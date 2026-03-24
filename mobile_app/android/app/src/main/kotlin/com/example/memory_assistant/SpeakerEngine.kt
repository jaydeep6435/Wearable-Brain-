package com.example.memory_assistant

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import org.json.JSONArray
import java.io.File
import java.util.UUID
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * SpeakerEngine — Offline speaker identification using 128-D voice embeddings.
 *
 * Each voice has a unique 128-dimensional x-vector "fingerprint".
 * We store enrolled voices in SQLite and compare new speech segments
 * against them using cosine similarity.
 *
 * Similarity > 0.6 → known speaker
 * Similarity < 0.6 → unknown speaker (offer to enroll)
 */
object SpeakerEngine {

    private const val TAG = "WBrain.Speaker"
    private const val SIMILARITY_THRESHOLD = 0.44
    private const val ENROLLED_STRONG_THRESHOLD = 0.56
    private const val ENROLLED_MARGIN_THRESHOLD = 0.03
    private const val SESSION_SIMILARITY_THRESHOLD = 0.34 // More tolerant within a single noisy recording
    private const val MAX_SAMPLES_PER_PROFILE = 240

    data class SpeakerProfile(
        val id: String,
        val name: String,
        val xvector: FloatArray,
        val sampleCount: Int,
        val createdAt: String
    )

    data class MatchStats(
        val name: String?,
        val similarity: Double,
        val threshold: Double,
        val margin: Double,
    )

    private val profiles = mutableListOf<SpeakerProfile>()
    private val profileSamples = mutableMapOf<String, MutableList<FloatArray>>()
    private var isLoaded = false

    fun reloadProfiles(context: Context) {
        isLoaded = false
        loadProfiles(context)
    }

    fun profileCount(): Int = profiles.size

    /**
     * Load all speaker profiles from the database.
     */
    fun loadProfiles(context: Context) {
        if (isLoaded) return
        profiles.clear()
        profileSamples.clear()

        try {
            val dbPath = context.getDatabasePath("memory.db")
            if (!dbPath.exists()) {
                isLoaded = true
                return
            }

            val db = SQLiteDatabase.openDatabase(
                dbPath.absolutePath, null, SQLiteDatabase.OPEN_READONLY
            )

            // Check if table exists
            val cursor = db.rawQuery(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='speaker_voiceprints'",
                null
            )
            val tableExists = cursor.moveToFirst()
            cursor.close()

            if (!tableExists) {
                db.close()
                isLoaded = true
                return
            }

            val c = db.rawQuery(
                "SELECT id, name, xvector, sample_count, created_at FROM speaker_voiceprints",
                null
            )
            while (c.moveToNext()) {
                val id = c.getString(0)
                val name = c.getString(1)
                val xvectorJson = c.getString(2)
                val sampleCount = c.getInt(3)
                val createdAt = c.getString(4) ?: ""

                val xvector = jsonToFloatArray(xvectorJson)
                if (xvector.isNotEmpty()) {
                    profiles.add(SpeakerProfile(id, name, xvector, sampleCount, createdAt))
                    profileSamples[id] = mutableListOf(xvector)
                }
            }
            c.close()

            // Load per-speaker fingerprint samples for stronger future matching.
            val samplesTable = db.rawQuery(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='speaker_voice_samples'",
                null
            )
            val hasSamplesTable = samplesTable.moveToFirst()
            samplesTable.close()

            if (hasSamplesTable) {
                val sc = db.rawQuery(
                    "SELECT profile_id, xvector FROM speaker_voice_samples ORDER BY created_at ASC",
                    null
                )
                while (sc.moveToNext()) {
                    val pid = sc.getString(0) ?: continue
                    val vec = jsonToFloatArray(sc.getString(1) ?: "")
                    if (vec.isNotEmpty()) {
                        profileSamples.getOrPut(pid) { mutableListOf() }.add(vec)
                    }
                }
                sc.close()
            }

            for (p in profiles) {
                if (profileSamples[p.id].isNullOrEmpty()) {
                    profileSamples[p.id] = mutableListOf(p.xvector)
                }
            }

            db.close()
            isLoaded = true
            Log.i(TAG, "Loaded ${profiles.size} speaker profiles")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading profiles: ${e.message}")
            isLoaded = true
        }
    }

    /**
     * Identify which stored speaker (if any) matches the given x-vector.
     * Returns the speaker name or null if no match found.
     */
    fun identifySpeaker(xvector: FloatArray): Pair<String?, Double> {
        val (bestMatch, bestSimilarity, bestThreshold, margin) = bestProfileWithThreshold(xvector)
        if (bestMatch == null) return null to 0.0

        return if (bestSimilarity >= bestThreshold && margin >= ENROLLED_MARGIN_THRESHOLD) {
            Log.d(TAG, "Identified: ${bestMatch?.name} (similarity=${"%.3f".format(bestSimilarity)})")
            bestMatch?.name to bestSimilarity
        } else {
            Log.d(TAG, "Unknown speaker (best=${"%.3f".format(bestSimilarity)}, margin=${"%.3f".format(margin)})")
            null to bestSimilarity
        }
    }

    /**
     * Return the best enrolled profile name and similarity without applying threshold gating.
     * Useful for second-pass relabeling of session clusters.
     */
    fun bestProfileForVector(xvector: FloatArray): Pair<String?, Double> {
        val (bestMatch, bestSimilarity, _, _) = bestProfileWithThreshold(xvector)
        return bestMatch?.name to bestSimilarity
    }

    fun bestProfileStats(xvector: FloatArray): MatchStats {
        val (bestMatch, bestSimilarity, bestThreshold, margin) = bestProfileWithThreshold(xvector)
        return MatchStats(bestMatch?.name, bestSimilarity, bestThreshold, margin)
    }

    private fun bestProfileWithThreshold(xvector: FloatArray): Quadruple<SpeakerProfile?, Double, Double, Double> {
        if (profiles.isEmpty() || xvector.isEmpty()) return Quadruple(null, 0.0, SIMILARITY_THRESHOLD, 0.0)

        var bestMatch: SpeakerProfile? = null
        var bestSimilarity = -1.0
        var secondBestSimilarity = -1.0
        var bestThreshold = SIMILARITY_THRESHOLD

        for (profile in profiles) {
            val samples = profileSamples[profile.id] ?: mutableListOf(profile.xvector)
            val simMean = cosineSimilarity(xvector, profile.xvector)
            var simBestSample = simMean
            for (s in samples) {
                val sim = cosineSimilarity(xvector, s)
                if (sim > simBestSample) simBestSample = sim
            }

            val sim = max(simMean * 0.9, simBestSample)
            val sampleCount = max(profile.sampleCount, samples.size)
            val adaptiveThreshold = SIMILARITY_THRESHOLD - min(0.06, (sampleCount - 1) * 0.005)
            if (sim > bestSimilarity) {
                secondBestSimilarity = bestSimilarity
                bestSimilarity = sim
                bestMatch = profile
                bestThreshold = adaptiveThreshold
            } else if (sim > secondBestSimilarity) {
                secondBestSimilarity = sim
            }
        }

        val margin = (bestSimilarity - secondBestSimilarity).coerceAtLeast(0.0)
        return Quadruple(bestMatch, bestSimilarity, bestThreshold, margin)
    }

    private data class Quadruple<A, B, C, D>(
        val first: A,
        val second: B,
        val third: C,
        val fourth: D,
    )

    /**
     * Enroll a new speaker or update an existing one.
     * The x-vector is averaged with any existing samples for better accuracy.
     */
    fun enrollSpeaker(context: Context, name: String, xvector: FloatArray): String {
        return enrollSpeaker(context, name, listOf(xvector))
    }

    fun enrollSpeaker(context: Context, name: String, xvectors: List<FloatArray>): String {
        val valid = xvectors.filter { it.isNotEmpty() }.map { normalizeVector(it) }
        if (valid.isEmpty()) return ""

        val centroid = centroid(valid) ?: return ""
        val existing = profiles.find { it.name.equals(name, ignoreCase = true) }

        return if (existing != null) {
            val oldSamples = profileSamples[existing.id] ?: mutableListOf(existing.xvector)
            val merged = (oldSamples + valid).takeLast(MAX_SAMPLES_PER_PROFILE)
            val avgVector = centroid(merged) ?: existing.xvector
            val newCount = existing.sampleCount + valid.size

            updateProfile(context, existing.id, avgVector, newCount)
            replaceSamples(context, existing.id, merged)
            profiles.removeAll { it.id == existing.id }
            profiles.add(existing.copy(xvector = avgVector, sampleCount = newCount))
            profileSamples[existing.id] = merged.toMutableList()

            Log.i(TAG, "Updated profile: $name (samples=$newCount)")
            existing.id
        } else {
            val id = UUID.randomUUID().toString()
            insertProfile(context, id, name, centroid, valid.size)
            replaceSamples(context, id, valid.takeLast(MAX_SAMPLES_PER_PROFILE))
            profiles.add(SpeakerProfile(id, name, centroid, valid.size, ""))
            profileSamples[id] = valid.takeLast(MAX_SAMPLES_PER_PROFILE).toMutableList()

            Log.i(TAG, "Enrolled new speaker: $name")
            id
        }
    }

    /**
     * Delete a speaker profile.
     */
    fun deleteProfile(context: Context, id: String) {
        try {
            val db = context.getDatabasePath("memory.db")
            val sqlDb = SQLiteDatabase.openDatabase(db.absolutePath, null, SQLiteDatabase.OPEN_READWRITE)
            ensureSpeakerTables(sqlDb)
            sqlDb.delete("speaker_voiceprints", "id = ?", arrayOf(id))
            sqlDb.delete("speaker_voice_samples", "profile_id = ?", arrayOf(id))
            sqlDb.close()
            profiles.removeAll { it.id == id }
            profileSamples.remove(id)
            Log.i(TAG, "Deleted profile: $id")
        } catch (e: Exception) {
            Log.e(TAG, "Error deleting profile: ${e.message}")
        }
    }

    /**
     * Get all profiles as a list of maps (for Flutter).
     */
    fun getProfiles(): List<Map<String, Any>> {
        return profiles.map { p ->
            mapOf(
                "id" to p.id,
                "name" to p.name,
                "sample_count" to p.sampleCount,
                "created_at" to p.createdAt
            )
        }
    }

    /**
     * Assign a label during conversation: tracks unknown speakers within a session.
     * Returns a consistent label like "Speaker 1", "Speaker 2" etc.
     */
    private val sessionSpeakers = mutableMapOf<Int, Pair<String, FloatArray>>()
    private val sessionSpeakerCounts = mutableMapOf<Int, Int>()
    private var nextUnknownId = 1

    fun resetSession() {
        sessionSpeakers.clear()
        sessionSpeakerCounts.clear()
        nextUnknownId = 1
    }

    /**
     * Identify or assign a speaker for a given x-vector within a conversation session.
     */
    fun getSessionSpeaker(xvector: FloatArray): String {
        // First, check enrolled profiles
        val (enrolledName, enrolledSim) = identifySpeaker(xvector)
        if (enrolledName != null && enrolledSim >= ENROLLED_STRONG_THRESHOLD) {
            Log.i(TAG, "  → Matched enrolled: $enrolledName (sim=${"%.3f".format(enrolledSim)})")
            return enrolledName
        } else if (enrolledName != null) {
            Log.i(TAG, "  → Enrolled match too weak (${"%.3f".format(enrolledSim)}), using session clustering")
        }

        // Check session-local speakers
        var bestSessionLabel: String? = null
        var bestSessionSim = -1.0
        var bestSessionId = -1
        for ((id, pair) in sessionSpeakers) {
            val sim = cosineSimilarity(xvector, pair.second)
            Log.d(TAG, "  Session ${pair.first} sim=${"%.3f".format(sim)}")
            if (sim >= SESSION_SIMILARITY_THRESHOLD && sim > bestSessionSim) {
                bestSessionSim = sim
                bestSessionLabel = pair.first
                bestSessionId = id
            }
        }

        if (bestSessionLabel != null && bestSessionId >= 0) {
            // Update the session speaker with averaged vector for better future matching
            val existing = sessionSpeakers[bestSessionId]!!
            val count = sessionSpeakerCounts.getOrDefault(bestSessionId, 1)
            val avgVec = averageVectors(existing.second, xvector, count)
            sessionSpeakers[bestSessionId] = existing.first to avgVec
            sessionSpeakerCounts[bestSessionId] = count + 1
            Log.i(TAG, "  → Session match: $bestSessionLabel (sim=${"%.3f".format(bestSessionSim)}, samples=${count+1})")
            return bestSessionLabel
        }

        // New unknown speaker in this session
        val label = "Speaker $nextUnknownId"
        sessionSpeakers[nextUnknownId] = label to xvector
        sessionSpeakerCounts[nextUnknownId] = 1
        nextUnknownId++
        Log.i(TAG, "  → New session speaker: $label")
        return label
    }

    /**
     * Assign a label using session clustering only (no enrolled profile matching).
     * Useful when initial turn boundaries are approximate and direct enrolled matching
     * can collapse all turns to one known person.
     */
    fun getSessionClusterLabel(xvector: FloatArray): String {
        var bestSessionLabel: String? = null
        var bestSessionSim = -1.0
        var bestSessionId = -1

        for ((id, pair) in sessionSpeakers) {
            val sim = cosineSimilarity(xvector, pair.second)
            if (sim >= SESSION_SIMILARITY_THRESHOLD && sim > bestSessionSim) {
                bestSessionSim = sim
                bestSessionLabel = pair.first
                bestSessionId = id
            }
        }

        if (bestSessionLabel != null && bestSessionId >= 0) {
            val existing = sessionSpeakers[bestSessionId]!!
            val count = sessionSpeakerCounts.getOrDefault(bestSessionId, 1)
            val avgVec = averageVectors(existing.second, xvector, count)
            sessionSpeakers[bestSessionId] = existing.first to avgVec
            sessionSpeakerCounts[bestSessionId] = count + 1
            return bestSessionLabel
        }

        val label = "Speaker $nextUnknownId"
        sessionSpeakers[nextUnknownId] = label to xvector
        sessionSpeakerCounts[nextUnknownId] = 1
        nextUnknownId++
        return label
    }

    fun matchSessionSpeakerOnly(xvector: FloatArray): String? {
        val (enrolledName, enrolledSim) = identifySpeaker(xvector)
        if (enrolledName != null && enrolledSim >= ENROLLED_STRONG_THRESHOLD) {
            Log.i(TAG, "  → Matched enrolled (session-only): $enrolledName (sim=${"%.3f".format(enrolledSim)})")
            return enrolledName
        }

        var bestSessionLabel: String? = null
        var bestSessionSim = -1.0
        for ((_, pair) in sessionSpeakers) {
            val sim = cosineSimilarity(xvector, pair.second)
            if (sim >= SESSION_SIMILARITY_THRESHOLD && sim > bestSessionSim) {
                bestSessionSim = sim
                bestSessionLabel = pair.first
            }
        }
        return bestSessionLabel
    }

    fun dominantSessionSpeaker(): String? {
        if (sessionSpeakerCounts.isEmpty()) return null
        val bestId = sessionSpeakerCounts.maxByOrNull { it.value }?.key ?: return null
        return sessionSpeakers[bestId]?.first
    }

    /**
     * Promote a session-only unknown speaker label (e.g., "Speaker 2")
     * into a persistent enrolled profile with a real name.
     */
    fun enrollSessionSpeaker(context: Context, sessionLabel: String, name: String): String? {
        val hit = sessionSpeakers.entries.firstOrNull {
            it.value.first.equals(sessionLabel, ignoreCase = true)
        } ?: return null

        val vec = hit.value.second
        if (vec.isEmpty()) return null

        val profileId = enrollSpeaker(context, name, listOf(vec))
        if (profileId.isNotBlank()) {
            sessionSpeakers[hit.key] = name to vec
        }
        return profileId.ifBlank { null }
    }

    // ── Math ─────────────────────────────────────────────────

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Double {
        if (a.size != b.size || a.isEmpty()) return 0.0
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom > 0) dot / denom else 0.0
    }

    private fun averageVectors(existing: FloatArray, newVec: FloatArray, existingCount: Int): FloatArray {
        val result = FloatArray(existing.size)
        val w = existingCount.toFloat() / (existingCount + 1)
        val nw = 1f / (existingCount + 1)
        for (i in existing.indices) {
            result[i] = existing[i] * w + newVec[i] * nw
        }
        return result
    }

    // ── DB Helpers ───────────────────────────────────────────

    private fun insertProfile(context: Context, id: String, name: String, xvector: FloatArray, sampleCount: Int) {
        try {
            val db = context.getDatabasePath("memory.db")
            val sqlDb = SQLiteDatabase.openDatabase(db.absolutePath, null, SQLiteDatabase.OPEN_READWRITE)
            ensureSpeakerTables(sqlDb)

            val cv = ContentValues().apply {
                put("id", id)
                put("name", name)
                put("xvector", floatArrayToJson(xvector))
                put("sample_count", sampleCount)
            }
            sqlDb.insertWithOnConflict("speaker_voiceprints", null, cv, SQLiteDatabase.CONFLICT_REPLACE)
            sqlDb.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error inserting profile: ${e.message}")
        }
    }

    private fun updateProfile(context: Context, id: String, xvector: FloatArray, sampleCount: Int) {
        try {
            val db = context.getDatabasePath("memory.db")
            val sqlDb = SQLiteDatabase.openDatabase(db.absolutePath, null, SQLiteDatabase.OPEN_READWRITE)
            ensureSpeakerTables(sqlDb)
            val cv = ContentValues().apply {
                put("xvector", floatArrayToJson(xvector))
                put("sample_count", sampleCount)
                put("updated_at", System.currentTimeMillis().toString())
            }
            sqlDb.update("speaker_voiceprints", cv, "id = ?", arrayOf(id))
            sqlDb.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error updating profile: ${e.message}")
        }
    }

    private fun replaceSamples(context: Context, profileId: String, samples: List<FloatArray>) {
        try {
            val db = context.getDatabasePath("memory.db")
            val sqlDb = SQLiteDatabase.openDatabase(db.absolutePath, null, SQLiteDatabase.OPEN_READWRITE)
            ensureSpeakerTables(sqlDb)
            sqlDb.delete("speaker_voice_samples", "profile_id = ?", arrayOf(profileId))

            val keep = samples.takeLast(MAX_SAMPLES_PER_PROFILE)
            keep.forEachIndexed { idx, v ->
                val cv = ContentValues().apply {
                    put("id", UUID.randomUUID().toString())
                    put("profile_id", profileId)
                    put("sample_index", idx)
                    put("xvector", floatArrayToJson(v))
                }
                sqlDb.insertWithOnConflict("speaker_voice_samples", null, cv, SQLiteDatabase.CONFLICT_REPLACE)
            }
            sqlDb.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error replacing profile samples: ${e.message}")
        }
    }

    private fun ensureSpeakerTables(db: SQLiteDatabase) {
        db.execSQL(
            """
                CREATE TABLE IF NOT EXISTS speaker_voiceprints (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    xvector TEXT NOT NULL,
                    sample_count INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """.trimIndent()
        )
        db.execSQL(
            """
                CREATE TABLE IF NOT EXISTS speaker_voice_samples (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    sample_index INTEGER DEFAULT 0,
                    xvector TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (profile_id) REFERENCES speaker_voiceprints(id)
                )
            """.trimIndent()
        )
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_voice_samples_profile ON speaker_voice_samples(profile_id)")
    }

    private fun normalizeVector(vec: FloatArray): FloatArray {
        var norm = 0.0
        for (v in vec) norm += (v * v)
        val denom = sqrt(norm).toFloat()
        if (denom <= 0f) return vec.copyOf()

        val out = FloatArray(vec.size)
        for (i in vec.indices) out[i] = vec[i] / denom
        return out
    }

    private fun centroid(vectors: List<FloatArray>): FloatArray? {
        if (vectors.isEmpty()) return null
        val dim = vectors.first().size
        if (dim == 0) return null

        val out = FloatArray(dim)
        var count = 0
        for (v in vectors) {
            if (v.size != dim) continue
            for (i in 0 until dim) out[i] += v[i]
            count++
        }
        if (count == 0) return null
        for (i in 0 until dim) out[i] /= count.toFloat()
        return normalizeVector(out)
    }

    // ── JSON Helpers ─────────────────────────────────────────

    private fun floatArrayToJson(arr: FloatArray): String {
        val ja = JSONArray()
        for (f in arr) ja.put(f.toDouble())
        return ja.toString()
    }

    private fun jsonToFloatArray(json: String): FloatArray {
        return try {
            val ja = JSONArray(json)
            FloatArray(ja.length()) { ja.getDouble(it).toFloat() }
        } catch (e: Exception) {
            floatArrayOf()
        }
    }
}
