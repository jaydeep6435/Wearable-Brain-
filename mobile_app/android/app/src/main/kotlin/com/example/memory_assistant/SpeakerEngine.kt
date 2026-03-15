package com.example.memory_assistant

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import org.json.JSONArray
import java.io.File
import java.util.UUID
import kotlin.math.sqrt

/**
 * SpeakerEngine — Offline speaker identification using Vosk x-vectors.
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
    private const val SIMILARITY_THRESHOLD = 0.42  // Lowered for better BT audio speaker separation
    private const val SESSION_SIMILARITY_THRESHOLD = 0.34 // More tolerant within a single noisy recording

    data class SpeakerProfile(
        val id: String,
        val name: String,
        val xvector: FloatArray,
        val sampleCount: Int,
        val createdAt: String
    )

    private val profiles = mutableListOf<SpeakerProfile>()
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
                }
            }
            c.close()
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
        if (profiles.isEmpty() || xvector.isEmpty()) return null to 0.0

        var bestMatch: SpeakerProfile? = null
        var bestSimilarity = -1.0

        for (profile in profiles) {
            val sim = cosineSimilarity(xvector, profile.xvector)
            if (sim > bestSimilarity) {
                bestSimilarity = sim
                bestMatch = profile
            }
        }

        return if (bestSimilarity >= SIMILARITY_THRESHOLD) {
            Log.d(TAG, "Identified: ${bestMatch?.name} (similarity=${"%.3f".format(bestSimilarity)})")
            bestMatch?.name to bestSimilarity
        } else {
            Log.d(TAG, "Unknown speaker (best=${"%.3f".format(bestSimilarity)})")
            null to bestSimilarity
        }
    }

    /**
     * Enroll a new speaker or update an existing one.
     * The x-vector is averaged with any existing samples for better accuracy.
     */
    fun enrollSpeaker(context: Context, name: String, xvector: FloatArray): String {
        val existing = profiles.find { it.name.equals(name, ignoreCase = true) }

        return if (existing != null) {
            // Average with existing vector for better accuracy
            val avgVector = averageVectors(existing.xvector, xvector, existing.sampleCount)
            val newCount = existing.sampleCount + 1

            updateProfile(context, existing.id, avgVector, newCount)
            profiles.removeAll { it.id == existing.id }
            profiles.add(existing.copy(xvector = avgVector, sampleCount = newCount))

            Log.i(TAG, "Updated profile: $name (samples=$newCount)")
            existing.id
        } else {
            val id = UUID.randomUUID().toString()
            insertProfile(context, id, name, xvector)
            profiles.add(SpeakerProfile(id, name, xvector, 1, ""))

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
            sqlDb.delete("speaker_voiceprints", "id = ?", arrayOf(id))
            sqlDb.close()
            profiles.removeAll { it.id == id }
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
        if (enrolledName != null) {
            Log.i(TAG, "  → Matched enrolled: $enrolledName (sim=${"%.3f".format(enrolledSim)})")
            return enrolledName
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

    fun matchSessionSpeakerOnly(xvector: FloatArray): String? {
        val (enrolledName, enrolledSim) = identifySpeaker(xvector)
        if (enrolledName != null) {
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

    private fun insertProfile(context: Context, id: String, name: String, xvector: FloatArray) {
        try {
            val db = context.getDatabasePath("memory.db")
            val sqlDb = SQLiteDatabase.openDatabase(db.absolutePath, null, SQLiteDatabase.OPEN_READWRITE)

            // Ensure table exists
            sqlDb.execSQL("""
                CREATE TABLE IF NOT EXISTS speaker_voiceprints (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    xvector TEXT NOT NULL,
                    sample_count INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            val cv = ContentValues().apply {
                put("id", id)
                put("name", name)
                put("xvector", floatArrayToJson(xvector))
                put("sample_count", 1)
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
