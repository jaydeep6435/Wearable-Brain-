package com.example.memory_assistant

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineWhisperModelConfig
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * SherpaTranscriber - Offline whisper transcription using sherpa-onnx.
 */
object SherpaTranscriber {

    private const val TAG = "WBrain.Sherpa"
    private const val MODEL_DIR_NAME = "sherpa-onnx-whisper-small.en"
    private const val MODEL_ARCHIVE_NAME = "sherpa-onnx-whisper-small.en.tar.bz2"
    private const val MODEL_URL =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.en.tar.bz2"
    private const val ENROLLED_RELABEL_THRESHOLD = 0.50
    private const val ENROLLED_REINFORCE_THRESHOLD = 0.62
    private const val ENROLLED_RELABEL_MARGIN = 0.02
    private const val MIN_ENROLLED_VOTES = 6
    private const val MIN_ENROLLED_VOTE_RATIO = 0.75
    private const val MAX_ENROLLMENT_CHUNKS_SAFETY = 240
    private const val ASR_MAX_CHUNK_SECONDS = 28

    private var recognizer: OfflineRecognizer? = null
    private var spkReady = true

    // ASR State
    var isAsrInitializing = false
    var isAsrPaused = false
    var asrInitError: String? = null
    var asrProgress = 0
    var asrBytesDownloaded = 0L
    var asrTotalBytes = 0L
    var asrStage = "idle" // idle|downloading|extracting|loading|ready|error

    // SPK State (logical compatibility for existing UI)
    var isSpkInitializing = false
    var isSpkPaused = false
    var spkInitError: String? = null
    var spkProgress = 0
    var spkBytesDownloaded = 0L
    var spkTotalBytes = 0L

    fun isAsrReady(): Boolean = recognizer != null
    fun isSpkReady(): Boolean = spkReady
    fun isReady(): Boolean = isAsrReady()

    fun initAsr(context: Context, onComplete: ((Boolean) -> Unit)? = null) {
        if (recognizer != null) { onComplete?.invoke(true); return }
        if (isAsrInitializing && !isAsrPaused) { onComplete?.invoke(false); return }

        isAsrInitializing = true
        isAsrPaused = false
        asrInitError = null
        asrStage = "loading"
        Log.i(TAG, "Initializing Sherpa-ONNX Whisper ASR")

        Thread {
            try {
                var modelRoot = resolveModelRoot(context)
                if (modelRoot == null) {
                    val archiveFile = File(context.cacheDir, MODEL_ARCHIVE_NAME)
                    asrStage = "downloading"
                    downloadFileResumable(MODEL_URL, archiveFile)
                    if (isAsrPaused) {
                        asrStage = "idle"
                        isAsrInitializing = false
                        return@Thread
                    }

                    // Prefer internal app storage for faster extraction I/O on many devices.
                    val targetParent = context.filesDir
                    asrProgress = 100
                    asrStage = "extracting"
                    val extractStartMs = System.currentTimeMillis()
                    extractTarBz2(archiveFile, targetParent)
                    Log.i(TAG, "Model extract completed in ${System.currentTimeMillis() - extractStartMs} ms")
                    ensureModelLayout(targetParent)
                    modelRoot = resolveModelRoot(context)
                }

                if (modelRoot == null) {
                    throw Exception("Model files not found after download/extract")
                }

                asrStage = "loading"
                recognizer = createRecognizer(modelRoot)
                isAsrInitializing = false
                asrProgress = 100
                asrStage = "ready"
                Log.i(TAG, "Sherpa model ready: ${modelRoot.absolutePath}")
                onComplete?.invoke(true)

            } catch (e: Throwable) {
                if (!isAsrPaused) {
                    isAsrInitializing = false
                    asrInitError = formatAsrInitError(e)
                    asrStage = "error"
                    Log.e(TAG, "ASR init failed: ${asrInitError}")
                    onComplete?.invoke(false)
                }
            }
        }.start()
    }

    private fun formatAsrInitError(t: Throwable): String {
        val raw = (t.message ?: t.toString()).trim()
        val lower = raw.lowercase()
        if ("failed to get field id" in lower && "decodingmethod" in lower) {
            return "Model engine compatibility error on this device build. Please tap Retry once. If it persists, reinstall latest app build. Details: $raw"
        }
        return raw.ifBlank { "Unknown ASR initialization error" }
    }

    fun pauseAsr() {
        // Pause is only meaningful while downloading bytes.
        if (asrStage == "downloading") {
            isAsrPaused = true
        }
    }
    fun resumeAsr(context: Context) { if (isAsrPaused) initAsr(context) }
    fun retryAsr(context: Context) {
        asrInitError = null
        isAsrInitializing = false
        isAsrPaused = false
        asrStage = "idle"
        recognizer?.release()
        recognizer = null
        initAsr(context)
    }

    fun initSpk(context: Context, onComplete: ((Boolean) -> Unit)? = null) {
        isSpkInitializing = true
        isSpkPaused = false
        spkInitError = ""
        spkReady = true
        spkProgress = 100
        isSpkInitializing = false
        onComplete?.invoke(true)
    }

    fun pauseSpk() { isSpkPaused = true }
    fun resumeSpk(context: Context) { if (isSpkPaused) initSpk(context) }
    fun retrySpk(context: Context) {
        spkInitError = null
        isSpkInitializing = false
        isSpkPaused = false
        initSpk(context)
    }

    fun transcribeWavWithSpeakers(
        wavPath: String,
        context: Context,
    ): List<Map<String, Any>> {
        val text = transcribeWav(wavPath).trim()
        if (text.isEmpty()) return emptyList()

        SpeakerEngine.loadProfiles(context)
        SpeakerEngine.resetSession()

        // Split transcript into sentence-like chunks.
        val pieces = text
            .split(Regex("(?<=[.!?])\\s+|\\n+"))
            .map { it.trim() }
            .filter { it.isNotBlank() }

        if (pieces.size <= 1) {
            return listOf(mapOf("speaker" to "Speaker 1", "text" to text))
        }

        // Merge tiny acknowledgement fragments with the next sentence.
        // Example: "Yes." + "What time does it start?" should be one turn.
        fun wordCount(s: String): Int = s
            .split(Regex("\\s+"))
            .count { it.isNotBlank() }

        val shortAcks = setOf(
            "yes", "yeah", "yep", "no", "ok", "okay", "sure", "hmm", "hm",
            "alright", "right", "fine", "perfect",
        )

        fun shouldMergeWithNext(current: String, next: String): Boolean {
            val normalized = current.lowercase()
                .replace(Regex("[^a-z\\s]"), "")
                .trim()
            val wc = wordCount(normalized)
            val hasTimeExpr = Regex("\\b\\d{1,2}\\s*(am|pm)\\b|\\b(am|pm)\\b", RegexOption.IGNORE_CASE)
                .containsMatchIn(current)

            if (next.isBlank()) return false
            if (hasTimeExpr) return false

            return wc <= 2 || normalized in shortAcks || current.trim().endsWith(",")
        }

        val turns = mutableListOf<String>()
        var i = 0
        while (i < pieces.size) {
            val cur = pieces[i]
            if (i + 1 < pieces.size && shouldMergeWithNext(cur, pieces[i + 1])) {
                turns.add("${cur.trim()} ${pieces[i + 1].trim()}")
                i += 2
            } else {
                turns.add(cur)
                i++
            }
        }

        val cleanedTurns = turns
            .map { it.replace(Regex("^\\s*>>\\s*"), "").trim() }
            .filter {
                it.isNotBlank() &&
                    !it.contains("[blank", ignoreCase = true) &&
                    !it.contains("metal clanking", ignoreCase = true)
            }

        if (cleanedTurns.isEmpty()) return emptyList()

        val wave = try {
            readWavePcm16(wavPath)
        } catch (_: Exception) {
            null
        }

        if (wave == null || wave.samples.isEmpty()) {
            return cleanedTurns.mapIndexed { idx, turn ->
                mapOf(
                    "speaker" to if (idx % 2 == 0) "Speaker 1" else "Speaker 2",
                    "text" to turn,
                )
            }
        }

        val totalChars = cleanedTurns.sumOf { it.length }.coerceAtLeast(1)
        val totalSamples = wave.samples.size
        var cursor = 0

        val acousticLabels = buildAcousticSpeakerWindows(wave.samples, wave.sampleRate)
        val turnVectors = mutableListOf<FloatArray?>()

        val segments = cleanedTurns.mapIndexed { idx, turn ->
            val span = ((totalSamples.toDouble() * turn.length.toDouble() / totalChars.toDouble()).toInt())
                .coerceAtLeast(1)
            val start = cursor.coerceIn(0, totalSamples - 1)
            val end = (start + span).coerceAtMost(totalSamples)
            cursor = end

            val vec = extractXVectorFromSamples(wave.samples, start, end)
            turnVectors.add(vec)

            val speaker = acousticLabels
                ?.let { pickSpeakerFromWindows(start, end, it) }
                ?: vec
                    ?.let { SpeakerEngine.getSessionClusterLabel(it) }
                ?: if (idx % 2 == 0) "Speaker 1" else "Speaker 2"

            mapOf(
                "speaker" to speaker,
                "text" to turn,
            )
        }

        val fingerprintAssigned = assignByTurnFingerprints(segments, turnVectors) ?: segments

        val diversified = enforceSpeakerDiversityFallback(fingerprintAssigned)
        val stabilized = regularizeUnstableTwoSpeakerRuns(diversified)
        val relabeled = relabelGenericSpeakersFromEnrollment(stabilized, turnVectors)
        reinforceMatchedEnrolledProfiles(context, relabeled, turnVectors)
        return relabeled
    }

    private fun regularizeUnstableTwoSpeakerRuns(
        segments: List<Map<String, Any>>,
    ): List<Map<String, Any>> {
        if (segments.size < 6) return segments

        val generic = Regex("^Speaker\\s+\\d+$", RegexOption.IGNORE_CASE)
        val labels = segments.map { (it["speaker"] ?: "").toString().trim() }
        val unique = labels.filter { generic.matches(it) }.distinct()
        if (unique.size != 2) return segments

        val runLengths = mutableListOf<Int>()
        var run = 1
        for (i in 1 until labels.size) {
            if (labels[i].equals(labels[i - 1], ignoreCase = true)) run++ else {
                runLengths.add(run)
                run = 1
            }
        }
        runLengths.add(run)

        val longestRun = runLengths.maxOrNull() ?: 1
        val c1 = labels.count { it.equals(unique[0], ignoreCase = true) }
        val c2 = labels.count { it.equals(unique[1], ignoreCase = true) }
        val dominantRatio = maxOf(c1, c2).toDouble() / labels.size.toDouble()

        // If generic labels are heavily imbalanced or streaky, enforce stable turn-taking.
        if (longestRun < 3 && dominantRatio <= 0.72) return segments

        Log.w(
            TAG,
            "Regularizing unstable two-speaker run: longestRun=$longestRun, dominantRatio=${"%.2f".format(dominantRatio)}"
        )

        val first = labels.firstOrNull { generic.matches(it) } ?: "Speaker 1"
        val second = unique.firstOrNull { !it.equals(first, ignoreCase = true) } ?: "Speaker 2"

        return segments.mapIndexed { idx, seg ->
            val target = if (idx % 2 == 0) first else second
            seg.toMutableMap().apply { this["speaker"] = target }
        }
    }

    private fun assignByTurnFingerprints(
        segments: List<Map<String, Any>>,
        vectors: List<FloatArray?>,
    ): List<Map<String, Any>>? {
        if (segments.isEmpty() || vectors.isEmpty()) return null

        val indexed = vectors.mapIndexedNotNull { idx, vec ->
            if (vec != null) idx to vec else null
        }
        if (indexed.size < 4) return null

        val first = indexed.first()
        val second = indexed.maxByOrNull { (_, v) -> 1.0 - cosineSimilarity(first.second, v) } ?: return null

        var c1 = first.second.copyOf()
        var c2 = second.second.copyOf()
        val labels = IntArray(indexed.size) { 0 }

        repeat(8) {
            var any1 = false
            var any2 = false
            for (i in indexed.indices) {
                val s1 = cosineSimilarity(indexed[i].second, c1)
                val s2 = cosineSimilarity(indexed[i].second, c2)
                labels[i] = if (s1 >= s2) 0 else 1
                if (labels[i] == 0) any1 = true else any2 = true
            }
            if (!any1 || !any2) return null

            val g1 = mutableListOf<FloatArray>()
            val g2 = mutableListOf<FloatArray>()
            for (i in indexed.indices) {
                if (labels[i] == 0) g1.add(indexed[i].second) else g2.add(indexed[i].second)
            }
            c1 = centroid(g1) ?: return null
            c2 = centroid(g2) ?: return null
        }

        // If centroids are nearly identical, fingerprint clustering is unreliable.
        val centerSim = cosineSimilarity(c1, c2)
        if (centerSim >= 0.992) return null

        val g1Count = labels.count { it == 0 }
        val g2Count = labels.size - g1Count
        val clusterBalance = minOf(g1Count, g2Count).toDouble() / maxOf(g1Count, g2Count).toDouble()
        if (clusterBalance < 0.20) {
            // Strong imbalance usually means one-speaker audio with noisy split.
            return null
        }

        val turnLabels = MutableList(segments.size) { -1 }
        for (i in indexed.indices) {
            turnLabels[indexed[i].first] = labels[i]
        }

        // Fill missing labels from nearest known neighbors.
        for (i in turnLabels.indices) {
            if (turnLabels[i] >= 0) continue
            val left = (i - 1 downTo 0).firstOrNull { turnLabels[it] >= 0 }
            val right = (i + 1 until turnLabels.size).firstOrNull { turnLabels[it] >= 0 }
            turnLabels[i] = when {
                left != null && right != null -> if (i - left <= right - i) turnLabels[left] else turnLabels[right]
                left != null -> turnLabels[left]
                right != null -> turnLabels[right]
                else -> if (i % 2 == 0) 0 else 1
            }
        }

        // Smooth isolated flip noise: A B A -> A A A.
        for (i in 1 until turnLabels.lastIndex) {
            val a = turnLabels[i - 1]
            val b = turnLabels[i]
            val c = turnLabels[i + 1]
            if (a == c && b != a) {
                turnLabels[i] = a
            }
        }

        val first0 = turnLabels.indexOfFirst { it == 0 }
        val first1 = turnLabels.indexOfFirst { it == 1 }
        val zeroIsSpeaker1 = first1 < 0 || (first0 >= 0 && first0 <= first1)

        return segments.mapIndexed { idx, seg ->
            val cls = turnLabels[idx]
            val label = if ((cls == 0) == zeroIsSpeaker1) "Speaker 1" else "Speaker 2"
            seg.toMutableMap().apply { this["speaker"] = label }
        }
    }

    private data class SpeakerWindow(
        val start: Int,
        val end: Int,
        val label: String,
    )

    private data class WindowVector(
        val start: Int,
        val end: Int,
        val vec: FloatArray,
        val rms: Double,
    )

    private fun buildAcousticSpeakerWindows(samples: FloatArray, sampleRate: Int): List<SpeakerWindow>? {
        if (samples.isEmpty() || sampleRate <= 0) return null

        val win = (sampleRate * 2).coerceAtLeast(1600)
        val hop = sampleRate.coerceAtLeast(800)

        val raw = mutableListOf<WindowVector>()
        var start = 0
        while (start < samples.size) {
            val end = (start + win).coerceAtMost(samples.size)
            if (end - start < 1600) break

            val rms = computeRms(samples, start, end)
            val vec = extractXVectorFromSamples(samples, start, end)
            if (vec != null) {
                raw.add(WindowVector(start, end, vec, rms))
            }

            if (end >= samples.size) break
            start += hop
        }

        if (raw.size < 4) return null

        val bestRms = raw.maxOf { it.rms }
        val voiced = raw.filter { it.rms >= bestRms * 0.12 }
        if (voiced.size < 4) return null

        val first = voiced.maxByOrNull { it.rms } ?: return null
        val second = voiced.minByOrNull { cosineSimilarity(first.vec, it.vec) } ?: return null

        var c1 = first.vec.copyOf()
        var c2 = second.vec.copyOf()
        var labels = IntArray(voiced.size) { 0 }

        repeat(8) {
            var any1 = false
            var any2 = false

            for (i in voiced.indices) {
                val s1 = cosineSimilarity(voiced[i].vec, c1)
                val s2 = cosineSimilarity(voiced[i].vec, c2)
                labels[i] = if (s1 >= s2) 0 else 1
                if (labels[i] == 0) any1 = true else any2 = true
            }

            if (!any1 || !any2) return null

            val g1 = mutableListOf<FloatArray>()
            val g2 = mutableListOf<FloatArray>()
            for (i in voiced.indices) {
                if (labels[i] == 0) g1.add(voiced[i].vec) else g2.add(voiced[i].vec)
            }

            c1 = centroid(g1) ?: return null
            c2 = centroid(g2) ?: return null
        }

        val centroidSimilarity = cosineSimilarity(c1, c2)
        if (centroidSimilarity >= 0.965) {
            // Two centroids are too similar -> likely one speaker.
            return null
        }

        val firstLabelIndex = labels.indexOfFirst { it == 0 }
        val secondLabelIndex = labels.indexOfFirst { it == 1 }
        if (firstLabelIndex < 0 || secondLabelIndex < 0) return null

        val mapZeroToSpeaker1 = voiced[firstLabelIndex].start <= voiced[secondLabelIndex].start

        return voiced.mapIndexed { idx, w ->
            val lbl = labels[idx]
            val speaker = if ((lbl == 0) == mapZeroToSpeaker1) "Speaker 1" else "Speaker 2"
            SpeakerWindow(w.start, w.end, speaker)
        }
    }

    private fun pickSpeakerFromWindows(start: Int, end: Int, windows: List<SpeakerWindow>): String? {
        if (windows.isEmpty()) return null

        val votes = mutableMapOf<String, Int>()
        for (w in windows) {
            val overlap = (minOf(end, w.end) - maxOf(start, w.start)).coerceAtLeast(0)
            if (overlap <= 0) continue
            votes[w.label] = votes.getOrDefault(w.label, 0) + overlap
        }

        return votes.maxByOrNull { it.value }?.key
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Double {
        if (a.isEmpty() || b.isEmpty() || a.size != b.size) return 0.0
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        val denom = sqrt(na) * sqrt(nb)
        return if (denom > 0.0) dot / denom else 0.0
    }

    private fun reinforceMatchedEnrolledProfiles(
        context: Context,
        segments: List<Map<String, Any>>,
        vectors: List<FloatArray?>,
    ) {
        if (segments.isEmpty() || vectors.isEmpty()) return

        val genericPattern = Regex("^Speaker\\s+\\d+$", RegexOption.IGNORE_CASE)
        val buckets = mutableMapOf<String, MutableList<FloatArray>>()

        for (i in segments.indices) {
            val label = (segments[i]["speaker"] ?: "").toString().trim()
            if (label.isBlank() || genericPattern.matches(label)) continue

            val vec = vectors.getOrNull(i) ?: continue
            val stats = SpeakerEngine.bestProfileStats(vec)
            if (!stats.name.isNullOrBlank() &&
                stats.name.equals(label, ignoreCase = true) &&
                stats.similarity >= ENROLLED_REINFORCE_THRESHOLD &&
                stats.margin >= ENROLLED_RELABEL_MARGIN
            ) {
                buckets.getOrPut(stats.name) { mutableListOf() }.add(vec)
            }
        }

        for ((name, vecs) in buckets) {
            if (vecs.size < MIN_ENROLLED_VOTES) {
                Log.i(TAG, "Skip reinforcement for $name: only ${vecs.size} strong match(es), need $MIN_ENROLLED_VOTES")
                continue
            }
            SpeakerEngine.enrollSpeaker(context, name, vecs.take(4))
            Log.i(TAG, "Reinforced enrolled profile: $name (+${vecs.size} segment sample(s))")
        }
    }

    private fun relabelGenericSpeakersFromEnrollment(
        segments: List<Map<String, Any>>,
        vectors: List<FloatArray?>,
    ): List<Map<String, Any>> {
        if (segments.isEmpty() || vectors.isEmpty()) return segments

        val genericPattern = Regex("^Speaker\\s+\\d+$", RegexOption.IGNORE_CASE)
        val genericLabels = segments.mapNotNull { seg ->
            val s = (seg["speaker"] ?: "").toString().trim()
            if (genericPattern.matches(s)) s else null
        }.distinct()
        if (genericLabels.isEmpty()) return segments

        data class Candidate(
            val label: String,
            val name: String,
            val sim: Double,
            val votes: Int,
            val total: Int,
        )
        val candidates = mutableListOf<Candidate>()

        for (label in genericLabels) {
            val vecs = mutableListOf<FloatArray>()
            for (i in segments.indices) {
                val sp = (segments[i]["speaker"] ?: "").toString()
                if (sp.equals(label, ignoreCase = true)) {
                    vectors.getOrNull(i)?.let { vecs.add(it) }
                }
            }
            if (vecs.isEmpty()) continue

            val c = centroid(vecs) ?: continue
            val stats = SpeakerEngine.bestProfileStats(c)

            var strongVotes = 0
            var voteSum = 0.0
            for (v in vecs) {
                val s = SpeakerEngine.bestProfileStats(v)
                if (!s.name.isNullOrBlank() &&
                    !stats.name.isNullOrBlank() &&
                    s.name.equals(stats.name, ignoreCase = true) &&
                    s.similarity >= ENROLLED_RELABEL_THRESHOLD &&
                    s.margin >= ENROLLED_RELABEL_MARGIN
                ) {
                    strongVotes++
                    voteSum += s.similarity
                }
            }
            val voteRatio = strongVotes.toDouble() / vecs.size.toDouble()
            val avgVoteSim = if (strongVotes > 0) voteSum / strongVotes.toDouble() else 0.0

            if (!stats.name.isNullOrBlank() &&
                stats.similarity >= ENROLLED_RELABEL_THRESHOLD &&
                stats.margin >= ENROLLED_RELABEL_MARGIN &&
                strongVotes >= MIN_ENROLLED_VOTES &&
                voteRatio >= MIN_ENROLLED_VOTE_RATIO &&
                avgVoteSim >= ENROLLED_RELABEL_THRESHOLD
            ) {
                candidates.add(Candidate(label, stats.name, stats.similarity, strongVotes, vecs.size))
            } else {
                Log.i(
                    TAG,
                    "Skip enrolled relabel for $label -> ${stats.name ?: "<none>"}: " +
                        "votes=$strongVotes/${vecs.size}, ratio=${"%.2f".format(voteRatio)}, " +
                        "sim=${"%.3f".format(stats.similarity)}, margin=${"%.3f".format(stats.margin)}"
                )
            }
        }

        if (candidates.isEmpty()) return segments

        val labelToName = mutableMapOf<String, String>()
        val usedNames = mutableSetOf<String>()
        candidates.sortedWith(compareByDescending<Candidate> { it.votes }.thenByDescending { it.sim }).forEach { c ->
            val key = c.name.lowercase()
            if (key !in usedNames && c.label !in labelToName) {
                usedNames.add(key)
                labelToName[c.label] = c.name
            }
        }
        if (labelToName.isEmpty()) return segments

        Log.i(TAG, "Relabeled from enrolled voice match: ${labelToName.entries.joinToString { "${it.key}->${it.value}" }}")

        return segments.map { seg ->
            val sp = (seg["speaker"] ?: "").toString()
            val target = labelToName.entries.firstOrNull { it.key.equals(sp, ignoreCase = true) }?.value
            if (target != null) {
                seg.toMutableMap().apply { this["speaker"] = target }
            } else {
                seg
            }
        }
    }

    private fun enforceSpeakerDiversityFallback(
        segments: List<Map<String, Any>>,
    ): List<Map<String, Any>> {
        if (segments.size < 2) return segments

        val speakers = segments.map { (it["speaker"] ?: "").toString().trim() }
        val uniqueSpeakers = speakers.filter { it.isNotBlank() }.distinct()
        if (uniqueSpeakers.size > 1) return segments

        val onlySpeaker = uniqueSpeakers.firstOrNull() ?: return segments
        if (!onlySpeaker.startsWith("Speaker ", ignoreCase = true)) {
            // Do not override a strong enrolled-name match.
            return segments
        }

        val turns = segments.map { (it["text"] ?: "").toString().trim() }
        if (!looksLikeDialogue(turns)) return segments

        val inferred = inferDialogueSpeakers(turns)
        val inferredUnique = inferred.distinct()
        if (inferredUnique.size < 2) return segments

        Log.w(TAG, "Speaker collapse detected ($onlySpeaker); applying dialogue fallback labels")
        return segments.mapIndexed { i, seg ->
            seg.toMutableMap().apply {
                this["speaker"] = inferred[i]
            }
        }
    }

    private fun looksLikeDialogue(turns: List<String>): Boolean {
        if (turns.size >= 3) return true
        if (turns.size < 2) return false

        val qCount = turns.count { isQuestionLike(it) }
        val cueCount = turns.count { isReplyLike(it) }
        return qCount >= 1 || cueCount >= 1
    }

    private fun inferDialogueSpeakers(turns: List<String>): List<String> {
        if (turns.isEmpty()) return emptyList()

        // Collapse fallback should be deterministic and stable.
        // For scripted conversations this avoids heuristic drift.
        return turns.mapIndexed { idx, _ ->
            if (idx % 2 == 0) "Speaker 1" else "Speaker 2"
        }
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

        var norm = 0.0
        for (x in out) norm += (x * x)
        val d = sqrt(norm).toFloat()
        if (d > 0f) {
            for (i in out.indices) out[i] /= d
        }
        return out
    }

    private fun isQuestionLike(text: String): Boolean {
        val t = text.trim().lowercase()
        if (t.contains("?")) return true
        return t.startsWith("when ") ||
            t.startsWith("what ") ||
            t.startsWith("where ") ||
            t.startsWith("why ") ||
            t.startsWith("how ") ||
            t.startsWith("can ") ||
            t.startsWith("could ") ||
            t.startsWith("will ") ||
            t.startsWith("would ") ||
            t.startsWith("do ") ||
            t.startsWith("did ") ||
            t.startsWith("are ") ||
            t.startsWith("is ")
    }

    private fun isReplyLike(text: String): Boolean {
        val t = text.trim().lowercase()
        return t.startsWith("sure") ||
            t.startsWith("yes") ||
            t.startsWith("yeah") ||
            t.startsWith("no") ||
            t.startsWith("okay") ||
            t.startsWith("ok") ||
            t.startsWith("alright") ||
            t.startsWith("fine")
    }

    /**
     * Build a deterministic 128-D voice embedding from PCM waveform.
     * This preserves enrollment/matching behavior without Sherpa x-vectors.
     */
    fun extractXVector(wavPath: String): FloatArray? {
        return try {
            val wave = readWavePcm16(wavPath)
            extractXVectorFromSamples(wave.samples, 0, wave.samples.size)
        } catch (e: Exception) {
            Log.e(TAG, "Voice embedding error: ${e.message}")
            null
        }
    }

    fun extractXVectors(wavPath: String, maxChunks: Int = 0): List<FloatArray> {
        return try {
            val wave = readWavePcm16(wavPath)
            val s = wave.samples
            if (s.size < 1600) return emptyList()

            // Use denser 1.0s windows with 0.5s stride to capture richer voice variation.
            val chunkSize = wave.sampleRate.coerceAtLeast(1600)
            val step = (wave.sampleRate / 2).coerceAtLeast(800)

            data class Candidate(val vec: FloatArray, val rms: Double)
            val candidates = mutableListOf<Candidate>()
            var start = 0
            while (start < s.size) {
                val end = (start + chunkSize).coerceAtMost(s.size)
                val rms = computeRms(s, start, end)
                val v = extractXVectorFromSamples(s, start, end)
                if (v != null) {
                    candidates.add(Candidate(v, rms))
                }
                if (end >= s.size) break
                start += step
            }

            // Keep only voice-active chunks for enrollment robustness.
            val selected = if (candidates.isNotEmpty()) {
                val sorted = candidates.sortedByDescending { it.rms }
                val best = sorted.first().rms
                val floor = best * 0.25
                val targetLimit = if (maxChunks > 0) {
                    maxChunks.coerceAtMost(MAX_ENROLLMENT_CHUNKS_SAFETY)
                } else {
                    MAX_ENROLLMENT_CHUNKS_SAFETY
                }
                sorted
                    .filter { it.rms >= floor }
                    .take(targetLimit)
                    .map { it.vec }
            } else {
                emptyList()
            }

            if (selected.isNotEmpty()) {
                selected
            } else {
                val fallback = mutableListOf<FloatArray>()
                extractXVectorFromSamples(s, 0, s.size)?.let { fallback.add(it) }
                fallback
            }
        } catch (e: Exception) {
            Log.e(TAG, "Voice multi-embedding error: ${e.message}")
            emptyList()
        }
    }

    private fun computeRms(samples: FloatArray, start: Int, end: Int): Double {
        val s0 = start.coerceIn(0, samples.size)
        val e0 = end.coerceIn(s0, samples.size)
        val len = (e0 - s0).coerceAtLeast(1)

        var sum = 0.0
        for (i in s0 until e0) {
            val v = samples[i].toDouble()
            sum += v * v
        }
        return sqrt(sum / len)
    }

    private fun extractXVectorFromSamples(samples: FloatArray, start: Int, end: Int): FloatArray? {
        val s0 = start.coerceIn(0, samples.size)
        val e0 = end.coerceIn(s0, samples.size)
        if (e0 - s0 < 1600) return null

        val dim = 128
        val block = maxOf(1, (e0 - s0) / dim)
        val vec = FloatArray(dim)

        for (i in 0 until dim) {
            val segStart = s0 + i * block
            if (segStart >= e0) break
            val segEnd = minOf(e0, segStart + block)

            var sumAbs = 0.0
            var sum = 0.0
            var zc = 0
            var prev = samples[segStart]

            for (j in segStart until segEnd) {
                val cur = samples[j]
                sumAbs += abs(cur)
                sum += cur
                if ((cur >= 0f) != (prev >= 0f)) zc++
                prev = cur
            }

            val len = (segEnd - segStart).coerceAtLeast(1)
            val energy = (sumAbs / len).toFloat()
            val bias = (sum / len).toFloat()
            val zcr = zc.toFloat() / len.toFloat()

            vec[i] = (energy * 0.80f) + (abs(bias) * 0.10f) + (zcr * 0.10f)
        }

        var norm = 0.0
        for (v in vec) norm += (v * v)
        val denom = sqrt(norm).toFloat()
        if (denom > 0f) {
            for (i in vec.indices) vec[i] /= denom
        }

        return vec
    }

    fun transcribeWav(wavPath: String): String {
        val r = recognizer ?: run {
            Log.e(TAG, "Sherpa recognizer not loaded")
            return ""
        }

        try {
            val wave = readWavePcm16(wavPath)
            val totalSamples = wave.samples.size
            val maxChunkSamples = (wave.sampleRate * ASR_MAX_CHUNK_SECONDS).coerceAtLeast(wave.sampleRate)

            if (totalSamples <= maxChunkSamples) {
                val raw = decodeSamples(r, wave.samples, wave.sampleRate)
                return cleanupTranscript(raw)
            }

            val chunks = mutableListOf<String>()
            var start = 0
            var chunkIndex = 0
            while (start < totalSamples) {
                val end = (start + maxChunkSamples).coerceAtMost(totalSamples)
                val slice = wave.samples.copyOfRange(start, end)
                val raw = decodeSamples(r, slice, wave.sampleRate)
                if (raw.isNotBlank()) {
                    chunks.add(raw)
                }
                chunkIndex++
                Log.i(
                    TAG,
                    "Long-audio decode chunk $chunkIndex: " +
                        "${String.format("%.1f", start.toDouble() / wave.sampleRate)}s-" +
                        "${String.format("%.1f", end.toDouble() / wave.sampleRate)}s"
                )
                start = end
            }

            val merged = chunks.joinToString(" ").trim()
            return cleanupTranscript(merged)

        } catch (e: Exception) {
            Log.e(TAG, "Transcription error: ${e.message}")
            return ""
        }
    }

    private fun decodeSamples(recognizer: OfflineRecognizer, samples: FloatArray, sampleRate: Int): String {
        val stream = recognizer.createStream()
        return try {
            stream.acceptWaveform(samples, sampleRate = sampleRate)
            recognizer.decode(stream)
            recognizer.getResult(stream).text.trim()
        } finally {
            stream.release()
        }
    }

    fun cleanupTranscript(text: String): String {
        if (text.isBlank()) return ""

        var cleaned = text.lowercase().trim()

        cleaned = cleaned.replace("\\b(\\w+)\\s+\\1\\b".toRegex(), "$1")
        cleaned = cleaned.replace("\\b(\\w+)\\s+\\1\\b".toRegex(), "$1")

        val fillerPattern = "\\bthe\\s+(?=(?:the|a|an|uh|um|huh)\\b)".toRegex()
        cleaned = fillerPattern.replace(cleaned, "")

        val noiseWords = setOf("the", "a", "uh", "um", "huh", "eh", "ah")
        val words = cleaned.split("\\s+".toRegex()).toMutableList()

        while (words.isNotEmpty() && words.first() in noiseWords) {
            words.removeAt(0)
        }
        while (words.isNotEmpty() && words.last() in noiseWords) {
            words.removeAt(words.lastIndex)
        }

        val result = mutableListOf<String>()
        for (i in words.indices) {
            val word = words[i]
            if (word in noiseWords) {
                val prev = if (i > 0) words[i - 1] else ""
                val next = if (i < words.size - 1) words[i + 1] else ""
                val keepWords = setOf(
                    "in", "on", "at", "by", "for", "with", "to",
                    "is", "was", "has", "have", "get", "take", "from", "about",
                    "near", "around", "after", "before", "during",
                )
                if (word == "the" && next.isNotEmpty() && next !in noiseWords &&
                    (prev in keepWords || prev.isEmpty() || prev.endsWith("."))
                ) {
                    result.add(word)
                } else if (word == "a" && next.isNotEmpty() && next !in noiseWords) {
                    result.add(word)
                }
            } else {
                result.add(word)
            }
        }

        cleaned = result.joinToString(" ").trim()

        if (cleaned.isNotEmpty()) {
            cleaned = cleaned.replaceFirstChar { it.uppercaseChar() }
            cleaned = cleaned.replace("\\. (\\w)".toRegex()) { match ->
                ". " + match.groupValues[1].uppercase()
            }
        }

        cleaned = cleaned.replace("\\s+".toRegex(), " ").trim()

        return cleaned
    }

    private data class WaveData(
        val samples: FloatArray,
        val sampleRate: Int,
    )

    private fun resolveModelRoot(context: Context): File? {
        val candidates = listOf(
            File(context.filesDir, MODEL_DIR_NAME),
            File(context.getExternalFilesDir(null), MODEL_DIR_NAME),
            File(context.cacheDir, MODEL_DIR_NAME),
        ).filterNotNull()

        candidates.forEach { dir ->
            if (dir.exists() && dir.isDirectory && findWhisperFiles(dir) != null) {
                return dir
            }
        }

        // Some archives extract with an extra nested root directory.
        // Search nearby directories so we reuse an existing model instead of re-downloading.
        val roots = listOfNotNull(
            context.getExternalFilesDir(null),
            context.filesDir,
            context.cacheDir,
        )
        roots.forEach { root ->
            val nested = findModelRootRecursively(root, maxDepth = 4)
            if (nested != null) return nested
        }

        return null
    }

    private fun findModelRootRecursively(root: File, maxDepth: Int): File? {
        if (!root.exists() || !root.isDirectory || maxDepth < 0) return null

        val queue = ArrayDeque<Pair<File, Int>>()
        queue.add(root to 0)

        while (queue.isNotEmpty()) {
            val (dir, depth) = queue.removeFirst()
            if (findWhisperFiles(dir) != null) {
                return dir
            }
            if (depth >= maxDepth) continue

            val children = dir.listFiles() ?: continue
            for (child in children) {
                if (child.isDirectory && !child.name.startsWith(".")) {
                    queue.add(child to (depth + 1))
                }
            }
        }

        return null
    }

    private fun findWhisperFiles(modelRoot: File): Triple<File, File, File>? {
        fun pick(names: List<String>): File? {
            for (name in names) {
                val f = File(modelRoot, name)
                if (f.exists()) return f
            }
            return null
        }

        val encoder = pick(
            listOf(
                "small.en-encoder.int8.onnx",
                "small.en-encoder.onnx",
                "encoder.int8.onnx",
                "encoder.onnx",
            )
        ) ?: return null

        val decoder = pick(
            listOf(
                "small.en-decoder.int8.onnx",
                "small.en-decoder.onnx",
                "decoder.int8.onnx",
                "decoder.onnx",
            )
        ) ?: return null

        val tokens = pick(
            listOf(
                "small.en-tokens.txt",
                "tokens.txt",
            )
        ) ?: return null

        return Triple(encoder, decoder, tokens)
    }

    private fun createRecognizer(modelRoot: File): OfflineRecognizer {
        val (encoder, decoder, tokens) = findWhisperFiles(modelRoot)
            ?: throw Exception("Missing Whisper files in ${modelRoot.absolutePath}")

        val whisper = OfflineWhisperModelConfig(
            encoder = encoder.absolutePath,
            decoder = decoder.absolutePath,
            language = "en",
            task = "transcribe",
            tailPaddings = 1000,
        )

        val model = OfflineModelConfig(
            whisper = whisper,
            tokens = tokens.absolutePath,
            numThreads = 2,
            provider = "cpu",
            modelType = "whisper",
            debug = false,
        )

        val cfg = OfflineRecognizerConfig(
            modelConfig = model,
            decodingMethod = "greedy_search",
        )

        return OfflineRecognizer(config = cfg)
    }

    private fun downloadFileResumable(url: String, targetFile: File) {
        targetFile.parentFile?.mkdirs()

        val existingSize = if (targetFile.exists()) targetFile.length() else 0L
        asrBytesDownloaded = existingSize

        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = 30000
        conn.readTimeout = 60000
        if (existingSize > 0) {
            conn.setRequestProperty("Range", "bytes=$existingSize-")
        }

        conn.connect()
        val responseCode = conn.responseCode
        if (responseCode != 200 && responseCode != 206) {
            if (responseCode == 416) {
                conn.disconnect()
                return
            }
            conn.disconnect()
            throw Exception("Download failed with HTTP $responseCode")
        }

        val contentLength = conn.contentLengthLong
        val total = if (responseCode == 206 && contentLength > 0) contentLength + existingSize else contentLength
        asrTotalBytes = if (total > 0) total else 0L

        val inputStream = BufferedInputStream(conn.inputStream)
        val output = RandomAccessFile(targetFile, "rw")

        if (responseCode == 200) {
            output.setLength(0)
            output.seek(0)
            asrBytesDownloaded = 0L
        } else {
            output.seek(existingSize)
        }

        val buffer = ByteArray(64 * 1024)
        try {
            while (!isAsrPaused) {
                val n = inputStream.read(buffer)
                if (n <= 0) break
                output.write(buffer, 0, n)
                asrBytesDownloaded += n

                if (asrTotalBytes > 0) {
                    val p = ((asrBytesDownloaded * 100L) / asrTotalBytes).toInt().coerceIn(0, 100)
                    asrProgress = p
                    if (asrBytesDownloaded >= asrTotalBytes) {
                        // Avoid waiting indefinitely for server EOF once we already
                        // downloaded the advertised content length.
                        asrProgress = 100
                        asrStage = "extracting"
                        break
                    }
                }
            }
        } finally {
            output.close()
            inputStream.close()
            conn.disconnect()
        }

        if (!isAsrPaused && asrTotalBytes > 0L && asrBytesDownloaded >= asrTotalBytes) {
            asrProgress = 100
        }

        if (!isAsrPaused && asrTotalBytes > 0L && asrBytesDownloaded < asrTotalBytes) {
            throw Exception("Model download interrupted")
        }
    }

    private fun extractTarBz2(archiveFile: File, outputDir: File) {
        outputDir.mkdirs()

        BufferedInputStream(FileInputStream(archiveFile)).use { fis ->
            BZip2CompressorInputStream(fis).use { bzis ->
                TarArchiveInputStream(bzis).use { tis ->
                    val buffer = ByteArray(1024 * 1024)
                    var hasEncoder = false
                    var hasDecoder = false
                    var hasTokens = false

                    fun updateFound(entryName: String) {
                        val lower = entryName.lowercase()
                        if (lower.endsWith("encoder.onnx") || lower.endsWith("encoder.int8.onnx")) {
                            hasEncoder = true
                        }
                        if (lower.endsWith("decoder.onnx") || lower.endsWith("decoder.int8.onnx")) {
                            hasDecoder = true
                        }
                        if (lower.endsWith("tokens.txt")) {
                            hasTokens = true
                        }
                    }

                    // Initialize presence from already-existing files to speed up retries.
                    findWhisperFiles(outputDir)?.let {
                        hasEncoder = true
                        hasDecoder = true
                        hasTokens = true
                    }

                    while (true) {
                        if (hasEncoder && hasDecoder && hasTokens) {
                            // Required assets are already available; skip remaining archive scan.
                            break
                        }

                        val entry = tis.nextTarEntry ?: break

                        if (!entry.isDirectory && !isWhisperAssetEntry(entry.name)) {
                            continue
                        }

                        val outFile = File(outputDir, entry.name)
                        val canonicalBase = outputDir.canonicalPath + File.separator
                        val canonicalOut = outFile.canonicalPath
                        if (!canonicalOut.startsWith(canonicalBase)) {
                            throw Exception("Unsafe archive entry: ${entry.name}")
                        }

                        if (entry.isDirectory) {
                            outFile.mkdirs()
                            continue
                        }

                        outFile.parentFile?.mkdirs()

                        // Skip rewriting files that already match size to speed up retries.
                        if (outFile.exists() && entry.size >= 0L && outFile.length() == entry.size) {
                            updateFound(outFile.name)
                            continue
                        }

                        FileOutputStream(outFile).use { rawFos ->
                            BufferedOutputStream(rawFos, 1024 * 1024).use { fos ->
                            while (true) {
                                val n = tis.read(buffer)
                                if (n <= 0) break
                                fos.write(buffer, 0, n)
                            }
                            }
                        }

                        updateFound(outFile.name)
                    }
                }
            }
        }
    }

    private fun isWhisperAssetEntry(entryName: String): Boolean {
        val lower = entryName.lowercase()
        return lower.endsWith("encoder.onnx") ||
            lower.endsWith("encoder.int8.onnx") ||
            lower.endsWith("decoder.onnx") ||
            lower.endsWith("decoder.int8.onnx") ||
            lower.endsWith("tokens.txt")
    }

    private fun ensureModelLayout(parentDir: File) {
        val modelDir = File(parentDir, MODEL_DIR_NAME)
        if (modelDir.exists() && modelDir.isDirectory) return

        val encoderCandidates = listOf(
            "small.en-encoder.int8.onnx",
            "small.en-encoder.onnx",
            "encoder.int8.onnx",
            "encoder.onnx",
        )
        val decoderCandidates = listOf(
            "small.en-decoder.int8.onnx",
            "small.en-decoder.onnx",
            "decoder.int8.onnx",
            "decoder.onnx",
        )
        val tokenCandidates = listOf("small.en-tokens.txt", "tokens.txt")

        val encoder = encoderCandidates.firstNotNullOfOrNull { name ->
            val f = File(parentDir, name)
            if (f.exists()) f else null
        }
        val decoder = decoderCandidates.firstNotNullOfOrNull { name ->
            val f = File(parentDir, name)
            if (f.exists()) f else null
        }
        val tokens = tokenCandidates.firstNotNullOfOrNull { name ->
            val f = File(parentDir, name)
            if (f.exists()) f else null
        }

        if (encoder != null && decoder != null && tokens != null) {
            modelDir.mkdirs()
            encoder.renameTo(File(modelDir, encoder.name))
            decoder.renameTo(File(modelDir, decoder.name))
            tokens.renameTo(File(modelDir, tokens.name))
        }
    }

    private fun readWavePcm16(wavPath: String): WaveData {
        RandomAccessFile(wavPath, "r").use { raf ->
            val riff = ByteArray(4)
            raf.readFully(riff)
            if (String(riff) != "RIFF") throw Exception("Invalid WAV header")

            raf.skipBytes(4)
            val wave = ByteArray(4)
            raf.readFully(wave)
            if (String(wave) != "WAVE") throw Exception("Invalid WAV type")

            var sampleRate = 16000
            var channels = 1
            var bitsPerSample = 16
            var dataOffset = 0L
            var dataSize = 0

            while (raf.filePointer < raf.length() - 8) {
                val chunkIdBytes = ByteArray(4)
                raf.readFully(chunkIdBytes)
                val chunkId = String(chunkIdBytes)
                val chunkSize = readIntLE(raf)

                when (chunkId) {
                    "fmt " -> {
                        val audioFormat = readShortLE(raf)
                        channels = readShortLE(raf)
                        sampleRate = readIntLE(raf)
                        raf.skipBytes(6)
                        bitsPerSample = readShortLE(raf)
                        if (chunkSize > 16) raf.skipBytes(chunkSize - 16)
                        if (audioFormat != 1) throw Exception("Only PCM WAV is supported")
                    }

                    "data" -> {
                        dataOffset = raf.filePointer
                        dataSize = chunkSize
                        raf.skipBytes(chunkSize)
                    }

                    else -> raf.skipBytes(chunkSize)
                }

                if ((chunkSize and 1) == 1 && raf.filePointer < raf.length()) {
                    raf.skipBytes(1)
                }
            }

            if (dataOffset <= 0 || dataSize <= 0) {
                throw Exception("WAV data chunk not found")
            }
            if (bitsPerSample != 16) {
                throw Exception("Only 16-bit WAV is supported")
            }

            raf.seek(dataOffset)
            val raw = ByteArray(dataSize)
            raf.readFully(raw)

            val shorts = ShortArray(dataSize / 2)
            ByteBuffer.wrap(raw)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asShortBuffer()
                .get(shorts)

            val mono = FloatArray(shorts.size / channels)
            var m = 0
            var i = 0
            while (i < shorts.size) {
                val s = shorts[i].toInt()
                mono[m++] = (s / 32768.0f).coerceIn(-1.0f, 1.0f)
                i += channels
            }

            return WaveData(samples = mono, sampleRate = sampleRate)
        }
    }

    private fun readShortLE(raf: RandomAccessFile): Int {
        val b0 = raf.read()
        val b1 = raf.read()
        return (b1 shl 8) or b0
    }

    private fun readIntLE(raf: RandomAccessFile): Int {
        val b0 = raf.read()
        val b1 = raf.read()
        val b2 = raf.read()
        val b3 = raf.read()
        return (b3 shl 24) or (b2 shl 16) or (b1 shl 8) or b0
    }

    fun getStatus(): Map<String, Any> = mapOf(
        "asr" to mapOf(
            "ready" to isAsrReady(),
            "initializing" to isAsrInitializing,
            "paused" to isAsrPaused,
            "stage" to asrStage,
            "error" to (asrInitError ?: ""),
            "progress" to asrProgress,
            "downloaded_mb" to (asrBytesDownloaded / 1024 / 1024).toInt(),
            "total_mb" to (asrTotalBytes / 1024 / 1024).toInt(),
            "model_name" to MODEL_DIR_NAME,
        ),
        "spk" to mapOf(
            "ready" to isSpkReady(),
            "initializing" to isSpkInitializing,
            "paused" to isSpkPaused,
            "error" to (spkInitError ?: ""),
            "progress" to spkProgress,
            "downloaded_mb" to (spkBytesDownloaded / 1024 / 1024).toInt(),
            "total_mb" to (spkTotalBytes / 1024 / 1024).toInt(),
            "model_name" to "speaker-embedding-compatible",
        ),
        "ready" to isAsrReady(),
    )
}