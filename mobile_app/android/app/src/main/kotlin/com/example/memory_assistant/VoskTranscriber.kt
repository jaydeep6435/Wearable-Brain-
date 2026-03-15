package com.example.memory_assistant

import android.content.Context
import android.util.Log
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.SpeakerModel
import org.json.JSONObject
import org.json.JSONArray
import java.io.*
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream

/**
 * VoskTranscriber — High-accuracy offline WAV transcription.
 *
 * Uses the lgraph model (128MB) for much better accuracy than
 * the small models. Downloads on first use, cached permanently.
 *
 * After recording stops, call transcribeWav() on a background thread
 * to process the complete WAV file.
 */
object VoskTranscriber {

    private const val TAG = "WBrain.Vosk"
    private const val MODEL_DIR_NAME = "vosk-model-en-in-0.5"

    // Higher-accuracy Indian-English model (larger, server-grade)
    private const val MODEL_URL =
        "https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip"

    // Speaker identification model (~12MB)
    private const val SPK_MODEL_URL =
        "https://alphacephei.com/vosk/models/vosk-model-spk-0.4.zip"
    private const val SPK_MODEL_DIR_NAME = "vosk-model-spk"

    private var model: Model? = null
    var spkModel: SpeakerModel? = null

    // ASR State
    var isAsrInitializing = false
    var isAsrPaused = false
    var asrInitError: String? = null
    var asrProgress = 0
    var asrBytesDownloaded = 0L
    var asrTotalBytes = 0L

    // SPK State
    var isSpkInitializing = false
    var isSpkPaused = false
    var spkInitError: String? = null
    var spkProgress = 0
    var spkBytesDownloaded = 0L
    var spkTotalBytes = 0L

    private var asrDownloadThread: Thread? = null
    private var spkDownloadThread: Thread? = null

    private const val RECOGNITION_CHUNK_BYTES = 8192
    private const val MIN_SPEECH_RMS = 35.0
    private const val SPEECH_HANGOVER_CHUNKS = 14
    private const val SPEECH_THRESHOLD_MULTIPLIER = 1.6
    private const val MAX_SPEECH_RMS_THRESHOLD = 220.0
    private const val NOISE_UPDATE_RATIO = 1.35

    fun isAsrReady(): Boolean = model != null
    fun isSpkReady(): Boolean = spkModel != null
    fun isReady(): Boolean = isAsrReady()

    private fun computeChunkRms(buffer: ByteArray, bytesRead: Int): Double {
        if (bytesRead < 2) return 0.0
        var sumSq = 0.0
        var samples = 0
        var i = 0
        while (i + 1 < bytesRead) {
            val sample = ((buffer[i].toInt() and 0xFF) or (buffer[i + 1].toInt() shl 8)).toShort().toInt()
            sumSq += sample.toDouble() * sample.toDouble()
            samples++
            i += 2
        }
        if (samples == 0) return 0.0
        return Math.sqrt(sumSq / samples)
    }

    // ── ASR MODEL INIT ──────────────────────────────────────

    fun initAsr(context: Context, onComplete: ((Boolean) -> Unit)? = null) {
        if (model != null) { onComplete?.invoke(true); return }
        if (isAsrInitializing && !isAsrPaused) { onComplete?.invoke(false); return }

        isAsrInitializing = true
        isAsrPaused = false
        asrInitError = null
        Log.i(TAG, "━━━ Initializing Vosk ASR (Speech) ━━━")

        asrDownloadThread = Thread {
            try {
                val modelDir = File(context.filesDir, MODEL_DIR_NAME)
                val zipFile = File(context.cacheDir, "vosk_model.zip")

                if (modelDir.exists() && modelDir.isDirectory &&
                    modelDir.listFiles()?.isNotEmpty() == true && !zipFile.exists()
                ) {
                    Log.i(TAG, "  ASR Model cached: ${modelDir.absolutePath}")
                } else {
                    downloadFileResumable(MODEL_URL, zipFile, isAsr = true)
                    if (isAsrPaused) {
                        Log.i(TAG, "  ASR Download paused")
                        return@Thread
                    }
                    Log.i(TAG, "  ✓ ASR Download complete. Extracting...")
                    try {
                        extractZip(zipFile, modelDir)
                        zipFile.delete()
                        Log.i(TAG, "  ✓ ASR Extraction complete")
                    } catch (e: Exception) {
                        zipFile.delete()
                        modelDir.deleteRecursively()
                        throw Exception("Model archive corrupted. Press Retry to restart.")
                    }
                }

                val actualModelDir = findModelDir(modelDir)
                    ?: throw Exception("Model files not found in $modelDir")

                model = Model(actualModelDir.absolutePath)
                isAsrInitializing = false
                Log.i(TAG, "  ✓ ASR model ready")
                onComplete?.invoke(true)

            } catch (e: Exception) {
                if (!isAsrPaused) {
                    isAsrInitializing = false
                    asrInitError = e.message
                    Log.e(TAG, "  ✗ ASR init failed: ${e.message}")
                    onComplete?.invoke(false)
                }
            }
        }
        asrDownloadThread?.start()
    }

    fun pauseAsr() { isAsrPaused = true }
    fun resumeAsr(context: Context) { if (isAsrPaused) initAsr(context) }
    fun retryAsr(context: Context) {
        asrInitError = null
        isAsrInitializing = false
        isAsrPaused = false
        initAsr(context)
    }

    // ── SPK MODEL INIT ──────────────────────────────────────

    fun initSpk(context: Context, onComplete: ((Boolean) -> Unit)? = null) {
        if (spkModel != null) { onComplete?.invoke(true); return }
        if (isSpkInitializing && !isSpkPaused) { onComplete?.invoke(false); return }

        isSpkInitializing = true
        isSpkPaused = false
        spkInitError = null
        Log.i(TAG, "━━━ Initializing Vosk SPK (Speaker) ━━━")

        spkDownloadThread = Thread {
            try {
                val spkModelDir = File(context.filesDir, SPK_MODEL_DIR_NAME)
                val spkZipFile = File(context.cacheDir, "vosk_spk_model.zip")

                if (spkModelDir.exists() && spkModelDir.isDirectory &&
                    spkModelDir.listFiles()?.isNotEmpty() == true && !spkZipFile.exists()
                ) {
                    Log.i(TAG, "  SPK Model cached: ${spkModelDir.absolutePath}")
                } else {
                    downloadFileResumable(SPK_MODEL_URL, spkZipFile, isAsr = false)
                    if (isSpkPaused) {
                        Log.i(TAG, "  SPK Download paused")
                        return@Thread
                    }
                    Log.i(TAG, "  ✓ SPK Download complete. Extracting...")
                    try {
                        extractZip(spkZipFile, spkModelDir)
                        spkZipFile.delete()
                        Log.i(TAG, "  ✓ SPK Extraction complete")
                    } catch (e: Exception) {
                        spkZipFile.delete()
                        spkModelDir.deleteRecursively()
                        throw Exception("Model archive corrupted. Press Retry to restart.")
                    }
                }

                val actualSpkDir = findModelDir(spkModelDir)
                    ?: throw Exception("Speaker model files not found")

                spkModel = SpeakerModel(actualSpkDir.absolutePath)
                isSpkInitializing = false
                Log.i(TAG, "  ✓ SPK model ready")
                onComplete?.invoke(true)

            } catch (e: Exception) {
                if (!isSpkPaused) {
                    isSpkInitializing = false
                    spkInitError = e.message
                    Log.e(TAG, "  ✗ SPK init failed: ${e.message}")
                    onComplete?.invoke(false)
                }
            }
        }
        spkDownloadThread?.start()
    }

    fun pauseSpk() { isSpkPaused = true }
    fun resumeSpk(context: Context) { if (isSpkPaused) initSpk(context) }
    fun retrySpk(context: Context) {
        spkInitError = null
        isSpkInitializing = false
        isSpkPaused = false
        initSpk(context)
    }

    /**
     * Transcribe WAV file WITH speaker diarization.
     * Returns a list of segments: [{speaker: "Name", text: "..."}]
     * Must be called from background thread.
     *
     * KEY DESIGN DECISIONS:
     * - Buffer size = 4096 bytes (~128ms at 16kHz). This is the OPTIMAL size for Vosk.
     *   Larger buffers (8K-32K) cause Vosk to batch internally and can miss phrase boundaries.
     * - Properly reads WAV header to find the exact data chunk offset.
     * - Post-merges consecutive segments from the same speaker for cleaner output.
     */
    fun transcribeWavWithSpeakers(wavPath: String, context: Context): List<Map<String, Any>> {
        val m = model ?: run {
            Log.e(TAG, "Model not loaded")
            return emptyList()
        }

        Log.i(TAG, "━━━ Vosk Diarized Transcription ━━━━━━━")
        Log.i(TAG, "  File: $wavPath")
        val startMs = System.currentTimeMillis()

        // Load the latest speaker profiles (voiceprints) before diarization
        SpeakerEngine.reloadProfiles(context)
        Log.i(TAG, "  Speaker profiles loaded: ${SpeakerEngine.profileCount()}")
        SpeakerEngine.resetSession()

        try {
            val file = File(wavPath)
            val fileSize = file.length()
            val headerInfo = readWavHeader(wavPath)
            val sampleRate = headerInfo["sampleRate"] ?: 16000
            val channels = headerInfo["channels"] ?: 1
            val bitsPerSample = headerInfo["bitsPerSample"] ?: 16
            val dataOffset = headerInfo["dataOffset"] ?: 44

            val durationSec = (fileSize - dataOffset).toDouble() / (sampleRate * channels * bitsPerSample / 8)
            Log.i(TAG, "  WAV: ${sampleRate}Hz, ${channels}ch, ${bitsPerSample}bit, ${String.format("%.1f", durationSec)}s, ${fileSize/1024}KB")
            Log.i(TAG, "  SpkModel=${spkModel != null}, DataOffset=$dataOffset")

            // Create recognizer WITH speaker model for x-vector extraction
            val recognizer = if (spkModel != null) {
                Log.i(TAG, "  ✓ Using Speaker Model for diarization")
                Recognizer(m, sampleRate.toFloat(), spkModel)
            } else {
                Log.w(TAG, "  ⚠ No Speaker Model — will use energy-based diarization")
                Recognizer(m, sampleRate.toFloat())
            }
            recognizer.setWords(true)

            val fis = FileInputStream(wavPath)
            fis.skip(dataOffset.toLong())

            // Larger chunk gives more context in noisy Bluetooth speech
            val buffer = ByteArray(RECOGNITION_CHUNK_BYTES)
            val rawSegments = mutableListOf<Map<String, Any>>()
            var totalBytes = 0
            var segCount = 0

            // Lightweight VAD gate to avoid feeding long silence/noise runs
            var noiseFloorRms = 20.0
            var noiseFloorSamples = 0
            var speechHangover = 0
            var skippedSilentChunks = 0

            var currentChunkEnergy = 0.0
            var currentChunkSamples = 0

            while (true) {
                val bytesRead = fis.read(buffer)
                if (bytesRead <= 0) break
                totalBytes += bytesRead

                val chunkRms = computeChunkRms(buffer, bytesRead)
                if (chunkRms > 0.0) {
                    val allowBootstrap = noiseFloorSamples < 6
                    val lowEnergyChunk = chunkRms <= (noiseFloorRms * NOISE_UPDATE_RATIO)
                    if (allowBootstrap || lowEnergyChunk) {
                        noiseFloorRms = if (noiseFloorSamples == 0) {
                            chunkRms
                        } else {
                            (noiseFloorRms * 0.94) + (chunkRms * 0.06)
                        }
                        noiseFloorSamples++
                    }
                }

                val speechThreshold = minOf(
                    MAX_SPEECH_RMS_THRESHOLD,
                    maxOf(MIN_SPEECH_RMS, noiseFloorRms * SPEECH_THRESHOLD_MULTIPLIER),
                )
                val speechLike = chunkRms >= speechThreshold
                if (speechLike) {
                    speechHangover = SPEECH_HANGOVER_CHUNKS
                } else if (speechHangover > 0) {
                    speechHangover--
                }

                if (!speechLike && speechHangover == 0) {
                    skippedSilentChunks++
                    continue
                }

                // Calculate energy (sum of squares) of this chunk for diarization
                var chunkEnergySum = 0.0
                for (i in 0 until bytesRead - 1 step 2) {
                    val sample = ((buffer[i].toInt() and 0xFF) or (buffer[i + 1].toInt() shl 8)).toShort()
                    chunkEnergySum += sample.toDouble() * sample.toDouble()
                }
                val samplesInChunk = bytesRead / 2
                currentChunkEnergy += chunkEnergySum
                currentChunkSamples += samplesInChunk

                if (recognizer.acceptWaveForm(buffer, bytesRead)) {
                    val resultJson = recognizer.result
                    val energy = if (currentChunkSamples > 0)
                        Math.sqrt(currentChunkEnergy / currentChunkSamples) else 0.0

                    val segment = parseSegmentRaw(resultJson, energy)
                    if (segment != null) {
                        rawSegments.add(segment)
                        segCount++
                        Log.d(TAG, "  [$segCount] [${segment["speaker"]}] energy=${String.format("%.0f", energy)} '${(segment["text"] as String).take(60)}'")
                    }

                    currentChunkEnergy = 0.0
                    currentChunkSamples = 0
                }
            }

            // Get final result — this contains any remaining unprocessed text
            val finalEnergy = if (currentChunkSamples > 0)
                Math.sqrt(currentChunkEnergy / currentChunkSamples) else 0.0
            val finalJson = recognizer.finalResult
            val finalSeg = parseSegmentRaw(finalJson, finalEnergy)
            if (finalSeg != null) {
                rawSegments.add(finalSeg)
                segCount++
                Log.d(TAG, "  [FINAL] [${finalSeg["speaker"]}] '${(finalSeg["text"] as String).take(60)}'")
            }

            recognizer.close()
            fis.close()

            Log.i(TAG, "  VAD gate: noise=${String.format("%.1f", noiseFloorRms)}, skipped=$skippedSilentChunks chunks")

            // ★ If ALL segments have same speaker label "Speaker" (no x-vectors worked),
            // use energy-based diarization as fallback
            val uniqueSpeakers = rawSegments.map { it["speaker"] }.toSet()
            if (uniqueSpeakers.size <= 1 && rawSegments.size > 1) {
                Log.i(TAG, "  ★ X-vector diarization failed — using ENERGY-BASED fallback")
                applyEnergyDiarization(rawSegments)
            }

            // ★ POST-PROCESS: Merge consecutive segments from the same speaker
            val mergedSegments = mergeConsecutiveSpeakerSegments(rawSegments)

            // ★ Apply transcript cleanup ONLY on the final merged text, NOT per-segment
            val cleanedSegments = mergedSegments.map { seg ->
                val cleanedText = cleanupTranscript(seg["text"].toString())
                if (cleanedText.isNotEmpty()) {
                    mapOf("speaker" to seg["speaker"]!!, "text" to cleanedText)
                } else null
            }.filterNotNull()

            val filteredSegments = cleanedSegments.filter { seg ->
                !isLowInformationSegment(seg["text"].toString())
            }.ifEmpty { cleanedSegments }

            val stabilizedSegments = stabilizeSpeakerLabels(filteredSegments, maxAnonymousSpeakers = 2)

            val elapsed = System.currentTimeMillis() - startMs
            val totalText = stabilizedSegments.joinToString(" ") { it["text"].toString() }
            Log.i(TAG, "  ✓ ${rawSegments.size} raw → ${mergedSegments.size} merged → ${cleanedSegments.size} cleaned")
            Log.i(TAG, "  ✓ ${totalBytes/1024}KB processed in ${elapsed}ms")
            Log.i(TAG, "  ✓ Speakers: ${stabilizedSegments.map { it["speaker"] }.toSet()}")
            if (totalText.length < 300) Log.i(TAG, "  ✓ Text: '$totalText'")
            Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            return stabilizedSegments

        } catch (e: Exception) {
            Log.e(TAG, "  ✗ Diarized transcription error: ${e.message}")
            e.printStackTrace()
            return emptyList()
        }
    }

    /**
     * Merge consecutive segments from the same speaker into a single segment.
     * This produces cleaner output: instead of
     *   Speaker 1: "Hello"
     *   Speaker 1: "how are you"
     * We get:
     *   Speaker 1: "Hello how are you"
     */
    private fun mergeConsecutiveSpeakerSegments(
        segments: List<Map<String, Any>>
    ): List<Map<String, Any>> {
        if (segments.isEmpty()) return segments

        val merged = mutableListOf<Map<String, Any>>()
        var currentSpeaker = segments[0]["speaker"] as String
        var currentText = StringBuilder(segments[0]["text"] as String)

        for (i in 1 until segments.size) {
            val speaker = segments[i]["speaker"] as String
            val text = segments[i]["text"] as String

            if (speaker == currentSpeaker) {
                // Same speaker — append text
                currentText.append(" ").append(text)
            } else {
                // Different speaker — save current and start new
                merged.add(mapOf("speaker" to currentSpeaker, "text" to currentText.toString()))
                currentSpeaker = speaker
                currentText = StringBuilder(text)
            }
        }

        // Don't forget the last segment
        merged.add(mapOf("speaker" to currentSpeaker, "text" to currentText.toString()))
        return merged
    }

    private fun stabilizeSpeakerLabels(
        segments: List<Map<String, Any>>,
        maxAnonymousSpeakers: Int = 2,
    ): List<Map<String, Any>> {
        if (segments.isEmpty()) return segments

        val speakers = segments.map { it["speaker"].toString() }.distinct()
        val hasNamedSpeaker = speakers.any { !it.matches(Regex("^Speaker\\s+\\d+$")) }

        var working = segments

        if (!hasNamedSpeaker && speakers.size > maxAnonymousSpeakers) {
            val scoreBySpeaker = mutableMapOf<String, Int>()
            for (seg in segments) {
                val spk = seg["speaker"].toString()
                val text = seg["text"].toString()
                scoreBySpeaker[spk] = scoreBySpeaker.getOrDefault(spk, 0) + text.length
            }

            val keep = scoreBySpeaker.entries
                .sortedByDescending { it.value }
                .take(maxAnonymousSpeakers)
                .map { it.key }
                .toSet()

            val remapped = mutableListOf<Map<String, Any>>()
            for (i in segments.indices) {
                val seg = segments[i]
                val current = seg["speaker"].toString()
                if (current in keep) {
                    remapped.add(seg)
                    continue
                }

                val prev = if (i > 0) segments[i - 1]["speaker"].toString() else ""
                val next = if (i < segments.lastIndex) segments[i + 1]["speaker"].toString() else ""
                val replacement = when {
                    prev in keep -> prev
                    next in keep -> next
                    else -> keep.first()
                }

                remapped.add(mapOf(
                    "speaker" to replacement,
                    "text" to seg["text"].toString(),
                ))
            }

            working = mergeConsecutiveSpeakerSegments(remapped)
        }

        if (!hasNamedSpeaker) {
            val map = linkedMapOf<String, String>()
            var index = 1
            working = working.map { seg ->
                val old = seg["speaker"].toString()
                val normalized = map.getOrPut(old) {
                    val label = "Speaker $index"
                    index += 1
                    label
                }
                mapOf(
                    "speaker" to normalized,
                    "text" to seg["text"].toString(),
                )
            }
        }

        return mergeConsecutiveSpeakerSegments(working)
    }

    private fun isLowInformationSegment(text: String): Boolean {
        val words = text.lowercase().split("\\s+".toRegex()).filter { it.isNotBlank() }
        if (words.isEmpty()) return true
        val filler = setOf("the", "it", "be", "could", "uh", "um", "hmm", "ah")
        val meaningful = words.count { it.length >= 3 && it !in filler }
        val fillerCount = words.count { it in filler }

        if (words.size <= 2 && meaningful == 0) return true
        if (meaningful == 0 && fillerCount >= 2) return true
        if (words.size >= 4 && (fillerCount.toDouble() / words.size) > 0.7) return true

        return false
    }

    /**
     * Extract the average x-vector from a WAV file for voice enrollment.
     * Returns the 128-dimensional speaker embedding or null.
     */
    fun extractXVector(wavPath: String): FloatArray? {
        if (model == null) {
            Log.e(TAG, "extractXVector: ASR model not loaded! Download it from Settings first.")
            return null
        }
        if (spkModel == null) {
            Log.e(TAG, "extractXVector: Speaker model not loaded! Download it from Settings first.")
            return null
        }
        val m = model!!
        val sm = spkModel!!

        try {
            val headerInfo = readWavHeader(wavPath)
            val sampleRate = headerInfo["sampleRate"] ?: 16000
            val dataOffset = headerInfo["dataOffset"] ?: 44

            val recognizer = Recognizer(m, sampleRate.toFloat(), sm)
            recognizer.setWords(true)

            val fis = FileInputStream(wavPath)
            fis.skip(dataOffset.toLong())

            val buffer = ByteArray(4096)
            val xvectors = mutableListOf<FloatArray>()

            while (true) {
                val bytesRead = fis.read(buffer)
                if (bytesRead <= 0) break

                if (recognizer.acceptWaveForm(buffer, bytesRead)) {
                    val json = recognizer.result
                    val obj = JSONObject(json)
                    val spkArray = obj.optJSONArray("spk")
                    if (spkArray != null && spkArray.length() > 0) {
                        xvectors.add(FloatArray(spkArray.length()) { spkArray.getDouble(it).toFloat() })
                    }
                }
            }

            // Also get final result
            val finalJson = recognizer.finalResult
            val finalObj = JSONObject(finalJson)
            val finalSpk = finalObj.optJSONArray("spk")
            if (finalSpk != null && finalSpk.length() > 0) {
                xvectors.add(FloatArray(finalSpk.length()) { finalSpk.getDouble(it).toFloat() })
            }

            recognizer.close()
            fis.close()

            if (xvectors.isEmpty()) {
                Log.w(TAG, "No x-vectors extracted from enrollment audio")
                return null
            }

            // Average all x-vectors for a robust profile
            val dim = xvectors[0].size
            val avg = FloatArray(dim)
            for (xv in xvectors) {
                for (i in xv.indices) avg[i] += xv[i]
            }
            for (i in avg.indices) avg[i] /= xvectors.size.toFloat()

            Log.i(TAG, "Extracted ${xvectors.size} x-vectors (dim=${dim}) for enrollment")
            return avg

        } catch (e: Exception) {
            Log.e(TAG, "X-vector extraction error: ${e.message}")
            return null
        }
    }

    /**
     * Parse a Vosk result JSON and extract text + speaker x-vector.
     */
    /**
     * Parse a Vosk result JSON into a segment map WITHOUT cleaning up text.
     * Text cleanup is done later on the merged output, not per-segment.
     * This prevents losing valid speech that's mixed with filler words.
     */
    private fun parseSegmentRaw(json: String, energy: Double): Map<String, Any>? {
        try {
            val obj = JSONObject(json)
            val text = obj.optString("text", "").trim()
            if (text.isEmpty()) return null

            val words = text.lowercase().split("\\s+".toRegex()).filter { it.isNotBlank() }
            val isShortSegment = words.size <= 3

            val spkArray = obj.optJSONArray("spk")
            val speaker = if (spkArray != null && spkArray.length() > 0) {
                val xvector = FloatArray(spkArray.length()) { spkArray.getDouble(it).toFloat() }
                Log.d(TAG, "  [xvec] dim=${xvector.size}, norm=${String.format("%.2f", xvector.map { it * it }.sum())}")
                if (isShortSegment) {
                    SpeakerEngine.matchSessionSpeakerOnly(xvector)
                        ?: SpeakerEngine.dominantSessionSpeaker()
                        ?: "Speaker 1"
                } else {
                    SpeakerEngine.getSessionSpeaker(xvector)
                }
            } else {
                Log.d(TAG, "  [xvec] NONE — no speaker vector in this segment")
                SpeakerEngine.dominantSessionSpeaker() ?: "Speaker 1"
            }

            return mapOf(
                "speaker" to speaker,
                "text" to text,
                "energy" to energy
            )
        } catch (e: Exception) {
            Log.e(TAG, "  Error parsing segment: ${e.message}")
            return null
        }
    }

    /**
     * Energy-based diarization fallback.
     * When x-vectors don't produce distinct speakers, we use audio energy
     * (RMS amplitude) to identify speakers:
     *   - Person WEARING the BT headset → CLOSE → LOUDER → higher energy
     *   - Other person → FAR → QUIETER → lower energy
     * 
     * We split at the median energy: above = "Speaker 1", below = "Speaker 2"
     */
    private fun applyEnergyDiarization(segments: MutableList<Map<String, Any>>) {
        if (segments.size < 2) return

        // Collect energies
        val energies = segments.map { (it["energy"] as? Double) ?: 0.0 }
        val sorted = energies.sorted()
        val median = sorted[sorted.size / 2]

        Log.i(TAG, "  [Energy] min=${String.format("%.0f", sorted.first())}, median=${String.format("%.0f", median)}, max=${String.format("%.0f", sorted.last())}")

        // Only apply if there's meaningful variation
        val maxE = sorted.last()
        val minE = sorted.first()
        if (maxE <= 0 || (maxE - minE) / maxE < 0.15) {
            Log.i(TAG, "  [Energy] Not enough variation — keeping single speaker")
            return
        }

        // Reassign speaker labels based on energy
        for (i in segments.indices) {
            val energy = (segments[i]["energy"] as? Double) ?: 0.0
            val label = if (energy >= median) "Speaker 1" else "Speaker 2"
            val mutable = segments[i].toMutableMap()
            mutable["speaker"] = label
            segments[i] = mutable
            Log.d(TAG, "  [Energy] Seg $i: energy=${String.format("%.0f", energy)} → $label")
        }
    }

    // Keep old method for backward compatibility (used by transcribeWav non-diarized)
    private fun parseSegmentWithSpeaker(json: String): Map<String, Any>? {
        return parseSegmentRaw(json, 0.0)
    }

    /**
     * Transcribe WAV and return a single merged string (backward-compatible).
     */
    fun transcribeWav(wavPath: String): String {
        val m = model ?: run {
            Log.e(TAG, "Model not loaded")
            return ""
        }

        Log.i(TAG, "━━━ Vosk Transcribing ━━━━━━━━━━━━━━━━━")
        Log.i(TAG, "  File: $wavPath")
        val startMs = System.currentTimeMillis()

        try {
            val file = File(wavPath)
            Log.i(TAG, "  File size: ${file.length() / 1024}KB")

            val headerInfo = readWavHeader(wavPath)
            val sampleRate = headerInfo["sampleRate"] ?: 16000
            val dataOffset = headerInfo["dataOffset"] ?: 44
            Log.i(TAG, "  WAV: ${sampleRate}Hz, ${headerInfo["channels"]}ch, ${headerInfo["bitsPerSample"]}bit, offset=$dataOffset")

            val recognizer = Recognizer(m, sampleRate.toFloat())
            recognizer.setWords(true)

            val fis = FileInputStream(wavPath)
            fis.skip(dataOffset.toLong())

            // Larger chunk gives better acoustic context in noisy recordings
            val buffer = ByteArray(RECOGNITION_CHUNK_BYTES)
            val segments = mutableListOf<String>()
            var totalBytes = 0

            // Lightweight VAD gate to avoid silence/noise dominating decoding
            var noiseFloorRms = 20.0
            var noiseFloorSamples = 0
            var speechHangover = 0
            var skippedSilentChunks = 0

            while (true) {
                val bytesRead = fis.read(buffer)
                if (bytesRead <= 0) break
                totalBytes += bytesRead

                val chunkRms = computeChunkRms(buffer, bytesRead)
                if (chunkRms > 0.0) {
                    val allowBootstrap = noiseFloorSamples < 6
                    val lowEnergyChunk = chunkRms <= (noiseFloorRms * NOISE_UPDATE_RATIO)
                    if (allowBootstrap || lowEnergyChunk) {
                        noiseFloorRms = if (noiseFloorSamples == 0) {
                            chunkRms
                        } else {
                            (noiseFloorRms * 0.94) + (chunkRms * 0.06)
                        }
                        noiseFloorSamples++
                    }
                }

                val speechThreshold = minOf(
                    MAX_SPEECH_RMS_THRESHOLD,
                    maxOf(MIN_SPEECH_RMS, noiseFloorRms * SPEECH_THRESHOLD_MULTIPLIER),
                )
                val speechLike = chunkRms >= speechThreshold
                if (speechLike) {
                    speechHangover = SPEECH_HANGOVER_CHUNKS
                } else if (speechHangover > 0) {
                    speechHangover--
                }

                if (!speechLike && speechHangover == 0) {
                    skippedSilentChunks++
                    continue
                }

                if (recognizer.acceptWaveForm(buffer, bytesRead)) {
                    val text = extractText(recognizer.result)
                    if (text.isNotBlank()) {
                        segments.add(text)
                        Log.d(TAG, "  Seg: '$text'")
                    }
                }
            }

            val finalText = extractText(recognizer.finalResult)
            if (finalText.isNotBlank()) {
                segments.add(finalText)
                Log.d(TAG, "  Final: '$finalText'")
            }

            recognizer.close()
            fis.close()

            val elapsed = System.currentTimeMillis() - startMs
            val rawResult = segments.joinToString(" ")
                .replace("\\s+".toRegex(), " ")
                .trim()
            // ★ Clean up Vosk artifacts from merged text
            val result = cleanupTranscript(rawResult)

            Log.i(TAG, "  VAD gate: noise=${String.format("%.1f", noiseFloorRms)}, skipped=$skippedSilentChunks chunks")
            Log.i(TAG, "  ✓ ${totalBytes / 1024}KB @ ${sampleRate}Hz → ${result.length} chars in ${elapsed}ms")
            Log.i(TAG, "  Result: '${result.take(200)}'")
            Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            return result

        } catch (e: Exception) {
            Log.e(TAG, "  ✗ Error: ${e.message}")
            e.printStackTrace()
            return ""
        }
    }

    private fun readWavHeader(wavPath: String): Map<String, Int> {
        try {
            val raf = RandomAccessFile(wavPath, "r")

            // Read fmt chunk at standard offset
            raf.seek(20)
            val audioFormat = readShortLE(raf)
            val channels = readShortLE(raf)
            val sampleRate = readIntLE(raf)
            val byteRate = readIntLE(raf)
            val blockAlign = readShortLE(raf)
            val bitsPerSample = readShortLE(raf)

            // Find the actual 'data' chunk (may not be at offset 44 if extra chunks exist)
            var dataOffset = 44  // default
            raf.seek(12)  // Skip RIFF header + WAVE
            while (raf.filePointer < raf.length() - 8) {
                val chunkId = ByteArray(4)
                raf.readFully(chunkId)
                val chunkSize = readIntLE(raf)
                val id = String(chunkId)
                if (id == "data") {
                    dataOffset = raf.filePointer.toInt()
                    break
                }
                // Skip this chunk
                raf.seek(raf.filePointer + chunkSize)
            }

            raf.close()
            return mapOf(
                "sampleRate" to sampleRate,
                "channels" to channels,
                "bitsPerSample" to bitsPerSample,
                "dataOffset" to dataOffset
            )
        } catch (e: Exception) {
            Log.w(TAG, "  WAV header read failed: ${e.message}")
            return mapOf("sampleRate" to 16000, "channels" to 1, "bitsPerSample" to 16, "dataOffset" to 44)
        }
    }

    private fun readShortLE(raf: RandomAccessFile): Int {
        val b0 = raf.read(); val b1 = raf.read()
        return (b1 shl 8) or b0
    }

    private fun readIntLE(raf: RandomAccessFile): Int {
        val b0 = raf.read(); val b1 = raf.read()
        val b2 = raf.read(); val b3 = raf.read()
        return (b3 shl 24) or (b2 shl 16) or (b1 shl 8) or b0
    }

    private fun extractText(json: String): String {
        val regex = """"text"\s*:\s*"([^"]*)"""".toRegex()
        return regex.find(json)?.groupValues?.getOrNull(1)?.trim() ?: ""
    }

    private fun findModelDir(root: File): File? {
        // ASR models usually have 'am' or 'conf'
        // Speaker models usually have 'mfcc.conf' or 'final.ext.raw'
        fun hasModelFiles(dir: File): Boolean {
            return File(dir, "am").exists() || 
                   File(dir, "conf").exists() || 
                   File(dir, "mfcc.conf").exists() || 
                   File(dir, "final.ext.raw").exists()
        }

        if (hasModelFiles(root)) return root
        root.listFiles()?.forEach { child ->
            if (child.isDirectory) {
                if (hasModelFiles(child)) return child
                child.listFiles()?.forEach { gc ->
                    if (gc.isDirectory && hasModelFiles(gc)) return gc
                }
            }
        }
        return null
    }

    private fun downloadFileResumable(url: String, targetFile: File, isAsr: Boolean) {
        val existingSize = if (targetFile.exists()) targetFile.length() else 0L
        if (isAsr) asrBytesDownloaded = existingSize else spkBytesDownloaded = existingSize

        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = 30000
        conn.readTimeout = 60000
        
        if (existingSize > 0) {
            conn.setRequestProperty("Range", "bytes=$existingSize-")
        }

        conn.connect()
        val responseCode = conn.responseCode
        
        if (responseCode != 200 && responseCode != 206) {
            if (responseCode == 416) return // Range not satisfiable, might be done
            throw Exception("Server returned code $responseCode")
        }

        val contentLength = conn.contentLength.toLong()
        val totalBytes = if (responseCode == 206) contentLength + existingSize else contentLength
        if (isAsr) asrTotalBytes = totalBytes else spkTotalBytes = totalBytes

        val inputStream = BufferedInputStream(conn.inputStream)
        val outputStream = RandomAccessFile(targetFile, "rw")
        
        if (responseCode == 200) {
            outputStream.setLength(0) // Truncate to rewrite from start
            outputStream.seek(0)
            if (isAsr) asrBytesDownloaded = 0L else spkBytesDownloaded = 0L
        } else {
            outputStream.seek(existingSize)
        }

        val buffer = ByteArray(32768)
        while (if (isAsr) !isAsrPaused else !isSpkPaused) {
            val n = inputStream.read(buffer)
            if (n <= 0) break
            outputStream.write(buffer, 0, n)
            
            if (isAsr) {
                asrBytesDownloaded += n
                val newProgress = if (asrTotalBytes > 0) ((asrBytesDownloaded * 100) / asrTotalBytes).toInt() else 0
                if (newProgress != asrProgress) {
                    asrProgress = newProgress
                    if (asrProgress % 5 == 0) Log.d(TAG, "  ASR Progress: $asrProgress%")
                }
            } else {
                spkBytesDownloaded += n
                val newProgress = if (spkTotalBytes > 0) ((spkBytesDownloaded * 100) / spkTotalBytes).toInt() else 0
                if (newProgress != spkProgress) {
                    spkProgress = newProgress
                    if (spkProgress % 5 == 0) Log.d(TAG, "  SPK Progress: $spkProgress%")
                }
            }
        }

        outputStream.close()
        inputStream.close()
        conn.disconnect()

        val paused = if (isAsr) isAsrPaused else isSpkPaused
        val bytesDownloaded = if (isAsr) asrBytesDownloaded else spkBytesDownloaded
        val total = if (isAsr) asrTotalBytes else spkTotalBytes
        
        if (!paused && total > 0 && bytesDownloaded < total) {
            throw Exception("Download incomplete (Network lost?)")
        }
    }

    private fun extractZip(zipFile: File, targetDir: File) {
        targetDir.mkdirs()
        val zis = ZipInputStream(BufferedInputStream(FileInputStream(zipFile)))
        var entry = zis.nextEntry
        while (entry != null) {
            val outFile = File(targetDir, entry.name)
            if (entry.isDirectory) {
                outFile.mkdirs()
            } else {
                outFile.parentFile?.mkdirs()
                FileOutputStream(outFile).use { fos ->
                    val buffer = ByteArray(32768)
                    while (true) {
                        val n = zis.read(buffer)
                        if (n <= 0) break
                        fos.write(buffer, 0, n)
                    }
                }
            }
            entry = zis.nextEntry
        }
        zis.close()
    }

    // ══════════════════════════════════════════════════════════
    //  TRANSCRIPT CLEANUP — Remove Vosk filler artifacts
    // ══════════════════════════════════════════════════════════

    /**
     * Clean up Vosk ASR output by removing common artifacts:
     * - Repeated filler words ("the the", "a a", "uh uh")
     * - Leading/trailing filler words ("the" at start/end of phrases)
     * - Common misrecognitions specific to noisy BT audio
     * - Apply basic sentence capitalization
     */
    fun cleanupTranscript(text: String): String {
        if (text.isBlank()) return ""

        var cleaned = text.lowercase().trim()

        // 1) Remove repeated consecutive words ("the the" → "the", "a a" → "a")
        cleaned = cleaned.replace("\\b(\\w+)\\s+\\1\\b".toRegex(), "$1")
        // Apply twice for triple repeats
        cleaned = cleaned.replace("\\b(\\w+)\\s+\\1\\b".toRegex(), "$1")

        // 2) Remove filler words that appear before/after content words
        //    Pattern: "the [content]" where "the" is noise inserted by Vosk
        //    We detect this by looking for "the" that doesn't fit grammatically
        val fillerPattern = "\\bthe\\s+(?=(?:the|a|an|uh|um|huh)\\b)".toRegex()
        cleaned = fillerPattern.replace(cleaned, "")

        // 3) Remove standalone noise words (pure filler segments)
        val noiseWords = setOf("the", "a", "uh", "um", "huh", "eh", "ah")
        val words = cleaned.split("\\s+".toRegex()).toMutableList()

        // Remove leading noise words
        while (words.isNotEmpty() && words.first() in noiseWords) {
            words.removeAt(0)
        }
        // Remove trailing noise words
        while (words.isNotEmpty() && words.last() in noiseWords) {
            words.removeAt(words.lastIndex)
        }

        // 4) Remove isolated "the" between punctuation-like breaks
        //    e.g., "call later the around six" → "call later around six"
        val result = mutableListOf<String>()
        for (i in words.indices) {
            val word = words[i]
            if (word in noiseWords) {
                // Keep if it makes grammatical sense (next word is a noun/adj)
                val prev = if (i > 0) words[i - 1] else ""
                val next = if (i < words.size - 1) words[i + 1] else ""
                // Only keep "the" if next word is NOT another noise word
                // and the previous word suggests an article is needed
                val keepWords = setOf("in", "on", "at", "by", "for", "with", "to",
                    "is", "was", "has", "have", "get", "take", "from", "about",
                    "near", "around", "after", "before", "during")
                if (word == "the" && next.isNotEmpty() && next !in noiseWords &&
                    (prev in keepWords || prev.isEmpty() || prev.endsWith("."))) {
                    result.add(word)
                } else if (word == "a" && next.isNotEmpty() && next !in noiseWords) {
                    result.add(word)
                }
                // else: skip this noise word
            } else {
                result.add(word)
            }
        }

        cleaned = result.joinToString(" ").trim()

        // 5) Capitalize first letter of each sentence
        if (cleaned.isNotEmpty()) {
            cleaned = cleaned.replaceFirstChar { it.uppercaseChar() }
            cleaned = cleaned.replace("\\. (\\w)".toRegex()) { match ->
                ". " + match.groupValues[1].uppercase()
            }
        }

        // 6) Clean up extra spaces
        cleaned = cleaned.replace("\\s+".toRegex(), " ").trim()

        return cleaned
    }

    fun getStatus(): Map<String, Any> = mapOf(
        "asr" to mapOf(
            "ready" to isAsrReady(),
            "initializing" to isAsrInitializing,
            "paused" to isAsrPaused,
            "error" to (asrInitError ?: ""),
            "progress" to asrProgress,
            "downloaded_mb" to (asrBytesDownloaded / 1024 / 1024).toInt(),
            "total_mb" to (asrTotalBytes / 1024 / 1024).toInt(),
            "model_name" to "vosk-model-en-in-0.5"
        ),
        "spk" to mapOf(
            "ready" to isSpkReady(),
            "initializing" to isSpkInitializing,
            "paused" to isSpkPaused,
            "error" to (spkInitError ?: ""),
            "progress" to spkProgress,
            "downloaded_mb" to (spkBytesDownloaded / 1024 / 1024).toInt(),
            "total_mb" to (spkTotalBytes / 1024 / 1024).toInt(),
            "model_name" to "vosk-model-spk-0.4"
        ),
        "ready" to isAsrReady()
    )
}
