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
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
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

    private var recognizer: OfflineRecognizer? = null
    private var spkReady = true

    // ASR State
    var isAsrInitializing = false
    var isAsrPaused = false
    var asrInitError: String? = null
    var asrProgress = 0
    var asrBytesDownloaded = 0L
    var asrTotalBytes = 0L

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
        Log.i(TAG, "Initializing Sherpa-ONNX Whisper ASR")

        Thread {
            try {
                var modelRoot = resolveModelRoot(context)
                if (modelRoot == null) {
                    val archiveFile = File(context.cacheDir, MODEL_ARCHIVE_NAME)
                    downloadFileResumable(MODEL_URL, archiveFile)
                    if (isAsrPaused) {
                        isAsrInitializing = false
                        return@Thread
                    }

                    val targetParent = context.getExternalFilesDir(null) ?: context.filesDir
                    extractTarBz2(archiveFile, targetParent)
                    ensureModelLayout(targetParent)
                    modelRoot = resolveModelRoot(context)
                }

                if (modelRoot == null) {
                    throw Exception("Model files not found after download/extract")
                }

                recognizer = createRecognizer(modelRoot)
                isAsrInitializing = false
                asrProgress = 100
                Log.i(TAG, "Sherpa model ready: ${modelRoot.absolutePath}")
                onComplete?.invoke(true)

            } catch (e: Exception) {
                if (!isAsrPaused) {
                    isAsrInitializing = false
                    asrInitError = e.message
                    Log.e(TAG, "ASR init failed: ${e.message}")
                    onComplete?.invoke(false)
                }
            }
        }.start()
    }

    fun pauseAsr() { isAsrPaused = true }
    fun resumeAsr(context: Context) { if (isAsrPaused) initAsr(context) }
    fun retryAsr(context: Context) {
        asrInitError = null
        isAsrInitializing = false
        isAsrPaused = false
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

    fun transcribeWavWithSpeakers(wavPath: String, context: Context): List<Map<String, Any>> {
        val text = transcribeWav(wavPath).trim()
        if (text.isEmpty()) return emptyList()
        return listOf(
            mapOf(
                "speaker" to "Speaker 1",
                "text" to text,
            )
        )
    }

    /**
     * Build a deterministic 128-D voice embedding from PCM waveform.
     * This preserves enrollment/matching behavior without Sherpa x-vectors.
     */
    fun extractXVector(wavPath: String): FloatArray? {
        return try {
            val wave = readWavePcm16(wavPath)
            val s = wave.samples
            if (s.size < 1600) return null

            val dim = 128
            val block = maxOf(1, s.size / dim)
            val vec = FloatArray(dim)

            for (i in 0 until dim) {
                val start = i * block
                if (start >= s.size) break
                val end = minOf(s.size, start + block)

                var sumAbs = 0.0
                var sum = 0.0
                var zc = 0
                var prev = s[start]

                for (j in start until end) {
                    val cur = s[j]
                    sumAbs += abs(cur)
                    sum += cur
                    if ((cur >= 0f) != (prev >= 0f)) zc++
                    prev = cur
                }

                val len = (end - start).coerceAtLeast(1)
                val energy = (sumAbs / len).toFloat()
                val bias = (sum / len).toFloat()
                val zcr = zc.toFloat() / len.toFloat()

                vec[i] = (energy * 0.80f) + (abs(bias) * 0.10f) + (zcr * 0.10f)
            }

            // L2 normalize
            var norm = 0.0
            for (v in vec) norm += (v * v)
            val denom = sqrt(norm).toFloat()
            if (denom > 0f) {
                for (i in vec.indices) vec[i] /= denom
            }

            vec
        } catch (e: Exception) {
            Log.e(TAG, "Voice embedding error: ${e.message}")
            null
        }
    }

    fun transcribeWav(wavPath: String): String {
        val r = recognizer ?: run {
            Log.e(TAG, "Sherpa recognizer not loaded")
            return ""
        }

        try {
            val wave = readWavePcm16(wavPath)
            val stream = r.createStream()
            stream.acceptWaveform(wave.samples, sampleRate = wave.sampleRate)
            r.decode(stream)

            val raw = r.getResult(stream).text.trim()
            stream.release()
            return cleanupTranscript(raw)

        } catch (e: Exception) {
            Log.e(TAG, "Transcription error: ${e.message}")
            return ""
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

        return candidates.firstOrNull { dir ->
            dir.exists() && dir.isDirectory && findWhisperFiles(dir) != null
        }
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
                    val p = ((asrBytesDownloaded * 100L) / asrTotalBytes).toInt().coerceIn(0, 99)
                    asrProgress = p
                }
            }
        } finally {
            output.close()
            inputStream.close()
            conn.disconnect()
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
                    while (true) {
                        val entry = tis.nextTarEntry ?: break

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
                        FileOutputStream(outFile).use { fos ->
                            val buffer = ByteArray(64 * 1024)
                            while (true) {
                                val n = tis.read(buffer)
                                if (n <= 0) break
                                fos.write(buffer, 0, n)
                            }
                        }
                    }
                }
            }
        }
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