package com.example.memory_assistant

import android.content.Context
import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.roundToInt

/**
 * Phi2ModelManager
 *
 * Downloads and manages a local Phi-2 GGUF model file in app-private storage.
 * This path does not rely on Ollama and is intended for offline-first mobile flow
 * after one-time model download.
 */
object Phi2ModelManager {

    private const val TAG = "WBrain.Phi2"
    private const val MODEL_FILE_NAME = "phi-2.Q4_K_M.gguf"
    private const val MODEL_URL = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
    private const val READY_MARKER_NAME = "phi-2.Q4_K_M.gguf.ready"

    @Volatile private var initializing = false
    @Volatile private var paused = false
    @Volatile private var ready = false
    @Volatile private var progress = 0
    @Volatile private var downloadedMb = 0
    @Volatile private var totalMb = 0
    @Volatile private var stage = "idle"
    @Volatile private var error = ""
    @Volatile private var cancelRequested = false

    private fun modelDir(context: Context): File = File(context.filesDir, "models/phi2")
    private fun modelFile(context: Context): File = File(modelDir(context), MODEL_FILE_NAME)
    private fun readyMarkerFile(context: Context): File = File(modelDir(context), READY_MARKER_NAME)

    private fun updateFromDisk(context: Context) {
        val file = modelFile(context)
        val marker = readyMarkerFile(context)
        if (file.exists() && marker.exists()) {
            ready = true
            initializing = false
            paused = false
            stage = "ready"
            downloadedMb = (file.length() / 1024L / 1024L).toInt()
            totalMb = if (totalMb > 0) totalMb else downloadedMb
            progress = 100
            error = ""
        } else {
            if (!initializing) {
                ready = false
                stage = "idle"
            }
            downloadedMb = if (file.exists()) (file.length() / 1024L / 1024L).toInt() else 0
            if (totalMb == 0) totalMb = 1900
        }
    }

    fun getStatus(context: Context): Map<String, Any> {
        updateFromDisk(context)
        return mapOf(
            "ready" to ready,
            "initializing" to initializing,
            "paused" to paused,
            "progress" to progress,
            "downloaded_mb" to downloadedMb,
            "total_mb" to totalMb,
            "stage" to stage,
            "error" to error,
            "model" to MODEL_FILE_NAME,
            "storage_path" to modelFile(context).absolutePath,
            "status" to if (ready) "online" else if (initializing) "downloading" else "offline",
            "reason" to if (!ready && !initializing && error.isBlank()) "Phi-2 model not downloaded yet" else "",
        )
    }

    fun startDownload(context: Context) {
        updateFromDisk(context)
        if (ready || initializing) return

        initializing = true
        paused = false
        cancelRequested = false
        stage = "downloading"
        progress = 0
        error = ""

        Thread {
            val dir = modelDir(context)
            if (!dir.exists()) dir.mkdirs()
            val file = modelFile(context)
            val marker = readyMarkerFile(context)

            try {
                if (marker.exists()) marker.delete()
                val existingBytes = if (file.exists()) file.length() else 0L
                val conn = URL(MODEL_URL).openConnection() as HttpURLConnection
                conn.requestMethod = "GET"
                conn.connectTimeout = 7000
                conn.readTimeout = 30000
                if (existingBytes > 0L) {
                    conn.setRequestProperty("Range", "bytes=$existingBytes-")
                }

                val code = conn.responseCode
                if (code !in listOf(HttpURLConnection.HTTP_OK, HttpURLConnection.HTTP_PARTIAL)) {
                    throw IllegalStateException("Model download failed: HTTP $code")
                }

                val serverLength = conn.contentLengthLong
                val expectedBytes = when {
                    serverLength > 0L && code == HttpURLConnection.HTTP_PARTIAL -> existingBytes + serverLength
                    serverLength > 0L -> serverLength
                    else -> 0L
                }
                totalMb = if (expectedBytes > 0L) (expectedBytes / 1024L / 1024L).toInt() else 1900

                val raf = RandomAccessFile(file, "rw")
                if (existingBytes > 0L && code == HttpURLConnection.HTTP_PARTIAL) {
                    raf.seek(existingBytes)
                } else {
                    raf.setLength(0L)
                }

                conn.inputStream.use { input ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        if (cancelRequested) {
                            paused = true
                            stage = "paused"
                            Log.i(TAG, "Download paused by user at ${file.length()} bytes")
                            break
                        }
                        val n = input.read(buffer)
                        if (n <= 0) break
                        raf.write(buffer, 0, n)

                        val current = file.length()
                        downloadedMb = (current / 1024L / 1024L).toInt()
                        if (expectedBytes > 0L) {
                            progress = ((current.toDouble() / expectedBytes.toDouble()) * 100.0)
                                .roundToInt()
                                .coerceIn(0, 100)
                        }
                    }
                }

                raf.close()
                conn.disconnect()

                if (!cancelRequested) {
                    stage = "verifying"
                    if (expectedBytes > 0L && file.length() < expectedBytes) {
                        stage = "error"
                        error = "Downloaded file is incomplete. Tap Retry."
                    } else {
                        marker.writeText("ready")
                    }
                    updateFromDisk(context)
                    if (!ready) {
                        stage = "error"
                        error = "Downloaded file is incomplete or invalid. Tap Retry."
                    } else {
                        stage = "ready"
                        progress = 100
                        Log.i(TAG, "Phi-2 model ready at ${file.absolutePath}")
                    }
                }
            } catch (e: Exception) {
                stage = "error"
                error = e.message ?: "Unknown download error"
                Log.e(TAG, "Phi-2 download failed", e)
            } finally {
                if (!cancelRequested) paused = false
                cancelRequested = false
                initializing = false
            }
        }.start()
    }

    fun pauseDownload() {
        if (!initializing) return
        cancelRequested = true
    }

    fun resumeDownload(context: Context) {
        if (!paused) return
        startDownload(context)
    }

    fun retryDownload(context: Context) {
        try {
            val file = modelFile(context)
            if (file.exists()) file.delete()
            val marker = readyMarkerFile(context)
            if (marker.exists()) marker.delete()
        } catch (_: Exception) {}

        initializing = false
        paused = false
        ready = false
        progress = 0
        downloadedMb = 0
        totalMb = 0
        stage = "idle"
        error = ""
        cancelRequested = false

        startDownload(context)
    }
}
