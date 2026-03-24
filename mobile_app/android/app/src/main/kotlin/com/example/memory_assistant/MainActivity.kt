package com.example.memory_assistant

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothClass
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.ByteArrayOutputStream
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.io.RandomAccessFile
import java.util.Calendar
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import org.json.JSONObject

/**
 * MainActivity — Full Voice Pipeline Bridge
 *
 * Complete flow:
 *   startRecording → real AudioRecord captures PCM → stopRecording
 *   → save WAV → Android SpeechRecognizer transcription
 *   → SimpleNlpProcessor extraction → SQLite storage
 *
 * Debug logs tagged "WBrain" for adb logcat:
 *   adb logcat -s WBrain:* WBrain.DB:* WBrain.NLP:*
 */
class MainActivity : FlutterActivity() {

    companion object {
        const val TAG = "WBrain"
        const val CHANNEL = "memory_assistant"
        const val SAMPLE_RATE = 16000
        const val PERMISSION_REQUEST_CODE = 200
    }

    private lateinit var db: MemoryDatabase
    private val mainHandler = Handler(Looper.getMainLooper())

    // ── Audio recording state ────────────────────────────────
    private var isRecording = false
    private var recordingThread: Thread? = null
    private var audioRecord: AudioRecord? = null
    private var pcmBuffer = ByteArrayOutputStream()
    private var recordingStartTime = 0L
    private var lastWavPath: String? = null
    private var actualRecordingSampleRate = SAMPLE_RATE  // Tracks real rate (8000 for BT SCO)
    private var lastClipPercent = 0.0
    private var lastRecordingAudioSource = MediaRecorder.AudioSource.MIC

    // ── Audio source state ───────────────────────────────────
    private var currentAudioSource = "microphone"
    private var audioSourceActive = false
    private var bluetoothDeviceName: String? = null
    private var btAudioBuffer = ByteArrayOutputStream()

    // ── Live transcription state ────────────────────────────
    private var liveRecognizer: SpeechRecognizer? = null
    private var liveTranscript = StringBuilder()
    private var transcriptReady = false
    private var transcriptSegmentCount = 0
    private var lastSegmentTime = 0L

    // ── Pending result for async speech recognition ──────────
    private var pendingResult: MethodChannel.Result? = null

    // ── SCO state tracking ───────────────────────────────
    private var scoConnected = false
    private var scoReceiver: android.content.BroadcastReceiver? = null
    private var pendingBtRecordAction: (() -> Unit)? = null

    // ── On-device answer model status (offline) ─────────────
    @Volatile private var llmInitializing = false
    @Volatile private var llmPaused = false
    @Volatile private var llmReady = false
    @Volatile private var llmProgress = 0
    @Volatile private var llmDownloadedMb = 0
    @Volatile private var llmTotalMb = 1900
    @Volatile private var llmStage = "idle"
    @Volatile private var llmError = ""
    @Volatile private var llmModel = "phi-2.Q4_K_M.gguf"
    @Volatile private var llmCancelRequested = false
    @Volatile private var llmEndpoint = "file://app/models/phi2"

    private fun refreshLlmReadyState() {
        val status = Phi2ModelManager.getStatus(applicationContext)
        llmReady = status["ready"] == true
        llmInitializing = status["initializing"] == true
        llmPaused = status["paused"] == true
        llmProgress = (status["progress"] as? Number)?.toInt() ?: 0
        llmDownloadedMb = (status["downloaded_mb"] as? Number)?.toInt() ?: 0
        llmTotalMb = (status["total_mb"] as? Number)?.toInt() ?: 1900
        llmStage = (status["stage"] ?: "idle").toString()
        llmError = (status["error"] ?: "").toString()
        llmModel = (status["model"] ?: llmModel).toString()
        llmEndpoint = (status["storage_path"] ?: llmEndpoint).toString()
    }

    private fun getLlmStatusMap(): Map<String, Any> {
        return mapOf(
            "ready" to llmReady,
            "initializing" to llmInitializing,
            "paused" to llmPaused,
            "progress" to llmProgress,
            "downloaded_mb" to llmDownloadedMb,
            "total_mb" to llmTotalMb,
            "stage" to llmStage,
            "error" to llmError,
            "model" to llmModel,
            "endpoint" to llmEndpoint,
            "candidates" to listOf("file://app/models/phi2"),
            "status" to if (llmReady) "online" else if (llmInitializing) "downloading" else "offline",
            "reason" to if (!llmReady && !llmInitializing && llmError.isBlank()) "Phi-2 model not downloaded yet" else ""
        )
    }

    private fun startLlmDownload() {
        Phi2ModelManager.startDownload(applicationContext)
        refreshLlmReadyState()
        Log.i(TAG, "Phi-2 local GGUF download requested.")
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        refreshLlmReadyState()

        // Initialize real database
        db = MemoryDatabase(applicationContext)
        Log.i(TAG, "═══════════════════════════════════════")
        Log.i(TAG, "  Memory Assistant Engine Started")
        Log.i(TAG, "  DB path: ${applicationContext.getDatabasePath("memory.db")}")
        Log.i(TAG, "  SpeechRecognizer available: ${SpeechRecognizer.isRecognitionAvailable(this)}")
        Log.i(TAG, "═══════════════════════════════════════")

        // Initialize sherpa-onnx offline ASR
        SherpaTranscriber.initAsr(applicationContext) { ready ->
            Log.i(TAG, "Sherpa offline ASR: ${if (ready) "✓ READY" else "✗ NOT AVAILABLE"}")
        }

        // Initialize speaker subsystem status
        SherpaTranscriber.initSpk(applicationContext) { ready ->
            Log.i(TAG, "Speaker subsystem: ${if (ready) "✓ READY" else "✗ NOT AVAILABLE"}")
        }

        // Register MethodChannel
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                handleMethodCall(call, result)
            }
    }

    // ── Permission check ─────────────────────────────────────

    private fun ensureRecordPermission(): Boolean {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                PERMISSION_REQUEST_CODE
            )
            return false
        }
        return true
    }

    // ── Bluetooth SCO Routing ────────────────────────────────

    private fun startBluetoothSco(): Boolean {
        val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        try {
            // Check if Bluetooth adapter exists and is enabled
            val btManager = getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager
            val btAdapter = btManager?.adapter
            if (btAdapter == null || !btAdapter.isEnabled) {
                Log.w(TAG, "  ✗ Bluetooth adapter not available or disabled")
                return false
            }

            // Check BLUETOOTH_CONNECT permission for Android 12+
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT)
                    != PackageManager.PERMISSION_GRANTED) {
                    Log.w(TAG, "  ✗ BLUETOOTH_CONNECT permission not granted")
                    ActivityCompat.requestPermissions(
                        this, arrayOf(Manifest.permission.BLUETOOTH_CONNECT), 201
                    )
                    return false
                }
            }

            // Check for ACTUALLY CONNECTED BT audio input devices
            val connectedInputs = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
            val btInput = connectedInputs.firstOrNull { d ->
                d.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO ||
                d.type == AudioDeviceInfo.TYPE_BLUETOOTH_A2DP ||
                d.type == AudioDeviceInfo.TYPE_BLE_HEADSET
            }

            if (btInput != null) {
                bluetoothDeviceName = btInput.productName?.toString()?.ifBlank { null }
                    ?: "Bluetooth Audio"
                Log.i(TAG, "  ✓ Connected BT input: $bluetoothDeviceName (type=${btInput.type})")
            } else {
                // Fallback: check bonded audio devices
                val bondedAudio = btAdapter.bondedDevices?.filter { device ->
                    val major = device.bluetoothClass?.majorDeviceClass
                    major == BluetoothClass.Device.Major.AUDIO_VIDEO ||
                    major == BluetoothClass.Device.Major.WEARABLE
                }?.firstOrNull()
                bluetoothDeviceName = bondedAudio?.name ?: "Bluetooth Device"
                Log.w(TAG, "  ⚠ No connected BT input found, using bonded: $bluetoothDeviceName")
            }

            // Register SCO state receiver BEFORE starting SCO
            registerScoReceiver()

            // Start SCO connection for microphone routing
            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            @Suppress("DEPRECATION")
            audioManager.startBluetoothSco()
            @Suppress("DEPRECATION")
            audioManager.isBluetoothScoOn = true

            Log.i(TAG, "  ✓ SCO requested, mode=IN_COMMUNICATION")
            Log.i(TAG, "  ✓ isBluetoothScoOn=${audioManager.isBluetoothScoOn}")

            // Fallback: if SCO doesn't connect within 3s, proceed anyway
            mainHandler.postDelayed({
                if (!scoConnected && currentAudioSource == "bluetooth") {
                    Log.w(TAG, "  ⚠ SCO timeout (3s) — proceeding with available input")
                    scoConnected = true  // force proceed
                    pendingBtRecordAction?.invoke()
                    pendingBtRecordAction = null
                }
            }, 3000)

            return true

        } catch (e: Exception) {
            Log.e(TAG, "  ✗ BT SCO failed: ${e.message}")
            return false
        }
    }

    private fun registerScoReceiver() {
        unregisterScoReceiver()  // clean previous
        scoReceiver = object : android.content.BroadcastReceiver() {
            override fun onReceive(context: android.content.Context?, intent: android.content.Intent?) {
                val state = intent?.getIntExtra(AudioManager.EXTRA_SCO_AUDIO_STATE, -1)
                Log.i(TAG, "  [SCO] State changed: $state")
                when (state) {
                    AudioManager.SCO_AUDIO_STATE_CONNECTED -> {
                        scoConnected = true
                        Log.i(TAG, "  [SCO] ✓ CONNECTED — BT mic is now active")
                        // Trigger pending recording action
                        pendingBtRecordAction?.invoke()
                        pendingBtRecordAction = null
                    }
                    AudioManager.SCO_AUDIO_STATE_DISCONNECTED -> {
                        Log.w(TAG, "  [SCO] ✗ DISCONNECTED")
                        scoConnected = false
                    }
                    AudioManager.SCO_AUDIO_STATE_CONNECTING -> {
                        Log.i(TAG, "  [SCO] ... CONNECTING")
                    }
                }
            }
        }
        val filter = android.content.IntentFilter(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED)
        registerReceiver(scoReceiver, filter)
        Log.d(TAG, "  [SCO] BroadcastReceiver registered")
    }

    private fun unregisterScoReceiver() {
        scoReceiver?.let {
            try { unregisterReceiver(it) } catch (_: Exception) {}
            scoReceiver = null
        }
    }

    private fun stopBluetoothSco() {
        try {
            unregisterScoReceiver()
            scoConnected = false
            pendingBtRecordAction = null
            val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
            @Suppress("DEPRECATION")
            audioManager.isBluetoothScoOn = false
            @Suppress("DEPRECATION")
            audioManager.stopBluetoothSco()
            audioManager.mode = AudioManager.MODE_NORMAL
            Log.i(TAG, "  ✓ Bluetooth SCO stopped, mode=NORMAL")
        } catch (e: Exception) {
            Log.e(TAG, "  BT SCO stop error: ${e.message}")
        }
    }

    // ── Real microphone recording ────────────────────────────

    private fun calculateRms(buffer: ByteArray): Double {
        var sum = 0.0
        for (i in 0 until buffer.size step 2) {
            val sample = (buffer[i].toInt() and 0xFF) or (buffer[i + 1].toInt() shl 8)
            sum += (sample * sample).toDouble()
        }
        return Math.sqrt(sum / (buffer.size / 2))
    }

    private fun startMicRecording() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.e(TAG, "  ✗ RECORD_AUDIO permission not granted!")
            return
        }

        val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager

        // Choose source: prefer VOICE_RECOGNITION for cleaner ASR on BT, fallback safely.
        var audioSource = MediaRecorder.AudioSource.MIC
        if (currentAudioSource == "bluetooth") {
            Log.i(TAG, "  SCO on: ${audioManager.isBluetoothScoOn}, connected: $scoConnected")
            val btCandidates = listOf(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                MediaRecorder.AudioSource.MIC,
            )
            val testBuf = AudioRecord.getMinBufferSize(
                8000,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            var chosen = MediaRecorder.AudioSource.VOICE_COMMUNICATION
            if (testBuf > 0) {
                for (candidate in btCandidates) {
                    val probe = AudioRecord(
                        candidate,
                        8000,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        testBuf * 2,
                    )
                    val ok = probe.state == AudioRecord.STATE_INITIALIZED
                    probe.release()
                    if (ok) {
                        chosen = candidate
                        break
                    }
                }
            }
            audioSource = chosen
            val sourceName = when (audioSource) {
                MediaRecorder.AudioSource.VOICE_RECOGNITION -> "VOICE_RECOGNITION"
                MediaRecorder.AudioSource.VOICE_COMMUNICATION -> "VOICE_COMMUNICATION"
                MediaRecorder.AudioSource.MIC -> "MIC"
                else -> "SOURCE_$audioSource"
            }
            Log.i(TAG, "  Using $sourceName for BT SCO")
        }
        lastRecordingAudioSource = audioSource

        // Log input device info
        val inputs = audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
        for (dev in inputs) {
            Log.d(TAG, "  [Input] type=${dev.type}, name=${dev.productName}, id=${dev.id}")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                val rates = dev.sampleRates
                if (rates != null && rates.isNotEmpty()) {
                    Log.d(TAG, "    Supported rates: ${rates.joinToString()}")
                }
            }
        }

        // ★ CRITICAL: Bluetooth SCO supports 8kHz (narrowband) or sometimes 16kHz (wideband)
        // We MUST detect the correct rate instead of forcing 16kHz
        val recordSampleRate: Int
        if (currentAudioSource == "bluetooth") {
            // Try 8kHz first (guaranteed by SCO), then 16kHz (wideband if supported)
            val rate8k = AudioRecord.getMinBufferSize(8000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            val rate16k = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)

            if (rate8k > 0) {
                // Try to create an AudioRecord at 8kHz to test
                val testRec = AudioRecord(audioSource, 8000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, rate8k * 2)
                if (testRec.state == AudioRecord.STATE_INITIALIZED) {
                    recordSampleRate = 8000
                    testRec.release()
                    Log.i(TAG, "  ★ BT SCO: Using 8kHz (narrowband) — will resample to 16kHz for Sherpa")
                } else {
                    testRec.release()
                    recordSampleRate = SAMPLE_RATE
                    Log.i(TAG, "  ★ BT SCO: 8kHz failed, using ${SAMPLE_RATE}Hz")
                }
            } else {
                recordSampleRate = SAMPLE_RATE
                Log.i(TAG, "  ★ BT SCO: 8kHz not supported, using ${SAMPLE_RATE}Hz")
            }
        } else {
            recordSampleRate = SAMPLE_RATE
        }

        actualRecordingSampleRate = recordSampleRate
        Log.i(TAG, "  ★ Recording at: ${recordSampleRate}Hz (target for Sherpa: ${SAMPLE_RATE}Hz)")

        val bufferSize = AudioRecord.getMinBufferSize(
            recordSampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        audioRecord = AudioRecord(
            audioSource,
            recordSampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize * 2
        )

        // Try to route to BT SCO device explicitly
        if (currentAudioSource == "bluetooth" && Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val scoInput = inputs.firstOrNull { it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO }
            if (scoInput != null) {
                val set = audioRecord?.setPreferredDevice(scoInput)
                Log.i(TAG, "  Routed AudioRecord to SCO device: ${scoInput.productName} (success=$set)")
            }
        }

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "  ✗ AudioRecord failed to initialize at ${recordSampleRate}Hz")

            // Fallback: try the other sample rate
            if (recordSampleRate != SAMPLE_RATE) {
                Log.i(TAG, "  Retrying at ${SAMPLE_RATE}Hz...")
                audioRecord?.release()
                val fallbackBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
                audioRecord = AudioRecord(audioSource, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, fallbackBuf * 2)
                actualRecordingSampleRate = SAMPLE_RATE
                if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                    Log.e(TAG, "  ✗ Fallback also failed!")
                    audioRecord?.release()
                    audioRecord = null
                    return
                }
            } else {
                audioRecord?.release()
                audioRecord = null
                return
            }
        }

        // Log actual recording state
        Log.i(TAG, "  ✓ AudioRecord initialized: rate=${actualRecordingSampleRate}Hz, buffer=${bufferSize}, source=${audioSource}")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            val routedDevice = audioRecord?.routedDevice
            Log.i(TAG, "  Routed to: ${routedDevice?.productName} (type=${routedDevice?.type})")
        }

        pcmBuffer.reset()
        audioRecord?.startRecording()
        Log.i(TAG, "  ✓ AudioRecord started (rate=${actualRecordingSampleRate}Hz)")

        recordingThread = Thread({
            val buffer = ByteArray(bufferSize)
            var totalBytes = 0
            var logCounter = 0
            var maxSample = 0
            var clippedSamples = 0
            var totalSamples = 0
            while (isRecording && audioRecord != null) {
                val read = audioRecord?.read(buffer, 0, buffer.size) ?: -1
                if (read > 0) {
                    synchronized(pcmBuffer) {
                        pcmBuffer.write(buffer, 0, read)
                    }
                    totalBytes += read

                    // Track peak amplitude
                    for (i in 0 until read - 1 step 2) {
                        val sample = Math.abs((buffer[i].toInt() and 0xFF) or (buffer[i + 1].toInt() shl 8))
                        if (sample > maxSample) maxSample = sample
                        totalSamples++
                        if (sample >= 32760) clippedSamples++
                    }

                    logCounter++
                    // Log every ~1 second
                    if (logCounter % (actualRecordingSampleRate * 2 / bufferSize) == 0) {
                        val seconds = totalBytes.toDouble() / (actualRecordingSampleRate * 2)
                        val clipPct = if (totalSamples > 0) (clippedSamples * 100.0 / totalSamples) else 0.0
                        Log.d(TAG, "  [Rec] ${String.format("%.1f", seconds)}s, ${totalBytes / 1024}KB, peak=$maxSample, clip=${String.format("%.2f", clipPct)}%, rate=${actualRecordingSampleRate}Hz, src=$currentAudioSource")
                    }
                }
            }
            lastClipPercent = if (totalSamples > 0) (clippedSamples * 100.0 / totalSamples) else 0.0
            Log.i(TAG, "  Recording thread ended. Total: ${totalBytes / 1024}KB, peak=$maxSample, clip=${String.format("%.2f", lastClipPercent)}%, rate=${actualRecordingSampleRate}Hz")
        }, "MicRecordThread")
        recordingThread?.isDaemon = true
        recordingThread?.start()
    }

    private fun stopMicRecording(): ByteArray? {
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        recordingThread?.join(2000)
        recordingThread = null

        var data = synchronized(pcmBuffer) {
            pcmBuffer.toByteArray().also { pcmBuffer.reset() }
        }
        if (data.isEmpty()) return null

        Log.i(TAG, "  Raw PCM: ${data.size / 1024}KB, recorded at ${actualRecordingSampleRate}Hz")
        if (lastClipPercent > 1.0) {
            Log.w(TAG, "  ⚠ Input clipping detected: ${String.format("%.2f", lastClipPercent)}% of samples clipped (source=$lastRecordingAudioSource)")
        }

        // ★ RESAMPLE 8kHz → 16kHz if recorded at 8kHz ★
        if (actualRecordingSampleRate != SAMPLE_RATE && actualRecordingSampleRate > 0) {
            Log.i(TAG, "  ★ Resampling ${actualRecordingSampleRate}Hz → ${SAMPLE_RATE}Hz...")
            data = resamplePcm(data, actualRecordingSampleRate, SAMPLE_RATE)
            Log.i(TAG, "  ★ Resampled: ${data.size / 1024}KB (now ${SAMPLE_RATE}Hz)")
        }

        // ★ AUDIO PREPROCESSING — Dramatically improves Sherpa accuracy ★
        data = preprocessAudio(data)

        return data
    }

    /**
     * Audio preprocessing pipeline for cleaner Sherpa transcription.
     * Steps: DC offset removal → noise gate → gain normalization → high-pass filter.
     */
    private fun preprocessAudio(input: ByteArray): ByteArray {
        val numSamples = input.size / 2
        if (numSamples < 100) return input

        // Read into 16-bit samples
        val samples = ShortArray(numSamples)
        for (i in 0 until numSamples) {
            samples[i] = ((input[i * 2].toInt() and 0xFF) or (input[i * 2 + 1].toInt() shl 8)).toShort()
        }

        // 1) DC offset removal — fix baseline drift from BT mics
        var sum = 0L
        for (s in samples) sum += s
        val dcOffset = (sum / numSamples).toInt()
        if (Math.abs(dcOffset) > 10) {
            for (i in samples.indices) {
                samples[i] = (samples[i] - dcOffset).coerceIn(-32768, 32767).toShort()
            }
            Log.d(TAG, "  [Preprocess] DC offset removed: $dcOffset")
        }

        val isBluetoothInput = currentAudioSource == "bluetooth"

        // 2) Soft noise gate (disabled for Bluetooth to preserve far/quiet speech)
        if (!isBluetoothInput) {
            var rmsSum = 0.0
            for (s in samples) rmsSum += s.toDouble() * s.toDouble()
            val rms = Math.sqrt(rmsSum / numSamples)
            val noiseFloor = (rms * 0.015).toInt().coerceIn(8, 40)
            var attenuatedCount = 0
            for (i in samples.indices) {
                if (Math.abs(samples[i].toInt()) < noiseFloor) {
                    samples[i] = (samples[i] * 0.35).toInt().coerceIn(-32768, 32767).toShort()
                    attenuatedCount++
                }
            }
            Log.d(TAG, "  [Preprocess] Soft gate: floor=$noiseFloor, attenuated ${attenuatedCount * 100 / numSamples}%")
        } else {
            Log.d(TAG, "  [Preprocess] Bluetooth mode: skipping noise gate")
        }

        // 3) Gain normalization — boost quiet BT audio to 80% of full range
        var maxAmp = 0
        for (s in samples) {
            val abs = Math.abs(s.toInt())
            if (abs > maxAmp) maxAmp = abs
        }
        if (maxAmp > 50 && maxAmp < 30000) {
            val targetPeak = if (isBluetoothInput) 28000.0 else 24000.0
            val maxGain = if (isBluetoothInput) 10.0 else 8.0
            val gain = (targetPeak / maxAmp).coerceAtMost(maxGain)
            for (i in samples.indices) {
                val boosted = (samples[i] * gain).toInt().coerceIn(-32767, 32767)
                samples[i] = boosted.toShort()
            }
            Log.d(TAG, "  [Preprocess] Gain: ${String.format("%.1f", gain)}x (peak $maxAmp → ${(maxAmp * gain).toInt()})")
        }

        // 4) Simple high-pass filter (skip on Bluetooth to avoid thinning speech)
        if (!isBluetoothInput) {
            var prev = 0.0
            val alpha = 0.92
            for (i in samples.indices) {
                val filtered = alpha * (prev + samples[i] - (if (i > 0) samples[i - 1] else samples[i]))
                prev = filtered
                samples[i] = filtered.toInt().coerceIn(-32767, 32767).toShort()
            }
        } else {
            Log.d(TAG, "  [Preprocess] Bluetooth mode: skipping high-pass filter")
        }

        Log.i(TAG, "  ★ Audio preprocessed: DC→NoiseGate→Gain→HPF")

        // Write back to byte array
        val output = ByteArray(numSamples * 2)
        for (i in 0 until numSamples) {
            output[i * 2] = (samples[i].toInt() and 0xFF).toByte()
            output[i * 2 + 1] = (samples[i].toInt() shr 8).toByte()
        }
        return output
    }

    /**
     * Resample 16-bit mono PCM from one sample rate to another using linear interpolation.
     * This produces much cleaner audio than simple sample duplication.
     */
    private fun resamplePcm(input: ByteArray, fromRate: Int, toRate: Int): ByteArray {
        val numInputSamples = input.size / 2
        val ratio = fromRate.toDouble() / toRate.toDouble()
        val numOutputSamples = (numInputSamples / ratio).toInt()
        val output = ByteArray(numOutputSamples * 2)

        // Read input samples into short array for processing
        val inputSamples = ShortArray(numInputSamples)
        for (i in 0 until numInputSamples) {
            inputSamples[i] = ((input[i * 2].toInt() and 0xFF) or (input[i * 2 + 1].toInt() shl 8)).toShort()
        }

        // Linear interpolation resampling
        for (i in 0 until numOutputSamples) {
            val srcPos = i * ratio
            val srcIdx = srcPos.toInt()
            val frac = srcPos - srcIdx

            val sample: Short = if (srcIdx + 1 < numInputSamples) {
                // Interpolate between two samples
                val s0 = inputSamples[srcIdx].toInt()
                val s1 = inputSamples[srcIdx + 1].toInt()
                (s0 + (s1 - s0) * frac).toInt().toShort()
            } else if (srcIdx < numInputSamples) {
                inputSamples[srcIdx]
            } else {
                0
            }

            output[i * 2] = (sample.toInt() and 0xFF).toByte()
            output[i * 2 + 1] = (sample.toInt() shr 8).toByte()
        }

        return output
    }

    // ── Save PCM as WAV ──────────────────────────────────────

    private fun saveWav(pcmData: ByteArray): String {
        val recordingsDir = File(applicationContext.filesDir, "recordings")
        recordingsDir.mkdirs()

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val wavFile = File(recordingsDir, "session_$timestamp.wav")

        // PCM data is ALWAYS 16kHz after resampling
        val wavSampleRate = SAMPLE_RATE

        FileOutputStream(wavFile).use { fos ->
            val totalDataLen = pcmData.size + 36
            val byteRate = wavSampleRate * 1 * 16 / 8

            // WAV header
            fos.write("RIFF".toByteArray())
            fos.write(intToByteArray(totalDataLen))
            fos.write("WAVE".toByteArray())
            fos.write("fmt ".toByteArray())
            fos.write(intToByteArray(16))        // Subchunk1Size (PCM)
            fos.write(shortToByteArray(1))       // AudioFormat (PCM)
            fos.write(shortToByteArray(1))       // NumChannels (Mono)
            fos.write(intToByteArray(wavSampleRate))
            fos.write(intToByteArray(byteRate))
            fos.write(shortToByteArray(2))       // BlockAlign
            fos.write(shortToByteArray(16))      // BitsPerSample
            fos.write("data".toByteArray())
            fos.write(intToByteArray(pcmData.size))
            fos.write(pcmData)
        }

        var maxAmp = 0
        for (i in 0 until pcmData.size step 2) {
            val sample = Math.abs((pcmData[i].toInt() and 0xFF) or (pcmData[i + 1].toInt() shl 8))
            if (sample > maxAmp) maxAmp = sample
        }

        val durationSec = pcmData.size.toDouble() / (wavSampleRate * 2)
        Log.i(TAG, "  ✓ WAV saved: ${wavFile.absolutePath}")
        Log.i(TAG, "    → Size: ${pcmData.size / 1024}KB, Rate: ${wavSampleRate}Hz, Duration: ${String.format("%.1f", durationSec)}s")
        Log.i(TAG, "    → Peak Amplitude: $maxAmp")
        return wavFile.absolutePath
    }

    private fun intToByteArray(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xff).toByte(),
            (value shr 8 and 0xff).toByte(),
            (value shr 16 and 0xff).toByte(),
            (value shr 24 and 0xff).toByte()
        )
    }

    private fun shortToByteArray(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xff).toByte(),
            (value shr 8 and 0xff).toByte()
        )
    }

    // ── Live Speech Recognition (runs DURING recording) ──────

    private fun startLiveTranscription() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            Log.w(TAG, "  SpeechRecognizer not available")
            return
        }

        liveTranscript.clear()
        transcriptReady = false

        liveRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            // Keep listening as long as possible — extended for full conversations
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 15000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 12000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, 60000L)
        }

        liveRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onResults(results: android.os.Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val text = matches?.firstOrNull() ?: ""
                if (text.isNotBlank()) {
                    transcriptSegmentCount++
                    val now = System.currentTimeMillis()
                    val gap = if (lastSegmentTime > 0) now - lastSegmentTime else 0
                    lastSegmentTime = now
                    if (liveTranscript.isNotEmpty()) liveTranscript.append(". ")
                    liveTranscript.append(text)
                    Log.i(TAG, "  [LiveASR] Segment #$transcriptSegmentCount (gap=${gap}ms): '${text.take(80)}'")
                    Log.i(TAG, "  [LiveASR] Total transcript: ${liveTranscript.length} chars")
                } else {
                    Log.w(TAG, "  [LiveASR] Empty result received")
                }
                transcriptReady = true
                // Restart if still recording — continuous capture
                if (isRecording) {
                    try {
                        liveRecognizer?.startListening(intent)
                        Log.d(TAG, "  [LiveASR] Restarted for continuous listening")
                    } catch (e: Exception) {
                        Log.e(TAG, "  [LiveASR] Restart failed: ${e.message}")
                    }
                }
            }

            override fun onPartialResults(partialResults: android.os.Bundle?) {
                val partial = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val text = partial?.firstOrNull() ?: ""
                if (text.isNotBlank()) {
                    Log.d(TAG, "  [LiveASR] Partial: '${text.take(50)}'")
                }
            }

            override fun onError(error: Int) {
                val errorMsg = when (error) {
                    SpeechRecognizer.ERROR_AUDIO -> "Audio error"
                    SpeechRecognizer.ERROR_NO_MATCH -> "No match"
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Timeout"
                    SpeechRecognizer.ERROR_CLIENT -> "Client error"
                    else -> "Error $error"
                }
                Log.w(TAG, "  [LiveASR] Error: $errorMsg")
                // Restart on timeout/no-match if still recording
                if (isRecording && (error == SpeechRecognizer.ERROR_NO_MATCH ||
                    error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT)) {
                    try {
                        liveRecognizer?.startListening(intent)
                        Log.d(TAG, "  [LiveASR] Restarted after $errorMsg")
                    } catch (e: Exception) {
                        Log.e(TAG, "  [LiveASR] Restart failed: ${e.message}")
                    }
                }
            }

            override fun onReadyForSpeech(params: android.os.Bundle?) {
                Log.d(TAG, "  [LiveASR] Ready for speech")
            }
            override fun onBeginningOfSpeech() {
                Log.d(TAG, "  [LiveASR] Speech started")
            }
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {
                Log.d(TAG, "  [LiveASR] Speech ended")
            }
            override fun onEvent(eventType: Int, params: android.os.Bundle?) {}
        })

        liveRecognizer?.startListening(intent)
        Log.i(TAG, "  ✓ Live transcription started")
    }

    private fun stopLiveTranscription(): String {
        try {
            liveRecognizer?.stopListening()
            liveRecognizer?.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "  LiveASR stop error: ${e.message}")
        }
        liveRecognizer = null
        val result = liveTranscript.toString().trim()
        Log.i(TAG, "  ✓ Live transcript collected (${result.length} chars): '${result.take(80)}'")
        return result
    }

    /**
     * BT Fallback: Play saved WAV through speaker and re-transcribe
     * via SpeechRecognizer from built-in mic.
     *
     * Called when BT recording captured audio (WAV peak > 500)
     * but SpeechRecognizer returned nothing.
     */
    private fun retryTranscriptionFromWav(
        wavPath: String,
        onComplete: (String) -> Unit
    ) {
        Log.i(TAG, "  [BT-Fallback] ━━━━━━━━━━━━━━━━━━━━━━━")
        Log.i(TAG, "  [BT-Fallback] Playing WAV → re-transcribing via built-in mic")

        val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager

        // Switch to NORMAL mode so built-in mic is active
        audioManager.mode = AudioManager.MODE_NORMAL
        audioManager.isSpeakerphoneOn = true

        // Set moderate volume (60% of max)
        val originalVol = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
        val maxVol = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, maxVol * 6 / 10, 0)

        val fallbackTranscript = StringBuilder()
        var segCount = 0
        var recognizerDone = false

        // Create fresh SpeechRecognizer for built-in mic
        val fbRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        val fbIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 10000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, 60000L)
        }

        fbRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onResults(results: android.os.Bundle?) {
                val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull() ?: ""
                if (text.isNotBlank()) {
                    segCount++
                    if (fallbackTranscript.isNotEmpty()) fallbackTranscript.append(". ")
                    fallbackTranscript.append(text)
                    Log.i(TAG, "  [BT-Fallback] Segment #$segCount: '${text.take(80)}'")
                }
            }
            override fun onError(error: Int) {
                val msg = when (error) {
                    SpeechRecognizer.ERROR_NO_MATCH -> "No match"
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Timeout"
                    SpeechRecognizer.ERROR_AUDIO -> "Audio error"
                    else -> "Error $error"
                }
                Log.w(TAG, "  [BT-Fallback] Recognizer error: $msg")
            }
            override fun onReadyForSpeech(p: android.os.Bundle?) {
                Log.d(TAG, "  [BT-Fallback] Recognizer ready")
            }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rms: Float) {}
            override fun onBufferReceived(b: ByteArray?) {}
            override fun onEndOfSpeech() {
                Log.d(TAG, "  [BT-Fallback] Speech ended")
            }
            override fun onPartialResults(p: android.os.Bundle?) {}
            override fun onEvent(t: Int, p: android.os.Bundle?) {}
        })

        // Start recognizer FIRST (needs to be listening before playback)
        fbRecognizer.startListening(fbIntent)
        Log.i(TAG, "  [BT-Fallback] Recognizer listening from built-in mic...")

        // Play WAV after 800ms (let recognizer initialize)
        mainHandler.postDelayed({
            try {
                val player = android.media.MediaPlayer().apply {
                    setDataSource(wavPath)
                    @Suppress("DEPRECATION")
                    setAudioStreamType(AudioManager.STREAM_MUSIC)
                    prepare()
                }
                Log.i(TAG, "  [BT-Fallback] Playing WAV (${player.duration}ms) through speaker")

                player.setOnCompletionListener {
                    Log.i(TAG, "  [BT-Fallback] Playback complete")
                    // Wait 3s after playback for recognizer to finish
                    mainHandler.postDelayed({
                        if (!recognizerDone) {
                            recognizerDone = true
                            try {
                                fbRecognizer.stopListening()
                                fbRecognizer.destroy()
                            } catch (_: Exception) {}

                            // Restore audio settings
                            audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, originalVol, 0)
                            audioManager.isSpeakerphoneOn = false

                            val result = fallbackTranscript.toString().trim()
                            Log.i(TAG, "  [BT-Fallback] Result: '${result.take(100)}' ($segCount segs)")
                            Log.i(TAG, "  [BT-Fallback] ━━━━━━━━━━━━━━━━━━━━━━━")
                            onComplete(result)
                        }
                    }, 3000)
                }
                player.start()
            } catch (e: Exception) {
                Log.e(TAG, "  [BT-Fallback] Playback error: ${e.message}")
                audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, originalVol, 0)
                audioManager.isSpeakerphoneOn = false
                onComplete("")
            }
        }, 800)

        // Hard timeout: 90 seconds max
        mainHandler.postDelayed({
            if (!recognizerDone) {
                recognizerDone = true
                Log.w(TAG, "  [BT-Fallback] Hard timeout (90s)")
                try { fbRecognizer.destroy() } catch (_: Exception) {}
                audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, originalVol, 0)
                audioManager.isSpeakerphoneOn = false
                onComplete(fallbackTranscript.toString().trim())
            }
        }, 90000)
    }

    // ── Voice Enrollment ─────────────────────────────────────

    private fun enrollFromAudio(wavPath: String, name: String, context: Context): Map<String, Any> {
        Log.i(TAG, "━━━ Voice Enrollment ━━━━━━━━━━━━━━━━━")
        Log.i(TAG, "  Name: $name, File: $wavPath")

        // Check if required models are loaded
        if (!SherpaTranscriber.isAsrReady()) {
            Log.w(TAG, "  ✗ ASR model not loaded yet")
            return mapOf(
                "status" to "error",
                "message" to "Speech Model not ready. Please download it from Settings first."
            )
        }
        if (!SherpaTranscriber.isSpkReady()) {
            Log.w(TAG, "  ✗ Speaker model not loaded yet")
            return mapOf(
                "status" to "error",
                "message" to "Speaker Model not ready. Please download it from Settings → Speech & Intelligence first."
            )
        }

        // maxChunks=0 lets transcriber keep all robust voice-active chunks (up to safety cap).
        val xvectors = SherpaTranscriber.extractXVectors(wavPath, maxChunks = 0)
        if (xvectors.isEmpty()) {
            Log.w(TAG, "  ✗ Could not extract voice fingerprint")
            return mapOf(
                "status" to "error",
                "message" to "Could not extract voice fingerprint. Please speak clearly for at least 5 seconds."
            )
        }

        val profileId = SpeakerEngine.enrollSpeaker(context, name, xvectors)
        Log.i(TAG, "  ✓ Enrolled: $name (id=$profileId, fingerprints=${xvectors.size})")
        Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return mapOf(
            "status" to "ok",
            "id" to profileId,
            "name" to name,
            "fingerprints" to xvectors.size,
            "message" to "Voice enrolled successfully! I'll recognize $name in future conversations."
        )
    }

    private fun isUnknownSpeakerLabel(label: String): Boolean {
        return Regex("^Speaker\\s+\\d+$", RegexOption.IGNORE_CASE).matches(label.trim())
    }

    private fun normalizePersonName(raw: String): String {
        return raw.lowercase(Locale.US)
            .split(Regex("\\s+"))
            .filter { it.isNotBlank() }
            .joinToString(" ") { token ->
                token.replaceFirstChar { c -> if (c.isLowerCase()) c.titlecase(Locale.US) else c.toString() }
            }
            .trim()
    }

    private fun extractAddressedNames(text: String): Set<String> {
        if (text.isBlank()) return emptySet()

        val stop = setOf(
            "the", "this", "that", "there", "where", "when", "what", "which", "with", "from",
            "please", "thanks", "thank", "hello", "hi", "hey", "okay", "ok", "listen",
            "doctor", "sir", "madam", "bro", "sis", "friend", "today", "tomorrow", "weekend",
            "sure", "later", "around", "time", "call", "quick", "available", "will", "be",
            "am", "pm", "morning", "evening", "night", "yes", "no", "alright", "fine",
            "overall", "also", "agree", "agreed", "great", "perfect", "done", "right", "then",
            "frontend", "backend", "front-end", "back-end", "database", "storage", "module", "modules",
            "ui", "demo", "review", "testing", "bugs", "submission", "everything",
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"
        )

        val selfIntro = Regex(
            "\\b(?:my name is|i am|i'm|this is)\\s+([A-Za-z][A-Za-z'-]{1,24}(?:\\s+[A-Za-z][A-Za-z'-]{1,24})?)",
            RegexOption.IGNORE_CASE
        )
        val selfNames = selfIntro.findAll(text)
            .map { normalizePersonName(it.groupValues[1]) }
            .toSet()

        val patterns = listOf(
            Regex("\\b(?:hey|hi|hello|listen|thanks|thank you|ok|okay)\\s+([A-Za-z][A-Za-z'-]{1,24})\\b", RegexOption.IGNORE_CASE),
            Regex("\\b(?:talking to|speaking to|with)\\s+([A-Za-z][A-Za-z'-]{1,24})\\b", RegexOption.IGNORE_CASE),
        )

        val out = linkedSetOf<String>()
        for (rx in patterns) {
            rx.findAll(text).forEach { m ->
                val raw = m.groupValues.getOrNull(1)?.trim().orEmpty()
                if (raw.length < 3) return@forEach
                val lower = raw
                    .lowercase(Locale.US)
                    .replace(Regex("[^a-z'-]"), "")
                    .trim(' ', '\'', '-')
                if (lower in stop) return@forEach
                if (lower.matches(Regex("\\d+"))) return@forEach
                if (lower.endsWith("am") || lower.endsWith("pm")) return@forEach
                val normalized = normalizePersonName(raw)
                if (normalized in selfNames) return@forEach
                out.add(normalized)
            }
        }
        return out
    }

    private fun autoNameUnknownSpeakers(
        context: Context,
        segments: List<Map<String, Any>>,
    ): Pair<List<Map<String, Any>>, List<String>> {
        if (segments.size < 2) return segments to emptyList()

        data class Turn(val speaker: String, val text: String)
        val turns = segments.mapNotNull { seg ->
            val speaker = (seg["speaker"] ?: "").toString().trim()
            val text = (seg["text"] ?: "").toString().trim()
            if (speaker.isBlank() || text.isBlank()) null else Turn(speaker, text)
        }
        if (turns.size < 2) return segments to emptyList()

        val unknownLabels = turns.map { it.speaker }.distinct().filter { isUnknownSpeakerLabel(it) }
        if (unknownLabels.isEmpty()) return segments to emptyList()
        val labelTurnCounts = turns.groupingBy { it.speaker }.eachCount()

        val votes = mutableMapOf<String, MutableMap<String, Int>>()
        for (i in turns.indices) {
            val names = extractAddressedNames(turns[i].text)
            if (names.isEmpty()) continue

            fun addVote(targetSpeaker: String, weight: Int) {
                val bucket = votes.getOrPut(targetSpeaker) { mutableMapOf() }
                for (name in names) {
                    bucket[name] = bucket.getOrDefault(name, 0) + weight
                }
            }

            var linked = false

            for (step in 1..3) {
                val j = i + step
                if (j >= turns.size) break
                val targetSpeaker = turns[j].speaker
                if (targetSpeaker.equals(turns[i].speaker, ignoreCase = true)) continue

                val weight = 4 - step // nearer reply gets higher weight
                addVote(targetSpeaker, weight)
                linked = true
                break
            }

            // If no future reply exists (e.g., name is spoken in the last turn),
            // map mention to the nearest previous opposing turn.
            if (!linked) {
                for (step in 1..3) {
                    val j = i - step
                    if (j < 0) break
                    val targetSpeaker = turns[j].speaker
                    if (targetSpeaker.equals(turns[i].speaker, ignoreCase = true)) continue

                    val weight = 3 - step
                    if (weight > 0) {
                        addVote(targetSpeaker, weight)
                    }
                    break
                }
            }
        }

        val tentative = mutableMapOf<String, Pair<String, Int>>()
        for (label in unknownLabels) {
            val options = votes[label] ?: continue
            val best = options.maxByOrNull { it.value } ?: continue
            if (best.value >= 6) {
                tentative[label] = best.key to best.value
            }
        }
        if (tentative.isEmpty()) return segments to emptyList()

        val finalMap = mutableMapOf<String, String>()
        val usedNames = mutableSetOf<String>()
        tentative.entries
            .sortedByDescending { it.value.second }
            .forEach { (label, pair) ->
                val name = pair.first
                val key = name.lowercase(Locale.US)
                if (key !in usedNames) {
                    usedNames.add(key)
                    finalMap[label] = name
                }
            }

        if (finalMap.isEmpty()) return segments to emptyList()

        val renamed = segments.map { seg ->
            val speaker = (seg["speaker"] ?: "").toString()
            val newName = finalMap.entries.firstOrNull { it.key.equals(speaker, ignoreCase = true) }?.value
            if (newName != null) {
                seg.toMutableMap().apply { this["speaker"] = newName }
            } else {
                seg
            }
        }

        val learned = mutableListOf<String>()
        for ((label, name) in finalMap) {
            val turnsForLabel = labelTurnCounts[label] ?: 0
            if (turnsForLabel >= 6) {
                val profileId = SpeakerEngine.enrollSessionSpeaker(context, label, name)
                if (!profileId.isNullOrBlank()) {
                    learned.add(name)
                    Log.i(TAG, "  ✓ Auto-enrolled from speech mention: $name (from $label, turns=$turnsForLabel)")
                }
            } else {
                Log.i(TAG, "  ↷ Skip auto-enroll for $name from $label (insufficient turns=$turnsForLabel)")
            }
        }

        return renamed to learned.distinct()
    }

    // ── Process captured text → events → DB ──────────────────

    private fun processAndStore(
        text: String,
        wavPath: String?,
        duration: Double,
        source: String,
        diarizedSegments: List<Map<String, Any>>? = null
    ): Map<String, Any?> {
        Log.i(TAG, "━━━ AUTO-PROCESS ━━━━━━━━━━━━━━━━━━━━━")
        Log.i(TAG, "  Transcript length: ${text.length}")
        Log.i(TAG, "  Source: $source")
        Log.i(TAG, "  Duration: ${duration}s")

        if (text.isBlank()) {
            // Even without transcript, save the audio reference
            if (wavPath != null) {
                val convId = db.saveConversation("[Audio: ${duration.toInt()}s recording]", "audio")
                Log.i(TAG, "  Saved audio reference (no transcript): $convId")
                return mapOf(
                    "status" to "ok",
                    "conversation_id" to convId,
                    "summary" to "Recorded ${duration.toInt()}s of audio. No speech detected for transcription.",
                    "events_saved" to 0,
                    "events_extracted" to 0,
                    "memory_count" to db.getMemoryCount(),
                    "audio_path" to wavPath,
                    "transcript" to ""
                )
            }
            return mapOf(
                "status" to "error",
                "summary" to "No speech captured. Try again or use text mode.",
                "events_saved" to 0
            )
        }

        // Step 1: Speaker separation
        val (speakerCount, structuredText, speakerSegmentCount) = if (!diarizedSegments.isNullOrEmpty()) {
            val (namedSegments, learnedNames) = autoNameUnknownSpeakers(this, diarizedSegments)
            if (learnedNames.isNotEmpty()) {
                Log.i(TAG, "  Auto-named unknown speaker(s): ${learnedNames.joinToString(", ")}")
            }

            val count = namedSegments.map { (it["speaker"] ?: "Speaker 1").toString() }.distinct().size
            val structured = if (count > 1) {
                namedSegments.joinToString("\n") {
                    "${(it["speaker"] ?: "Speaker 1")}: ${(it["text"] ?: "").toString()}"
                }
            } else {
                text
            }
            Triple(count, structured, namedSegments.size)
        } else {
            val speakerSegments = SimpleNlpProcessor.separateSpeakers(text)
            val count = speakerSegments.map { it.speaker }.distinct().size
            val structured = if (count > 1) {
                speakerSegments.joinToString("\n") { "${it.speaker}: ${it.text}" }
            } else text
            Triple(count, structured, speakerSegments.size)
        }
        Log.i(TAG, "  Step 0: $speakerCount speaker(s), $speakerSegmentCount segments")

        // Step 1: Save conversation (structured if multi-speaker)
        val convId = db.saveConversation(structuredText, source)
        Log.i(TAG, "  Step 1/4: Saved conversation ${convId.take(8)} (${structuredText.length} chars)")

        // Step 2: Extract and save events (from full text, not partial)
        val extracted = SimpleNlpProcessor.extractEvents(text)
        var savedEvents = 0
        for (ev in extracted) {
            val saved = db.saveEvent(convId, ev.type, ev.description, ev.date, ev.time, ev.person)
            if (saved != null) savedEvents++
        }

        // Fallback persistence: if event extractor finds nothing, keep key memory notes searchable.
        if (savedEvents == 0) {
            val fallbackNotes = SimpleNlpProcessor.extractKeyPoints(text).take(3)
            for (note in fallbackNotes) {
                val saved = db.saveEvent(convId, "note", note, null, null, null)
                if (saved != null) savedEvents++
            }
            if (fallbackNotes.isNotEmpty()) {
                Log.i(TAG, "  Step 2/4: Added ${fallbackNotes.size} fallback note event(s)")
            }
        }
        Log.i(TAG, "  Step 2/4: Extracted ${extracted.size}, saved $savedEvents events")

        // Step 3: Generate and save summary
        val summary = SimpleNlpProcessor.summarize(text)
        val keyPoints = SimpleNlpProcessor.extractKeyPoints(text)
        db.saveSummary(convId, summary, keyPoints.joinToString("\n"))
        Log.i(TAG, "  Step 3/4: Summary saved (${keyPoints.size} key points)")

        val memCount = db.getMemoryCount()
        val urgentItems = db.getUrgentItems(48)
        Log.i(TAG, "  Step 4/4: speakers=$speakerCount, memories=$memCount, urgent=${urgentItems.size}")
        Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return mapOf(
            "status" to "ok",
            "conversation_id" to convId,
            "summary" to summary,
            "key_points" to keyPoints,
            "events_extracted" to extracted.size,
            "events_saved" to savedEvents,
            "urgent_items" to urgentItems.take(5),
            "important_items" to extracted.filter { it.type in listOf("medication", "meeting") }.map {
                mapOf("type" to it.type, "description" to it.description)
            },
            "memory_count" to memCount,
            "transcript" to text,
            "audio_path" to (wavPath ?: "")
        )
    }

    // ══════════════════════════════════════════════════════════
    //  METHOD CHANNEL HANDLER
    // ══════════════════════════════════════════════════════════

    private fun parseEventDateTime(rawDate: String?, rawTime: String?): Date? {
        val datePart = rawDate?.trim().orEmpty()
        val timePart = rawTime?.trim().orEmpty()
        if (timePart.isBlank()) return null

        val now = Calendar.getInstance()
        val eventCal = Calendar.getInstance()

        // Resolve date
        val lowerDate = datePart.lowercase(Locale.US)
        when {
            lowerDate.isBlank() || lowerDate.contains("today") -> {
                // Keep current day.
            }
            lowerDate.contains("tomorrow") -> {
                eventCal.add(Calendar.DAY_OF_YEAR, 1)
            }
            else -> {
                val datePatterns = listOf(
                    "yyyy-MM-dd", "yyyy/MM/dd", "dd-MM-yyyy", "dd/MM/yyyy",
                    "MMM d, yyyy", "MMMM d, yyyy", "MMM d yyyy", "MMMM d yyyy",
                    "MMM d", "MMMM d", "d MMM", "d MMMM",
                    "EEE", "EEEE"
                )
                var parsedDate: Date? = null
                for (pattern in datePatterns) {
                    try {
                        val fmt = SimpleDateFormat(pattern, Locale.US)
                        fmt.isLenient = true
                        parsedDate = fmt.parse(datePart)
                        if (parsedDate != null) {
                            val parsedCal = Calendar.getInstance().apply { time = parsedDate }
                            if (pattern == "EEE" || pattern == "EEEE") {
                                val targetDow = parsedCal.get(Calendar.DAY_OF_WEEK)
                                val diff = (targetDow - now.get(Calendar.DAY_OF_WEEK) + 7) % 7
                                eventCal.time = now.time
                                eventCal.add(Calendar.DAY_OF_YEAR, if (diff == 0) 7 else diff)
                            } else {
                                eventCal.set(Calendar.YEAR, parsedCal.get(Calendar.YEAR))
                                eventCal.set(Calendar.MONTH, parsedCal.get(Calendar.MONTH))
                                eventCal.set(Calendar.DAY_OF_MONTH, parsedCal.get(Calendar.DAY_OF_MONTH))

                                // If year was not present in input, prefer current year.
                                if (!pattern.contains("yyyy")) {
                                    eventCal.set(Calendar.YEAR, now.get(Calendar.YEAR))
                                }
                            }
                            break
                        }
                    } catch (_: Exception) {
                    }
                }
            }
        }

        // Resolve time
        val normalizedTime = timePart
            .replace(Regex("(?i)(\\d)(am|pm)"), "$1 $2")
            .replace(".", "")
            .trim()

        val timePatterns = listOf("HH:mm", "H:mm", "h:mm a", "h a", "ha", "HHmm")
        var parsedTime: Date? = null
        for (pattern in timePatterns) {
            try {
                val fmt = SimpleDateFormat(pattern, Locale.US)
                fmt.isLenient = true
                parsedTime = fmt.parse(normalizedTime)
                if (parsedTime != null) break
            } catch (_: Exception) {
            }
        }
        if (parsedTime == null) return null

        val timeCal = Calendar.getInstance().apply { time = parsedTime }
        eventCal.set(Calendar.HOUR_OF_DAY, timeCal.get(Calendar.HOUR_OF_DAY))
        eventCal.set(Calendar.MINUTE, timeCal.get(Calendar.MINUTE))
        eventCal.set(Calendar.SECOND, 0)
        eventCal.set(Calendar.MILLISECOND, 0)

        // If date was implicit/today and time already passed, move to next day.
        if ((lowerDate.isBlank() || lowerDate.contains("today")) && eventCal.timeInMillis < now.timeInMillis) {
            eventCal.add(Calendar.DAY_OF_YEAR, 1)
        }

        return eventCal.time
    }

    private fun formatDate(date: Date): String {
        return SimpleDateFormat("yyyy-MM-dd", Locale.US).format(date)
    }

    private fun formatTime(date: Date): String {
        return SimpleDateFormat("HH:mm", Locale.US).format(date)
    }

    private fun handleMethodCall(call: MethodCall, result: MethodChannel.Result) {
        Log.d(TAG, "→ MethodCall: ${call.method}")
        try {
            when (call.method) {

                // ── Health Check ──────────────────────────
                "isReady" -> {
                    Log.i(TAG, "isReady → true (DB initialized)")
                    result.success(true)
                }

                // ── Process Text (manual text input) ─────
                "processText" -> {
                    val text = call.argument<String>("text") ?: ""
                    if (text.isBlank()) {
                        result.success(mapOf("status" to "error", "error" to "No text provided"))
                        return
                    }
                    val res = processAndStore(text, null, 0.0, "text")
                    result.success(res)
                }

                // ── Process Audio (from file) ────────────
                "processAudio" -> {
                    val filePath = call.argument<String>("file_path") ?: ""
                    Log.i(TAG, "━━━ processAudio ━━━━━━━━━━━━━━━━━━━━")
                    Log.i(TAG, "  File: $filePath")

                    // For file-based audio, store reference with metadata
                    val res = processAndStore(
                        "[Audio file: ${filePath.substringAfterLast("/")}]",
                        filePath, 0.0, "audio_file"
                    )
                    result.success(res)
                }

                // ══════════════════════════════════════════
                //  RECORDING SESSION — THE CRITICAL FIX
                // ══════════════════════════════════════════

                "startRecording", "startBackgroundListening" -> {
                    Log.i(TAG, "━━━ startRecording ━━━━━━━━━━━━━━━━━━")
                    Log.i(TAG, "  Source: $currentAudioSource")
                    Log.i(TAG, "  Mode: AudioRecord ONLY (post-recording Sherpa)")

                    if (isRecording) {
                        result.success(mapOf("status" to "already_recording"))
                        return
                    }

                    if (!ensureRecordPermission()) {
                        result.success(mapOf(
                            "status" to "permission_required",
                            "error" to "Microphone permission needed"
                        ))
                        return
                    }

                    isRecording = true
                    audioSourceActive = true
                    recordingStartTime = System.currentTimeMillis()
                    liveTranscript.clear()

                    // ★ AudioRecord ONLY — NO SpeechRecognizer ★
                    if (currentAudioSource == "bluetooth") {
                        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
                        if (scoConnected || am.isBluetoothScoOn) {
                            // SCO already active — start recording immediately
                            startMicRecording()
                        } else {
                            // SCO not connected yet — initiate it, then record when connected
                            Log.i(TAG, "  BT SCO not connected, initiating SCO...")
                            pendingBtRecordAction = { startMicRecording() }
                            val scoStarted = startBluetoothSco()
                            if (!scoStarted) {
                                Log.w(TAG, "  ⚠ SCO failed to start, falling back to built-in mic")
                                pendingBtRecordAction = null
                                startMicRecording()
                            }
                        }
                    } else {
                        startMicRecording()
                    }

                    Log.i(TAG, "  ✓ Recording started")
                    Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    result.success(mapOf(
                        "status" to "recording",
                        "source" to currentAudioSource,
                        "active" to true
                    ))
                }

                "stopRecording", "stopBackgroundListening" -> {
                    Log.i(TAG, "━━━ stopRecording ━━━━━━━━━━━━━━━━━━━")
                    val duration = if (recordingStartTime > 0)
                        (System.currentTimeMillis() - recordingStartTime) / 1000.0 else 0.0
                    Log.i(TAG, "  Duration: ${duration}s, Source: $currentAudioSource")

                    if (!isRecording) {
                        result.success(mapOf("status" to "not_recording"))
                        return
                    }

                    isRecording = false
                    audioSourceActive = false
                    pendingBtRecordAction = null

                    // ★ Step 1: Save WAV (before stopping SCO) ★
                    val pcmData = stopMicRecording()
                    var wavPath: String? = null
                    var wavPeakAmplitude = 0
                    if (pcmData != null && pcmData.isNotEmpty()) {
                        wavPath = saveWav(pcmData)
                        lastWavPath = wavPath
                        for (i in 0 until pcmData.size - 1 step 2) {
                            val sample = Math.abs((pcmData[i].toInt() and 0xFF) or (pcmData[i + 1].toInt() shl 8))
                            if (sample > wavPeakAmplitude) wavPeakAmplitude = sample
                        }
                        Log.i(TAG, "  ✓ WAV saved: $wavPath (${pcmData.size / 1024}KB, peak=$wavPeakAmplitude)")
                    } else {
                        Log.w(TAG, "  ✗ No PCM data captured")
                    }

                    // Stop BT SCO after WAV saved
                    if (currentAudioSource == "bluetooth") stopBluetoothSco()
                    recordingStartTime = 0

                    if (wavPath == null || wavPeakAmplitude < 30) { // Very low threshold — catches quiet BT audio from far away
                        Log.w(TAG, "  ⚠ No audio captured (peak=$wavPeakAmplitude)")
                        result.success(mapOf(
                            "status" to "ok",
                            "summary" to "No speech detected. Try speaking louder.",
                            "events_saved" to 0
                        ))
                        Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        return
                    }

                    // ★ Step 2: Transcribe with sherpa-onnx on-device (background thread) ★
                    val finalWavPath = wavPath
                    val finalDuration = duration
                    val finalSource = if (currentAudioSource == "bluetooth") "bluetooth" else "microphone"

                    if (!SherpaTranscriber.isReady()) {
                        Log.w(TAG, "  ⚠ Speech model still initializing...")
                        result.success(mapOf(
                            "status" to "ok",
                            "summary" to "Speech model initializing. Please wait and try again.",
                            "events_saved" to 0,
                            "audio_path" to finalWavPath
                        ))
                        Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        return
                    }

                    Log.i(TAG, "  ★ Transcribing with sherpa-onnx (post-recording)...")

                    val ctx = this

                    Thread {
                        // Use diarized transcription
                        var segments = SherpaTranscriber.transcribeWavWithSpeakers(finalWavPath, ctx)

                        val autoNamed = autoNameUnknownSpeakers(ctx, segments)
                        segments = autoNamed.first
                        if (autoNamed.second.isNotEmpty()) {
                            Log.i(TAG, "  ✓ Learned new speaker(s): ${autoNamed.second.joinToString(", ")}")
                        }

                        val diarizedTranscript = segments.joinToString(" ") { it["text"].toString() }
                            .replace("\\s+".toRegex(), " ").trim()
                        val diarizedNormalized = SherpaTranscriber.cleanupTranscript(diarizedTranscript).trim()

                        fun transcriptQuality(text: String): Double {
                            if (text.isBlank()) return -999.0
                            val filler = setOf("the", "it", "be", "could", "uh", "um", "hmm")
                            val words = text.lowercase().split("\\s+".toRegex()).filter { it.isNotBlank() }
                            if (words.isEmpty()) return -999.0
                            val meaningful = words.count { it.length >= 3 && it !in filler }
                            val fillerCount = words.count { it in filler }
                            val unique = words.toSet().size
                            return (meaningful * 2.0) + (unique * 0.3) - (fillerCount * 1.2)
                        }

                        var transcript = diarizedNormalized
                        var transcriptSource = "sherpa_diarized"

                        // Quality fallback: if diarized transcript is too short for duration,
                        // run plain ASR pass and use the cleaner text for processing.
                        val minExpectedChars = (finalDuration * 6.0).toInt() // conservative floor
                        val diarizedScore = transcriptQuality(diarizedNormalized)
                        if (transcript.length < minExpectedChars || transcript.length < 40 || diarizedScore < 8.0) {
                            Log.w(TAG, "  ⚠ Diarized transcript seems short (${transcript.length} chars for ${"%.1f".format(finalDuration)}s), running plain fallback...")
                            val plainText = SherpaTranscriber.transcribeWav(finalWavPath).trim()
                            val plainNormalized = SherpaTranscriber.cleanupTranscript(plainText).trim()
                            val plainScore = transcriptQuality(plainNormalized)

                            if (plainNormalized.isNotEmpty() &&
                                (plainScore > diarizedScore + 1.0 || plainNormalized.length > transcript.length + 15)
                            ) {
                                transcript = plainNormalized
                                transcriptSource = "sherpa_plain_fallback"
                                Log.i(TAG, "  ✓ Fallback improved transcript (score ${"%.1f".format(diarizedScore)} → ${"%.1f".format(plainScore)})")
                            }
                        }

                        runOnUiThread {
                            if (transcript.isNotEmpty()) {
                                // Build diarized text for display: "Speaker: text"
                                val diarizedText = segments.joinToString("\n") {
                                    "${it["speaker"]}: ${it["text"]}"
                                }

                                if (segments.isNotEmpty()) {
                                    Log.i(TAG, "  ✓ Sherpa diarized: ${segments.size} segments")
                                    segments.forEach { Log.i(TAG, "    [${it["speaker"]}] ${it["text"].toString().take(60)}") }
                                } else {
                                    Log.w(TAG, "  ⚠ No diarized segments; using plain transcript fallback")
                                }

                                val processResult = processAndStore(
                                    transcript,
                                    finalWavPath,
                                    finalDuration,
                                    finalSource,
                                    diarizedSegments = if (segments.isNotEmpty()) segments else null
                                )
                                val enriched = processResult.toMutableMap()
                                enriched["transcription_source"] = transcriptSource
                                enriched["diarized_text"] = diarizedText
                                enriched["speaker_segments"] = segments
                                enriched["full_transcript"] = transcript
                                try { result.success(enriched) } catch (_: Exception) {}
                            } else {
                                Log.w(TAG, "  ⚠ Sherpa returned empty")
                                try {
                                    result.success(mapOf(
                                        "status" to "ok",
                                        "summary" to "Audio captured (${finalDuration.toInt()}s) but no speech recognized.",
                                        "events_saved" to 0,
                                        "audio_path" to finalWavPath
                                    ))
                                } catch (_: Exception) {}
                            }
                            Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        }
                    }.start()
                }

                // ── Query Memory ──────────────────────────
                "queryMemory" -> {
                    val question = call.argument<String>("question") ?: ""
                    Log.i(TAG, "━━━ queryMemory ━━━━━━━━━━━━━━━━━━━━━")
                    Log.i(TAG, "  Question: '$question'")
                    Log.i(TAG, "  mode: llm-only")

                    if (question.isBlank()) {
                        result.success(mapOf(
                            "answer" to "Please ask a question.",
                            "results" to emptyList<Any>(),
                            "method" to "keyword"
                        ))
                        return
                    }

                    val queryResult = db.chatWithMemory(question)
                    val answer = queryResult["answer"] as? String ?: ""
                    val resultCount = (queryResult["results"] as? List<*>)?.size ?: 0
                    Log.i(TAG, "  Answer: ${answer.take(80)}...")
                    Log.i(TAG, "  Results: $resultCount")
                    Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                    result.success(queryResult)
                }

                // ── Intelligence Mode: Chat with Memory (LLM) ──
                "chatWithMemory" -> {
                    val question = call.argument<String>("question") ?: ""
                    if (question.isBlank()) {
                        result.success(mapOf(
                            "answer" to "Please ask me a question about your memories.",
                            "related_events" to emptyList<Any>(),
                            "confidence" to "none",
                            "context_used" to false
                        ))
                        return
                    }
                    // Run on background thread (LLM call is blocking HTTP)
                    Thread {
                        try {
                            val chatResult = db.chatWithMemory(question)
                            runOnUiThread { result.success(chatResult) }
                        } catch (e: Exception) {
                            Log.e(TAG, "chatWithMemory error: ${e.message}")
                            runOnUiThread {
                                result.success(mapOf(
                                    "answer" to "I'm having trouble thinking right now. Please try again.",
                                    "related_events" to emptyList<Any>(),
                                    "confidence" to "none",
                                    "context_used" to false,
                                    "mode" to "error"
                                ))
                            }
                        }
                    }.start()
                }

                // ── Intelligence Mode: Urgent Items ──────
                "getUrgentItems" -> {
                    val hours = call.argument<Int>("hours") ?: 48
                    val items = db.getUrgentItems(hours)
                    result.success(mapOf("items" to items, "count" to items.size))
                }

                // ── Events ────────────────────────────────
                "getEvents" -> {
                    val typeFilter = call.argument<String>("type")
                    val events = db.getAllEvents(typeFilter)
                    Log.i(TAG, "getEvents(type=$typeFilter) → ${events.size} results")
                    result.success(mapOf("events" to events, "count" to events.size))
                }

                "getUpcoming" -> {
                    val windowMinutes = call.argument<Int>("minutes") ?: 60
                    val nowMs = System.currentTimeMillis()
                    val maxMs = nowMs + windowMinutes * 60_000L

                    val upcoming = db.getAllEvents()
                        .mapNotNull { event ->
                            val rawDate = event["raw_date"] as? String
                            val rawTime = event["raw_time"] as? String
                            val parsed = parseEventDateTime(rawDate, rawTime) ?: return@mapNotNull null
                            val eventMs = parsed.time
                            if (eventMs < nowMs || eventMs > maxMs) return@mapNotNull null

                            val minutesUntil = ((eventMs - nowMs) / 60_000L).toInt()
                            event.toMutableMap().apply {
                                put("parsed_date", formatDate(parsed))
                                put("parsed_time", formatTime(parsed))
                                put("minutes_until", minutesUntil)
                                put("event_epoch_ms", eventMs)
                            }
                        }
                        .sortedBy { (it["minutes_until"] as? Int) ?: Int.MAX_VALUE }

                    Log.i(TAG, "getUpcoming($windowMinutes min) -> ${upcoming.size} results")
                    result.success(mapOf("events" to upcoming, "count" to upcoming.size))
                }

                // ── Sherpa Status & Controls ──────────────────────────
                "getSherpaStatus" -> {
                    result.success(SherpaTranscriber.getStatus())
                }
                // ASR Controls
                "startAsrDownload" -> {
                    SherpaTranscriber.initAsr(this)
                    result.success(true)
                }
                "pauseAsrDownload" -> {
                    SherpaTranscriber.pauseAsr()
                    result.success(true)
                }
                "resumeAsrDownload" -> {
                    SherpaTranscriber.resumeAsr(this)
                    result.success(true)
                }
                "retryAsrDownload" -> {
                    SherpaTranscriber.retryAsr(this)
                    result.success(true)
                }
                // SPK Controls
                "startSpkDownload" -> {
                    SherpaTranscriber.initSpk(this)
                    result.success(true)
                }
                "pauseSpkDownload" -> {
                    SherpaTranscriber.pauseSpk()
                    result.success(true)
                }
                "resumeSpkDownload" -> {
                    SherpaTranscriber.resumeSpk(this)
                    result.success(true)
                }
                "retrySpkDownload" -> {
                    SherpaTranscriber.retrySpk(this)
                    result.success(true)
                }

                // ── Speaker Profiles ──────────────────────────
                "getSpeakerProfiles" -> {
                    SpeakerEngine.loadProfiles(this)
                    result.success(mapOf(
                        "profiles" to SpeakerEngine.getProfiles(),
                        "count" to SpeakerEngine.getProfiles().size
                    ))
                }
                "enrollVoice" -> {
                    val name = call.argument<String>("name") ?: ""
                    val audioPath = call.argument<String>("audio_path") ?: ""
                    Log.i(TAG, "━━━ enrollVoice ━━━━━━━━━━━━━━━━━━━━━")
                    Log.i(TAG, "  Name: $name, Audio: $audioPath")

                    if (name.isBlank()) {
                        result.success(mapOf("status" to "error", "message" to "Name is required"))
                        return
                    }

                    if (!SherpaTranscriber.isReady()) {
                        result.success(mapOf("status" to "error", "message" to "Speech model not ready yet"))
                        return
                    }

                    val ctx = this
                    Thread {
                        try {
                            // Process enrollment audio and persist voice profile.
                            val enrollResult = enrollFromAudio(audioPath, name, ctx)
                            
                            runOnUiThread {
                                try {
                                    result.success(enrollResult)
                                } catch (e: Exception) {
                                    Log.e(TAG, "Result already sent or error: ${e.message}")
                                }
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Enrollment error: ${e.message}")
                            runOnUiThread {
                                val errorMap = HashMap<String, Any>()
                                errorMap["status"] = "error"
                                errorMap["message"] = e.message ?: "Unknown error"
                                try {
                                    result.success(errorMap)
                                } catch (e2: Exception) {
                                    Log.e(TAG, "Could not send error result: ${e2.message}")
                                }
                            }
                        }
                    }.start()
                }
                "deleteSpeakerProfile" -> {
                    val id = call.argument<String>("id") ?: ""
                    SpeakerEngine.deleteProfile(this, id)
                    result.success(mapOf("status" to "ok"))
                }
                "getSpeakers" -> {
                    SpeakerEngine.loadProfiles(this)
                    val profiles = SpeakerEngine.getProfiles()
                    result.success(mapOf(
                        "speakers" to profiles,
                        "count" to profiles.size
                    ))
                }

                // ── Audio Source ──────────────────────────
                "setAudioSource" -> {
                    val sourceType = call.argument<String>("source_type") ?: "microphone"
                    val deviceName = call.argument<String>("device_name")
                    Log.i(TAG, "━━━ setAudioSource ━━━━━━━━━━━━━━━━━━")
                    Log.i(TAG, "  Type: $sourceType")
                    Log.i(TAG, "  Device: ${deviceName ?: "default"}")

                    // Stop previous BT SCO if switching away from bluetooth
                    if (currentAudioSource == "bluetooth" && sourceType != "bluetooth") {
                        stopBluetoothSco()
                    }

                    if (sourceType == "bluetooth") {
                        val scoOk = startBluetoothSco()
                        if (!scoOk) {
                            Log.w(TAG, "  ⚠ BT SCO failed — falling back to microphone")
                            currentAudioSource = "microphone"
                            audioSourceActive = true
                            result.success(mapOf(
                                "status" to "fallback",
                                "type" to "microphone",
                                "active" to true,
                                "device_name" to "",
                                "error" to "Bluetooth not available. Using phone microphone."
                            ))
                            return
                        }
                        bluetoothDeviceName = deviceName ?: bluetoothDeviceName ?: "Bluetooth Device"
                        btAudioBuffer.reset()
                        currentAudioSource = "bluetooth"
                        audioSourceActive = true
                        Log.i(TAG, "  ✓ Bluetooth source activated: $bluetoothDeviceName")
                    } else {
                        currentAudioSource = sourceType
                        audioSourceActive = true
                    }

                    Log.i(TAG, "  Final source: $currentAudioSource")
                    Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                    result.success(mapOf(
                        "status" to "ok",
                        "type" to currentAudioSource,
                        "active" to true,
                        "device_name" to (bluetoothDeviceName ?: "")
                    ))
                }

                "pushBluetoothAudio" -> {
                    val pcmData = call.argument<ByteArray>("pcm_data")
                    if (pcmData != null && isRecording) {
                        btAudioBuffer.write(pcmData)
                        val rms = calculateRms(pcmData)
                        Log.v(TAG, "pushBT: +${pcmData.size}b (total: ${btAudioBuffer.size()}) RMS: ${String.format("%.2f", rms)}")
                        result.success(mapOf(
                            "status" to "ok",
                            "samples_written" to pcmData.size,
                            "buffer_size" to btAudioBuffer.size(),
                            "rms" to rms
                        ))
                    } else {
                        val reason = if (pcmData == null) "no data" else "not recording"
                        Log.w(TAG, "pushBT rejected: $reason")
                        result.success(mapOf("status" to "error", "error" to reason))
                    }
                }

                "getAudioSourceInfo" -> {
                    val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
                    val inputs = am.getDevices(AudioManager.GET_DEVICES_INPUTS)
                    val inputList = inputs.map { "${it.productName}(type=${it.type})" }
                    Log.d(TAG, "getAudioSourceInfo → src=$currentAudioSource, sco=$scoConnected, mode=${am.mode}")
                    Log.d(TAG, "  Inputs: $inputList")
                    result.success(mapOf(
                        "type" to currentAudioSource,
                        "active" to audioSourceActive,
                        "device_name" to (bluetoothDeviceName ?: ""),
                        "is_recording" to isRecording,
                        "buffer_size" to btAudioBuffer.size(),
                        "sco_connected" to scoConnected,
                        "bluetooth_sco_on" to am.isBluetoothScoOn,
                        "audio_mode" to am.mode,
                        "input_devices" to inputList
                    ))
                }

                "getBluetoothDevices" -> {
                    Log.i(TAG, "━━━ getBluetoothDevices ━━━━━━━━━━━━━")
                    val devices = mutableListOf<Map<String, Any>>()
                    try {
                        val btManager = getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager
                        val btAdapter = btManager?.adapter
                        if (btAdapter != null && btAdapter.isEnabled) {
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
                                ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT)
                                != PackageManager.PERMISSION_GRANTED) {
                                Log.w(TAG, "  BT permission not granted")
                            } else {
                                btAdapter.bondedDevices?.forEach { device ->
                                    val major = device.bluetoothClass?.majorDeviceClass ?: -1
                                    val isAudio = major == BluetoothClass.Device.Major.AUDIO_VIDEO ||
                                                  major == BluetoothClass.Device.Major.WEARABLE
                                    devices.add(mapOf(
                                        "name" to (device.name ?: "Unknown"),
                                        "address" to device.address,
                                        "type" to (if (isAudio) "audio" else "other"),
                                        "major_class" to major
                                    ))
                                    Log.i(TAG, "  ${device.name} [${if (isAudio) "AUDIO" else "other"}] ${device.address}")
                                }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "  BT enumeration error: ${e.message}")
                    }
                    Log.i(TAG, "  Total: ${devices.size} devices")
                    Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    result.success(mapOf("devices" to devices))
                }

                // ── Stats & Debug ─────────────────────────
                "getStats" -> {
                    val stats = db.getStats()
                    Log.i(TAG, "getStats → $stats")
                    result.success(stats)
                }

                "getMemoryCount" -> {
                    val count = db.getMemoryCount()
                    Log.i(TAG, "getMemoryCount → $count")
                    result.success(mapOf("count" to count))
                }

                "getRecordings" -> {
                    val dir = File(applicationContext.filesDir, "recordings")
                    val files = dir.listFiles()?.map {
                        mapOf("path" to it.absolutePath, "name" to it.name, "size" to it.length())
                    } ?: emptyList()
                    result.success(mapOf("recordings" to files, "count" to files.size))
                }

                // ── Speakers ──────────────────────────────
                "assignSpeaker" -> {
                    result.success(mapOf("status" to "ok",
                        "label" to (call.argument<String>("label") ?: ""),
                        "name" to (call.argument<String>("name") ?: "")))
                }

                // ── Backup ────────────────────────────────
                "createBackup" -> result.success(mapOf("status" to "ok"))
                "restoreBackup" -> result.success(mapOf("status" to "ok"))
                "verifyBackup" -> result.success(mapOf("status" to "ok", "valid" to true))
                "listBackups" -> result.success(emptyList<Any>())

                // ── LLM / Worker ──────────────────────────
                "checkLlmStatus" -> {
                    Thread {
                        refreshLlmReadyState()
                        val status = getLlmStatusMap()
                        mainHandler.post { result.success(status) }
                    }.start()
                }
                "getLlmEndpoint" -> {
                    result.success(mapOf("endpoint" to llmEndpoint, "candidates" to listOf("file://app/models/phi2")))
                }
                "setLlmEndpoint" -> {
                    result.success(mapOf("status" to "error", "message" to "Endpoint override is disabled in local Phi-2 mode"))
                }
                "startLlmDownload" -> {
                    startLlmDownload()
                    result.success(mapOf("status" to "started"))
                }
                "pauseLlmDownload" -> {
                    Phi2ModelManager.pauseDownload()
                    refreshLlmReadyState()
                    result.success(mapOf("status" to "pausing"))
                }
                "resumeLlmDownload" -> {
                    Phi2ModelManager.resumeDownload(applicationContext)
                    refreshLlmReadyState()
                    result.success(mapOf("status" to "resumed"))
                }
                "retryLlmDownload" -> {
                    Phi2ModelManager.retryDownload(applicationContext)
                    refreshLlmReadyState()
                    result.success(mapOf("status" to "retrying"))
                }

                "getWorkerStatus" -> result.success(mapOf(
                    "is_running" to isRecording,
                    "source" to currentAudioSource,
                    "active" to audioSourceActive,
                    "mode" to if (isRecording) "recording" else "idle"
                ))

                // ── Phase Q/R ─────────────────────────────
                "getResourceStats" -> {
                    val stats = db.getStats()
                    result.success(stats + mapOf(
                        "memory_mb" to (Runtime.getRuntime().totalMemory() / 1024 / 1024),
                        "free_mb" to (Runtime.getRuntime().freeMemory() / 1024 / 1024)
                    ))
                }

                "getMemoryPatterns" -> result.success(emptyList<Any>())
                "getReinforcementItems" -> result.success(emptyList<Any>())
                "markItemShown" -> result.success(null)
                "checkEscalations" -> result.success(emptyList<Any>())

                "generateDailyBrief" -> {
                    val stats = db.getStats()
                    val convCount = stats["total_conversations"] as? Int ?: 0
                    val eventCount = stats["total_events"] as? Int ?: 0
                    val events = db.getAllEvents()
                    val meetings = events.filter { (it["type"] as? String) == "meeting" }
                    val meds = events.filter { (it["type"] as? String) == "medication" }
                    val tasks = events.filter { (it["type"] as? String) == "task" }

                    val briefParts = mutableListOf<String>()
                    briefParts.add("You have $convCount conversation(s) and $eventCount event(s) stored.")
                    if (meetings.isNotEmpty()) {
                        briefParts.add("${meetings.size} meeting(s): ${meetings.joinToString(", ") { 
                            (it["description"] as? String ?: "").take(40)
                        }}")
                    }
                    if (meds.isNotEmpty()) {
                        briefParts.add("${meds.size} medication reminder(s)")
                    }
                    if (tasks.isNotEmpty()) {
                        briefParts.add("${tasks.size} task(s) pending")
                    }

                    result.success(mapOf(
                        "greeting" to "Good day! Here's your memory summary.",
                        "total_conversations" to convCount,
                        "total_events" to eventCount,
                        "brief" to briefParts.joinToString("\n"),
                        "meetings" to meetings.size,
                        "medications" to meds.size,
                        "tasks" to tasks.size
                    ))
                }

                "setConfigFlag" -> {
                    val key = call.argument<String>("key") ?: ""
                    val value = call.argument<Boolean>("value") ?: false
                    Log.i(TAG, "setConfigFlag: $key = $value")
                    result.success(mapOf("status" to "ok", "key" to key, "value" to value))
                }

                else -> {
                    Log.w(TAG, "⚠ Unhandled method: ${call.method}")
                    result.notImplemented()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "✗ Error in ${call.method}: ${e.message}", e)
            result.error("BRIDGE_ERROR", "Error in ${call.method}: ${e.message}", e.stackTraceToString())
        }
    }
}
