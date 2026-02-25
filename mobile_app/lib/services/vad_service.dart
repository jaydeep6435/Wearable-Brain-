/// VAD Service — Voice Activity Detection with Auto-Recording
///
/// Monitors microphone amplitude (dBFS) to detect speech.
/// When speech is detected → keeps recording the segment.
/// When silence returns for 3s → stops recording → uploads to backend.
///
/// State machine: Idle → Listening → Recording → Processing → Listening
///
/// Approach: Always records to a temp file while monitoring amplitude.
/// If speech never occurs, the probe file is discarded.
/// If speech occurs, the file becomes the segment.

import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'api_service.dart';

// ── State enum ────────────────────────────────────────────────
enum VadState { idle, listening, recording, processing }

// ── Callback types ────────────────────────────────────────────
typedef VadStateCallback = void Function(VadState state);
typedef VadResultCallback = void Function(Map<String, dynamic> result);
typedef VadErrorCallback = void Function(String error);
typedef VadAmplitudeCallback = void Function(double dBFS);

class VadService {
  // ── Configuration ───────────────────────────────────────────
  /// Amplitude above this → speech detected (typical speech: -20 to -10 dBFS)
  static const double speechThreshold = -30.0;

  /// Seconds of silence before stopping recording
  static const int silenceTimeoutSec = 3;

  /// Maximum recording length in seconds (auto-stop safety)
  static const int maxSegmentSec = 120;

  /// Minimum recording length in seconds (skip false triggers)
  static const int minSegmentSec = 2;

  /// Cooldown between uploads in seconds (debounce)
  static const int debounceSec = 5;

  /// Amplitude poll interval in milliseconds
  static const int pollIntervalMs = 250;

  // ── State ───────────────────────────────────────────────────
  final AudioRecorder _recorder = AudioRecorder();
  VadState _state = VadState.idle;
  VadState get state => _state;

  StreamSubscription<Amplitude>? _amplitudeSub;
  Timer? _silenceTimer;
  Timer? _maxDurationTimer;
  Timer? _probeRestartTimer;
  DateTime? _speechStartTime;
  DateTime? _lastUploadTime;
  String? _currentPath;
  int _segmentCount = 0;
  int get segmentCount => _segmentCount;
  bool _useLlm = false;
  bool _disposed = false;

  // ── Callbacks ───────────────────────────────────────────────
  VadStateCallback? onStateChanged;
  VadResultCallback? onSegmentProcessed;
  VadErrorCallback? onError;
  VadAmplitudeCallback? onAmplitude;

  // ── Public API ──────────────────────────────────────────────

  /// Start listening for speech
  Future<bool> startListening({bool useLlm = false}) async {
    if (_state != VadState.idle) return false;
    _disposed = false;

    // Check microphone permission
    if (!await _recorder.hasPermission()) {
      onError?.call('Microphone permission denied');
      return false;
    }

    _useLlm = useLlm;
    _segmentCount = 0;
    _setState(VadState.listening);

    // Start the probe recording (monitors amplitude)
    await _startProbe();

    debugPrint('[VAD] Listening started (threshold: $speechThreshold dBFS)');
    return true;
  }

  /// Stop listening entirely
  Future<void> stopListening() async {
    _disposed = true;
    _silenceTimer?.cancel();
    _maxDurationTimer?.cancel();
    _probeRestartTimer?.cancel();
    _amplitudeSub?.cancel();

    try {
      await _recorder.cancel();
    } catch (_) {}

    // Clean up probe file
    _deleteFile(_currentPath);
    _currentPath = null;

    _setState(VadState.idle);
    debugPrint('[VAD] Listening stopped');
  }

  /// Dispose resources
  void dispose() {
    _disposed = true;
    stopListening();
    _recorder.dispose();
  }

  // ── Probe Recording ─────────────────────────────────────────
  // We record to a temp file and monitor amplitude.
  // If amplitude stays below threshold → discard file, start new probe.
  // If amplitude rises above threshold → transition to "recording" state.

  Future<void> _startProbe() async {
    if (_disposed || _state == VadState.idle) return;

    try {
      final dir = await getTemporaryDirectory();
      _currentPath = p.join(
        dir.path,
        'vad_${DateTime.now().millisecondsSinceEpoch}.m4a',
      );

      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.aacLc,
          sampleRate: 16000,
          numChannels: 1,
          autoGain: true,
          echoCancel: true,
          noiseSuppress: true,
        ),
        path: _currentPath!,
      );

      // Monitor amplitude
      _amplitudeSub?.cancel();
      _amplitudeSub = _recorder
          .onAmplitudeChanged(const Duration(milliseconds: pollIntervalMs))
          .listen(_onAmplitude);

      // Auto-restart probe every 10s to keep files small (if no speech)
      _probeRestartTimer?.cancel();
      _probeRestartTimer = Timer(const Duration(seconds: 10), () {
        if (_state == VadState.listening && !_disposed) {
          _restartProbe();
        }
      });
    } catch (e) {
      debugPrint('[VAD] Probe start error: $e');
      // Retry after delay
      if (!_disposed && _state != VadState.idle) {
        _probeRestartTimer = Timer(const Duration(seconds: 2), _startProbe);
      }
    }
  }

  Future<void> _restartProbe() async {
    if (_disposed || _state != VadState.listening) return;
    _amplitudeSub?.cancel();

    try {
      await _recorder.cancel();
    } catch (_) {}

    _deleteFile(_currentPath);
    await _startProbe();
  }

  // ── Amplitude Handler ───────────────────────────────────────

  void _onAmplitude(Amplitude amp) {
    if (_disposed) return;
    final dBFS = amp.current;
    onAmplitude?.call(dBFS);

    if (_state == VadState.listening) {
      // In listening mode: check if speech starts
      if (dBFS > speechThreshold) {
        _onSpeechStart();
      }
    } else if (_state == VadState.recording) {
      // In recording mode: check for silence
      if (dBFS > speechThreshold) {
        // Speech continuing — reset silence timer
        _silenceTimer?.cancel();
        _silenceTimer = null;
      } else {
        // Silence detected — start countdown
        _silenceTimer ??= Timer(
          const Duration(seconds: silenceTimeoutSec),
          _onSilenceTimeout,
        );
      }
    }
  }

  // ── Speech Detection ────────────────────────────────────────

  void _onSpeechStart() {
    if (_state != VadState.listening) return;

    debugPrint('[VAD] Speech detected!');
    _probeRestartTimer?.cancel();
    _speechStartTime = DateTime.now();
    _setState(VadState.recording);

    // The current probe recording IS our segment now — just keep recording.
    // Set max duration safety timer
    _maxDurationTimer = Timer(
      const Duration(seconds: maxSegmentSec),
      () {
        debugPrint('[VAD] Max segment duration reached');
        _onSilenceTimeout();
      },
    );
  }

  // ── Silence → Stop Recording ────────────────────────────────

  Future<void> _onSilenceTimeout() async {
    if (_state != VadState.recording || _disposed) return;

    _silenceTimer?.cancel();
    _maxDurationTimer?.cancel();
    _amplitudeSub?.cancel();

    // Calculate duration
    final duration = _speechStartTime != null
        ? DateTime.now().difference(_speechStartTime!)
        : Duration.zero;

    try {
      final path = await _recorder.stop();

      if (duration.inSeconds < minSegmentSec) {
        debugPrint('[VAD] Segment too short (${duration.inSeconds}s) — skipping');
        _deleteFile(path);
        _backToListening();
        return;
      }

      debugPrint('[VAD] Segment complete: ${duration.inSeconds}s → $path');

      // Debounce check
      if (_lastUploadTime != null) {
        final since = DateTime.now().difference(_lastUploadTime!);
        if (since.inSeconds < debounceSec) {
          final wait = debounceSec - since.inSeconds;
          debugPrint('[VAD] Debounce: waiting ${wait}s');
          await Future.delayed(Duration(seconds: wait));
        }
      }

      if (path != null && path.isNotEmpty && !_disposed) {
        await _autoUpload(path);
      } else {
        _backToListening();
      }
    } catch (e) {
      debugPrint('[VAD] Stop error: $e');
      onError?.call('Recording stop failed: $e');
      _backToListening();
    }
  }

  // ── Auto Upload ─────────────────────────────────────────────

  Future<void> _autoUpload(String filePath) async {
    _setState(VadState.processing);

    try {
      debugPrint('[VAD] Uploading: $filePath');
      final result = await ApiService.processAudio(filePath, useLlm: _useLlm);

      _lastUploadTime = DateTime.now();
      _segmentCount++;

      final transcription = result['transcription'] ?? '';
      if (transcription.toString().trim().isEmpty) {
        debugPrint('[VAD] Empty transcription — no clear speech');
      } else {
        debugPrint('[VAD] ✅ Transcribed: ${transcription.toString().substring(
            0, transcription.toString().length.clamp(0, 80))}...');
        onSegmentProcessed?.call(result);
      }
    } catch (e) {
      debugPrint('[VAD] Upload error: $e');
      onError?.call('Upload failed: $e');
    }

    // Clean up segment file
    _deleteFile(filePath);

    // Resume listening
    _backToListening();
  }

  // ── Helpers ─────────────────────────────────────────────────

  void _backToListening() {
    if (_disposed || _state == VadState.idle) return;
    _setState(VadState.listening);
    _startProbe();
  }

  void _setState(VadState newState) {
    if (_state == newState) return;
    _state = newState;
    onStateChanged?.call(newState);
    debugPrint('[VAD] State → $newState');
  }

  void _deleteFile(String? path) {
    if (path == null) return;
    try {
      final file = File(path);
      if (file.existsSync()) file.deleteSync();
    } catch (_) {}
  }
}
