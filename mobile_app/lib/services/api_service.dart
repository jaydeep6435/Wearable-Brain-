/// API Service — MethodChannel Bridge to Python Engine
///
/// Replaces the old HTTP-based communication with direct
/// MethodChannel invocations. No network. No ports. No HTTP.
///
/// The Python engine runs as a background service and is
/// invoked directly via platform channels.
///
/// Channel: 'memory_assistant'

import 'package:flutter/services.dart';

class ApiService {
  static const _channel = MethodChannel('memory_assistant');

  // ── Processing ────────────────────────────────────────────

  /// Process conversation text through the full pipeline
  static Future<Map<String, dynamic>> processText(
    String text, {
    bool useLlm = false,
  }) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'processText',
        {'text': text, 'use_llm': useLlm},
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to process text: ${e.message}');
    }
  }

  /// Process audio file through the full pipeline
  static Future<Map<String, dynamic>> processAudio(
    String filePath, {
    bool useLlm = false,
  }) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'processAudio',
        {'file_path': filePath, 'use_llm': useLlm},
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to process audio: ${e.message}');
    }
  }

  // ── Session Recording ─────────────────────────────────────

  /// Start recording a conversation session
  static Future<Map<String, dynamic>> startRecording() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'startRecording',
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to start recording: ${e.message}');
    }
  }

  /// Stop recording and process through the full pipeline
  static Future<Map<String, dynamic>> stopRecording() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'stopRecording',
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to stop recording: ${e.message}');
    }
  }

  /// List all saved recordings
  static Future<Map<String, dynamic>> getRecordings() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'getRecordings',
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to list recordings: ${e.message}');
    }
  }

  // ── Query ─────────────────────────────────────────────────

  /// Query conversation memory
  static Future<Map<String, dynamic>> queryMemory(
    String question, {
    bool useLlm = false,
  }) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'queryMemory',
        {'question': question, 'use_llm': useLlm},
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to query: ${e.message}');
    }
  }

  // ── Events & Reminders ────────────────────────────────────

  /// Get all events (optional type filter)
  static Future<Map<String, dynamic>> getEvents({String? type}) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'getEvents',
        {'type': type},
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to get events: ${e.message}');
    }
  }

  /// Get upcoming events (within N minutes)
  static Future<Map<String, dynamic>> getUpcoming({int minutes = 60}) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'getUpcoming',
        {'minutes': minutes},
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to get upcoming events: ${e.message}');
    }
  }

  // ── Speakers ──────────────────────────────────────────────

  /// Get all speaker profiles
  static Future<Map<String, dynamic>> getSpeakers() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'getSpeakers',
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to get speakers: ${e.message}');
    }
  }

  /// Assign a name to a speaker label (e.g., SPEAKER_00 → "Doctor")
  static Future<Map<String, dynamic>> assignSpeaker(
    String label,
    String name,
  ) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'assignSpeaker',
        {'label': label, 'name': name},
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to assign speaker: ${e.message}');
    }
  }

  // ── Backup & Restore ─────────────────────────────────────

  /// Create a secure backup of the entire database
  static Future<Map<String, dynamic>> createBackup(String path) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'createBackup',
        {'path': path},
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to create backup: ${e.message}');
    }
  }

  /// Restore a database from a backup file
  static Future<Map<String, dynamic>> restoreBackup(String path) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'restoreBackup',
        {'path': path},
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to restore backup: ${e.message}');
    }
  }

  /// Verify backup file integrity
  static Future<Map<String, dynamic>> verifyBackup(String path) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'verifyBackup',
        {'path': path},
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to verify backup: ${e.message}');
    }
  }

  /// List all backup files in a directory
  static Future<List<dynamic>> listBackups(String directory) async {
    try {
      final result = await _channel.invokeListMethod<dynamic>(
        'listBackups',
        {'directory': directory},
      );
      return result ?? [];
    } on PlatformException catch (e) {
      throw Exception('Failed to list backups: ${e.message}');
    }
  }

  // ── Wearable Audio Sources ───────────────────────────────

  /// Switch audio source: "microphone", "bluetooth", or "file"
  static Future<Map<String, dynamic>> setAudioSource(
    String sourceType, {
    String? deviceName,
    String? filePath,
  }) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'setAudioSource',
        {
          'source_type': sourceType,
          'device_name': deviceName,
          'file_path': filePath,
        },
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to set audio source: ${e.message}');
    }
  }

  /// Push raw PCM audio from a Bluetooth device
  static Future<Map<String, dynamic>> pushBluetoothAudio(
    Uint8List pcmData,
  ) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'pushBluetoothAudio',
        {'pcm_data': pcmData},
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to push BT audio: ${e.message}');
    }
  }

  /// Get info about the currently active audio source
  static Future<Map<String, dynamic>> getAudioSourceInfo() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'getAudioSourceInfo',
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to get source info: ${e.message}');
    }
  }

  // ── Status & Health ───────────────────────────────────────

  /// Get engine statistics
  static Future<Map<String, dynamic>> getStats() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'getStats',
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to get stats: ${e.message}');
    }
  }

  /// Check LLM availability
  static Future<Map<String, dynamic>> checkLlmStatus() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'checkLlmStatus',
      );
      return result ?? {};
    } catch (_) {
      return {'status': 'error'};
    }
  }

  /// Get background worker status (recording/VAD)
  static Future<Map<String, dynamic>> getWorkerStatus() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'getWorkerStatus',
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to get worker status: ${e.message}');
    }
  }

  /// Health check — is the engine ready?
  static Future<bool> checkServer() async {
    try {
      final result = await _channel.invokeMethod<bool>('isReady');
      return result ?? false;
    } catch (_) {
      return false;
    }
  }

  // ── Phase Q/R: Prioritization & Reinforcement ─────────────

  /// Get resource usage statistics
  static Future<Map<String, dynamic>> getResourceStats() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'getResourceStats',
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to get resource stats: ${e.message}');
    }
  }

  /// Get urgent events (medication/appointments within N hours)
  static Future<List<dynamic>> getUrgentItems({int hours = 24}) async {
    try {
      final result = await _channel.invokeMethod<List<dynamic>>(
        'getUrgentItems',
        {'hours': hours},
      );
      return result ?? [];
    } on PlatformException catch (e) {
      throw Exception('Failed to get urgent items: ${e.message}');
    }
  }

  /// Get recurring conversation patterns
  static Future<List<dynamic>> getMemoryPatterns({
    int minFrequency = 1,
  }) async {
    try {
      final result = await _channel.invokeMethod<List<dynamic>>(
        'getMemoryPatterns',
        {'min_frequency': minFrequency},
      );
      return result ?? [];
    } on PlatformException catch (e) {
      throw Exception('Failed to get patterns: ${e.message}');
    }
  }

  /// Get critical events needing re-display
  static Future<List<dynamic>> getReinforcementItems() async {
    try {
      final result = await _channel.invokeMethod<List<dynamic>>(
        'getReinforcementItems',
      );
      return result ?? [];
    } on PlatformException catch (e) {
      throw Exception('Failed to get reinforcement items: ${e.message}');
    }
  }

  /// Mark a critical event as shown to the user
  static Future<void> markItemShown(String eventId) async {
    try {
      await _channel.invokeMethod('markItemShown', {'event_id': eventId});
    } on PlatformException catch (e) {
      throw Exception('Failed to mark item shown: ${e.message}');
    }
  }

  /// Check for missed/overdue events and escalate
  static Future<List<dynamic>> checkEscalations() async {
    try {
      final result = await _channel.invokeMethod<List<dynamic>>(
        'checkEscalations',
      );
      return result ?? [];
    } on PlatformException catch (e) {
      throw Exception('Failed to check escalations: ${e.message}');
    }
  }

  /// Generate a calm, structured daily summary
  static Future<Map<String, dynamic>> generateDailyBrief() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'generateDailyBrief',
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to generate daily brief: ${e.message}');
    }
  }

  /// Start VAD-based background listening (hands-free)
  static Future<Map<String, dynamic>> startBackgroundListening() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'startBackgroundListening',
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to start listening: ${e.message}');
    }
  }

  /// Stop VAD background listener
  static Future<Map<String, dynamic>> stopBackgroundListening() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'stopBackgroundListening',
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to stop listening: ${e.message}');
    }
  }

  /// Toggle a config flag at runtime (SIMPLIFIED_MODE, LOW_RESOURCE_MODE)
  static Future<Map<String, dynamic>> setConfigFlag(
    String key,
    bool value,
  ) async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'setConfigFlag',
        {'key': key, 'value': value},
      );
      return result ?? {};
    } on PlatformException catch (e) {
      throw Exception('Failed to set config: ${e.message}');
    }
  }

  /// Get total memory count (debug method)
  static Future<int> getMemoryCount() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'getMemoryCount',
      );
      return (result?['count'] as int?) ?? 0;
    } catch (_) {
      return 0;
    }
  }
}
