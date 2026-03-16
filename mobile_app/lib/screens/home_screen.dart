/// Record Screen — Simplified Alzheimer-Friendly UI
///
/// Two primary modes:
///   1. Wearable Listening — Start/Stop button, auto-processing
///   2. Text Input — Simple text area with Send button
///
/// States: IDLE → LISTENING → PROCESSING → READY
///
/// No developer buttons. No Flask. No technical labels.
/// Large buttons, calm colors, minimal text.

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/api_service.dart';

/// App states for the record screen
enum AppState { idle, listening, processing, ready }

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  AppState _state = AppState.idle;
  String _statusMessage = 'Ready to listen';
  Map<String, dynamic>? _lastResult;
  final TextEditingController _textController = TextEditingController();
  bool _showTextMode = false;
  String _currentSource = 'microphone';
  String _btDeviceName = '';
  Map<String, dynamic> _sherpaStatus = {};
  Timer? _sherpaTimer;

  // Pulse animation for listening state
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.15).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
    _loadInitialState();
    _startStatusTimer();
  }

  void _startStatusTimer() {
    // Poll Sherpa status every 5 seconds on home screen
    _sherpaTimer = Timer.periodic(const Duration(seconds: 5), (timer) async {
      try {
        final status = await ApiService.getSherpaStatus();
        if (!mounted) return;
        setState(() => _sherpaStatus = status);
        
        // If ready, stop fast polling
        if (status['ready'] == true) {
          timer.cancel();
          // Periodically check every 30s just in case
          _sherpaTimer = Timer.periodic(const Duration(seconds: 30), (t) async {
            final s = await ApiService.getSherpaStatus();
            if (!mounted) return;
            setState(() => _sherpaStatus = s);
          });
        }
      } catch (_) {}
    });
  }

  Future<void> _loadInitialState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final saved = prefs.getString('audio_source') ?? 'microphone';
      final info = await ApiService.getAudioSourceInfo();
      final status = await ApiService.getSherpaStatus();
      if (!mounted) return;
      setState(() {
        _currentSource = saved;
        _btDeviceName = (info['device_name'] ?? '').toString();
        _sherpaStatus = status;
      });
      // Ensure engine is set to saved source
      await ApiService.setAudioSource(saved);
    } catch (_) {}
  }

  // ── Start Listening ──────────────────────────────────────

  Future<void> _startListening() async {
    setState(() {
      _state = AppState.listening;
      _statusMessage = 'Listening...';
      _lastResult = null;
    });
    _pulseController.repeat(reverse: true);

    try {
      await ApiService.startBackgroundListening();
    } catch (e) {
      // Fallback to session recording if background listening unavailable
      try {
        await ApiService.startRecording();
      } catch (_) {}
    }
  }

  // ── Stop Listening ───────────────────────────────────────

  Future<void> _stopListening() async {
    _pulseController.stop();
    _pulseController.reset();

    setState(() {
      _state = AppState.processing;
      _statusMessage = 'Processing your conversation...';
    });

    try {
      // Stop and auto-process
      final result = await ApiService.stopRecording();
      if (!mounted) return;
      setState(() {
        _state = AppState.ready;
        _lastResult = result;
        _statusMessage = 'Done! Memory saved.';
      });

      // Auto-reset after showing result
      Future.delayed(const Duration(seconds: 8), () {
        if (mounted && _state == AppState.ready) {
          setState(() {
            _state = AppState.idle;
            _statusMessage = 'Ready to listen';
          });
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _state = AppState.idle;
        _statusMessage = 'Could not process. Try again.';
      });
    }
  }

  // ── Text Mode ────────────────────────────────────────────

  Future<void> _sendText() async {
    final text = _textController.text.trim();
    if (text.isEmpty) return;

    setState(() {
      _state = AppState.processing;
      _statusMessage = 'Understanding your words...';
    });

    try {
      final result = await ApiService.processText(text);
      if (!mounted) return;
      _textController.clear();
      setState(() {
        _state = AppState.ready;
        _lastResult = result;
        _statusMessage = 'Got it! Memory saved.';
      });

      Future.delayed(const Duration(seconds: 8), () {
        if (mounted && _state == AppState.ready) {
          setState(() {
            _state = AppState.idle;
            _statusMessage = 'Ready to listen';
          });
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _state = AppState.idle;
        _statusMessage = 'Could not understand. Try again.';
      });
    }
  }

  // ── Build ────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Record',
          style: TextStyle(fontSize: 24, fontWeight: FontWeight.w600),
        ),
        centerTitle: true,
        actions: [
          // Toggle text mode
          IconButton(
            icon: Icon(
              _showTextMode ? Icons.mic : Icons.keyboard,
              size: 28,
            ),
            tooltip: _showTextMode ? 'Voice mode' : 'Type instead',
            onPressed: () {
              setState(() => _showTextMode = !_showTextMode);
            },
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            children: [
              const Spacer(flex: 1),

              // ── Source Indicator (PART 5) ──
              _buildSourceIndicator(cs),

              const SizedBox(height: 8),

              // ── Status Message ──
              Text(
                _statusMessage,
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w500,
                  color: _state == AppState.ready
                      ? Colors.green.shade700
                      : cs.onSurface.withValues(alpha: 0.7),
                ),
                textAlign: TextAlign.center,
              ),

              const SizedBox(height: 32),

              // ── Main Action Area ──
              if (_showTextMode)
                _buildTextMode(cs)
              else
                _buildListeningMode(cs),

              const SizedBox(height: 24),

              // ── Result Preview ──
              if (_lastResult != null && _state == AppState.ready)
                _buildResultPreview(cs),

              _buildSherpaModelProgress(cs),

              const Spacer(flex: 2),
            ],
          ),
        ),
      ),
    );
  }

  // ── Listening Mode ───────────────────────────────────────

  Widget _buildListeningMode(ColorScheme cs) {
    switch (_state) {
      case AppState.idle:
      case AppState.ready:
        return _buildStartButton();

      case AppState.listening:
        return _buildStopButton();

      case AppState.processing:
        return _buildProcessingIndicator(cs);
    }
  }

  Widget _buildStartButton() {
    return SizedBox(
      width: 180,
      height: 180,
      child: ElevatedButton(
        onPressed: _startListening,
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF2E7D32), // Green
          foregroundColor: Colors.white,
          shape: const CircleBorder(),
          elevation: 6,
          shadowColor: Colors.green.withValues(alpha: 0.4),
        ),
        child: const Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.mic, size: 56),
            SizedBox(height: 8),
            Text(
              'Start',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStopButton() {
    return ScaleTransition(
      scale: _pulseAnimation,
      child: SizedBox(
        width: 180,
        height: 180,
        child: ElevatedButton(
          onPressed: _stopListening,
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFFC62828), // Red
            foregroundColor: Colors.white,
            shape: const CircleBorder(),
            elevation: 8,
            shadowColor: Colors.red.withValues(alpha: 0.4),
          ),
          child: const Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.stop, size: 56),
              SizedBox(height: 8),
              Text(
                'Stop',
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildProcessingIndicator(ColorScheme cs) {
    return Column(
      children: [
        SizedBox(
          width: 120,
          height: 120,
          child: CircularProgressIndicator(
            strokeWidth: 6,
            color: cs.primary,
          ),
        ),
        const SizedBox(height: 24),
        Text(
          'Processing audio...',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w600,
            color: cs.onSurface,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          '🎙️ Transcribing speech\n🔍 Extracting events\n💾 Saving to memory',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 14,
            color: cs.onSurface.withValues(alpha: 0.5),
            height: 1.6,
          ),
        ),
      ],
    );
  }

  // ── Text Mode ────────────────────────────────────────────

  Widget _buildTextMode(ColorScheme cs) {
    final isProcessing = _state == AppState.processing;

    return Column(
      children: [
        TextField(
          controller: _textController,
          maxLines: 4,
          enabled: !isProcessing,
          style: const TextStyle(fontSize: 17),
          decoration: InputDecoration(
            hintText: 'Type what was said...',
            hintStyle: TextStyle(fontSize: 17, color: cs.outline),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            contentPadding: const EdgeInsets.all(20),
          ),
        ),
        const SizedBox(height: 16),
        SizedBox(
          width: double.infinity,
          height: 56,
          child: FilledButton.icon(
            onPressed: isProcessing ? null : _sendText,
            icon: isProcessing
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white,
                    ),
                  )
                : const Icon(Icons.send, size: 24),
            label: Text(
              isProcessing ? 'Understanding...' : 'Save Memory',
              style: const TextStyle(fontSize: 18),
            ),
            style: FilledButton.styleFrom(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
            ),
          ),
        ),
      ],
    );
  }

  // ── Result Preview ───────────────────────────────────────

  Widget _buildResultPreview(ColorScheme cs) {
    final summary = _lastResult?['summary'] ?? '';
    final eventCount = (_lastResult?['events_saved'] ?? 0);
    final diarizedText = _lastResult?['diarized_text'] ?? '';
    final fullTranscript = _lastResult?['full_transcript'] ?? '';

    if (summary.toString().isEmpty) return const SizedBox.shrink();

    return Card(
      elevation: 0,
      color: Colors.green.withValues(alpha: 0.08),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.check_circle, color: Colors.green, size: 24),
                const SizedBox(width: 8),
                Text(
                  'Saved!',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: Colors.green.shade700,
                  ),
                ),
                const Spacer(),
                if (eventCount > 0)
                  Text(
                    '$eventCount event${eventCount > 1 ? 's' : ''} found',
                    style: TextStyle(
                      fontSize: 14,
                      color: cs.outline,
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 12),
            // ── Show diarized text with speaker labels if available ──
            if (diarizedText.toString().isNotEmpty)
              ...[
                ..._buildDiarizedLines(diarizedText.toString(), cs),
                if (fullTranscript.toString().trim().isNotEmpty &&
                    fullTranscript.toString().trim().length > diarizedText.toString().trim().length + 20) ...[
                  const SizedBox(height: 8),
                  Text(
                    fullTranscript.toString(),
                    style: TextStyle(
                      fontSize: 14,
                      height: 1.4,
                      color: cs.onSurface.withValues(alpha: 0.72),
                    ),
                    maxLines: 4,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ]
            else
              Text(
                summary.toString(),
                style: TextStyle(
                  fontSize: 16,
                  height: 1.5,
                  color: cs.onSurface.withValues(alpha: 0.8),
                ),
                maxLines: 6,
                overflow: TextOverflow.ellipsis,
              ),
          ],
        ),
      ),
    );
  }

  /// Parse diarized text lines ("Speaker 1: text\nSpeaker 2: text")
  /// and render with color-coded speaker badges.
  List<Widget> _buildDiarizedLines(String diarizedText, ColorScheme cs) {
    final lines = diarizedText.split('\n').where((l) => l.trim().isNotEmpty).toList();
    final speakerColors = [
      Colors.blue,
      Colors.deepOrange,
      Colors.teal,
      Colors.purple,
      Colors.brown,
    ];
    final Map<String, int> speakerColorMap = {};
    int nextColor = 0;

    return lines.take(8).map((line) {
      final colonIdx = line.indexOf(':');
      if (colonIdx > 0 && colonIdx < 30) {
        final speaker = line.substring(0, colonIdx).trim();
        final text = line.substring(colonIdx + 1).trim();

        // Assign consistent color per speaker
        if (!speakerColorMap.containsKey(speaker)) {
          speakerColorMap[speaker] = nextColor++;
        }
        final color = speakerColors[speakerColorMap[speaker]! % speakerColors.length];

        return Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  speaker,
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: color,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  text,
                  style: TextStyle(
                    fontSize: 15,
                    height: 1.4,
                    color: cs.onSurface.withValues(alpha: 0.85),
                  ),
                ),
              ),
            ],
          ),
        );
      } else {
        return Padding(
          padding: const EdgeInsets.only(bottom: 4),
          child: Text(
            line,
            style: TextStyle(fontSize: 15, color: cs.onSurface.withValues(alpha: 0.8)),
          ),
        );
      }
    }).toList();
  }

  // ── Source Indicator ──────────────────────────────────

  Widget _buildSourceIndicator(ColorScheme cs) {
    final icon = _currentSource == 'bluetooth'
        ? Icons.bluetooth_audio
        : _currentSource == 'file'
            ? Icons.insert_drive_file
            : Icons.mic;
    final label = _currentSource == 'bluetooth'
        ? 'Bluetooth${_btDeviceName.isNotEmpty ? ': $_btDeviceName' : ''}'
        : _currentSource == 'file'
            ? 'File Input'
            : 'Microphone';
    final color = _currentSource == 'bluetooth'
        ? Colors.blue
        : _currentSource == 'file'
            ? Colors.orange
            : Colors.green;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 18, color: color),
          const SizedBox(width: 8),
          Text(
            'Source: $label',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSherpaModelProgress(ColorScheme cs) {
    final asr = (_sherpaStatus['asr'] is Map)
        ? Map<String, dynamic>.from(_sherpaStatus['asr'])
        : <String, dynamic>{};

    if (asr['ready'] == true || _sherpaStatus.isEmpty) {
      return const SizedBox.shrink();
    }

    final isInitializing = asr['initializing'] == true;
    final progress = (asr['progress'] as int? ?? 0);
    final downloadedMb = (asr['downloaded_mb'] as int? ?? 0);
    final totalMb = (asr['total_mb'] as int? ?? 0);
    final errorText = (asr['error'] ?? '').toString();
    
    // Only show if it's actually downloading or has an error
    if (!isInitializing && progress == 0 && errorText.isEmpty) {
      return const SizedBox.shrink();
    }

    return Padding(
      padding: const EdgeInsets.only(top: 16),
      child: Card(
        elevation: 0,
        color: cs.primaryContainer.withValues(alpha: 0.4),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              Row(
                children: [
                  Icon(Icons.download, color: cs.primary, size: 20),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      isInitializing 
                        ? (totalMb > 0
                          ? 'Downloading Speech Model: $downloadedMb / $totalMb MB'
                          : 'Downloading Speech Model: $progress%')
                        : errorText.isNotEmpty
                          ? 'Model Error: $errorText'
                            : 'Preparing Speech Engine...',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: cs.onPrimaryContainer,
                      ),
                    ),
                  ),
                ],
              ),
              if (isInitializing && progress > 0) ...[
                const SizedBox(height: 12),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: (progress.clamp(0, 100)) / 100.0,
                    backgroundColor: cs.primaryContainer,
                    color: cs.primary,
                    minHeight: 6,
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _textController.dispose();
    super.dispose();
  }
}
