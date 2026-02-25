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
          'Please wait...',
          style: TextStyle(
            fontSize: 18,
            color: cs.onSurface.withValues(alpha: 0.6),
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
            Text(
              summary.toString(),
              style: TextStyle(
                fontSize: 16,
                height: 1.5,
                color: cs.onSurface.withValues(alpha: 0.8),
              ),
              maxLines: 4,
              overflow: TextOverflow.ellipsis,
            ),
          ],
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
