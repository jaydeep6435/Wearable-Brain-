/// Voice Enrollment Screen — Record your voice to create a voice profile
///
/// Flow: Enter name → Read 3 sentences aloud (25s total) → Save voiceprint
/// The recorded audio is processed by VoskTranscriber to extract
/// a 128-dimensional x-vector "voice fingerprint".
///
/// Multiple sentences ensure the model captures a wide range of phonemes,
/// vowels, consonants, and vocal patterns for accurate future identification.

import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import '../services/api_service.dart';

class VoiceEnrollmentScreen extends StatefulWidget {
  const VoiceEnrollmentScreen({super.key});

  @override
  State<VoiceEnrollmentScreen> createState() => _VoiceEnrollmentScreenState();
}

class _VoiceEnrollmentScreenState extends State<VoiceEnrollmentScreen>
    with SingleTickerProviderStateMixin {
  final _nameController = TextEditingController();
  String _status = 'idle'; // idle, recording, processing, success, error
  String _message = '';
  int _recordSeconds = 0;
  Timer? _timer;

  // ── Recording Configuration ──
  static const int _totalRecordDuration = 25; // seconds
  static const int _sentencesPerSession = 3;

  // ── Sentence Bank (phonetically rich, covering diverse sounds) ──
  final List<String> _allSentences = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "The sunlight strikes raindrops in the air and they act as a prism forming a rainbow.",
    "Please call Stella and ask her to bring these things with her from the store.",
    "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for later.",
    "A large size in stockings is hard to sell during the warm summer months.",
    "The boy was there when the sun rose over the quiet mountain village.",
    "A rod is used to catch pink salmon in the cold rushing streams.",
    "The source of the huge river is the clear spring hidden deep in the forest.",
    "Kick the ball straight and follow through with your whole body.",
    "Help the woman get back to her feet after she stumbled on the path.",
    "She had your dark suit in greasy wash water all year long.",
    "Don't ask me to carry an oily rag like that to the store.",
    "His shirt was red and his shoes were dark blue and very old.",
  ];

  // Currently displayed sentences (3 picked for this session)
  List<String> _sessionSentences = [];
  int _currentSentenceIndex = 0;

  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.2).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }

  /// Pick 3 random non-repeating sentences from the bank
  void _pickSessionSentences() {
    final rng = Random();
    final shuffled = List<String>.from(_allSentences)..shuffle(rng);
    _sessionSentences = shuffled.take(_sentencesPerSession).toList();
    _currentSentenceIndex = 0;
  }

  Future<void> _startRecording() async {
    final name = _nameController.text.trim();
    if (name.isEmpty) {
      setState(() {
        _message = 'Please enter a name first';
        _status = 'error';
      });
      return;
    }

    // Check if both models are ready before recording
    try {
      final voskStatus = await ApiService.getVoskStatus();
      final asr = (voskStatus['asr'] is Map)
          ? Map<String, dynamic>.from(voskStatus['asr'])
          : <String, dynamic>{};
      final spk = (voskStatus['spk'] is Map)
          ? Map<String, dynamic>.from(voskStatus['spk'])
          : <String, dynamic>{};

      final asrReady = (voskStatus['asrReady'] == true) || (asr['ready'] == true);
      final spkReady = (voskStatus['spkReady'] == true) || (spk['ready'] == true);
      
      if (!asrReady) {
        setState(() {
          _message = 'Speech Model not downloaded yet.\nPlease download it from Settings → Speech & Intelligence.';
          _status = 'error';
        });
        return;
      }
      if (!spkReady) {
        setState(() {
          _message = 'Speaker Identification Model not downloaded yet.\nPlease download it from Settings → Speech & Intelligence.';
          _status = 'error';
        });
        return;
      }
    } catch (_) {
      // If we can't check, proceed anyway — backend will catch it
    }

    _pickSessionSentences();

    setState(() {
      _status = 'recording';
      _recordSeconds = 0;
      _currentSentenceIndex = 0;
      _message = 'Read each sentence out loud clearly';
    });
    _pulseController.repeat(reverse: true);

    try {
      await ApiService.startRecording();
    } catch (e) {
      setState(() {
        _status = 'error';
        _message = 'Could not start recording: $e';
      });
      _pulseController.stop();
      return;
    }

    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {
        _recordSeconds++;
        // Advance sentence every ~8 seconds
        final nextIdx = (_recordSeconds / 8).floor();
        if (nextIdx < _sentencesPerSession && nextIdx != _currentSentenceIndex) {
          _currentSentenceIndex = nextIdx;
        }
      });
      if (_recordSeconds >= _totalRecordDuration) {
        timer.cancel();
        _stopAndEnroll();
      }
    });
  }

  Future<void> _stopAndEnroll() async {
    _pulseController.stop();
    _pulseController.reset();

    setState(() {
      _status = 'processing';
      _message = 'Analyzing your voice patterns...';
    });

    try {
      final stopResult = await ApiService.stopRecording();
      final audioPath = stopResult['audio_path']?.toString() ?? '';

      if (audioPath.isEmpty) {
        setState(() {
          _status = 'error';
          _message = 'No audio was captured. Please try again.';
        });
        return;
      }

      final enrollResult = await ApiService.enrollVoice(
        _nameController.text.trim(),
        audioPath,
      );

      if (enrollResult['status'] == 'ok') {
        setState(() {
          _status = 'success';
          _message = enrollResult['message'] ?? 'Voice enrolled successfully!';
        });
      } else {
        setState(() {
          _status = 'error';
          _message = enrollResult['message'] ??
              'Could not save voice profile. Please try again.';
        });
      }
    } catch (e) {
      setState(() {
        _status = 'error';
        _message = 'Error: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final remaining = _totalRecordDuration - _recordSeconds;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Enroll Voice'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Header
              const Icon(Icons.record_voice_over, size: 48, color: Colors.deepPurple),
              const SizedBox(height: 12),
              Text(
                'Voice Enrollment',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: cs.onSurface,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              Text(
                'Read 3 sentences aloud (~25 seconds) so I can\nlearn your unique voice for future recognition.',
                style: TextStyle(fontSize: 15, color: cs.onSurface.withValues(alpha: 0.6)),
                textAlign: TextAlign.center,
              ),

              const SizedBox(height: 28),

              // Name input
              TextField(
                controller: _nameController,
                enabled: _status == 'idle' || _status == 'error',
                textCapitalization: TextCapitalization.words,
                style: const TextStyle(fontSize: 18),
                decoration: InputDecoration(
                  labelText: 'Your Name',
                  hintText: 'e.g., David, Mom, Doctor',
                  prefixIcon: const Icon(Icons.person),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 20, vertical: 18,
                  ),
                ),
              ),

              const SizedBox(height: 28),

              // ── Recording UI ──
              if (_status == 'recording') ...[
                // Sentence Progress Dots
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: List.generate(_sentencesPerSession, (i) {
                    final isActive = i == _currentSentenceIndex;
                    final isDone = i < _currentSentenceIndex;
                    return Container(
                      width: isDone ? 12 : (isActive ? 14 : 10),
                      height: isDone ? 12 : (isActive ? 14 : 10),
                      margin: const EdgeInsets.symmetric(horizontal: 6),
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: isDone
                            ? Colors.green
                            : isActive
                                ? cs.primary
                                : cs.outlineVariant,
                        border: isActive
                            ? Border.all(color: cs.primary, width: 2)
                            : null,
                      ),
                      child: isDone
                          ? const Icon(Icons.check, size: 8, color: Colors.white)
                          : null,
                    );
                  }),
                ),
                const SizedBox(height: 6),
                Text(
                  'Sentence ${_currentSentenceIndex + 1} of $_sentencesPerSession',
                  style: TextStyle(
                    fontSize: 13,
                    color: cs.onSurface.withValues(alpha: 0.5),
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),

                // Reading Prompt Card
                AnimatedSwitcher(
                  duration: const Duration(milliseconds: 400),
                  child: Card(
                    key: ValueKey(_currentSentenceIndex),
                    elevation: 0,
                    color: cs.primaryContainer.withValues(alpha: 0.3),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                      side: BorderSide(color: cs.primaryContainer),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(24),
                      child: Column(
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.volume_up, size: 18, color: cs.primary),
                              const SizedBox(width: 8),
                              Text(
                                'READ OUT LOUD',
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.bold,
                                  color: cs.primary,
                                  letterSpacing: 1.2,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 14),
                          Text(
                            '"${_sessionSentences.isNotEmpty ? _sessionSentences[_currentSentenceIndex] : ""}"',
                            style: TextStyle(
                              fontSize: 19,
                              fontWeight: FontWeight.w500,
                              color: cs.onSurface,
                              height: 1.5,
                              fontStyle: FontStyle.italic,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Timer Pulse
                Center(
                  child: ScaleTransition(
                    scale: _pulseAnimation,
                    child: Container(
                      width: 90,
                      height: 90,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.red.withValues(alpha: 0.15),
                        border: Border.all(color: Colors.red, width: 3),
                      ),
                      child: Center(
                        child: Text(
                          '$remaining',
                          style: const TextStyle(
                            fontSize: 32,
                            fontWeight: FontWeight.bold,
                            color: Colors.red,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: _recordSeconds / _totalRecordDuration.toDouble(),
                    backgroundColor: cs.surfaceContainerHighest,
                    color: Colors.red,
                    minHeight: 6,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  '${_recordSeconds}s / ${_totalRecordDuration}s',
                  style: TextStyle(
                    fontSize: 13,
                    color: cs.onSurface.withValues(alpha: 0.5),
                  ),
                  textAlign: TextAlign.center,
                ),
              ],

              if (_status == 'processing')
                Column(
                  children: [
                    const SizedBox(height: 40),
                    const SizedBox(
                      width: 60, height: 60,
                      child: CircularProgressIndicator(strokeWidth: 4),
                    ),
                    const SizedBox(height: 16),
                  ],
                ),

              if (_status == 'success')
                Column(
                  children: [
                    const SizedBox(height: 40),
                    const Icon(Icons.check_circle, color: Colors.green, size: 64),
                    const SizedBox(height: 16),
                  ],
                ),

              // Status message
              if (_message.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  child: Text(
                    _message,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                      color: _status == 'error'
                          ? Colors.red.shade700
                          : _status == 'success'
                              ? Colors.green.shade700
                              : cs.onSurface.withValues(alpha: 0.7),
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),

              const SizedBox(height: 32),

              // Action buttons
              if (_status == 'idle' || _status == 'error')
                SizedBox(
                  height: 56,
                  child: FilledButton.icon(
                    onPressed: _startRecording,
                    icon: const Icon(Icons.mic, size: 24),
                    label: const Text(
                      'Start Recording',
                      style: TextStyle(fontSize: 18),
                    ),
                    style: FilledButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                  ),
                ),

              if (_status == 'success')
                SizedBox(
                  height: 56,
                  child: FilledButton.icon(
                    onPressed: () => Navigator.pop(context, true),
                    icon: const Icon(Icons.check, size: 24),
                    label: const Text(
                      'Done',
                      style: TextStyle(fontSize: 18),
                    ),
                    style: FilledButton.styleFrom(
                      backgroundColor: Colors.green,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                  ),
                ),

              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _timer?.cancel();
    _pulseController.dispose();
    _nameController.dispose();
    super.dispose();
  }
}
