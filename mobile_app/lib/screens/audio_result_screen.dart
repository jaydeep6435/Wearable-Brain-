/// Audio Result Screen — Shows transcription + summary + events
///
/// After audio is processed, displays the full results including
/// the transcribed text, AI summary, and extracted events.

import 'package:flutter/material.dart';
import 'result_screen.dart';

class AudioResultScreen extends StatelessWidget {
  final Map<String, dynamic> data;

  const AudioResultScreen({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final transcription = data['transcription'] ?? 'No transcription';
    final llmUsed = data['llm_used'] ?? false;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Audio Results'),
        centerTitle: true,
        actions: [
          if (llmUsed)
            const Padding(
              padding: EdgeInsets.only(right: 12),
              child: Chip(
                label: Text('🤖 AI', style: TextStyle(fontSize: 11)),
                backgroundColor: Color(0x30673AB7),
                side: BorderSide.none,
                padding: EdgeInsets.zero,
                visualDensity: VisualDensity.compact,
              ),
            ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Transcription Section ───────────────────────
            Card(
              elevation: 1,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(6),
                          decoration: BoxDecoration(
                            color: Colors.teal.withAlpha(25),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Icon(Icons.mic, color: Colors.teal, size: 20),
                        ),
                        const SizedBox(width: 10),
                        const Text(
                          'Transcription',
                          style: TextStyle(
                            fontSize: 17,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.grey.withAlpha(15),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: Colors.grey.withAlpha(30)),
                      ),
                      child: Text(
                        transcription,
                        style: const TextStyle(
                          fontSize: 14,
                          height: 1.5,
                          fontStyle: FontStyle.italic,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // ── Reuse ResultScreen content ──────────────────
            ResultScreen(data: data, embedded: true),
          ],
        ),
      ),
    );
  }
}
