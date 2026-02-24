/// Home Screen — Text input + Process button + LLM toggle
///
/// User enters conversation text and sends it to the Flask API.
/// Toggle switch enables AI-powered (LLM) processing via Ollama.
/// Results (summary + events) are shown on the Results screen.

import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _textController = TextEditingController();
  bool _isLoading = false;
  bool _serverOnline = false;
  bool _useLlm = false;
  String _llmStatus = 'checking...';

  @override
  void initState() {
    super.initState();
    _checkServer();
    _checkLlm();
    // Pre-fill with sample text for demo
    _textController.text =
        'I have a doctor appointment tomorrow at 10 AM. '
        'Don\'t forget to take your medicine after breakfast. '
        'We need to call the pharmacy to refill the prescription. '
        'Your son David is visiting this weekend.';
  }

  Future<void> _checkServer() async {
    final online = await ApiService.checkServer();
    setState(() => _serverOnline = online);
  }

  Future<void> _checkLlm() async {
    final result = await ApiService.checkLlmStatus();
    setState(() {
      _llmStatus = result['status'] ?? 'error';
    });
  }

  Future<void> _processText() async {
    if (_textController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter some text')),
      );
      return;
    }

    setState(() => _isLoading = true);

    try {
      final result = await ApiService.processText(
        _textController.text,
        useLlm: _useLlm,
      );
      if (!mounted) return;
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(data: result),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Memory Assistant'),
        centerTitle: true,
        actions: [
          // Server status indicator
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Icon(
              Icons.circle,
              size: 14,
              color: _serverOnline ? Colors.greenAccent : Colors.redAccent,
            ),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Header
            const Text(
              '📝 Enter Conversation Text',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'Paste or type a conversation. The AI will extract events, '
              'create a summary, and store it in memory.',
              style: TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 12),

            // 🤖 LLM Toggle
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              decoration: BoxDecoration(
                color: _useLlm
                    ? Colors.deepPurple.withAlpha(30)
                    : Theme.of(context).colorScheme.surfaceContainerHighest.withAlpha(80),
                borderRadius: BorderRadius.circular(12),
                border: _useLlm
                    ? Border.all(color: Colors.deepPurple.withAlpha(100))
                    : null,
              ),
              child: Row(
                children: [
                  Icon(
                    _useLlm ? Icons.auto_awesome : Icons.rule,
                    size: 20,
                    color: _useLlm ? Colors.deepPurple : Colors.grey,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _useLlm ? '🤖 AI Mode (LLM)' : '⚡ Fast Mode (Rule-based)',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 13,
                            color: _useLlm ? Colors.deepPurple : null,
                          ),
                        ),
                        Text(
                          _useLlm
                              ? 'Smarter results • Ollama: $_llmStatus'
                              : 'Quick processing • No LLM needed',
                          style: const TextStyle(fontSize: 11, color: Colors.grey),
                        ),
                      ],
                    ),
                  ),
                  Switch(
                    value: _useLlm,
                    onChanged: (val) {
                      setState(() => _useLlm = val);
                      if (val) _checkLlm();
                    },
                    activeThumbColor: Colors.deepPurple,
                  ),
                ],
              ),
            ),

            // LLM offline warning
            if (_useLlm && _llmStatus != 'online')
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  '⚠️ Ollama is $_llmStatus. Run: ollama serve',
                  style: const TextStyle(color: Colors.orange, fontSize: 11),
                  textAlign: TextAlign.center,
                ),
              ),

            const SizedBox(height: 12),

            // Text input
            Expanded(
              child: TextField(
                controller: _textController,
                maxLines: null,
                expands: true,
                textAlignVertical: TextAlignVertical.top,
                decoration: InputDecoration(
                  hintText: 'Type or paste conversation text here...',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  filled: true,
                  fillColor: Theme.of(context).colorScheme.surfaceContainerHighest.withAlpha(80),
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Process button
            SizedBox(
              height: 50,
              child: ElevatedButton.icon(
                onPressed: _isLoading ? null : _processText,
                icon: _isLoading
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : Icon(_useLlm ? Icons.auto_awesome : Icons.flash_on),
                label: Text(_isLoading
                    ? (_useLlm ? 'AI Processing...' : 'Processing...')
                    : (_useLlm ? '🤖 Process with AI' : '⚡ Process Text')),
                style: ElevatedButton.styleFrom(
                  backgroundColor: _useLlm ? Colors.deepPurple : null,
                  foregroundColor: _useLlm ? Colors.white : null,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ),

            if (!_serverOnline) ...[
              const SizedBox(height: 8),
              const Text(
                '⚠️ Flask server not reachable. Run: python api.py',
                style: TextStyle(color: Colors.red, fontSize: 12),
                textAlign: TextAlign.center,
              ),
            ],
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }
}
