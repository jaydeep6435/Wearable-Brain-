/// Home Screen — Text input + Process button
///
/// User enters conversation text and sends it to the Flask API.
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

  @override
  void initState() {
    super.initState();
    _checkServer();
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

  Future<void> _processText() async {
    if (_textController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter some text')),
      );
      return;
    }

    setState(() => _isLoading = true);

    try {
      final result = await ApiService.processText(_textController.text);
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
            const SizedBox(height: 16),

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
                    : const Icon(Icons.auto_awesome),
                label: Text(_isLoading ? 'Processing...' : 'Process Text'),
                style: ElevatedButton.styleFrom(
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
