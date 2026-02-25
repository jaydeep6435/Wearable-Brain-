/// Search Screen — Query Memories with Ranked Results
///
/// Natural language search with importance badges and speaker filtering.
/// Replaces the old Query screen with a richer search experience.

import 'package:flutter/material.dart';
import '../services/api_service.dart';

class SearchScreen extends StatefulWidget {
  const SearchScreen({super.key});

  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  final _controller = TextEditingController();
  List<dynamic> _results = [];
  Map<String, dynamic>? _answer;
  bool _searching = false;
  final _recentQueries = <String>[];

  Future<void> _search() async {
    final q = _controller.text.trim();
    if (q.isEmpty) return;

    setState(() {
      _searching = true;
      _answer = null;
      _results = [];
    });

    try {
      final result = await ApiService.queryMemory(q);
      if (!mounted) return;
      setState(() {
        _answer = result;
        _results = (result['results'] as List<dynamic>?) ?? [];
        _searching = false;
        if (!_recentQueries.contains(q)) {
          _recentQueries.insert(0, q);
          if (_recentQueries.length > 5) _recentQueries.removeLast();
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _searching = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Search failed: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Search Memories',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
        ),
        centerTitle: true,
      ),
      body: Column(
        children: [
          // ── Search Bar ──
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: SearchBar(
              controller: _controller,
              hintText: 'Ask anything... (e.g. "When is my appointment?")',
              hintStyle: WidgetStatePropertyAll(
                TextStyle(fontSize: 16, color: cs.outline),
              ),
              textStyle: const WidgetStatePropertyAll(
                TextStyle(fontSize: 17),
              ),
              leading: const Padding(
                padding: EdgeInsets.only(left: 8),
                child: Icon(Icons.search),
              ),
              trailing: [
                if (_controller.text.isNotEmpty)
                  IconButton(
                    icon: const Icon(Icons.clear),
                    onPressed: () {
                      _controller.clear();
                      setState(() {
                        _answer = null;
                        _results = [];
                      });
                    },
                  ),
                IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: _search,
                ),
              ],
              onSubmitted: (_) => _search(),
              onChanged: (_) => setState(() {}),
            ),
          ),

          // ── Quick Suggestions ──
          if (_answer == null && _recentQueries.isEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Wrap(
                spacing: 8,
                children: [
                  _buildSuggestionChip('When is my appointment?'),
                  _buildSuggestionChip('What medicine?'),
                  _buildSuggestionChip('Who is visiting?'),
                  _buildSuggestionChip('What did the doctor say?'),
                ],
              ),
            ),

          // ── Recent Queries ──
          if (_answer == null && _recentQueries.isNotEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Wrap(
                spacing: 8,
                children: _recentQueries
                    .map((q) => ActionChip(
                          label: Text(q, style: const TextStyle(fontSize: 14)),
                          onPressed: () {
                            _controller.text = q;
                            _search();
                          },
                        ))
                    .toList(),
              ),
            ),

          const SizedBox(height: 8),

          // ── Results ──
          Expanded(
            child: _searching
                ? const Center(child: CircularProgressIndicator())
                : _answer != null
                    ? _buildResultArea(cs)
                    : Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.manage_search,
                                size: 64, color: cs.outline),
                            const SizedBox(height: 12),
                            Text(
                              'Ask a question about your memories',
                              style: TextStyle(
                                fontSize: 17,
                                color: cs.outline,
                              ),
                            ),
                          ],
                        ),
                      ),
          ),
        ],
      ),
    );
  }

  Widget _buildSuggestionChip(String text) {
    return ActionChip(
      label: Text(text, style: const TextStyle(fontSize: 14)),
      avatar: const Icon(Icons.lightbulb_outline, size: 18),
      onPressed: () {
        _controller.text = text;
        _search();
      },
    );
  }

  Widget _buildResultArea(ColorScheme cs) {
    final answerText = _answer?['answer'] as String? ?? '';

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Main answer
        if (answerText.isNotEmpty)
          Card(
            elevation: 0,
            color: cs.primaryContainer.withValues(alpha: 0.3),
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.auto_awesome, color: cs.primary, size: 24),
                      const SizedBox(width: 8),
                      Text(
                        'Answer',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                          color: cs.primary,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Text(
                    answerText,
                    style: TextStyle(
                      fontSize: 18,
                      height: 1.5,
                      color: cs.onSurface,
                    ),
                  ),
                ],
              ),
            ),
          ),

        // Result cards
        if (_results.isNotEmpty) ...[
          const SizedBox(height: 16),
          Text(
            'Related Memories (${_results.length})',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: cs.outline,
            ),
          ),
          const SizedBox(height: 8),
          ..._results.map((r) => _buildResultCard(r, cs)),
        ],
      ],
    );
  }

  Widget _buildResultCard(dynamic result, ColorScheme cs) {
    // Handle both formats:
    //   Python semantic search: {"document": {...}, "score": 0.8}
    //   Kotlin bridge:         {"type": "...", "description": "...", ...}
    final Map doc;
    final double score;
    if (result is Map && result.containsKey('document')) {
      doc = (result['document'] as Map?) ?? {};
      score = (result['blended_score'] ?? result['score'] ?? 0.0).toDouble();
    } else if (result is Map) {
      doc = result;
      score = 0.0;
    } else {
      doc = {};
      score = 0.0;
    }
    final importance = doc['importance_score'] ?? doc['importance'] ?? 0;
    final text = doc['text'] ?? doc['description'] ?? '';
    final type = doc['type'] ?? '';
    final person = doc['person'];
    final rawTime = doc['raw_time'];

    return Card(
      elevation: 0,
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: ListTile(
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        leading: Icon(
          type == 'meeting' ? Icons.event :
          type == 'medication' ? Icons.medication :
          type == 'task' ? Icons.task_alt :
          Icons.chat_bubble_outline,
          color: cs.primary,
        ),
        title: Text(
          text.toString(),
          style: const TextStyle(fontSize: 16),
          maxLines: 3,
          overflow: TextOverflow.ellipsis,
        ),
        subtitle: Row(
          children: [
            if (type.toString().isNotEmpty)
              _buildScoreBadge(type.toString().toUpperCase(), cs.primary),
            if (score > 0) ...[
              const SizedBox(width: 6),
              _buildScoreBadge(
                '${(score * 100).toStringAsFixed(0)}%',
                cs.secondary,
              ),
            ],
            if (person != null && person.toString().isNotEmpty) ...[
              const SizedBox(width: 6),
              _buildScoreBadge(person.toString(), Colors.teal),
            ],
            if (rawTime != null && rawTime.toString().isNotEmpty) ...[
              const SizedBox(width: 6),
              _buildScoreBadge(rawTime.toString(), Colors.orange),
            ],
            if (importance is int && importance >= 5) ...[
              const SizedBox(width: 6),
              _buildScoreBadge('Important', Colors.red),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildScoreBadge(String label, Color color) {
    return Container(
      margin: const EdgeInsets.only(top: 4),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.w600,
          color: color,
        ),
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}
