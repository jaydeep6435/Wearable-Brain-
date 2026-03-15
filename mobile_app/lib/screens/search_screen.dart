/// Search Screen — Conversational Memory Assistant (Intelligence Mode)
///
/// Chat-style UI: user asks questions, assistant answers with
/// structured event cards below. Uses chatWithMemory API.

import 'package:flutter/material.dart';
import '../services/api_service.dart';

class SearchScreen extends StatefulWidget {
  const SearchScreen({super.key});

  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _ChatMessage {
  final String text;
  final bool isUser;
  final String? confidence;
  final List<dynamic> relatedEvents;
  final DateTime timestamp;

  _ChatMessage({
    required this.text,
    required this.isUser,
    this.confidence,
    this.relatedEvents = const [],
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}

class _SearchScreenState extends State<SearchScreen> {
  final _controller = TextEditingController();
  final _scrollController = ScrollController();
  final List<_ChatMessage> _messages = [];
  bool _thinking = false;

  @override
  void initState() {
    super.initState();
    // Welcome message
    _messages.add(_ChatMessage(
      text: "Hello! I'm your memory assistant. Ask me anything about your conversations, appointments, or medications.",
      isUser: false,
      confidence: "high",
    ));
  }

  Future<void> _sendMessage() async {
    final q = _controller.text.trim();
    if (q.isEmpty) return;

    setState(() {
      _messages.add(_ChatMessage(text: q, isUser: true));
      _thinking = true;
    });
    _controller.clear();
    _scrollToBottom();

    try {
      final result = await ApiService.chatWithMemory(q);
      if (!mounted) return;

      final answer = result['answer'] as String? ?? "I couldn't find anything about that in my memory.";
      final confidence = result['confidence'] as String? ?? 'low';
      final related = result['related_events'] as List<dynamic>? ?? [];

      setState(() {
        _messages.add(_ChatMessage(
          text: answer,
          isUser: false,
          confidence: confidence,
          relatedEvents: related,
        ));
        _thinking = false;
      });
      _scrollToBottom();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _messages.add(_ChatMessage(
          text: "I'm having trouble right now. Please try again.",
          isUser: false,
          confidence: "none",
        ));
        _thinking = false;
      });
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 150), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Memory Assistant',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_sweep),
            tooltip: 'Clear chat',
            onPressed: () {
              setState(() {
                _messages.clear();
                _messages.add(_ChatMessage(
                  text: "Chat cleared. Ask me anything!",
                  isUser: false,
                  confidence: "high",
                ));
              });
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // ── Chat Messages ──
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
              itemCount: _messages.length + (_thinking ? 1 : 0),
              itemBuilder: (context, index) {
                if (index == _messages.length && _thinking) {
                  return _buildThinkingBubble(cs);
                }
                return _buildMessageBubble(_messages[index], cs);
              },
            ),
          ),

          // ── Quick Suggestions (show when few messages) ──
          if (_messages.length <= 2)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: [
                    _buildSuggestionChip('When is my appointment?', cs),
                    _buildSuggestionChip('What medicine do I take?', cs),
                    _buildSuggestionChip('Who is visiting?', cs),
                    _buildSuggestionChip('What happened today?', cs),
                  ],
                ),
              ),
            ),

          // ── Input Area ──
          Container(
            padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
            decoration: BoxDecoration(
              color: cs.surface,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.05),
                  blurRadius: 10,
                  offset: const Offset(0, -2),
                ),
              ],
            ),
            child: SafeArea(
              top: false,
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      decoration: InputDecoration(
                        hintText: 'Ask about your memories...',
                        hintStyle: TextStyle(color: cs.outline),
                        filled: true,
                        fillColor: cs.surfaceContainerHighest.withValues(alpha: 0.5),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(28),
                          borderSide: BorderSide.none,
                        ),
                      ),
                      style: const TextStyle(fontSize: 17),
                      onSubmitted: (_) => _sendMessage(),
                      textInputAction: TextInputAction.send,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [cs.primary, cs.tertiary],
                      ),
                      shape: BoxShape.circle,
                    ),
                    child: IconButton(
                      icon: const Icon(Icons.send, color: Colors.white),
                      onPressed: _thinking ? null : _sendMessage,
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

  // ── Chat Bubble ──────────────────────────────────────────

  Widget _buildMessageBubble(_ChatMessage msg, ColorScheme cs) {
    if (msg.isUser) {
      return Align(
        alignment: Alignment.centerRight,
        child: Container(
          margin: const EdgeInsets.only(bottom: 8, left: 60),
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [cs.primary, cs.primary.withValues(alpha: 0.85)],
            ),
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(20),
              topRight: Radius.circular(20),
              bottomLeft: Radius.circular(20),
              bottomRight: Radius.circular(4),
            ),
          ),
          child: Text(
            msg.text,
            style: TextStyle(
              fontSize: 17,
              color: cs.onPrimary,
              height: 1.4,
            ),
          ),
        ),
      );
    }

    // Assistant message
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 8, right: 40),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Answer bubble
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: cs.surfaceContainerHighest.withValues(alpha: 0.6),
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(4),
                  topRight: Radius.circular(20),
                  bottomLeft: Radius.circular(20),
                  bottomRight: Radius.circular(20),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Confidence indicator
                  if (msg.confidence != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.auto_awesome,
                            size: 16,
                            color: _confidenceColor(msg.confidence!, cs),
                          ),
                          const SizedBox(width: 4),
                          Text(
                            'Memory Assistant',
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                              color: _confidenceColor(msg.confidence!, cs),
                            ),
                          ),
                        ],
                      ),
                    ),
                  Text(
                    msg.text,
                    style: TextStyle(
                      fontSize: 17,
                      height: 1.5,
                      color: cs.onSurface,
                    ),
                  ),
                ],
              ),
            ),

            // Event chips removed — only clean message shown

          ],
        ),
      ),
    );
  }

  // ── Event Card (below answer) ──────────────────────────

  Widget _buildEventChip(dynamic event, ColorScheme cs) {
    final map = event is Map ? event : <String, dynamic>{};
    final type = (map['type'] ?? '').toString();
    final desc = (map['description'] ?? '').toString();
    final date = (map['raw_date'] ?? '').toString();
    final time = (map['raw_time'] ?? '').toString();
    final person = (map['person'] ?? '').toString();

    final icon = type == 'meeting' ? Icons.event
        : type == 'medication' ? Icons.medication
        : type == 'task' ? Icons.task_alt
        : Icons.info_outline;

    final color = type == 'meeting' ? Colors.blue
        : type == 'medication' ? Colors.red
        : type == 'task' ? Colors.orange
        : cs.primary;

    return Container(
      margin: const EdgeInsets.only(bottom: 4, left: 4),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withValues(alpha: 0.2)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 18, color: color),
          const SizedBox(width: 8),
          Flexible(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  desc,
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: cs.onSurface),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                if (date.isNotEmpty || time.isNotEmpty || person.isNotEmpty)
                  Text(
                    [if (date.isNotEmpty) '📅 $date', if (time.isNotEmpty) '⏰ $time', if (person.isNotEmpty) '👤 $person']
                        .join('  '),
                    style: TextStyle(fontSize: 12, color: cs.outline),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Thinking Indicator ─────────────────────────────────

  Widget _buildThinkingBubble(ColorScheme cs) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 8, right: 100),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        decoration: BoxDecoration(
          color: cs.surfaceContainerHighest.withValues(alpha: 0.6),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 16, height: 16,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: cs.primary,
              ),
            ),
            const SizedBox(width: 12),
            Text(
              'Searching memories...',
              style: TextStyle(
                fontSize: 15,
                fontStyle: FontStyle.italic,
                color: cs.outline,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Suggestion Chip ────────────────────────────────────

  Widget _buildSuggestionChip(String text, ColorScheme cs) {
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: ActionChip(
        label: Text(text, style: const TextStyle(fontSize: 14)),
        avatar: Icon(Icons.lightbulb_outline, size: 18, color: cs.primary),
        onPressed: () {
          _controller.text = text;
          _sendMessage();
        },
      ),
    );
  }

  Color _confidenceColor(String confidence, ColorScheme cs) {
    switch (confidence) {
      case 'high': return Colors.green;
      case 'medium': return Colors.orange;
      case 'low': return Colors.grey;
      default: return cs.outline;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }
}
