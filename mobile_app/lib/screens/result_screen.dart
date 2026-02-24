/// Result Screen — Professional display of summary + extracted events
///
/// Clean, organized display with sections for:
///  - AI Summary (or extractive summary)
///  - Extracted Events by category (meetings, tasks, medications)
///  - Memory status

import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> data;

  const ResultScreen({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final summary = data['summary'] ?? 'No summary';
    final events = (data['events'] as List?) ?? [];
    final totalInMemory = data['total_events_in_memory'] ?? 0;
    final llmUsed = data['llm_used'] ?? false;

    // Group events by type
    final meetings = events.where((e) => e['type'] == 'meeting').toList();
    final tasks = events.where((e) => e['type'] == 'task').toList();
    final medications = events.where((e) => e['type'] == 'medication').toList();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Analysis Results'),
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
            // ── Summary Section ──────────────────────────────
            _buildSectionCard(
              context,
              icon: llmUsed ? Icons.auto_awesome : Icons.summarize,
              iconColor: llmUsed ? Colors.deepPurple : Colors.blue,
              title: llmUsed ? 'AI Summary' : 'Summary',
              child: _buildSummaryContent(summary),
            ),
            const SizedBox(height: 16),

            // ── Events Overview ──────────────────────────────
            _buildQuickStats(meetings.length, tasks.length, medications.length),
            const SizedBox(height: 16),

            // ── Meetings ─────────────────────────────────────
            if (meetings.isNotEmpty) ...[
              _buildEventSection(
                context,
                icon: Icons.groups,
                color: Colors.blue,
                title: 'Appointments & Meetings',
                events: meetings,
              ),
              const SizedBox(height: 12),
            ],

            // ── Tasks ────────────────────────────────────────
            if (tasks.isNotEmpty) ...[
              _buildEventSection(
                context,
                icon: Icons.task_alt,
                color: Colors.orange,
                title: 'Tasks & Reminders',
                events: tasks,
              ),
              const SizedBox(height: 12),
            ],

            // ── Medications ──────────────────────────────────
            if (medications.isNotEmpty) ...[
              _buildEventSection(
                context,
                icon: Icons.medication,
                color: Colors.red,
                title: 'Medications',
                events: medications,
              ),
              const SizedBox(height: 12),
            ],

            // ── No events message ────────────────────────────
            if (events.isEmpty)
              Card(
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Padding(
                  padding: EdgeInsets.all(20),
                  child: Column(
                    children: [
                      Icon(Icons.search_off, size: 40, color: Colors.grey),
                      SizedBox(height: 8),
                      Text(
                        'No events found in this text',
                        style: TextStyle(color: Colors.grey, fontSize: 15),
                      ),
                    ],
                  ),
                ),
              ),

            const SizedBox(height: 16),

            // ── Memory status ────────────────────────────────
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerHighest.withAlpha(60),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.storage, size: 16, color: Colors.grey),
                  const SizedBox(width: 6),
                  Text(
                    '$totalInMemory events stored in memory',
                    style: const TextStyle(color: Colors.grey, fontSize: 13),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Summary Content ──────────────────────────────────────────
  Widget _buildSummaryContent(String summary) {
    // Split by bullet points or newlines for cleaner display
    final lines = summary
        .split(RegExp(r'\n|(?=- )'))
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty)
        .toList();

    if (lines.length <= 1) {
      return Text(
        summary.trim(),
        style: const TextStyle(fontSize: 15, height: 1.5),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: lines.map((line) {
        final cleanLine = line.startsWith('- ') ? line.substring(2) : line;
        return Padding(
          padding: const EdgeInsets.only(bottom: 6),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Padding(
                padding: EdgeInsets.only(top: 6, right: 8),
                child: Icon(Icons.circle, size: 6, color: Colors.deepPurple),
              ),
              Expanded(
                child: Text(
                  cleanLine,
                  style: const TextStyle(fontSize: 14, height: 1.4),
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }

  // ── Section Card Container ───────────────────────────────────
  Widget _buildSectionCard(
    BuildContext context, {
    required IconData icon,
    required Color iconColor,
    required String title,
    required Widget child,
  }) {
    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
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
                    color: iconColor.withAlpha(25),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(icon, color: iconColor, size: 20),
                ),
                const SizedBox(width: 10),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }

  // ── Quick Stats Row ──────────────────────────────────────────
  Widget _buildQuickStats(int meetings, int tasks, int medications) {
    return Row(
      children: [
        _buildStatChip(Icons.groups, '$meetings', 'Meetings', Colors.blue),
        const SizedBox(width: 8),
        _buildStatChip(Icons.task_alt, '$tasks', 'Tasks', Colors.orange),
        const SizedBox(width: 8),
        _buildStatChip(Icons.medication, '$medications', 'Meds', Colors.red),
      ],
    );
  }

  Widget _buildStatChip(IconData icon, String count, String label, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
        decoration: BoxDecoration(
          color: color.withAlpha(15),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: color.withAlpha(40)),
        ),
        child: Column(
          children: [
            Icon(icon, color: color, size: 22),
            const SizedBox(height: 4),
            Text(
              count,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            Text(
              label,
              style: TextStyle(fontSize: 11, color: color.withAlpha(180)),
            ),
          ],
        ),
      ),
    );
  }

  // ── Event Section ────────────────────────────────────────────
  Widget _buildEventSection(
    BuildContext context, {
    required IconData icon,
    required Color color,
    required String title,
    required List events,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Section header
        Row(
          children: [
            Icon(icon, size: 18, color: color),
            const SizedBox(width: 6),
            Text(
              title,
              style: TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        // Event cards
        ...events.map((event) => _buildEventCard(event, color)),
      ],
    );
  }

  Widget _buildEventCard(Map<String, dynamic> event, Color color) {
    final desc = event['description'] ?? 'No description';
    final rawDate = event['raw_date'];
    final parsedDate = event['parsed_date'];
    final time = event['time'];
    final parsedTime = event['parsed_time'];
    final person = event['person'];
    final isFromLlm = event['source'] == 'llm';

    // Build info chips
    List<Widget> infoChips = [];

    if (parsedDate != null || rawDate != null) {
      String dateText = parsedDate ?? rawDate;
      if (parsedTime != null || time != null) {
        dateText += ' at ${parsedTime ?? time}';
      }
      infoChips.add(_infoTag(Icons.calendar_today, dateText, Colors.blue));
    }

    if (person != null) {
      infoChips.add(_infoTag(Icons.person, person, Colors.teal));
    }

    if (isFromLlm) {
      infoChips.add(_infoTag(Icons.auto_awesome, 'AI', Colors.deepPurple));
    }

    return Card(
      margin: const EdgeInsets.only(bottom: 6),
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(10),
        side: BorderSide(color: color.withAlpha(30)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              desc,
              style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
            ),
            if (infoChips.isNotEmpty) ...[
              const SizedBox(height: 6),
              Wrap(spacing: 6, runSpacing: 4, children: infoChips),
            ],
          ],
        ),
      ),
    );
  }

  Widget _infoTag(IconData icon, String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withAlpha(20),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: color),
          const SizedBox(width: 4),
          Text(
            text,
            style: TextStyle(fontSize: 11, color: color, fontWeight: FontWeight.w500),
          ),
        ],
      ),
    );
  }
}
