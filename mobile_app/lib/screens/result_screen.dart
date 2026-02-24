/// Result Screen — Shows summary + extracted events
///
/// Displays the API response after processing text.

import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> data;

  const ResultScreen({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final summary = data['summary'] ?? 'No summary';
    final events = (data['events'] as List?) ?? [];
    final totalInMemory = data['total_events_in_memory'] ?? 0;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Results'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Summary card
            Card(
              elevation: 2,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(
                      children: [
                        Icon(Icons.summarize, color: Colors.blue),
                        SizedBox(width: 8),
                        Text(
                          'Summary',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    const Divider(),
                    Text(summary, style: const TextStyle(fontSize: 15)),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Events header
            Row(
              children: [
                const Icon(Icons.event_note, color: Colors.deepPurple),
                const SizedBox(width: 8),
                Text(
                  'Extracted Events (${events.length})',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),

            // Event list
            if (events.isEmpty)
              const Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Text('No events found.'),
                ),
              )
            else
              ...events.map((event) => _buildEventCard(event)),

            const SizedBox(height: 16),

            // Memory count
            Center(
              child: Text(
                '📦 Total events in memory: $totalInMemory',
                style: const TextStyle(color: Colors.grey),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEventCard(Map<String, dynamic> event) {
    final type = (event['type'] ?? 'unknown').toString().toUpperCase();
    final desc = event['description'] ?? 'No description';
    final rawDate = event['raw_date'];
    final parsedDate = event['parsed_date'];
    final time = event['time'];
    final parsedTime = event['parsed_time'];
    final person = event['person'];

    IconData icon;
    Color color;
    switch (type) {
      case 'MEETING':
        icon = Icons.groups;
        color = Colors.blue;
        break;
      case 'TASK':
        icon = Icons.task_alt;
        color = Colors.orange;
        break;
      case 'MEDICATION':
        icon = Icons.medication;
        color = Colors.red;
        break;
      default:
        icon = Icons.event;
        color = Colors.grey;
    }

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: color.withAlpha(40),
          child: Icon(icon, color: color, size: 22),
        ),
        title: Text(desc, style: const TextStyle(fontWeight: FontWeight.w500)),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (parsedDate != null)
              Text('📅 $parsedDate${parsedTime != null ? ' at $parsedTime' : ''}'),
            if (rawDate != null && parsedDate == null)
              Text('📅 $rawDate${time != null ? ' at $time' : ''}'),
            if (person != null) Text('👤 $person'),
          ],
        ),
        trailing: Chip(
          label: Text(type, style: const TextStyle(fontSize: 10)),
          backgroundColor: color.withAlpha(30),
          side: BorderSide.none,
          padding: EdgeInsets.zero,
        ),
      ),
    );
  }
}
