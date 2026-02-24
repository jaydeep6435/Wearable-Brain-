/// Reminder Screen — Shows upcoming reminders and today's schedule
///
/// Fetches data from GET /reminders and GET /events endpoints.

import 'package:flutter/material.dart';
import '../services/api_service.dart';

class ReminderScreen extends StatefulWidget {
  const ReminderScreen({super.key});

  @override
  State<ReminderScreen> createState() => _ReminderScreenState();
}

class _ReminderScreenState extends State<ReminderScreen> {
  List<dynamic> _upcoming = [];
  List<dynamic> _todaySchedule = [];
  List<dynamic> _allEvents = [];
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final reminders = await ApiService.getReminders(minutes: 1440); // 24h
      final events = await ApiService.getEvents();

      setState(() {
        _upcoming = reminders['upcoming'] ?? [];
        _todaySchedule = reminders['todays_schedule'] ?? [];
        _allEvents = events['events'] ?? [];
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Reminders'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadData,
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.error_outline, size: 48, color: Colors.red),
                      const SizedBox(height: 8),
                      Text('Error: $_error', textAlign: TextAlign.center),
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: _loadData,
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                )
              : RefreshIndicator(
                  onRefresh: _loadData,
                  child: SingleChildScrollView(
                    physics: const AlwaysScrollableScrollPhysics(),
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        // Upcoming reminders
                        _buildSection(
                          '⏰ Upcoming (Next 24h)',
                          _upcoming,
                          emptyText: 'No upcoming reminders',
                          showMinutes: true,
                        ),
                        const SizedBox(height: 16),

                        // Today's schedule
                        _buildSection(
                          '📅 Today\'s Schedule',
                          _todaySchedule,
                          emptyText: 'Nothing scheduled today',
                        ),
                        const SizedBox(height: 16),

                        // All events
                        _buildSection(
                          '📦 All Events in Memory (${_allEvents.length})',
                          _allEvents,
                          emptyText: 'No events stored yet',
                        ),
                      ],
                    ),
                  ),
                ),
    );
  }

  Widget _buildSection(String title, List<dynamic> items,
      {String emptyText = 'None', bool showMinutes = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        if (items.isEmpty)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  const Icon(Icons.info_outline, color: Colors.grey),
                  const SizedBox(width: 8),
                  Text(emptyText, style: const TextStyle(color: Colors.grey)),
                ],
              ),
            ),
          )
        else
          ...items.map((item) => _buildEventTile(item, showMinutes: showMinutes)),
      ],
    );
  }

  Widget _buildEventTile(dynamic item, {bool showMinutes = false}) {
    final type = (item['type'] ?? 'event').toString().toUpperCase();
    final desc = item['description'] ?? 'Unknown';
    final parsedDate = item['parsed_date'];
    final parsedTime = item['parsed_time'] ?? item['time'];
    final minsUntil = item['minutes_until'];

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

    String subtitle = '';
    if (parsedDate != null) subtitle += '📅 $parsedDate';
    if (parsedTime != null) subtitle += '  🕐 $parsedTime';
    if (showMinutes && minsUntil != null) {
      subtitle += '  (in $minsUntil min)';
    }

    return Card(
      margin: const EdgeInsets.only(bottom: 6),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: ListTile(
        dense: true,
        leading: CircleAvatar(
          backgroundColor: color.withAlpha(40),
          radius: 18,
          child: Icon(icon, color: color, size: 20),
        ),
        title: Text(desc, style: const TextStyle(fontWeight: FontWeight.w500)),
        subtitle: subtitle.isNotEmpty ? Text(subtitle) : null,
        trailing: Text(type, style: TextStyle(color: color, fontSize: 11)),
      ),
    );
  }
}
