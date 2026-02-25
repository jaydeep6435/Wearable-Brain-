/// Reminder Screen — Shows upcoming reminders + notification toggle
///
/// Fetches data from GET /reminders and GET /events endpoints.
/// Allows enabling push notifications that fire even when app is closed.

import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/notification_service.dart';

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

  // Notification state
  bool _notificationsEnabled = false;
  int _scheduledCount = 0;
  bool _syncing = false;

  @override
  void initState() {
    super.initState();
    _loadNotificationState();
    _loadData();
  }

  Future<void> _loadNotificationState() async {
    final enabled = await NotificationService.isEnabled();
    if (mounted) setState(() => _notificationsEnabled = enabled);
  }

  Future<void> _toggleNotifications(bool enabled) async {
    setState(() => _syncing = true);

    if (enabled) {
      // Request permission first
      final granted = await NotificationService.requestPermission();
      if (!granted) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('❌ Notification permission denied')),
          );
          setState(() => _syncing = false);
        }
        return;
      }

      // Enable and sync
      await NotificationService.setEnabled(true);
      final count = await NotificationService.syncWithBackend();
      if (mounted) {
        setState(() {
          _notificationsEnabled = true;
          _scheduledCount = count;
          _syncing = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('🔔 Notifications ON — $count events scheduled')),
        );
      }
    } else {
      // Disable and cancel all
      await NotificationService.setEnabled(false);
      await NotificationService.cancelAll();
      if (mounted) {
        setState(() {
          _notificationsEnabled = false;
          _scheduledCount = 0;
          _syncing = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('🔕 Notifications OFF')),
        );
      }
    }
  }

  Future<void> _testNotification() async {
    await NotificationService.showInstant(
      title: '🔔 Test Reminder',
      body: '⏰ This is a test notification from Memory Assistant!',
    );
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('✅ Test notification sent!')),
      );
    }
  }

  Future<void> _loadData() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final reminders = await ApiService.getUpcoming(minutes: 1440); // 24h
      final events = await ApiService.getEvents();

      if (mounted) {
        setState(() {
          _upcoming = reminders['upcoming'] ?? [];
          _todaySchedule = reminders['todays_schedule'] ?? [];
          _allEvents = events['events'] ?? [];
          _isLoading = false;
        });
      }

      // Auto-sync notifications if enabled
      if (_notificationsEnabled) {
        final count = await NotificationService.syncWithBackend();
        if (mounted) setState(() => _scheduledCount = count);
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _isLoading = false;
        });
      }
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
                        // ── Notification Controls ──────────────────
                        _buildNotificationCard(),
                        const SizedBox(height: 16),

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

  // ── Notification Toggle Card ────────────────────────────────
  Widget _buildNotificationCard() {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            // Toggle Row
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: _notificationsEnabled
                        ? Colors.orange.withAlpha(30)
                        : Colors.grey.withAlpha(20),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(
                    _notificationsEnabled
                        ? Icons.notifications_active
                        : Icons.notifications_off,
                    color: _notificationsEnabled ? Colors.orange : Colors.grey,
                    size: 24,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _notificationsEnabled
                            ? '🔔 Push Notifications ON'
                            : '🔕 Push Notifications OFF',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 14,
                          color: _notificationsEnabled ? Colors.orange.shade800 : null,
                        ),
                      ),
                      Text(
                        _notificationsEnabled
                            ? '$_scheduledCount events scheduled'
                            : 'Enable to get reminders when app is closed',
                        style: const TextStyle(fontSize: 11, color: Colors.grey),
                      ),
                    ],
                  ),
                ),
                if (_syncing)
                  const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                else
                  Switch(
                    value: _notificationsEnabled,
                    onChanged: _toggleNotifications,
                    activeThumbColor: Colors.orange,
                  ),
              ],
            ),

            // Test button (only shown when enabled)
            if (_notificationsEnabled) ...[
              const SizedBox(height: 8),
              SizedBox(
                width: double.infinity,
                height: 36,
                child: OutlinedButton.icon(
                  onPressed: _testNotification,
                  icon: const Icon(Icons.send, size: 16),
                  label: const Text('Send Test Notification', style: TextStyle(fontSize: 12)),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.orange,
                    side: BorderSide(color: Colors.orange.withAlpha(100)),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                ),
              ),
            ],
          ],
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
