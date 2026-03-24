import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_timezone/flutter_timezone.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:timezone/data/latest_all.dart' as tz;
import 'package:timezone/timezone.dart' as tz;

import 'api_service.dart';

class ReminderNotificationService {
  ReminderNotificationService._();

  static final ReminderNotificationService instance = ReminderNotificationService._();

  static const String _channelId = 'upcoming_events_channel';
  static const String _channelName = 'Upcoming Events';
  static const String _channelDescription =
      'Notifications for events that happen in the next hour';
  static const String _scheduledIdsKey = 'scheduled_notification_ids_v1';
  static const int _horizonMinutes = 60 * 24 * 30;
  static const List<int> _leadMinutes = <int>[15, 10, 5, 0];

  final FlutterLocalNotificationsPlugin _notifications =
      FlutterLocalNotificationsPlugin();

  bool _initialized = false;
  bool _scheduling = false;

  Future<void> initialize() async {
    if (_initialized) return;

    const androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
    const settings = InitializationSettings(android: androidSettings);

    tz.initializeTimeZones();
    try {
      final timezoneName = await FlutterTimezone.getLocalTimezone();
      tz.setLocalLocation(tz.getLocation(timezoneName));
    } catch (e) {
      debugPrint('Timezone setup fallback: $e');
    }

    await _notifications.initialize(settings);
    await _requestPermissions();

    _initialized = true;
  }

  Future<void> startMonitoring() async {
    await initialize();
    await refreshSchedules();
  }

  void stopMonitoring() {}

  Future<void> refreshSchedules() async {
    if (!_initialized || _scheduling) return;
    _scheduling = true;

    try {
      final result = await ApiService.getUpcoming(minutes: _horizonMinutes);
      final events = (result['events'] as List<dynamic>? ?? <dynamic>[])
          .whereType<Map>()
          .toList();

      final oldIds = await _loadScheduledIds();
      for (final id in oldIds) {
        await _notifications.cancel(id);
      }

      final newIds = <int>{};
      final nowMs = DateTime.now().millisecondsSinceEpoch;

      for (final event in events) {
        final key = _eventKey(event);
        final eventEpochMs = _toInt(event['event_epoch_ms']);
        if (eventEpochMs == null || eventEpochMs <= nowMs) continue;

        final eventTime = DateTime.fromMillisecondsSinceEpoch(eventEpochMs);
        final description =
            (event['description'] ?? 'Upcoming event').toString().trim();

        for (final lead in _leadMinutes) {
          final triggerEpochMs = eventEpochMs - (lead * 60 * 1000);
          if (triggerEpochMs <= nowMs) continue;

          final triggerTime = DateTime.fromMillisecondsSinceEpoch(triggerEpochMs);
          final notificationId = _notificationId(key, lead);
          final title = lead == 0 ? 'Event is starting now' : 'Reminder: in $lead minutes';
          final body = '$description\n'
              'Event at ${eventTime.toLocal().year.toString().padLeft(4, '0')}-'
              '${eventTime.toLocal().month.toString().padLeft(2, '0')}-'
              '${eventTime.toLocal().day.toString().padLeft(2, '0')} '
              '${eventTime.toLocal().hour.toString().padLeft(2, '0')}:'
              '${eventTime.toLocal().minute.toString().padLeft(2, '0')}';

          await _notifications.zonedSchedule(
            notificationId,
            title,
            body,
            tz.TZDateTime.from(triggerTime, tz.local),
            const NotificationDetails(
              android: AndroidNotificationDetails(
                _channelId,
                _channelName,
                channelDescription: _channelDescription,
                importance: Importance.high,
                priority: Priority.high,
              ),
            ),
            androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
            uiLocalNotificationDateInterpretation:
                UILocalNotificationDateInterpretation.absoluteTime,
          );

          newIds.add(notificationId);
        }
      }

      await _saveScheduledIds(newIds);
    } catch (e) {
      debugPrint('Reminder schedule refresh failed: $e');
    } finally {
      _scheduling = false;
    }
  }

  int _notificationId(String eventKey, int lead) {
    return '$eventKey|$lead'.hashCode & 0x7fffffff;
  }

  Future<void> _requestPermissions() async {
    final androidPlugin = _notifications.resolvePlatformSpecificImplementation<
        AndroidFlutterLocalNotificationsPlugin>();
    await androidPlugin?.requestNotificationsPermission();
    await androidPlugin?.requestExactAlarmsPermission();
  }

  Future<Set<int>> _loadScheduledIds() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getStringList(_scheduledIdsKey) ?? <String>[];
    return saved.map(int.tryParse).whereType<int>().toSet();
  }

  Future<void> _saveScheduledIds(Set<int> ids) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(
      _scheduledIdsKey,
      ids.take(1000).map((e) => e.toString()).toList(),
    );
  }

  String _eventKey(Map<dynamic, dynamic> event) {
    final id = (event['id'] ?? '').toString();
    if (id.isNotEmpty) return id;

    final type = (event['type'] ?? '').toString();
    final description = (event['description'] ?? '').toString();
    final date = (event['parsed_date'] ?? event['raw_date'] ?? '').toString();
    final time = (event['parsed_time'] ?? event['raw_time'] ?? '').toString();
    return '$type|$description|$date|$time';
  }

  int? _toInt(dynamic value) {
    if (value is int) return value;
    if (value is double) return value.round();
    if (value is String) return int.tryParse(value);
    return null;
  }
}
