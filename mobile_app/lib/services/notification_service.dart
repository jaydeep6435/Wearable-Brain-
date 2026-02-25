/// Notification Service — Schedules local push notifications for events
///
/// Uses flutter_local_notifications to show system notifications
/// even when the app is minimized or closed.
///
/// Usage:
///   await NotificationService.init();
///   await NotificationService.scheduleEventNotification(event);

import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:timezone/timezone.dart' as tz;
import 'package:timezone/data/latest.dart' as tz_data;
import 'package:shared_preferences/shared_preferences.dart';
import 'api_service.dart';

class NotificationService {
  static final FlutterLocalNotificationsPlugin _plugin =
      FlutterLocalNotificationsPlugin();

  static bool _initialized = false;

  static const String _prefKey = 'notifications_enabled';

  // ── Initialize ──────────────────────────────────────────────
  static Future<void> init() async {
    if (_initialized) return;

    // Initialize timezone data
    tz_data.initializeTimeZones();

    // Android settings
    const androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');

    // Init settings
    const initSettings = InitializationSettings(android: androidSettings);

    await _plugin.initialize(
      initSettings,
      onDidReceiveNotificationResponse: _onNotificationTap,
    );

    _initialized = true;
    debugPrint('[NotificationService] Initialized');
  }

  // ── Handle notification tap ─────────────────────────────────
  static void _onNotificationTap(NotificationResponse response) {
    debugPrint('[NotificationService] Notification tapped: ${response.payload}');
    // App is already open or will be opened by the system
  }

  // ── Request permission (Android 13+) ────────────────────────
  static Future<bool> requestPermission() async {
    final android = _plugin.resolvePlatformSpecificImplementation<
        AndroidFlutterLocalNotificationsPlugin>();
    if (android != null) {
      final granted = await android.requestNotificationsPermission();
      debugPrint('[NotificationService] Permission granted: $granted');
      return granted ?? false;
    }
    return true; // Non-Android platforms
  }

  // ── Toggle state (persist with SharedPreferences) ───────────
  static Future<bool> isEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_prefKey) ?? false;
  }

  static Future<void> setEnabled(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_prefKey, enabled);
    debugPrint('[NotificationService] Enabled: $enabled');
  }

  // ── Schedule a notification for an event ────────────────────
  static Future<void> scheduleEventNotification({
    required int id,
    required String title,
    required String body,
    required DateTime scheduledTime,
  }) async {
    // Don't schedule in the past
    if (scheduledTime.isBefore(DateTime.now())) {
      debugPrint('[NotificationService] Skipping past event: $title');
      return;
    }

    final tzScheduledTime = tz.TZDateTime.from(scheduledTime, tz.local);

    const androidDetails = AndroidNotificationDetails(
      'memory_reminders',
      'Memory Reminders',
      channelDescription: 'Reminders for upcoming events',
      importance: Importance.high,
      priority: Priority.high,
      icon: '@mipmap/ic_launcher',
      playSound: true,
      enableVibration: true,
      styleInformation: BigTextStyleInformation(''),
    );

    const details = NotificationDetails(android: androidDetails);

    await _plugin.zonedSchedule(
      id,
      title,
      body,
      tzScheduledTime,
      details,
      androidScheduleMode: AndroidScheduleMode.inexactAllowWhileIdle,
      matchDateTimeComponents: null,
    );

    debugPrint('[NotificationService] Scheduled: "$title" at $scheduledTime');
  }

  // ── Show an instant notification (for testing) ──────────────
  static Future<void> showInstant({
    required String title,
    required String body,
  }) async {
    const androidDetails = AndroidNotificationDetails(
      'memory_reminders',
      'Memory Reminders',
      channelDescription: 'Reminders for upcoming events',
      importance: Importance.high,
      priority: Priority.high,
      icon: '@mipmap/ic_launcher',
    );

    const details = NotificationDetails(android: androidDetails);

    await _plugin.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      title,
      body,
      details,
    );
  }

  // ── Cancel all scheduled notifications ──────────────────────
  static Future<void> cancelAll() async {
    await _plugin.cancelAll();
    debugPrint('[NotificationService] All notifications cancelled');
  }

  // ── Sync with backend — schedule notifications for events ───
  static Future<int> syncWithBackend() async {
    try {
      // Fetch upcoming events (next 24 hours)
      final data = await ApiService.getUpcoming(minutes: 1440);
      final upcoming = data['upcoming'] as List<dynamic>? ?? [];

      // Cancel existing and reschedule
      await cancelAll();

      int scheduled = 0;

      for (int i = 0; i < upcoming.length; i++) {
        final event = upcoming[i];
        final desc = event['description'] ?? 'Upcoming event';
        final eventDatetimeStr = event['event_datetime'];
        final minsUntil = event['minutes_until'] as int?;
        final type = (event['type'] ?? 'event').toString().toUpperCase();
        final timeStr = event['parsed_time'] ?? event['time'] ?? '';

        DateTime? eventDt;

        // Try to parse ISO datetime from backend
        if (eventDatetimeStr != null) {
          eventDt = DateTime.tryParse(eventDatetimeStr);
        }

        // Fallback: use minutes_until
        if (eventDt == null && minsUntil != null) {
          eventDt = DateTime.now().add(Duration(minutes: minsUntil));
        }

        if (eventDt == null) continue;

        // Schedule notification 15 minutes before the event
        final notifyTime = eventDt.subtract(const Duration(minutes: 15));

        // Build notification text
        String icon = '⏰';
        if (type == 'MEDICATION') icon = '💊';
        if (type == 'MEETING') icon = '👥';
        if (type == 'TASK') icon = '✅';

        String body = '$icon $desc';
        if (timeStr.isNotEmpty) body += ' at $timeStr';

        // If notify time is in the past but event is in the future,
        // schedule for now + 30 seconds as an immediate alert
        DateTime scheduleAt;
        if (notifyTime.isBefore(DateTime.now()) && eventDt.isAfter(DateTime.now())) {
          scheduleAt = DateTime.now().add(const Duration(seconds: 30));
        } else {
          scheduleAt = notifyTime;
        }

        await scheduleEventNotification(
          id: i + 100, // Offset to avoid ID collision
          title: '🔔 Reminder: $type',
          body: body,
          scheduledTime: scheduleAt,
        );

        scheduled++;
      }

      debugPrint('[NotificationService] Synced: $scheduled notifications scheduled');
      return scheduled;
    } catch (e) {
      debugPrint('[NotificationService] Sync error: $e');
      return 0;
    }
  }
}
