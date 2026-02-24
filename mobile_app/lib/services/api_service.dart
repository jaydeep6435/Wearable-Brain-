/// API Service — connects Flutter app to local Flask API
///
/// Base URL: http://10.0.2.2:5000 (Android emulator → localhost)
///           http://127.0.0.1:5000 (iOS/desktop)

import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  // Android emulator maps 10.0.2.2 → host machine's localhost
  // Change this to your computer's IP for physical device testing
  static String get baseUrl {
    if (Platform.isAndroid) {
      return 'http://10.0.2.2:5000';
    }
    return 'http://127.0.0.1:5000';
  }

  // -- POST /process_text -------------------------------------------------
  static Future<Map<String, dynamic>> processText(String text) async {
    final response = await http.post(
      Uri.parse('$baseUrl/process_text'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'text': text}),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to process text: ${response.body}');
    }
  }

  // -- POST /query --------------------------------------------------------
  static Future<Map<String, dynamic>> queryMemory(String question) async {
    final response = await http.post(
      Uri.parse('$baseUrl/query'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'question': question}),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to query: ${response.body}');
    }
  }

  // -- GET /events --------------------------------------------------------
  static Future<Map<String, dynamic>> getEvents({String? type}) async {
    String url = '$baseUrl/events';
    if (type != null) url += '?type=$type';

    final response = await http.get(Uri.parse(url));

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to get events: ${response.body}');
    }
  }

  // -- GET /reminders -----------------------------------------------------
  static Future<Map<String, dynamic>> getReminders({int minutes = 60}) async {
    final response = await http.get(
      Uri.parse('$baseUrl/reminders?minutes=$minutes'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to get reminders: ${response.body}');
    }
  }

  // -- GET / (health check) -----------------------------------------------
  static Future<bool> checkServer() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/'))
          .timeout(const Duration(seconds: 3));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
