/// Memory Assistant — Flutter Mobile App
///
/// Fully offline Alzheimer's Memory Assistant.
/// All features accessible via MethodChannel — no terminal dependency.
///
/// Navigation (5 tabs):
///   1. Dashboard  — Daily brief, urgent items, reinforcement
///   2. Conversation — Record/process text/audio
///   3. Search     — Query memories with ranked results
///   4. Reminders  — Upcoming events & alerts
///   5. Settings   — Toggles, backup, stats

import 'package:flutter/material.dart';
import 'screens/dashboard_screen.dart';
import 'screens/home_screen.dart';
import 'screens/search_screen.dart';
import 'screens/reminder_screen.dart';
import 'screens/settings_screen.dart';
import 'services/notification_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await NotificationService.init();
  runApp(const MemoryAssistantApp());
}

class MemoryAssistantApp extends StatelessWidget {
  const MemoryAssistantApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Memory Assistant',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF5B6ABF),
        useMaterial3: true,
        brightness: Brightness.light,
        textTheme: const TextTheme(
          bodyLarge: TextStyle(fontSize: 17),
          bodyMedium: TextStyle(fontSize: 16),
          titleLarge: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
        ),
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: const Color(0xFF5B6ABF),
        useMaterial3: true,
        brightness: Brightness.dark,
        textTheme: const TextTheme(
          bodyLarge: TextStyle(fontSize: 17),
          bodyMedium: TextStyle(fontSize: 16),
          titleLarge: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
        ),
      ),
      themeMode: ThemeMode.system,
      home: const MainNavigation(),
    );
  }
}

class MainNavigation extends StatefulWidget {
  const MainNavigation({super.key});

  @override
  State<MainNavigation> createState() => _MainNavigationState();
}

class _MainNavigationState extends State<MainNavigation> {
  int _currentIndex = 0;

  final List<Widget> _screens = const [
    DashboardScreen(),
    HomeScreen(),
    SearchScreen(),
    ReminderScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_currentIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (index) {
          setState(() => _currentIndex = index);
        },
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.dashboard_outlined),
            selectedIcon: Icon(Icons.dashboard),
            label: 'My Day',
          ),
          NavigationDestination(
            icon: Icon(Icons.mic_outlined),
            selectedIcon: Icon(Icons.mic),
            label: 'Record',
          ),
          NavigationDestination(
            icon: Icon(Icons.search),
            selectedIcon: Icon(Icons.manage_search),
            label: 'Search',
          ),
          NavigationDestination(
            icon: Icon(Icons.notifications_outlined),
            selectedIcon: Icon(Icons.notifications),
            label: 'Reminders',
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}
