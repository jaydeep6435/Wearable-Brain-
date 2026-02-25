/// Dashboard Screen — Daily Brief, Urgent Items, Reinforcement
///
/// Primary screen for Alzheimer's patients.
/// Shows calm daily summary, urgent items needing attention,
/// and critical items needing reinforcement.
///
/// UX: Large fonts, calm colors, max 2–3 actions per card.

import 'package:flutter/material.dart';
import '../services/api_service.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  Map<String, dynamic> _brief = {};
  List<dynamic> _urgentItems = [];
  List<dynamic> _reinforcementItems = [];
  List<dynamic> _escalated = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadAll();
  }

  Future<void> _loadAll() async {
    setState(() => _loading = true);
    try {
      final results = await Future.wait([
        ApiService.generateDailyBrief(),
        ApiService.getUrgentItems(),
        ApiService.getReinforcementItems(),
        ApiService.checkEscalations(),
      ]);
      if (!mounted) return;
      setState(() {
        _brief = results[0] as Map<String, dynamic>;
        _urgentItems = results[1] as List<dynamic>;
        _reinforcementItems = results[2] as List<dynamic>;
        _escalated = results[3] as List<dynamic>;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  Future<void> _markShown(String eventId) async {
    try {
      await ApiService.markItemShown(eventId);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Got it! ✓'),
          duration: Duration(seconds: 1),
        ),
      );
      _loadAll();
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'My Day',
          style: TextStyle(fontSize: 24, fontWeight: FontWeight.w600),
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadAll,
            tooltip: 'Refresh',
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _loadAll,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  // ── Daily Brief Card ──
                  _buildDailyBriefCard(cs),
                  const SizedBox(height: 16),

                  // ── Urgent Items ──
                  if (_urgentItems.isNotEmpty) ...[
                    _buildSectionHeader('⚠️  Needs Attention', cs.error),
                    ..._urgentItems.map((item) => _buildUrgentCard(item, cs)),
                    const SizedBox(height: 16),
                  ],

                  // ── Escalated Items ──
                  if (_escalated.isNotEmpty) ...[
                    _buildSectionHeader('🔴  Overdue', cs.error),
                    ..._escalated.map((item) => _buildEscalatedCard(item, cs)),
                    const SizedBox(height: 16),
                  ],

                  // ── Reinforcement Items ──
                  if (_reinforcementItems.isNotEmpty) ...[
                    _buildSectionHeader('🔔  Remember', cs.primary),
                    ..._reinforcementItems
                        .map((item) => _buildReinforcementCard(item, cs)),
                    const SizedBox(height: 16),
                  ],

                  // ── All Clear ──
                  if (_urgentItems.isEmpty &&
                      _escalated.isEmpty &&
                      _reinforcementItems.isEmpty)
                    _buildAllClearCard(cs),
                ],
              ),
            ),
    );
  }

  Widget _buildDailyBriefCard(ColorScheme cs) {
    final greeting =
        _brief['greeting'] as String? ?? "Here's your day.";
    final summaryText =
        _brief['summary_text'] as String? ?? 'Everything is okay.';
    final closing =
        _brief['closing'] as String? ?? 'Take your time.';

    return Card(
      elevation: 0,
      color: cs.primaryContainer.withValues(alpha: 0.3),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.wb_sunny, color: cs.primary, size: 28),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    greeting,
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      color: cs.onSurface,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              summaryText,
              style: TextStyle(
                fontSize: 17,
                height: 1.6,
                color: cs.onSurface.withValues(alpha: 0.85),
              ),
            ),
            const SizedBox(height: 12),
            Text(
              closing,
              style: TextStyle(
                fontSize: 15,
                fontStyle: FontStyle.italic,
                color: cs.onSurface.withValues(alpha: 0.6),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionHeader(String title, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text(
        title,
        style: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.bold,
          color: color,
        ),
      ),
    );
  }

  Widget _buildUrgentCard(dynamic item, ColorScheme cs) {
    final desc = item is Map ? (item['description'] ?? 'Event') : '$item';
    final type = item is Map ? (item['type'] ?? '') : '';
    final icon = _iconForType(type.toString());

    return Card(
      elevation: 1,
      color: cs.errorContainer.withValues(alpha: 0.5),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
        leading: Icon(icon, color: cs.error, size: 32),
        title: Text(
          desc.toString(),
          style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w500),
        ),
        subtitle: type.toString().isNotEmpty
            ? Text(
                type.toString().toUpperCase(),
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: cs.error,
                ),
              )
            : null,
      ),
    );
  }

  Widget _buildEscalatedCard(dynamic item, ColorScheme cs) {
    final desc = item is Map ? (item['description'] ?? 'Event') : '$item';
    final level = item is Map ? (item['escalation_level'] ?? 0) : 0;

    return Card(
      elevation: 2,
      color: Colors.red.shade50,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: Colors.red.shade300, width: 1.5),
      ),
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
        leading: Badge(
          label: Text('L$level'),
          backgroundColor: Colors.red,
          child: const Icon(Icons.warning_amber, color: Colors.red, size: 32),
        ),
        title: Text(
          desc.toString(),
          style: const TextStyle(
            fontSize: 17,
            fontWeight: FontWeight.w600,
          ),
        ),
        subtitle: Text(
          'Escalation Level $level — Needs immediate attention',
          style: TextStyle(fontSize: 13, color: Colors.red.shade700),
        ),
      ),
    );
  }

  Widget _buildReinforcementCard(dynamic item, ColorScheme cs) {
    final desc = item is Map ? (item['description'] ?? 'Event') : '$item';
    final eventId = item is Map ? (item['id'] ?? '') : '';
    final shownCount = item is Map ? (item['shown_count'] ?? 0) : 0;

    return Card(
      elevation: 0,
      color: cs.tertiaryContainer.withValues(alpha: 0.4),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
        leading: Icon(Icons.repeat, color: cs.tertiary, size: 28),
        title: Text(
          desc.toString(),
          style: const TextStyle(fontSize: 17),
        ),
        subtitle: shownCount > 0
            ? Text(
                'Shown $shownCount time${shownCount > 1 ? 's' : ''}',
                style: TextStyle(fontSize: 13, color: cs.outline),
              )
            : null,
        trailing: eventId.toString().isNotEmpty
            ? FilledButton.tonal(
                onPressed: () => _markShown(eventId.toString()),
                child: const Text('Got it'),
              )
            : null,
      ),
    );
  }

  Widget _buildAllClearCard(ColorScheme cs) {
    return Card(
      elevation: 0,
      color: Colors.green.withValues(alpha: 0.1),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: const Padding(
        padding: EdgeInsets.all(32),
        child: Column(
          children: [
            Icon(Icons.check_circle, color: Colors.green, size: 48),
            SizedBox(height: 12),
            Text(
              'All clear!',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
            ),
            SizedBox(height: 4),
            Text(
              'Nothing urgent right now. Relax.',
              style: TextStyle(fontSize: 16, color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }

  IconData _iconForType(String type) {
    switch (type.toLowerCase()) {
      case 'medication':
        return Icons.medication;
      case 'meeting':
      case 'appointment':
        return Icons.calendar_today;
      case 'task':
        return Icons.task_alt;
      case 'visit':
        return Icons.person;
      default:
        return Icons.event;
    }
  }
}
