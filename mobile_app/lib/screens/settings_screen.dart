/// Settings Screen — Config Toggles, Backup, Stats, Audio Source
///
/// Allows toggling:
///   - Simplified Mode (fewer items, shorter summaries)
///   - Low Resource Mode (smaller models, disable embeddings)
///
/// Also provides:
///   - Backup / Restore controls
///   - Resource stats
///   - Audio source selection
///   - Speaker profile management

import 'package:flutter/material.dart';
import '../services/api_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _simplifiedMode = false;
  bool _lowResourceMode = false;
  Map<String, dynamic> _stats = {};
  Map<String, dynamic> _resourceStats = {};
  List<dynamic> _speakers = [];
  Map<String, dynamic> _audioSource = {};
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
        ApiService.getStats(),
        ApiService.getResourceStats(),
        ApiService.getSpeakers(),
        ApiService.getAudioSourceInfo(),
      ]);
      if (!mounted) return;
      setState(() {
        _stats = results[0] as Map<String, dynamic>;
        _resourceStats = results[1] as Map<String, dynamic>;
        _speakers = results[2] as List<dynamic>;
        _audioSource = results[3] as Map<String, dynamic>;

        // Read current config state from stats
        final config = _stats['config'] as Map?;
        if (config != null) {
          _simplifiedMode = config['simplified_mode'] == true;
          _lowResourceMode = config['low_resource_mode'] == true;
        }
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  Future<void> _toggleSimplifiedMode(bool value) async {
    try {
      await ApiService.setConfigFlag('SIMPLIFIED_MODE', value);
      setState(() => _simplifiedMode = value);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed: $e')),
      );
    }
  }

  Future<void> _toggleLowResourceMode(bool value) async {
    try {
      await ApiService.setConfigFlag('LOW_RESOURCE_MODE', value);
      setState(() => _lowResourceMode = value);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed: $e')),
      );
    }
  }

  Future<void> _createBackup() async {
    try {
      final result = await ApiService.createBackup('backup');
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Backup created: ${result['path'] ?? 'done'}'),
          backgroundColor: Colors.green,
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Backup failed: $e')),
      );
    }
  }

  Future<void> _restoreBackup() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Restore Backup?'),
        content: const Text(
          'This will replace all current data with the backup.\n'
          'This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(backgroundColor: Colors.red),
            child: const Text('Restore'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    try {
      final result = await ApiService.restoreBackup('backup');
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Restored: ${result['status'] ?? 'done'}'),
          backgroundColor: Colors.green,
        ),
      );
      _loadAll();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Restore failed: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Settings',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
        ),
        centerTitle: true,
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                // ── Mode Toggles ──
                _buildSectionTitle('Understanding', cs),
                Card(
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    children: [
                      SwitchListTile(
                        title: const Text(
                          'Simplified View',
                          style: TextStyle(fontSize: 17),
                        ),
                        subtitle: const Text(
                          'Show only the most important things',
                          style: TextStyle(fontSize: 14),
                        ),
                        secondary: Icon(
                          Icons.visibility_off,
                          color: cs.primary,
                        ),
                        value: _simplifiedMode,
                        onChanged: _toggleSimplifiedMode,
                      ),
                      const Divider(height: 1, indent: 16, endIndent: 16),
                      SwitchListTile(
                        title: const Text(
                          'Battery Saver',
                          style: TextStyle(fontSize: 17),
                        ),
                        subtitle: const Text(
                          'Use less power, slightly less detail',
                          style: TextStyle(fontSize: 14),
                        ),
                        secondary: Icon(
                          Icons.battery_saver,
                          color: cs.primary,
                        ),
                        value: _lowResourceMode,
                        onChanged: _toggleLowResourceMode,
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 24),

                // ── Backup / Restore ──
                _buildSectionTitle('Data Management', cs),
                Card(
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    children: [
                      ListTile(
                        leading: Icon(Icons.backup, color: cs.primary),
                        title: const Text(
                          'Create Backup',
                          style: TextStyle(fontSize: 17),
                        ),
                        subtitle: const Text('Save all data securely'),
                        trailing: const Icon(Icons.chevron_right),
                        onTap: _createBackup,
                      ),
                      const Divider(height: 1, indent: 16, endIndent: 16),
                      ListTile(
                        leading: const Icon(Icons.restore, color: Colors.orange),
                        title: const Text(
                          'Restore Backup',
                          style: TextStyle(fontSize: 17),
                        ),
                        subtitle: const Text('Replace data from backup'),
                        trailing: const Icon(Icons.chevron_right),
                        onTap: _restoreBackup,
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 24),

                // ── Audio Source ──
                _buildSectionTitle('Audio Source', cs),
                Card(
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: ListTile(
                    leading: Icon(
                      _audioSourceIcon(),
                      color: cs.primary,
                    ),
                    title: Text(
                      'Current: ${_audioSource['type'] ?? 'microphone'}',
                      style: const TextStyle(fontSize: 17),
                    ),
                    subtitle: Text(
                      'Active: ${_audioSource['active'] ?? false}',
                      style: const TextStyle(fontSize: 14),
                    ),
                  ),
                ),

                const SizedBox(height: 24),

                // ── Speaker Profiles ──
                _buildSectionTitle(
                  'Speaker Profiles (${_speakers.length})',
                  cs,
                ),
                if (_speakers.isEmpty)
                  Card(
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: const ListTile(
                      leading: Icon(Icons.person_off, color: Colors.grey),
                      title: Text('No speaker profiles yet'),
                      subtitle: Text(
                        'Profiles are created when conversations are recorded',
                      ),
                    ),
                  ),
                ..._speakers.map((s) => Card(
                      elevation: 0,
                      margin: const EdgeInsets.only(bottom: 6),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: ListTile(
                        leading: CircleAvatar(
                          child: Text(
                            (s['display_name'] ?? s['speaker_label'] ?? '?')
                                .toString()
                                .substring(0, 1)
                                .toUpperCase(),
                          ),
                        ),
                        title: Text(
                          s['display_name'] ?? s['speaker_label'] ?? 'Unknown',
                          style: const TextStyle(fontSize: 16),
                        ),
                        subtitle: Text(s['speaker_label'] ?? ''),
                      ),
                    )),

                const SizedBox(height: 24),

                // ── System Stats ──
                _buildSectionTitle('System', cs),
                Card(
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      children: [
                        _buildStatRow(
                          'Events stored',
                          '${_stats['total_events'] ?? 0}',
                        ),
                        _buildStatRow(
                          'Conversations',
                          '${_stats['total_conversations'] ?? 0}',
                        ),
                        if (_resourceStats.isNotEmpty) ...[
                          _buildStatRow(
                            'Memory (est.)',
                            '${_resourceStats['estimated_memory_mb'] ?? '?'} MB',
                          ),
                          _buildStatRow(
                            'Active threads',
                            '${_resourceStats['active_threads'] ?? 0}',
                          ),
                        ],
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 32),
              ],
            ),
    );
  }

  Widget _buildSectionTitle(String title, ColorScheme cs) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8, left: 4),
      child: Text(
        title,
        style: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w600,
          color: cs.outline,
          letterSpacing: 0.5,
        ),
      ),
    );
  }

  Widget _buildStatRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 15)),
          Text(
            value,
            style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }

  IconData _audioSourceIcon() {
    switch (_audioSource['type']?.toString().toLowerCase()) {
      case 'bluetooth':
        return Icons.bluetooth;
      case 'file':
        return Icons.insert_drive_file;
      default:
        return Icons.mic;
    }
  }
}
