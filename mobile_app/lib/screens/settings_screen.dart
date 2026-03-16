/// Settings Screen — Config Toggles, Audio Source Selector, Backup, Stats
///
/// PART 1: Audio Input Source selector with radio buttons
///   - Phone Microphone
///   - Bluetooth / Earbuds
///   - File (Testing)
///
/// Persists selection via SharedPreferences.
/// On app start, loads saved source and calls setAudioSource().

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/api_service.dart';
import 'voice_enrollment_screen.dart';

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
  Map<String, dynamic> _sherpaStatus = {};
  Timer? _sherpaTimer;
  bool _loading = true;

  // Audio source selection
  String _selectedSource = 'microphone';
  String _btDeviceName = '';
  bool _sourceChanging = false;

  @override
  void initState() {
    super.initState();
    _loadAll();
    _startStatusTimer();
  }

  @override
  void dispose() {
    _sherpaTimer?.cancel();
    super.dispose();
  }

  void _startStatusTimer() {
    // Poll Sherpa status every 2 seconds while settings are open
    _sherpaTimer = Timer.periodic(const Duration(seconds: 2), (timer) async {
      try {
        final status = await ApiService.getSherpaStatus();
        if (!mounted) return;
        setState(() {
          _sherpaStatus = status;
        });
        
        // If ready, slow down polling or stop
        if (status['ready'] == true) {
          timer.cancel();
          _startSlowTimer(); // Poll less frequently or stop
        }
      } catch (_) {}
    });
  }

  void _startSlowTimer() {
    _sherpaTimer = Timer.periodic(const Duration(seconds: 10), (timer) async {
      try {
        final status = await ApiService.getSherpaStatus();
        if (!mounted) return;
        setState(() {
          _sherpaStatus = status;
        });
      } catch (_) {}
    });
  }

  Future<void> _loadAll() async {
    setState(() => _loading = true);

    // ★ Load saved preference FIRST (never loses state)
    final prefs = await SharedPreferences.getInstance();
    final savedSource = prefs.getString('audio_source') ?? 'microphone';
    if (mounted) {
      setState(() {
        _selectedSource = savedSource;
      });
    }

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
        final speakersResult = results[2] as Map<String, dynamic>;
        _speakers = (speakersResult['speakers'] as List<dynamic>?) ?? [];
        _audioSource = results[3] as Map<String, dynamic>;

        _btDeviceName = (_audioSource['device_name'] ?? '').toString();

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

    // Load Sherpa status separately so it never blocks the main settings
    try {
      final sherpa = await ApiService.getSherpaStatus();
      if (!mounted) return;
      setState(() => _sherpaStatus = sherpa);
    } catch (_) {
      // Sherpa status will be loaded by the periodic timer
    }
  }

  Future<void> _setAudioSource(String source) async {
    if (_sourceChanging) return;
    setState(() => _sourceChanging = true);

    try {
      final result = await ApiService.setAudioSource(source);
      final prefs = await SharedPreferences.getInstance();

      if (!mounted) return;

      final status = result['status'] ?? '';
      final actualType = result['type'] ?? source;
      final error = result['error'] as String?;

      if (status == 'fallback') {
        // BT failed, engine fell back to microphone
        await prefs.setString('audio_source', 'microphone');
        setState(() {
          _selectedSource = 'microphone';
          _btDeviceName = '';
          _sourceChanging = false;
        });
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(error ?? 'Bluetooth unavailable. Using microphone.'),
              backgroundColor: Colors.orange,
            ),
          );
        }
      } else {
        await prefs.setString('audio_source', actualType);
        setState(() {
          _selectedSource = actualType;
          _btDeviceName = (result['device_name'] ?? '').toString();
          _sourceChanging = false;
        });
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Audio source: ${_sourceDisplayName(actualType)}'),
              backgroundColor: Colors.green,
              duration: const Duration(seconds: 2),
            ),
          );
        }
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _sourceChanging = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to set audio source: $e')),
      );
    }
  }

  String _sourceDisplayName(String source) {
    switch (source) {
      case 'bluetooth': return 'Bluetooth / Earbuds';
      case 'file': return 'File (Testing)';
      default: return 'Phone Microphone';
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
                // ── Audio Input Source ── (PART 1)
                _buildSectionTitle('Audio Input Source', cs),
                Card(
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    children: [
                      _buildAudioSourceTile(
                        'microphone',
                        'Phone Microphone',
                        'Built-in device microphone',
                        Icons.mic,
                        cs,
                      ),
                      const Divider(height: 1, indent: 56, endIndent: 16),
                      _buildAudioSourceTile(
                        'bluetooth',
                        'Bluetooth / Earbuds',
                        _btDeviceName.isNotEmpty
                            ? 'Device: $_btDeviceName'
                            : 'Connect earbuds first',
                        Icons.bluetooth_audio,
                        cs,
                      ),
                      const Divider(height: 1, indent: 56, endIndent: 16),
                      _buildAudioSourceTile(
                        'file',
                        'File (Testing)',
                        'Process a pre-recorded file',
                        Icons.insert_drive_file,
                        cs,
                      ),
                    ],
                  ),
                ),

                // Active source indicator
                if (_selectedSource == 'bluetooth' && _btDeviceName.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 8, left: 4),
                    child: Row(
                      children: [
                        Icon(Icons.check_circle, size: 16, color: Colors.green.shade700),
                        const SizedBox(width: 6),
                        Text(
                          'Connected: $_btDeviceName',
                          style: TextStyle(
                            fontSize: 13,
                            color: Colors.green.shade700,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),

                const SizedBox(height: 24),

                // ── Speech & Intelligence ──
                _buildSectionTitle('Speech & Intelligence', cs),
                
                // Horizontal scroll view for multiple models
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  clipBehavior: Clip.none,
                  child: Row(
                    children: [
                      // ASR Model Card (Speech)
                      SizedBox(
                        width: MediaQuery.of(context).size.width * 0.85,
                        child: _buildModelCard(
                          title: 'High-Accuracy Speech Model',
                          description: 'Required for offline transcription',
                          icon: Icons.psychology,
                          status: _sherpaStatus['asr'] ?? {},
                          defaultSize: 130,
                          onStart: ApiService.startAsrDownload,
                          onPause: ApiService.pauseAsrDownload,
                          onResume: ApiService.resumeAsrDownload,
                          onRetry: ApiService.retryAsrDownload,
                          cs: cs,
                        ),
                      ),
                      const SizedBox(width: 12),
                      
                      // SPK Model Card (Speaker ID)
                      SizedBox(
                        width: MediaQuery.of(context).size.width * 0.85,
                        child: _buildModelCard(
                          title: 'Speaker Identification Model',
                          description: 'Identifies WHO is speaking',
                          icon: Icons.record_voice_over,
                          status: _sherpaStatus['spk'] ?? {},
                          defaultSize: 12,
                          onStart: ApiService.startSpkDownload,
                          onPause: ApiService.pauseSpkDownload,
                          onResume: ApiService.resumeSpkDownload,
                          onRetry: ApiService.retrySpkDownload,
                          cs: cs,
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 24),

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

                // ── Speaker Profiles ──
                _buildSectionTitle(
                  'Voice Profiles (${_speakers.length})',
                  cs,
                ),
                // Enroll button
                Card(
                  elevation: 0,
                  margin: const EdgeInsets.only(bottom: 8),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: ListTile(
                    leading: CircleAvatar(
                      backgroundColor: cs.primaryContainer,
                      child: Icon(Icons.add, color: cs.primary),
                    ),
                    title: const Text(
                      'Enroll a Voice',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                    ),
                    subtitle: const Text('Record speech to identify a person'),
                    trailing: Icon(Icons.chevron_right, color: cs.outline),
                    onTap: () async {
                      final enrolled = await Navigator.push<bool>(
                        context,
                        MaterialPageRoute(
                          builder: (_) => const VoiceEnrollmentScreen(),
                        ),
                      );
                      if (enrolled == true) _loadAll();
                    },
                  ),
                ),
                if (_speakers.isEmpty)
                  Card(
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: const ListTile(
                      leading: Icon(Icons.person_off, color: Colors.grey),
                      title: Text('No voice profiles yet'),
                      subtitle: Text(
                        'Enroll voices to identify speakers in conversations',
                      ),
                    ),
                  ),
                ..._speakers.map((s) {
                  final sp = s is Map ? s : {};
                  final name = sp['name']?.toString() ?? sp['display_name']?.toString() ?? 'Unknown';
                  final samples = sp['sample_count'] ?? 1;
                  final id = sp['id']?.toString() ?? '';

                  return Dismissible(
                    key: Key(id),
                    direction: DismissDirection.endToStart,
                    background: Container(
                      alignment: Alignment.centerRight,
                      padding: const EdgeInsets.only(right: 20),
                      decoration: BoxDecoration(
                        color: Colors.red.shade100,
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Icon(Icons.delete, color: Colors.red.shade700),
                    ),
                    onDismissed: (_) {
                      ApiService.deleteSpeakerProfile(id);
                      setState(() => _speakers.remove(s));
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text('Deleted profile: $name')),
                      );
                    },
                    child: Card(
                      elevation: 0,
                      margin: const EdgeInsets.only(bottom: 6),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: ListTile(
                        leading: CircleAvatar(
                          child: Text(
                            name.substring(0, 1).toUpperCase(),
                          ),
                        ),
                        title: Text(
                          name,
                          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
                        ),
                        subtitle: Text('$samples voice sample${samples > 1 ? 's' : ''}'),
                        trailing: IconButton(
                          icon: Icon(Icons.delete_outline, color: Colors.red.shade400),
                          onPressed: () {
                            showDialog(
                              context: context,
                              builder: (ctx) => AlertDialog(
                                title: const Text('Delete Profile?'),
                                content: Text('Are you sure you want to delete the voice profile for $name?'),
                                actions: [
                                  TextButton(
                                    onPressed: () => Navigator.pop(ctx),
                                    child: const Text('Cancel'),
                                  ),
                                  TextButton(
                                    onPressed: () {
                                      Navigator.pop(ctx);
                                      ApiService.deleteSpeakerProfile(id);
                                      setState(() => _speakers.remove(s));
                                      ScaffoldMessenger.of(context).showSnackBar(
                                        SnackBar(content: Text('Deleted profile: $name')),
                                      );
                                    },
                                    child: const Text('Delete', style: TextStyle(color: Colors.red)),
                                  ),
                                ],
                              ),
                            );
                          },
                        ),
                      ),
                    ),
                  );
                }),

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
                        _buildStatRow(
                          'Audio source',
                          _sourceDisplayName(_selectedSource),
                        ),
                        if (_resourceStats.isNotEmpty) ...[
                          _buildStatRow(
                            'Memory (est.)',
                            '${_resourceStats['memory_mb'] ?? _resourceStats['estimated_memory_mb'] ?? '?'} MB',
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

  // ── Audio Source Radio Tile ──────────────────────────────

  Widget _buildAudioSourceTile(
    String value,
    String title,
    String subtitle,
    IconData icon,
    ColorScheme cs,
  ) {
    final selected = _selectedSource == value;
    return RadioListTile<String>(
      value: value,
      groupValue: _selectedSource,
      onChanged: _sourceChanging
          ? null
          : (v) {
              if (v != null) _setAudioSource(v);
            },
      title: Text(
        title,
        style: TextStyle(
          fontSize: 17,
          fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
        ),
      ),
      subtitle: Text(subtitle, style: const TextStyle(fontSize: 14)),
      secondary: Icon(
        icon,
        color: selected ? cs.primary : cs.outline,
        size: 28,
      ),
      activeColor: cs.primary,
      controlAffinity: ListTileControlAffinity.trailing,
    );
  }

  Widget _buildSectionTitle(String title, ColorScheme cs) {
    return Padding(
      padding: const EdgeInsets.only(left: 8, bottom: 8, top: 8),
      child: Text(
        title.toUpperCase(),
        style: TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.bold,
          color: cs.primary,
          letterSpacing: 1.2,
        ),
      ),
    );
  }

  Widget _buildModelCard({
    required String title,
    required String description,
    required IconData icon,
    required Map status,
    required int defaultSize,
    required VoidCallback onStart,
    required VoidCallback onPause,
    required VoidCallback onResume,
    required VoidCallback onRetry,
    required ColorScheme cs,
  }) {
    final isReady = status['ready'] == true;
    final isInit = status['initializing'] == true;
    final isPaused = status['paused'] == true;
    final error = (status['error'] ?? '').toString();
    final downloaded = status['downloaded_mb'] ?? 0;
    final total = (status['total_mb'] ?? 0) > 0 ? status['total_mb'] : defaultSize;
    final progress = status['progress'] as int? ?? 0;

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: isReady ? Colors.green : cs.primary, size: 32),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 2),
                      Text(
                        status.isEmpty 
                            ? 'Checking status...'
                            : isReady
                                ? 'Ready - $description'
                                : isPaused
                                    ? 'Paused at $downloaded MB'
                                    : isInit
                                        ? 'Downloading: $downloaded / $total MB'
                                        : error.isNotEmpty
                                            ? 'Failed: $error'
                                            : 'Ready to download (~$defaultSize MB)',
                        style: TextStyle(
                          fontSize: 13,
                          color: isReady
                              ? Colors.green.shade700
                              : error.isNotEmpty
                                  ? Colors.red.shade700
                                  : cs.onSurfaceVariant,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
                if (isReady)
                  const Icon(Icons.check_circle, color: Colors.green, size: 24),
              ],
            ),
            
            if (isInit || isPaused || downloaded > 0) ...[
              const SizedBox(height: 16),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: progress / 100.0,
                  backgroundColor: cs.surfaceContainerHighest,
                  color: isPaused ? cs.outline : cs.primary,
                  minHeight: 8,
                ),
              ),
            ],

            const SizedBox(height: 16),
            Row(
              children: [
                if (!isReady) ...[
                  if (isInit && !isPaused)
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: onPause,
                        icon: const Icon(Icons.pause, size: 18),
                        label: const Text('Pause'),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: cs.onSurface,
                          side: BorderSide(color: cs.outlineVariant),
                        ),
                      ),
                    )
                  else if (isPaused)
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: onResume,
                        icon: const Icon(Icons.play_arrow, size: 18),
                        label: const Text('Resume'),
                      ),
                    )
                  else if (error.isNotEmpty)
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: onRetry,
                        icon: const Icon(Icons.refresh, size: 18),
                        label: const Text('Retry'),
                        style: FilledButton.styleFrom(
                          backgroundColor: Colors.red.shade600,
                          foregroundColor: Colors.white,
                        ),
                      ),
                    )
                  else if (!isInit)
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: onStart,
                        icon: const Icon(Icons.download, size: 18),
                        label: const Text('Download'),
                      ),
                    ),
                ] else ...[
                    const Spacer(),
                    Text(
                      'Optimized for offline use',
                      style: TextStyle(fontSize: 12, fontStyle: FontStyle.italic, color: cs.onSurfaceVariant),
                    ),
                ],
              ],
            ),
          ],
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
}
