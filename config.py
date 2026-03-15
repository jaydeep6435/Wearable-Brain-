"""
Configuration — Memory Assistant Settings
============================================
Central configuration for the Wearable Brain memory assistant.

LOW_RESOURCE_MODE gates heavy features for mid-range Android
devices (4–6GB RAM). When enabled:
  - Whisper uses 'tiny' model (~75MB) instead of 'base' (~140MB)
  - Sentence embedding search is disabled (TF-IDF only)
  - Fewer voiceprint embeddings stored per speaker
  - Higher VAD threshold (less frequent processing)
  - Shorter max recording duration

Override via environment variable:
    WBRAIN_LOW_RESOURCE=1  python main.py
"""

import os


# ── Resource Mode ──────────────────────────────────────────────

LOW_RESOURCE_MODE = os.environ.get("WBRAIN_LOW_RESOURCE", "0") == "1"
"""Enable low-resource mode for constrained devices."""


# ── ASR (WhisperX + faster-whisper) ────────────────────────────

WHISPER_MODEL_SIZE = "tiny" if LOW_RESOURCE_MODE else "base"
"""faster-whisper model: 'tiny' (~39MB), 'base' (~74MB), 'small' (~244MB),
'medium' (~769MB), 'large-v3' (~1.5GB)."""

WHISPER_COMPUTE_TYPE = "int8"
"""CTranslate2 compute type: 'int8' (fastest CPU), 'float16' (GPU), 'float32'."""

VAD_THRESHOLD = 0.5
"""Silero VAD speech probability threshold (0.0-1.0). Lower = more sensitive."""

HF_TOKEN = os.environ.get("HF_TOKEN", "")
"""HuggingFace token for downloading pyannote diarization models."""


# ── Semantic Search ────────────────────────────────────────────

ENABLE_EMBEDDINGS = not LOW_RESOURCE_MODE
"""Enable sentence-transformer embedding search (all-MiniLM-L6-v2).
When False, falls back to TF-IDF only."""


# ── Voice Fingerprinting ──────────────────────────────────────

MAX_VOICEPRINTS_PER_SPEAKER = 5 if LOW_RESOURCE_MODE else 10
"""Maximum voice embeddings stored per speaker for adaptive matching."""


# ── VAD / Audio Worker ─────────────────────────────────────────

VAD_SILENCE_THRESHOLD = 500.0 if LOW_RESOURCE_MODE else 300.0
"""RMS amplitude threshold for voice activity detection."""

MAX_RECORDING_DURATION_SEC = 60.0 if LOW_RESOURCE_MODE else 120.0
"""Maximum continuous recording duration per segment (seconds)."""

PROCESSING_COOLDOWN_SEC = 5.0 if LOW_RESOURCE_MODE else 2.0
"""Minimum interval between consecutive ASR processing jobs."""

RING_BUFFER_SECONDS = 15.0 if LOW_RESOURCE_MODE else 30.0
"""Ring buffer size for Bluetooth audio source (seconds)."""

MIN_SPEECH_DURATION_SEC = 2.0 if LOW_RESOURCE_MODE else 1.5
"""Minimum speech duration to trigger processing (seconds)."""

SILENCE_TIMEOUT_SEC = 3.0 if LOW_RESOURCE_MODE else 2.0
"""Silence duration to stop recording in VAD mode (seconds)."""


# ── Audio ──────────────────────────────────────────────────────

SAMPLE_RATE = 16000
"""Default audio sample rate in Hz (16kHz for Whisper)."""

CHANNELS = 1
"""Audio channels (1 = mono)."""

CHUNK_DURATION_MS = 30
"""Duration of each audio read chunk in milliseconds."""


# ── Debug ──────────────────────────────────────────────────────

DEBUG_TIMING = os.environ.get("WBRAIN_DEBUG_TIMING", "0") == "1"
"""Enable timing logs for model loading and processing steps."""


# ── Cognitive Simplification (Phase R) ─────────────────────────

SIMPLIFIED_MODE = os.environ.get("WBRAIN_SIMPLIFIED", "0") == "1"
"""Reduce cognitive load: fewer summary points, only high-importance results."""

MAX_SUMMARY_POINTS = 2 if SIMPLIFIED_MODE else 5
"""Maximum key summary points shown to the user."""

MIN_DISPLAY_IMPORTANCE = 3 if SIMPLIFIED_MODE else 0
"""Minimum importance_score for events shown in simplified mode."""

REINFORCEMENT_INTERVAL_HOURS = 12
"""Re-show critical events if not viewed within this many hours."""

ESCALATION_MAX_LEVEL = 3
"""Maximum escalation level for missed/overdue events (0-3)."""


# ── Summary ────────────────────────────────────────────────────

def get_config_summary() -> dict:
    """Return a snapshot of current configuration."""
    return {
        "low_resource_mode": LOW_RESOURCE_MODE,
        "whisper_model": WHISPER_MODEL_SIZE,
        "enable_embeddings": ENABLE_EMBEDDINGS,
        "max_voiceprints_per_speaker": MAX_VOICEPRINTS_PER_SPEAKER,
        "vad_silence_threshold": VAD_SILENCE_THRESHOLD,
        "max_recording_duration_sec": MAX_RECORDING_DURATION_SEC,
        "processing_cooldown_sec": PROCESSING_COOLDOWN_SEC,
        "ring_buffer_seconds": RING_BUFFER_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "debug_timing": DEBUG_TIMING,
        "simplified_mode": SIMPLIFIED_MODE,
        "max_summary_points": MAX_SUMMARY_POINTS,
        "reinforcement_interval_hours": REINFORCEMENT_INTERVAL_HOURS,
        "escalation_max_level": ESCALATION_MAX_LEVEL,
    }


if __name__ == "__main__":
    import json
    print("Wearable Brain Configuration")
    print("=" * 40)
    print(json.dumps(get_config_summary(), indent=2))
