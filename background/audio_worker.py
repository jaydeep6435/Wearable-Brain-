"""
Background Audio Worker — Session Recording + VAD Modes
=========================================================
Two modes of operation:

1. SESSION MODE (primary):
   - User explicitly starts/stops recording
   - Full conversation captured as one WAV
   - Saved permanently to recordings/
   - Processed after stop → full pipeline

2. VAD MODE (optional, for wearable/hands-free):
   - Continuous listening with Voice Activity Detection
   - Auto-detects speech, saves chunks, processes automatically

Accepts any AudioSource (mic, file, Bluetooth, ESP32).

Usage (Session Mode):
    from audio.microphone import MicrophoneSource

    worker = AudioWorker(engine, audio_source=MicrophoneSource())
    worker.start_recording()
    # ... conversation happens ...
    result = worker.stop_recording()

Usage (VAD Mode):
    worker = AudioWorker(engine, audio_source=MicrophoneSource())
    worker.start_vad_listening()
    worker.stop()

No cloud. No network. CPU-friendly.
"""

import os
import sys
import time
import wave
import threading
import numpy as np
from datetime import datetime

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Directories
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "recordings")
TMP_AUDIO_DIR = os.path.join(PROJECT_ROOT, "tmp_audio")


class AudioWorker:
    """
    Background audio capture with two modes:

    1. Session Recording — record full conversation, process after stop
    2. VAD Listening — auto-detect speech and process chunks

    Accepts any AudioSource implementation (mic, file, Bluetooth, etc.).
    Thread-safe. Uses daemon threads that die with the main process.
    """

    def __init__(
        self,
        engine=None,
        audio_source=None,
        sample_rate: int = None,
        channels: int = None,
        chunk_duration_ms: int = None,
        # VAD settings
        silence_threshold: float = None,
        min_speech_sec: float = None,
        max_record_sec: float = None,
        silence_timeout_sec: float = None,
    ):
        """
        Initialize the audio worker.

        All parameters default to values from config.py when not specified.

        Args:
            engine: MemoryAssistantEngine instance for processing.
            audio_source: AudioSource implementation. If None, defaults
                          to MicrophoneSource when recording starts.
            sample_rate: Audio sample rate in Hz (16000 for Whisper).
            channels: Number of audio channels (1 = mono).
            chunk_duration_ms: Duration of each audio read chunk in ms.
            silence_threshold: RMS threshold for VAD mode.
            min_speech_sec: Minimum speech duration for VAD mode.
            max_record_sec: Maximum recording duration for VAD mode.
            silence_timeout_sec: Silence timeout for VAD mode.
        """
        # Load config defaults
        try:
            import config as cfg
        except ImportError:
            cfg = None

        def _cfg(name, fallback):
            if cfg:
                return getattr(cfg, name, fallback)
            return fallback

        self._engine = engine
        self._audio_source = audio_source
        self.sample_rate = sample_rate or _cfg("SAMPLE_RATE", 16000)
        self.channels = channels or _cfg("CHANNELS", 1)
        self.chunk_duration_ms = chunk_duration_ms or _cfg("CHUNK_DURATION_MS", 30)
        self.silence_threshold = silence_threshold or _cfg("VAD_SILENCE_THRESHOLD", 50.0) # Lowered for debug
        self.min_speech_sec = min_speech_sec or _cfg("MIN_SPEECH_DURATION_SEC", 1.5)
        self.max_record_sec = max_record_sec or _cfg("MAX_RECORDING_DURATION_SEC", 120.0)
        self.silence_timeout_sec = silence_timeout_sec or _cfg("SILENCE_TIMEOUT_SEC", 2.0)
        self._cooldown_sec = _cfg("PROCESSING_COOLDOWN_SEC", 2.0)

        # Thread management
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Mode tracking
        self._mode: str | None = None  # "session" or "vad"

        # Session recording state
        self._session_frames: list = []
        self._session_start_time: float | None = None
        self._session_file: str | None = None
        self._processing_result: dict | None = None

        # Statistics
        self._recordings_count = 0
        self._total_processed = 0
        self._last_recording_time: str | None = None
        self._started_at: str | None = None
        self._errors: list[str] = []
        self._last_process_time: float = 0  # Cooldown tracking

        # Computed values
        self._chunk_samples = int(self.sample_rate * self.chunk_duration_ms / 1000)

    # ── Audio Source Management ────────────────────────────────

    def _get_source(self):
        """
        Get or create the audio source.

        If no source was provided, creates a MicrophoneSource.
        This lazy creation avoids importing sounddevice at init time.
        """
        if self._audio_source is not None:
            return self._audio_source

        # Default to microphone
        from audio.microphone import MicrophoneSource
        self._audio_source = MicrophoneSource(
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        return self._audio_source

    @property
    def audio_source(self):
        """The current audio source (may be None if not yet created)."""
        return self._audio_source

    @audio_source.setter
    def audio_source(self, source):
        """Set a new audio source. Must be done while worker is stopped."""
        if self.is_running():
            raise RuntimeError("Cannot change audio source while running")
        self._audio_source = source

    def _source_name(self) -> str:
        """Human-readable name of the current audio source."""
        src = self._audio_source
        if src is None:
            return "MicrophoneSource (default)"
        return type(src).__name__

    # ════════════════════════════════════════════════════════════
    #  SESSION RECORDING MODE (Primary)
    # ════════════════════════════════════════════════════════════

    def start_recording(self) -> dict:
        """
        Start recording a full conversation session.

        Audio is captured continuously until stop_recording() is called.
        The complete recording is saved as a WAV file and then processed.

        Returns:
            dict with status info.
        """
        if self.is_running():
            return {"status": "already_running", "mode": self._mode}

        os.makedirs(RECORDINGS_DIR, exist_ok=True)

        # Start the audio source
        source = self._get_source()
        try:
            source.start()
        except RuntimeError as e:
            return {"status": "error", "error": str(e)}

        self._mode = "session"
        self._session_frames = []
        self._session_start_time = time.time()
        self._stop_event.clear()
        self._errors.clear()
        self._processing_result = None
        self._started_at = datetime.now().isoformat()

        self._thread = threading.Thread(
            target=self._record_loop,
            name="AudioWorker-Session",
            daemon=True,
        )
        self._thread.start()

        print(f"[Worker] Session recording started ({self._source_name()})")
        print("[Worker] Speak naturally. Call stop_recording() when done.")

        return {
            "status": "recording",
            "mode": "session",
            "source": self._source_name(),
            "started_at": self._started_at,
        }

    def stop_recording(self, process: bool = True) -> dict:
        """
        Stop the session recording, save the WAV, and optionally process it.

        Args:
            process: If True, run the full pipeline on the saved recording.

        Returns:
            dict with file path, duration, and processing results.
        """
        if not self.is_running() or self._mode != "session":
            return {"status": "not_recording"}

        # Signal thread to stop
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        # Stop the audio source
        source = self._audio_source
        if source:
            source.stop()

        # Calculate duration
        duration = 0.0
        if self._session_start_time:
            duration = time.time() - self._session_start_time

        # Save the recording
        if not self._session_frames:
            self._thread = None
            self._mode = None
            return {"status": "empty", "duration": 0.0, "message": "No audio captured"}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(RECORDINGS_DIR, f"session_{timestamp}.wav")

        try:
            audio_data = np.concatenate(self._session_frames, axis=0)
            actual_duration = len(audio_data) / self.sample_rate
            self._save_wav(wav_path, audio_data)
            self._recordings_count += 1
            self._last_recording_time = datetime.now().isoformat()
            self._session_file = wav_path

            print(f"[Worker] Session saved: {wav_path} ({actual_duration:.1f}s)")

            # Process through engine
            result = {"status": "saved", "file": wav_path, "duration": actual_duration}

            if process and self._engine is not None:
                print("[Worker] Processing recording through pipeline...")
                try:
                    pipeline_result = self._engine.process_audio(wav_path)
                    self._total_processed += 1
                    self._processing_result = pipeline_result

                    result["processing"] = {
                        "transcription": pipeline_result.get("transcription", "")[:200],
                        "num_speakers": pipeline_result.get("num_speakers", 1),
                        "summary": pipeline_result.get("summary", ""),
                        "events_saved": pipeline_result.get("events_saved", 0),
                        "events": pipeline_result.get("events", []),
                        "conversation_id": pipeline_result.get("conversation_id"),
                    }
                    result["status"] = "processed"

                    events = pipeline_result.get("events_saved", 0)
                    print(f"[Worker] Done: {events} events extracted")

                except Exception as e:
                    error_msg = f"Processing error: {e}"
                    self._errors.append(error_msg)
                    result["error"] = error_msg
                    print(f"[Worker] {error_msg}")
            else:
                print("[Worker] Recording saved (processing skipped)")

            return result

        except Exception as e:
            error_msg = f"Save error: {e}"
            self._errors.append(error_msg)
            return {"status": "error", "error": error_msg}

        finally:
            self._session_frames = []
            self._thread = None
            self._mode = None

    def _record_loop(self):
        """Continuous recording loop for session mode using AudioSource."""
        source = self._audio_source
        if source is None:
            self._errors.append("No audio source configured")
            return

        try:
            while not self._stop_event.is_set():
                chunk = source.read_chunk(self._chunk_samples)
                if chunk.size > 0:
                    self._session_frames.append(chunk.copy())
                    # Periodic logging: every ~2 seconds
                    if len(self._session_frames) % (2000 // self.chunk_duration_ms) == 0:
                        rms = self._compute_rms(chunk)
                        max_amp = np.max(np.abs(chunk))
                        elapsed = len(self._session_frames) * self.chunk_duration_ms / 1000
                        print(f"[Worker][REC] {elapsed:.0f}s | frames={len(self._session_frames)} | "
                              f"RMS={rms:.1f} | MaxAmp={max_amp} | source={self._source_name()}")

        except Exception as e:
            error_msg = f"Recording error: {e}"
            self._errors.append(error_msg)
            print(f"[Worker] {error_msg}")

    # ════════════════════════════════════════════════════════════
    #  VAD LISTENING MODE (Optional, for hands-free/wearable)
    # ════════════════════════════════════════════════════════════

    def start_vad_listening(self) -> None:
        """
        Start background VAD listening (auto-detect speech).

        Non-blocking. Records speech chunks automatically and
        processes them via the engine. For hands-free use.
        """
        if self.is_running():
            print("[Worker] Already running")
            return

        os.makedirs(TMP_AUDIO_DIR, exist_ok=True)
        self._cleanup_tmp()

        # Start the audio source
        source = self._get_source()
        try:
            source.start()
        except RuntimeError as e:
            self._errors.append(str(e))
            print(f"[Worker] {e}")
            return

        self._mode = "vad"
        self._stop_event.clear()
        self._errors.clear()
        self._started_at = datetime.now().isoformat()

        self._thread = threading.Thread(
            target=self._vad_listen_loop,
            name="AudioWorker-VAD",
            daemon=True,
        )
        self._thread.start()
        print(f"[Worker] VAD listening started ({self._source_name()}, threshold={self.silence_threshold})")

    def _vad_listen_loop(self):
        """VAD listening loop using AudioSource."""
        source = self._audio_source
        if source is None:
            return

        recording_frames = []
        is_recording = False
        silence_start = None
        recording_start = None

        try:
            while not self._stop_event.is_set():
                chunk = source.read_chunk(self._chunk_samples)
                if chunk.size == 0:
                    # Source exhausted (e.g., FileSource finished)
                    if not source.is_active:
                        break
                    continue

                rms = self._compute_rms(chunk)

                if rms > self.silence_threshold:
                    if not is_recording:
                        is_recording = True
                        recording_start = time.time()
                        recording_frames = []

                    recording_frames.append(chunk.copy())
                    silence_start = None

                    elapsed = time.time() - recording_start
                    if elapsed >= self.max_record_sec:
                        self._finalize_vad_chunk(recording_frames)
                        is_recording = False
                        recording_frames = []

                elif is_recording:
                    recording_frames.append(chunk.copy())

                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= self.silence_timeout_sec:
                        duration = time.time() - recording_start
                        if duration >= self.min_speech_sec:
                            self._finalize_vad_chunk(recording_frames)
                        is_recording = False
                        recording_frames = []
                        silence_start = None

        except Exception as e:
            self._errors.append(f"VAD error: {e}")

        # Stop the source when VAD loop ends
        if source:
            source.stop()

    def _finalize_vad_chunk(self, frames: list) -> None:
        """Save a VAD-detected speech chunk and process it."""
        if not frames:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(TMP_AUDIO_DIR, f"vad_{timestamp}.wav")

        try:
            audio_data = np.concatenate(frames, axis=0)
            self._save_wav(wav_path, audio_data)
            self._recordings_count += 1
            self._last_recording_time = datetime.now().isoformat()

            if self._engine:
                result = self._engine.process_audio(wav_path)
                self._total_processed += 1
                if "error" in result:
                    self._errors.append(result["error"])

            self._delete_file(wav_path)

        except Exception as e:
            self._errors.append(f"VAD chunk error: {e}")

    # ════════════════════════════════════════════════════════════
    #  COMMON API
    # ════════════════════════════════════════════════════════════

    def stop(self) -> None:
        """Stop any running mode (session or VAD)."""
        if not self.is_running():
            return

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._mode = None

        # Stop the audio source
        if self._audio_source:
            self._audio_source.stop()

    def is_running(self) -> bool:
        """Check if the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def mode(self) -> str | None:
        """Current mode: 'session', 'vad', or None."""
        return self._mode if self.is_running() else None

    @property
    def session_duration(self) -> float:
        """Duration of current session recording in seconds."""
        if self._mode == "session" and self._session_start_time:
            return time.time() - self._session_start_time
        return 0.0

    def status(self) -> dict:
        """Get full worker status."""
        return {
            "running": self.is_running(),
            "mode": self.mode,
            "source": self._source_name(),
            "started_at": self._started_at,
            "session_duration": round(self.session_duration, 1),
            "recordings_captured": self._recordings_count,
            "recordings_processed": self._total_processed,
            "last_recording": self._last_recording_time,
            "last_file": self._session_file,
            "errors": self._errors[-5:],
            "config": {
                "sample_rate": self.sample_rate,
                "silence_threshold": self.silence_threshold,
                "min_speech_sec": self.min_speech_sec,
                "max_record_sec": self.max_record_sec,
                "silence_timeout_sec": self.silence_timeout_sec,
            },
        }

    def get_last_result(self) -> dict | None:
        """Get the processing result from the last stop_recording() call."""
        return self._processing_result

    def list_recordings(self) -> list[dict]:
        """List all saved recordings in the recordings/ directory."""
        if not os.path.isdir(RECORDINGS_DIR):
            return []

        recordings = []
        for f in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
            if f.endswith(".wav"):
                fp = os.path.join(RECORDINGS_DIR, f)
                size_mb = os.path.getsize(fp) / (1024 * 1024)
                recordings.append({
                    "file": f,
                    "path": fp,
                    "size_mb": round(size_mb, 2),
                    "modified": datetime.fromtimestamp(
                        os.path.getmtime(fp)
                    ).isoformat(),
                })
        return recordings

    # ════════════════════════════════════════════════════════════
    #  UTILITIES
    # ════════════════════════════════════════════════════════════

    def _save_wav(self, path: str, audio_data: np.ndarray) -> None:
        """Save numpy audio array as 16-bit WAV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

    @staticmethod
    def _compute_rms(audio_chunk: np.ndarray) -> float:
        """Compute Root Mean Square amplitude of an audio chunk."""
        if audio_chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)))

    @staticmethod
    def _delete_file(path: str) -> None:
        """Safely delete a file."""
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass

    def _cleanup_tmp(self) -> None:
        """Remove all WAV files from tmp_audio directory."""
        if not os.path.isdir(TMP_AUDIO_DIR):
            return
        count = 0
        for f in os.listdir(TMP_AUDIO_DIR):
            fp = os.path.join(TMP_AUDIO_DIR, f)
            if os.path.isfile(fp) and f.endswith(".wav"):
                self._delete_file(fp)
                count += 1
        if count:
            print(f"[Worker] Cleaned up {count} temp files")


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("AudioWorker — Quick Test")
    print("=" * 40)

    # Test RMS
    silence = np.zeros(480, dtype=np.int16)
    loud = np.full(480, 5000, dtype=np.int16)
    print(f"RMS silence: {AudioWorker._compute_rms(silence):.1f}")
    print(f"RMS loud:    {AudioWorker._compute_rms(loud):.1f}")

    # Test status
    worker = AudioWorker(engine=None)
    print(f"Status: {worker.status()}")
    print(f"Source: {worker._source_name()}")
    print(f"Recordings: {worker.list_recordings()}")
    print("Quick test passed!")
