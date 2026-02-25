"""
Speaker Diarizer — Who spoke when?
=====================================
Detects speaker segments in an audio file using pyannote.audio.

pyannote.audio runs fully offline after downloading the model once.
It outputs timestamped speaker labels:
  [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5}, ...]

Fallback: If pyannote is not installed, returns a single-speaker
segment covering the entire audio duration (graceful degradation).

Model: pyannote/speaker-diarization-3.1
  - Requires a HuggingFace token on first download
  - ~100 MB model, runs on CPU
  - After download, works fully offline

Setup:
  pip install pyannote.audio
  # Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
  # Then: huggingface-cli login
"""

import os
import warnings

# Suppress noisy warnings from pyannote/torch
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Check if pyannote is available
PYANNOTE_AVAILABLE = False
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PyannotePipeline = None


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio (offline).

    Uses a class-level singleton for the pipeline model so it loads
    only once across all instances and requests.

    Usage:
        diarizer = SpeakerDiarizer()
        segments = diarizer.diarize("recording.wav")
        # [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5}, ...]
    """

    # Default model
    DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"

    # ── Class-level singleton for the pipeline ─────────────────
    _shared_pipeline = None      # Loaded once, shared across instances
    _pipeline_load_attempted = False

    def __init__(
        self,
        model: str = None,
        hf_token: str = None,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
    ):
        """
        Initialize the diarizer.

        Args:
            model: HuggingFace model name or local path.
            hf_token: HuggingFace auth token (needed for first download).
            num_speakers: Exact number of speakers (if known).
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
        """
        self.model_name = model or self.DEFAULT_MODEL
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._available = PYANNOTE_AVAILABLE

        # CPU optimization: limit threads to avoid oversubscription
        if PYANNOTE_AVAILABLE:
            try:
                import torch
                cpu_count = os.cpu_count() or 4
                torch.set_num_threads(min(cpu_count, 4))
            except Exception:
                pass

    @property
    def _pipeline(self):
        """Access the shared singleton pipeline."""
        return SpeakerDiarizer._shared_pipeline

    @_pipeline.setter
    def _pipeline(self, value):
        SpeakerDiarizer._shared_pipeline = value

    @property
    def is_available(self) -> bool:
        """Check if pyannote is installed and usable."""
        return self._available

    def _load_pipeline(self):
        """
        Lazy-load the pyannote pipeline (singleton).

        The model loads once and is shared across all SpeakerDiarizer instances.
        Subsequent calls are no-ops. This prevents memory leaks from reloading.
        """
        # Already loaded
        if SpeakerDiarizer._shared_pipeline is not None:
            return

        # Already tried and failed
        if SpeakerDiarizer._pipeline_load_attempted:
            return

        if not self._available:
            print("[Diarizer] pyannote.audio not installed — using fallback")
            return

        SpeakerDiarizer._pipeline_load_attempted = True

        try:
            print(f"[Diarizer] Loading model: {self.model_name}")
            kwargs = {}
            if self.hf_token:
                kwargs["use_auth_token"] = self.hf_token

            SpeakerDiarizer._shared_pipeline = PyannotePipeline.from_pretrained(
                self.model_name, **kwargs
            )

            # Force CPU mode
            import torch
            SpeakerDiarizer._shared_pipeline.to(torch.device("cpu"))

            print("[Diarizer] Model loaded successfully (CPU, singleton)")
        except Exception as e:
            error_msg = str(e)
            print(f"[Diarizer] Failed to load model: {error_msg}")

            if "401" in error_msg or "token" in error_msg.lower():
                print("[Diarizer] HINT: Set HF_TOKEN environment variable:")
                print("[Diarizer]   $env:HF_TOKEN='hf_your_token_here'")
                print("[Diarizer]   Also accept terms at:")
                print(f"[Diarizer]   https://huggingface.co/{self.model_name}")

            print("[Diarizer] Falling back to single-speaker mode")
            self._available = False

    def diarize(self, audio_path: str) -> list[dict]:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file (wav, mp3, m4a).

        Returns:
            List of speaker segments:
            [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5},
                {"speaker": "SPEAKER_01", "start": 3.5, "end": 7.2},
                ...
            ]
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Try pyannote first
        if self._available:
            return self._diarize_pyannote(audio_path)

        # Fallback: single speaker for entire audio
        return self._diarize_fallback(audio_path)

    def _diarize_pyannote(self, audio_path: str) -> list[dict]:
        """Run pyannote speaker diarization."""
        self._load_pipeline()

        if self._pipeline is None:
            return self._diarize_fallback(audio_path)

        try:
            print(f"[Diarizer] Diarizing: {audio_path}")

            # Build parameters
            params = {}
            if self.num_speakers is not None:
                params["num_speakers"] = self.num_speakers
            if self.min_speakers is not None:
                params["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                params["max_speakers"] = self.max_speakers

            # Run diarization
            diarization = self._pipeline(audio_path, **params)

            # Convert to list of dicts
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                })

            # Merge adjacent segments from the same speaker
            segments = self._merge_adjacent(segments)

            print(f"[Diarizer] Found {len(segments)} segments, "
                  f"{len(set(s['speaker'] for s in segments))} speakers")

            return segments

        except Exception as e:
            print(f"[Diarizer] Error during diarization: {e}")
            return self._diarize_fallback(audio_path)

    def _diarize_fallback(self, audio_path: str) -> list[dict]:
        """
        Fallback: return a single segment covering the entire file.
        Uses audio duration from file metadata or estimates from file size.
        """
        duration = self._get_audio_duration(audio_path)
        print(f"[Diarizer] Fallback: single speaker, {duration:.1f}s")

        return [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": round(duration, 2),
            }
        ]

    @staticmethod
    def _merge_adjacent(segments: list[dict], gap_threshold: float = 0.5) -> list[dict]:
        """
        Merge adjacent segments from the same speaker.
        If two consecutive segments from the same speaker are less than
        gap_threshold seconds apart, merge them into one.
        """
        if not segments:
            return segments

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            last = merged[-1]
            if (
                seg["speaker"] == last["speaker"]
                and seg["start"] - last["end"] < gap_threshold
            ):
                last["end"] = seg["end"]
            else:
                merged.append(seg.copy())

        return merged

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        """
        Get audio duration in seconds.
        Tries multiple methods: wave module, mutagen, or file-size estimate.
        """
        # Method 1: wave module (for .wav files)
        if audio_path.lower().endswith(".wav"):
            try:
                import wave
                with wave.open(audio_path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    return frames / rate
            except Exception:
                pass

        # Method 2: mutagen (if installed)
        try:
            from mutagen import File as MutagenFile
            audio = MutagenFile(audio_path)
            if audio and audio.info:
                return audio.info.length
        except ImportError:
            pass
        except Exception:
            pass

        # Method 3: Rough estimate from file size
        # ~32 KB/s for compressed audio (m4a), ~176 KB/s for 16-bit 44.1kHz wav
        file_size = os.path.getsize(audio_path)
        if audio_path.lower().endswith(".wav"):
            return file_size / (16000 * 2)  # 16kHz 16-bit mono
        else:
            return file_size / 32000  # compressed



# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    diarizer = SpeakerDiarizer()
    print(f"pyannote available: {diarizer.is_available}")

    # Test with a sample file if it exists
    test_files = ["test_audio.wav", "test_speech.wav"]
    for f in test_files:
        if os.path.isfile(f):
            print(f"\nDiarizing: {f}")
            segments = diarizer.diarize(f)
            print(json.dumps(segments, indent=2))
            break
    else:
        print("No test audio files found. Skipping diarization test.")
