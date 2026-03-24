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
import sys
import warnings
import wave

import numpy as np

# Suppress noisy warnings from pyannote/torch
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Lazy import state for pyannote. Importing torch/pyannote at module import
# time can be expensive or unstable on unsupported Python versions.
PYANNOTE_AVAILABLE = None
PyannotePipeline = None


def _get_pyannote_pipeline_class():
    """Try importing pyannote Pipeline lazily and cache the result."""
    global PYANNOTE_AVAILABLE, PyannotePipeline

    if PYANNOTE_AVAILABLE is not None:
        return PyannotePipeline

    try:
        from pyannote.audio import Pipeline as _Pipeline

        PyannotePipeline = _Pipeline
        PYANNOTE_AVAILABLE = True
    except Exception as e:
        print(f"[Diarizer] pyannote unavailable: {e}")
        PyannotePipeline = None
        PYANNOTE_AVAILABLE = False

    return PyannotePipeline


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

        # pyannote/torch stacks are frequently incompatible on very new
        # Python versions. Keep the pipeline responsive by defaulting to
        # fallback mode in that case unless explicitly forced.
        force_pyannote = os.environ.get("WBRAIN_FORCE_PYANNOTE", "0") == "1"
        if sys.version_info >= (3, 13) and not force_pyannote:
            self._available = False
            print("[Diarizer] pyannote disabled on Python >= 3.13 (set WBRAIN_FORCE_PYANNOTE=1 to override)")
        else:
            self._available = _get_pyannote_pipeline_class() is not None

        # CPU optimization: limit threads to avoid oversubscription
        if self._available:
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
                kwargs["token"] = self.hf_token

            pipeline_cls = _get_pyannote_pipeline_class()
            if pipeline_cls is None:
                raise RuntimeError("pyannote is not available in this environment")

            SpeakerDiarizer._shared_pipeline = pipeline_cls.from_pretrained(
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

            # Run diarization. On newer Python environments pyannote's
            # internal AudioDecoder path may fail; in that case use a
            # preloaded waveform dictionary input instead.
            try:
                diarization = self._pipeline(audio_path, **params)
            except Exception as path_err:
                if "AudioDecoder" not in str(path_err):
                    raise

                waveform = self._load_wav_as_tensor(audio_path, target_sr=16000)
                if waveform is None:
                    raise

                diarization = self._pipeline(
                    {"waveform": waveform, "sample_rate": 16000},
                    **params,
                )

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

    @staticmethod
    def _load_wav_as_tensor(audio_path: str, target_sr: int = 16000):
        """Load a PCM WAV file into a torch tensor for pyannote input."""
        try:
            import torch
        except Exception:
            return None

        try:
            with wave.open(audio_path, "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
        except Exception:
            return None

        if sampwidth != 2:
            return None

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        waveform = torch.from_numpy(audio).unsqueeze(0)
        if sr == target_sr:
            return waveform

        return torch.nn.functional.interpolate(
            waveform.unsqueeze(0),
            size=int(waveform.shape[-1] * float(target_sr) / float(sr)),
            mode="linear",
            align_corners=False,
        ).squeeze(0)

    def _diarize_fallback(self, audio_path: str) -> list[dict]:
        """
        Fallback diarization:
          1) Try lightweight energy-based 2-speaker segmentation.
          2) If unreliable, return a single segment covering the full file.
        """
        segments = self._energy_diarize(audio_path)
        if segments:
            speakers = len(set(s["speaker"] for s in segments))
            print(f"[Diarizer] Fallback energy diarization: {len(segments)} segments, {speakers} speakers")
            return segments

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
    def _energy_diarize(audio_path: str) -> list[dict]:
        """Heuristic 2-speaker diarization from frame-level energy patterns."""
        try:
            with wave.open(audio_path, "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                if sampwidth != 2:
                    return []

                frame_sec = 0.8
                frame_samples = max(1, int(sr * frame_sec))
                energies: list[float] = []

                while True:
                    raw = wf.readframes(frame_samples)
                    if not raw:
                        break
                    chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    if chunk.size == 0:
                        continue
                    if channels > 1:
                        chunk = chunk.reshape(-1, channels).mean(axis=1)
                    rms = float(np.sqrt(np.mean(np.square(chunk))))
                    energies.append(rms)
        except Exception:
            return []

        if len(energies) < 4:
            return []

        p10 = float(np.percentile(energies, 10))
        p90 = float(np.percentile(energies, 90))
        if p90 <= 0 or (p90 - p10) / p90 < 0.22:
            # Not enough energy spread to robustly separate speakers.
            return []

        threshold = float(np.median(energies))
        segments: list[dict] = []
        for i, e in enumerate(energies):
            label = "SPEAKER_00" if e >= threshold else "SPEAKER_01"
            start = round(i * 0.8, 2)
            end = round((i + 1) * 0.8, 2)
            if segments and segments[-1]["speaker"] == label:
                segments[-1]["end"] = end
            else:
                segments.append({"speaker": label, "start": start, "end": end})

        # Remove very short fragments introduced by threshold noise.
        cleaned = [s for s in segments if (s["end"] - s["start"]) >= 0.6]
        if not cleaned:
            cleaned = segments

        speaker_count = len(set(s["speaker"] for s in cleaned))
        return cleaned if speaker_count >= 2 else []

    @staticmethod
    def _merge_adjacent(segments: list[dict], gap_threshold: float = 1.0) -> list[dict]:
        """
        Merge adjacent segments from the same speaker.
        If two consecutive segments from the same speaker are less than
        gap_threshold seconds apart, merge them into one.
        Also filters out very short segments (< 0.5s) as noise.
        """
        if not segments:
            return segments

        # Filter out very short segments (noise)
        min_duration = 0.5
        filtered = [s for s in segments if (s["end"] - s["start"]) >= min_duration]
        if not filtered:
            filtered = segments  # Keep all if filtering removes everything

        merged = [filtered[0].copy()]
        for seg in filtered[1:]:
            last = merged[-1]
            if (
                seg["speaker"] == last["speaker"]
                and seg["start"] - last["end"] < gap_threshold
            ):
                last["end"] = seg["end"]
            else:
                merged.append(seg.copy())

        num_speakers = len(set(s["speaker"] for s in merged))
        print(f"[Diarizer] Merged: {len(segments)} → {len(merged)} segments, {num_speakers} speakers")

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
