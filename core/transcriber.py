"""
<<<<<<< HEAD
Transcriber Module — Audio → Text
==================================
Uses OpenAI Whisper (offline) to convert an audio file into text.
Whisper downloads the model on first run (~140 MB for 'base') and
works fully offline after that.
"""

import os
import shutil
import whisper


def transcribe_audio(file_path: str, model_size: str = "base") -> dict:
=======
Transcriber Module -- High-Accuracy Speech Pipeline
=====================================================
Complete offline speech-to-text pipeline:

  Audio -> Silero VAD (speech detection)
       -> faster-whisper (transcription with timestamps)
       -> pyannote (speaker diarization)
       -> Merge (speaker-labeled transcript)

Stack:
  - faster-whisper (CTranslate2 backend, 4x faster than OpenAI Whisper)
  - Silero VAD (speech/silence detection)
  - pyannote/speaker-diarization (speaker separation)

The model downloads on first use and is cached permanently.
All models are singletons -- loaded once, reused for all calls.

API is backward-compatible:
    transcribe_audio(file_path) -> {"text": "...", "segments": [...]}
"""

import os
import time
import wave
import numpy as np

# =====================================================================
# Module-level caches (singletons)
# =====================================================================
_whisper_model = None
_whisper_model_size = None
_vad_model = None
_vad_utils = None


# =====================================================================
# 1. faster-whisper model loader
# =====================================================================

def _get_model(model_size: str = None):
    """
    Get or load the faster-whisper model (singleton).

    Args:
        model_size: Model size ('tiny', 'base', 'small', 'medium', 'large-v3').
                    If None, reads from config.

    Returns:
        faster_whisper.WhisperModel instance.
    """
    global _whisper_model, _whisper_model_size

    if model_size is None:
        try:
            from config import WHISPER_MODEL_SIZE
            model_size = WHISPER_MODEL_SIZE
        except ImportError:
            model_size = "base"

    # Return cached model if same size
    if _whisper_model is not None and _whisper_model_size == model_size:
        print(f"[Transcriber] Reusing cached '{model_size}' model")
        return _whisper_model

    # Determine compute type
    try:
        from config import WHISPER_COMPUTE_TYPE
        compute_type = WHISPER_COMPUTE_TYPE
    except (ImportError, AttributeError):
        compute_type = "int8"

    t0 = time.time()
    print(f"[Transcriber] Loading faster-whisper '{model_size}' (compute={compute_type})...")

    from faster_whisper import WhisperModel

    _whisper_model = WhisperModel(
        model_size,
        device="cpu",
        compute_type=compute_type,
    )
    _whisper_model_size = model_size
    elapsed = time.time() - t0
    print(f"[Transcriber] Model loaded in {elapsed:.1f}s (cached)")

    return _whisper_model


# =====================================================================
# 2. Audio Loader (bypasses broken torchcodec)
# =====================================================================

def _load_wav_as_tensor(audio_path: str, target_sr: int = 16000):
>>>>>>> 5ca5d8f (feat: improve mobile transcription and diarization pipeline)
    """
    Load a WAV file as a torch tensor using Python's built-in wave module.
    This bypasses the broken torchaudio/torchcodec audio I/O on Python 3.14.

    Returns:
        torch.Tensor of shape (1, num_samples) at target_sr, or None on error.
    """
    import torch

    try:
        with wave.open(audio_path, "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.getnframes()
            raw = wf.readframes(frames)
    except Exception:
        # Not a standard WAV -- try soundfile as fallback
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path, dtype='float32')
            if data.ndim > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data).unsqueeze(0)
            if sr != target_sr:
                waveform = _resample_tensor(waveform, sr, target_sr)
            return waveform
        except Exception:
            pass

        # Last resort: use ffmpeg to convert to raw PCM
        try:
            import subprocess
            import tempfile
            tmp_path = os.path.join(tempfile.gettempdir(), "_wbrain_tmp.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", str(target_sr),
                 "-ac", "1", "-f", "wav", tmp_path],
                capture_output=True, timeout=30,
            )
            with wave.open(tmp_path, "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.getnframes()
                raw = wf.readframes(frames)
            os.unlink(tmp_path)
            audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio_np = audio_np.reshape(-1, channels).mean(axis=1)
            return torch.from_numpy(audio_np).unsqueeze(0)
        except Exception as e3:
            print(f"[Audio] Cannot read {audio_path}: {e3}")
            return None

    # Convert raw bytes to numpy
    if sampwidth == 2:
        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio_np = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio_np = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        print(f"[Audio] Unsupported sample width: {sampwidth}")
        return None

    # Convert stereo to mono
    if channels > 1:
        audio_np = audio_np.reshape(-1, channels).mean(axis=1)

    waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)

    # Resample if needed
    if sr != target_sr:
        waveform = _resample_tensor(waveform, sr, target_sr)

    return waveform


def _resample_tensor(waveform, orig_sr: int, target_sr: int):
    """Simple linear interpolation resampling."""
    import torch
    if orig_sr == target_sr:
        return waveform
    ratio = target_sr / orig_sr
    new_length = int(waveform.shape[-1] * ratio)
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0), size=new_length, mode='linear', align_corners=False
    ).squeeze(0)


# =====================================================================
# 3. Silero VAD -- Voice Activity Detection
# =====================================================================

def _load_vad():
    """Load Silero VAD model (singleton)."""
    global _vad_model, _vad_utils
    if _vad_model is not None:
        return _vad_model, _vad_utils

    try:
        import torch
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True,
        )
        print("[VAD] Silero VAD loaded")
        return _vad_model, _vad_utils
    except Exception as e:
        print(f"[VAD] Silero VAD not available: {e}")
        return None, None


def detect_speech_segments(audio_path: str, threshold: float = None):
    """
    Use Silero VAD to find speech regions in audio.
    Uses built-in WAV reader to bypass broken torchcodec.
    """
    if threshold is None:
        try:
            from config import VAD_THRESHOLD
            threshold = VAD_THRESHOLD
        except (ImportError, AttributeError):
            threshold = 0.5

    vad_model, vad_utils = _load_vad()
    if vad_model is None:
        return None

    try:
        (get_speech_timestamps, _, _, _, _) = vad_utils

        # Load audio using our own reader (bypasses torchcodec)
        waveform = _load_wav_as_tensor(audio_path, target_sr=16000)
        if waveform is None:
            print("[VAD] Failed to load audio")
            return None

        # Silero VAD expects 1D tensor
        wav = waveform.squeeze()

        speech_timestamps = get_speech_timestamps(
            wav, vad_model,
            threshold=threshold,
            sampling_rate=16000,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
            speech_pad_ms=100,
        )

        if not speech_timestamps:
            return []

        segments = []
        for ts in speech_timestamps:
            segments.append((ts['start'] / 16000.0, ts['end'] / 16000.0))

        total_speech = sum(e - s for s, e in segments)
        print(f"[VAD] {len(segments)} speech segments, {total_speech:.1f}s total")
        return segments

    except Exception as e:
        print(f"[VAD] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# =====================================================================
# 3. pyannote Speaker Diarization
# =====================================================================

def diarize_audio(audio_path: str):
    """
    Run pyannote speaker diarization.

    Returns:
        List of {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5}
        or None if diarization unavailable.
    """
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError:
        print("[Diarize] pyannote.audio not installed")
        return None

    try:
        # Get HF token
        try:
            from config import HF_TOKEN
            hf_token = HF_TOKEN
        except (ImportError, AttributeError):
            hf_token = os.environ.get("HF_TOKEN", "")

        if not hf_token:
            print("[Diarize] No HF_TOKEN -- cannot load diarization model")
            return None

        # Use the community model (no license agreement needed)
        model_name = "pyannote/speaker-diarization-3.1"

        t0 = time.time()
        print(f"[Diarize] Loading {model_name}...")

        pipeline = PyannotePipeline.from_pretrained(
            model_name, token=hf_token
        )

        import torch
        pipeline.to(torch.device("cpu"))

        print(f"[Diarize] Model loaded in {time.time() - t0:.1f}s")

        # Pre-load audio as waveform dict (bypasses broken torchcodec)
        t0 = time.time()
        waveform = _load_wav_as_tensor(audio_path, target_sr=16000)
        if waveform is None:
            print("[Diarize] Failed to load audio")
            return None

        # pyannote expects {"waveform": tensor, "sample_rate": int}
        audio_input = {"waveform": waveform, "sample_rate": 16000}

        # Run diarization on pre-loaded waveform
        diarization = pipeline(audio_input)

        # pyannote 3.1+ returns DiarizeOutput; extract the Annotation
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        else:
            annotation = diarization  # older pyannote returns Annotation directly

        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
            })

        # Merge adjacent same-speaker segments
        segments = _merge_adjacent_segments(segments)

        elapsed = time.time() - t0
        speakers = set(s["speaker"] for s in segments)
        print(f"[Diarize] {len(segments)} segments, {len(speakers)} speakers in {elapsed:.1f}s")
        return segments

    except Exception as e:
        print(f"[Diarize] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _merge_adjacent_segments(segments, gap_threshold=1.0):
    """Merge consecutive segments from same speaker with < gap_threshold gap."""
    if not segments:
        return segments

    # Filter very short segments (noise)
    filtered = [s for s in segments if (s["end"] - s["start"]) >= 0.3]
    if not filtered:
        filtered = segments

    merged = [filtered[0].copy()]
    for seg in filtered[1:]:
        last = merged[-1]
        if (seg["speaker"] == last["speaker"]
                and seg["start"] - last["end"] < gap_threshold):
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    return merged


# =====================================================================
# 4. Segment Merge -- Align Whisper text with diarization speakers
# =====================================================================

def _assign_speakers(whisper_segments, diarize_segments):
    """
    Assign speaker labels to Whisper transcript segments using
    diarization output. Uses maximum overlap matching.

    Returns list of {"speaker": "Speaker 1", "start": ..., "end": ..., "text": ...}
    """
    if not diarize_segments:
        return [
            {
                "speaker": "Speaker 1",
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
            }
            for seg in whisper_segments
            if seg.get("text", "").strip()
        ]

    # Map pyannote speaker IDs to friendly names
    unique_speakers = list(dict.fromkeys(s["speaker"] for s in diarize_segments))
    speaker_map = {spk: f"Speaker {i+1}" for i, spk in enumerate(unique_speakers)}

    result = []
    for seg in whisper_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)

        # Find best overlapping diarization segment
        best_speaker = unique_speakers[0] if unique_speakers else "SPEAKER_00"
        best_overlap = 0

        for dseg in diarize_segments:
            overlap_start = max(seg_start, dseg["start"])
            overlap_end = min(seg_end, dseg["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg["speaker"]

        result.append({
            "speaker": speaker_map.get(best_speaker, "Speaker 1"),
            "start": round(seg_start, 2),
            "end": round(seg_end, 2),
            "text": text,
        })

    # Merge consecutive segments from same speaker
    if not result:
        return result

    merged = [result[0].copy()]
    for seg in result[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"]:
            last["text"] += " " + seg["text"]
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    return merged


# =====================================================================
# 5. Audio Debug Info
# =====================================================================

def _read_wav_info(file_path: str) -> dict:
    """Read WAV file metadata for debugging."""
    try:
        with wave.open(file_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            duration = frames / rate if rate > 0 else 0

            wf.rewind()
            raw = wf.readframes(frames)
            audio = np.frombuffer(raw, dtype=np.int16)
            peak = int(np.max(np.abs(audio))) if audio.size > 0 else 0
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size > 0 else 0

            return {
                "sample_rate": rate, "channels": channels,
                "sample_width": sampwidth, "duration_sec": duration,
                "frames": frames, "peak_amplitude": peak, "rms": rms,
            }
    except Exception as e:
        return {"error": str(e)}


# =====================================================================
# 6. Main Pipeline -- transcribe_audio()
# =====================================================================

def transcribe_audio(file_path: str, model_size: str = None,
                     enable_diarization: bool = True) -> dict:
    """
    Full speech pipeline: VAD -> Transcription -> Diarization -> Merge.

    Args:
<<<<<<< HEAD
        file_path  : Path to the audio file (wav, mp3, m4a, etc.)
        model_size : Whisper model size — "tiny", "base", "small", "medium", "large"
                     Smaller = faster but less accurate. "base" is a good default.
=======
        file_path       : Path to audio file (wav, mp3, m4a, etc.)
        model_size      : Whisper model size override.
        enable_diarization: Whether to run speaker diarization.
>>>>>>> 5ca5d8f (feat: improve mobile transcription and diarization pipeline)

    Returns:
        dict with keys:
            "text"     -- full transcription string
            "segments" -- list of speaker-labeled segments
            "speakers" -- list of detected speakers
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: '{file_path}'")

    print(f"\n[Pipeline] ========== Speech Pipeline ==========")
    print(f"[Pipeline]   File: {file_path}")
    t_start = time.time()

<<<<<<< HEAD
    # Step 1: Load the Whisper model (downloads once, cached afterwards)
    print(f"[Transcriber] Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)

    # Step 2: Run transcription on the audio file
    print(f"[Transcriber] Transcribing: {file_path}")
    result = model.transcribe(file_path)

    # Step 3: Return the full text and detailed segments
    print(f"[Transcriber] Done — {len(result['segments'])} segments found.")
=======
    # -- Step 0: Audio quality check --
    wav_info = _read_wav_info(file_path)
    if "error" not in wav_info:
        print(f"[Pipeline]   WAV: {wav_info['sample_rate']}Hz, "
              f"{wav_info['channels']}ch, "
              f"{wav_info['duration_sec']:.1f}s, "
              f"peak={wav_info['peak_amplitude']}, "
              f"RMS={wav_info['rms']:.0f}")
        if wav_info['peak_amplitude'] < 100:
            print(f"[Pipeline]   WARNING: Very low audio level!")

    # -- Step 1: Silero VAD --
    print(f"[Pipeline]   Step 1/4: Voice Activity Detection...")
    vad_segments = detect_speech_segments(file_path)

    if vad_segments is not None and len(vad_segments) == 0:
        print(f"[Pipeline]   No speech detected by VAD")
        return {"text": "", "segments": [], "speakers": []}

    # -- Step 2: faster-whisper transcription --
    print(f"[Pipeline]   Step 2/4: Transcription (faster-whisper)...")
    model = _get_model(model_size)

    t0 = time.time()
    segments_iter, info = model.transcribe(
        file_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
            speech_pad_ms=100,
        ),
        language="en",
        condition_on_previous_text=True,
    )

    whisper_segments = []
    full_text_parts = []

    for segment in segments_iter:
        seg_dict = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
        }
        if segment.words:
            seg_dict["words"] = [
                {
                    "word": w.word.strip(),
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                    "probability": round(w.probability, 3),
                }
                for w in segment.words
            ]
        whisper_segments.append(seg_dict)
        full_text_parts.append(segment.text.strip())

    elapsed_asr = time.time() - t0
    full_text = " ".join(full_text_parts).strip()
    print(f"[Pipeline]   Transcribed: {len(whisper_segments)} segments, "
          f"{len(full_text)} chars in {elapsed_asr:.1f}s")
    print(f"[Pipeline]   Language: {info.language} (prob={info.language_probability:.2f})")

    if not full_text:
        print(f"[Pipeline]   No text transcribed")
        return {"text": "", "segments": [], "speakers": []}

    # -- Step 3: Speaker diarization (pyannote) --
    diarize_segments = None
    if enable_diarization:
        print(f"[Pipeline]   Step 3/4: Speaker Diarization (pyannote)...")
        diarize_segments = diarize_audio(file_path)
    else:
        print(f"[Pipeline]   Step 3/4: Diarization skipped")

    # -- Step 4: Merge transcription + speakers --
    print(f"[Pipeline]   Step 4/4: Merging segments...")
    final_segments = _assign_speakers(whisper_segments, diarize_segments)

    speakers = list(dict.fromkeys(s["speaker"] for s in final_segments))

    elapsed_total = time.time() - t_start
    print(f"[Pipeline]   DONE: {len(final_segments)} segments, "
          f"{len(speakers)} speakers in {elapsed_total:.1f}s")

    if len(full_text) < 300:
        print(f"[Pipeline]   Text: '{full_text}'")
    else:
        print(f"[Pipeline]   Text: '{full_text[:200]}...'")

    # Print speaker-labeled output
    for seg in final_segments:
        print(f"[Pipeline]   [{seg['speaker']}] {seg['text'][:80]}")

    print(f"[Pipeline] ========================================\n")

>>>>>>> 5ca5d8f (feat: improve mobile transcription and diarization pipeline)
    return {
        "text": full_text,
        "segments": final_segments,
        "speakers": speakers,
        "vad_segments": vad_segments,
    }


# =====================================================================
# 7. Convenience: transcribe with speakers (for engine compatibility)
# =====================================================================

def transcribe_with_speakers(file_path: str, model_size: str = None) -> dict:
    """
    Convenience wrapper: always enables diarization.
    Returns same format as transcribe_audio.
    """
    return transcribe_audio(file_path, model_size=model_size, enable_diarization=True)


# =====================================================================
# CLI Test
# =====================================================================
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m core.transcriber <audio_file> [--no-diarize]")
        sys.exit(1)

    audio_file = sys.argv[1]
    do_diarize = "--no-diarize" not in sys.argv

    result = transcribe_audio(audio_file, enable_diarization=do_diarize)

    print("\n--- Full Transcription ---")
    print(result["text"])

    print(f"\n--- {len(result['segments'])} Speaker Segments ---")
    for seg in result["segments"]:
        print(f"  [{seg['speaker']}] [{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text']}")

    print(f"\n--- Speakers: {result['speakers']} ---")
