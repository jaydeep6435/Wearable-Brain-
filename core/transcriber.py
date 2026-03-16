"""
Transcriber Module -- Offline ASR + Optional Multi-Speaker Diarization
========================================================================

Pipeline:
  Audio -> faster-whisper transcription -> optional pyannote diarization
        -> speaker assignment (timestamp overlap)

This module keeps backward-compatible APIs used across the project:
  - transcribe_audio(...)
  - transcribe_with_speakers(...)
  - _get_model(...)
  - detect_speech_segments(...)
"""

from __future__ import annotations

import os
import time
import wave
from typing import Any

import numpy as np

# Singleton caches
_whisper_model = None
_whisper_model_size = None


def _load_audio_mono(file_path: str, target_sr: int = 16000):
    """Load audio into mono float32 numpy array at target_sr."""
    # Fast path for PCM WAV files (no external codecs required).
    try:
        if file_path.lower().endswith(".wav"):
            with wave.open(file_path, "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())

            if sampwidth == 2:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 4:
                audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = None

            if audio is not None:
                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)
                if sr != target_sr and audio.size > 1:
                    old_x = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
                    new_len = max(1, int(audio.size * float(target_sr) / float(sr)))
                    new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
                    audio = np.interp(new_x, old_x, audio).astype(np.float32)
                return audio.astype(np.float32), target_sr
    except Exception:
        pass

    try:
        from pydub import AudioSegment

        seg = AudioSegment.from_file(file_path)
        seg = seg.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
        raw = np.array(seg.get_array_of_samples(), dtype=np.float32)
        audio = (raw / 32768.0).astype(np.float32)
        return audio, target_sr
    except Exception:
        return None, None


def _segment_features(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray | None:
    """Extract compact acoustic features from one time range."""
    s = max(0, int(start * sr))
    e = min(audio.size, int(end * sr))
    if e - s < int(0.20 * sr):
        return None

    x = audio[s:e]
    if x.size < 64:
        return None

    energy = float(np.sqrt(np.mean(np.square(x))) + 1e-8)
    zc = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))

    nfft = 1
    while nfft < x.size:
        nfft *= 2
    win = np.hanning(x.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(x * win, n=nfft)).astype(np.float32)
    if spec.size < 8:
        return None

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr).astype(np.float32)
    spec_sum = float(spec.sum() + 1e-8)
    centroid = float((freqs * spec).sum() / spec_sum)
    spread = float(np.sqrt(((freqs - centroid) ** 2 * spec).sum() / spec_sum))

    # Coarse spectral envelope (8 bands)
    bands = np.array_split(spec, 8)
    band_energy = np.array([float(b.mean()) if b.size else 0.0 for b in bands], dtype=np.float32)
    band_energy = band_energy / (band_energy.sum() + 1e-8)

    feat = np.concatenate(
        [
            np.array([np.log(energy), zc, centroid / 4000.0, spread / 4000.0], dtype=np.float32),
            band_energy,
        ]
    )
    return feat


def _kmeans2(x: np.ndarray, max_iter: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Minimal deterministic 2-means for small feature sets."""
    if x.shape[0] < 2:
        return np.zeros(x.shape[0], dtype=np.int32), np.zeros((2, x.shape[1]), dtype=np.float32)

    c0 = x[0].copy()
    # Pick farthest point from c0 for stable init
    d = np.sum((x - c0) ** 2, axis=1)
    c1 = x[int(np.argmax(d))].copy()
    centers = np.stack([c0, c1], axis=0)

    labels = np.zeros(x.shape[0], dtype=np.int32)
    for _ in range(max_iter):
        d0 = np.sum((x - centers[0]) ** 2, axis=1)
        d1 = np.sum((x - centers[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in (0, 1):
            pts = x[labels == k]
            if pts.size > 0:
                centers[k] = pts.mean(axis=0)

    return labels, centers


def _heuristic_diarize_from_asr(file_path: str, whisper_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fallback diarization from ASR timestamps + acoustic clustering."""
    if len(whisper_segments) < 4:
        return []

    audio, sr = _load_audio_mono(file_path, target_sr=16000)
    if audio is None or sr is None:
        return []

    feats = []
    seg_idx = []
    for i, seg in enumerate(whisper_segments):
        f = _segment_features(audio, sr, float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
        if f is not None:
            feats.append(f)
            seg_idx.append(i)

    if len(feats) < 4:
        return []

    x = np.stack(feats).astype(np.float32)
    # Standardize features
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    xz = (x - mean) / std

    labels, centers = _kmeans2(xz)

    # Separation quality gate
    sep = float(np.linalg.norm(centers[0] - centers[1]))
    if sep < 1.6:
        return []

    # Majority balance gate: avoid tiny outlier cluster being treated as speaker
    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    if min(n0, n1) < 2:
        return []

    raw_segments = []
    full_labels = [0] * len(whisper_segments)
    for i, idx in enumerate(seg_idx):
        full_labels[idx] = int(labels[i])

    for i, seg in enumerate(whisper_segments):
        label = f"SPEAKER_0{full_labels[i]}"
        raw_segments.append(
            {
                "speaker": label,
                "start": round(float(seg.get("start", 0.0)), 2),
                "end": round(float(seg.get("end", 0.0)), 2),
            }
        )

    # Merge consecutive same-label spans
    merged = [raw_segments[0].copy()]
    for seg in raw_segments[1:]:
        if seg["speaker"] == merged[-1]["speaker"] and seg["start"] - merged[-1]["end"] <= 1.0:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    speakers = len(set(s["speaker"] for s in merged))
    return merged if speakers >= 2 else []


def _dialogue_turn_diarize(whisper_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Text/turn-taking fallback for short two-person dialog clips."""
    if len(whisper_segments) < 4:
        return []

    starts = [float(s.get("start", 0.0)) for s in whisper_segments]
    ends = [float(s.get("end", 0.0)) for s in whisper_segments]
    texts = [str(s.get("text", "")).strip().lower() for s in whisper_segments]

    durations = [max(0.0, e - st) for st, e in zip(starts, ends)]
    pauses = [max(0.0, starts[i] - ends[i - 1]) for i in range(1, len(starts))]

    if np.mean(durations) > 4.0:
        return []
    if pauses and float(np.median(pauses)) < 0.6:
        return []

    has_question = any(t.endswith("?") for t in texts)
    response_tokens = ("yes", "no", "okay", "ok", "sure", "right", "yeah", "yep")
    has_response = any(any(t.startswith(tok) for tok in response_tokens) for t in texts)
    if not (has_question and has_response):
        return []

    out = []
    for i, seg in enumerate(whisper_segments):
        out.append(
            {
                "speaker": f"SPEAKER_0{i % 2}",
                "start": round(float(seg.get("start", 0.0)), 2),
                "end": round(float(seg.get("end", 0.0)), 2),
            }
        )

    return out


def _get_model(model_size: str | None = None):
    """Load and cache faster-whisper model."""
    global _whisper_model, _whisper_model_size

    if model_size is None:
        try:
            from config import WHISPER_MODEL_SIZE

            model_size = WHISPER_MODEL_SIZE
        except (ImportError, AttributeError):
            model_size = "base"

    if _whisper_model is not None and _whisper_model_size == model_size:
        return _whisper_model

    try:
        from config import WHISPER_COMPUTE_TYPE

        compute_type = WHISPER_COMPUTE_TYPE
    except (ImportError, AttributeError):
        compute_type = "int8"

    from faster_whisper import WhisperModel

    t0 = time.time()
    print(f"[Transcriber] Loading faster-whisper '{model_size}' (compute={compute_type})...")
    _whisper_model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
    _whisper_model_size = model_size
    print(f"[Transcriber] Model ready in {time.time() - t0:.1f}s")

    return _whisper_model


def _read_wav_info(file_path: str) -> dict[str, Any]:
    """Read basic WAV metadata used for light validation and fallback timing."""
    with wave.open(file_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        channels = wf.getnchannels()
        duration = frames / rate if rate else 0.0
        return {
            "frames": frames,
            "sample_rate": rate,
            "channels": channels,
            "duration_sec": float(duration),
        }


def _energy_based_speech_regions(file_path: str, frame_ms: int = 200) -> list[tuple[float, float]]:
    """Fallback speech-region detector when Silero VAD is unavailable."""
    try:
        with wave.open(file_path, "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                info = _read_wav_info(file_path)
                return [(0.0, round(info["duration_sec"], 2))]

            frame_samples = max(1, int(rate * frame_ms / 1000.0))

            rms_values: list[float] = []
            while True:
                raw = wf.readframes(frame_samples)
                if not raw:
                    break
                chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                if channels > 1:
                    chunk = chunk.reshape(-1, channels).mean(axis=1)
                if chunk.size == 0:
                    rms_values.append(0.0)
                else:
                    rms_values.append(float(np.sqrt(np.mean(np.square(chunk)))))
    except Exception:
        return None

    if not rms_values:
        return []

    threshold = max(200.0, float(np.percentile(rms_values, 60)))
    regions: list[tuple[float, float]] = []
    start_idx = None
    for i, rms in enumerate(rms_values):
        if rms >= threshold and start_idx is None:
            start_idx = i
        elif rms < threshold and start_idx is not None:
            s = start_idx * frame_ms / 1000.0
            e = i * frame_ms / 1000.0
            if e - s >= 0.3:
                regions.append((round(s, 2), round(e, 2)))
            start_idx = None

    if start_idx is not None:
        s = start_idx * frame_ms / 1000.0
        e = len(rms_values) * frame_ms / 1000.0
        if e - s >= 0.3:
            regions.append((round(s, 2), round(e, 2)))

    if not regions:
        info = _read_wav_info(file_path)
        return [(0.0, round(info["duration_sec"], 2))]

    return regions


def _load_vad():
    """Lazy-load Silero VAD if available."""
    try:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )
        return model, utils
    except Exception:
        return None, None


def _load_wav_tensor(file_path: str, target_sr: int = 16000):
    """Load WAV into torch tensor for Silero VAD."""
    import torch

    with wave.open(file_path, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        return None

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    waveform = torch.from_numpy(audio).unsqueeze(0)
    if sr == target_sr:
        return waveform

    resampled = torch.nn.functional.interpolate(
        waveform.unsqueeze(0),
        size=int(waveform.shape[-1] * float(target_sr) / float(sr)),
        mode="linear",
        align_corners=False,
    ).squeeze(0)
    return resampled


def detect_speech_segments(audio_path: str, threshold: float | None = None):
    """
    Return list of speech regions as tuples: (start_sec, end_sec).

    Returns None only if file cannot be read.
    """
    if not os.path.isfile(audio_path):
        return None

    if threshold is None:
        try:
            from config import VAD_THRESHOLD

            threshold = float(VAD_THRESHOLD)
        except (ImportError, AttributeError, TypeError, ValueError):
            threshold = 0.5

    model, utils = _load_vad()
    if model is None or utils is None:
        return _energy_based_speech_regions(audio_path)

    try:
        get_speech_timestamps = utils[0]
        wav = _load_wav_tensor(audio_path, target_sr=16000)
        if wav is None:
            return _energy_based_speech_regions(audio_path)

        ts = get_speech_timestamps(
            wav.squeeze(),
            model,
            threshold=threshold,
            sampling_rate=16000,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
            speech_pad_ms=100,
        )
        if not ts:
            return []

        return [
            (round(item["start"] / 16000.0, 2), round(item["end"] / 16000.0, 2))
            for item in ts
        ]
    except Exception:
        return _energy_based_speech_regions(audio_path)


def _assign_speakers(
    whisper_segments: list[dict[str, Any]], diarization_segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Assign speaker labels to ASR segments via max-overlap alignment."""
    if not whisper_segments:
        return []

    if not diarization_segments:
        return [
            {
                "speaker": "Speaker 1",
                "start": round(float(seg.get("start", 0.0)), 2),
                "end": round(float(seg.get("end", 0.0)), 2),
                "text": seg.get("text", "").strip(),
            }
            for seg in whisper_segments
            if seg.get("text", "").strip()
        ]

    speaker_order: list[str] = []
    for seg in diarization_segments:
        label = str(seg.get("speaker", "SPEAKER_00"))
        if label not in speaker_order:
            speaker_order.append(label)

    if not speaker_order:
        speaker_order = ["SPEAKER_00"]

    speaker_map = {label: f"Speaker {idx + 1}" for idx, label in enumerate(speaker_order)}

    labeled: list[dict[str, Any]] = []
    for seg in whisper_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        best_speaker = speaker_order[0]
        best_overlap = -1.0

        for dia in diarization_segments:
            d_start = float(dia.get("start", 0.0))
            d_end = float(dia.get("end", 0.0))
            overlap = max(0.0, min(end, d_end) - max(start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(dia.get("speaker", speaker_order[0]))

        labeled.append(
            {
                "speaker": speaker_map.get(best_speaker, "Speaker 1"),
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
            }
        )

    if not labeled:
        return labeled

    merged = [labeled[0].copy()]
    for seg in labeled[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and seg["start"] - prev["end"] <= 0.8:
            prev["end"] = seg["end"]
            prev["text"] = f"{prev['text']} {seg['text']}".strip()
        else:
            merged.append(seg.copy())

    return merged


def transcribe_audio(
    file_path: str,
    model_size: str | None = None,
    enable_diarization: bool = True,
) -> dict[str, Any]:
    """
    Transcribe audio and optionally return speaker-labeled segments.

    Returns:
      {
        "text": str,
        "segments": [{speaker,start,end,text}, ...],
        "speakers": ["Speaker 1", ...],
        "vad_segments": [(start,end), ...]
      }
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: '{file_path}'")

    t0 = time.time()
    model = _get_model(model_size)

    vad_segments = detect_speech_segments(file_path)
    if vad_segments == []:
        print("[Transcriber] VAD found no speech; continuing with ASR fallback pass")

    def _run_asr_pass(vad_filter: bool) -> tuple[list[dict[str, Any]], str]:
        segments_iter, _info = model.transcribe(
            file_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=vad_filter,
            language="en",
            condition_on_previous_text=True,
        )

        segments_local: list[dict[str, Any]] = []
        text_local: list[str] = []
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            segments_local.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                }
            )
            text_local.append(text)

        return segments_local, " ".join(text_local).strip()

    whisper_segments, full_text = _run_asr_pass(vad_filter=True)
    if not full_text:
        print("[Transcriber] First ASR pass empty; retrying without Whisper VAD filter")
        whisper_segments, full_text = _run_asr_pass(vad_filter=False)
    if not full_text:
        return {
            "text": "",
            "segments": [],
            "speakers": [],
            "vad_segments": vad_segments or [],
        }

    diarization_segments: list[dict[str, Any]] | None = None
    if enable_diarization:
        try:
            from diarization.diarizer import SpeakerDiarizer

            diarizer = SpeakerDiarizer()
            diarization_segments = diarizer.diarize(file_path)
        except Exception as e:
            print(f"[Transcriber] Diarization skipped: {e}")
            diarization_segments = None

    # If pyannote/fallback produced one speaker, try ASR-timestamp acoustic clustering.
    if enable_diarization:
        speaker_count = len(set(s.get("speaker") for s in (diarization_segments or []) if s.get("speaker")))
        if speaker_count < 2:
            heuristic = _heuristic_diarize_from_asr(file_path, whisper_segments)
            if heuristic:
                diarization_segments = heuristic
                print(f"[Transcriber] Heuristic diarization activated: {len(heuristic)} segments, 2 speakers")
            else:
                turn_based = _dialogue_turn_diarize(whisper_segments)
                if turn_based:
                    diarization_segments = turn_based
                    print(f"[Transcriber] Turn-taking diarization activated: {len(turn_based)} segments, 2 speakers")

    final_segments = _assign_speakers(whisper_segments, diarization_segments or [])
    speakers = list(dict.fromkeys(seg["speaker"] for seg in final_segments))

    print(
        "[Transcriber] Done in "
        f"{time.time() - t0:.1f}s | text={len(full_text)} chars | "
        f"segments={len(final_segments)} | speakers={len(speakers)}"
    )

    return {
        "text": full_text,
        "segments": final_segments,
        "speakers": speakers,
        "vad_segments": vad_segments or [],
    }


def transcribe_with_speakers(file_path: str, model_size: str | None = None) -> dict[str, Any]:
    """Compatibility wrapper for explicit speaker-aware transcription."""
    return transcribe_audio(file_path, model_size=model_size, enable_diarization=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.transcriber <audio_file>")
        raise SystemExit(1)

    output = transcribe_audio(sys.argv[1], enable_diarization=True)
    print("\n--- Transcript ---")
    print(output["text"])
    print("\n--- Segments ---")
    for item in output["segments"]:
        print(
            f"[{item['speaker']}] "
            f"[{item['start']:.2f}s-{item['end']:.2f}s] "
            f"{item['text']}"
        )
