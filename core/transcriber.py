"""
Transcriber Module — Audio → Text
==================================
Uses OpenAI Whisper (offline) to convert an audio file into text.
Whisper downloads the model on first run (~140 MB for 'base') and
works fully offline after that.

Model is cached as a singleton — loaded once, reused for all calls.
Model size is controlled by config.WHISPER_MODEL_SIZE.
"""

import os
import shutil
import time
import whisper

# Module-level model cache: {model_size: model}
_model_cache: dict = {}


def _get_model(model_size: str = None):
    """
    Get or load the Whisper model (singleton per size).

    Args:
        model_size: Override model size. If None, reads from config.

    Returns:
        Loaded Whisper model.
    """
    if model_size is None:
        try:
            from config import WHISPER_MODEL_SIZE, DEBUG_TIMING
            model_size = WHISPER_MODEL_SIZE
        except ImportError:
            model_size = "base"

    if model_size in _model_cache:
        print(f"[Transcriber] Reusing cached '{model_size}' model")
        return _model_cache[model_size]

    try:
        debug = __import__("config").DEBUG_TIMING
    except (ImportError, AttributeError):
        debug = False

    t0 = time.time()
    print(f"[Transcriber] Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)
    elapsed = time.time() - t0

    _model_cache[model_size] = model
    print(f"[Transcriber] Model loaded in {elapsed:.1f}s (cached for reuse)")

    if debug:
        print(f"[Transcriber][DEBUG] Load time: {elapsed:.3f}s, size: {model_size}")

    return model


def transcribe_audio(file_path: str, model_size: str = None) -> dict:
    """
    Transcribe an audio file to text using Whisper.

    Args:
        file_path  : Path to the audio file (wav, mp3, m4a, etc.)
        model_size : Whisper model size override. If None, uses config default.

    Returns:
        dict with keys:
            "text"     — full transcription as a single string
            "segments" — list of timestamped segments from Whisper
    """
    # Pre-flight check: does the audio file exist?
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Audio file not found: '{file_path}'\n"
            f"  Please provide the full path to an existing .wav / .mp3 / .m4a file."
        )

    # Pre-flight check: is ffmpeg available?
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg is not installed or not in your PATH.\n"
            "  Whisper needs ffmpeg to decode audio files.\n"
            "  Install it with:  winget install Gyan.FFmpeg\n"
            "  Then restart your terminal so the PATH is updated."
        )

    # Load model (cached singleton)
    model = _get_model(model_size)

    # Run transcription
    t0 = time.time()
    print(f"[Transcriber] Transcribing: {file_path}")
    result = model.transcribe(file_path)
    elapsed = time.time() - t0

    print(f"[Transcriber] Done — {len(result['segments'])} segments in {elapsed:.1f}s")
    return {
        "text": result["text"].strip(),
        "segments": result["segments"],
    }


# ---------------------------------------------------------------------------
# Quick test: run this file directly to transcribe a sample audio file
# Usage:  python -m core.transcriber path/to/audio.wav
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.transcriber <audio_file>")
        sys.exit(1)

    output = transcribe_audio(sys.argv[1])
    print("\n--- Full Transcription ---")
    print(output["text"])
