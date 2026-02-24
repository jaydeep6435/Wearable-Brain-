"""
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
    """
    Transcribe an audio file to text using Whisper.

    Args:
        file_path  : Path to the audio file (wav, mp3, m4a, etc.)
        model_size : Whisper model size — "tiny", "base", "small", "medium", "large"
                     Smaller = faster but less accurate. "base" is a good default.

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

    # Step 1: Load the Whisper model (downloads once, cached afterwards)
    print(f"[Transcriber] Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)

    # Step 2: Run transcription on the audio file
    print(f"[Transcriber] Transcribing: {file_path}")
    result = model.transcribe(file_path)

    # Step 3: Return the full text and detailed segments
    print(f"[Transcriber] Done — {len(result['segments'])} segments found.")
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
