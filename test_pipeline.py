"""
Test Script -- Verify Speech Pipeline Accuracy
================================================
Tests the new faster-whisper + Silero VAD + pyannote pipeline.

Usage:
    python test_pipeline.py                  # Test with default test_speech.wav
    python test_pipeline.py path/to/audio.wav  # Test with custom file
    python test_pipeline.py --no-diarize     # Skip diarization
"""

import os
import sys
import json
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.transcriber import transcribe_audio, _get_model, detect_speech_segments


def test_model_loading():
    """Test 1: Model loads successfully."""
    print("=" * 50)
    print("TEST 1: Model Loading")
    print("=" * 50)

    t0 = time.time()
    model = _get_model()
    elapsed = time.time() - t0

    assert model is not None, "Model failed to load!"
    print(f"  PASS: Model loaded in {elapsed:.1f}s\n")


def test_vad(audio_path):
    """Test 2: VAD detects speech."""
    print("=" * 50)
    print("TEST 2: Voice Activity Detection")
    print("=" * 50)

    segments = detect_speech_segments(audio_path)

    if segments is None:
        print("  SKIP: Silero VAD not available\n")
        return

    print(f"  Speech segments found: {len(segments)}")
    for i, (start, end) in enumerate(segments):
        print(f"    [{i+1}] {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")

    assert len(segments) > 0, "VAD found no speech!"
    print(f"  PASS: {len(segments)} speech segments detected\n")


def test_transcription(audio_path, enable_diarization):
    """Test 3: Full pipeline transcription."""
    print("=" * 50)
    print("TEST 3: Full Pipeline (ASR + Diarization)")
    print("=" * 50)

    result = transcribe_audio(
        audio_path,
        enable_diarization=enable_diarization,
    )

    text = result.get("text", "")
    segments = result.get("segments", [])
    speakers = result.get("speakers", [])

    print(f"\n  --- Results ---")
    print(f"  Full text ({len(text)} chars):")
    print(f"    '{text}'")
    print(f"\n  Speakers: {speakers}")
    print(f"  Segments: {len(segments)}")

    for seg in segments:
        print(f"    [{seg['speaker']}] [{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text']}")

    assert len(text) > 0, "No text transcribed!"
    print(f"\n  PASS: Transcription successful\n")
    return result


def test_expected_content(result):
    """Test 4: Check expected words are in the transcription."""
    print("=" * 50)
    print("TEST 4: Content Accuracy Check")
    print("=" * 50)

    text = result.get("text", "").lower()

    # These keywords should appear in a medical conversation
    expected_keywords = ["doctor", "appointment", "medicine", "pharmacy", "prescription"]
    found = [kw for kw in expected_keywords if kw in text]
    missed = [kw for kw in expected_keywords if kw not in text]

    print(f"  Found keywords: {found}")
    if missed:
        print(f"  Missed keywords: {missed}")

    accuracy = len(found) / len(expected_keywords) * 100
    print(f"  Keyword accuracy: {accuracy:.0f}%")

    if accuracy >= 60:
        print(f"  PASS: {accuracy:.0f}% keyword accuracy\n")
    else:
        print(f"  WARN: Low keyword accuracy ({accuracy:.0f}%)\n")


def main():
    # Find test audio
    audio_path = None
    do_diarize = True

    for arg in sys.argv[1:]:
        if arg == "--no-diarize":
            do_diarize = False
        elif os.path.isfile(arg):
            audio_path = arg

    if audio_path is None:
        # Try default test files
        for f in ["test_speech.wav", "test_audio.wav"]:
            if os.path.isfile(f):
                audio_path = f
                break

    if audio_path is None:
        print("ERROR: No audio file found.")
        print("Usage: python test_pipeline.py <audio_file>")
        sys.exit(1)

    print(f"\nTesting with: {audio_path}")
    print(f"Diarization: {'enabled' if do_diarize else 'disabled'}")
    print()

    # Run tests
    test_model_loading()
    test_vad(audio_path)
    result = test_transcription(audio_path, do_diarize)
    test_expected_content(result)

    print("=" * 50)
    print("ALL TESTS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
