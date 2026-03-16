import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.transcriber import transcribe_audio


def run(path: str):
    print(f"=== {path} ===")
    result = transcribe_audio(path, enable_diarization=True)
    speakers = result.get("speakers", [])
    segments = result.get("segments", [])
    print("SPEAKERS", speakers)
    print("COUNT", len(speakers))
    print("SEGMENTS", len(segments))
    for s in segments:
        print(f"[{s['speaker']}] {s['start']}->{s['end']} {s['text']}")
    print()


run(r"test_multi_speaker.wav")
run(r"project test.wav")
run(r"wearable brain testing 1 .wav")
