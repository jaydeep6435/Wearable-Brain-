import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from diarization.diarizer import SpeakerDiarizer

print("PY_EXE", sys.executable)
print("PY_VER", sys.version)

d = SpeakerDiarizer()
print("PYANNOTE_AVAILABLE", d.is_available)
