import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diarization.diarizer import SpeakerDiarizer

print('STEP A: create diarizer', flush=True)
d = SpeakerDiarizer()
print('STEP B: available', d.is_available, flush=True)
print('STEP C: diarize start', flush=True)
segs = d.diarize('project test.wav')
print('STEP D: diarize done', len(segs), flush=True)
print(sorted(set(s['speaker'] for s in segs)), flush=True)
