import os
import sys

print('STEP 1: start', flush=True)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print('STEP 2: path set', flush=True)

print('STEP 3: importing diarizer module', flush=True)
import diarization.diarizer as dz
print('STEP 4: diarizer imported', flush=True)

print('STEP 5: loading pipeline class', flush=True)
cls = dz._get_pyannote_pipeline_class()
print('STEP 6: pipeline class loaded?', cls is not None, flush=True)

print('STEP 7: creating diarizer', flush=True)
d = dz.SpeakerDiarizer()
print('STEP 8: diarizer available?', d.is_available, flush=True)
