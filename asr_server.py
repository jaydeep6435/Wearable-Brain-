"""
ASR Server — PC-side Whisper + Diarization HTTP Server
=======================================================
Lightweight HTTP server that accepts WAV files from the phone
and returns accurate transcriptions using Whisper + speaker
diarization using pyannote.

Runs on port 5050. No Flask — uses Python's built-in http.server.

Endpoints:
  POST /transcribe  — WAV → text transcript
  POST /diarize     — WAV → speaker segments
  POST /process     — WAV → transcript + speakers + merged output
  GET  /health      — server status check

Usage:
  python asr_server.py
"""

import io
import json
import os
import sys
import time
import tempfile
import wave
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.transcriber import transcribe_audio

# Try to import diarizer
DIARIZER_AVAILABLE = False
try:
    from diarization.diarizer import SpeakerDiarizer
    DIARIZER_AVAILABLE = True
except ImportError:
    print("[ASR Server] ⚠ Diarization not available (pyannote not installed)")

# ── Configuration ──────────────────────────────────────────

HOST = "0.0.0.0"  # Listen on all interfaces so phone can reach it
PORT = 5050
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB max

# Pre-load faster-whisper model at startup
print("=" * 50)
print("  ASR Server — Loading faster-whisper model...")
print("=" * 50)
try:
    from core.transcriber import _get_model
    _get_model()  # Pre-load
    WHISPER_READY = True
    print("[ASR Server] ✓ faster-whisper model loaded")
except Exception as e:
    WHISPER_READY = False
    print(f"[ASR Server] ✗ faster-whisper load failed: {e}")

# Pre-load diarizer
diarizer = None
if DIARIZER_AVAILABLE:
    try:
        diarizer = SpeakerDiarizer()
        print("[ASR Server] ✓ Diarizer initialized")
    except Exception as e:
        print(f"[ASR Server] ⚠ Diarizer init failed: {e}")
        diarizer = None


class ASRHandler(BaseHTTPRequestHandler):
    """Handle transcription and diarization requests."""

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[ASR Server] {args[0]}")

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _read_body(self):
        """Read request body."""
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_UPLOAD_SIZE:
            return None
        return self.rfile.read(length)

    def _save_temp_wav(self, data):
        """Save uploaded data to a temp WAV file."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=tempfile.gettempdir()
        )
        tmp.write(data)
        tmp.close()
        return tmp.name

    def _get_wav_info(self, wav_path):
        """Get WAV file metadata."""
        try:
            with wave.open(wav_path, "rb") as wf:
                return {
                    "channels": wf.getnchannels(),
                    "sample_rate": wf.getframerate(),
                    "frames": wf.getnframes(),
                    "duration": wf.getnframes() / wf.getframerate(),
                }
        except Exception:
            return {}

    # ── GET handlers ────────────────────────────────────────

    def do_GET(self):
        if self.path == "/health":
            self._set_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "whisper": WHISPER_READY,
                "diarizer": diarizer is not None,
            }).encode())
        else:
            self._set_headers(404)
            self.wfile.write(b'{"error": "not found"}')

    # ── POST handlers ───────────────────────────────────────

    def do_POST(self):
        if self.path == "/transcribe":
            self._handle_transcribe()
        elif self.path == "/diarize":
            self._handle_diarize()
        elif self.path == "/process":
            self._handle_process()
        else:
            self._set_headers(404)
            self.wfile.write(b'{"error": "not found"}')

    def _handle_transcribe(self):
        """Transcribe WAV using Whisper."""
        t0 = time.time()
        print("\n━━━ /transcribe ━━━━━━━━━━━━━━━━━━━━━━━")

        data = self._read_body()
        if not data:
            self._set_headers(400)
            self.wfile.write(b'{"error": "no data"}')
            return

        wav_path = self._save_temp_wav(data)
        info = self._get_wav_info(wav_path)
        print(f"  WAV: {len(data)//1024}KB, {info.get('duration', 0):.1f}s, "
              f"{info.get('sample_rate', 0)}Hz")

        try:
            result = transcribe_audio(wav_path)
            elapsed = time.time() - t0

            response = {
                "text": result["text"],
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip(),
                    }
                    for seg in result.get("segments", [])
                ],
                "duration": info.get("duration", 0),
                "elapsed": round(elapsed, 2),
            }

            print(f"  ✓ Transcribed in {elapsed:.1f}s: '{result['text'][:100]}'")
            self._set_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            print(f"  ✗ Error: {e}")
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
        finally:
            try:
                os.unlink(wav_path)
            except Exception:
                pass

    def _handle_diarize(self):
        """Diarize WAV using pyannote."""
        t0 = time.time()
        print("\n━━━ /diarize ━━━━━━━━━━━━━━━━━━━━━━━━━━")

        if diarizer is None:
            self._set_headers(503)
            self.wfile.write(b'{"error": "diarizer not available"}')
            return

        data = self._read_body()
        if not data:
            self._set_headers(400)
            self.wfile.write(b'{"error": "no data"}')
            return

        wav_path = self._save_temp_wav(data)

        try:
            segments = diarizer.diarize(wav_path)
            elapsed = time.time() - t0
            speakers = set(s["speaker"] for s in segments)

            print(f"  ✓ {len(segments)} segments, {len(speakers)} speakers in {elapsed:.1f}s")
            self._set_headers()
            self.wfile.write(json.dumps({
                "segments": segments,
                "speakers": list(speakers),
                "elapsed": round(elapsed, 2),
            }).encode())

        except Exception as e:
            print(f"  ✗ Error: {e}")
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
        finally:
            try:
                os.unlink(wav_path)
            except Exception:
                pass

    def _handle_process(self):
        """
        Full pipeline: VAD + Transcribe + Diarize + Merge.
        Uses the unified pipeline from core.transcriber.
        """
        t0 = time.time()
        print("\n=== /process (full pipeline) ===========")

        data = self._read_body()
        if not data:
            self._set_headers(400)
            self.wfile.write(b'{"error": "no data"}')
            return

        wav_path = self._save_temp_wav(data)
        info = self._get_wav_info(wav_path)
        print(f"  WAV: {len(data)//1024}KB, {info.get('duration', 0):.1f}s")

        try:
            # Use the unified pipeline (VAD + ASR + diarization + merge)
            result = transcribe_audio(wav_path, enable_diarization=(diarizer is not None))

            elapsed = time.time() - t0
            speakers = result.get("speakers", ["Speaker 1"])

            response = {
                "text": result["text"],
                "segments": result.get("segments", []),
                "speakers": speakers,
                "speaker_count": len(speakers),
                "duration": info.get("duration", 0),
                "timings": {
                    "total": round(elapsed, 2),
                },
            }

            print(f"  Done in {elapsed:.1f}s -- {len(speakers)} speakers")
            print(f"  Text: '{result['text'][:80]}'")
            print("========================================\n")

            self._set_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())
        finally:
            try:
                os.unlink(wav_path)
            except Exception:
                pass


def _merge_transcript_speakers(whisper_segments, diarize_segments):
    """
    Merge Whisper transcript segments with speaker labels from diarization.

    Each Whisper segment has {start, end, text}.
    Each diarize segment has {speaker, start, end}.

    For each Whisper segment, find the overlapping diarize segment
    and assign the speaker label.
    """
    if not diarize_segments:
        # No diarization — assign all to SPEAKER_0
        return [
            {
                "speaker": "SPEAKER_0",
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
            }
            for seg in whisper_segments
        ]

    merged = []
    for seg in whisper_segments:
        seg_mid = (seg.get("start", 0) + seg.get("end", 0)) / 2
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Find the diarize segment that covers the midpoint
        speaker = "SPEAKER_0"
        best_overlap = 0
        for dseg in diarize_segments:
            overlap_start = max(seg.get("start", 0), dseg.get("start", 0))
            overlap_end = min(seg.get("end", 0), dseg.get("end", 0))
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                speaker = dseg["speaker"]

        merged.append({
            "speaker": speaker,
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "text": text,
        })

    return merged


def main():
    server = HTTPServer((HOST, PORT), ASRHandler)
    print(f"\n{'=' * 50}")
    print(f"  ASR Server running on http://{HOST}:{PORT}")
    print(f"  Whisper: {'✓' if WHISPER_READY else '✗'}")
    print(f"  Diarizer: {'✓' if diarizer else '✗'}")
    print(f"{'=' * 50}")
    print(f"\n  Endpoints:")
    print(f"    POST /transcribe  — WAV → text")
    print(f"    POST /diarize     — WAV → speaker segments")
    print(f"    POST /process     — WAV → full pipeline")
    print(f"    GET  /health      — status check")
    print(f"\n  Waiting for requests...\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ASR Server] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
