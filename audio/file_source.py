"""
File Audio Source
=================

AudioSource-compatible WAV file reader for replay/testing.
"""

from __future__ import annotations

import wave
import numpy as np


class FileSource:
    def __init__(self, file_path: str, sample_rate: int = 16000):
        self._file_path = file_path
        self._sample_rate = sample_rate
        self._channels = 1
        self._active = False
        self._wf = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self) -> None:
        if self._active:
            return
        self._wf = wave.open(self._file_path, "rb")
        self._channels = self._wf.getnchannels()
        self._sample_rate = self._wf.getframerate()
        self._active = True

    def stop(self) -> None:
        if self._wf is not None:
            try:
                self._wf.close()
            except Exception:
                pass
        self._wf = None
        self._active = False

    def read_chunk(self, num_samples: int) -> np.ndarray:
        if not self._active or self._wf is None:
            return np.array([], dtype=np.int16)

        raw = self._wf.readframes(num_samples)
        if not raw:
            self.stop()
            return np.array([], dtype=np.int16)

        arr = np.frombuffer(raw, dtype=np.int16)
        if self._channels > 1:
            arr = arr.reshape(-1, self._channels)
        else:
            arr = arr.reshape(-1, 1)
        return arr
