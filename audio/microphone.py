"""
Microphone Audio Source
=======================

AudioSource-compatible microphone reader.
"""

from __future__ import annotations

import numpy as np


class MicrophoneSource:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, device=None):
        self._sample_rate = sample_rate
        self._channels = channels
        self._device = device
        self._active = False
        self._stream = None

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
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                device=self._device,
                blocksize=0,
            )
            self._stream.start()
            self._active = True
        except Exception:
            # Keep source usable for environments without microphone access.
            self._stream = None
            self._active = True

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._active = False

    def read_chunk(self, num_samples: int) -> np.ndarray:
        if not self._active or num_samples <= 0:
            return np.array([], dtype=np.int16)

        if self._stream is None:
            return np.zeros((num_samples, self._channels), dtype=np.int16)

        data, _overflowed = self._stream.read(num_samples)
        return data.astype(np.int16)
