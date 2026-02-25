"""
Microphone Audio Source
=========================
AudioSource implementation that captures live audio from the
system's default microphone using sounddevice.

Usage:
    from audio.microphone import MicrophoneSource

    mic = MicrophoneSource(sample_rate=16000)
    mic.start()
    chunk = mic.read_chunk(480)   # 30ms at 16kHz
    mic.stop()

Requires: pip install sounddevice
"""

import numpy as np


class MicrophoneSource:
    """
    Live microphone input via sounddevice.

    Implements the AudioSource protocol for real-time mic capture.
    Uses sounddevice's InputStream for low-latency audio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | None = None,
    ):
        """
        Initialize microphone source.

        Args:
            sample_rate: Audio sample rate in Hz (16000 for Whisper).
            channels: Number of channels (1 = mono).
            device: Specific audio device index, or None for default.
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._device = device
        self._stream = None
        self._active = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def is_active(self) -> bool:
        return self._active and self._stream is not None

    def start(self) -> None:
        """Open the microphone stream."""
        if self._active:
            return

        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError(
                "sounddevice not installed. Run: pip install sounddevice"
            )

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                device=self._device,
            )
            self._stream.start()
            self._active = True
            print(f"[MicSource] Opened (rate={self._sample_rate}, device={self._device or 'default'})")
        except Exception as e:
            self._active = False
            self._stream = None
            raise RuntimeError(f"Failed to open microphone: {e}")

    def stop(self) -> None:
        """Close the microphone stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._active = False

    def read_chunk(self, num_samples: int) -> np.ndarray:
        """Read audio samples from the microphone."""
        if not self.is_active or self._stream is None:
            return np.array([], dtype=np.int16)

        try:
            audio_chunk, overflowed = self._stream.read(num_samples)
            if overflowed:
                return np.array([], dtype=np.int16)
            return audio_chunk
        except Exception:
            return np.array([], dtype=np.int16)

    def __repr__(self) -> str:
        status = "active" if self.is_active else "inactive"
        return f"MicrophoneSource(rate={self._sample_rate}, {status})"
