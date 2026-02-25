"""
Audio Source Protocol
=======================
Abstract interface for audio input sources.

All audio sources must implement this protocol so that the
AudioWorker can accept input from any source: microphone,
file, Bluetooth earbuds, ESP32, etc.

Usage:
    from audio.source import AudioSource
    from audio.microphone import MicrophoneSource

    source: AudioSource = MicrophoneSource()
    source.start()
    chunk = source.read_chunk()
    source.stop()
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class AudioSource(Protocol):
    """
    Protocol for audio input sources.

    Any class implementing these methods can be used as an audio
    source for the AudioWorker. This enables hot-swapping between
    microphone, file playback, Bluetooth, or any future input.

    All audio data must be 16-bit mono PCM at the configured sample rate.
    """

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz (e.g., 16000)."""
        ...

    @property
    def channels(self) -> int:
        """Number of audio channels (1 = mono)."""
        ...

    @property
    def is_active(self) -> bool:
        """Whether the source is currently streaming audio."""
        ...

    def start(self) -> None:
        """
        Open the audio source and begin streaming.

        After calling start(), read_chunk() should return audio data.
        Raises RuntimeError if the source cannot be opened.
        """
        ...

    def stop(self) -> None:
        """
        Close the audio source and release resources.

        After calling stop(), read_chunk() should return empty arrays.
        Safe to call multiple times.
        """
        ...

    def read_chunk(self, num_samples: int) -> np.ndarray:
        """
        Read a chunk of audio samples.

        Args:
            num_samples: Number of samples to read.

        Returns:
            numpy array of int16 audio data, shape (num_samples, channels).
            Returns empty array if source is not active or no data available.
        """
        ...
