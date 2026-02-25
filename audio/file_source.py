"""
File Audio Source
===================
AudioSource implementation that reads audio from a WAV file.

Used for:
- Testing the pipeline without a real microphone
- Replaying saved recordings through the pipeline
- Simulating real-time audio from pre-recorded files

Usage:
    from audio.file_source import FileSource

    source = FileSource("recording.wav")
    source.start()
    while source.is_active:
        chunk = source.read_chunk(480)
        # process chunk...
    source.stop()
"""

import os
import wave
import numpy as np


class FileSource:
    """
    Audio source that reads from a WAV file.

    Implements the AudioSource protocol, delivering audio data
    from a file as if it were a live stream. Useful for testing
    and replaying saved recordings.
    """

    def __init__(self, file_path: str, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize file source.

        Args:
            file_path: Path to WAV file.
            sample_rate: Expected sample rate (will resample if mismatched).
            channels: Expected channels (1 = mono).
        """
        self._file_path = file_path
        self._sample_rate = sample_rate
        self._channels = channels
        self._audio_data: np.ndarray | None = None
        self._position = 0
        self._active = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def is_active(self) -> bool:
        return self._active and self._audio_data is not None and self._position < len(self._audio_data)

    def start(self) -> None:
        """Load the WAV file into memory."""
        if self._active:
            return

        if not os.path.isfile(self._file_path):
            raise RuntimeError(f"Audio file not found: {self._file_path}")

        try:
            with wave.open(self._file_path, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()

                raw_data = wf.readframes(n_frames)

            # Convert to numpy int16
            if sampwidth == 2:
                audio = np.frombuffer(raw_data, dtype=np.int16)
            elif sampwidth == 1:
                audio = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.int16) - 128) * 256
            else:
                raise RuntimeError(f"Unsupported sample width: {sampwidth}")

            # Convert stereo to mono if needed
            if n_channels > 1 and self._channels == 1:
                audio = audio.reshape(-1, n_channels)[:, 0]

            # Reshape for consistency
            audio = audio.reshape(-1, self._channels)

            self._audio_data = audio
            self._position = 0
            self._active = True

            duration = len(audio) / framerate
            print(f"[FileSource] Loaded: {self._file_path} ({duration:.1f}s, {framerate}Hz)")

        except Exception as e:
            self._active = False
            raise RuntimeError(f"Failed to load audio file: {e}")

    def stop(self) -> None:
        """Release the audio data."""
        self._active = False
        self._audio_data = None
        self._position = 0

    def read_chunk(self, num_samples: int) -> np.ndarray:
        """Read next chunk of samples from the file."""
        if not self.is_active or self._audio_data is None:
            return np.array([], dtype=np.int16)

        end = min(self._position + num_samples, len(self._audio_data))
        chunk = self._audio_data[self._position:end]
        self._position = end

        # If we've consumed all data, mark as inactive
        if self._position >= len(self._audio_data):
            self._active = False

        return chunk

    @property
    def remaining_samples(self) -> int:
        """Number of unread samples remaining."""
        if self._audio_data is None:
            return 0
        return max(0, len(self._audio_data) - self._position)

    @property
    def total_samples(self) -> int:
        """Total number of samples in the file."""
        if self._audio_data is None:
            return 0
        return len(self._audio_data)

    def reset(self) -> None:
        """Reset read position to the beginning."""
        self._position = 0
        if self._audio_data is not None:
            self._active = True

    def __repr__(self) -> str:
        status = "active" if self.is_active else "inactive"
        return f"FileSource({os.path.basename(self._file_path)}, {status})"
