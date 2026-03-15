"""
Bluetooth Audio Source — Push-Based Wearable Input
====================================================
AudioSource implementation for Bluetooth/BLE audio devices.

Architecture:
  Flutter app connects to Bluetooth device (earbuds, ESP32, etc.)
  → Flutter receives raw PCM audio frames
  → Flutter pushes frames via MethodChannel to Python
  → BluetoothAudioSource buffers frames in a thread-safe ring buffer
  → AudioWorker reads frames via read_chunk()

Python is transport-agnostic — it only receives PCM data.
No Bluetooth stack in Python. No network. Fully offline.

Usage:
    source = BluetoothAudioSource(sample_rate=16000)
    source.start()

    # Flutter pushes frames:
    source.push_audio(pcm_bytes)

    # AudioWorker reads:
    chunk = source.read_chunk(480)

    source.stop()
"""

import threading
import time
import numpy as np


# Default ring buffer: 30 seconds of 16kHz mono int16
DEFAULT_BUFFER_SECONDS = 30


class BluetoothAudioSource:
    """
    Push-based audio source for Bluetooth/BLE wearable devices.

    Implements the AudioSource protocol. Audio is pushed in from
    an external source (Flutter MethodChannel) and read out by
    the AudioWorker.

    Thread-safe: push_audio() and read_chunk() can be called
    from different threads simultaneously.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        buffer_seconds: float = DEFAULT_BUFFER_SECONDS,
        device_name: str = "Bluetooth Device",
    ):
        """
        Initialize Bluetooth audio source.

        Args:
            sample_rate: Audio sample rate in Hz (16000 for Whisper).
            channels: Number of channels (1 = mono).
            buffer_seconds: Ring buffer size in seconds.
            device_name: Human-readable device name for logging.
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._device_name = device_name
        self._active = False
        self._connected = False

        # Ring buffer (circular, thread-safe)
        buffer_samples = int(sample_rate * buffer_seconds)
        self._buffer = np.zeros(buffer_samples, dtype=np.int16)
        self._buffer_size = buffer_samples
        self._write_pos = 0
        self._read_pos = 0
        self._available = 0  # Samples available to read
        self._lock = threading.Lock()

        # Statistics
        self._total_pushed = 0
        self._total_read = 0
        self._overflows = 0
        self._last_push_time: float | None = None

    # ── AudioSource Protocol ──────────────────────────────────

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Connection State ──────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """Whether the Bluetooth device is currently connected."""
        return self._connected

    @property
    def device_name(self) -> str:
        """Human-readable name of the connected device."""
        return self._device_name

    def set_connected(self, connected: bool, device_name: str = None) -> None:
        """
        Update connection state (called by Flutter bridge).

        Args:
            connected: Whether a device is connected.
            device_name: Optional name of the connected device.
        """
        self._connected = connected
        if device_name:
            self._device_name = device_name

        if connected:
            print(f"[BTSource] Device connected: {self._device_name}")
        else:
            print(f"[BTSource] Device disconnected: {self._device_name}")

    def start(self) -> None:
        """Activate the source for reading."""
        if self._active:
            return

        self._active = True
        self._clear_buffer()
        print(f"[BTSource] Started (rate={self._sample_rate}, device={self._device_name})")

    def stop(self) -> None:
        """Deactivate the source and clear buffer."""
        self._active = False
        self._connected = False
        self._clear_buffer()
        print(f"[BTSource] Stopped")

    def read_chunk(self, num_samples: int) -> np.ndarray:
        """
        Read audio samples from the ring buffer.

        Returns available data (may be less than num_samples).
        Returns empty array if source is not active.
        """
        if not self._active:
            return np.array([], dtype=np.int16)

        with self._lock:
            if self._available == 0:
                return np.array([], dtype=np.int16)

            # Read up to num_samples
            to_read = min(num_samples, self._available)
            result = np.empty(to_read, dtype=np.int16)

            # Handle wrap-around
            first_chunk = min(to_read, self._buffer_size - self._read_pos)
            result[:first_chunk] = self._buffer[
                self._read_pos : self._read_pos + first_chunk
            ]

            if first_chunk < to_read:
                remainder = to_read - first_chunk
                result[first_chunk:] = self._buffer[:remainder]

            self._read_pos = (self._read_pos + to_read) % self._buffer_size
            self._available -= to_read
            self._total_read += to_read

        return result.reshape(-1, 1) if self._channels == 1 else result

    # ── Push Interface (for Flutter bridge) ───────────────────

    def push_audio(self, pcm_data: bytes | np.ndarray) -> int:
        """
        Push raw PCM audio frames into the buffer.

        Called by Flutter via MethodChannel when Bluetooth audio
        frames arrive.

        Args:
            pcm_data: Raw PCM bytes (int16 little-endian) or numpy array.

        Returns:
            Number of samples written.
        """
        if not self._active:
            return 0

        # Convert to numpy
        if isinstance(pcm_data, bytes):
            samples = np.frombuffer(pcm_data, dtype=np.int16)
        elif isinstance(pcm_data, np.ndarray):
            samples = pcm_data.astype(np.int16).ravel()
        else:
            return 0

        n = len(samples)
        if n == 0:
            return 0

        with self._lock:
            # Check for overflow
            space = self._buffer_size - self._available
            if n > space:
                # Buffer overflow — drop oldest data
                overflow = n - space
                self._read_pos = (self._read_pos + overflow) % self._buffer_size
                self._available -= overflow
                self._overflows += 1

            # Write samples (handle wrap-around)
            first_chunk = min(n, self._buffer_size - self._write_pos)
            self._buffer[self._write_pos : self._write_pos + first_chunk] = (
                samples[:first_chunk]
            )

            if first_chunk < n:
                remainder = n - first_chunk
                self._buffer[:remainder] = samples[first_chunk:]

            self._write_pos = (self._write_pos + n) % self._buffer_size
            self._available += n
            self._total_pushed += n
            self._last_push_time = time.time()

            # DEBUG TELEMETRY
            if n > 0:
                rms = np.sqrt(np.mean(samples.astype(np.float64)**2))
                max_amp = np.max(np.abs(samples))
                # Only log every ~1 second to avoid spam (at 16kHz, ~16000 samples)
                if self._total_pushed % 16000 < n:
                    print(f"[BTSource][DEBUG] Pushed {n} samples. Total: {self._total_pushed}. "
                          f"RMS: {rms:.2f}, Max Amp: {max_amp}, Buffer Available: {self._available}")

        return n

    # ── Internal ──────────────────────────────────────────────

    def _clear_buffer(self) -> None:
        """Reset the ring buffer."""
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._available = 0

    def get_stats(self) -> dict:
        """Get source statistics."""
        return {
            "device": self._device_name,
            "connected": self._connected,
            "active": self._active,
            "sample_rate": self._sample_rate,
            "buffer_available": self._available,
            "buffer_capacity": self._buffer_size,
            "total_pushed": self._total_pushed,
            "total_read": self._total_read,
            "overflows": self._overflows,
            "last_push_time": self._last_push_time,
        }

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        conn = "connected" if self._connected else "disconnected"
        return f"BluetoothAudioSource({self._device_name}, {status}, {conn})"
