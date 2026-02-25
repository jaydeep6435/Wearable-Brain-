"""
Database Encryption — AES-256-GCM File-Level Encryption
=========================================================
Encrypts the SQLite database file at rest using AES-256-GCM.

Key derivation: PBKDF2-HMAC-SHA256, 480,000 iterations.
File format: [16-byte salt][12-byte nonce][ciphertext][16-byte GCM tag]

The passphrase is NEVER stored, logged, or hardcoded.
It must be provided via:
  1. MEMORY_ASSISTANT_KEY environment variable
  2. config.json file ({"encryption_key": "..."})
  3. None → unencrypted mode (backward compatible)

Usage:
    enc = DatabaseEncryption("my-secure-passphrase")
    enc.encrypt_file("memory.db", "memory.db.enc")
    enc.decrypt_file("memory.db.enc", "memory.db.tmp")
"""

import os
import json

# Lazy import — only loaded when encryption is actually used
_cryptography_available = None


def _check_cryptography():
    """Check if cryptography library is installed."""
    global _cryptography_available
    if _cryptography_available is None:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            _cryptography_available = True
        except ImportError:
            _cryptography_available = False
    return _cryptography_available


# ── Constants ──────────────────────────────────────────────────

SALT_LENGTH = 16
NONCE_LENGTH = 12
KDF_ITERATIONS = 480_000
KEY_LENGTH = 32  # AES-256

# Magic bytes for encrypted file detection
ENCRYPTED_HEADER = b"WBENC1"  # Wearable Brain ENCrypted v1


class DatabaseEncryption:
    """
    AES-256-GCM file encryption for SQLite databases.

    Thread-safe: each operation is self-contained.
    No state is retained between encrypt/decrypt calls.
    """

    def __init__(self, passphrase: str):
        """
        Initialize with a passphrase.

        Args:
            passphrase: Secret passphrase for key derivation.
                        NEVER log or store this value.
        """
        if not passphrase:
            raise ValueError("Passphrase cannot be empty")

        if not _check_cryptography():
            raise ImportError(
                "cryptography library required for encryption. "
                "Install with: pip install cryptography"
            )

        # Store passphrase bytes — never log
        self._passphrase = passphrase.encode("utf-8")

    def derive_key(self, salt: bytes) -> bytes:
        """
        Derive a 256-bit encryption key from passphrase + salt.

        Uses PBKDF2-HMAC-SHA256 with 480,000 iterations.

        Args:
            salt: 16-byte random salt.

        Returns:
            32-byte derived key.
        """
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_LENGTH,
            salt=salt,
            iterations=KDF_ITERATIONS,
        )
        return kdf.derive(self._passphrase)

    def encrypt_file(self, src_path: str, dst_path: str) -> None:
        """
        Encrypt a file using AES-256-GCM.

        Output format: [WBENC1][salt][nonce][ciphertext+tag]

        Args:
            src_path: Path to plaintext file.
            dst_path: Path to write encrypted file.
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Source file not found: {src_path}")

        # Read plaintext
        with open(src_path, "rb") as f:
            plaintext = f.read()

        # Generate salt and nonce
        salt = os.urandom(SALT_LENGTH)
        nonce = os.urandom(NONCE_LENGTH)

        # Derive key and encrypt
        key = self.derive_key(salt)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Write: header + salt + nonce + ciphertext (includes GCM tag)
        dst_dir = os.path.dirname(dst_path)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)

        with open(dst_path, "wb") as f:
            f.write(ENCRYPTED_HEADER)
            f.write(salt)
            f.write(nonce)
            f.write(ciphertext)

    def decrypt_file(self, src_path: str, dst_path: str) -> None:
        """
        Decrypt a file encrypted with encrypt_file().

        Args:
            src_path: Path to encrypted file.
            dst_path: Path to write decrypted file.

        Raises:
            ValueError: If file is not encrypted or has wrong format.
            cryptography.exceptions.InvalidTag: If passphrase is wrong.
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Encrypted file not found: {src_path}")

        with open(src_path, "rb") as f:
            data = f.read()

        header_len = len(ENCRYPTED_HEADER)

        # Validate header
        if len(data) < header_len + SALT_LENGTH + NONCE_LENGTH:
            raise ValueError("File too small to be encrypted")

        if data[:header_len] != ENCRYPTED_HEADER:
            raise ValueError("Not a valid encrypted database file")

        # Extract components
        offset = header_len
        salt = data[offset:offset + SALT_LENGTH]
        offset += SALT_LENGTH
        nonce = data[offset:offset + NONCE_LENGTH]
        offset += NONCE_LENGTH
        ciphertext = data[offset:]

        # Derive key and decrypt
        key = self.derive_key(salt)
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        # Write decrypted file
        dst_dir = os.path.dirname(dst_path)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)

        with open(dst_path, "wb") as f:
            f.write(plaintext)

    @staticmethod
    def is_encrypted(file_path: str) -> bool:
        """
        Check if a file is an encrypted database.

        Args:
            file_path: Path to file to check.

        Returns:
            True if file starts with the encrypted header.
        """
        if not os.path.isfile(file_path):
            return False

        try:
            with open(file_path, "rb") as f:
                header = f.read(len(ENCRYPTED_HEADER))
            return header == ENCRYPTED_HEADER
        except Exception:
            return False

    @staticmethod
    def is_sqlite(file_path: str) -> bool:
        """
        Check if a file is a plaintext SQLite database.

        Args:
            file_path: Path to file to check.

        Returns:
            True if file starts with SQLite magic bytes.
        """
        if not os.path.isfile(file_path):
            return False

        try:
            with open(file_path, "rb") as f:
                header = f.read(16)
            return header[:6] == b"SQLite"
        except Exception:
            return False


# ── Passphrase Resolution ──────────────────────────────────────

def get_passphrase() -> str | None:
    """
    Get the encryption passphrase from environment or config.

    Resolution order:
      1. MEMORY_ASSISTANT_KEY environment variable
      2. config.json in project root ({"encryption_key": "..."})
      3. None (unencrypted mode)

    Returns:
        Passphrase string, or None if no key is configured.
    """
    # 1. Environment variable
    key = os.environ.get("MEMORY_ASSISTANT_KEY")
    if key:
        return key

    # 2. Config file
    config_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.json"),
    ]

    for config_path in config_paths:
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                key = config.get("encryption_key")
                if key:
                    return key
            except (json.JSONDecodeError, IOError):
                pass

    # 3. No key configured → unencrypted mode
    return None
