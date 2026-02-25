"""
Backup Manager — Secure Backup & Restore for Memory Database
==============================================================
Creates encrypted backup archives (.wbbak) of the SQLite database
with SHA-256 integrity verification.

Backup format:
  .wbbak = tar archive containing:
    - memory.db.enc (or memory.db for unencrypted)
    - manifest.json (sha256, timestamp, version, tables)

No plaintext exposure during backup/restore.
No cloud. No network. Fully offline.

Usage:
    mgr = BackupManager(db)
    mgr.create_backup("/path/to/backup.wbbak")
    mgr.verify_backup("/path/to/backup.wbbak")
    mgr.restore_backup("/path/to/backup.wbbak")
"""

import os
import json
import hashlib
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone


# Backup file extension
BACKUP_EXTENSION = ".wbbak"
MANIFEST_NAME = "manifest.json"
BACKUP_VERSION = "1.0.0"


class BackupManager:
    """
    Manages secure backup/restore of the memory database.

    Supports both encrypted (AES-256-GCM) and plaintext databases.
    Backups include SHA-256 integrity hashes and metadata manifests.
    """

    def __init__(self, db):
        """
        Initialize with a Database instance.

        Args:
            db: storage.db.Database instance.
        """
        self.db = db

    # ── Create Backup ─────────────────────────────────────────

    def create_backup(self, destination_path: str) -> dict:
        """
        Create a secure backup archive.

        Steps:
          1. WAL checkpoint (flush pending writes)
          2. Save encrypted state (if encrypted mode)
          3. Copy the database file
          4. Compute SHA-256 hash
          5. Build manifest with metadata
          6. Pack into .wbbak tar archive

        Args:
            destination_path: Where to write the backup file.
                              Will add .wbbak extension if missing.

        Returns:
            dict with: status, path, size_bytes, sha256, timestamp
        """
        if not destination_path.endswith(BACKUP_EXTENSION):
            destination_path += BACKUP_EXTENSION

        # Prevent overwriting without explicit path
        if os.path.exists(destination_path):
            return {
                "status": "error",
                "error": f"Backup file already exists: {destination_path}",
            }

        print(f"[Backup] Creating backup → {destination_path}")

        # Step 1: WAL checkpoint
        self._wal_checkpoint()

        # Step 2: Save encrypted (flush if encrypted)
        if self.db.is_encrypted:
            self.db.save_encrypted()

        # Step 3: Determine source file
        source_path = self._get_source_path()
        if not source_path or not os.path.isfile(source_path):
            return {
                "status": "error",
                "error": "Database file not found for backup",
            }

        # Step 4: Compute SHA-256
        sha256 = self._compute_sha256(source_path)

        # Step 5: Build manifest
        manifest = self._build_manifest(source_path, sha256)

        # Step 6: Pack into tar archive
        try:
            db_basename = os.path.basename(source_path)
            self._ensure_directory(destination_path)

            with tarfile.open(destination_path, "w:gz") as tar:
                # Add database file
                tar.add(source_path, arcname=db_basename)

                # Add manifest
                manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
                self._add_bytes_to_tar(tar, MANIFEST_NAME, manifest_bytes)

            size = os.path.getsize(destination_path)
            print(f"[Backup] Backup complete: {size:,} bytes, SHA-256: {sha256[:16]}...")

            return {
                "status": "success",
                "path": destination_path,
                "size_bytes": size,
                "sha256": sha256,
                "timestamp": manifest["timestamp"],
                "encrypted": manifest["encrypted"],
            }

        except Exception as e:
            # Clean up partial file
            if os.path.exists(destination_path):
                os.remove(destination_path)
            return {"status": "error", "error": str(e)}

    # ── Verify Backup ─────────────────────────────────────────

    def verify_backup(self, file_path: str) -> dict:
        """
        Verify the integrity of a backup file.

        Checks:
          - File exists and is a valid tar archive
          - Contains manifest.json
          - Contains database file
          - SHA-256 hash matches manifest

        Args:
            file_path: Path to the .wbbak backup file.

        Returns:
            dict with: valid (bool), manifest, sha256_match, errors
        """
        errors = []

        if not os.path.isfile(file_path):
            return {"valid": False, "errors": ["File not found"]}

        try:
            with tarfile.open(file_path, "r:gz") as tar:
                members = tar.getnames()

                # Check manifest exists
                if MANIFEST_NAME not in members:
                    errors.append("Missing manifest.json")
                    return {"valid": False, "errors": errors}

                # Read manifest
                manifest_file = tar.extractfile(MANIFEST_NAME)
                manifest = json.loads(manifest_file.read().decode("utf-8"))

                db_filename = manifest.get("db_filename")
                if not db_filename or db_filename not in members:
                    errors.append(f"Missing database file: {db_filename}")
                    return {"valid": False, "errors": errors}

                # Extract DB to temp and verify SHA-256
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tar.extract(db_filename, tmp_dir)
                    extracted_path = os.path.join(tmp_dir, db_filename)

                    actual_sha = self._compute_sha256(extracted_path)
                    expected_sha = manifest.get("sha256", "")

                    sha_match = actual_sha == expected_sha
                    if not sha_match:
                        errors.append(
                            f"SHA-256 mismatch: expected {expected_sha[:16]}..., "
                            f"got {actual_sha[:16]}..."
                        )

                return {
                    "valid": sha_match and len(errors) == 0,
                    "manifest": manifest,
                    "sha256_match": sha_match,
                    "errors": errors,
                }

        except tarfile.TarError as e:
            return {"valid": False, "errors": [f"Invalid archive: {e}"]}
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}

    # ── Restore Backup ────────────────────────────────────────

    def restore_backup(self, source_path: str) -> dict:
        """
        Restore a database from a backup archive.

        Steps:
          1. Verify backup integrity (SHA-256)
          2. Extract database file from archive
          3. Replace current database file
          4. Return status (caller must reconnect engine)

        WARNING: This overwrites the current database!

        Args:
            source_path: Path to the .wbbak backup file.

        Returns:
            dict with: status, restored_from, timestamp, needs_restart
        """
        # Step 1: Verify integrity
        verification = self.verify_backup(source_path)
        if not verification.get("valid"):
            return {
                "status": "error",
                "error": "Backup integrity check failed",
                "details": verification.get("errors", []),
            }

        manifest = verification["manifest"]
        db_filename = manifest["db_filename"]

        print(f"[Backup] Restoring from: {source_path}")
        print(f"[Backup]   Created: {manifest.get('timestamp', 'unknown')}")
        print(f"[Backup]   Encrypted: {manifest.get('encrypted', False)}")

        try:
            # Step 2: Extract to temp
            with tempfile.TemporaryDirectory() as tmp_dir:
                with tarfile.open(source_path, "r:gz") as tar:
                    tar.extract(db_filename, tmp_dir)

                extracted_path = os.path.join(tmp_dir, db_filename)

                # Step 3: Determine destination path
                dest_path = self._get_restore_destination(manifest)

                # Close current DB connection before replacing
                self.db.close()

                # Copy extracted file to destination
                self._ensure_directory(dest_path)
                shutil.copy2(extracted_path, dest_path)

            print(f"[Backup] Restored to: {dest_path}")

            return {
                "status": "success",
                "restored_from": source_path,
                "restored_to": dest_path,
                "timestamp": manifest.get("timestamp"),
                "encrypted": manifest.get("encrypted", False),
                "needs_restart": True,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── List Backups ──────────────────────────────────────────

    def list_backups(self, directory: str) -> list[dict]:
        """
        List all backup files in a directory.

        Args:
            directory: Directory to scan for .wbbak files.

        Returns:
            List of dicts with: path, size_bytes, modified.
        """
        backups = []
        if not os.path.isdir(directory):
            return backups

        for name in sorted(os.listdir(directory)):
            if name.endswith(BACKUP_EXTENSION):
                full_path = os.path.join(directory, name)
                stat = os.stat(full_path)
                backups.append({
                    "path": full_path,
                    "filename": name,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                })

        return backups

    # ── Internal Helpers ──────────────────────────────────────

    def _wal_checkpoint(self):
        """Flush WAL journal to main database file."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db._working_path)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        except Exception:
            pass  # Non-critical — WAL may not exist

    def _get_source_path(self) -> str:
        """Get the database file path to back up."""
        if self.db.is_encrypted:
            enc_path = self.db.db_path + ".enc"
            if os.path.isfile(enc_path):
                return enc_path
        # Fallback: working path (plaintext)
        return self.db._working_path

    def _get_restore_destination(self, manifest: dict) -> str:
        """Determine where to place the restored database file."""
        if manifest.get("encrypted"):
            return self.db.db_path + ".enc"
        return self.db.db_path

    def _build_manifest(self, source_path: str, sha256: str) -> dict:
        """Build the backup manifest with metadata."""
        stats = self.db.get_stats()
        return {
            "version": BACKUP_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "db_filename": os.path.basename(source_path),
            "sha256": sha256,
            "size_bytes": os.path.getsize(source_path),
            "encrypted": self.db.is_encrypted,
            "tables": stats,
        }

    @staticmethod
    def _compute_sha256(file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _add_bytes_to_tar(tar: tarfile.TarFile, name: str, data: bytes):
        """Add raw bytes as a file to a tar archive."""
        import io
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    @staticmethod
    def _ensure_directory(file_path: str):
        """Ensure the parent directory exists."""
        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
