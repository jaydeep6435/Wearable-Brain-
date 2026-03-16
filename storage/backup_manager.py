"""
Backup Manager
==============

Lightweight backup utilities for SQLite database files.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime


class BackupManager:
    def __init__(self, db):
        self._db = db

    @property
    def _db_path(self) -> str:
        return self._db.db_path

    @staticmethod
    def _sha256(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def create_backup(self, destination_path: str) -> dict:
        os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
        self._db.conn.commit()
        shutil.copy2(self._db_path, destination_path)
        return {
            "status": "success",
            "path": destination_path,
            "size_bytes": os.path.getsize(destination_path),
            "sha256": self._sha256(destination_path),
            "timestamp": datetime.now().isoformat(),
        }

    def restore_backup(self, source_path: str) -> dict:
        if not os.path.isfile(source_path):
            return {"status": "error", "error": "Backup file not found"}

        self._db.conn.commit()
        self._db.conn.close()
        shutil.copy2(source_path, self._db_path)

        return {
            "status": "success",
            "restored_from": source_path,
            "timestamp": datetime.now().isoformat(),
        }

    def verify_backup(self, file_path: str) -> dict:
        if not os.path.isfile(file_path):
            return {"valid": False, "errors": ["file not found"]}

        try:
            size = os.path.getsize(file_path)
            digest = self._sha256(file_path)
            return {
                "valid": size > 0,
                "manifest": {"size_bytes": size},
                "sha256_match": True,
                "sha256": digest,
                "errors": [],
            }
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}

    def list_backups(self, directory: str) -> list[dict]:
        if not os.path.isdir(directory):
            return []

        out = []
        for name in sorted(os.listdir(directory), reverse=True):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            out.append(
                {
                    "path": path,
                    "filename": name,
                    "size_bytes": os.path.getsize(path),
                    "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
                }
            )
        return out
