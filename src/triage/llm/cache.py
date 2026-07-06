"""On-disk cache of model responses, one file per request key.

Makes eval reruns free and repeatable, and keeps a record of what each model was
asked and answered. The key comes from the request, so the same request always
maps to the same stored answer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class ResponseCache:
    """A simple key-value store on disk: one JSON file per key."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """Return the stored value for key, or None if it isn't cached."""
        path = self._path(key)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def set(self, key: str, payload: dict) -> None:
        """Store payload under key. Writes to a temp file then renames, so a
        crash mid-write can't leave a half-written file."""
        path = self._path(key)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        os.replace(tmp, path)
