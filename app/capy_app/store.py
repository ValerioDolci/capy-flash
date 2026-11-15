"""Settings and conversations on disk: small JSON files, written atomically."""

import json
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths

DEFAULT_SETTINGS: Dict[str, Any] = {
    # kind "endpoint": a server somewhere (URL + key). kind "local": the engine this app runs on this computer
    # (the URL is then http://127.0.0.1:<engine.port>/v1); only /api/engine/use sets it.
    "backend": {
        "kind": "endpoint",
        "name": "capy-flash",
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "",
    },
    "model": "",
    "thinking": True,
    "temperature": None,
    "max_tokens": None,
    "system_prompt": "",
    "theme": "dark",
    "language": "auto",
    "recent_folders": [],
    # Per-model context/output limits for OpenCode (the endpoint does not always report them).
    "model_limits": {},
    # The engine on this computer: the catalog model to run, NINFER_PORT, whether it starts with the app, and
    # the install folder (empty: ~/capy-flash, as install.sh). The folder decides which launcher is executed, so
    # the page cannot change it: capy-app --engine-dir.
    "engine": {"model": "", "port": 8080, "autostart": True, "dir": ""},
    "setup_done": False,
}

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def write_json_atomic(path: Path, value: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=1)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


class Settings:
    def __init__(self, directory: Optional[Path] = None):
        self.path = (directory or paths.config_dir()) / "app.json"
        self._lock = threading.Lock()
        self._value = self._load()

    def _load(self) -> Dict[str, Any]:
        self.stored = set()  # the keys the file on disk had (older versions lack the newer ones)
        try:
            with open(self.path, encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                self.stored = set(stored)
                return _merge(DEFAULT_SETTINGS, stored)
        except (OSError, ValueError):
            pass
        return json.loads(json.dumps(DEFAULT_SETTINGS))

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._value))

    def update(self, change: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(change) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError("unknown settings: " + ", ".join(sorted(unknown)))
        with self._lock:
            self._value = _merge(self._value, change)
            write_json_atomic(self.path, self._value)
            return json.loads(json.dumps(self._value))

    def public(self) -> Dict[str, Any]:
        """The settings the page may see: the API key is reduced to whether it is set."""
        value = self.get()
        key = value["backend"].pop("api_key", "")
        value["backend"]["has_api_key"] = bool(key)
        return value

    def existed(self) -> bool:
        return self.path.exists()


class Conversations:
    def __init__(self, directory: Optional[Path] = None):
        self.dir = (directory or paths.data_dir()) / "conversations"
        self._lock = threading.Lock()

    def _path(self, conversation_id: str) -> Path:
        if not _ID.match(conversation_id or ""):
            raise ValueError("bad conversation id")
        return self.dir / (conversation_id + ".json")

    def list(self) -> List[Dict[str, Any]]:
        out = []
        if not self.dir.is_dir():
            return out
        for p in self.dir.glob("*.json"):
            try:
                with open(p, encoding="utf-8") as f:
                    c = json.load(f)
                out.append(
                    {
                        "id": c.get("id", p.stem),
                        "title": c.get("title", ""),
                        "model": c.get("model", ""),
                        "updated": c.get("updated", 0),
                        "messages": len(c.get("messages", [])),
                    }
                )
            except (OSError, ValueError):
                continue
        out.sort(key=lambda c: c["updated"], reverse=True)
        return out

    def get(self, conversation_id: str) -> Dict[str, Any]:
        with open(self._path(conversation_id), encoding="utf-8") as f:
            return json.load(f)

    def save(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(conversation, dict) or not isinstance(conversation.get("messages", []), list):
            raise ValueError("a conversation is an object with a messages list")
        conversation = dict(conversation)
        conversation.setdefault("id", uuid.uuid4().hex[:16])
        now = time.time()
        conversation.setdefault("created", now)
        conversation["updated"] = now
        with self._lock:
            write_json_atomic(self._path(conversation["id"]), conversation)
        return conversation

    def delete(self, conversation_id: str) -> None:
        with self._lock:
            try:
                os.unlink(self._path(conversation_id))
            except FileNotFoundError:
                pass
