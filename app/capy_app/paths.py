"""Where the app keeps its settings, conversations and runtime files, per platform."""

import os
import sys
from pathlib import Path

APP_NAME = "capy-flash"


def _home() -> Path:
    return Path(os.path.expanduser("~"))


def config_dir() -> Path:
    """Settings (they may hold an API key: written 0600)."""
    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_CONFIG_HOME") or str(_home() / ".config")
    return Path(base) / APP_NAME


def data_dir() -> Path:
    """Conversations."""
    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(_home() / ".local" / "share")
    return Path(base) / APP_NAME


def runtime_dir() -> Path:
    """The running instance's port and token, and helper scripts (0700)."""
    if sys.platform == "darwin":
        return config_dir() / "run"
    base = os.environ.get("XDG_RUNTIME_DIR")
    return Path(base) / APP_NAME if base else config_dir() / "run"


def ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def app_root() -> Path:
    """The app/ directory (holds web/ and launch/)."""
    return Path(__file__).resolve().parent.parent
