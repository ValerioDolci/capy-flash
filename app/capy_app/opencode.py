"""OpenCode: a provider config for the current endpoint, and a terminal that opens OpenCode in a project folder.

The app never edits the user's OpenCode config. It writes its own file (<config dir>/opencode.json) and launches
OpenCode with OPENCODE_CONFIG pointing at it; OpenCode merges that file over the user's global config, so their
settings, providers and keys stay as they are. The API key, if any, is not written into the config: it goes to a
0600 file referenced with {file:...}.

Format: OpenCode 1.x ("provider" / "npm" / "options", https://opencode.ai/v1/docs/providers/). OpenCode 2 changed
the schema ("providers" / "package" / "settings"); detect() reports the major version so the UI can say so.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths
from .store import write_json_atomic

PROVIDER_ID = "capy-flash"
DEFAULT_LIMIT = {"context": 131072, "output": 32768}


def find_binary() -> Optional[str]:
    found = shutil.which("opencode")
    if found:
        return found
    for candidate in (
        "~/.opencode/bin/opencode",
        "~/.npm-global/bin/opencode",
        "~/.local/bin/opencode",
        "/opt/homebrew/bin/opencode",
        "/usr/local/bin/opencode",
    ):
        p = os.path.expanduser(candidate)
        if os.access(p, os.X_OK):
            return p
    return None


def detect() -> Dict[str, Any]:
    binary = find_binary()
    if not binary:
        return {"installed": False}
    version = ""
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15)
        match = re.search(r"\d+\.\d+\.\d+", out.stdout + out.stderr)
        version = match.group(0) if match else ""
    except (OSError, subprocess.SubprocessError):
        pass
    major = int(version.split(".")[0]) if version else 0
    return {"installed": True, "binary": binary, "version": version, "supported": major == 1 or not version}


def config_path() -> Path:
    return paths.config_dir() / "opencode.json"


def key_path() -> Path:
    return paths.config_dir() / "opencode-api-key"


def build_config(
    backend_name: str,
    base_url: str,
    api_key: str,
    models: List[Dict[str, Any]],
    limits: Dict[str, Dict[str, int]],
    default_model: str = "",
) -> Dict[str, Any]:
    """OpenCode 1.x config with one provider holding the endpoint's chat models."""
    entries: Dict[str, Any] = {}
    for m in models:
        if m.get("kind", "chat") != "chat":
            continue
        limit = dict(DEFAULT_LIMIT)
        if m.get("context"):
            limit["context"] = int(m["context"])
        limit.update({k: int(v) for k, v in (limits.get(m["id"]) or {}).items() if k in ("context", "output")})
        entries[m["id"]] = {"name": m.get("name") or m["id"], "limit": limit}
    options: Dict[str, Any] = {"baseURL": base_url}
    if api_key:
        options["apiKey"] = "{file:" + str(key_path()) + "}"
    config: Dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": f"capy-flash ({backend_name})",
                "options": options,
                "models": entries,
            }
        },
    }
    if default_model and default_model in entries:
        config["model"] = f"{PROVIDER_ID}/{default_model}"
    return config


def write_config(config: Dict[str, Any], api_key: str) -> Path:
    paths.ensure_private_dir(paths.config_dir())
    if api_key:
        kp = key_path()
        fd = os.open(str(kp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(api_key)
        os.chmod(kp, 0o600)
    else:
        try:
            os.unlink(key_path())
        except FileNotFoundError:
            pass
    path = config_path()
    write_json_atomic(path, config, mode=0o600)
    return path


def launch_script(binary: str, folder: str, model: str) -> Path:
    """A small script that runs OpenCode in `folder` with our config merged in."""
    run = paths.ensure_private_dir(paths.runtime_dir())
    suffix = ".command" if sys.platform == "darwin" else ".sh"
    script = run / ("open-opencode" + suffix)
    args = [binary] + (["-m", f"{PROVIDER_ID}/{model}"] if model else [])
    body = "\n".join(
        [
            "#!/bin/sh",
            "# written by capy-flash app: OpenCode with the capy-flash provider (the user's config is merged, not edited)",
            f"cd {shlex.quote(folder)} || exit 1",
            f"export OPENCODE_CONFIG={shlex.quote(str(config_path()))}",
            "exec " + " ".join(shlex.quote(a) for a in args),
            "",
        ]
    )
    fd = os.open(str(script), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
    with os.fdopen(fd, "w") as f:
        f.write(body)
    os.chmod(script, 0o700)
    return script


def terminal_command(script: Path) -> List[str]:
    if sys.platform == "darwin":
        return ["open", "-a", "Terminal", str(script)]
    for term, args in (
        ("x-terminal-emulator", ["-e"]),
        ("gnome-terminal", ["--"]),
        ("konsole", ["-e"]),
        ("xfce4-terminal", ["-x"]),
        ("kitty", []),
        ("alacritty", ["-e"]),
        ("xterm", ["-e"]),
    ):
        found = shutil.which(term)
        if found:
            return [found] + args + [str(script)]
    raise RuntimeError("no terminal emulator found (tried x-terminal-emulator, gnome-terminal, konsole, xterm, ...)")


def open_in_terminal(folder: str, model: str) -> Dict[str, Any]:
    info = detect()
    if not info.get("installed"):
        raise RuntimeError("OpenCode is not installed (https://opencode.ai)")
    if not os.path.isdir(folder):
        raise ValueError("not a folder: " + folder)
    if not config_path().is_file():
        raise RuntimeError("configure OpenCode first")
    script = launch_script(info["binary"], folder, model)
    command = terminal_command(script)
    subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
    return {"script": str(script), "command": command}


def manual_snippet(config: Dict[str, Any]) -> str:
    """What to paste into ~/.config/opencode/opencode.json to use the provider without the app."""
    provider = json.loads(json.dumps(config["provider"]))
    options = provider[PROVIDER_ID]["options"]
    if "apiKey" in options:
        options["apiKey"] = "{env:CAPY_FLASH_API_KEY}"
    return json.dumps({"provider": provider}, indent=2)
