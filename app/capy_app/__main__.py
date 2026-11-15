"""Start the app (or bring up the running one) and open its window.

python3 -m capy_app                 start, or open the window of the instance already running
python3 -m capy_app --install       add a launcher: ~/Applications/capy-flash.app (macOS) or a .desktop entry (Linux)
python3 -m capy_app --no-window     serve only and print the URL (e.g. to open it by hand)
python3 -m capy_app --engine-dir D  where the engine and the models are installed (default ~/capy-flash)
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import urllib.request
import webbrowser
from pathlib import Path

from . import __version__, paths
from .local import discover_endpoints
from .security import COOKIE, new_token
from .server import bind, make_app
from .store import Conversations, Settings, write_json_atomic

INSTANCE = "instance.json"


def _instance_file() -> Path:
    return paths.ensure_private_dir(paths.runtime_dir()) / INSTANCE


def running_instance():
    """The URL of an instance that answers, or None."""
    try:
        with open(_instance_file(), encoding="utf-8") as f:
            info = json.load(f)
        os.kill(int(info["pid"]), 0)
        req = urllib.request.Request(
            f"http://127.0.0.1:{info['port']}/api/state", headers={"Cookie": f"{COOKIE}={info['token']}"}
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            if r.status == 200:
                return info["url"]
    except Exception:
        return None
    return None


def discover_endpoint():
    """First run: the first local endpoint whose /v1/models answers without a key, as backend settings (or None)."""
    found = discover_endpoints()
    if not found:
        return None
    return {k: found[0][k] for k in ("kind", "name", "base_url", "api_key")}


def open_window(url: str) -> None:
    """A browser window without tabs or address bar when a Chromium browser is there; else the default browser."""
    if sys.platform == "darwin":
        for name in ("Google Chrome", "Chromium", "Microsoft Edge", "Brave Browser"):
            if os.path.isdir(f"/Applications/{name}.app") or os.path.isdir(
                os.path.expanduser(f"~/Applications/{name}.app")
            ):
                subprocess.Popen(["open", "-na", name, "--args", f"--app={url}"])
                return
        subprocess.Popen(["open", url])
        return
    for name in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
        "brave-browser",
    ):
        found = shutil.which(name)
        if found:
            subprocess.Popen(
                [found, f"--app={url}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
            )
            return
    webbrowser.open(url)


def _copy_app(dest: Path) -> Path:
    """A copy of the app (code, page, launcher) at `dest`, so the installed app does not depend on where this
    checkout or tarball lives. Re-installing replaces it; settings and conversations live elsewhere."""
    src = paths.app_root()
    if src.resolve() == dest.resolve():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
    (dest / "launch" / "capy-app").chmod(0o755)
    return dest


def install() -> str:
    if sys.platform == "darwin":
        bundle = Path(os.path.expanduser("~/Applications/capy-flash.app"))
        macos = bundle / "Contents" / "MacOS"
        resources = bundle / "Contents" / "Resources"
        macos.mkdir(parents=True, exist_ok=True)
        resources.mkdir(parents=True, exist_ok=True)
        root = _copy_app(resources / "app")
        launcher = root / "launch" / "capy-app"
        icon = root / "web" / "icon.svg"
        exe = macos / "capy-flash"
        exe.write_text(f'#!/bin/sh\nexec "{launcher}" "$@"\n')
        exe.chmod(0o755)
        icon_name = _mac_icon(icon, resources)
        plist = {
            "CFBundleName": "capy-flash",
            "CFBundleDisplayName": "capy-flash",
            "CFBundleIdentifier": "io.github.valeriodolci.capy-flash.app",
            "CFBundleVersion": __version__,
            "CFBundleShortVersionString": __version__,
            "CFBundleExecutable": "capy-flash",
            "CFBundlePackageType": "APPL",
            "LSUIElement": True,
        }
        if icon_name:
            plist["CFBundleIconFile"] = icon_name
        import plistlib

        with open(bundle / "Contents" / "Info.plist", "wb") as f:
            plistlib.dump(plist, f)
        return str(bundle)
    data = Path(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"))
    root = _copy_app(data / "capy-flash" / "app")
    launcher = root / "launch" / "capy-app"
    icon = root / "web" / "icon.svg"
    entry = "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            "Name=capy-flash",
            "Comment=Chat with capy-flash models",
            f'Exec="{launcher}"',
            f"Icon={icon}",
            "Terminal=false",
            "Categories=Development;Utility;",
            "",
        ]
    )
    base = data / "applications"
    base.mkdir(parents=True, exist_ok=True)
    target = base / "capy-flash.desktop"
    target.write_text(entry)
    target.chmod(0o755)
    desktop = Path(os.path.expanduser("~/Desktop"))
    if desktop.is_dir():
        copy = desktop / "capy-flash.desktop"
        copy.write_text(entry)
        copy.chmod(0o755)
        if shutil.which("gio"):
            subprocess.run(["gio", "set", str(copy), "metadata::trusted", "true"], capture_output=True)
    return str(target)


def _mac_icon(svg: Path, resources: Path):
    """Best effort: SVG -> .icns with the tools every macOS has (qlmanage, sips, iconutil)."""
    if not svg.is_file():
        return None
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["qlmanage", "-t", "-s", "1024", "-o", tmp, str(svg)], capture_output=True, timeout=60)
            png = Path(tmp) / (svg.name + ".png")
            if not png.is_file():
                return None
            iconset = Path(tmp) / "icon.iconset"
            iconset.mkdir()
            for size in (16, 32, 128, 256, 512):
                for scale in (1, 2):
                    px = size * scale
                    name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                    subprocess.run(
                        ["sips", "-z", str(px), str(px), str(png), "--out", str(iconset / name)],
                        capture_output=True,
                        timeout=60,
                    )
            out = resources / "icon.icns"
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], capture_output=True, timeout=60)
            return "icon" if out.is_file() else None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(prog="capy-app", description="capy-flash app " + __version__)
    ap.add_argument("--install", action="store_true", help="add a launcher to the applications menu / folder")
    ap.add_argument("--no-window", action="store_true", help="do not open a window; print the URL")
    ap.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: a free one)")
    ap.add_argument(
        "--engine-dir",
        metavar="DIR",
        help="where the engine and the models are installed (default ~/capy-flash, as install.sh); remembered",
    )
    a = ap.parse_args()

    if a.install:
        print("installed:", install())
        return 0

    url = running_instance()
    if url:
        if a.no_window:
            print(url)
        else:
            open_window(url)
        return 0

    token = new_token()
    settings = Settings()
    if not settings.existed():
        # First run: an endpoint that already answers stays the default; the page then shows the guided setup.
        found = discover_endpoint()
        if found:
            settings.update({"backend": found})
    elif "setup_done" not in settings.stored:
        settings.update({"setup_done": True})  # settings from a version before the guided setup: nothing to ask
    if a.engine_dir:
        settings.update({"engine": {"dir": str(Path(os.path.expanduser(a.engine_dir)).resolve())}})
    app = make_app(paths.app_root() / "web", settings, Conversations(), token)
    server, port = bind(app, "127.0.0.1", a.port)
    url = f"http://127.0.0.1:{port}/?t={token}"
    write_json_atomic(_instance_file(), {"pid": os.getpid(), "port": port, "token": token, "url": url})
    print(f"capy-flash app {__version__} on http://127.0.0.1:{port}/", flush=True)
    if a.no_window:
        print(url, flush=True)
    else:
        open_window(url)
    # "Run capy-flash on this computer" was chosen earlier: the engine starts with the app (unless turned off).
    threading.Thread(target=app.local.autostart, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: app.shutdown_soon())  # stop the engine too, not just the window
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        try:
            os.unlink(_instance_file())
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
