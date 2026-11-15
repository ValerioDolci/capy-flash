"""capy-flash on this computer, as the app's API sees it: the machine, what is installed, the engine, tuning.

One object ties machine.py, installer.py, engine.py and tuning.py to the settings, and holds the rules about
who may do what: the engine only starts when the settings say "local" (the user chose to run capy-flash here),
and what is downloaded comes from the catalog, never from a request.
"""

import json
import os
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import catalog, machine, paths
from .backends import model_kind
from .downloads import DownloadError, Fetcher
from .engine import Engine, EngineError, launch_plan, shell_command
from .installer import Busy, Installer
from .store import Settings
from .tuning import ProfileStore, Tuner, signature

# First run: the usual local OpenAI-compatible endpoints, in order (capy-flash's own default port first).
CANDIDATES = (
    ("capy-flash", "http://127.0.0.1:8080/v1"),
    ("vramble", "http://127.0.0.1:8099/v1"),
    ("vLLM", "http://127.0.0.1:8000/v1"),
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("Ollama", "http://127.0.0.1:11434/v1"),
)


def probe_endpoint(name: str, url: str, timeout: float = 0.8) -> Optional[Dict[str, Any]]:
    """Backend settings for `url` if its /v1/models answers without a key, else None."""
    try:
        with urllib.request.urlopen(url + "/models", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8")).get("data")
            if r.status == 200 and data:
                chat = [m for m in data if isinstance(m, dict) and model_kind(m) == "chat"]
                return {"kind": "endpoint", "name": name, "base_url": url, "api_key": "", "models": len(chat)}
    except Exception:
        pass
    return None


def discover_endpoints(candidates=CANDIDATES, timeout: float = 0.8) -> List[Dict[str, Any]]:
    """The local endpoints that answer, in the candidates' order (probed at the same time)."""
    found: List[Optional[Dict[str, Any]]] = [None] * len(candidates)

    def probe(i: int, name: str, url: str) -> None:
        found[i] = probe_endpoint(name, url, timeout)

    threads = [threading.Thread(target=probe, args=(i, n, u), daemon=True) for i, (n, u) in enumerate(candidates)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 1)
    return [f for f in found if f]


class Refused(Exception):
    """A request the app will not carry out; `code` lets the page say it in the user's language."""

    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class Local:
    def __init__(
        self,
        settings: Settings,
        system: Optional[machine.System] = None,
        root: Optional[Path] = None,
        fetcher: Optional[Fetcher] = None,
        sources: Optional[Dict[str, str]] = None,
        state_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        run_dir: Optional[Path] = None,
        candidates=CANDIDATES,
    ):
        self.settings = settings
        self.system = system or machine.System()
        configured = settings.get()["engine"].get("dir") or ""
        self.root = Path(root or (os.path.expanduser(configured) if configured else machine.default_dir()))
        self.machine = machine.Machine(self.system, self.root, ttl=15.0)
        self.installer = Installer(self.root, fetcher, self.system, **(sources or {}))
        self.profiles = ProfileStore((state_dir or paths.config_dir()) / "tuning.json")
        self.engine = Engine(
            self.installer,
            self.machine,
            port=self.port,
            allowed=self.is_local,
            profile_for=lambda key: self.profiles.flags(key, self.machine.facts()),
            log_path=(log_dir or paths.data_dir()) / "engine.log",
            run_dir=run_dir or paths.runtime_dir(),
            on_ready=self.profiles.note_load,
        )
        self.tuner = Tuner(
            self.engine,
            self.machine,
            self.profiles,
            version=lambda: (self.installer.release() or {}).get("version", ""),
        )
        self.installer.on_done = self._installed
        self.candidates = candidates
        self._restart_after_update = False

    # ---- settings -------------------------------------------------------------------------------------------
    def is_local(self) -> bool:
        return self.settings.get()["backend"].get("kind") == "local"

    def port(self) -> int:
        return int(self.settings.get()["engine"].get("port") or catalog.DEFAULT_PORT)

    def chosen(self) -> str:
        return self.settings.get()["engine"].get("model") or ""

    # ---- what the page shows --------------------------------------------------------------------------------
    def models(self, facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        installed = {}
        for m in catalog.MODELS:
            installed[m["key"]] = self.installer.model_status(m["key"])
        recommended = machine.recommend(facts, [k for k, s in installed.items() if s["installed"]])
        out = []
        for m in catalog.MODELS:
            status = installed[m["key"]]
            record = self.profiles.record(m["key"])
            out.append(
                {
                    "key": m["key"],
                    "family": m["family"],
                    "title": m["title"],
                    "edition": m["edition"],
                    "size_gb": m["size_gb"],
                    "need_gpus": m["need_gpus"],
                    "need_ram_gib": m["need_ram_gib"],
                    "need_disk_gb": m["need_disk_gb"],
                    "context": m["context"],
                    "vision": m["vision"],
                    "gpus": machine.gpus_for(m, facts),
                    "optional": bool(m.get("optional")),
                    "fit": machine.check(m, facts, status["installed"]),
                    "installed": status["installed"],
                    "verified": status["verified"],
                    "partial": status["partial"],
                    "bytes": status["bytes"],
                    # what is on disk for it, and whose: see Installer.model_status
                    "state": status["state"],
                    "found": status["found"],
                    "folder": status["folder"],
                    "deletable": status["deletable"],
                    "mismatch": status["mismatch"][:6],
                    "recommended": m["key"] == recommended,
                    "tuned": bool(record and record.get("active")),
                }
            )
        return out

    def supported(self, facts: Dict[str, Any]) -> bool:
        """Some model can run here (disk aside: that is fixable, and said per model)."""
        if not facts["platform_ok"]:
            return False
        return any(
            all(l["level"] != "fail" or l["code"] == "disk" for l in machine.check(m, facts)["lines"])
            for m in catalog.MODELS
        )

    def why_not(self, facts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Why the least demanding model cannot run here."""
        easiest = catalog.model("27b-16gb")
        return [l for l in machine.check(easiest, facts)["lines"] if l["level"] == "fail"]

    def snapshot(self) -> Dict[str, Any]:
        facts = self.machine.facts()
        chosen = self.chosen()
        model = catalog.model(chosen)
        # In this order: a tuning run saves its profile before it says "done", an installation finishes its files
        # before its job does. Reading the state first never shows "done" without the result.
        tuning = self.tuner.snapshot()
        job = self.installer.job.snapshot() if self.installer.job else None
        record = self.profiles.record(chosen) if model else None
        profile = None
        if record:
            profile = {
                k: record.get(k)
                for k in (
                    "active",
                    "flags",
                    "before",
                    "after",
                    "noise",
                    "baselines",
                    "trials",
                    "measured_at",
                    "engine_version",
                    "long_context",
                    "seconds",
                )
            }
            profile["stale"] = record.get("machine") != signature(facts)
        command = ""
        if model and self.installer.release():
            command = shell_command(
                launch_plan(model, facts, self.installer, self.port(), self.profiles.flags(chosen, facts))
            )
        engine_settings = self.settings.get()["engine"]
        return {
            "mode": "local" if self.is_local() else "endpoint",
            "supported": self.supported(facts),
            "why_not": self.why_not(facts) if not self.supported(facts) else [],
            "machine": facts,
            "dir": str(self.root),
            "release": self.installer.release(),
            "model": chosen,
            "models": self.models(facts),
            "process": self.engine.snapshot(),
            "job": job,
            "tuning": tuning,
            "profile": profile,
            "settings": {"port": self.port(), "autostart": bool(engine_settings.get("autostart", True))},
            "command": command,
            "lossy": catalog.LOSSY_SWITCHES,
            "lossless": catalog.LOSSLESS_SETTINGS,
        }

    def setup(self) -> Dict[str, Any]:
        """The first-run facts: the machine, the servers that answer, what the app proposes."""
        facts = self.machine.facts(fresh=True)
        endpoints = discover_endpoints(self.candidates)
        for e in endpoints:
            e.pop("api_key", None)
        supported = self.supported(facts)
        installed = self.installer.installed_models()
        return {
            "machine": facts,
            "supported": supported,
            "why_not": [] if supported else self.why_not(facts),
            "recommended": machine.recommend(facts, installed) if supported else None,
            "endpoints": endpoints,
            "installed": installed,
            "proposal": "endpoint" if endpoints else "local" if supported else "connect",
            "busy_gpus": machine.busy_gpus(facts["gpu_processes"]) if supported else [],
        }

    # ---- actions --------------------------------------------------------------------------------------------
    def _model(self, key: Any) -> Dict[str, Any]:
        model = catalog.model(key) if isinstance(key, str) else None
        if model is None:
            raise Refused("unknown_model", "unknown model", 400)
        return model

    def _not_while_tuning(self) -> None:
        if self.tuner.state == "running":
            raise Refused("tuning", "a tuning run is in progress")

    def _fit(self, model: Dict[str, Any], installed: bool) -> None:
        verdict = machine.check(model, self.machine.facts(fresh=True), installed)
        failed = [l for l in verdict["lines"] if l["level"] == "fail"]
        if failed:
            raise Refused("unsupported", failed[0]["text"])

    def use(self, key: Any, as_is: bool = False) -> None:
        """Run capy-flash on this computer with this model: the only way into "local" mode.
        `as_is`: the user accepts the files found in the model's folder without verification."""
        model = self._model(key)
        self._fit(model, self.installer.model_status(model["key"])["installed"] or as_is)
        self._not_while_tuning()
        if as_is:
            try:
                self.installer.accept_as_is(model["key"])
            except DownloadError as e:
                raise Refused(e.code, e.message)
        changed = self.chosen() != model["key"]
        self.settings.update({"backend": {"kind": "local"}, "engine": {"model": model["key"]}, "setup_done": True})
        if changed and self.engine.state in ("starting", "running"):
            self.engine.stop()
        self.autostart()

    def install(self, key: Any, new_folder: bool = False) -> Dict[str, Any]:
        """Download (or verify what is there). `new_folder`: next to a folder the app did not create."""
        model = self._model(key)
        status = self.installer.model_status(model["key"])
        self._fit(model, status["installed"] and not new_folder)
        try:
            return self.installer.start(model["key"], new_folder=bool(new_folder)).snapshot()
        except Busy as e:
            raise Refused("busy", str(e))

    def remove(self, key: Any) -> None:
        model = self._model(key)
        if self.engine.model == model["key"] and self.engine.state in ("starting", "running", "stopping"):
            raise Refused("in_use", "stop the engine before deleting the model it is running")
        if not self.installer.model_status(model["key"])["deletable"]:
            raise Refused("not_ours", "the app did not download these files and does not delete them")
        try:
            self.installer.remove_model(model["key"])
        except Busy as e:
            raise Refused("busy", str(e))
        except DownloadError as e:
            raise Refused(e.code, e.message)

    def start(self) -> None:
        self._not_while_tuning()
        try:
            self.engine.start(self.chosen())
        except EngineError as e:
            raise Refused(e.code, e.message)

    def stop(self) -> None:
        """Stop the engine this app started, and a tuning run first."""
        if self.tuner.state == "running":
            self.tuner.cancel(restore=False)
            self.tuner.wait(60)
        self.engine.stop()

    def autostart(self) -> None:
        """Start the chosen model if the user asked for that and everything is in place; never raises."""
        if not self.is_local() or not self.settings.get()["engine"].get("autostart", True):
            return
        key = self.chosen()
        if not key or self.engine.state in ("starting", "running") or self.tuner.state == "running":
            return
        if self.installer.release() is None or not self.installer.model_status(key)["installed"]:
            return
        try:
            self.engine.start(key)
        except EngineError:
            pass  # the reason is in the engine's snapshot

    def _installed(self, job) -> None:
        if job.state != "done":
            return
        if self._restart_after_update:
            self._restart_after_update = False
            if self.is_local() and self.chosen():
                try:
                    self.engine.start(self.chosen())
                except EngineError:
                    pass
            return
        if job.model and job.model == self.chosen():
            self.autostart()

    def engine_settings(self, change: Dict[str, Any]) -> None:
        update: Dict[str, Any] = {}
        if "port" in change:
            port = change["port"]
            if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
                raise Refused("bad_port", "the port must be a number between 1024 and 65535", 400)
            if self.engine.state in ("starting", "running") and port != self.port():
                raise Refused("in_use", "stop the engine before changing its port")
            update["port"] = port
        if "autostart" in change:
            update["autostart"] = bool(change["autostart"])
        if update:
            self.settings.update({"engine": update})

    def check_update(self) -> Dict[str, Any]:
        try:
            latest = self.installer.latest()
        except DownloadError as e:
            raise Refused(e.code, e.message, 502)
        current = (self.installer.release() or {}).get("version")
        return {
            "installed": current,
            "latest": latest["version"],
            "newer": bool(current) and _version_tuple(latest["version"]) > _version_tuple(current),
        }

    def update(self) -> Dict[str, Any]:
        self._not_while_tuning()
        was_running = self.engine.state in ("starting", "running")
        try:
            job = self.installer.start(None, update=True)
        except Busy as e:
            raise Refused("busy", str(e))
        if was_running:
            self.engine.stop()
            self._restart_after_update = True
        return job.snapshot()

    def tune_plan(self, key: Any, long_context: bool) -> Dict[str, Any]:
        return self.tuner.plan(self._model(key or self.chosen())["key"], long_context)

    def tune(self, long_context: bool) -> Dict[str, Any]:
        if self.installer.job and self.installer.job.state == "running":
            raise Refused("busy", "wait for the download to finish")
        try:
            return self.tuner.start(self._model(self.chosen())["key"], long_context)
        except EngineError as e:
            raise Refused(e.code, e.message)

    def set_profile_active(self, active: bool) -> None:
        """Revert (or apply again) this machine's profile for the chosen model, and restart the engine with it."""
        self._not_while_tuning()
        key = self._model(self.chosen())["key"]
        if not self.profiles.set_active(key, active):
            raise Refused("no_profile", "nothing has been measured for this model yet")
        if self.engine.state in ("starting", "running") and self.engine.model == key:
            try:
                self.engine.start(key)
            except EngineError as e:
                raise Refused(e.code, e.message)

    def close(self) -> None:
        self.installer.cancel()
        if self.tuner.state == "running":
            self.tuner.cancel(restore=False)
            self.tuner.wait(30)
        self.engine.close()


def _version_tuple(version: str):
    parts = []
    for piece in version.lstrip("v").split("-")[0].split("."):
        parts.append(int(piece) if piece.isdigit() else 0)
    return tuple(parts)
