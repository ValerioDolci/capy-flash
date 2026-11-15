"""Start, watch and stop the capy-flash engine the app installed on this computer.

The engine is started with the release's own launcher (serve-flash-next.sh, serve-tp2.sh, serve-27b-16gb.sh), the
way install.sh prints it, plus the flags of this machine's tuned profile. One place decides whether it may start
(Engine.start): only when the user chose to run capy-flash here, on a machine that passes install.sh's checks,
with nothing else answering on the port and nobody else holding the GPUs. So the app never starts a second
engine next to a server such as vramble / llama-swap.
"""

import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import catalog, machine
from .downloads import DownloadError
from .installer import Installer
from .store import write_json_atomic

RESTART_EXIT = 2  # ninfer-serve exits with 2 when the Engine fails: restart it (docs/serving.md)
MAX_RESTARTS = 3
RESTART_WINDOW = 600.0
SETTLE_SECONDS = 12.0  # after stopping an engine: how long its VRAM, RAM and port may take to be free again
SETTLE_CODES = ("gpu_busy", "ram", "port_busy")


class EngineError(Exception):
    """code: not_local | unsupported | ram | no_release | no_model | unverified | port_busy | gpu_busy | busy | spawn"""

    def __init__(self, code: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}


def clean_profile(model: Dict[str, Any], profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Only the lossless settings that apply to `model`, with valid values; anything else is dropped."""
    p = profile or {}
    return {
        s["key"]: p[s["key"]]
        for s in catalog.LOSSLESS_SETTINGS
        if s["key"] in p and catalog.applies(s, model) and _valid(s, model, p[s["key"]])
    }


def _valid(setting: Dict[str, Any], model: Dict[str, Any], value: Any) -> bool:
    if "range" in setting:
        low, high = setting["range"]
        return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high
    choices = model["gpus"] if setting["choices"] is None else setting["choices"]
    return any(type(value) is type(choice) and value == choice for choice in choices)


def profile_flags(profile: Dict[str, Any]) -> List[str]:
    """The engine flags of a profile clean_profile returned."""
    flags: List[str] = []
    for s in catalog.LOSSLESS_SETTINGS:
        if s["flag"] and s["key"] in profile:
            flags += [s["flag"]] if profile[s["key"]] is True else [s["flag"], str(profile[s["key"]])]
    return flags


def launch_plan(
    model: Dict[str, Any],
    facts: Dict[str, Any],
    installer: Installer,
    port: int,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The command and environment for `model` on this machine: install.sh's command plus the tuned flags."""
    profile = clean_profile(model, profile)
    usable = [g["index"] for g in facts["gpus"] if g["usable"]]
    gpus = profile.get("gpus") or machine.gpus_for(model, facts)
    gpus = min(gpus, max(len(usable), 1))
    devices = ",".join(str(i) for i in (usable[:gpus] or list(range(gpus))))
    argv = [str(installer.launcher(model)), str(installer.artifact(model["key"]))] + list(model["flags"])
    if gpus == 2:
        argv += list(model["flags_2gpu"])
    argv += profile_flags(profile)
    env = {"NINFER_HOST": "127.0.0.1", "NINFER_PORT": str(port)}
    if model["launcher"] == "serve-27b-16gb.sh":
        env["NINFER_DEVICE"] = devices.split(",")[0]
    else:
        env["NINFER_DEVICES"] = devices
    return {"argv": argv, "env": env, "gpus": gpus, "profile": profile}


def shell_command(plan: Dict[str, Any]) -> str:
    """The same start, as a line for a terminal, a systemd unit or a llama-swap entry."""
    import shlex

    env = " ".join(
        f"{k}={shlex.quote(v)}" for k, v in sorted(plan["env"].items()) if not (k == "NINFER_HOST" and v == "127.0.0.1")
    )
    return (env + " " if env else "") + " ".join(shlex.quote(a) for a in plan["argv"])


def child_environment(extra: Dict[str, str]) -> Dict[str, str]:
    """The app's environment without the speed switches that change the numbers, plus the launcher's settings."""
    env = {k: v for k, v in os.environ.items() if k not in catalog.LOSSY_ENV}
    env.update(extra)
    return env


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _command_line(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        pass
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


class Engine:
    def __init__(
        self,
        installer: Installer,
        machine_: machine.Machine,
        port: Callable[[], int],
        allowed: Callable[[], bool],
        profile_for: Callable[[str], Dict[str, Any]],
        log_path: Path,
        run_dir: Path,
        on_ready: Optional[Callable[[str, float], None]] = None,
    ):
        self.installer = installer
        self.machine = machine_
        self._port = port
        self._allowed = allowed
        self._profile_for = profile_for
        self.log_path = Path(log_path)
        self.record_path = Path(run_dir) / "engine.json"
        self.on_ready = on_ready
        self._lock = threading.RLock()  # the fields below
        self._op = threading.RLock()  # one start or stop at a time (they wait for processes)
        self.command = ""
        self._proc: Optional[subprocess.Popen] = None
        self._generation = 0
        self._stopping = False
        self._stopped_at = 0.0
        self._restart_times: List[float] = []
        self.state = "stopped"  # stopped | starting | running | stopping | failed
        self.model: Optional[str] = None
        self.pid: Optional[int] = None
        self.port: Optional[int] = None
        self.profile: Dict[str, Any] = {}
        self.gpus = 0
        self.started_at: Optional[float] = None
        self.load_seconds: Optional[float] = None
        self.restarts = 0
        self.exit_code: Optional[int] = None
        self.error: Optional[Dict[str, Any]] = None
        self.adopted = False
        self._adopt()

    # ---- facts ------------------------------------------------------------------------------------------------
    def base_url(self, port: Optional[int] = None) -> str:
        return f"http://127.0.0.1:{port or self.port or self._port()}/v1"

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "model": self.model,
                "pid": self.pid,
                "port": self.port or self._port(),
                "profile": dict(self.profile),
                "gpus": self.gpus,
                "started_at": self.started_at,
                "load_seconds": self.load_seconds,
                "restarts": self.restarts,
                "exit_code": self.exit_code,
                "error": self.error,
                "adopted": self.adopted,
                "command": self.command,
            }

    def log_tail(self, lines: int = 200) -> List[str]:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 256 * 1024))
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        return text.splitlines()[-max(1, min(lines, 2000)) :]

    def healthy(self, port: Optional[int] = None) -> bool:
        try:
            url = f"http://127.0.0.1:{port or self.port or self._port()}/health"
            with urllib.request.urlopen(url, timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    # ---- start ------------------------------------------------------------------------------------------------
    def check(self, key: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Everything that must hold before the engine starts. Returns the launch plan or raises EngineError."""
        model = catalog.model(key or "")
        if model is None:
            raise EngineError("no_model", "choose a model first")
        if not self._allowed():
            raise EngineError(
                "not_local",
                'this app is set to use a server: choose "run capy-flash on this '
                'computer" before starting an engine here',
            )
        facts = self.machine.facts(fresh=True)
        on_disk = self.installer.model_status(key)
        installed = on_disk["installed"]
        verdict = machine.check(model, facts, installed=True)
        failed = [l for l in verdict["lines"] if l["level"] == "fail"]
        if failed:
            raise EngineError("unsupported", failed[0]["text"], {"line": failed[0]})
        release = self.installer.release()
        if release is None:
            raise EngineError("no_release", "the capy-flash engine is not installed yet")
        if not installed and on_disk["found"] in ("foreign", "different", "outside"):
            # Files are there, but nothing the app has verified: it runs them only if the user says "as it is".
            raise EngineError(
                "unverified",
                f"{model['title']} ({model['edition']}) in {on_disk['folder']} is not "
                "verified: verify it, download the current version, or choose to use it as it is",
                {"found": on_disk["found"], "folder": on_disk["folder"]},
            )
        if not installed:
            raise EngineError("no_model", f"{model['title']} ({model['edition']}) is not downloaded yet")
        short = [l for l in verdict["lines"] if l["code"] == "ram" and l["level"] == "warn"]
        if short:
            raise EngineError("ram", short[0]["text"], {"line": short[0]})
        port = self._port()
        if _port_answers(port):
            raise EngineError(
                "port_busy",
                f"something already answers on port {port} of this computer (another "
                "capy-flash?). Use it as a server, or change the engine's port",
                {"port": port},
            )
        busy = machine.busy_gpus(facts["gpu_processes"], ignore_pids=(self.pid,) if self.pid else ())
        if busy:
            names = ", ".join(
                f"{os.path.basename(p['name']) or 'pid'} ({p['pid']}, {p['memory_mib'] // 1024} GiB)" for p in busy[:4]
            )
            raise EngineError(
                "gpu_busy",
                "the GPUs are in use by another program: " + names + ". The app does not start an engine next to it",
                {"processes": busy},
            )
        if not release["verified"]:  # installed by install.sh: check it once before running it
            try:
                self.installer.verify_current()
            except DownloadError as e:
                raise EngineError("no_release", e.message)
        return launch_plan(model, facts, self.installer, port, self._profile_for(key) if profile is None else profile)

    def start(self, key: str, profile: Optional[Dict[str, Any]] = None) -> None:
        """Start `key` (stopping what runs now). `profile`: the tuner's candidate; None: this machine's profile."""
        with self._op:
            self._halt()
            while True:
                try:
                    plan = self.check(key, profile)
                    break
                except EngineError as e:
                    # Right after a stop (a restart, a tuning run) the engine that just exited may still show up
                    # in nvidia-smi, in MemAvailable or on its port for a moment: look again before refusing.
                    if e.code in SETTLE_CODES and time.time() - self._stopped_at < SETTLE_SECONDS:
                        time.sleep(0.5)
                        continue
                    with self._lock:
                        self.state, self.model = "stopped", key
                        self.error = {"code": e.code, "message": e.message, "detail": e.detail}
                    raise
            with self._lock:
                self._restart_times = []
                self.restarts = 0
                self._spawn(key, plan)

    def _spawn(self, key: str, plan: Dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists() and self.restarts == 0:
            os.replace(self.log_path, self.log_path.with_name(self.log_path.name + ".1"))
        log = open(self.log_path, "ab")
        log.write(("# capy-flash app, %s: %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), shell_command(plan))).encode())
        log.flush()
        try:
            proc = subprocess.Popen(
                plan["argv"],
                env=child_environment(plan["env"]),
                cwd=str(self.installer.root),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as e:
            log.close()
            self.state, self.error = "failed", {"code": "spawn", "message": f"cannot start the engine: {e}"}
            raise EngineError("spawn", f"cannot start the engine: {e}")
        log.close()
        self._generation += 1
        self._proc, self._stopping = proc, False
        self.state, self.model, self.pid, self.port = "starting", key, proc.pid, int(plan["env"]["NINFER_PORT"])
        self.profile, self.gpus = plan["profile"], plan["gpus"]
        self.started_at, self.load_seconds, self.exit_code, self.error, self.adopted = (
            time.time(),
            None,
            None,
            None,
            False,
        )
        self.command = shell_command(plan)
        try:
            write_json_atomic(
                self.record_path,
                {
                    "pid": proc.pid,
                    "model": key,
                    "port": self.port,
                    "profile": self.profile,
                    "gpus": self.gpus,
                    "started_at": self.started_at,
                    "artifact": plan["argv"][1],
                },
            )
        except OSError:
            pass
        threading.Thread(target=self._watch, args=(self._generation,), daemon=True, name="capy-engine").start()

    def _watch(self, generation: int) -> None:
        while True:
            with self._lock:
                if generation != self._generation:
                    return
                proc, pid, state = self._proc, self.pid, self.state
            code = proc.poll() if proc is not None else (None if pid and _alive(pid) else -1)
            if code is not None:
                return self._exited(generation, code)
            if state == "starting" and self.healthy():
                with self._lock:
                    if generation != self._generation:
                        return
                    self.state = "running"
                    adopted = self.adopted
                    if not adopted:
                        self.load_seconds = time.time() - (self.started_at or time.time())
                    model, seconds = self.model, self.load_seconds
                if self.on_ready and model and not adopted:
                    try:
                        self.on_ready(model, seconds)
                    except Exception:
                        pass
            time.sleep(0.2 if state == "starting" else 1.0)

    def _exited(self, generation: int, code: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            was, key, profile = self.state, self.model, dict(self.profile)
            self._proc, self.pid, self.exit_code = None, None, code
            self._forget()
            if self._stopping:
                self.state = "stopped"
                return
            now = time.time()
            self._restart_times = [t for t in self._restart_times if now - t < RESTART_WINDOW]
            restart = code == RESTART_EXIT and was == "running" and len(self._restart_times) < MAX_RESTARTS and key
            if restart:
                self._restart_times.append(now)
                self.state = "starting"
            else:
                tail = [l for l in self.log_tail(14) if not l.startswith("# capy-flash app")]
                what = "the engine stopped while loading" if was == "starting" else "the engine stopped"
                self.state = "failed"
                self.error = {
                    "code": "exit",
                    "message": f"{what} (exit code {code})" if code >= 0 else what,
                    "exit_code": code,
                    "log": tail[-8:],
                }
                return
        try:  # exit code 2: the Engine failed and asks to be started again
            plan = self.check(key, profile)
        except EngineError as e:
            with self._lock:
                if generation == self._generation:
                    self.state, self.error = "failed", {"code": e.code, "message": e.message}
            return
        with self._lock:
            if generation == self._generation and not self._stopping:
                self.restarts += 1
                self._spawn(key, plan)

    # ---- stop -------------------------------------------------------------------------------------------------
    def stop(self, timeout: float = 20.0) -> None:
        with self._op:
            self._halt(timeout)

    def _halt(self, timeout: float = 20.0) -> None:
        with self._lock:
            proc, pid = self._proc, self.pid
            self._stopping = True
            self._generation += 1  # the watcher of the old process, and a restart it may be preparing, are void
            if not pid:
                self.state = "stopped"
                return
            self.state = "stopping"
        _signal_group(pid, signal.SIGTERM)
        deadline = time.time() + timeout
        while time.time() < deadline and _running(proc, pid):
            time.sleep(0.05)
        if _running(proc, pid):
            _signal_group(pid, signal.SIGKILL)
            deadline = time.time() + 5
            while time.time() < deadline and _running(proc, pid):
                time.sleep(0.05)
        with self._lock:
            self._proc, self.pid = None, None
            self.state, self.exit_code, self.error = "stopped", None, None
            self._stopped_at = time.time()
            self._forget()

    def wait_ready(self, timeout: float = 900.0, cancel: Optional[threading.Event] = None) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.state == "running":
                return True
            if self.state in ("failed", "stopped") or (cancel is not None and cancel.is_set()):
                return False
            time.sleep(0.1)
        return False

    def close(self) -> None:
        """The app is quitting: the engine it started goes with it."""
        try:
            self.stop()
        except Exception:
            pass

    # ---- a process left by an app that died -------------------------------------------------------------------
    def _forget(self) -> None:
        try:
            os.unlink(self.record_path)
        except OSError:
            pass

    def _adopt(self) -> None:
        try:
            with open(self.record_path, encoding="utf-8") as f:
                record = json.load(f)
            pid = int(record["pid"])
            key = record["model"]
        except (OSError, ValueError, KeyError, TypeError):
            return
        # The pid may belong to something else by now: it must still be an engine serving that model's file.
        line = _command_line(pid) if _alive(pid) else ""
        artifact = str(record.get("artifact") or "")
        if catalog.model(key) is None or not artifact or artifact not in line:
            return self._forget()
        self.pid, self.model, self.port = pid, key, int(record.get("port") or self._port())
        self.profile, self.gpus = record.get("profile") or {}, int(record.get("gpus") or 0)
        self.started_at, self.adopted = record.get("started_at"), True
        self.state = "starting"
        self._generation += 1
        threading.Thread(target=self._watch, args=(self._generation,), daemon=True, name="capy-engine").start()


def _running(proc: Optional[subprocess.Popen], pid: int) -> bool:
    return proc.poll() is None if proc is not None else _alive(pid)


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)  # the launcher was started in its own session: the group is the engine and its children
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _port_answers(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False
