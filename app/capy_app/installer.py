"""Install capy-flash where install.sh puts it, so the command line and the app see the same files:

    <root>/capy-flash-vX.Y.Z/    a release (bin/ninfer-serve, launchers, SHA256SUMS)
    <root>/current               symlink to the release in use
    <root>/models/<name>/        one model repository from Hugging Face, with its SHA256SUMS
    <root>/.capy-app/            the app's own: what it verified and created, and every temporary file

The sources are fixed (catalog.py). Nothing downloaded is used before its checksum matched: the release tarball
against its published .sha256, the unpacked release against its own SHA256SUMS, every model file against the
repository's SHA256SUMS (files it does not list: against the digest Hugging Face publishes for them).

What is already on disk belongs to whoever put it there:
- the app deletes or replaces only files it created itself (it records each one with size, time and digest), and
  writes its temporary files only under <root>/.capy-app/tmp;
- a file counts as installed only after its digest matched, and a model folder is one unit: if any file found in
  it differs from the repository, nothing in it is adopted or changed. The folder is reported as a different
  version, and the model can be downloaded into a new folder next to it, or used as it is if the user says so;
- nothing is read for verification or written through a link that leads outside <root>: a model folder that is
  such a link can only be used as it is.
"""

import collections
import errno
import json
import os
import re
import secrets
import shutil
import stat
import tarfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import catalog
from .downloads import (
    DownloadError,
    Fetcher,
    digest_of,
    ensure_space,
    parse_sums,
    part_path,
    safe_relative,
    sha256_file,
)
from .machine import System
from .store import write_json_atomic

RELEASE_DIR = re.compile(r"^capy-flash-(v\d+\.\d+\.\d+(?:[.-][A-Za-z0-9.]{1,24})?)$")
STATE_DIR = ".capy-app"
SKIP_FILES = (".gitattributes",)


class Busy(Exception):
    pass


class Job:
    """One installation, as the page sees it."""

    def __init__(self, model: Optional[str]):
        self.id = secrets.token_hex(4)
        self.model = model
        self.state = "running"  # running | done | failed | cancelled
        self.phase = "release"  # release | model
        self.step = "prepare"  # prepare | download | verify | unpack
        self.file = ""
        self.files_done = 0
        self.files_total = 0
        self.bytes_done = 0
        self.bytes_total = 0
        self.step_done = 0  # bytes hashed in a verify step
        self.step_total = 0
        self.error: Optional[Dict[str, str]] = None
        self.version = ""
        self.cancel = threading.Event()
        self.started = time.time()
        self.finished: Optional[float] = None
        self._samples: "collections.deque" = collections.deque(maxlen=24)
        self._lock = threading.Lock()

    def add(self, n: int) -> None:
        with self._lock:
            self.bytes_done += n
            now = time.monotonic()
            if not self._samples or now - self._samples[-1][0] >= 0.5:
                self._samples.append((now, self.bytes_done))

    def add_step(self, n: int) -> None:
        self.step_done += n

    def speed(self) -> float:
        with self._lock:
            if len(self._samples) < 2:
                return 0.0
            (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]
            stale = time.monotonic() - t1 > 5
        return 0.0 if stale or t1 <= t0 else max(0.0, (b1 - b0) / (t1 - t0))

    def snapshot(self) -> Dict[str, Any]:
        speed = self.speed()
        left = max(self.bytes_total - self.bytes_done, 0)
        # One number for the page: the download when there is one, else the check of what is already on disk.
        if self.bytes_total:
            percent = self.bytes_done / self.bytes_total
        else:
            percent = self.step_done / self.step_total if self.step_total else 0.0
        return {
            "id": self.id,
            "model": self.model,
            "state": self.state,
            "phase": self.phase,
            "step": self.step,
            "file": self.file,
            "files_done": self.files_done,
            "files_total": self.files_total,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "step_done": self.step_done,
            "step_total": self.step_total,
            "speed": speed,
            "version": self.version,
            "percent": max(0.0, min(1.0, percent)),
            "eta": (left / speed) if speed > 0 and self.state == "running" else None,
            "error": self.error,
        }


class Installer:
    def __init__(
        self,
        root: Path,
        fetcher: Optional[Fetcher] = None,
        system: Optional[System] = None,
        github_api: str = "https://api.github.com",
        github: str = "https://github.com",
        hf: str = "https://huggingface.co",
    ):
        self.root = Path(root)
        self.fetcher = fetcher or Fetcher()
        self.system = system or System()
        self.github_api = github_api.rstrip("/")
        self.github = github.rstrip("/")
        self.hf = hf.rstrip("/")
        self.job: Optional[Job] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.on_done = None  # called with the job when an installation ends

    # ---- the app's own corner of the root ----------------------------------------------------------------------
    def _state_path(self) -> Path:
        return self.root / STATE_DIR / "state.json"

    def _state(self) -> Dict[str, Any]:
        try:
            with open(self._state_path(), encoding="utf-8") as f:
                value = json.load(f)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self._require_inside(self._state_path())
        write_json_atomic(self._state_path(), state, mode=0o644)

    def _tmp(self) -> Path:
        """Where every temporary file of the app goes: never next to files it did not create."""
        tmp = self.root / STATE_DIR / "tmp"
        self._require_inside(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

    def contains(self, path: Path) -> bool:
        """`path` (it may not exist yet) is inside the root once every link on the way is resolved."""
        root = self.root.resolve()
        resolved = Path(path).resolve()
        return resolved == root or root in resolved.parents

    def _require_inside(self, path: Path) -> None:
        if not self.contains(path):
            raise DownloadError("outside", f"{path} leads outside {self.root}: the app does not write there")

    # ---- releases ----------------------------------------------------------------------------------------------
    def release(self) -> Optional[Dict[str, Any]]:
        """The release `current` points at: version, path, whether the app verified it, libraries it misses."""
        link = self.root / "current"
        try:
            path = link.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        match = RELEASE_DIR.match(path.name)
        if not match or path.parent != self.root.resolve() or not (path / catalog.ENGINE_BINARY).is_file():
            return None
        record = (self._state().get("releases") or {}).get(match.group(1)) or {}
        sums = path / "SHA256SUMS"
        verified = bool(record) and sums.is_file() and record.get("sums") == sha256_file(sums)
        return {
            "version": match.group(1),
            "path": str(path),
            "verified": verified,
            "missing_libs": record.get("missing_libs") or [] if verified else [],
        }

    def launcher(self, model: Dict[str, Any]) -> Path:
        return self.root / "current" / model["launcher"]

    # ---- models: where each one is, and whose files those are ---------------------------------------------------
    def _model_entry(self, key: str) -> Dict[str, Any]:
        entry = (self._state().get("models") or {}).get(key)
        return entry if isinstance(entry, dict) else {}

    def _set_model_entry(self, key: str, **changes: Any) -> None:
        state = self._state()
        entry = dict((state.get("models") or {}).get(key) or {})
        for name, value in changes.items():
            if value is None:
                entry.pop(name, None)
            else:
                entry[name] = value
        state.setdefault("models", {})[key] = entry
        self._save_state(state)

    def _folder_name(self, key: str) -> str:
        """models/<key>, or models/<key>-<revision> when the app downloaded next to a folder it did not create."""
        if catalog.model(key) is None:
            raise ValueError("unknown model: " + str(key))
        name = self._model_entry(key).get("folder")
        if isinstance(name, str) and re.match("^" + re.escape(key) + "-[0-9a-f]{8}$", name):
            return name
        return key

    def model_dir(self, key: str) -> Path:
        return self.root / "models" / self._folder_name(key)

    def artifact(self, key: str) -> Path:
        return self.model_dir(key) / catalog.model(key)["artifact"]

    def _record_path(self, name: str) -> Path:
        return self.root / STATE_DIR / "models" / (name + ".json")

    def _record(self, name: str) -> Dict[str, Any]:
        """What the app knows about a model folder: per file, size, time, digest and whether the app created it."""
        try:
            with open(self._record_path(name), encoding="utf-8") as f:
                value = json.load(f)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_record(self, name: str, record: Dict[str, Any]) -> None:
        self._require_inside(self._record_path(name))
        write_json_atomic(self._record_path(name), record, mode=0o644)

    def _links_out(self, folder: Path) -> bool:
        """The folder, or something in it, is a link that leads outside the root."""
        if not self.contains(folder):
            return True
        try:
            return any(p.is_symlink() and not self.contains(p) for p in folder.iterdir())
        except OSError:
            return False

    def model_status(self, key: str) -> Dict[str, Any]:
        """Cheap (no hashing). `found` is what is on disk:
        absent | partial (an app download not finished) | ready (every file verified) | foreign (files the app has
        not verified) | different (files that do not match the repository, or changed since they were verified) |
        outside (a link out of the root). `state` is the same, or "as_is" when the user chose to use an unverified
        folder as it is. Only ready and as_is count as installed."""
        model = catalog.model(key)
        if model is None:
            raise ValueError("unknown model: " + str(key))
        name = self._folder_name(key)
        folder = self.root / "models" / name
        out: Dict[str, Any] = {
            "state": "absent",
            "found": "absent",
            "installed": False,
            "verified": False,
            "partial": False,
            "bytes": 0,
            "folder": str(folder),
            "deletable": False,
            "mismatch": [],
            "as_is": False,
        }
        if not folder.is_symlink() and not folder.exists():
            return out
        record = self._record(name)
        files = record.get("files") or {}
        if self._links_out(folder):
            found = "outside"
        else:
            entries = [p for p in folder.iterdir()]
            out["bytes"] = sum(p.stat().st_size for p in entries if p.is_file() and not p.is_symlink())
            changed = sorted(n for n, st in files.items() if not _same(folder / n, st))
            seen = record.get("seen") or {}
            ours = bool(record.get("created_folder")) or any(st.get("created") for st in files.values())
            if record.get("complete") and not changed and (folder / model["artifact"]).is_file():
                found = "ready"
            elif changed:
                found, out["mismatch"] = "different", changed
            elif record.get("verdict") == "different" and seen and all(_same(folder / n, st) for n, st in seen.items()):
                found, out["mismatch"] = "different", list(record.get("mismatch") or [])
            elif record and not record.get("complete") and record.get("verdict") != "different" and (ours or files):
                found = "partial"
            elif not entries:
                found = "absent"
            else:
                found = "foreign"
            out["deletable"] = found in ("ready", "partial") and ours
        as_is = (
            bool(self._model_entry(key).get("as_is")) and found != "ready" and (folder / model["artifact"]).is_file()
        )
        out.update(
            found=found,
            state="as_is" if as_is else found,
            as_is=as_is,
            installed=found == "ready" or as_is,
            verified=found == "ready",
            partial=found == "partial",
        )
        return out

    def installed_models(self) -> List[str]:
        return [m["key"] for m in catalog.MODELS if self.model_status(m["key"])["installed"]]

    def accept_as_is(self, key: str) -> None:
        """The user's choice: use what is in the model folder without verifying or changing it."""
        if not self.artifact(key).is_file():
            raise DownloadError("layout", f"{self.artifact(key)} is not there")
        self._set_model_entry(key, as_is=True)

    def remove_model(self, key: str) -> Dict[str, Any]:
        """Delete what the app downloaded for this model: only files it created and that are still as it left them.
        Anything else in the folder stays, and so does a folder the app did not create."""
        status = self.model_status(key)
        if self.job and self.job.state == "running" and self.job.model == key:
            raise Busy("this model is being downloaded")
        name = self._folder_name(key)
        folder = self.root / "models" / name
        if status["found"] == "outside":
            raise DownloadError("outside", f"{folder} leads outside {self.root}: the app does not delete there")
        record = self._record(name)
        removed = 0
        for path, st in (record.get("files") or {}).items():
            target = folder / path
            if st.get("created") and safe_relative(path) and not target.is_symlink() and _same(target, st):
                target.unlink()
                removed += 1
        shutil.rmtree(self._tmp() / name, ignore_errors=True)
        try:
            self._record_path(name).unlink()
        except OSError:
            pass
        left: List[str] = []
        if folder.is_dir():
            if record.get("created_folder"):
                for sub in sorted((d for d in folder.rglob("*") if d.is_dir()), reverse=True):
                    try:
                        sub.rmdir()
                    except OSError:
                        pass
                try:
                    folder.rmdir()  # only when nothing else is in it
                except OSError:
                    pass
            if folder.is_dir():
                left = sorted(p.name for p in folder.iterdir())
        # Back to models/<key> when the folder the app had made next to it is gone.
        self._set_model_entry(key, as_is=None, folder=name if name != key and folder.exists() else None)
        return {"removed": removed, "left": left}

    # ---- jobs -----------------------------------------------------------------------------------------------
    def start(
        self, model: Optional[str], version: Optional[str] = None, update: bool = False, new_folder: bool = False
    ) -> Job:
        """Install the release (when missing, or `update`) and then `model` (when given), in the background.
        `new_folder`: download the model next to a folder the app did not create, instead of refusing."""
        if model is not None and catalog.model(model) is None:
            raise ValueError("unknown model: " + str(model))
        if version is not None:
            catalog.tarball_name(version)  # validates the tag
        with self._lock:
            if self.job and self.job.state == "running":
                raise Busy("another installation is running")
            job = Job(model)
            self.job = job
            self._thread = threading.Thread(
                target=self._run, args=(job, version, update, new_folder), daemon=True, name="capy-install"
            )
            self._thread.start()
            return job

    def cancel(self, job_id: Optional[str] = None) -> bool:
        job = self.job
        if job and job.state == "running" and (job_id is None or job.id == job_id):
            job.cancel.set()
            return True
        return False

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _run(self, job: Job, version: Optional[str], update: bool, new_folder: bool = False) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            current = self.release()
            if update or version or current is None or not current["verified"]:
                self._install_release(job, version, update)
            if job.model:
                self._install_model(job, job.model, new_folder)
            job.state = "done"
        except DownloadError as e:
            job.state = "cancelled" if e.code == "cancelled" else "failed"
            job.error = None if e.code == "cancelled" else {"code": e.code, "message": e.message}
        except Exception as e:  # a bug or an unexpected file system error: report it, keep the app up
            job.state = "failed"
            job.error = {"code": "internal", "message": f"{type(e).__name__}: {e}"}
        finally:
            job.finished = time.time()
            if self.on_done:
                try:
                    self.on_done(job)
                except Exception:
                    pass

    # ---- release ----------------------------------------------------------------------------------------------
    def latest(self) -> Dict[str, Any]:
        """Tag and tarball size of the latest release."""
        data = self.fetcher.json(f"{self.github_api}/repos/{catalog.GITHUB_REPO}/releases/latest")
        tag = data.get("tag_name") if isinstance(data, dict) else None
        if not isinstance(tag, str) or not catalog.VERSION_RE.match(tag):
            raise DownloadError("layout", "cannot find the latest capy-flash release")
        name = catalog.tarball_name(tag)
        size = next(
            (a.get("size") for a in data.get("assets") or [] if isinstance(a, dict) and a.get("name") == name), None
        )
        return {"version": tag, "size": size if isinstance(size, int) else None}

    def _created_release(self, version: str) -> bool:
        return bool(((self._state().get("releases") or {}).get(version) or {}).get("created"))

    def _install_release(self, job: Job, version: Optional[str], update: bool) -> None:
        job.phase, job.step = "release", "prepare"
        current = self.release()
        size = None
        if version is None:
            if current and not update:
                version = current["version"]  # installed (e.g. by install.sh) but not verified by the app yet
            else:
                latest = self.latest()
                version, size = latest["version"], latest["size"]
        job.version = version
        target = self.root / ("capy-flash-" + version)
        theirs = (
            f"{target} is there, the app did not install it and it does not pass its checksums. The app "
            "will not change it: move it away, or install it again with install.sh"
        )
        if target.is_symlink():
            raise DownloadError("foreign_release", f"{target} is a link: the app does not install through it")
        if (target / catalog.ENGINE_BINARY).is_file():
            job.step = "verify"
            try:
                self._verify_release(target, job)
                return self._adopt_release(target, version)
            except DownloadError as e:
                if e.code != "checksum":
                    raise
                if not self._created_release(version):
                    raise DownloadError("foreign_release", theirs)
                shutil.rmtree(target)  # the app's own release, damaged: fetch it again
        elif target.exists():
            if not self._created_release(version):
                raise DownloadError("foreign_release", theirs)
            shutil.rmtree(target)
        tarball = catalog.tarball_name(version)
        url = f"{self.github}/{catalog.GITHUB_REPO}/releases/download/{version}/{tarball}"
        published = parse_sums(self.fetcher.text(url + ".sha256", limit=4096)).get(tarball)
        if not published:
            raise DownloadError("layout", "the release has no published checksum for " + tarball)
        tmp = self._tmp()
        have = part_path(tmp / tarball).stat().st_size if part_path(tmp / tarball).exists() else 0
        if size:
            ensure_space(self.root, 3 * size - have)
            job.bytes_total += size
        job.bytes_done += have
        job.step, job.file, job.files_total = "download", tarball, job.files_total + 1
        part = self.fetcher.download(url, tmp / tarball, size, job.add, job.cancel)
        if not size:
            job.bytes_total += part.stat().st_size
        job.step, job.step_done, job.step_total = "verify", 0, part.stat().st_size
        if sha256_file(part, job.add_step, job.cancel) != published:
            part.unlink()
            raise DownloadError(
                "checksum",
                f"{tarball} does not match its published checksum (the download was corrupt and has been deleted)",
            )
        job.step = "unpack"
        unpack = Path(_mkdtemp(tmp, "unpack."))
        try:
            top = _extract(part, unpack)
            job.step = "verify"
            self._verify_release(top, job)
            if target.exists() or target.is_symlink():
                raise DownloadError("foreign_release", f"{target} appeared during the installation: not replaced")
            os.replace(top, target)
        finally:
            shutil.rmtree(unpack, ignore_errors=True)
        part.unlink()
        job.files_done += 1
        self._adopt_release(target, version, created=True)

    def _verify_release(self, folder: Path, job: Optional[Job] = None) -> None:
        """`sha256sum -c SHA256SUMS` in the release folder; the engine and the launchers must be listed."""
        sums_file = folder / "SHA256SUMS"
        if not sums_file.is_file():
            raise DownloadError("checksum", "the release has no SHA256SUMS")
        sums = parse_sums(sums_file.read_text(errors="replace"))
        needed = {catalog.ENGINE_BINARY} | {m["launcher"] for m in catalog.MODELS if not m.get("optional")}
        if not needed <= set(sums):
            raise DownloadError("checksum", "the release's SHA256SUMS does not cover the engine and its launchers")
        if job:
            job.step_done, job.step_total = 0, sum((folder / n).stat().st_size for n in sums if (folder / n).is_file())
        for name, digest in sums.items():
            path = folder / name
            if not safe_relative(name) or not path.is_file() or path.is_symlink():
                raise DownloadError("checksum", f"release verification failed: {name} is missing")
            if sha256_file(path, job.add_step if job else None, job.cancel if job else None) != digest:
                raise DownloadError("checksum", f"release verification failed: {name} does not match SHA256SUMS")

    def verify_current(self) -> None:
        """Check a release the app did not install itself (install.sh did) before it is run for the first time."""
        release = self.release()
        if release is None:
            raise DownloadError("layout", "the capy-flash engine is not installed")
        if not release["verified"]:
            self._verify_release(Path(release["path"]))
            self._adopt_release(Path(release["path"]), release["version"])

    def _adopt_release(self, folder: Path, version: str, created: bool = False) -> None:
        """Record the verification, look for missing libraries, point `current` at the release."""
        missing = self._missing_libs(folder)
        state = self._state()
        previous = (state.get("releases") or {}).get(version) or {}
        state.setdefault("releases", {})[version] = {
            "sums": sha256_file(folder / "SHA256SUMS"),
            "verified_at": int(time.time()),
            "missing_libs": missing,
            "created": bool(created or previous.get("created")),
        }
        self._save_state(state)
        link = self.root / "current"
        if link.is_symlink():
            try:
                resolved = link.resolve(strict=True)
            except (OSError, RuntimeError):
                resolved = None
            if resolved is not None and resolved == folder.resolve():
                return  # already there (install.sh's link stays as it is)
            # `current` is the pointer install.sh and the app share. It may be moved from one release of this root
            # to another; a link somebody pointed elsewhere (a build of their own) is theirs.
            ours = state.get("current") == os.readlink(link)
            in_root = (
                resolved is not None and resolved.parent == self.root.resolve() and RELEASE_DIR.match(resolved.name)
            )
            if not (ours or in_root):
                raise DownloadError(
                    "foreign_current",
                    f"{link} points to {os.readlink(link)}, which the app did not "
                    "set: it will not change it. Remove that link to let the app use its release",
                )
        elif link.exists():
            raise DownloadError("foreign_current", f"{link} is not a link: the app will not replace it")
        tmp = self.root / (".current." + secrets.token_hex(4))  # a name of the app's own, next to the link
        os.symlink(folder.name, tmp)  # relative: ~/capy-flash can be moved as a whole
        os.replace(tmp, link)  # atomic, like `ln -sfn`
        state["current"] = folder.name
        self._save_state(state)

    def _missing_libs(self, folder: Path) -> List[str]:
        """install.sh's ldd check: shared libraries the engine needs and the system does not have."""
        if self.system.uname() != ("Linux", "x86_64"):
            return []
        out = self.system.run(
            ["env", "LD_LIBRARY_PATH=" + str(folder / "lib"), "ldd", str(folder / catalog.ENGINE_BINARY)]
        )
        return [line.split()[0] for line in (out or "").splitlines() if "not found" in line and line.split()]

    # ---- models -----------------------------------------------------------------------------------------------
    def listing(self, key: str) -> Dict[str, Any]:
        """The model repository at its current commit: revision and files (path, size, expected digest)."""
        repo = catalog.model(key)["hf_repo"]
        if not repo:
            raise DownloadError(
                "unpublished",
                f"{key} is not published yet: convert it (docs/weight-conversion.md) into {self.artifact(key)} "
                "and use it as it is",
            )
        info = self.fetcher.json(f"{self.hf}/api/models/{repo}")
        revision = info.get("sha") if isinstance(info, dict) else None
        if not isinstance(revision, str) or not re.match(r"^[0-9a-f]{40}$", revision):
            raise DownloadError("layout", "cannot read the model repository " + repo)
        tree = self.fetcher.json(f"{self.hf}/api/models/{repo}/tree/{revision}?recursive=true")
        base = f"{self.hf}/{repo}/resolve/{revision}/"
        sums = parse_sums(self.fetcher.text(base + "SHA256SUMS", limit=1 << 20))
        files = []
        for entry in tree if isinstance(tree, list) else []:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue
            path, size = entry.get("path"), entry.get("size")
            if not isinstance(path, str) or not isinstance(size, int) or path in SKIP_FILES:
                continue
            if not safe_relative(path) or path.endswith(".part") or path.startswith(STATE_DIR):
                raise DownloadError("layout", "unexpected file name in the model repository: " + path[:80])
            lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
            digest = sums.get(path) or lfs.get("oid") or entry.get("oid")
            if not isinstance(digest, str) or not re.match(r"^([0-9a-f]{64}|[0-9a-f]{40})$", digest):
                raise DownloadError("layout", "no published checksum for " + path)
            files.append({"path": path, "size": size, "digest": digest, "url": base + urllib.parse.quote(path)})
        artifact = catalog.model(key)["artifact"]
        if artifact not in sums or not any(f["path"] == artifact for f in files):
            raise DownloadError("layout", f"the repository's SHA256SUMS does not list {artifact}")
        files.sort(key=lambda f: (f["size"], f["path"]))  # small files first: licences and checksums land early
        return {"revision": revision, "files": files}

    def _install_model(self, job: Job, key: str, new_folder: bool = False) -> None:
        job.phase, job.step, job.file = "model", "prepare", ""
        listing = self.listing(key)
        revision, files = listing["revision"], listing["files"]
        status = self.model_status(key)
        if new_folder and status["found"] in ("different", "outside", "foreign"):
            # Next to the folder the app did not create, under a name that says which revision this is.
            self._set_model_entry(key, folder=f"{key}-{revision[:8]}", as_is=None)
            status = self.model_status(key)
        name = self._folder_name(key)
        folder = self.root / "models" / name
        if status["found"] == "outside":
            raise DownloadError(
                "outside",
                f"{folder} leads outside {self.root}. The app does not read or change it; "
                "it can be used as it is, or the model can be downloaded into a new folder",
            )
        self._require_inside(folder)  # also before it exists: `models` itself may be a link
        record = self._record(name)
        if not folder.exists():
            folder.mkdir(parents=True)
            record = {"key": key, "created_folder": True, "files": {}}
        elif not record:
            record = {"key": key, "created_folder": False, "files": {}}
        known: Dict[str, Any] = record.setdefault("files", {})

        def own(path: str) -> bool:
            """A file the app created, still as the app left it: the only kind it may replace."""
            st = known.get(path)
            return bool(st and st.get("created") and _same(folder / path, st))

        # 1. What is already there: checked against the repository as one unit, without touching anything.
        present = [f for f in files if (folder / f["path"]).exists() or (folder / f["path"]).is_symlink()]
        todo = [
            f
            for f in present
            if not (_same(folder / f["path"], known.get(f["path"])) and known[f["path"]].get("digest") == f["digest"])
        ]
        job.step, job.step_done = "verify", 0
        job.step_total = sum((folder / f["path"]).stat().st_size for f in todo if (folder / f["path"]).is_file())
        verified: Dict[str, Any] = {}
        mismatch: List[str] = []
        for f in todo:
            path = folder / f["path"]
            job.file = f["path"]
            if path.is_symlink() or not path.is_file():
                mismatch.append(f["path"])
            elif path.stat().st_size != f["size"]:
                job.add_step(path.stat().st_size)
                mismatch.append(f["path"])
            elif digest_of(path, f["digest"], job.add_step, job.cancel) != f["digest"]:
                mismatch.append(f["path"])
            else:  # a size that matches is not enough: only the digest makes a file count
                verified[f["path"]] = dict(_stamp(path), digest=f["digest"], created=own(f["path"]))
        stale = {m for m in mismatch if own(m)}  # the app's own files, of an earlier revision: it replaces them
        theirs = sorted(m for m in mismatch if m not in stale)
        if theirs:
            # Somebody else's files of another version. Nothing here is adopted, nothing is changed.
            record["files"] = {n: st for n, st in known.items() if own(n)}
            record.update(
                complete=False,
                verdict="different",
                mismatch=theirs,
                revision=revision,
                seen={f["path"]: _stamp(folder / f["path"]) for f in present if (folder / f["path"]).is_file()},
            )
            self._save_record(name, record)
            raise DownloadError(
                "different_version",
                f"a different version of this model is in {folder} "
                f"({', '.join(theirs[:3])}{', ...' if len(theirs) > 3 else ''} do not match the "
                "repository). The app did not create it and will not change it",
            )
        known.update(verified)
        for gone in [n for n in known if not (folder / n).is_file()]:
            del known[gone]
        tmp = self._tmp() / name
        if record.get("revision") != revision:
            shutil.rmtree(tmp, ignore_errors=True)  # parts of another revision cannot be continued
        record.update(revision=revision, complete=False)
        for stale_key in ("verdict", "mismatch", "seen"):
            record.pop(stale_key, None)
        self._save_record(name, record)

        # 2. What is missing: downloaded under the app's own temporary names, verified, then put in place.
        missing = [f for f in files if f["path"] not in known or f["path"] in stale]

        def parked(f: Dict[str, Any]) -> int:
            part = part_path(tmp / f["path"])
            return min(part.stat().st_size, f["size"]) if part.is_file() else 0

        have = sum(parked(f) for f in missing)
        need = sum(f["size"] for f in missing)
        ensure_space(folder, need - have)
        job.bytes_total += need
        job.bytes_done += have
        job.files_total += len(files)
        job.files_done += len(files) - len(missing)
        for f in missing:
            job.file, job.step = f["path"], "download"
            part = self.fetcher.download(f["url"], tmp / f["path"], f["size"], job.add, job.cancel)
            job.step, job.step_done, job.step_total = "verify", 0, f["size"]
            if digest_of(part, f["digest"], job.add_step, job.cancel) != f["digest"]:
                part.unlink()
                job.bytes_done -= f["size"]
                raise DownloadError(
                    "checksum",
                    f"{f['path']} does not match its published checksum (the download "
                    "was corrupt and has been deleted)",
                )
            dest = folder / f["path"]
            self._place(part, dest, replace=f["path"] in stale and own(f["path"]))
            known[f["path"]] = dict(_stamp(dest), digest=f["digest"], created=True)
            self._save_record(name, record)
            job.files_done += 1
        record["complete"] = True
        self._save_record(name, record)
        shutil.rmtree(tmp, ignore_errors=True)
        self._set_model_entry(key, as_is=None)

    def _place(self, part: Path, dest: Path, replace: bool = False) -> None:
        """Move a verified download to its final name. An existing file is replaced only if the app created it."""
        if dest.is_symlink() or not self.contains(dest.parent):
            raise DownloadError("outside", f"{dest} leads outside {self.root}: the app does not write there")
        if dest.exists() and not replace:
            raise DownloadError(
                "different_version",
                f"{dest} appeared while it was being downloaded: the app did not create it and will not replace it",
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(part, dest)
        except OSError as e:
            if e.errno != errno.EXDEV:
                raise DownloadError("disk", f"cannot write {dest}: {e}")
            # The models are on another file system than the app's temporary files: copy, then rename in place.
            side = dest.with_name(f".{dest.name}.capy-app-{secrets.token_hex(4)}")
            try:
                shutil.copyfile(part, side)
                os.replace(side, dest)
            except OSError as copy_error:
                try:
                    side.unlink()
                except OSError:
                    pass
                raise DownloadError("disk", f"cannot write {dest}: {copy_error}")
            part.unlink()


def _stamp(path: Path) -> Dict[str, int]:
    st = path.stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def _same(path: Path, stamp: Optional[Dict[str, int]]) -> bool:
    """The file is the one the app verified (same size and modification time)."""
    if not stamp or not path.is_file():
        return False
    st = path.stat()
    return st.st_size == stamp.get("size") and int(st.st_mtime) == stamp.get("mtime")


def _mkdtemp(parent: Path, prefix: str) -> str:
    import tempfile

    return tempfile.mkdtemp(prefix=prefix, dir=str(parent))


def _extract(tarball: Path, into: Path) -> Path:
    """Unpack a verified release tarball that holds one top folder; refuse anything that would leave `into`."""
    root = into.resolve()
    with tarfile.open(tarball, "r:gz") as tar:
        members = tar.getmembers()
        tops = set()
        for m in members:
            name = m.name[2:] if m.name.startswith("./") else m.name
            name = name.rstrip("/")
            if not safe_relative(name):
                raise DownloadError("layout", "unsafe path in the release archive: " + m.name[:80])
            tops.add(name.split("/")[0])
            if m.isdir() or m.isreg():
                pass
            elif m.issym() or m.islnk():
                base = (root / name).parent if m.issym() else root
                target = (base / m.linkname).resolve()
                if os.path.isabs(m.linkname) or (target != root and root not in target.parents):
                    raise DownloadError("layout", "unsafe link in the release archive: " + m.name[:80])
            else:
                raise DownloadError("layout", "unexpected entry in the release archive: " + m.name[:80])
            m.mode &= 0o777  # no setuid / setgid / sticky
            m.uid, m.gid, m.uname, m.gname = os.getuid(), os.getgid(), "", ""  # owned by whoever installs
        if len(tops) != 1:
            raise DownloadError("layout", "the release archive should hold exactly one folder")
        if hasattr(tarfile, "tar_filter"):
            tar.extractall(str(root), members=members, filter="tar")
        else:
            tar.extractall(str(root), members=members)
    top = root / tops.pop()
    if not top.is_dir() or top.is_symlink():
        raise DownloadError("layout", "the release archive should hold exactly one folder")
    for p in top.rglob("*"):
        if p.is_file() and not p.is_symlink():
            mode = stat.S_IMODE(p.stat().st_mode)
            os.chmod(p, (mode | 0o600) & 0o755)
    return top
