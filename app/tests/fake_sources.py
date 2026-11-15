"""A local stand-in for GitHub releases and Hugging Face, for the tests and the demo.

It serves a release tarball (built here from a fake release tree whose launchers start tests/fake_engine.py) with
its .sha256, and model repositories (commit, file tree, SHA256SUMS, files with Range support). Faults can be
injected per file: break the connection after N bytes, serve corrupt bytes, ignore Range, limit the speed.
"""

import hashlib
import io
import json
import os
import socket
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = "ValerioDolci/capy-flash"
ENGINE = Path(__file__).resolve().parent / "fake_engine.py"
LAUNCHERS = {
    "serve-flash-next.sh": (
        "devices=${NINFER_DEVICES:-0,1}\n"
        'if [ "${devices#*,}" = "$devices" ]; then placement="--profile flash-next-1gpu --device $devices"\n'
        'else placement="--profile flash-next --devices $devices"; fi\n'
        'exec "$here/bin/ninfer-serve" "$artifact" $placement --host "${NINFER_HOST:-127.0.0.1}" '
        '--port "${NINFER_PORT:-8080}" --model-id "${NINFER_MODEL_ID:-capy-flash}" "$@"\n'
    ),
    "serve-tp2.sh": (
        'exec "$here/bin/ninfer-serve" "$artifact" --profile 27b --host "${NINFER_HOST:-127.0.0.1}" '
        '--port "${NINFER_PORT:-8080}" --devices "${NINFER_DEVICES:-0,1}" '
        '--model-id "${NINFER_MODEL_ID:-qwen3.8-27b}" "$@"\n'
    ),
    "serve-27b-16gb.sh": (
        'exec "$here/bin/ninfer-serve" "$artifact" --profile 27b-16gb '
        '--host "${NINFER_HOST:-127.0.0.1}" --port "${NINFER_PORT:-8080}" '
        '--device "${NINFER_DEVICE:-0}" --model-id "${NINFER_MODEL_ID:-qwen3.8-27b}" "$@"\n'
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def release_files(python: str = sys.executable) -> dict:
    """A release tree shaped like the real tarball: bin/ninfer-serve, three launchers, lib/, SHA256SUMS."""
    head = '#!/bin/sh\nhere=$(cd "$(dirname "$0")" && pwd)\nartifact=$1; shift\n'
    files = {
        "bin/ninfer-serve": f'#!/bin/sh\nexec "{python}" "{ENGINE}" "$@"\n'.encode(),
        "lib/libcudart.so.13": b"\x7fELF fake cuda runtime" + b"\0" * 2048,
        "chat_templates/qwen3_8.jinja": b"{{ messages }}\n",
        "README.md": b"# capy-flash (fake release)\n",
        "COMMIT": b"version fake\n",
    }
    for name, body in LAUNCHERS.items():
        files[name] = (head + body).encode()
    files["SHA256SUMS"] = "".join(f"{sha256(data)}  {name}\n" for name, data in sorted(files.items())).encode()
    return files


def tarball(version: str, files: dict, top: str = "", extra=None) -> bytes:
    """A .tar.gz with one top folder. `extra`: a list of TarInfo-like tuples (name, type, linkname) to add."""
    top = top or f"capy-flash-{version}-linux-x86_64-sm120a-ubuntu26.04"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        folder = tarfile.TarInfo(top + "/")
        folder.type, folder.mode = tarfile.DIRTYPE, 0o775
        tar.addfile(folder)
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            info.mode = 0o775 if name.endswith(".sh") or name.startswith("bin/") else 0o664
            tar.addfile(info, io.BytesIO(data))
        for name, kind, link in extra or []:
            info = tarfile.TarInfo(name)
            info.type, info.linkname = kind, link
            tar.addfile(info)
    return buf.getvalue()


def model_files(artifact: str, size: int = 300_000, parts: int = 0) -> dict:
    """A model repository: the artifact (and split parts), a small text file, a README outside SHA256SUMS."""
    files = {
        artifact: os.urandom(size),
        "README.md": b"# a fake model\n",
        "NOTICE": b"notice\n",
        ".gitattributes": b"*.ninfer filter=lfs\n",
    }
    for i in range(1, parts + 1):
        files[f"{artifact}.part-{i:04d}"] = os.urandom(size)
    listed = [n for n in files if n not in ("README.md", ".gitattributes")]
    files["SHA256SUMS"] = "".join(f"{sha256(files[n])}  {n}\n" for n in sorted(listed)).encode()
    return files


class Sources:
    def __init__(self, version: str = "v0.12.0"):
        self.version = version
        self.release = tarball(version, release_files())
        self.release_sha = sha256(self.release)
        self.repos = {}  # hf repo -> {"revision": sha, "files": {path: bytes}}
        self.break_after = {}  # file name -> bytes after which the connection is cut (once)
        self.corrupt = set()  # file names served with a flipped byte
        self.ignore_range = False
        self.rate = 0  # bytes per second (0: unlimited)
        self.requests = []  # (path, Range header)
        self.fail_status = {}  # path prefix -> HTTP status
        self.on_break = None  # called with the file name when a connection is cut
        sources = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                sources.requests.append((path, self.headers.get("Range")))
                for prefix, status in sources.fail_status.items():
                    if path.startswith(prefix):
                        return self._bytes(status, b"{}", "application/json")
                found = sources.lookup(path)
                if found is None:
                    return self._bytes(404, b"not found", "text/plain")
                name, data, ctype = found
                if name in sources.corrupt:
                    data = data[:-1] + bytes([data[-1] ^ 0xFF])
                start = 0
                status = 200
                rng = self.headers.get("Range")
                if rng and not sources.ignore_range:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    status = 206
                body = data[start:]
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.end_headers()
                cut = sources.break_after.pop(name, None)
                end = len(body) if cut is None else max(0, min(len(body), cut - start))
                sent = 0
                try:
                    while sent < end:
                        piece = body[sent : min(sent + 16384, end)]
                        self.wfile.write(piece)
                        self.wfile.flush()
                        sent += len(piece)
                        if sources.rate:
                            time.sleep(len(piece) / sources.rate)
                    if end < len(body):  # the connection drops in the middle of the file
                        self.close_connection = True
                        if sources.on_break:
                            sources.on_break(name)
                        try:
                            self.connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

            def _bytes(self, status, data, ctype):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def add_model(self, repo: str, files: dict, revision: str = "") -> None:
        self.repos[repo] = {"revision": revision or hashlib.sha1(repo.encode()).hexdigest(), "files": files}

    def lookup(self, path: str):
        name = f"capy-flash-{self.version}-linux-x86_64-sm120a-ubuntu26.04.tar.gz"
        if path == f"/repos/{REPO}/releases/latest":
            body = {"tag_name": self.version, "assets": [{"name": name, "size": len(self.release)}]}
            return "latest", json.dumps(body).encode(), "application/json"
        if path == f"/{REPO}/releases/download/{self.version}/{name}":
            return name, self.release, "application/gzip"
        if path == f"/{REPO}/releases/download/{self.version}/{name}.sha256":
            return name + ".sha256", f"{self.release_sha}  {name}\n".encode(), "text/plain"
        for repo, entry in self.repos.items():
            revision, files = entry["revision"], entry["files"]
            if path == f"/api/models/{repo}":
                return "info", json.dumps({"id": repo, "sha": revision}).encode(), "application/json"
            if path == f"/api/models/{repo}/tree/{revision}":
                tree = []
                for file, data in sorted(files.items()):
                    item = {"type": "file", "path": file, "size": len(data), "oid": git_oid(data)}
                    if file.endswith(".ninfer") or ".ninfer.part-" in file:  # as LFS files: sha256 in lfs.oid
                        item["oid"] = git_oid(b"pointer " + file.encode())
                        item["lfs"] = {"oid": sha256(data), "size": len(data)}
                    tree.append(item)
                return "tree", json.dumps(tree).encode(), "application/json"
            prefix = f"/{repo}/resolve/{revision}/"
            if path.startswith(prefix) and path[len(prefix) :] in files:
                file = path[len(prefix) :]
                return file, files[file], "application/octet-stream"
        return None

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def make_install(root: Path, models=(), version: str = "v0.12.0", accept: bool = True) -> None:
    """What install.sh leaves in <root>: a release with `current` pointing at it, and model folders."""
    from capy_app import catalog

    release = Path(root) / ("capy-flash-" + version)
    for name, data in release_files().items():
        path = release / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o755 if name.endswith(".sh") or name.startswith("bin/") else 0o644)
    link = Path(root) / "current"
    if not link.exists():
        os.symlink(release.name, link)
    from capy_app.installer import Installer

    for key in models:
        folder = Path(root) / "models" / key
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in model_files(catalog.model(key)["artifact"], size=2048).items():
            (folder / name).write_bytes(data)
        if accept:  # the user's "use it as it is": these tests are about running a model, not about verifying it
            Installer(Path(root)).accept_as_is(key)
