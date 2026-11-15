"""Downloads with the standard library: HTTPS only, resumable, verified before anything is used.

A file is written as <name>.part, continued with a Range request when a part is already there, hashed in full once
complete and compared with the published digest, and only then renamed to its final name. A file that does not match
is deleted. Cancelling keeps the part, so the next attempt continues from it.
"""

import hashlib
import http.client
import json
import os
import shutil
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import __version__

CHUNK = 1 << 20
Progress = Optional[Callable[[int], None]]


class DownloadError(Exception):
    """code: insecure | http | network | disk | checksum | cancelled | layout"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    allow_http = False

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not self.allow_http and urllib.parse.urlsplit(newurl).scheme != "https":
            raise DownloadError("insecure", "refused a redirect to a non-HTTPS address")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Fetcher:
    """HTTP GETs for the installer. `allow_http` exists for the tests' local server only: nothing the page or the
    settings can say reaches it."""

    def __init__(self, allow_http: bool = False, timeout: float = 30.0, retries: int = 4, backoff: float = 1.0):
        self.allow_http = allow_http
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        redirects = _HttpsOnlyRedirects()
        redirects.allow_http = allow_http
        self._opener = urllib.request.build_opener(
            redirects, urllib.request.HTTPSHandler(context=ssl.create_default_context())
        )
        self._opener.addheaders = [("User-Agent", "capy-flash-app/" + __version__)]

    def open(self, url: str, headers: Optional[Dict[str, str]] = None):
        scheme = urllib.parse.urlsplit(url).scheme
        if scheme != "https" and not (self.allow_http and scheme == "http"):
            raise DownloadError("insecure", "only HTTPS downloads are allowed: " + url)
        try:
            return self._opener.open(urllib.request.Request(url, headers=headers or {}), timeout=self.timeout)
        except urllib.error.HTTPError:
            raise
        except DownloadError:
            raise
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            raise DownloadError("network", f"{_host(url)} did not answer: {getattr(e, 'reason', e)}")

    def bytes(self, url: str, limit: int = 8 << 20) -> bytes:
        try:
            with self.open(url) as r:
                data = r.read(limit + 1)
        except urllib.error.HTTPError as e:
            raise DownloadError("http", f"{_host(url)} answered HTTP {e.code} for {urllib.parse.urlsplit(url).path}")
        except (OSError, http.client.HTTPException) as e:
            raise DownloadError("network", f"{_host(url)}: {e}")
        if len(data) > limit:
            raise DownloadError("layout", "unexpectedly large answer from " + _host(url))
        return data

    def text(self, url: str, limit: int = 8 << 20) -> str:
        return self.bytes(url, limit).decode("utf-8", errors="replace")

    def json(self, url: str) -> Any:
        try:
            return json.loads(self.text(url))
        except ValueError:
            raise DownloadError("layout", _host(url) + " did not answer with JSON")

    def download(
        self,
        url: str,
        dest: Path,
        size: Optional[int] = None,
        progress: Progress = None,
        cancel: Optional[threading.Event] = None,
    ) -> Path:
        """Fetch `url` into `<dest>.part`, continuing a part that is already there. Returns the part's path: the
        caller verifies it and renames it. `progress(n)` gets the bytes of every chunk written; the bytes of a part
        found on disk are the caller's to count, and are given back (negative n) if the part has to be dropped."""
        part = part_path(dest)
        part.parent.mkdir(parents=True, exist_ok=True)
        failures = 0
        while True:
            have = part.stat().st_size if part.exists() else 0
            if size is not None and have > size:  # a part from another file or version
                part.unlink()
                if progress:
                    progress(-have)
                have = 0
            if size is not None and have == size:
                return part
            try:
                got = self._attempt(url, part, have, size, progress, cancel)
                if got:
                    return part
            except DownloadError as e:
                if e.code != "network":
                    raise
                now = part.stat().st_size if part.exists() else 0
                failures = 0 if now > have else failures + 1  # an attempt that moved forward is not a failure
                if failures > self.retries:
                    raise
                pause = self.backoff * min(failures + 1, 5)
                if cancel is None:
                    time.sleep(pause)
                elif cancel.wait(pause):
                    raise DownloadError("cancelled", "cancelled")

    def _attempt(
        self,
        url: str,
        part: Path,
        have: int,
        size: Optional[int],
        progress: Progress,
        cancel: Optional[threading.Event],
    ) -> bool:
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            response = self.open(url, headers)
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:  # the part is not a prefix of this file (or is already all of it)
                part.unlink()
                if progress:
                    progress(-have)
                raise DownloadError("network", "the server refused the range; starting again")
            transient = e.code >= 500 or e.code in (408, 429)  # worth another try, unlike a 404
            raise DownloadError(
                "network" if transient else "http",
                f"{_host(url)} answered HTTP {e.code} for {urllib.parse.urlsplit(url).path}",
            )
        with response:
            mode = "ab"
            if have and response.status == 206:
                start = _range_start(response.headers.get("Content-Range"))
                if start != have:
                    raise DownloadError("network", "the server answered another range than the one asked")
            elif have:  # 200: the server ignored the range and sends the file from its first byte
                mode = "wb"
                if progress:
                    progress(-have)
            announced = response.headers.get("Content-Length")
            received = 0
            with open(part, mode) as f:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise DownloadError("cancelled", "cancelled")
                    try:
                        chunk = response.read1(CHUNK)  # what has arrived: steady progress, prompt cancel
                    except (OSError, http.client.HTTPException, socket.timeout) as e:
                        raise DownloadError("network", f"the connection to {_host(url)} broke: {e}")
                    if not chunk:
                        break
                    try:
                        f.write(chunk)
                    except OSError as e:
                        raise DownloadError("disk", f"cannot write {part.name}: {e}")
                    received += len(chunk)
                    if progress:
                        progress(len(chunk))
        # A connection that closes early ends the read without an error: compare with what was announced.
        short = announced is not None and announced.isdigit() and received < int(announced)
        if short or (size is not None and part.stat().st_size < size):
            raise DownloadError("network", f"the connection to {_host(url)} closed early")
        return True


def part_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part")


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).netloc or url


def _range_start(content_range: Optional[str]) -> int:
    try:  # "bytes 100-199/200"
        return int((content_range or "").split()[1].split("-")[0])
    except (IndexError, ValueError):
        return -1


def sha256_file(path: Path, progress: Progress = None, cancel: Optional[threading.Event] = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if cancel is not None and cancel.is_set():
                raise DownloadError("cancelled", "cancelled")
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
            if progress:
                progress(len(chunk))
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """The object id git (and Hugging Face) give a file that is not in LFS."""
    h = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_of(path: Path, expected: str, progress: Progress = None, cancel: Optional[threading.Event] = None) -> str:
    """The digest of `path` in the form of `expected`: SHA-256 (64 hex digits) or a git blob id (40)."""
    return git_blob_sha1(path) if len(expected) == 40 else sha256_file(path, progress, cancel)


def verify_and_keep(
    part: Path, dest: Path, expected: str, progress: Progress = None, cancel: Optional[threading.Event] = None
) -> None:
    """Rename the part to its final name only if its digest is the published one; otherwise delete it."""
    actual = digest_of(part, expected, progress, cancel)
    if actual != expected.lower():
        try:
            part.unlink()
        except OSError:
            pass
        raise DownloadError(
            "checksum",
            f"{dest.name} does not match its published checksum (the download was corrupt and has been deleted)",
        )
    os.replace(part, dest)


def parse_sums(text: str) -> Dict[str, str]:
    """`sha256sum` output: "<digest>  <path>" per line (also "<digest> *<path>")."""
    sums: Dict[str, str] = {}
    for line in text.splitlines():
        digest, _, name = line.strip().partition(" ")
        name = name.strip().lstrip("*")
        if name.startswith("./"):
            name = name[2:]
        if len(digest) == 64 and all(c in "0123456789abcdefABCDEF" for c in digest) and name:
            sums[name] = digest.lower()
    return sums


def safe_relative(name: str) -> bool:
    """A path that stays inside the folder it is joined to."""
    if not name or name.startswith(("/", "\\")) or "\\" in name or "\0" in name:
        return False
    parts = name.split("/")
    return all(p not in ("", ".", "..") for p in parts)


def ensure_space(directory: Path, need: int, margin: int = 1 << 30) -> None:
    probe = directory
    while not probe.is_dir():
        probe = probe.parent
    free = shutil.disk_usage(str(probe)).free
    if free < need + margin:
        raise DownloadError(
            "disk",
            "not enough free disk in %s: %.1f GB needed, %.1f GB free" % (directory, (need + margin) / 1e9, free / 1e9),
        )
