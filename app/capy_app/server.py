"""The app's local HTTP server: the page, a small JSON API, and a streaming proxy to the backend's /v1."""

import http.client
import json
import os
import ssl
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import __version__, catalog, opencode
from .backends import EndpointBackend, has_images, model_vision, strip_images
from .local import Local, Refused
from .security import Guard
from .store import Conversations, Settings

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json",
}
CSP = (
    "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
MAX_BODY = 64 * 1024 * 1024  # images travel inline (data URLs)
# Request headers forwarded to the backend; everything else (Cookie, Origin, Host, ...) stays here.
FORWARD = ("Content-Type", "Accept", "Anthropic-Version", "Anthropic-Beta")
# What POST /api/settings may change. The backend's kind and the engine's settings are not here: "local" is set by
# /api/engine/use (which checks the machine) and the engine has its own, validated endpoint.
PAGE_SETTINGS = (
    "backend",
    "model",
    "thinking",
    "temperature",
    "max_tokens",
    "system_prompt",
    "theme",
    "language",
    "model_limits",
)
PAGE_BACKEND = ("name", "base_url", "api_key")


class App:
    def __init__(
        self, web_dir: Path, settings: Settings, conversations: Conversations, token: str, local: Optional[Local] = None
    ):
        self.web_dir = web_dir
        self.settings = settings
        self.conversations = conversations
        self.token = token
        self.local = local or Local(settings)
        self.guard: Optional[Guard] = None
        self.static = {"/" + p.name: p for p in web_dir.iterdir() if p.is_file() and p.suffix in STATIC_TYPES}
        self.static["/"] = web_dir / "index.html"
        self.server: Optional[ThreadingHTTPServer] = None
        self._vision: Dict[Tuple[str, str], Optional[bool]] = {}  # (server, model id) -> reads images?

    def backend(self) -> EndpointBackend:
        """Where chat requests go: the configured server, or the engine this app runs on this computer."""
        if self.local.is_local():
            return EndpointBackend(self.local.engine.base_url(self.local.port()), "", "capy-flash")
        return EndpointBackend.from_settings(self.settings.get()["backend"])

    def bind(self, host: str, port: int) -> ThreadingHTTPServer:
        app = self

        class Handler(AppHandler):
            pass

        Handler.app = app
        server = ThreadingHTTPServer((host, port), Handler)
        server.daemon_threads = True
        self.server = server
        self.guard = Guard(self.token, server.server_address[1])
        return server

    def models(self) -> list:
        """The backend's models, each with `vision`: True, False, or None when the app cannot tell."""
        backend = self.backend()
        models = backend.models()
        chosen = catalog.model(self.local.chosen()) if self.local.is_local() else None
        for m in models:
            key = (backend.base_url, m["id"])
            if chosen is not None:
                m["vision"] = bool(chosen["vision"])  # the engine on this computer: the catalog knows
            elif self._vision.get(key) is False:
                m["vision"] = False  # learned: the server refused an image for it
            self._vision[key] = m["vision"]
        return models

    def vision_of(self, backend: EndpointBackend, model_id: str) -> Optional[bool]:
        if self.local.is_local():
            chosen = catalog.model(self.local.chosen())
            if chosen is not None:
                return bool(chosen["vision"])
        key = (backend.base_url, model_id)
        return self._vision[key] if key in self._vision else model_vision({"id": model_id})

    def learn_no_vision(self, backend: EndpointBackend, model_id: str) -> None:
        self._vision[(backend.base_url, model_id)] = False

    def shutdown_soon(self) -> None:
        if self.server:
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    def close(self) -> None:
        """The app is quitting: stop the engine it started and any download."""
        self.local.close()


class HttpError(Exception):
    def __init__(self, status: int, message: str, code: str = ""):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


class AppHandler(BaseHTTPRequestHandler):
    app: App
    server_version = "capy-flash-app/" + __version__
    protocol_version = "HTTP/1.0"  # the proxy streams close-delimited bodies

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet: no request lines on stderr
        pass

    # ---- plumbing -------------------------------------------------------------------------------------------
    def _common_headers(self) -> None:
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")

    def _send(self, status: int, body: bytes, content_type: str, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self._common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, value: Any) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str, code: str = "") -> None:
        error = {"message": message}
        if code:
            error["code"] = code  # lets the page say it in the user's language
        self._json(status, {"error": error})

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise HttpError(413, "request too large")
        return self.rfile.read(length) if length > 0 else b""

    def _json_body(self) -> Dict[str, Any]:
        try:
            value = json.loads(self._body() or b"{}")
        except ValueError:
            raise HttpError(400, "the body is not JSON")
        if not isinstance(value, dict):
            raise HttpError(400, "the body must be a JSON object")
        return value

    # ---- entry points ---------------------------------------------------------------------------------------
    def do_GET(self) -> None:
        self._dispatch()

    def do_HEAD(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def do_PUT(self) -> None:
        self._dispatch()

    def do_DELETE(self) -> None:
        self._dispatch()

    def do_OPTIONS(self) -> None:  # no CORS, ever
        self._error(405, "method not allowed")

    def _dispatch(self) -> None:
        guard = self.app.guard
        try:
            if guard is None or not guard.host_ok(self.headers):
                return self._error(421, "unexpected Host")
            url = urllib.parse.urlsplit(self.path)
            path = url.path
            query = urllib.parse.parse_qs(url.query)
            if path.startswith("/api/") or path.startswith("/v1/"):
                if not guard.cookie_ok(self.headers):
                    return self._error(401, "open the app from its launcher (session token missing)")
                if self.command not in ("GET", "HEAD") and not guard.same_origin(self.headers):
                    return self._error(403, "cross-origin request refused")
                if path.startswith("/v1/"):
                    return self._proxy(path[len("/v1") :], url.query)
                return self._api(path, query)
            if self.command not in ("GET", "HEAD"):
                return self._error(405, "method not allowed")
            return self._static(path, query)
        except HttpError as e:
            self._error(e.status, e.message, e.code)
        except Refused as e:
            self._error(e.status, e.message, e.code)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # keep the server up; report the failure to the page
            try:
                self._error(500, f"{type(e).__name__}: {e}")
            except Exception:
                pass

    # ---- static page ----------------------------------------------------------------------------------------
    def _static(self, path: str, query: Dict[str, Any]) -> None:
        if path == "/" and "t" in query:
            if not self.app.guard.token_ok(query["t"][0]):
                return self._error(401, "wrong session token")
            # Set the session cookie, then drop the token from the address bar.
            self.send_response(303)
            self._common_headers()
            self.send_header("Set-Cookie", self.app.guard.set_cookie_header())
            self.send_header("Location", "/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        file = self.app.static.get(path)
        if file is None or not file.is_file():
            return self._error(404, "not found")
        self._send(200, file.read_bytes(), STATIC_TYPES[file.suffix])

    # ---- JSON API -------------------------------------------------------------------------------------------
    def _api(self, path: str, query: Dict[str, Any]) -> None:
        app, method = self.app, self.command
        if path == "/api/state" and method == "GET":
            facts = app.local.machine.facts()
            return self._json(
                200,
                {
                    "version": __version__,
                    "settings": app.settings.public(),
                    "opencode": opencode.detect(),
                    "home": os.path.expanduser("~"),
                    "platform": {
                        "os": facts["os"],
                        "arch": facts["arch"],
                        "can_run_engine": app.local.supported(facts),
                    },
                },
            )
        if path == "/api/settings" and method == "POST":
            change = self._json_body()
            refused = [k for k in change if k not in PAGE_SETTINGS]
            if refused:
                raise HttpError(400, "these settings cannot be changed here: " + ", ".join(sorted(refused)))
            if "backend" in change:
                if not isinstance(change["backend"], dict):
                    raise HttpError(400, "backend must be an object")
                b = {k: v for k, v in change["backend"].items() if k in PAGE_BACKEND}
                if len(b) != len(change["backend"]):
                    raise HttpError(400, "only the server's name, URL and key can be changed here")
                if b.get("api_key") is None:  # null/absent: keep the stored key
                    b.pop("api_key", None)
                try:  # validate the URL before storing it
                    EndpointBackend(b.get("base_url", app.settings.get()["backend"]["base_url"]))
                except ValueError as e:
                    raise HttpError(400, str(e))
                if "base_url" in b:  # a server was chosen: the engine this app may be running here stops
                    b["kind"] = "endpoint"
                    if app.local.is_local():
                        app.local.stop()
                    change["setup_done"] = True
                change["backend"] = b
            try:
                app.settings.update(change)
            except ValueError as e:
                raise HttpError(400, str(e))
            return self._json(200, app.settings.public())
        if path.startswith("/api/setup") or path.startswith("/api/engine") or path.startswith("/api/tune"):
            return self._local_api(path, query)
        if path == "/api/check" and method == "GET":
            return self._json(200, app.backend().check())
        if path == "/api/models" and method == "GET":
            # A server that is off is a state the page shows, not a failed request: 200 with the reason.
            try:
                return self._json(200, {"models": app.models(), "error": None})
            except Exception as e:
                return self._json(200, {"models": [], "error": str(getattr(e, "reason", e))})
        if path == "/api/conversations" and method == "GET":
            return self._json(200, {"conversations": app.conversations.list()})
        if path.startswith("/api/conversations/"):
            cid = path[len("/api/conversations/") :]
            try:
                if method == "GET":
                    return self._json(200, app.conversations.get(cid))
                if method == "PUT":
                    body = self._json_body()
                    body["id"] = cid
                    return self._json(200, app.conversations.save(body))
                if method == "DELETE":
                    app.conversations.delete(cid)
                    return self._json(200, {"ok": True})
            except FileNotFoundError:
                raise HttpError(404, "no such conversation")
            except ValueError as e:
                raise HttpError(400, str(e))
        if path == "/api/folders" and method == "GET":
            return self._json(200, list_folders((query.get("path") or [""])[0]))
        if path == "/api/opencode/configure" and method == "POST":
            return self._json(200, self._opencode_configure(self._json_body()))
        if path == "/api/opencode/launch" and method == "POST":
            body = self._json_body()
            folder = os.path.expanduser(str(body.get("folder") or ""))
            try:
                result = opencode.open_in_terminal(folder, str(body.get("model") or ""))
            except (RuntimeError, ValueError) as e:
                raise HttpError(400, str(e))
            recent = [folder] + [f for f in app.settings.get()["recent_folders"] if f != folder]
            app.settings.update({"recent_folders": recent[:8]})
            return self._json(200, result)
        if path == "/api/snippets" and method == "GET":
            return self._json(200, snippets(app.backend(), (query.get("model") or [""])[0]))
        if path == "/api/quit" and method == "POST":
            self._json(200, {"ok": True})
            app.close()
            app.shutdown_soon()
            return
        raise HttpError(404, "no such API")

    # ---- capy-flash on this computer: setup, engine, models, tuning ------------------------------------------
    def _local_api(self, path: str, query: Dict[str, Any]) -> None:
        """Reading is GET; everything that installs, starts, stops or changes is POST (so: same-origin checked)."""
        local, method = self.app.local, self.command
        if method == "GET":
            if path == "/api/setup":
                value = local.setup()
                value["setup_done"] = bool(self.app.settings.get().get("setup_done"))
                return self._json(200, value)
            if path == "/api/engine":
                return self._json(200, local.snapshot())
            if path == "/api/engine/log":
                try:
                    lines = int((query.get("lines") or ["200"])[0])
                except ValueError:
                    lines = 200
                return self._json(200, {"lines": local.engine.log_tail(lines)})
            if path == "/api/tune/plan":
                long_context = (query.get("long") or ["0"])[0] == "1"
                return self._json(200, local.tune_plan((query.get("model") or [""])[0], long_context))
            raise HttpError(405 if path in LOCAL_POSTS else 404, "use POST" if path in LOCAL_POSTS else "no such API")
        if method != "POST" or path not in LOCAL_POSTS:
            raise HttpError(405 if path in LOCAL_POSTS or path in LOCAL_GETS else 404, "method not allowed")
        body = self._json_body()
        if path == "/api/setup/done":
            self.app.settings.update({"setup_done": True})
        elif path == "/api/engine/use":
            local.use(body.get("model"), as_is=body.get("as_is") is True)
        elif path == "/api/engine/install":
            local.install(body.get("model"), new_folder=body.get("new_folder") is True)
        elif path == "/api/engine/cancel":
            local.installer.cancel(str(body.get("job") or "") or None)
        elif path == "/api/engine/remove":
            local.remove(body.get("model"))
        elif path == "/api/engine/start":
            local.start()
        elif path == "/api/engine/stop":
            local.stop()
        elif path == "/api/engine/settings":
            local.engine_settings(body)
        elif path == "/api/engine/check-update":
            return self._json(200, local.check_update())
        elif path == "/api/engine/update":
            local.update()
        elif path == "/api/tune/start":
            local.tune(bool(body.get("long_context")))
        elif path == "/api/tune/cancel":
            local.tuner.cancel()
        elif path == "/api/tune/revert":
            local.set_profile_active(False)
        elif path == "/api/tune/apply":
            local.set_profile_active(True)
        return self._json(200, local.snapshot())

    def _opencode_configure(self, body: Dict[str, Any]) -> Dict[str, Any]:
        app = self.app
        settings = app.settings.get()
        backend = app.backend()
        try:
            models = app.models()
        except Exception as e:
            raise HttpError(502, "the endpoint did not answer: " + str(getattr(e, "reason", e)))
        config = opencode.build_config(
            backend.name,
            backend.base_url,
            backend.api_key,
            models,
            settings.get("model_limits") or {},
            str(body.get("model") or settings["model"]),
        )
        path = opencode.write_config(config, backend.api_key)
        chat = [m for m in models if m["kind"] == "chat"]
        return {
            "path": str(path),
            "models": len(chat),
            "detect": opencode.detect(),
            "manual": opencode.manual_snippet(config),
        }

    # ---- proxy to the backend -------------------------------------------------------------------------------
    def _proxy(self, sub_path: str, query: str) -> None:
        backend = self.app.backend()
        target = urllib.parse.urlsplit(backend.url(sub_path) + ("?" + query if query else ""))
        body = self._body() if self.command in ("POST", "PUT") else None
        headers = {k: self.headers[k] for k in FORWARD if self.headers.get(k)}
        headers.update(backend.headers())
        # Images go only to models that read them. A model without vision refuses the whole request (400), and
        # since the image stays in the history, every later turn too: so they are left out of what is sent.
        payload, model_id, images, dropped = None, "", False, False
        if self.command == "POST" and sub_path == "/chat/completions" and body:
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if isinstance(payload, dict) and has_images(payload.get("messages")):
                model_id, images = str(payload.get("model") or ""), True
                if self.app.vision_of(backend, model_id) is False:
                    body, dropped = _without_images(payload), True

        def send(data: Optional[bytes]) -> Tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
            connection = _connection(target)
            try:
                connection.request(
                    self.command, target.path + ("?" + target.query if target.query else ""), body=data, headers=headers
                )
                return connection, connection.getresponse()
            except (OSError, http.client.HTTPException) as e:
                connection.close()
                raise HttpError(502, f"the endpoint {backend.base_url} did not answer: {e}")

        conn, upstream = send(body)
        first = b""
        try:
            if upstream.status == 400 and images and not dropped:
                # A model the app knew nothing about: if it says it has no vision, remember that and send the
                # conversation again without the images.
                first = upstream.read(65536)
                if b"vision" in first.lower():
                    conn.close()
                    self.app.learn_no_vision(backend, model_id)
                    conn, upstream = send(_without_images(payload))
                    first, dropped = b"", True
            self.send_response(upstream.status)
            self._common_headers()
            for k in ("Content-Type", "X-Request-Id"):
                if upstream.getheader(k):
                    self.send_header(k, upstream.getheader(k))
            if dropped:
                self.send_header("X-Capy-Images", "dropped")  # the page says so under the answer
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            if first:
                self.wfile.write(first)
                self.wfile.flush()
            while True:
                chunk = upstream.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            conn.close()  # also when the page stopped reading: the backend sees the cancel


def _without_images(payload: Dict[str, Any]) -> bytes:
    return json.dumps(dict(payload, messages=strip_images(payload["messages"])), ensure_ascii=False).encode("utf-8")


LOCAL_GETS = ("/api/setup", "/api/engine", "/api/engine/log", "/api/tune/plan")
LOCAL_POSTS = (
    "/api/setup/done",
    "/api/engine/use",
    "/api/engine/install",
    "/api/engine/cancel",
    "/api/engine/remove",
    "/api/engine/start",
    "/api/engine/stop",
    "/api/engine/settings",
    "/api/engine/check-update",
    "/api/engine/update",
    "/api/tune/start",
    "/api/tune/cancel",
    "/api/tune/revert",
    "/api/tune/apply",
)


def _connection(target: urllib.parse.SplitResult) -> http.client.HTTPConnection:
    port = target.port or (443 if target.scheme == "https" else 80)
    if target.scheme == "https":
        return http.client.HTTPSConnection(target.hostname, port, timeout=900, context=ssl.create_default_context())
    return http.client.HTTPConnection(target.hostname, port, timeout=900)


def list_folders(path: str) -> Dict[str, Any]:
    """Sub-folders of `path` (default: the home folder), for picking an OpenCode project."""
    p = Path(os.path.expanduser(path or "~")).resolve()
    while not p.is_dir() and p.parent != p:  # a folder that is gone (a recent project, deleted): its nearest parent
        p = p.parent
    dirs = []
    try:
        for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if child.name.startswith(".") or not child.is_dir():
                continue
            dirs.append(child.name)
    except PermissionError:
        pass
    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "dirs": dirs[:500], "git": (p / ".git").exists()}


def snippets(backend: EndpointBackend, model: str) -> Dict[str, str]:
    model = model or "MODEL"
    key = backend.api_key and "$CAPY_FLASH_API_KEY"
    auth = f' \\\n  -H "Authorization: Bearer {key}"' if key else ""
    py_key = 'os.environ["CAPY_FLASH_API_KEY"]' if key else '"none"'
    py_import = "import os\n" if key else ""
    return {
        "curl": (
            f"curl {backend.url('/chat/completions')}{auth} \\\n  -H 'Content-Type: application/json' \\\n"
            f'  -d \'{{"model": "{model}", "messages": [{{"role": "user", "content": "Hello"}}]}}\''
        ),
        "python": (
            py_import + "from openai import OpenAI\n\n"
            f'client = OpenAI(base_url="{backend.base_url}", api_key={py_key})\n'
            f'reply = client.chat.completions.create(model="{model}",\n'
            '    messages=[{"role": "user", "content": "Hello"}])\n'
            "print(reply.choices[0].message.content)"
        ),
        "claude_code": (
            f"ANTHROPIC_BASE_URL={backend.anthropic_base()} \\\n"
            f"ANTHROPIC_AUTH_TOKEN={key or 'none'} \\\n"
            f"claude --model {model}"
        ),
    }


def make_app(
    web_dir: Path, settings: Settings, conversations: Conversations, token: str, local: Optional[Local] = None
) -> App:
    return App(web_dir, settings, conversations, token, local)


def bind(app: App, host: str = "127.0.0.1", port: int = 0) -> Tuple[ThreadingHTTPServer, int]:
    server = app.bind(host, port)
    return server, server.server_address[1]
