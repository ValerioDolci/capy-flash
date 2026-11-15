"""Who may talk to the app's local HTTP server.

A page from any website can send requests to 127.0.0.1, and a rebound DNS name can make them same-origin. So:
- the Host header must name this server (127.0.0.1:<port> or localhost:<port>): stops DNS rebinding;
- /api and /v1 need the per-launch session token, carried by an HttpOnly SameSite=Strict cookie that only the
  launcher's first URL (/?t=<token>) can set;
- requests that change state must not come from another origin (Origin / Sec-Fetch-Site);
- no CORS headers are ever sent.
"""

import hmac
import secrets
from http.cookies import SimpleCookie
from typing import Mapping, Optional

COOKIE = "capy_t"


def new_token() -> str:
    return secrets.token_urlsafe(24)


class Guard:
    def __init__(self, token: str, port: int):
        self.token = token
        self.port = port
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        self.origins = {f"http://{h}" for h in self.hosts}

    def host_ok(self, headers: Mapping[str, str]) -> bool:
        return (headers.get("Host") or "").strip().lower() in self.hosts

    def token_ok(self, value: Optional[str]) -> bool:
        return bool(value) and hmac.compare_digest(value.encode(), self.token.encode())

    def cookie_ok(self, headers: Mapping[str, str]) -> bool:
        raw = headers.get("Cookie")
        if not raw:
            return False
        jar = SimpleCookie()
        try:
            jar.load(raw)
        except Exception:
            return False
        morsel = jar.get(COOKIE)
        return morsel is not None and self.token_ok(morsel.value)

    def same_origin(self, headers: Mapping[str, str]) -> bool:
        """For requests that change state: refuse anything a browser marks as coming from elsewhere."""
        site = (headers.get("Sec-Fetch-Site") or "").lower()
        if site and site not in ("same-origin", "none"):
            return False
        origin = headers.get("Origin")
        if origin is not None and origin.lower() not in self.origins:
            return False
        return True

    def set_cookie_header(self) -> str:
        return f"{COOKIE}={self.token}; Path=/; HttpOnly; SameSite=Strict"
