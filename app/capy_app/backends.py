"""Where the models live.

EndpointBackend: any OpenAI-compatible server (capy-flash's ninfer-serve, llama-swap, vramble, ...). The engine the
app runs on this computer (engine.py) is reached the same way, on its local port.
"""

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# Models that do not chat (an endpoint such as llama-swap also serves speech, OCR or embeddings).
_NOT_CHAT = re.compile(
    r"(^|[-_ ])(asr|tts|whisper|embed|embedding|rerank|reranker|ocr|stt|speech)([-_ ]|$)", re.IGNORECASE
)


def model_kind(entry: Dict[str, Any]) -> str:
    text = " ".join(str(entry.get(k, "")) for k in ("id", "name"))
    return "other" if _NOT_CHAT.search(text) else "chat"


_FLASH_NEXT = re.compile(r"flash[-_ ]?next", re.IGNORECASE)
_QWEN_27B = re.compile(r"qwen[-_ ]?(3[._]8)?[-_ ]?27b", re.IGNORECASE)
IMAGE_PARTS = ("image_url", "image", "input_image")
OMITTED = "(image not sent: this model does not read images)"


def model_vision(entry: Dict[str, Any]) -> Optional[bool]:
    """Does this model read images? True / False for the capy-flash models, recognised by id or name; None when
    the app cannot tell (any other model of an endpoint).

    The 27B artifacts carry the vision tower and their launchers load it. Flash-Next is text only unless it is
    the separate Vision artifact: an image sent to it is refused (400, vision disabled)."""
    text = " ".join(str(entry.get(k, "")) for k in ("id", "name"))
    if _FLASH_NEXT.search(text):
        return "vision" in text.lower()
    if _QWEN_27B.search(text):
        return True
    if str(entry.get("id", "")).lower() == "capy-flash":  # the launcher's model id for Flash-Next
        return False
    return None


def has_images(messages: Any) -> bool:
    return isinstance(messages, list) and any(
        isinstance(m, dict)
        and isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in m["content"])
        for m in messages
    )


def strip_images(messages: List[Any]) -> List[Any]:
    """The same conversation without its image parts, for a model that does not read images. The text of every
    turn stays; a turn that was only an image keeps a short note in its place, so the turns still alternate."""
    out = []
    for m in messages:
        if not (isinstance(m, dict) and isinstance(m.get("content"), list)):
            out.append(m)
            continue
        kept = [p for p in m["content"] if not (isinstance(p, dict) and p.get("type") in IMAGE_PARTS)]
        if len(kept) == len(m["content"]):
            out.append(m)
            continue
        texts = [p.get("text", "") for p in kept if isinstance(p, dict) and p.get("type") == "text"]
        if len(texts) == len(kept):  # only text is left: the plain form every server accepts
            content: Any = "\n".join(t for t in texts if t) or OMITTED
        else:
            content = kept
        out.append(dict(m, content=content))
    return out


class EndpointBackend:
    def __init__(self, base_url: str, api_key: str = "", name: str = ""):
        base_url = (base_url or "").strip().rstrip("/")
        if not re.match(r"^https?://", base_url):
            raise ValueError("the endpoint URL must start with http:// or https://")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self.base_url = base_url
        self.api_key = api_key or ""
        self.name = name or "endpoint"

    @classmethod
    def from_settings(cls, backend: Dict[str, Any]) -> "EndpointBackend":
        return cls(backend.get("base_url", ""), backend.get("api_key", ""), backend.get("name", ""))

    def headers(self) -> Dict[str, str]:
        return {"Authorization": "Bearer " + self.api_key} if self.api_key else {}

    def url(self, path: str) -> str:
        """`path` is what follows /v1 (e.g. "/chat/completions")."""
        return self.base_url + path

    def anthropic_base(self) -> str:
        """The base for Anthropic clients (they append /v1/messages themselves)."""
        return self.base_url[: -len("/v1")]

    def models(self, timeout: float = 10.0) -> List[Dict[str, Any]]:
        req = urllib.request.Request(self.url("/models"), headers=self.headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        out = []
        for m in data.get("data", []):
            if not isinstance(m, dict) or not m.get("id"):
                continue
            out.append(
                {
                    "id": m["id"],
                    "name": m.get("name") or m["id"],
                    "description": m.get("description") or "",
                    "kind": model_kind(m),
                    "vision": model_vision(m),
                    "context": _context_of(m),
                }
            )
        out.sort(key=lambda m: (m["kind"] != "chat", m["id"]))
        return out

    def check(self, timeout: float = 5.0) -> Dict[str, Any]:
        try:
            models = self.models(timeout=timeout)
            return {"ok": True, "models": len(models)}
        except urllib.error.HTTPError as e:
            return {
                "ok": False,
                "error": f"HTTP {e.code}" + (" (wrong or missing API key?)" if e.code in (401, 403) else ""),
            }
        except Exception as e:  # unreachable, refused, timeout, not JSON
            return {"ok": False, "error": str(getattr(e, "reason", e))}


def _context_of(m: Dict[str, Any]) -> Optional[int]:
    for key in ("max_model_len", "context_length", "max_context", "context_window"):
        value = m.get(key)
        if isinstance(value, int) and value > 0:
            return value
    meta = m.get("meta")
    if isinstance(meta, dict):
        for key in ("max_model_len", "context_length", "n_ctx_train", "n_ctx"):
            value = meta.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return None
