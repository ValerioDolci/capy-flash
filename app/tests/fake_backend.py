"""A tiny OpenAI-compatible server for the tests: /v1/models and a streaming /v1/chat/completions."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = [
    {"id": "qwen27b", "object": "model", "name": "Qwen3.8-27B", "max_model_len": 262144},
    {"id": "flashnext", "object": "model", "name": "Qwen3.8-Flash-Next"},
    {"id": "embed", "object": "model", "name": "Qwen3-Embedding-8B Q8_0"},
]


class FakeBackend:
    def __init__(self, api_key: str = "", delay: float = 0.15):
        self.api_key = api_key
        self.delay = delay
        self.requests = []  # (method, path, headers, body)
        self.aborted = threading.Event()  # a stream whose reader went away
        self.parts = 3
        self.no_vision = set()  # model ids that answer 400 vision_disabled to a request with an image
        self.bad_request = ""  # a 400 with this message for every chat request (not about images)
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _auth_ok(self):
                return not fake.api_key or self.headers.get("Authorization") == "Bearer " + fake.api_key

            def do_GET(self):
                fake.requests.append(("GET", self.path, dict(self.headers), b""))
                if not self._auth_ok():
                    return self._plain(401, b'{"error":"unauthorized"}')
                if self.path == "/v1/models":
                    return self._plain(200, json.dumps({"object": "list", "data": MODELS}).encode())
                self._plain(404, b"{}")

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                fake.requests.append(("POST", self.path, dict(self.headers), body))
                if not self._auth_ok():
                    return self._plain(401, b'{"error":"unauthorized"}')
                if self.path != "/v1/chat/completions":
                    return self._plain(404, b"{}")
                req = json.loads(body)
                if fake.bad_request:
                    return self._plain(400, json.dumps({"error": {"message": fake.bad_request}}).encode())
                images = any(
                    isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
                    for m in req.get("messages", [])
                )
                if images and req.get("model") in fake.no_vision:
                    return self._plain(
                        400,
                        json.dumps(
                            {
                                "error": {
                                    "code": "vision_disabled",
                                    "message": "this server was started without --vision",
                                }
                            }
                        ).encode(),
                    )
                if not req.get("stream"):
                    return self._plain(
                        200,
                        json.dumps(
                            {"choices": [{"message": {"role": "assistant", "content": "Hello there."}}]}
                        ).encode(),
                    )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                parts = [{"reasoning_content": "Thinking."}] + [{"content": " word"}] * (fake.parts - 1)
                try:
                    for p in parts:
                        self._chunk("data: " + json.dumps({"choices": [{"delta": p}]}) + "\n\n")
                        time.sleep(fake.delay)
                except (BrokenPipeError, ConnectionResetError):
                    fake.aborted.set()
                    return
                self._chunk(
                    "data: "
                    + json.dumps({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}})
                    + "\n\n"
                )
                self._chunk("data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

            def _chunk(self, text):
                data = text.encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()

            def _plain(self, status, data):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.handle_error = lambda *a: None  # a client that hangs up is part of the tests
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
