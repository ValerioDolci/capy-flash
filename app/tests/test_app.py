"""capy-flash app tests (standard library only): python3 -m unittest discover -s tests (from app/)."""

import http.client
import json
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capy_app import backends, opencode, paths  # noqa: E402
from capy_app.server import bind, make_app  # noqa: E402
from capy_app.store import Conversations, Settings  # noqa: E402
from fake_backend import FakeBackend  # noqa: E402

WEB = Path(__file__).resolve().parent.parent / "web"


class AppCase(unittest.TestCase):
    api_key = ""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._saved = (paths.config_dir, paths.runtime_dir, paths.data_dir)
        paths.config_dir = lambda: root / "config"
        paths.runtime_dir = lambda: root / "run"
        paths.data_dir = lambda: root / "data"
        self.fake = FakeBackend(api_key=self.api_key)
        self.settings = Settings(root / "config")
        self.settings.update({"backend": {"base_url": self.fake.url, "api_key": self.api_key, "name": "fake"}})
        self.token = "test-token-123"
        self.app = make_app(WEB, self.settings, Conversations(root / "data"), self.token)
        self.server, self.port = bind(self.app, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.cookie = f"capy_t={self.token}"
        self.origin = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.fake.close()
        paths.config_dir, paths.runtime_dir, paths.data_dir = self._saved
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None, host=None, cookie=True):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Host": host or f"127.0.0.1:{self.port}"}
        if cookie:
            h["Cookie"] = self.cookie
        if body is not None:
            body = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
            h["Origin"] = self.origin
        h.update(headers or {})
        conn.request(method, path, body=body, headers=h)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, dict(r.getheaders()), data


class TestSecurity(AppCase):
    def test_wrong_host_refused(self):
        status, _, _ = self.request("GET", "/api/state", host="evil.example:80")
        self.assertEqual(status, 421)
        status, _, _ = self.request("GET", "/", host=f"rebind.example:{self.port}")
        self.assertEqual(status, 421)

    def test_api_needs_session_cookie(self):
        self.assertEqual(self.request("GET", "/api/state", cookie=False)[0], 401)
        self.assertEqual(self.request("GET", "/v1/models", cookie=False)[0], 401)
        status, _, _ = self.request("GET", "/api/state", headers={"Cookie": "capy_t=wrong"}, cookie=False)
        self.assertEqual(status, 401)

    def test_token_url_sets_cookie_and_redirects(self):
        status, headers, _ = self.request("GET", f"/?t={self.token}", cookie=False)
        self.assertEqual(status, 303)
        self.assertEqual(headers["Location"], "/")
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", headers["Set-Cookie"])
        self.assertEqual(self.request("GET", "/?t=nope", cookie=False)[0], 401)

    def test_cross_origin_writes_refused(self):
        for extra in ({"Origin": "https://evil.example"}, {"Sec-Fetch-Site": "cross-site"}):
            status, _, _ = self.request(
                "POST", "/api/settings", headers=dict(extra, **{"Content-Type": "application/json"})
            )
            self.assertEqual(status, 403, extra)
        self.assertEqual(self.request("POST", "/api/settings", body={"theme": "light"})[0], 200)
        self.assertEqual(self.request("OPTIONS", "/api/settings")[0], 405)

    def test_static_headers(self):
        status, headers, data = self.request("GET", "/", cookie=False)
        self.assertEqual(status, 200)
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(self.request("GET", "/../capy_app/server.py", cookie=False)[0], 404)


class TestSettingsAndModels(AppCase):
    def test_api_key_never_reaches_the_page(self):
        self.settings.update({"backend": {"api_key": "sekret"}})
        _, _, data = self.request("GET", "/api/state")
        self.assertNotIn(b"sekret", data)
        self.assertTrue(json.loads(data)["settings"]["backend"]["has_api_key"])
        # null keeps the stored key
        self.request("POST", "/api/settings", body={"backend": {"name": "x", "api_key": None}})
        self.assertEqual(self.settings.get()["backend"]["api_key"], "sekret")

    def test_bad_url_refused(self):
        status, _, _ = self.request("POST", "/api/settings", body={"backend": {"base_url": "file:///etc/passwd"}})
        self.assertEqual(status, 400)
        self.assertEqual(self.request("POST", "/api/settings", body={"nope": 1})[0], 400)

    def test_models_and_kinds(self):
        status, _, data = self.request("GET", "/api/models")
        self.assertEqual(status, 200)
        models = {m["id"]: m for m in json.loads(data)["models"]}
        self.assertEqual(models["qwen27b"]["kind"], "chat")
        self.assertEqual(models["qwen27b"]["context"], 262144)
        self.assertEqual(models["embed"]["kind"], "other")

    def test_settings_file_private(self):
        mode = stat.S_IMODE(os.stat(self.settings.path).st_mode)
        self.assertEqual(mode, 0o600)


class TestProxy(AppCase):
    def test_stream_is_not_buffered(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        body = json.dumps({"model": "qwen27b", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        t0 = time.time()
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={
                "Host": f"127.0.0.1:{self.port}",
                "Cookie": self.cookie,
                "Origin": self.origin,
                "Content-Type": "application/json",
            },
        )
        r = conn.getresponse()
        self.assertEqual(r.status, 200)
        first = r.read1(65536)
        t_first = time.time() - t0
        rest = r.read()
        t_all = time.time() - t0
        conn.close()
        self.assertIn(b"reasoning_content", first)
        self.assertLess(t_first, t_all - 0.2, "the first event waited for the whole answer")
        self.assertIn(b"[DONE]", first + rest)
        # the app's cookie and origin stay here
        _, _, headers, _ = self.fake.requests[-1]
        self.assertNotIn("Cookie", headers)
        self.assertNotIn("Origin", headers)

    def test_stop_reaches_the_backend(self):
        # The page aborts the fetch (Stop): the app must drop the upstream connection, so the engine cancels.
        self.fake.parts = 200
        self.fake.delay = 0.05
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        body = json.dumps({"model": "qwen27b", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={
                "Host": f"127.0.0.1:{self.port}",
                "Cookie": self.cookie,
                "Origin": self.origin,
                "Content-Type": "application/json",
            },
        )
        r = conn.getresponse()
        r.read1(65536)
        r.close()  # the response owns the socket (close-delimited body)
        conn.close()
        self.assertTrue(self.fake.aborted.wait(5), "the backend kept streaming after the page stopped")

    def test_unreachable_backend(self):
        self.settings.update({"backend": {"base_url": "http://127.0.0.1:1/v1"}})
        status, _, data = self.request("GET", "/api/models")
        self.assertEqual((status, json.loads(data)["models"]), (200, []))
        self.assertTrue(json.loads(data)["error"])
        _, _, data = self.request("GET", "/api/check")
        self.assertFalse(json.loads(data)["ok"])


IMAGE = "data:image/png;base64,iVBORw0KGgo="


def with_image(text):
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": IMAGE}}],
    }


class TestImages(AppCase):
    """Images go only to models that read them: Flash-Next has no vision, and one image in the history would make
    every later turn of the conversation fail with 400 vision_disabled."""

    def chat(self, model, messages):
        return self.request("POST", "/v1/chat/completions", body={"model": model, "messages": messages})

    def sent(self):
        return json.loads(self.fake.requests[-1][3])["messages"]

    def test_which_models_read_images(self):
        vision = lambda **m: backends.model_vision(m)
        for m in (
            {"id": "qwen27b"},
            {"id": "qwen27b-fast"},
            {"id": "qwen3.8-27b"},
            {"id": "x", "name": "Qwen3.8-27B QUASAR NVFP4"},
            {"id": "qwen3_8_27b_gsq3"},
            {"id": "flashnext-vision"},
            {"id": "y", "name": "Qwen3.8-Flash-Next NVFP4 Vision"},
        ):
            self.assertIs(vision(**m), True, m)
        for m in (
            {"id": "flashnext"},
            {"id": "flashnext-64gb"},
            {"id": "capy-flash"},
            {"id": "z", "name": "Qwen3.8-Flash-Next NVFP4"},
        ):
            self.assertIs(vision(**m), False, m)
        for m in ({"id": "llama-3.3-70b"}, {"id": "gemma3"}, {"id": "embed"}):
            self.assertIsNone(vision(**m), m)
        models = {m["id"]: m["vision"] for m in json.loads(self.request("GET", "/api/models")[2])["models"]}
        self.assertEqual((models["qwen27b"], models["flashnext"]), (True, False))

    def test_strip_images_keeps_the_text_and_the_turns(self):
        history = [
            {"role": "system", "content": "Be brief."},
            with_image("What is in the picture?"),
            {"role": "assistant", "content": "A capybara."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMAGE}}]},
            {"role": "user", "content": "And now?"},
        ]
        self.assertTrue(backends.has_images(history))
        out = backends.strip_images(history)
        self.assertFalse(backends.has_images(out))
        self.assertEqual([m["role"] for m in out], [m["role"] for m in history])
        self.assertEqual(out[1]["content"], "What is in the picture?")
        self.assertEqual(out[3]["content"], backends.OMITTED, "a turn that was only an image is not left empty")
        self.assertEqual((out[0], out[2], out[4]), (history[0], history[2], history[4]))
        self.assertEqual(history[1]["content"][1]["type"], "image_url", "the caller's history is not modified")
        self.assertFalse(backends.has_images([{"role": "user", "content": "plain"}]))

    def test_history_with_an_image_still_works_on_a_model_without_vision(self):
        # A conversation started on qwen27b with an image, continued on flashnext.
        self.fake.no_vision = {"flashnext"}
        history = [
            with_image("What is in the picture?"),
            {"role": "assistant", "content": "A capybara."},
            {"role": "user", "content": "Tell me more about it."},
        ]
        status, headers, _ = self.chat("flashnext", history)
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-Capy-Images"), "dropped")
        self.assertEqual(
            self.sent(),
            [
                {"role": "user", "content": "What is in the picture?"},
                {"role": "assistant", "content": "A capybara."},
                {"role": "user", "content": "Tell me more about it."},
            ],
        )
        self.assertEqual(len([r for r in self.fake.requests if r[0] == "POST"]), 1, "known without asking the server")

    def test_images_reach_a_model_with_vision_untouched(self):
        status, headers, _ = self.chat("qwen27b", [with_image("What is in the picture?")])
        self.assertEqual(status, 200)
        self.assertNotIn("X-Capy-Images", headers)
        self.assertEqual(self.sent(), [with_image("What is in the picture?")])

    def test_unknown_model_that_refuses_images_is_retried_without_them_and_remembered(self):
        self.fake.no_vision = {"mystery-7b"}
        status, headers, _ = self.chat("mystery-7b", [with_image("Look.")])
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-Capy-Images"), "dropped")
        posts = [json.loads(r[3])["messages"] for r in self.fake.requests if r[0] == "POST"]
        self.assertEqual(posts, [[with_image("Look.")], [{"role": "user", "content": "Look."}]])
        self.chat("mystery-7b", [with_image("Again.")])  # now the app knows: no refused attempt first
        self.assertEqual(len([r for r in self.fake.requests if r[0] == "POST"]), 3)
        self.assertEqual(self.sent(), [{"role": "user", "content": "Again."}])

    def test_other_errors_pass_through(self):
        self.fake.bad_request = "max_tokens is too large"
        status, headers, data = self.chat("mystery-7b", [with_image("Look.")])
        self.assertEqual(status, 400)
        self.assertIn(b"max_tokens is too large", data)
        self.assertNotIn("X-Capy-Images", headers)
        self.assertEqual(len([r for r in self.fake.requests if r[0] == "POST"]), 1)
        status, _, data = self.chat("flashnext", [{"role": "user", "content": "no image here"}])
        self.assertEqual((status, b"max_tokens" in data), (400, True))


class TestProxyWithKey(AppCase):
    api_key = "backend-key"

    def test_key_added_by_the_app(self):
        status, _, _ = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        _, _, headers, _ = self.fake.requests[-1]
        self.assertEqual(headers.get("Authorization"), "Bearer backend-key")

    def test_wrong_key_reported(self):
        self.settings.update({"backend": {"api_key": "wrong"}})
        _, _, data = self.request("GET", "/api/check")
        self.assertIn("401", json.loads(data)["error"])


class TestConversations(AppCase):
    def test_crud(self):
        conv = {"title": "Test", "model": "qwen27b", "messages": [{"role": "user", "content": "hi"}]}
        status, _, data = self.request("PUT", "/api/conversations/abc123", body=conv)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["id"], "abc123")
        listing = json.loads(self.request("GET", "/api/conversations")[2])["conversations"]
        self.assertEqual([c["id"] for c in listing], ["abc123"])
        self.assertEqual(json.loads(self.request("GET", "/api/conversations/abc123")[2])["title"], "Test")
        self.assertEqual(self.request("PUT", "/api/conversations/..%2Fx", body=conv)[0], 400)
        self.assertEqual(self.request("DELETE", "/api/conversations/abc123", body={})[0], 200)
        self.assertEqual(self.request("GET", "/api/conversations/abc123")[0], 404)


class TestOpenCode(AppCase):
    def test_config(self):
        models = [
            {"id": "qwen27b", "name": "Q", "kind": "chat", "context": 262144},
            {"id": "embed", "name": "E", "kind": "other", "context": None},
        ]
        cfg = opencode.build_config("fake", "http://h/v1", "k", models, {"qwen27b": {"output": 8192}}, "qwen27b")
        p = cfg["provider"]["capy-flash"]
        self.assertEqual(p["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(list(p["models"]), ["qwen27b"])
        self.assertEqual(p["models"]["qwen27b"]["limit"], {"context": 262144, "output": 8192})
        self.assertTrue(p["options"]["apiKey"].startswith("{file:"))
        self.assertEqual(cfg["model"], "capy-flash/qwen27b")
        path = opencode.write_config(cfg, "k")
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(Path(opencode.key_path()).read_text(), "k")
        self.assertIn("{env:CAPY_FLASH_API_KEY}", opencode.manual_snippet(cfg))
        script = opencode.launch_script("/bin/opencode", "/tmp/my project", "qwen27b")
        text = script.read_text()
        self.assertIn("cd '/tmp/my project'", text)
        self.assertIn("OPENCODE_CONFIG=", text)
        self.assertIn("-m capy-flash/qwen27b", text)

    def test_configure_endpoint(self):
        status, _, data = self.request("POST", "/api/opencode/configure", body={"model": "qwen27b"})
        self.assertEqual(status, 200, data)
        written = json.loads(Path(json.loads(data)["path"]).read_text())
        self.assertIn("qwen27b", written["provider"]["capy-flash"]["models"])
        self.assertNotIn("embed", written["provider"]["capy-flash"]["models"])

    def test_folders(self):
        _, _, data = self.request("GET", "/api/folders?path=" + self.tmp.name)
        listing = json.loads(data)
        self.assertEqual(listing["path"], str(Path(self.tmp.name).resolve()))
        self.assertIn("config", listing["dirs"])
        gone = json.loads(self.request("GET", "/api/folders?path=" + self.tmp.name + "/config/no/such")[2])
        self.assertEqual(gone["path"], str(Path(self.tmp.name).resolve() / "config"), "a deleted folder: its parent")


if __name__ == "__main__":
    unittest.main()
