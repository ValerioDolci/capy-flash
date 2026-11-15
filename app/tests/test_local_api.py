"""The API for capy-flash on this computer (setup, install, engine, tuning) and its security rules."""

import http.client
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capy_app import catalog, paths  # noqa: E402
from capy_app.downloads import Fetcher  # noqa: E402
from capy_app.local import Local  # noqa: E402
from capy_app.server import LOCAL_POSTS, bind, make_app  # noqa: E402
from capy_app.store import Conversations, Settings  # noqa: E402
from fake_backend import FakeBackend  # noqa: E402
from fake_sources import Sources, make_install, model_files  # noqa: E402
from test_engine import KNOBS, free_port  # noqa: E402
from test_machine import FakeSystem  # noqa: E402

WEB = Path(__file__).resolve().parent.parent / "web"
M27 = catalog.model("27b-16gb")


class LocalCase(unittest.TestCase):
    system_args = {}
    candidates = ()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self._saved = (paths.config_dir, paths.runtime_dir, paths.data_dir)
        paths.config_dir = lambda: base / "config"
        paths.runtime_dir = lambda: base / "run"
        paths.data_dir = lambda: base / "data"
        self.env = {k: os.environ.get(k) for k in KNOBS}
        for k in KNOBS:
            os.environ.pop(k, None)
        self.starts_file = base / "starts.jsonl"
        os.environ["FAKE_ENGINE_RECORD"] = str(self.starts_file)
        self.sources = Sources()
        self.sources.add_model(M27["hf_repo"], model_files(M27["artifact"], size=40_000))
        self.fake = FakeBackend()
        self.root = base / "capy-flash"
        self.system = FakeSystem(**self.system_args)
        self.settings = Settings(base / "config")
        self.settings.update({"backend": {"base_url": self.fake.url, "name": "fake"}, "engine": {"port": free_port()}})
        urls = {"github_api": self.sources.url, "github": self.sources.url, "hf": self.sources.url}
        self.local = Local(
            self.settings,
            system=self.system,
            root=self.root,
            fetcher=Fetcher(allow_http=True, timeout=5, retries=0, backoff=0.01),
            sources=urls,
            state_dir=base / "config",
            log_dir=base / "data",
            run_dir=base / "run",
            candidates=self.candidates or (("fake", "http://127.0.0.1:9/v1"),),
        )
        self.token = "test-token-123"
        self.app = make_app(WEB, self.settings, Conversations(base / "data"), self.token, self.local)
        self.server, self.port = bind(self.app, "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.origin = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.local.close()
        self.local.installer.wait(10)
        self.server.shutdown()
        self.server.server_close()
        self.fake.close()
        self.sources.close()
        paths.config_dir, paths.runtime_dir, paths.data_dir = self._saved
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None, cookie=True, origin=True):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = {"Host": f"127.0.0.1:{self.port}"}
        if cookie:
            h["Cookie"] = f"capy_t={self.token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
            if origin:
                h["Origin"] = self.origin
        h.update(headers or {})
        conn.request(method, path, body=data, headers=h)
        r = conn.getresponse()
        raw = r.read()
        conn.close()
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, raw

    def post(self, path, body=None):
        return self.request("POST", path, body or {})

    def engine_state(self):
        return self.request("GET", "/api/engine")[1]

    def wait_for(self, what, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.engine_state()
            if what(state):
                return state
            time.sleep(0.05)
        self.fail("timed out; last state: " + json.dumps(state)[:1500])

    def starts(self):
        return self.starts_file.read_text().splitlines() if self.starts_file.exists() else []


class TestSetup(LocalCase):
    def test_linux_with_gpus_and_nothing_installed(self):
        status, setup = self.request("GET", "/api/setup")
        self.assertEqual(status, 200)
        self.assertEqual(
            (setup["supported"], setup["recommended"], setup["proposal"], setup["setup_done"]),
            (True, "flash-next", "local", False),
        )
        self.assertEqual(setup["machine"]["usable_gpus"], 2)
        state = self.engine_state()
        self.assertEqual((state["mode"], state["supported"], state["release"]), ("endpoint", True, None))
        fits = {m["key"]: m["fit"]["ok"] for m in state["models"]}
        self.assertTrue(all(fits.values()))
        self.assertEqual([m["key"] for m in state["models"] if m["recommended"]], ["flash-next"])
        self.assertEqual(self.post("/api/setup/done")[0], 200)
        self.assertTrue(self.request("GET", "/api/setup")[1]["setup_done"])

    def test_state_says_what_the_platform_can_do(self):
        _, state = self.request("GET", "/api/state")
        self.assertEqual(state["platform"], {"os": "Linux", "arch": "x86_64", "can_run_engine": True})
        self.assertEqual(state["settings"]["backend"]["kind"], "endpoint")


class TestSetupWithEndpoint(LocalCase):
    def setUp(self):
        super().setUp()
        self.local.candidates = (("vramble", self.fake.url), ("nothing", "http://127.0.0.1:9/v1"))

    def test_a_server_that_answers_stays_the_proposal(self):
        self.system.processes = "4242, /usr/local/bin/llama-server, 14210\n"
        _, setup = self.request("GET", "/api/setup")
        self.assertEqual(setup["proposal"], "endpoint")
        self.assertEqual(
            setup["endpoints"],
            [{"kind": "endpoint", "name": "vramble", "base_url": self.fake.url, "models": 2}],
            "chat models: the embedding model is not counted",
        )
        self.assertEqual([p["pid"] for p in setup["busy_gpus"]], [4242])


class TestMac(LocalCase):
    system_args = {"os_name": "Darwin", "arch": "arm64"}

    def test_a_mac_is_a_client(self):
        _, setup = self.request("GET", "/api/setup")
        self.assertEqual((setup["supported"], setup["recommended"], setup["proposal"]), (False, None, "connect"))
        self.assertEqual([l["code"] for l in setup["why_not"]], ["platform"])
        _, state = self.request("GET", "/api/state")
        self.assertFalse(state["platform"]["can_run_engine"])
        for path, body in (("/api/engine/use", {"model": "27b-16gb"}), ("/api/engine/install", {"model": "27b-16gb"})):
            status, answer = self.post(path, body)
            self.assertEqual((status, answer["error"]["code"]), (409, "unsupported"), path)
        self.assertEqual(self.post("/api/engine/start")[1]["error"]["code"], "no_model")
        self.assertEqual(self.settings.get()["backend"]["kind"], "endpoint")
        self.assertIsNone(self.local.installer.job)
        self.assertEqual(self.sources.requests, [], "nothing was downloaded")


class TestSecurity(LocalCase):
    def test_every_action_is_post_with_token_and_same_origin(self):
        make_install(self.root, ["27b-16gb"])
        for path in LOCAL_POSTS:
            body = {"model": "27b-16gb"}
            self.assertEqual(self.request("POST", path, body, cookie=False)[0], 401, path)
            self.assertEqual(self.request("POST", path, body, headers={"Origin": "https://evil.example"})[0], 403, path)
            self.assertEqual(self.request("POST", path, body, headers={"Sec-Fetch-Site": "cross-site"})[0], 403, path)
            self.assertEqual(
                self.request(
                    "POST", path, body, headers={"Sec-Fetch-Site": "same-site", "Origin": "http://127.0.0.1:1"}
                )[0],
                403,
                path,
            )
            self.assertEqual(self.request("GET", path)[0], 405, path)
            self.assertEqual(self.request("PUT", path, body)[0], 405, path)
        self.assertEqual(self.starts(), [], "no engine was started")
        self.assertIsNone(self.local.installer.job)
        self.assertEqual(self.settings.get()["backend"]["kind"], "endpoint")
        for path in ("/api/setup", "/api/engine", "/api/engine/log", "/api/tune/plan?model=27b"):
            self.assertEqual(self.request("GET", path, cookie=False)[0], 401, path)
            self.assertEqual(self.request("GET", path)[0], 200, path)
            self.assertEqual(self.request("POST", path.split("?")[0], {})[0], 405, path)
        self.assertEqual(self.request("GET", "/api/engine/nope")[0], 404)

    def test_settings_cannot_switch_to_local_or_touch_the_engine(self):
        for change in (
            {"backend": {"kind": "local"}},
            {"engine": {"dir": "/tmp/evil"}},
            {"engine": {"model": "27b"}},
            {"setup_done": True},
            {"backend": {"base_url": self.fake.url, "kind": "local"}},
            {"recent_folders": ["/"]},
        ):
            self.assertEqual(self.post("/api/settings", change)[0], 400, change)
        value = self.settings.get()
        self.assertEqual(
            (value["backend"]["kind"], value["engine"]["dir"], value["engine"]["model"]), ("endpoint", "", "")
        )

    def test_engine_does_not_start_unless_the_user_chose_local(self):
        make_install(self.root, ["27b-16gb"])
        self.settings.update({"engine": {"model": "27b-16gb"}})  # even with a model set and installed
        status, answer = self.post("/api/engine/start")
        self.assertEqual((status, answer["error"]["code"]), (409, "not_local"))
        status, answer = self.post("/api/tune/start")
        self.assertEqual((status, answer["error"]["code"]), (409, "not_local"))
        self.local.autostart()
        self.assertEqual(self.starts(), [])
        self.assertEqual(self.engine_state()["process"]["state"], "stopped")

    def test_gpus_held_by_another_server_refuse_the_start(self):
        make_install(self.root, ["27b-16gb"])
        self.assertEqual(self.post("/api/engine/use", {"model": "27b-16gb"})[0], 200)
        self.wait_for(lambda s: s["process"]["state"] == "running")
        self.post("/api/engine/stop")
        self.system.processes = "4242, /usr/local/bin/llama-server, 14210\n"
        self.local.machine.facts(fresh=True)
        from capy_app import engine

        settle, engine.SETTLE_SECONDS = engine.SETTLE_SECONDS, 0.3  # (it looks again for a while after a stop)
        try:
            status, answer = self.post("/api/engine/start")
        finally:
            engine.SETTLE_SECONDS = settle
        self.assertEqual((status, answer["error"]["code"]), (409, "gpu_busy"))
        self.assertEqual(len(self.starts()), 1)
        self.assertIn("llama-server", json.dumps(self.engine_state()["process"]["error"]))

    def test_only_catalog_models_and_fixed_sources(self):
        for body in ({"model": "nope"}, {"model": "../../etc"}, {"model": ["27b"]}, {}):
            self.assertEqual(self.post("/api/engine/install", body)[0], 400, body)
            self.assertEqual(self.post("/api/engine/use", body)[0], 400, body)
            self.assertEqual(self.post("/api/engine/remove", body)[0], 400, body)
        self.assertEqual(self.sources.requests, [])
        # a URL or a folder in the request changes nothing: the sources and the place are the app's
        status, _ = self.post(
            "/api/engine/install",
            {"model": "27b-16gb", "url": "http://127.0.0.1:9/evil", "dir": "/tmp/evil", "repo": "evil/repo"},
        )
        self.assertEqual(status, 200)
        self.wait_for(lambda s: s["job"] and s["job"]["state"] == "done")
        self.assertTrue(
            all(
                p.startswith(("/repos/ValerioDolci/", "/ValerioDolci/", "/api/models/Feyd89/", "/Feyd89/"))
                for p, _ in self.sources.requests
            )
        )
        self.assertTrue((self.root / "models" / "27b-16gb" / M27["artifact"]).is_file())
        self.assertFalse(Path("/tmp/evil").exists())

    def test_other_version_on_disk_is_the_users_decision(self):
        # A folder of another revision is in models/27b-16gb: the app reports it and waits for a choice.
        make_install(self.root, ["27b-16gb"], accept=False)
        folder = self.root / "models" / "27b-16gb"
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
        self.assertEqual(self.post("/api/engine/use", {"model": "27b-16gb"})[0], 200)
        model = lambda: [m for m in self.engine_state()["models"] if m["key"] == "27b-16gb"][0]
        self.assertEqual((model()["state"], model()["installed"]), ("foreign", False))
        self.assertEqual(self.post("/api/engine/start")[1]["error"]["code"], "unverified")
        self.post("/api/engine/install", {"model": "27b-16gb"})
        state = self.wait_for(lambda s: s["job"] and s["job"]["state"] == "failed")
        self.assertEqual(state["job"]["error"]["code"], "different_version")
        m = model()
        self.assertEqual((m["state"], m["folder"], m["deletable"]), ("different", str(folder), False))
        self.assertIn(M27["artifact"], m["mismatch"])
        self.assertEqual(self.post("/api/engine/remove", {"model": "27b-16gb"})[1]["error"]["code"], "not_ours")
        self.assertEqual(self.starts(), [])
        # choice 1: the current version, in a new folder next to it
        self.assertEqual(self.post("/api/engine/install", {"model": "27b-16gb", "new_folder": True})[0], 200)
        state = self.wait_for(lambda s: s["process"]["state"] == "running")
        self.assertRegex(model()["folder"], r"models/27b-16gb-[0-9a-f]{8}$")
        self.assertRegex(state["command"], r"models/27b-16gb-[0-9a-f]{8}/")
        self.assertEqual({p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}, before)
        # delete the app's download: the other version is there again, and still nobody's to run...
        self.post("/api/engine/stop")
        self.assertEqual(self.post("/api/engine/remove", {"model": "27b-16gb"})[0], 200)
        self.assertEqual(self.post("/api/engine/start")[1]["error"]["code"], "unverified")
        # choice 2: ...unless the user says "use it as it is"
        self.assertEqual(self.post("/api/engine/use", {"model": "27b-16gb", "as_is": True})[0], 200)
        state = self.wait_for(lambda s: s["process"]["state"] == "running")
        self.assertEqual((model()["state"], model()["verified"]), ("as_is", False))
        self.assertIn(str(folder) + "/", state["command"])
        self.assertEqual({p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}, before)

    def test_engine_settings_are_validated(self):
        for bad in ({"port": 80}, {"port": "8080"}, {"port": 70000}, {"port": True}):
            self.assertEqual(self.post("/api/engine/settings", bad)[0], 400, bad)
        self.assertEqual(self.post("/api/engine/settings", {"port": 9123, "autostart": False, "dir": "/x"})[0], 200)
        engine = self.settings.get()["engine"]
        self.assertEqual((engine["port"], engine["autostart"], engine["dir"]), (9123, False, ""))


class TestFlow(LocalCase):
    def test_install_run_chat_tune_revert_and_back_to_a_server(self):
        # choose to run here: nothing is installed yet, so nothing starts
        status, state = self.post("/api/engine/use", {"model": "27b-16gb"})
        self.assertEqual((status, state["mode"], state["model"]), (200, "local", "27b-16gb"))
        self.assertEqual(state["process"]["state"], "stopped")
        self.assertTrue(self.settings.get()["setup_done"])
        # install: release, then the model; the engine starts by itself when both are verified
        status, state = self.post("/api/engine/install", {"model": "27b-16gb"})
        self.assertEqual((status, state["job"]["state"]), (200, "running"))
        self.assertEqual(self.post("/api/engine/install", {"model": "27b-16gb"})[1]["error"]["code"], "busy")
        state = self.wait_for(lambda s: s["process"]["state"] == "running")
        self.assertEqual(state["job"]["state"], "done")
        self.assertEqual((state["release"]["version"], state["release"]["verified"]), ("v0.12.0", True))
        model = [m for m in state["models"] if m["key"] == "27b-16gb"][0]
        self.assertEqual((model["installed"], model["verified"]), (True, True))
        self.assertIn("serve-27b-16gb.sh", state["command"])
        # the chat goes to the engine on this computer
        status, models = self.request("GET", "/api/models")
        self.assertEqual((status, [m["id"] for m in models["models"]]), (200, ["qwen3.8-27b"]))
        self.assertIs(models["models"][0]["vision"], True, "the 27B reads images: the catalog says so")
        status, reply = self.request(
            "POST",
            "/v1/chat/completions",
            {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        )
        self.assertEqual(status, 200)
        self.assertIn("timings", reply)
        status, log = self.request("GET", "/api/engine/log?lines=50")
        self.assertIn("listening on", "\n".join(log["lines"]))
        # tuning: for the 27B, a measurement
        status, plan = self.request("GET", "/api/tune/plan")
        self.assertEqual((status, plan["tunable"]), (200, False))
        self.assertEqual(self.post("/api/tune/start")[0], 200)
        state = self.wait_for(lambda s: s["tuning"]["state"] == "done")
        self.assertGreater(state["profile"]["before"]["decode_tps"], 50)
        self.assertEqual((state["profile"]["active"], state["profile"]["stale"]), (True, False))
        status, state = self.post("/api/tune/revert")
        self.assertEqual((status, state["profile"]["active"]), (200, False))
        self.assertEqual(self.post("/api/tune/apply")[1]["profile"]["active"], True)
        # the model cannot be deleted while it runs
        self.assertEqual(self.post("/api/engine/remove", {"model": "27b-16gb"})[1]["error"]["code"], "in_use")
        # back to a server: the engine this app started stops
        self.wait_for(lambda s: s["process"]["state"] == "running")
        pid = self.engine_state()["process"]["pid"]
        status, _ = self.post("/api/settings", {"backend": {"name": "fake", "base_url": self.fake.url}})
        self.assertEqual(status, 200)
        state = self.engine_state()
        self.assertEqual((state["mode"], state["process"]["state"]), ("endpoint", "stopped"))
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertIn("qwen27b", [m["id"] for m in self.request("GET", "/api/models")[1]["models"]])
        self.assertEqual(self.post("/api/engine/remove", {"model": "27b-16gb"})[0], 200)
        self.assertFalse((self.root / "models" / "27b-16gb").exists())

    def test_cancel_a_download_and_continue_it(self):
        self.post("/api/engine/use", {"model": "27b-16gb"})
        self.sources.rate = 30_000
        _, state = self.post("/api/engine/install", {"model": "27b-16gb"})
        state = self.wait_for(lambda s: s["job"]["phase"] == "model" and s["job"]["bytes_done"] > 0)
        self.assertEqual(self.post("/api/engine/cancel", {"job": state["job"]["id"]})[0], 200)
        state = self.wait_for(lambda s: s["job"]["state"] == "cancelled")
        self.assertEqual(state["process"]["state"], "stopped")
        self.assertEqual(self.starts(), [])
        self.sources.rate = 0
        self.post("/api/engine/install", {"model": "27b-16gb"})
        self.wait_for(lambda s: s["process"]["state"] == "running")

    def test_failed_download_reports_and_does_not_start(self):
        self.post("/api/engine/use", {"model": "27b-16gb"})
        self.sources.corrupt.add(M27["artifact"])
        self.post("/api/engine/install", {"model": "27b-16gb"})
        state = self.wait_for(lambda s: s["job"] and s["job"]["state"] == "failed")
        self.assertEqual(state["job"]["error"]["code"], "checksum")
        self.assertEqual(state["process"]["state"], "stopped")
        self.assertEqual(self.starts(), [], "a model that failed verification is never run")
        self.assertEqual(self.post("/api/engine/start")[1]["error"]["code"], "no_model")

    def test_quit_stops_the_engine(self):
        make_install(self.root, ["27b-16gb"])
        self.post("/api/engine/use", {"model": "27b-16gb"})
        state = self.wait_for(lambda s: s["process"]["state"] == "running")
        pid = state["process"]["pid"]
        self.assertEqual(self.post("/api/quit")[0], 200)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail("the engine survived Quit")

    def test_autostart_can_be_turned_off(self):
        make_install(self.root, ["27b-16gb"])
        self.post("/api/engine/settings", {"autostart": False})
        self.post("/api/engine/use", {"model": "27b-16gb"})
        time.sleep(0.3)
        self.assertEqual(self.engine_state()["process"]["state"], "stopped")
        self.assertEqual(self.post("/api/engine/start")[0], 200)
        self.wait_for(lambda s: s["process"]["state"] == "running")

    def test_update_check(self):
        make_install(self.root, [])
        status, answer = self.post("/api/engine/check-update")
        self.assertEqual((status, answer), (200, {"installed": "v0.12.0", "latest": "v0.12.0", "newer": False}))
        self.sources.version = "v0.12.1"
        self.assertEqual(self.post("/api/engine/check-update")[1]["newer"], True)


if __name__ == "__main__":
    unittest.main()
