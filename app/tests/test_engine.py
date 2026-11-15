"""Engine supervision with a fake release whose launchers start tests/fake_engine.py."""

import json
import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capy_app import catalog, machine  # noqa: E402
from capy_app.engine import Engine, EngineError, _alive, clean_profile, launch_plan, profile_flags, shell_command  # noqa: E402
from capy_app.installer import Installer  # noqa: E402
from fake_sources import make_install  # noqa: E402
from test_machine import GPUS_1X5060, FakeSystem  # noqa: E402

KNOBS = (
    "FAKE_ENGINE_LOAD",
    "FAKE_ENGINE_FAIL",
    "FAKE_ENGINE_CRASH_ONCE",
    "FAKE_ENGINE_CORES",
    "FAKE_ENGINE_RECORD",
    "FAKE_ENGINE_SLOW_FIRST",
    "FAKE_ENGINE_COLD_ONCE",
    "FAKE_ENGINE_COLD_FACTOR",
    "NINFER_Q8_A8",
    "NINFER_TP_AR_PAYLOAD",
    "NINFER_ATTN_PV_F16ACC",
)


def _kill(pid):
    try:
        os.kill(pid, 9)
    except OSError:
        pass


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EngineCase(unittest.TestCase):
    models = ("flash-next", "flash-next-64gb", "27b", "27b-16gb", "flash-next-vision")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "capy-flash"
        make_install(self.root, self.models)
        self.saved = {k: os.environ.get(k) for k in KNOBS}
        for k in KNOBS:
            os.environ.pop(k, None)
        self.record = Path(self.tmp.name) / "starts.jsonl"
        os.environ["FAKE_ENGINE_RECORD"] = str(self.record)
        self.system = FakeSystem()
        self.machine = machine.Machine(self.system, self.root, ttl=0)
        self.installer = Installer(self.root, system=self.system)
        self.port = free_port()
        self.allowed = True
        self.profiles = {}
        self.engines = []
        self.engine = self.make()

    def make(self):
        engine = Engine(
            self.installer,
            self.machine,
            lambda: self.port,
            lambda: self.allowed,
            lambda key: self.profiles.get(key, {}),
            Path(self.tmp.name) / "engine.log",
            Path(self.tmp.name) / "run",
        )
        self.engines.append(engine)
        return engine

    def tearDown(self):
        for engine in self.engines:
            engine.close()
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def starts(self):
        return [json.loads(line) for line in self.record.read_text().splitlines()] if self.record.exists() else []

    def wait_state(self, state, engine=None, timeout=15):
        engine = engine or self.engine
        deadline = time.time() + timeout
        while time.time() < deadline and engine.state != state:
            time.sleep(0.02)
        self.assertEqual(engine.state, state, engine.snapshot())


class TestPlan(EngineCase):
    def plan(self, key, profile=None, system=None):
        facts = machine.probe(system or self.system, self.root)
        return launch_plan(catalog.model(key), facts, self.installer, 8080, profile)

    def test_commands_are_install_sh_commands(self):
        current, models = str(self.root / "current"), str(self.root / "models")
        p = self.plan("flash-next")
        self.assertEqual(
            p["argv"],
            [current + "/serve-flash-next.sh", models + "/flash-next/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer"],
        )
        self.assertEqual(p["env"], {"NINFER_HOST": "127.0.0.1", "NINFER_PORT": "8080", "NINFER_DEVICES": "0,1"})
        p = self.plan("flash-next-64gb")  # two GPUs: the smaller RAM caches install.sh adds
        self.assertEqual(p["argv"][2:], ["--host-kv-mib", "1024", "--host-state-slots", "2"])
        p = self.plan("flash-next-64gb", system=FakeSystem(gpus=GPUS_1X5060))  # one GPU: NINFER_DEVICES=0, no flags
        self.assertEqual((p["argv"][2:], p["env"]["NINFER_DEVICES"], p["gpus"]), ([], "0", 1))
        p = self.plan("27b")
        self.assertEqual((os.path.basename(p["argv"][0]), p["env"]["NINFER_DEVICES"]), ("serve-tp2.sh", "0,1"))
        p = self.plan("27b-16gb")
        self.assertEqual((os.path.basename(p["argv"][0]), p["env"].get("NINFER_DEVICE")), ("serve-27b-16gb.sh", "0"))
        self.assertNotIn("NINFER_DEVICES", p["env"])
        self.assertEqual(self.plan("flash-next-vision")["argv"][2:], ["--vision"])

    def test_usable_gpus_only(self):
        mixed = "0, NVIDIA GeForce GTX 1080, 8192, 6.1\n1, NVIDIA GeForce RTX 5080, 16303, 12.0\n2, NVIDIA GeForce RTX 5080, 16303, 12.0\n"
        self.assertEqual(self.plan("27b", system=FakeSystem(gpus=mixed))["env"]["NINFER_DEVICES"], "1,2")

    def test_profile_flags_are_lossless_and_validated(self):
        fn, dense = catalog.model("flash-next"), catalog.model("27b")
        dirty = {
            "expert_cpu_threads": 10,
            "miss_split": "calibrate",
            "stream_min_tokens": 128,
            "kv_ram_active": True,
            "NINFER_Q8_A8": "128",
            "extra": "--api-key x",
            "gpus": 7,
        }
        clean = clean_profile(fn, dirty)
        self.assertEqual(
            clean,
            {"expert_cpu_threads": 10, "miss_split": "calibrate", "stream_min_tokens": 128, "kv_ram_active": True},
        )
        self.assertEqual(
            profile_flags(clean),
            [
                "--expert-cpu-threads",
                "10",
                "--miss-split",
                "calibrate",
                "--stream-min-tokens",
                "128",
                "--kv-ram-active",
            ],
        )
        self.assertEqual(
            clean_profile(
                fn,
                {"expert_cpu_threads": 0, "miss_split": "; rm -rf", "stream_min_tokens": "9", "kv_ram_active": "yes"},
            ),
            {},
        )
        self.assertEqual(clean_profile(dense, dirty), {}, "the 27B runs on the GPUs: none of these apply")
        self.assertEqual(
            clean_profile(catalog.model("flash-next-64gb"), {"miss_split": "auto", "gpus": 1}), {"gpus": 1}
        )
        p = self.plan("flash-next-64gb", {"gpus": 1, "expert_cpu_threads": 11})
        self.assertEqual((p["env"]["NINFER_DEVICES"], p["argv"][2:]), ("0", ["--expert-cpu-threads", "11"]))
        self.assertIn("NINFER_DEVICES=0 ", shell_command(p))
        self.assertNotIn("NINFER_HOST", shell_command(p))


class TestRun(EngineCase):
    def test_start_health_stop(self):
        os.environ["FAKE_ENGINE_LOAD"] = "0.4"
        self.engine.start("flash-next")
        self.assertEqual(self.engine.state, "starting")
        pid = self.engine.pid
        self.wait_state("running")
        snap = self.engine.snapshot()
        self.assertGreaterEqual(snap["load_seconds"], 0.3)
        self.assertEqual((snap["model"], snap["gpus"], snap["port"]), ("flash-next", 2, self.port))
        start = self.starts()[0]
        self.assertIn("--port", start["argv"])
        self.assertEqual(start["env"]["NINFER_HOST"], "127.0.0.1")
        log = "\n".join(self.engine.log_tail())
        self.assertIn("# capy-flash app", log)
        self.assertIn("routed experts: 6 CPU threads", log)
        self.engine.stop()
        self.assertEqual(self.engine.state, "stopped")
        self.assertFalse(_alive(pid))
        self.assertFalse((Path(self.tmp.name) / "run" / "engine.json").exists())

    def test_tuned_profile_reaches_the_command_line(self):
        self.profiles["flash-next"] = {"expert_cpu_threads": 10, "miss_split": "calibrate"}
        self.engine.start("flash-next")
        self.wait_state("running")
        argv = self.starts()[0]["argv"]
        self.assertEqual(argv[argv.index("--expert-cpu-threads") + 1], "10")
        self.assertEqual(argv[argv.index("--miss-split") + 1], "calibrate")
        self.engine.start("flash-next", {})  # the tuner's baseline: no profile
        self.wait_state("running")
        self.assertNotIn("--expert-cpu-threads", self.starts()[1]["argv"])
        self.assertEqual(len(self.starts()), 2)

    def test_lossy_switches_never_reach_the_engine(self):
        os.environ.update({"NINFER_Q8_A8": "128", "NINFER_TP_AR_PAYLOAD": "int8", "NINFER_ATTN_PV_F16ACC": "1"})
        self.engine.start("flash-next")
        self.wait_state("running")
        env = self.starts()[0]["env"]
        for name in catalog.LOSSY_ENV:
            self.assertNotIn(name, env)

    def test_startup_failure_is_reported_with_the_log(self):
        os.environ["FAKE_ENGINE_FAIL"] = "1"
        self.engine.start("27b")
        self.wait_state("failed")
        error = self.engine.snapshot()["error"]
        self.assertEqual(error["exit_code"], 1)
        self.assertIn("CUDA out of memory", "\n".join(error["log"]))
        self.assertEqual(len(self.starts()), 1, "a startup failure is not retried")

    def test_exit_2_restarts(self):
        os.environ["FAKE_ENGINE_CRASH_ONCE"] = str(Path(self.tmp.name) / "crashed")
        self.engine.start("flash-next")
        self.wait_state("running")
        first = self.engine.pid
        try:
            import urllib.request

            urllib.request.urlopen(
                urllib.request.Request(self.engine.base_url() + "/chat/completions", data=b"{}"), timeout=5
            )
        except Exception:
            pass  # the fake engine dies with exit code 2 on this request
        deadline = time.time() + 15
        snap = self.engine.snapshot()
        while time.time() < deadline and not (snap["state"] == "running" and snap["restarts"] == 1):
            time.sleep(0.02)
            snap = self.engine.snapshot()
        self.assertEqual((snap["state"], snap["restarts"]), ("running", 1))
        self.assertNotEqual(snap["pid"], first)
        self.assertEqual(len(self.starts()), 2)

    def test_restart_waits_for_the_stopped_engine_to_leave_the_gpus(self):
        self.engine.start("flash-next")
        self.wait_state("running")
        first = self.engine.pid
        # nvidia-smi still lists the engine that was just stopped, for the first two looks
        looks = []
        real = self.system.run

        def run(argv, timeout=10.0):
            if "--query-compute-apps=pid,process_name,used_memory" in argv:
                looks.append(1)
                return f"{first}, ninfer-serve, 14500\n" if len(looks) <= 2 else ""
            return real(argv, timeout)

        self.system.run = run
        self.engine.start("flash-next", {"expert_cpu_threads": 8})
        self.wait_state("running")
        self.assertGreaterEqual(len(looks), 3)
        self.assertNotEqual(self.engine.pid, first)

    def test_stop_during_load(self):
        os.environ["FAKE_ENGINE_LOAD"] = "30"
        self.engine.start("flash-next")
        pid = self.engine.pid
        time.sleep(0.3)
        self.engine.stop()
        self.assertEqual(self.engine.state, "stopped")
        self.assertFalse(_alive(pid))

    def test_engine_left_by_a_dead_app_is_found_and_stopped(self):
        self.engine.start("flash-next")
        self.wait_state("running")
        pid = self.engine.pid
        self.addCleanup(_kill, pid)  # whatever this test finds, no fake engine is left behind
        self.engines.remove(self.engine)  # the app died without stopping its engine
        second = self.make()
        self.wait_state("running", second)
        self.assertEqual((second.pid, second.model, second.adopted), (pid, "flash-next", True))
        second.stop()
        self.assertFalse(_alive(pid))

    def test_stale_record_is_ignored(self):
        run = Path(self.tmp.name) / "run"
        run.mkdir()
        (run / "engine.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "model": "flash-next",
                    "port": self.port,
                    "artifact": "/x/models/flash-next/model.ninfer",
                }
            )
        )
        engine = self.make()  # that pid is this test, not an engine
        self.assertEqual((engine.state, engine.pid), ("stopped", None))


class TestGuards(EngineCase):
    def refused(self, key="flash-next"):
        with self.assertRaises(EngineError) as c:
            self.engine.start(key)
        self.assertEqual(self.starts(), [], "no process was started")
        self.assertEqual(self.engine.state, "stopped")
        return c.exception

    def test_only_when_the_user_chose_to_run_here(self):
        self.allowed = False
        self.assertEqual(self.refused().code, "not_local")

    def test_not_on_a_mac(self):
        self.system.os_name, self.system.arch = "Darwin", "arm64"
        self.assertEqual(self.refused().code, "unsupported")

    def test_not_without_suitable_gpus(self):
        self.system.gpus = "0, NVIDIA GeForce RTX 4090, 24564, 8.9\n"
        self.assertEqual(self.refused("27b-16gb").code, "unsupported")

    def test_not_next_to_another_server_on_the_gpus(self):
        self.system.processes = "4242, /usr/local/bin/llama-server, 14210\n4243, /usr/local/bin/llama-server, 13990\n"
        e = self.refused()
        self.assertEqual(e.code, "gpu_busy")
        self.assertIn("llama-server", e.message)

    def test_not_when_the_port_already_answers(self):
        with socket.socket() as other:
            other.bind(("127.0.0.1", self.port))
            other.listen(1)
            self.assertEqual(self.refused().code, "port_busy")

    def test_not_without_enough_free_ram(self):
        self.system.files["/proc/meminfo"] = "MemTotal: 98765432 kB\nMemAvailable: 41943040 kB\n"
        self.assertEqual(self.refused().code, "ram")
        self.assertEqual(self.engine.snapshot()["error"]["code"], "ram")

    def test_not_before_the_model_is_there(self):
        import shutil

        shutil.rmtree(self.root / "models" / "27b")
        self.assertEqual(self.refused("27b").code, "no_model")
        self.assertEqual(self.refused("nope").code, "no_model")

    def test_not_files_nobody_verified(self):
        # On disk (install.sh, a copy, an older download) but neither verified by the app nor accepted by the user.
        self.installer._set_model_entry("27b", as_is=None)
        e = self.refused("27b")
        self.assertEqual((e.code, e.detail["found"]), ("unverified", "foreign"))
        self.assertIn(str(self.root / "models" / "27b"), e.message)

    def test_not_a_release_that_fails_verification(self):
        (self.root / "current" / "bin" / "ninfer-serve").write_text("#!/bin/sh\necho tampered\n")
        e = self.refused()
        self.assertEqual(e.code, "no_release")
        self.assertIn("SHA256SUMS", e.message)


if __name__ == "__main__":
    unittest.main()
