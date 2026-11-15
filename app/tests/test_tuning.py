"""Tuning orchestration against the fake engine, whose reported timings depend on the flags it was started with
(tests/fake_engine.py: 10 expert threads are best, calibrate > auto > off, 128-token streaming helps short
prompts, two GPUs beat one, KV in RAM helps only long contexts; it can also make the first request of a process
slow, or the whole first process, as a cold machine does)."""

import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capy_app import catalog, tuning  # noqa: E402
from capy_app.engine import EngineError  # noqa: E402
from capy_app.tuning import (
    ProfileStore,
    Tuner,
    filler,
    gain,
    reference_and_noise,
    signature,
    steps_for,  # noqa: E402
    worth_it,
)
from test_engine import EngineCase  # noqa: E402
from test_machine import GPUS_1X5060, FakeSystem  # noqa: E402


class TuneCase(EngineCase):
    def setUp(self):
        super().setUp()
        self.store = ProfileStore(Path(self.tmp.name) / "tuning.json")
        self.engine._profile_for = lambda key: self.store.flags(key, self.machine.facts())
        self.engine.on_ready = self.store.note_load
        self.tuner = Tuner(self.engine, self.machine, self.store, version=lambda: "v0.12.0")

    def run_tuning(self, key, long_context=False):
        self.tuner.start(key, long_context)
        self.tuner.wait(120)
        return self.tuner.snapshot()

    def flags_of(self, start):
        argv = start["argv"]
        out = {}
        for flag in ("--expert-cpu-threads", "--miss-split", "--stream-min-tokens"):
            if flag in argv:
                out[flag] = argv[argv.index(flag) + 1]
        if "--kv-ram-active" in argv:
            out["--kv-ram-active"] = True
        return out


class TestBench(unittest.TestCase):
    def test_prompts_are_fixed(self):
        self.assertEqual(filler(50), filler(50))
        self.assertEqual(len(filler(tuning.PREFILL_WORDS).split()), tuning.PREFILL_WORDS)
        self.assertNotEqual(filler(50, seed=1), filler(50, seed=2))

    def test_decision_rule(self):
        base = {"decode_tps": 100.0, "prefill_tps": 4000.0, "short_ms": 200.0}
        self.assertAlmostEqual(gain({"short_ms": 180.0}, base, "short_ms"), 0.10)  # lower is better
        self.assertTrue(worth_it(dict(base, decode_tps=104.0), base, "decode_tps"))
        self.assertFalse(worth_it(dict(base, decode_tps=102.0), base, "decode_tps"), "under 3 %: noise")
        self.assertFalse(
            worth_it(dict(base, decode_tps=110.0, prefill_tps=3500.0), base, "decode_tps"),
            "faster decode must not cost 12 % of prefill",
        )
        self.assertTrue(worth_it(dict(base, decode_tps=110.0, prefill_tps=3950.0), base, "decode_tps"))

    def test_a_gain_must_also_exceed_the_noise(self):
        # The defaults measured twice (A ... A): the better value is the reference, the distance is the noise.
        first = {"decode_tps": 92.0, "prefill_tps": 3008.0, "short_ms": 330.0}  # a cold first run
        second = {"decode_tps": 100.0, "prefill_tps": 3440.0, "short_ms": 300.0}
        reference, noise = reference_and_noise(first, second)
        self.assertEqual(reference, second)
        self.assertAlmostEqual(noise["decode_tps"], 0.08)
        self.assertAlmostEqual(noise["prefill_tps"], 0.1256, places=3)
        self.assertAlmostEqual(noise["short_ms"], 0.10)
        candidate = dict(second, decode_tps=106.0)  # +6 %: above the 3 % rule...
        self.assertTrue(worth_it(candidate, reference, "decode_tps"))
        self.assertFalse(worth_it(candidate, reference, "decode_tps", noise), "...but inside this run's 8 % noise")
        self.assertTrue(worth_it(dict(second, decode_tps=110.0), reference, "decode_tps", noise))
        self.assertFalse(
            worth_it(dict(second, decode_tps=102.0), reference, "decode_tps", {"decode_tps": 0.0}), "3 % still"
        )
        # another number may move within the noise without disqualifying a real gain
        self.assertTrue(worth_it(dict(second, decode_tps=110.0, prefill_tps=3200.0), reference, "decode_tps", noise))
        self.assertFalse(worth_it(dict(second, decode_tps=110.0, prefill_tps=2900.0), reference, "decode_tps", noise))

    def test_steps_per_model(self):
        two = {"usable_gpus": 2}
        ids = lambda key, facts=two, long=False: [s["id"] for s in steps_for(catalog.model(key), facts, long)]
        self.assertEqual(ids("flash-next"), ["threads", "miss_split", "stream"])
        self.assertEqual(ids("flash-next", long=True), ["threads", "miss_split", "stream", "kv_ram"])
        self.assertEqual(ids("flash-next-64gb"), ["threads", "stream", "gpus"], "no miss split with GGUF experts")
        self.assertEqual(ids("flash-next-64gb", {"usable_gpus": 1}), ["threads", "stream"])
        self.assertEqual(ids("27b"), [])
        self.assertEqual(ids("27b-16gb", long=True), [])


class TestTuner(TuneCase):
    def test_flash_next_finds_the_best_lossless_settings(self):
        plan = self.tuner.plan("flash-next")
        self.assertTrue(plan["tunable"])
        self.assertGreater(plan["seconds"], 60 * 5, "an honest estimate: several engine starts")
        snap = self.run_tuning("flash-next")
        self.assertEqual(snap["state"], "done", snap["error"])
        record = self.store.record("flash-next")
        self.assertEqual(
            record["flags"], {"expert_cpu_threads": 10, "miss_split": "calibrate", "stream_min_tokens": 128}
        )
        self.assertTrue(record["active"])
        self.assertEqual(record["engine_version"], "v0.12.0")
        before, after = record["before"], record["after"]
        self.assertAlmostEqual(before["decode_tps"], 138.0, delta=0.2)  # 6 threads (auto), miss split off
        self.assertAlmostEqual(after["decode_tps"], 160.5, delta=0.2)  # 10 threads, calibrate
        self.assertLess(after["short_ms"], before["short_ms"] * 0.95)
        self.assertGreaterEqual(after["prefill_tps"], before["prefill_tps"] * 0.97)
        labels = [(t["step"], t["label"]) for t in record["trials"]]
        self.assertEqual(
            labels,
            [
                ("baseline", ""),
                ("threads", "8"),
                ("threads", "10"),
                ("threads", "12"),
                ("miss_split", "auto"),
                ("miss_split", "calibrate"),
                ("stream", "128"),
                ("stream", "512"),
                ("baseline_again", ""),
                ("final", ""),
            ],
            "the defaults, each setting on its own, the defaults again, the kept settings together",
        )
        self.assertEqual(
            [t["label"] for t in record["trials"] if t["adopted"] and t["step"] != "final"], ["10", "calibrate", "128"]
        )
        self.assertTrue(record["trials"][-1]["adopted"])
        self.assertEqual(record["trials"][-1]["metrics"], after, "the numbers shown are those of the final settings")
        self.assertEqual([b["decode_tps"] for b in record["baselines"]], [138.0, 138.0])
        self.assertEqual(record["noise"], {"decode_tps": 0.0, "prefill_tps": 0.0, "short_ms": 0.0})
        for start in self.starts()[1:-1]:  # every candidate is the defaults plus one setting
            self.assertLessEqual(len(self.flags_of(start)), 1, start["argv"])
        # the engine now runs with the profile, and a later start uses it too
        self.assertEqual(self.engine.state, "running")
        self.assertEqual(
            self.flags_of(self.starts()[-1]),
            {"--expert-cpu-threads": "10", "--miss-split": "calibrate", "--stream-min-tokens": "128"},
        )
        self.assertEqual(self.store.flags("flash-next", self.machine.facts()), record["flags"])
        for start in self.starts():  # never a switch that changes the output
            for name in catalog.LOSSY_ENV:
                self.assertNotIn(name, start["env"])
            self.assertFalse([a for a in start["argv"] if "NINFER" in a])

    def test_revert_and_apply_again(self):
        self.run_tuning("flash-next")
        self.assertTrue(self.store.set_active("flash-next", False))
        self.assertEqual(self.store.flags("flash-next", self.machine.facts()), {})
        self.engine.start("flash-next")
        self.wait_state("running")
        self.assertEqual(self.flags_of(self.starts()[-1]), {}, "reverted: the launcher's defaults")
        self.assertIsNotNone(self.store.record("flash-next"), "the measurements are kept")
        self.store.set_active("flash-next", True)
        self.engine.start("flash-next")
        self.wait_state("running")
        self.assertEqual(self.flags_of(self.starts()[-1])["--expert-cpu-threads"], "10")
        self.assertFalse(self.store.set_active("27b", True), "nothing measured for that model")

    def test_baseline_reuses_a_running_engine(self):
        self.engine.start("flash-next")
        self.wait_state("running")
        self.run_tuning("flash-next")
        self.assertEqual(self.flags_of(self.starts()[1]), {"--expert-cpu-threads": "8"}, "no restart for the baseline")

    def test_dense_model_is_measured_not_tuned(self):
        plan = self.tuner.plan("27b")
        self.assertFalse(plan["tunable"])
        snap = self.run_tuning("27b")
        self.assertEqual(snap["state"], "done", snap["error"])
        record = self.store.record("27b")
        self.assertEqual(record["flags"], {})
        self.assertEqual(record["before"], record["after"])
        self.assertEqual(len(self.starts()), 1)

    def test_64gb_edition_tries_one_gpu_and_keeps_two(self):
        snap = self.run_tuning("flash-next-64gb")
        self.assertEqual(snap["state"], "done", snap["error"])
        record = self.store.record("flash-next-64gb")
        steps = {t["step"] for t in record["trials"]}
        self.assertEqual(steps, {"baseline", "threads", "stream", "gpus", "baseline_again", "final"})
        self.assertNotIn("gpus", record["flags"], "one GPU is slower here: not adopted")
        one = [s for s in self.starts() if s["env"].get("NINFER_DEVICES") == "0"]
        self.assertEqual(len(one), 1)
        self.assertNotIn("--host-kv-mib", one[0]["argv"], "install.sh's two-GPU flags are not passed with one GPU")
        self.assertIn("--host-kv-mib", self.starts()[0]["argv"])

    def test_kv_in_ram_only_when_long_contexts_gain(self):
        snap = self.run_tuning("flash-next", long_context=True)
        self.assertEqual(snap["state"], "done", snap["error"])
        record = self.store.record("flash-next")
        self.assertTrue(record["flags"].get("kv_ram_active"))
        self.assertIn("long_decode_tps", record["after"])
        self.assertEqual(self.flags_of(self.starts()[-1]).get("--kv-ram-active"), True)

    def test_candidate_that_cannot_start_is_skipped(self):
        os.environ["FAKE_ENGINE_CORES"] = "7"  # the engine refuses more threads than cores
        snap = self.run_tuning("flash-next")
        self.assertEqual(snap["state"], "done", snap["error"])
        failed = [t for t in snap["trials"] if t["error"]]
        self.assertEqual([(t["step"], t["label"]) for t in failed], [("threads", "8")])
        self.assertIn("exit code 1", failed[0]["error"])
        self.assertNotEqual(self.store.record("flash-next")["flags"].get("expert_cpu_threads"), 8)
        self.assertEqual(self.engine.state, "running")

    def test_threads_are_not_tried_when_the_log_does_not_say_the_engines_count(self):
        log_tail = self.engine.log_tail
        self.engine.log_tail = lambda lines=200: [l for l in log_tail(lines) if "routed experts" not in l]
        snap = self.run_tuning("flash-next")
        self.assertEqual(snap["state"], "done", snap["error"])
        record = self.store.record("flash-next")
        self.assertNotIn("threads", [t["step"] for t in record["trials"]])
        self.assertEqual(record["flags"], {"miss_split": "calibrate", "stream_min_tokens": 128})
        self.assertEqual(snap["done"], snap["total"], "the skipped step's runs are taken off the count")

    def test_cancel_restores_the_engine_and_saves_nothing(self):
        os.environ["FAKE_ENGINE_LOAD"] = "0.3"
        self.tuner.start("flash-next")
        deadline = time.time() + 20
        while time.time() < deadline and self.tuner.snapshot()["done"] < 2:
            time.sleep(0.02)
        self.assertTrue(self.tuner.cancel())
        self.tuner.wait(30)
        self.assertEqual(self.tuner.snapshot()["state"], "cancelled")
        self.assertIsNone(self.store.record("flash-next"))
        self.wait_state("running")
        self.assertEqual(self.flags_of(self.starts()[-1]), {}, "back to how it ran before")

    def test_refused_like_any_engine_start(self):
        self.allowed = False
        with self.assertRaises(EngineError) as c:
            self.tuner.start("flash-next")
        self.assertEqual(c.exception.code, "not_local")
        self.assertEqual(self.starts(), [])
        self.assertEqual(self.tuner.snapshot()["state"], "idle")

    def test_profile_from_other_hardware_is_not_applied(self):
        self.run_tuning("flash-next")
        other = FakeSystem(gpus=GPUS_1X5060)
        from capy_app import machine

        facts = machine.probe(other, self.root)
        self.assertNotEqual(signature(facts), self.store.record("flash-next")["machine"])
        self.assertEqual(self.store.flags("flash-next", facts), {})

    def test_load_time_feeds_the_estimate(self):
        os.environ["FAKE_ENGINE_LOAD"] = "0.5"
        self.engine.start("flash-next")
        self.wait_state("running")
        self.assertGreaterEqual(self.store.load_seconds("flash-next"), 0.4)
        measured = self.tuner.plan("flash-next")["seconds"]
        self.store.note_load("flash-next", 90.0)
        self.assertLess(
            measured, self.tuner.plan("flash-next")["seconds"] / 2, "the measured load time, not the default"
        )

    def test_first_request_after_every_start_is_not_measured(self):
        # The first request of every engine process is 15 % slower: one discarded warm-up request absorbs it.
        os.environ["FAKE_ENGINE_SLOW_FIRST"] = "1"
        snap = self.run_tuning("flash-next")
        self.assertEqual(snap["state"], "done", snap["error"])
        record = self.store.record("flash-next")
        for trial in record["trials"]:
            self.assertGreaterEqual(trial["metrics"]["prefill_tps"], 4200.0, (trial["step"], trial["label"]))
        self.assertAlmostEqual(record["before"]["decode_tps"], 138.0, delta=0.2)
        self.assertEqual(
            record["flags"], {"expert_cpu_threads": 10, "miss_split": "calibrate", "stream_min_tokens": 128}
        )

    def test_cold_first_baseline_does_not_fake_a_gain(self):
        # A cold machine: the first measurement of the defaults is slower than everything after it (here by
        # 12 %, for the whole first engine process), so every candidate looks like a gain against it. On this
        # machine the engine's own thread count (10) is already the best: 12 threads are really 4 % slower.
        os.environ["FAKE_ENGINE_COLD_ONCE"] = str(Path(self.tmp.name) / "warmed-up")
        os.environ["FAKE_ENGINE_CORES"] = "20"
        snap = self.run_tuning("flash-next")
        self.assertEqual(snap["state"], "done", snap["error"])
        record = self.store.record("flash-next")
        cold, warm = record["baselines"]
        self.assertAlmostEqual(cold["decode_tps"], 132.0, delta=0.2)
        self.assertAlmostEqual(warm["decode_tps"], 150.0, delta=0.2)
        twelve = [t for t in record["trials"] if (t["step"], t["label"]) == ("threads", "12")][0]
        self.assertGreater(gain(twelve["metrics"], cold, "decode_tps"), 0.09, "+9 % against the cold baseline...")
        self.assertLess(gain(twelve["metrics"], warm, "decode_tps"), 0, "...and slower than the defaults in fact")
        self.assertFalse(twelve["adopted"])
        self.assertEqual(record["before"], warm, "the reference is the better of the two baselines")
        self.assertAlmostEqual(record["noise"]["decode_tps"], 0.12, delta=0.002)
        self.assertEqual(snap["noise"], record["noise"])
        # with 12 % between two runs of the same defaults, the real +7 % of calibrate cannot be told from noise
        self.assertEqual(record["flags"], {}, "nothing is adopted below the measured noise")
        self.assertEqual([t for t in record["trials"] if t["adopted"]], [])
        self.assertEqual(record["after"], record["before"])
        self.assertEqual(self.flags_of(self.starts()[-1]), {}, "the engine is left with its defaults")

    def test_gains_above_the_noise_are_still_kept(self):
        os.environ["FAKE_ENGINE_COLD_ONCE"] = str(Path(self.tmp.name) / "warmed-up")
        os.environ["FAKE_ENGINE_COLD_FACTOR"] = "0.97"  # a first run 3 % slower: noise just above the 3 % rule
        os.environ["FAKE_ENGINE_CORES"] = "20"
        snap = self.run_tuning("flash-next")
        self.assertEqual(snap["state"], "done", snap["error"])
        record = self.store.record("flash-next")
        self.assertAlmostEqual(record["noise"]["decode_tps"], 0.03, delta=0.002)
        self.assertEqual(
            record["flags"],
            {"miss_split": "calibrate", "stream_min_tokens": 128},
            "+7 % and +10 % clear both the 3 % rule and the 3 % noise; 12 threads (-4 %) do not",
        )
        self.assertAlmostEqual(record["before"]["decode_tps"], 150.0, delta=0.2)
        self.assertAlmostEqual(record["after"]["decode_tps"], 160.5, delta=0.2)

    def test_settings_that_do_not_hold_together_fall_back_to_the_best_one(self):
        # Force the "measured together" check to fail: the combination is measured as worse than the defaults.
        real = tuning.measure

        def measure(base, model_id, long_context=False, cancel=None, nonce=None):
            metrics = real(base, model_id, long_context, cancel, nonce)
            if len(self.flags_of(self.starts()[-1])) > 1:
                metrics = {k: (v * 2 if k == "short_ms" else v * 0.5) for k, v in metrics.items()}
            return metrics

        tuning.measure = measure
        try:
            snap = self.run_tuning("flash-next")
        finally:
            tuning.measure = real
        self.assertEqual(snap["state"], "done", snap["error"])
        record = self.store.record("flash-next")
        self.assertEqual(record["flags"], {"stream_min_tokens": 128}, "the single setting with the largest gain")
        self.assertEqual([(t["step"], t["label"]) for t in record["trials"] if t["adopted"]], [("stream", "128")])
        self.assertEqual(self.flags_of(self.starts()[-1]), {"--stream-min-tokens": "128"})


if __name__ == "__main__":
    unittest.main()
