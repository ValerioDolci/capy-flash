"""Agentic latency and report tests."""
import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BENCH = Path(__file__).resolve().parents[1] / "bench" / "agentic_ab"
sys.path.insert(0, str(BENCH))
import runner
import analyze


def chunk(delta=None, finish=None):
    return {"choices": [{"delta": delta or {}, "finish_reason": finish}]}


def tool(arguments, name="run", index=0):
    return {"index": index, "id": "call_%d" % index,
            "function": {"name": name, "arguments": arguments}}


class ClientMeasurementTests(unittest.TestCase):
    def run_stream(self, events):
        clock = [10_000_000_000]

        class Response:
            status = 200

            def __iter__(self):
                for seconds, event in events:
                    clock[0] = 10_000_000_000 + int(seconds * 1e9)
                    if isinstance(event, Exception):
                        raise event
                    payload = event if isinstance(event, str) else json.dumps(event)
                    yield ("data: " + payload + "\n").encode()

        class Connection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return Response()

            def close(self):
                pass

        client = runner.ArmClient("a", {"personas": {}, "actors": []}, "model", "/unused", 8192)
        with patch.object(runner.http.client, "HTTPConnection", Connection), \
             patch.object(runner.time, "perf_counter_ns", side_effect=lambda: clock[0]), \
             patch.object(runner.time, "time", side_effect=[100, 50, 20, 10, 5, 1, 0]):
            return client.post([], [], 128, 42)

    def test_reasoning_is_not_tool_readiness_and_wall_clock_cannot_change_latency(self):
        result = self.run_stream([
            (0.1, chunk({"role": "assistant"})),
            (1, chunk({"reasoning_content": "think"})),
            (2, chunk({"content": "calling"})),
            (3, chunk({"tool_calls": [tool('{"x":')]})),
            (4, chunk({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]})),
            (5, chunk(finish="tool_calls")), (6, "[DONE]")])
        self.assertEqual(result.client_latency_seconds, {
            "first_output": 1.0, "first_reasoning": 1.0, "first_content": 2.0,
            "first_tool_delta": 3.0, "tool_ready": 5.0, "response_end": 6.0})
        self.assertEqual(result.valid_tool_calls, 1)
        self.assertEqual(result.invalid_tool_calls, 0)
        self.assertTrue(result.stream_complete)

    def test_json_prefix_is_not_ready_until_terminal_tool_confirmation(self):
        result = self.run_stream([(1, chunk({"tool_calls": [tool("{}")] })),
                                  (2, chunk(finish="length")), (3, "[DONE]")])
        self.assertIsNone(result.client_latency_seconds["tool_ready"])
        self.assertIsNone(result.valid_tool_calls)

    def test_invalid_arguments_and_missing_names_are_counted_at_terminal(self):
        calls = [tool("{}"), tool("[]", index=1), tool("{", index=2),
                 tool("{}", name="", index=3), tool('{"x":NaN}', index=4)]
        result = self.run_stream([(1, chunk({"tool_calls": calls})),
                                  (2, chunk(finish="tool_calls")), (3, "[DONE]")])
        self.assertEqual(result.valid_tool_calls, 1)
        self.assertEqual(result.invalid_tool_calls, 4)
        self.assertEqual(result.client_latency_seconds["tool_ready"], 2.0)

    def test_all_invalid_calls_never_become_ready(self):
        result = self.run_stream([(1, chunk({"tool_calls": [tool("null")]})),
                                  (2, chunk(finish="tool_calls")), (3, "[DONE]")])
        self.assertIsNone(result.client_latency_seconds["tool_ready"])
        self.assertEqual(result.invalid_tool_calls, 1)

    def test_transport_failure_keeps_observations_without_inventing_readiness(self):
        result = self.run_stream([(1, chunk({"tool_calls": [tool("{}")] })),
                                  (2, OSError("connection lost"))])
        self.assertEqual(result.status, "error")
        self.assertEqual(result.client_latency_seconds["first_tool_delta"], 1.0)
        self.assertIsNone(result.client_latency_seconds["tool_ready"])
        self.assertFalse(result.stream_complete)

    def test_eof_without_done_is_identified_as_incomplete(self):
        result = self.run_stream([(1, chunk({"content": "partial"}))])
        self.assertFalse(result.stream_complete)
        self.assertIsNone(result.client_latency_seconds["tool_ready"])

    def test_done_without_terminal_choice_is_incomplete(self):
        result = self.run_stream([(1, chunk({"content": "partial"})), (2, "[DONE]")])
        self.assertFalse(result.stream_complete)


class SummaryMeasurementTests(unittest.TestCase):
    def test_latency_comparisons_do_not_reward_missing_or_failed_tools(self):
        def request(seed, ready, status="ok"):
            return {"seed": seed, "cls": "loop", "status": status, "stream_complete": True,
                    "client_latency_seconds": {"tool_ready": ready}}

        seeds, distributions = analyze.paired_tool_readiness({
            "a": [request(1, 4), request(2, 100), request(3, 5)],
            "b": [request(1, 6), request(2, None), request(3, 1, "error")]})
        self.assertEqual(seeds, [1])
        self.assertEqual(distributions["a"], {"n": 1, "median": 4, "p90": 4, "max": 4})
        self.assertEqual(distributions["b"]["median"], 6)
        seeds, distributions = analyze.paired_tool_readiness({"a": [], "b": []})
        self.assertEqual(seeds, [])
        self.assertIsNone(distributions["a"]["median"])

    def test_client_jsonl_server_join_and_single_seed_and_aggregate_reports(self):
        result = runner.Result()
        result.status = "ok"
        result.finish = "tool_calls"
        result.stream_complete = True
        result.t_send, result.t_first, result.t_done = 100, 101, 106
        result.client_latency_seconds.update(first_output=1, first_tool_delta=3,
                                              tool_ready=5, response_end=6)
        result.tool_calls = [{"id": "call_0", "name": "run", "arguments": "{}"}]
        result.valid_tool_calls, result.invalid_tool_calls = 1, 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {"arms": ["a", "b"], "labels": {"a": "Baseline", "b": "Candidate"},
                   "short": {"a": "A", "b": "B"}, "harness": "linux", "model": "model.ninfer",
                   "agent_max_tokens": 128, "seed": 42, "scale": 0.1, "corpus_commit": "fixture",
                   "exes": {"a": "serve-a", "b": "serve-b"}, "flags": []}
            (root / "config.json").write_text(json.dumps(cfg))
            server = {"event": "request_done", "timestamp_unix_ms": 106000,
                      "request": {"sampling": {"seed": 42}},
                      "result": {"prompt_tokens": 100, "completion_tokens": 10,
                                 "model_thinking_tokens": 6, "finish_reason": "stop_token",
                                 "tool_call_parse": {"marker_seen": True,
                                     "fallback_reason": "none", "schema_mismatch_arguments": 0}},
                      "timings_seconds": {"ttft": 1, "prefill": 0.5, "decode": 4, "total": 6}}
            for arm in cfg["arms"]:
                target = root / arm
                target.mkdir()
                client = runner.ArmClient(arm, {"personas": {}, "actors": []}, "model",
                                          str(target), 8192)
                with patch.object(client, "post", return_value=result), patch.object(runner, "log"):
                    client.request({"name": "agent"}, {"tag": "turn"}, [], [], 128, 42, "loop")
                (target / "request_log.jsonl").write_text(json.dumps(server) + "\n")
            with patch("sys.stdout", new_callable=io.StringIO):
                analyze.main([str(root)])
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["a"]["agentic"]["thinking_fraction"], 0.6)
            self.assertEqual(summary["a"]["agentic"]["latency_seconds"]["tool_ready"]["median"], 5)
            report = (root / "report.md").read_text()
            self.assertIn("| tool_ready | 1 | 5.00 | 5.00 | 5.00 |", report)
            self.assertIn("| turn | ok | True | 1.00 | n/a | n/a | 3.00 | 5.00 | 6.00 | none |", report)
            combined = root / "combined"
            combined.mkdir()
            with patch("sys.stdout", new_callable=io.StringIO):
                analyze.main(["--aggregate", str(combined), str(root)])
            self.assertIn("Client time to terminal JSON-object tool, maximum (s, common requests)",
                          (combined / "report.md").read_text())

    def test_distributions_cover_successes_and_expose_missing_failed_and_parser_cases(self):
        requests = [
            {"cls": "loop", "status": "ok", "stream_complete": True,
             "client_latency_seconds": {"first_output": 1, "tool_ready": 4},
             "valid_tool_calls": 1, "invalid_tool_calls": 0,
             "tool_calls": 1, "completion": 10, "thinking": 6,
             "server": {}, "thinking_observed": True, "tool_call_parse": {"marker_seen": True,
                 "fallback_reason": "none", "schema_mismatch_arguments": 2}},
            {"cls": "loop", "status": "ok", "stream_complete": True,
             "client_latency_seconds": {"first_output": 2, "tool_ready": None},
             "valid_tool_calls": None, "invalid_tool_calls": None,
             "tool_calls": 0, "completion": 10, "thinking": 2,
             "server": {}, "thinking_observed": True, "tool_call_parse": {"marker_seen": True,
                 "fallback_reason": "trailing_content", "schema_mismatch_arguments": 0}},
            {"cls": "loop", "status": "error", "stream_complete": False,
             "client_latency_seconds": {"first_output": 3}},
            {"cls": "loop", "status": "ok", "stream_complete": False,
             "client_latency_seconds": {"first_output": 4}},
            {"cls": "aborted", "status": "aborted"},
            {"cls": "loop", "status": "ok"},
        ]
        measured = analyze.agentic_measurements(requests)
        self.assertEqual(measured["requests"], 5)
        self.assertEqual(measured["instrumented_requests"], 4)
        self.assertEqual(measured["complete_requests"], 2)
        self.assertEqual(measured["client_errors"], 1)
        self.assertEqual(measured["incomplete_streams"], 2)
        self.assertEqual(measured["latency_seconds"]["tool_ready"], {
            "n": 1, "median": 4, "p90": 4, "max": 4})
        self.assertEqual(measured["latency_seconds"]["first_output"]["n"], 2)
        self.assertEqual(measured["tool_ready_requests"], 1)
        self.assertEqual(measured["without_tool_calls"], 1)
        self.assertEqual(measured["parser_fallbacks"], {"trailing_content": 1})
        self.assertEqual(measured["schema_mismatch_arguments"], 2)
        self.assertEqual(measured["thinking_fraction"], 0.4)
        self.assertEqual(measured["latency_seconds"]["first_content"]["n"], 0)
        self.assertIsNone(measured["latency_seconds"]["first_content"]["median"])
        report = analyze.agentic_table(measured)
        self.assertIn("| tool_ready | 1 | 4.00 | 4.00 | 4.00 |", report)
        self.assertIn('"trailing_content": 1', report)
        self.assertIn("2 incomplete streams", report)

    def test_no_evidence_is_not_zero_latency_or_zero_parser_failures(self):
        measured = analyze.agentic_measurements([{"cls": "loop", "status": "ok"}])
        self.assertEqual(measured["parser_observed_requests"], 0)
        self.assertIsNone(measured["thinking_fraction"])
        self.assertIsNone(measured["latency_seconds"]["tool_ready"]["max"])


if __name__ == "__main__":
    unittest.main()
