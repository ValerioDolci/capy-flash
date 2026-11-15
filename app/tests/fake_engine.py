#!/usr/bin/env python3
"""A stand-in for ninfer-serve, for the tests and the demo: same command line shape, /health, /v1/models and
/v1/chat/completions with llama.cpp-style `timings` that depend on the flags it was started with.

    fake_engine.py <artifact> [--host H] [--port N] [--model-id ID] [--devices A,B | --device N]
                   [--expert-cpu-threads N] [--miss-split M] [--stream-min-tokens N] [--kv-ram-active] ...

Environment (tests): FAKE_ENGINE_LOAD (seconds before /health answers), FAKE_ENGINE_FAIL (exit 1 at startup),
FAKE_ENGINE_CRASH_ONCE (a file: if missing, create it and exit 2 on the first chat request), FAKE_ENGINE_CORES
(physical cores: more threads than this stops the startup, as the real engine does), FAKE_ENGINE_RECORD (a file
that gets one JSON line per start: argv and the NINFER_* environment), FAKE_ENGINE_TOKEN_DELAY (seconds per
streamed chunk), FAKE_ENGINE_SLOW_FIRST (the first N requests of every process report 15 % lower rates: the cost
of a first request), FAKE_ENGINE_COLD_ONCE (a file: the process that finds it missing creates it and reports
FAKE_ENGINE_COLD_FACTOR, default 0.88, of every rate for as long as it lives: a cold first run of the machine).
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse(argv):
    opts = {
        "artifact": argv[0] if argv else "",
        "host": "127.0.0.1",
        "port": 8080,
        "model_id": "capy-flash",
        "threads": None,
        "miss_split": "off",
        "stream_min": 256,
        "kv_ram": False,
        "devices": "0,1",
    }
    i = 1
    while i < len(argv):
        a = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else ""
        if a == "--host":
            opts["host"] = nxt
            i += 1
        elif a == "--port":
            opts["port"] = int(nxt)
            i += 1
        elif a == "--model-id":
            opts["model_id"] = nxt
            i += 1
        elif a == "--devices":
            opts["devices"] = nxt
            i += 1
        elif a == "--device":
            opts["devices"] = nxt
            i += 1
        elif a == "--expert-cpu-threads":
            opts["threads"] = int(nxt)
            i += 1
        elif a == "--miss-split":
            opts["miss_split"] = nxt
            i += 1
        elif a == "--stream-min-tokens":
            opts["stream_min"] = int(nxt)
            i += 1
        elif a == "--kv-ram-active":
            opts["kv_ram"] = True
        i += 1
    return opts


def speeds(opts, cores, prompt_tokens):
    """A made-up but deterministic machine: 10 expert threads are best, calibrate beats auto beats off, two GPUs
    beat one, streaming from 128 tokens helps short prompts, KV in RAM helps only long contexts."""
    auto = max(6, cores // 2)
    threads = opts["threads"] or auto
    decode = 150.0 * (1.0 - 0.02 * abs(threads - 10))
    if threads >= cores:
        decode *= 0.6  # no core left for the main thread
    decode *= {"off": 1.0, "auto": 1.03, "calibrate": 1.07}.get(opts["miss_split"], 1.0)
    two = "," in opts["devices"]
    decode *= 1.0 if two else 0.55
    prefill = 4200.0 if two else 2900.0
    if opts["kv_ram"]:
        decode *= 1.06 if prompt_tokens > 12000 else 0.99
        prefill *= 0.98
    if prompt_tokens < 1000:
        prefill = 1900.0 if opts["stream_min"] <= 128 else 1700.0 if opts["stream_min"] <= 256 else 1100.0
    return decode, prefill


def main():
    opts = parse(sys.argv[1:])
    cores = int(os.environ.get("FAKE_ENGINE_CORES", "12"))
    record = os.environ.get("FAKE_ENGINE_RECORD")
    if record:
        with open(record, "a") as f:
            f.write(
                json.dumps(
                    {"argv": sys.argv[1:], "env": {k: v for k, v in os.environ.items() if k.startswith("NINFER_")}}
                )
                + "\n"
            )
    print(f"ninfer-serve (fake) loading {os.path.basename(opts['artifact'])}", flush=True)
    if os.environ.get("FAKE_ENGINE_FAIL"):
        print("error: CUDA out of memory while loading the weights", file=sys.stderr, flush=True)
        sys.exit(1)
    if opts["threads"] and opts["threads"] > cores:
        print("error: host MoE pool: fewer physical cores than requested threads", file=sys.stderr, flush=True)
        sys.exit(1)
    auto = max(6, cores // 2)
    rule = "set by the caller (--expert-cpu-threads)" if opts["threads"] else "auto: max(6, cores / 2)"
    print(
        f"routed experts: {opts['threads'] or auto} CPU threads for the missing experts ({rule}; {cores} physical "
        f"cores, {cores * 2} logical CPUs allowed, 1 L3 domains, 1 NUMA nodes)",
        flush=True,
    )
    if opts["miss_split"] == "calibrate":
        print("miss split calibrate: PCIe copy 0.41 ms per expert", flush=True)
    ready_at = time.time() + float(os.environ.get("FAKE_ENGINE_LOAD", "0"))
    slow_first = int(os.environ.get("FAKE_ENGINE_SLOW_FIRST", "0"))
    cold_marker = os.environ.get("FAKE_ENGINE_COLD_ONCE")
    cold = 1.0
    if cold_marker and not os.path.exists(cold_marker):
        open(cold_marker, "w").close()
        cold = float(os.environ.get("FAKE_ENGINE_COLD_FACTOR", "0.88"))
    crash_once = os.environ.get("FAKE_ENGINE_CRASH_ONCE")
    delay = float(os.environ.get("FAKE_ENGINE_TOKEN_DELAY", "0"))
    state = {"calls": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                if time.time() < ready_at:
                    return self._json(503, {"status": "loading"})
                return self._json(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self._json(
                    200,
                    {"object": "list", "data": [{"id": opts["model_id"], "object": "model", "max_model_len": 262144}]},
                )
            self._json(404, {})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path != "/v1/chat/completions":
                return self._json(404, {})
            if crash_once and not os.path.exists(crash_once):
                open(crash_once, "w").close()
                print("fatal: Engine worker failed", file=sys.stderr, flush=True)
                os._exit(2)
            req = json.loads(body or b"{}")
            text = " ".join(str(m.get("content", "")) for m in req.get("messages", []))
            prompt_tokens = max(1, len(text) // 4)
            want = int(req.get("max_tokens") or 64)
            decode, prefill = speeds(opts, cores, prompt_tokens)
            state["calls"] += 1
            slow = cold * (0.85 if state["calls"] <= slow_first else 1.0)
            decode, prefill = decode * slow, prefill * slow
            if opts["miss_split"] == "calibrate" and state["calls"] == 2:
                print("miss split calibrate: fitted fixed 0.9 ms + 0.21 ms per expert", flush=True)
            timings = {
                "cache_n": 0,
                "prompt_n": prompt_tokens,
                "prompt_ms": prompt_tokens / prefill * 1000.0,
                "prompt_per_second": prefill,
                "predicted_n": want,
                "predicted_ms": (want - 1) / decode * 1000.0,
                "predicted_per_second": decode,
            }
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": want}
            words = (
                "capy-flash keeps the experts that do not fit in VRAM in RAM and computes them on the CPU, "
                "next to the GPUs. "
            ).split(" ")
            if not req.get("stream"):
                content = " ".join(words[i % len(words)] for i in range(min(want, 60)))
                return self._json(
                    200,
                    {
                        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "length"}],
                        "usage": usage,
                        "timings": timings,
                    },
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(obj):
                data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()

            try:
                for i in range(min(want, 80)):
                    chunk({"choices": [{"delta": {"content": words[i % len(words)] + " "}}]})
                    if delay:
                        time.sleep(delay)
                chunk({"choices": [], "usage": usage, "timings": timings})
                chunk("[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer((opts["host"], opts["port"]), Handler)
    server.daemon_threads = True
    print(f"listening on http://{opts['host']}:{opts['port']}", flush=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
