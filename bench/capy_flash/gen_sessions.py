#!/usr/bin/env python3
"""Synthetic agentic sessions shaped like a real production log of a coding agent (2 201 tool-history
requests): prompt p50 35k / p90 84k tokens, new tokens per turn p50 3.8k, output p50 254.
Each session: a system prompt (~8k tokens: tool descriptions + project docs), a task, then turns of
[fixed assistant message, tool output (source code / docs from the ninfer tree)] until ~MAX tokens.
Deterministic (seed) for a given source tree. Token counts with the model tokenizer.
Usage: gen_sessions.py <tokenizer.json> <source tree> <out sessions.json> <sessions> <target tokens>
The sessions.json next to this script is the one every published number used (do not regenerate to compare)."""

import json
import os
import random
import sys
from tokenizers import Tokenizer

TOK = Tokenizer.from_file(sys.argv[1])
ntok = lambda s: len(TOK.encode(s).ids)
ROOT = sys.argv[2]
out, nsess, target = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rng = random.Random(71)

files = []
for d, _, fs in os.walk(ROOT):
    if "/.git" in d or "/build" in d or "/third_party" in d:
        continue
    for f in fs:
        if f.endswith((".cpp", ".h", ".cu", ".inc", ".md", ".py", ".cmake", ".txt")):
            p = os.path.join(d, f)
            if 2000 < os.path.getsize(p) < 400000:
                files.append(p)
files.sort()
docs = [p for p in files if p.endswith(".md")]

tools = []
for name, desc in [
    ("read_file", "Read a file of the repository, optionally a line range."),
    ("grep", "Search the repository with a regular expression."),
    ("list_dir", "List a directory."),
    ("edit_file", "Replace an exact string in a file."),
    ("run", "Run a shell command in the sandbox and return stdout/stderr."),
    ("build", "Build CMake targets with ninja."),
    ("test", "Run ctest with a filter."),
    ("git_diff", "Show the working tree diff."),
    ("git_log", "Show recent commits."),
    ("web_fetch", "Fetch a URL and return text."),
]:
    tools.append(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": desc
                + " "
                + " ".join(
                    [
                        "Returns the result as plain text; large results are truncated at 20 000 characters, errors are returned with a non-zero status field and the tool never throws."
                    ]
                    * 2
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repository-relative path."},
                        "pattern": {"type": "string"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "command": {"type": "string"},
                        "timeout_s": {"type": "integer"},
                    },
                    "required": [],
                },
            },
        }
    )
system = (
    "You are a senior C++/CUDA engineer working on ninfer, a tensor-parallel inference engine. Use the tools to read code before changing it, keep changes minimal, run the tests.\n\n# Tools\n"
    + json.dumps(tools, indent=1)
    + "\n\n# Project notes\n"
)
for p in docs:
    if ntok(system) > 8000:
        break
    system += "\n## " + os.path.relpath(p, ROOT) + "\n" + open(p, errors="ignore").read()[:6000]
tasks = [
    "Find why the tensor-parallel prefill is slower on short prompts and propose a fix.",
    "Review the expert cache policy for correctness issues under elastic resizing.",
    "Explain how the MTP verification window is committed and where host time is spent.",
    "Audit the all-reduce paths for cross-device copies that could fail without peer access.",
]


def chunk_of(path, n_tokens):
    text = open(path, errors="ignore").read()
    ids = TOK.encode(text).ids
    if len(ids) > n_tokens:
        s = rng.randrange(0, len(ids) - n_tokens)
        ids = ids[s : s + n_tokens]
    return TOK.decode(ids)


sessions = []
for k in range(nsess):
    turns, total = [], ntok(system) + 60
    while total < target and len(turns) < 40:
        want = int(min(12000, max(300, rng.lognormvariate(8.0, 0.8))))  # median ~3k tokens
        parts, got = [], 0
        while got < want * 0.9:
            p = rng.choice(files)
            c = chunk_of(p, want - got)
            t = ntok(c)
            if t < 50:
                continue
            parts.append(f"===== {os.path.relpath(p, ROOT)} =====\n{c}")
            got += t
        tool_out = "Tool result (status 0):\n" + "\n".join(parts)
        f0 = os.path.relpath(rng.choice(files), ROOT)
        asst = (
            f"I will inspect `{f0}` and the related code next to check how the data flows across the call sites. "
            + "So far the relevant path is clear; the open question is the ordering between the host and the device streams. "
            * rng.randint(1, 3)
        )
        turns.append({"assistant": asst, "tool": tool_out})
        total += ntok(asst) + ntok(tool_out) + 20
    sessions.append({"task": tasks[k % len(tasks)], "turns": turns, "approx_tokens": total})
    print(f"session {k}: {len(turns)} turns, ~{total} tokens", flush=True)
json.dump({"system": system, "system_tokens": ntok(system), "sessions": sessions}, open(out, "w"))
