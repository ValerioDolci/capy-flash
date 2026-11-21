"""Real Inkling 2 Mini turns for the frontend tests, from the greedy reference runs.

Reads the greedy outputs of tools/ref_ink/greedy.py (one <name>.json per prompt: prompt ids,
generated ids, whether generation stopped on EOS) and the prompt definitions
(tools/ref_ink/prompts/prompts.json), and writes a small JSON file with, per short prompt, the
messages, tools and effort, the reference prompt ids and the generated ids. The C++ frontend test
(tests/models/inkling2/test_frontend.cpp) renders the same prompts and replays the generated ids
through an output session. Prompts with a long attached text are skipped.

Usage: python3 frontend_greedy_fixture.py <greedy dir> <prompts.json> <output json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("greedy_dir")
    parser.add_argument("prompts")
    parser.add_argument("output")
    args = parser.parse_args()

    prompts = {p["name"]: p for p in json.loads(Path(args.prompts).read_text(encoding="utf-8"))}
    cases = []
    for path in sorted(Path(args.greedy_dir).glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        prompt = prompts.get(run["name"])
        if prompt is None or "long_text" in prompt:
            continue
        case = {"name": run["name"], "messages": prompt["messages"], "effort": run["effort"]}
        if prompt.get("tools"):
            case["tools"] = prompt["tools"]
        case["prompt_ids"] = run["prompt_ids"]
        case["generated_ids"] = run["generated_ids"]
        case["stopped_on_eos"] = run["stopped_on_eos"]
        case["text"] = run["text"]
        cases.append(case)
    fixture = {"source": "tools/ref_ink/greedy.py (transformers, CPU, greedy)", "cases": cases}
    Path(args.output).write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
