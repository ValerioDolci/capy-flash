#!/usr/bin/env python3
"""Ratchets: per-file counts of patterns the project is phasing out. A count may only go down.

tools/ratchets/<rule>.allow holds one "<count> <path>" line per file that still has occurrences. The check
fails when a file exceeds its allowance or is not listed (a new occurrence), and also when a file is below
its allowance or has none left: the change that removes occurrences updates the list, so the improvement
stays. `--update` rewrites the lists from the tree; use it only after removing occurrences.

Usage: tools/check_ratchets.py [--update]
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("src", "include", "apps")
SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inc"}

# rule -> (pattern, comments only, why)
RULES = {
    "getenv": (
        re.compile(r"\bgetenv\s*\("),
        False,
        "behaviour is selected by documented flags; validation hooks are listed in docs/capy-flash/diagnostics.md",
    ),
    "source_include": (
        re.compile(r'#\s*include\s+"[^"]+\.(?:cu|cpp|inc)"'),
        False,
        "a translation unit does not include another one; shared code goes in headers",
    ),
    "comment_state": (
        re.compile(r"\bT\d{2,3}\b|\b20\d\d-[01]\d\b|\bv0\.\d+|\bthis fork\b|\bOur file\b|\bupstream\b"),
        True,
        "comments describe the code as it is: no measurement IDs, dates, versions or history",
    ),
    "stderr": (
        re.compile(r"std::cerr|fprintf\s*\(\s*stderr|fputs\s*\([^;]*,\s*stderr\s*\)"),
        False,
        "messages go through the logger with a level",
    ),
}


STRING_LITERAL = re.compile(r'"(?:\\.|[^"\\])*"')


def comment_text(lines):
    """Yield the comment part of each line (// and /* */ comments; string literals are skipped)."""
    in_block = False
    for line in lines:
        out = []
        rest = line if in_block else STRING_LITERAL.sub('""', line)
        while rest:
            if in_block:
                end = rest.find("*/")
                if end < 0:
                    out.append(rest)
                    rest = ""
                else:
                    out.append(rest[:end])
                    rest = rest[end + 2 :]
                    in_block = False
                continue
            line_comment, block_comment = rest.find("//"), rest.find("/*")
            if line_comment >= 0 and (block_comment < 0 or line_comment < block_comment):
                out.append(rest[line_comment + 2 :])
                rest = ""
            elif block_comment >= 0:
                rest = rest[block_comment + 2 :]
                in_block = True
            else:
                rest = ""
        yield " ".join(out)


def count(rule):
    pattern, comments_only, _ = RULES[rule]
    counts = {}
    for top in SOURCES:
        for path in sorted((ROOT / top).rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            texts = comment_text(lines) if comments_only else lines
            n = sum(len(pattern.findall(text)) for text in texts)
            if n:
                counts[path.relative_to(ROOT).as_posix()] = n
    return counts


def allow_path(rule):
    return ROOT / "tools" / "ratchets" / f"{rule}.allow"


def read_allow(rule):
    allowed = {}
    path = allow_path(rule)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                n, file = line.split(maxsplit=1)
                allowed[file] = int(n)
    return allowed


def write_allow(rule, counts):
    _, _, why = RULES[rule]
    body = [f"# {rule}: {why}.", "# <count> <path>; counts may only go down (tools/check_ratchets.py)."]
    body += [f"{n} {file}" for file, n in sorted(counts.items())]
    allow_path(rule).parent.mkdir(parents=True, exist_ok=True)
    allow_path(rule).write_text("\n".join(body) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="rewrite the allow lists from the tree")
    args = parser.parse_args()
    failures = []
    for rule in RULES:
        counts = count(rule)
        if args.update:
            write_allow(rule, counts)
            print(f"{rule}: {sum(counts.values())} in {len(counts)} files")
            continue
        allowed = read_allow(rule)
        for file, n in sorted(counts.items()):
            limit = allowed.get(file)
            if limit is None:
                failures.append(f"{rule}: {file} has {n}, not allowed ({RULES[rule][2]})")
            elif n > limit:
                failures.append(f"{rule}: {file} has {n}, allowed {limit} ({RULES[rule][2]})")
            elif n < limit:
                failures.append(f"{rule}: {file} went down to {n} from {limit}: update tools/ratchets/{rule}.allow")
        for file in sorted(set(allowed) - set(counts)):
            failures.append(f"{rule}: {file} has none left: remove it from tools/ratchets/{rule}.allow")
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
