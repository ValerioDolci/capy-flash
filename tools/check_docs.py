#!/usr/bin/env python3
"""Documentation guard: what the code accepts must be in the user docs.

  - every `ninfer-serve` flag parsed in src/serve/serve_options.cpp appears in the option tables of
    the serving guide, docs/serving.md and docs/serving/*.md (a table row naming it in backticks);
  - every NINFER_* environment variable read with getenv() under src/ appears in
    docs/capy-flash/diagnostics.md or the serving guide;
  - every conversion recipe in tools/convert/official_recipes.py appears in docs/weight-conversion.md.

Usage: check_docs.py [repository root]. Exit 1 with the list of what is missing.
"""

import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
read = lambda p: (root / p).read_text(encoding="utf-8")
sources = [
    p for d in ("src", "include") for p in (root / d).rglob("*") if p.suffix in {".cpp", ".h", ".cu", ".cuh", ".inc"}
]
missing = []

serving_pages = ["docs/serving.md"] + sorted(str(p.relative_to(root)) for p in (root / "docs/serving").glob("*.md"))
serving = "\n".join(read(p) for p in serving_pages)
table_rows = "\n".join(l for l in serving.splitlines() if l.startswith("|"))
flags = sorted(set(re.findall(r'arg == "(--[a-z0-9-]+)"', read("src/serve/serve_options.cpp"))))
missing += [
    f"serve flag {f} not in an option table of docs/serving.md or docs/serving/"
    for f in flags
    if f"`{f}`" not in table_rows and f"`{f} " not in table_rows
]

env_docs = read("docs/capy-flash/diagnostics.md") + serving
envs = sorted({m for p in sources for m in re.findall(r'getenv\("(NINFER_[A-Z0-9_]+)"', p.read_text(errors="replace"))})
missing += [
    f"environment variable {e} not in docs/capy-flash/diagnostics.md, docs/serving.md or docs/serving/"
    for e in envs
    if not re.search(rf"`{e}`", env_docs)
]

recipes_src = read("tools/convert/official_recipes.py")
block = recipes_src[recipes_src.index("RECIPES = {") :]
recipes = re.findall(r'^\s+"([a-z0-9_]+)":', block[: block.index("}")], re.M)
conversion = read("docs/weight-conversion.md")
missing += [f"recipe {r} not in docs/weight-conversion.md" for r in recipes if f"`{r}`" not in conversion]

if missing:
    print("\n".join(missing))
    print(f"check_docs: {len(missing)} missing")
    sys.exit(1)
print(f"check_docs: {len(flags)} flags, {len(envs)} variables, {len(recipes)} recipes documented")
