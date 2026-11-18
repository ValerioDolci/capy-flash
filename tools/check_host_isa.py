#!/usr/bin/env python3
"""Checks that no AVX / AVX2 / AVX-512 instruction runs outside the run-time-dispatched host kernels.

The product targets baseline x86-64 and picks the expert kernels with cpuid (host_isa(),
src/ops/host_moe/host_nvfp4.h), so a binary must start on a CPU without AVX-512 (Zen 2/3,
Intel client) and without AVX2. A VEX/EVEX instruction in any other function (a static initializer,
a COMDAT copy of an inline/template function emitted by a file compiled with -mavx*, a probe)
would raise SIGILL there. This scans the disassembly of the given object files / static archives /
build directories and fails on such functions.

Allowed: functions whose demangled name starts with one of ALLOWED_PREFIXES (the ISA namespaces,
entered only after host_isa_supported()). Static initializers (_GLOBAL__sub_I_*) are never allowed.

usage: check_host_isa.py PATH... (files ending in .o / .a, or directories searched for them)
"""
import os
import re
import subprocess
import sys

ALLOWED_PREFIXES = (
    "ninfer::ops::host::detail::avx2::",
    "ninfer::ops::host::detail::avx512::",
    "ninfer::ops::host::gguf::detail::avx2::",
    "ninfer::ops::host::gguf::detail::avx512::",
)
# The qualified name of a demangled function, after an optional return type (templates).
QUALIFIED = re.compile(r"^(?:[^()]*?\s)?((?:[A-Za-z_]\w*|\(anonymous namespace\))::[^(]*)\(")

# VEX/EVEX-only forms: AVX mnemonics start with 'v' (vmovaps, vpshufb, vfmadd..., vzeroupper),
# AVX-512 mask ops with 'k'; ymm/zmm/k registers in operands.
WIDE_MNEMONIC = re.compile(r"^(v[a-z0-9]+|k(?:and|or|xor|not|mov|shift|test|add|unpck)[a-z0-9]*)$")
WIDE_OPERAND = re.compile(r"%(?:[yz]mm\d+|k[1-7])\b")
FUNC = re.compile(r"^[0-9a-f]+ <(.+)>:$")


def objects(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith((".o", ".a")):
                        yield os.path.join(root, f)
        else:
            yield p


def wide_functions(path):
    out = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", path],
                         capture_output=True, text=True, check=False).stdout
    current, hits = None, {}
    for line in out.splitlines():
        m = FUNC.match(line)
        if m:
            current = m.group(1)
            continue
        parts = line.split("\t")
        if current is None or len(parts) < 2:
            continue
        insn = parts[-1].strip().split(None, 1)
        if not insn:
            continue
        mnemonic = insn[0]
        operands = insn[1] if len(insn) > 1 else ""
        if WIDE_MNEMONIC.match(mnemonic) or WIDE_OPERAND.search(operands):
            hits.setdefault(current, mnemonic)
    return hits


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    bad, scanned, allowed = [], 0, 0
    for path in objects(argv[1:]):
        scanned += 1
        for func, mnemonic in wide_functions(path).items():
            base = re.sub(r" \[clone [^\]]*\]$", "", func)
            m = QUALIFIED.match(base)
            name = m.group(1) if m else base
            if base.startswith("_GLOBAL__sub_I") or not name.startswith(ALLOWED_PREFIXES):
                bad.append((path, func, mnemonic))
            else:
                allowed += 1
    for path, func, mnemonic in bad:
        print(f"WIDE {os.path.basename(path)}: {func} ({mnemonic})")
    print(f"{scanned} objects, {allowed} dispatched wide functions, {len(bad)} undispatched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
