"""What this computer is, and which capy-flash models it can run: the rules of install.sh --check.

Everything read from outside goes through a System object, so the tests feed fixture outputs of nvidia-smi and
/proc and the Linux-only logic runs anywhere.
"""

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import catalog

MIN_GPU_MIB = 15000  # install.sh: memory.total >= 15000 and compute_cap == 12.0
COMPUTE_CAP = "12.0"
MIN_DRIVER = 580  # CUDA 13.1 needs R580 or newer
AVX512 = ("avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx2", "fma", "f16c")
AVX2 = ("avx2", "fma", "f16c")
BUSY_MIB = 1024  # a process holding this much GPU memory means the GPU is in use


class System:
    """The outside world, as the probe sees it."""

    def uname(self) -> Tuple[str, str]:
        return platform.system(), platform.machine()

    def run(self, argv: List[str], timeout: float = 10.0) -> Optional[str]:
        """stdout of a command, or None when it is missing or fails."""
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout if out.returncode == 0 else None

    def read(self, path: str) -> Optional[str]:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None

    def disk_free(self, path: Path) -> Optional[int]:
        """Free bytes where `path` is or will be (the nearest existing parent, like install.sh)."""
        probe = Path(path)
        while not probe.is_dir():
            if probe.parent == probe:
                return None
            probe = probe.parent
        try:
            return shutil.disk_usage(str(probe)).free
        except OSError:
            return None


def parse_gpus(csv: str) -> List[Dict[str, Any]]:
    """`nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits`."""
    gpus = []
    for line in (csv or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        try:
            memory = int(float(parts[2]))
        except ValueError:
            memory = 0
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_mib": memory,
                "compute_cap": parts[3],
                "usable": memory >= MIN_GPU_MIB and parts[3] == COMPUTE_CAP,
            }
        )
    return gpus


def parse_gpu_processes(csv: str) -> List[Dict[str, Any]]:
    """`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`."""
    out = []
    for line in (csv or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        try:
            memory = int(float(parts[2]))
        except ValueError:
            memory = 0
        out.append({"pid": int(parts[0]), "name": parts[1], "memory_mib": memory})
    return out


def parse_meminfo(text: str) -> Dict[str, int]:
    """MemTotal / MemAvailable in whole GiB (truncated, as install.sh's awk does)."""
    values = {}
    for line in (text or "").splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if key in ("MemTotal", "MemAvailable") and fields and fields[0].isdigit():
            values[key] = int(fields[0]) // 1048576
    return {"total_gib": values.get("MemTotal", 0), "available_gib": values.get("MemAvailable", 0)}


def parse_cpuinfo(text: str) -> Dict[str, Any]:
    flags: List[str] = []
    model = ""
    logical = 0
    cores = set()
    physical_id = "0"
    for line in (text or "").splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "processor":
            logical += 1
        elif key == "model name" and not model:
            model = value
        elif key == "flags" and not flags:
            flags = value.split()
        elif key == "physical id":
            physical_id = value
        elif key == "core id":
            cores.add((physical_id, value))
    have = set(flags)
    missing512 = [f for f in AVX512 if f not in have]
    missing2 = [f for f in AVX2 if f not in have]
    simd = "avx512" if not missing512 else "avx2" if not missing2 else ""
    return {
        "model": model,
        "logical": logical,
        "physical_cores": len(cores) or logical,
        "simd": simd,
        "missing": missing2 if not simd else [],
    }


def probe(system: System, directory: Path) -> Dict[str, Any]:
    """The facts install.sh looks at. Never raises: a missing tool is a fact too."""
    os_name, arch = system.uname()
    out: Dict[str, Any] = {
        "os": os_name,
        "arch": arch,
        "platform_ok": os_name == "Linux" and arch == "x86_64",
        "dir": str(directory),
        "gpus": [],
        "usable_gpus": 0,
        "nvidia_smi": False,
        "driver": "",
        "driver_ok": False,
        "gpu_processes": [],
        "ram_total_gib": 0,
        "ram_available_gib": 0,
        "cpu": {"model": "", "logical": 0, "physical_cores": 0, "simd": "", "missing": []},
        "disk_free_gb": None,
    }
    free = system.disk_free(directory)
    if free is not None:
        out["disk_free_gb"] = free // 10**9
    if not out["platform_ok"]:
        return out
    csv = system.run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits"])
    if csv is not None:
        out["nvidia_smi"] = True
        out["gpus"] = parse_gpus(csv)
        out["usable_gpus"] = sum(1 for g in out["gpus"] if g["usable"])
        driver = (system.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) or "").strip()
        out["driver"] = driver.splitlines()[0].strip() if driver else ""
        major = out["driver"].split(".")[0]
        out["driver_ok"] = major.isdigit() and int(major) >= MIN_DRIVER
        out["gpu_processes"] = gpu_processes(system)
    memory = parse_meminfo(system.read("/proc/meminfo") or "")
    out["ram_total_gib"], out["ram_available_gib"] = memory["total_gib"], memory["available_gib"]
    out["cpu"] = parse_cpuinfo(system.read("/proc/cpuinfo") or "")
    return out


def gpu_processes(system: System) -> List[Dict[str, Any]]:
    csv = system.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    )
    return parse_gpu_processes(csv or "")


def check(model: Dict[str, Any], facts: Dict[str, Any], installed: bool = False) -> Dict[str, Any]:
    """install.sh --check for one model: lines of ok / warn / fail. `installed`: its files are already on disk."""
    lines: List[Dict[str, Any]] = []

    def add(level: str, code: str, text: str, **params: Any) -> None:
        lines.append({"level": level, "code": code, "text": text, "params": params})

    if facts["platform_ok"]:
        add("ok", "platform", "Linux x86_64")
    else:
        add(
            "fail",
            "platform",
            f"Linux x86_64 is required (this is {facts['os']} {facts['arch']})",
            os=facts["os"],
            arch=facts["arch"],
        )
        return {"ok": False, "lines": lines}

    need = model["need_gpus"]
    if not facts["nvidia_smi"]:
        add("fail", "no_nvidia_smi", "nvidia-smi not found: install the NVIDIA driver (R580 or newer)")
    else:
        usable = facts["usable_gpus"]
        names = [f"{g['name']} ({round(g['memory_mib'] / 1024)} GB, sm {g['compute_cap']})" for g in facts["gpus"]]
        if usable >= need:
            first = next(g["name"] for g in facts["gpus"] if g["usable"])
            add("ok", "gpu", f"{usable} sm_120 GPUs with 16 GB ({first})", usable=usable, need=need, name=first)
        else:
            add(
                "fail",
                "gpu",
                f"{need} sm_120 GPU(s) (RTX 50 series) with 16 GB are needed; found: " + (", ".join(names) or "none"),
                usable=usable,
                need=need,
                found=names,
            )
        if facts["driver_ok"]:
            add("ok", "driver", f"NVIDIA driver {facts['driver']}", version=facts["driver"])
        else:
            add(
                "fail",
                "driver",
                f"NVIDIA driver {facts['driver'] or '?'} is too old: CUDA 13.1 needs R580 or newer",
                version=facts["driver"],
            )

    if model["need_simd"]:
        cpu = facts["cpu"]
        if cpu["simd"] == "avx512":
            add("ok", "simd", "CPU has AVX-512 F/BW/VL/VNNI", kind="avx512")
        elif cpu["simd"] == "avx2":
            add("ok", "simd", "CPU has AVX2/FMA/F16C (the AVX2 expert kernels run)", kind="avx2")
        else:
            add(
                "fail",
                "simd",
                "the CPU lacks " + " ".join(cpu["missing"]) + ": Flash-Next computes the experts "
                "outside VRAM with AVX2 or AVX-512",
                kind="",
                missing=cpu["missing"],
            )

    total, available, ram = facts["ram_total_gib"], facts["ram_available_gib"], model["need_ram_gib"]
    if total < ram:
        add(
            "fail",
            "ram",
            f"about {ram} GiB of free RAM are needed, the machine has {total} GiB in total",
            need=ram,
            total=total,
            available=available,
        )
    elif available < ram:
        add(
            "warn",
            "ram",
            f"{available} GiB of RAM free now, {ram} GiB needed at startup: stop other models before serving",
            need=ram,
            total=total,
            available=available,
        )
    else:
        add("ok", "ram", f"RAM: {available} GiB free of {total} GiB", need=ram, total=total, available=available)

    free, disk = facts["disk_free_gb"], model["need_disk_gb"]
    if not installed and free is not None and free < disk:
        add(
            "fail",
            "disk",
            f"{disk} GB of disk are needed in {facts['dir']}, {free} GB free",
            need=disk,
            free=free,
            dir=facts["dir"],
        )
    else:
        add("ok", "disk", f"disk: {free} GB free in {facts['dir']}", need=disk, free=free, dir=facts["dir"])
    return {"ok": not any(l["level"] == "fail" for l in lines), "lines": lines}


def recommend(facts: Dict[str, Any], installed: Optional[List[str]] = None) -> Optional[str]:
    """The README's "Pick a model" table: the best model for the usable GPUs that passes every check."""
    installed = installed or []
    for key in catalog.PREFERENCE.get(min(facts["usable_gpus"], 2), []):
        if check(catalog.model(key), facts, key in installed)["ok"]:
            return key
    return None


def gpus_for(model: Dict[str, Any], facts: Dict[str, Any]) -> int:
    """How many GPUs the model runs on here by default: the most the launcher supports and the machine has."""
    return max([n for n in model["gpus"] if n <= max(facts["usable_gpus"], 1)] or [model["need_gpus"]])


def busy_gpus(processes: List[Dict[str, Any]], ignore_pids: Tuple[int, ...] = ()) -> List[Dict[str, Any]]:
    """Processes that hold GPU memory: another server (llama-swap, vramble, a second engine) owns the GPUs."""
    return [p for p in processes if p["memory_mib"] >= BUSY_MIB and p["pid"] not in ignore_pids]


class Machine:
    """A cached probe (nvidia-smi takes a moment; the page polls)."""

    def __init__(self, system: Optional[System] = None, directory: Optional[Path] = None, ttl: float = 5.0):
        self.system = system or System()
        self.directory = directory or default_dir()
        self.ttl = ttl
        self._at = 0.0
        self._facts: Optional[Dict[str, Any]] = None

    def facts(self, fresh: bool = False) -> Dict[str, Any]:
        if fresh or self._facts is None or time.monotonic() - self._at > self.ttl:
            self._facts = probe(self.system, self.directory)
            self._at = time.monotonic()
        return self._facts


def default_dir() -> Path:
    """install.sh's default: ~/capy-flash."""
    return Path(os.path.expanduser("~")) / "capy-flash"
