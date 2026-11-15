"""The hardware probe and install.sh's rules, on fixture outputs of nvidia-smi and /proc."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capy_app import catalog, machine  # noqa: E402

GPUS_2X5070 = "0, NVIDIA GeForce RTX 5070 Ti, 16303, 12.0\n1, NVIDIA GeForce RTX 5070 Ti, 16303, 12.0\n"
GPUS_1X5060 = "0, NVIDIA GeForce RTX 5060 Ti, 16311, 12.0\n"
GPUS_OLD = "0, NVIDIA GeForce RTX 4090, 24564, 8.9\n1, NVIDIA GeForce RTX 5070, 12227, 12.0\n"
FLAGS_AVX512 = "fpu vme sse sse2 avx avx2 fma f16c avx512f avx512bw avx512vl avx512_vnni"
FLAGS_AVX2 = "fpu vme sse sse2 avx avx2 fma f16c"
FLAGS_OLD = "fpu vme sse sse2 avx"


def cpuinfo(flags, cores=12, threads=2, model="AMD Ryzen 9 9900X 12-Core Processor"):
    out = []
    for n in range(cores * threads):
        out.append(
            f"processor\t: {n}\nmodel name\t: {model}\nphysical id\t: 0\ncore id\t\t: {n % cores}\nflags\t\t: {flags}\n"
        )
    return "\n".join(out)


def meminfo(total_gib, available_gib):
    return (
        f"MemTotal:       {total_gib * 1048576 + 5000} kB\nMemFree:         1000000 kB\n"
        f"MemAvailable:   {available_gib * 1048576 + 5000} kB\nCmaTotal:              0 kB\n"
    )


class FakeSystem(machine.System):
    def __init__(
        self,
        os_name="Linux",
        arch="x86_64",
        gpus=GPUS_2X5070,
        driver="580.95.05\n580.95.05\n",
        flags=FLAGS_AVX512,
        ram=(94, 86),
        free_gb=410,
        processes="",
        cores=12,
    ):
        self.os_name, self.arch, self.gpus, self.driver = os_name, arch, gpus, driver
        self.files = {"/proc/meminfo": meminfo(*ram), "/proc/cpuinfo": cpuinfo(flags, cores)}
        self.free_gb, self.processes = free_gb, processes
        self.commands = []

    def uname(self):
        return self.os_name, self.arch

    def run(self, argv, timeout=10.0):
        self.commands.append(argv)
        if argv[0] == "nvidia-smi":
            if self.gpus is None:
                return None
            if "--query-compute-apps=pid,process_name,used_memory" in argv:
                return self.processes
            if "--query-gpu=driver_version" in argv:
                return self.driver
            return self.gpus
        return None

    def read(self, path):
        return self.files.get(path)

    def disk_free(self, path):
        return self.free_gb * 10**9


def facts(**kw):
    return machine.probe(FakeSystem(**kw), Path("/home/u/capy-flash"))


def levels(result):
    return {l["code"]: l["level"] for l in result["lines"]}


class TestParsing(unittest.TestCase):
    def test_gpus(self):
        gpus = machine.parse_gpus(GPUS_OLD + "garbage line\n")
        self.assertEqual(
            [(g["index"], g["memory_mib"], g["usable"]) for g in gpus], [(0, 24564, False), (1, 12227, False)]
        )
        self.assertTrue(all(g["usable"] for g in machine.parse_gpus(GPUS_2X5070)))
        self.assertEqual(machine.parse_gpus(""), [])

    def test_meminfo_truncates_like_install_sh(self):
        self.assertEqual(
            machine.parse_meminfo("MemTotal:       98765432 kB\nMemAvailable:   73400319 kB\n"),
            {"total_gib": 94, "available_gib": 69},
        )  # 69.99 GiB is 69, not 70

    def test_cpuinfo(self):
        cpu = machine.parse_cpuinfo(cpuinfo(FLAGS_AVX512))
        self.assertEqual((cpu["physical_cores"], cpu["logical"], cpu["simd"]), (12, 24, "avx512"))
        self.assertEqual(machine.parse_cpuinfo(cpuinfo(FLAGS_AVX2, 16, 1))["simd"], "avx2")
        old = machine.parse_cpuinfo(cpuinfo(FLAGS_OLD))
        self.assertEqual((old["simd"], old["missing"]), ("", ["avx2", "fma", "f16c"]))

    def test_gpu_processes(self):
        text = "4242, /opt/llama-swap/llama-server, 14210\n911, python3, 312\n"
        self.assertEqual(
            machine.parse_gpu_processes(text)[0],
            {"pid": 4242, "name": "/opt/llama-swap/llama-server", "memory_mib": 14210},
        )
        busy = machine.busy_gpus(machine.parse_gpu_processes(text))
        self.assertEqual([p["pid"] for p in busy], [4242])
        self.assertEqual(machine.busy_gpus(machine.parse_gpu_processes(text), ignore_pids=(4242,)), [])


class TestRules(unittest.TestCase):
    def test_workstation_runs_everything(self):
        f = facts()
        self.assertEqual((f["usable_gpus"], f["driver_ok"], f["ram_total_gib"], f["disk_free_gb"]), (2, True, 94, 410))
        for m in catalog.MODELS:
            self.assertTrue(machine.check(m, f)["ok"], m["key"])
        self.assertEqual(machine.recommend(f), "flash-next")

    def test_mac_cannot_run_the_engine(self):
        system = FakeSystem(os_name="Darwin", arch="arm64")
        f = machine.probe(system, Path("/Users/u/capy-flash"))
        self.assertFalse(f["platform_ok"])
        self.assertEqual(system.commands, [], "no nvidia-smi on a Mac")
        result = machine.check(catalog.model("27b-16gb"), f)
        self.assertFalse(result["ok"])
        self.assertEqual(levels(result), {"platform": "fail"})
        self.assertIsNone(machine.recommend(f))

    def test_pick_a_model_table(self):
        # two GPUs, 64 GB machine: the 64 GB edition; two GPUs, little RAM: the NVFP4 27B
        self.assertEqual(machine.recommend(facts(ram=(62, 58))), "flash-next-64gb")
        self.assertEqual(machine.recommend(facts(ram=(31, 26))), "27b")
        # one GPU: the 64 GB edition with enough RAM, else the 3-bit 27B
        self.assertEqual(machine.recommend(facts(gpus=GPUS_1X5060, ram=(62, 58))), "flash-next-64gb")
        self.assertEqual(machine.recommend(facts(gpus=GPUS_1X5060, ram=(31, 26))), "27b-16gb")
        # Flash-Next needs AVX2 or AVX-512; the 27B does not
        self.assertEqual(machine.recommend(facts(flags=FLAGS_AVX2)), "flash-next")
        self.assertEqual(machine.recommend(facts(flags=FLAGS_OLD)), "27b")
        # not enough disk for Flash-Next: the next row that fits
        self.assertEqual(machine.recommend(facts(free_gb=60)), "27b")
        self.assertEqual(machine.recommend(facts(free_gb=60), installed=["flash-next"]), "flash-next")

    def test_fail_lines_name_the_reason(self):
        f = facts(gpus=GPUS_OLD)
        result = machine.check(catalog.model("27b-16gb"), f)
        self.assertEqual(levels(result)["gpu"], "fail")
        self.assertIn("RTX 4090", [l for l in result["lines"] if l["code"] == "gpu"][0]["text"])
        self.assertIsNone(machine.recommend(f))
        self.assertEqual(levels(machine.check(catalog.model("27b"), facts(driver="570.86.16\n")))["driver"], "fail")
        self.assertEqual(levels(machine.check(catalog.model("27b"), facts(gpus=None)))["no_nvidia_smi"], "fail")
        self.assertEqual(levels(machine.check(catalog.model("flash-next"), facts(ram=(62, 58))))["ram"], "fail")
        # enough RAM in total but busy now: a warning, as in install.sh
        warn = machine.check(catalog.model("flash-next"), facts(ram=(94, 40)))
        self.assertTrue(warn["ok"])
        self.assertEqual(levels(warn)["ram"], "warn")
        self.assertEqual(levels(machine.check(catalog.model("flash-next"), facts(free_gb=100)))["disk"], "fail")
        self.assertEqual(
            levels(machine.check(catalog.model("flash-next"), facts(free_gb=100), installed=True))["disk"], "ok"
        )

    def test_gpus_for(self):
        f2, f1 = facts(), facts(gpus=GPUS_1X5060)
        self.assertEqual(machine.gpus_for(catalog.model("flash-next-64gb"), f2), 2)
        self.assertEqual(machine.gpus_for(catalog.model("flash-next-64gb"), f1), 1)
        self.assertEqual(machine.gpus_for(catalog.model("27b-16gb"), f2), 1)

    def test_catalog_matches_install_sh(self):
        want = {
            "flash-next": (75, 135, True, 2),
            "flash-next-64gb": (56, 118, True, 1),
            "27b": (8, 20, False, 2),
            "27b-16gb": (8, 15, False, 1),
        }
        for key, row in want.items():
            m = catalog.model(key)
            self.assertEqual((m["need_ram_gib"], m["need_disk_gb"], m["need_simd"], m["need_gpus"]), row, key)
        self.assertEqual(catalog.tarball_name("v0.12.0"), "capy-flash-v0.12.0-linux-x86_64-sm120a-ubuntu26.04.tar.gz")
        for bad in ("", "latest", "v1", "v0.1.0/../x", "v0.1.0 ; rm"):
            with self.assertRaises(ValueError):
                catalog.tarball_name(bad)


if __name__ == "__main__":
    unittest.main()
