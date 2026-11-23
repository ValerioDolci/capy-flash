"""The capy-flash releases and models the app can install: the facts of install.sh and the README, in one place.

Nothing here comes from the page or from the settings: what is downloaded, from where, and which launcher runs it is
fixed in this file.
"""

import re
from typing import Any, Dict, List, Optional

GITHUB_REPO = "ValerioDolci/capy-flash"
TARBALL = "capy-flash-{version}-linux-x86_64-sm120a-ubuntu26.04.tar.gz"
VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+(?:[.-][A-Za-z0-9.]{1,24})?$")
ENGINE_BINARY = "bin/ninfer-serve"
DEFAULT_PORT = 8080  # NINFER_PORT's default

# Requirements as install.sh checks them (need_*), sizes as the Hugging Face repositories report them.
# gpus: the GPU counts the launcher supports; flags_2gpu / flags: what install.sh prints after the artifact path.
MODELS: List[Dict[str, Any]] = [
    {
        "key": "flash-next",
        "family": "flash-next",
        "title": "Qwen3.8-Flash-Next",
        "edition": "NVFP4",
        "hf_repo": "Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash",
        "artifact": "qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer",
        "launcher": "serve-flash-next.sh",
        "model_id": "capy-flash",
        "experts": "nvfp4",
        "need_gpus": 2,
        "gpus": [2],
        "need_ram_gib": 75,
        "need_disk_gb": 135,
        "need_simd": True,
        "size_gb": 126.6,
        "context": 262144,
        "vision": False,
        "flags": [],
        "flags_2gpu": [],
    },
    {
        "key": "flash-next-64gb",
        "family": "flash-next",
        "title": "Qwen3.8-Flash-Next",
        "edition": "64 GB",
        "hf_repo": "Feyd89/Qwen3.8-Flash-Next-GSQ-RCO-capy-flash",
        "artifact": "qwen3_8_flash_next_gsqrco_mtpB.ninfer",
        "launcher": "serve-flash-next.sh",
        "model_id": "capy-flash",
        "experts": "gguf",
        "need_gpus": 1,
        "gpus": [1, 2],
        "need_ram_gib": 56,
        "need_disk_gb": 118,
        "need_simd": True,
        "size_gb": 108.9,
        "context": 262144,
        "vision": False,
        "flags": [],
        "flags_2gpu": ["--host-kv-mib", "1024", "--host-state-slots", "2"],
    },
    {
        "key": "27b",
        "family": "27b",
        "title": "Qwen3.8-27B",
        "edition": "NVFP4",
        "hf_repo": "Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer",
        "artifact": "qwen3_8_27b_quasar_nvfp4_dflash2.ninfer",
        "launcher": "serve-tp2.sh",
        "model_id": "qwen3.8-27b",
        "experts": "",
        "need_gpus": 2,
        "gpus": [2],
        "need_ram_gib": 8,
        "need_disk_gb": 22,
        "need_simd": False,
        "size_gb": 19.2,
        "context": 196608,
        "vision": True,
        "flags": [],
        "flags_2gpu": [],
    },
    {
        "key": "27b-16gb",
        "family": "27b",
        "title": "Qwen3.8-27B",
        "edition": "3-bit",
        "hf_repo": "Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash",
        "artifact": "qwen3_8_27b_gsq3.ninfer",
        "launcher": "serve-27b-16gb.sh",
        "model_id": "qwen3.8-27b",
        "experts": "",
        "need_gpus": 1,
        "gpus": [1],
        "need_ram_gib": 8,
        "need_disk_gb": 15,
        "need_simd": False,
        "size_gb": 12.5,
        "context": 65536,
        "vision": True,
        "flags": [],
        "flags_2gpu": [],
    },
    {
        # The NVFP4 Flash-Next plus the Q8 vision tower. install.sh never fetches it; the README serves it with
        # `serve-flash-next.sh <vision artifact> --vision`. Requirements: those of flash-next (same weights).
        "key": "flash-next-vision",
        "family": "flash-next",
        "title": "Qwen3.8-Flash-Next",
        "edition": "NVFP4 + Vision",
        "hf_repo": "Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash",
        "artifact": "qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer",
        "launcher": "serve-flash-next.sh",
        "model_id": "capy-flash",
        "experts": "nvfp4",
        "need_gpus": 2,
        "gpus": [2],
        "need_ram_gib": 75,
        "need_disk_gb": 136,
        "need_simd": True,
        "size_gb": 127.1,
        "context": 262144,
        "vision": True,
        "flags": ["--vision"],
        "flags_2gpu": [],
        "optional": True,
    },
    {
        # Text only, one request at a time; verified on two 16 GB GPUs (one 32 GB GPU is not verified, so the app
        # offers two). The artifact holds the text model and its DSpark drafter, which the launcher turns on (same
        # text as without it): the released weights repackaged without requantizing. Its launcher ships only
        # in the releases that serve Inkling, so a release without it still verifies (Installer._verify_release).
        "key": "inkling2-mini",
        "family": "inkling2",
        "title": "Inkling 2 Mini",
        "edition": "NVFP4",
        "hf_repo": "Feyd89/Inkling-2-Mini-NVFP4-capy-flash",
        "artifact": "capy-flash-inkling2-mini-nvfp4.ninfer",
        "launcher": "capy-serve-inkling2-mini.sh",
        "model_id": "inkling-2-mini",
        "experts": "",
        "need_gpus": 2,
        "gpus": [2],
        "need_ram_gib": 8,
        "need_disk_gb": 23,
        "need_simd": False,
        "size_gb": 19.7,
        "context": 253952,
        "vision": False,
        "flags": [],
        "flags_2gpu": [],
        "optional": True,
    },
]

# "Pick a model" in the README, as an order of preference for each number of usable GPUs.
PREFERENCE = {2: ["flash-next", "flash-next-64gb", "27b", "27b-16gb"], 1: ["flash-next-64gb", "27b-16gb"]}

# The speed switches that change the numbers (README, "Optional speed switches"). The app never sets them and
# removes them from the engine's environment; the page shows this table under "Advanced".
LOSSY_SWITCHES = [
    {
        "env": "NINFER_TP_AR_PAYLOAD",
        "value": "int8",
        "applies": "flash-next",
        "gain": "+7-8 % prefill",
        "loss": "perplexity -0.22 % (NVFP4), -0.26 % (64 GB edition)",
    },
    {
        "env": "NINFER_Q8_A8",
        "value": "128",
        "applies": "flash-next",
        "gain": "+7-8 % prefill",
        "loss": "perplexity +0.21 % (NVFP4), +0.004 % (64 GB edition)",
    },
    {
        "env": "NINFER_ATTN_PV_F16ACC",
        "value": "1",
        "applies": "27b",
        "gain": "+6 % prefill at 16k, +25 % at 128k",
        "loss": "perplexity +1.1 % at 64k; GSM8K 0.9629 vs 0.9666",
    },
]
LOSSY_ENV = tuple(s["env"] for s in LOSSY_SWITCHES)

# The settings that leave the output unchanged: the only ones the tuner tries and a profile (tuning.json) may hold,
# in the order they are tried. key: the profile's field; flag: the engine's (None: the GPU count, which goes into
# NINFER_DEVICES); range or choices: the values a profile may hold (choices None: the model's GPU counts); tries:
# what the tuner measures (None: worked out from the machine); family, experts: the models it applies to (None:
# any); metric: the number it is judged on; runs: the most engine starts its step takes.
LOSSLESS_SETTINGS = [
    {
        "id": "threads",
        "key": "expert_cpu_threads",
        "flag": "--expert-cpu-threads",
        "range": (1, 64),
        "tries": None,
        "family": "flash-next",
        "experts": None,
        "metric": "decode_tps",
        "runs": 3,
    },
    # NVFP4 experts only: the GGUF expert banks of the 64 GB edition force it off
    {
        "id": "miss_split",
        "key": "miss_split",
        "flag": "--miss-split",
        "choices": ("auto", "calibrate", "off"),
        "tries": ("auto", "calibrate"),
        "family": "flash-next",
        "experts": "nvfp4",
        "metric": "decode_tps",
        "runs": 2,
    },
    {
        "id": "stream",
        "key": "stream_min_tokens",
        "flag": "--stream-min-tokens",
        "range": (0, 65536),
        "tries": (128, 512),
        "family": "flash-next",
        "experts": None,
        "metric": "short_ms",
        "runs": 2,
    },
    {
        "id": "gpus",
        "key": "gpus",
        "flag": None,
        "choices": None,
        "tries": None,
        "family": None,
        "experts": None,
        "metric": "decode_tps",
        "runs": 1,
    },
    {
        "id": "kv_ram",
        "key": "kv_ram_active",
        "flag": "--kv-ram-active",
        "choices": (True,),
        "tries": (True,),
        "family": "flash-next",
        "experts": None,
        "metric": "long_decode_tps",
        "runs": 1,
    },
]


def applies(setting: Dict[str, Any], model: Dict[str, Any]) -> bool:
    return setting["family"] in (None, model["family"]) and setting["experts"] in (None, model["experts"])


def model(key: str) -> Optional[Dict[str, Any]]:
    for m in MODELS:
        if m["key"] == key:
            return m
    return None


def tarball_name(version: str) -> str:
    if not VERSION_RE.match(version or ""):
        raise ValueError("not a release tag: %r" % (version,))
    return TARBALL.format(version=version)
