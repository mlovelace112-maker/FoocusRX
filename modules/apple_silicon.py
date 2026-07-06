"""Apple Silicon detection and MPS tuning for FoocusRX.

Runs as a lightweight module import. Sets a handful of environment
variables that tune the MPS backend for the detected chip generation:

  * M1 / M2 / M3   -> no BF16 hardware; leave dtype hints alone.
  * M4 / M5+       -> BF16 has native hardware support; hint bf16 to
                      downstream code that reads FOOOCUS_PREFERRED_DTYPE.
  * M5+            -> per-GPU-core Neural Accelerators (matmul hardware).
                      Set FOOOCUS_HAS_NEURAL_ACCELERATORS=1 so upstream
                      code can decide whether to enable mtlflashattn.

Also picks a saner default for PYTORCH_MPS_HIGH_WATERMARK_RATIO than
upstream's blanket ``0.0`` (which disables the memory cap entirely):

  * >= 32 GB unified memory: 0.0 (upstream behavior; we have headroom).
  * 16-31 GB:                0.85 (leave the OS ~15%).
  * < 16 GB:                 0.7  (leave the OS ~30%).

Nothing here fails hard. If detection doesn't work, the module returns
without setting anything and Fooocus behaves as before.
"""

from __future__ import annotations

import os
import platform
import subprocess


def _sysctl(key: str) -> str:
    try:
        out = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", key],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        return out.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


def detect_chip_brand() -> str:
    """Return the ``machdep.cpu.brand_string`` sysctl value, e.g.
    ``'Apple M5 Pro'``. Empty string on non-Mac or on failure.
    """
    if platform.system() != "Darwin":
        return ""
    return _sysctl("machdep.cpu.brand_string")


def detect_chip_generation(brand: str) -> int:
    """Extract the M-series generation number from a brand string.

    ``'Apple M5 Pro'`` -> 5, ``'Apple M1 Max'`` -> 1, unknown -> 0.
    """
    if not brand:
        return 0
    # Look for 'M' followed by digits (M1, M2, ... M99).
    import re
    m = re.search(r"\bM(\d+)\b", brand)
    return int(m.group(1)) if m else 0


def detect_memory_gb() -> int:
    """Return total physical memory in GiB, or 0 on failure."""
    raw = _sysctl("hw.memsize")
    if not raw.isdigit():
        return 0
    return int(raw) // (1024 ** 3)


def configure(verbose: bool = True) -> dict:
    """Detect the chip and set MPS-tuning env vars. Idempotent.

    Returns a dict describing what was detected and set, for logging.
    """
    info: dict = {
        "system": platform.system(),
        "arch": platform.machine(),
        "brand": "",
        "generation": 0,
        "memory_gb": 0,
        "applied": {},
    }

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return info  # Not Apple Silicon; nothing to do.

    brand = detect_chip_brand()
    gen = detect_chip_generation(brand)
    mem_gb = detect_memory_gb()
    info.update(brand=brand, generation=gen, memory_gb=mem_gb)

    applied = info["applied"]

    # BF16 native hardware landed with M4. Hint downstream code so it can
    # pick bf16 over fp16 when the chip supports it in hardware.
    # (FOOOCUS_PREFERRED_DTYPE is FoocusRX-internal; a follow-up PR wires
    # it into modules that currently hard-code torch.float16.)
    if gen >= 4:
        os.environ.setdefault("FOOOCUS_PREFERRED_DTYPE", "bfloat16")
        applied["FOOOCUS_PREFERRED_DTYPE"] = os.environ["FOOOCUS_PREFERRED_DTYPE"]

    # M5+ has per-GPU-core Neural Accelerators (matmul hardware).
    # Kernels like mtlflashattn.v2r target this. Expose a flag so
    # follow-up code can gate the drop-in shim on it.
    if gen >= 5:
        os.environ.setdefault("FOOOCUS_HAS_NEURAL_ACCELERATORS", "1")
        applied["FOOOCUS_HAS_NEURAL_ACCELERATORS"] = "1"

    # Sane memory watermark. Upstream sets 0.0 unconditionally which
    # removes MPS's OOM guard entirely — fine on 64+ GB, dangerous on
    # 16 GB where SDXL can push the whole machine into swap and lock up
    # the UI. Only downgrade if the user hasn't set it explicitly.
    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        if mem_gb >= 32:
            ratio = "0.0"
        elif mem_gb >= 16:
            ratio = "0.85"
        else:
            ratio = "0.7"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = ratio
        applied["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = ratio

    # MPS CPU fallback for ops the backend doesn't implement yet.
    # Upstream sets this too; we set it via setdefault to avoid clobbering
    # a user override.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    applied.setdefault(
        "PYTORCH_ENABLE_MPS_FALLBACK",
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
    )

    if verbose:
        _log(info)
    return info


def _log(info: dict) -> None:
    brand = info["brand"] or "unknown"
    gen = info["generation"]
    mem = info["memory_gb"]
    gen_desc = f"M{gen}" if gen else "unknown generation"
    print(f"[apple_silicon] detected: {brand} ({gen_desc}, {mem} GB unified memory)")
    for k, v in info["applied"].items():
        print(f"[apple_silicon]   {k}={v}")
    if gen >= 5:
        print(
            "[apple_silicon] M5+ detected: for a 3-11x attention speedup, "
            "install mtlflashattn (`pip install mtlflashattn`)."
        )


if __name__ == "__main__":
    # Handy for debugging: `python -m modules.apple_silicon`
    import json
    print(json.dumps(configure(verbose=False), indent=2))
