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


def detect_performance_cores() -> int:
    """Return the number of performance (P) cores, or 0 on failure.

    On Apple Silicon, ``hw.perflevel0`` is the P-cluster and
    ``hw.perflevel1`` is the E-cluster. We want P-cores for CPU-bound
    torch work (matmul on CPU-fallback ops, tokenizer, image IO, etc.).
    """
    raw = _sysctl("hw.perflevel0.physicalcpu")
    return int(raw) if raw.isdigit() else 0


_onnx_providers_logged = False


def onnxruntime_providers(log=print) -> list[str] | None:
    """Preferred onnxruntime execution provider order for the small ONNX
    side models FoocusRX runs directly (wd14 tagger, rembg background
    removal): CoreML first so Apple's ANE/GPU handles them, CPU as the
    fallback onnxruntime always ships.

    This does NOT put the SDXL UNet on the Neural Engine — PyTorch/MPS
    has no ANE backend, and getting the main diffusion model onto the ANE
    would mean a full CoreML conversion (Apple's ml-stable-diffusion
    approach), which is a separate pipeline incompatible with Fooocus's
    dynamic LoRA/ControlNet stack and arbitrary user-supplied checkpoints.
    This only covers the auxiliary models that are already plain ONNX.

    Returns None (onnxruntime's own default ordering) on non-Mac or when
    disabled via FOOOCUS_DISABLE_COREML=1, so callers can pass the result
    straight through to ``providers=``.
    """
    global _onnx_providers_logged
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    if os.environ.get("FOOOCUS_DISABLE_COREML", "").strip() == "1":
        return None
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:
        return None
    if "CoreMLExecutionProvider" not in available:
        return None
    providers = ["CoreMLExecutionProvider"]
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not _onnx_providers_logged:
        log(f"[apple_silicon] onnxruntime side models (tagger/background "
            f"removal) will prefer the Neural Engine via CoreML: {providers}")
        _onnx_providers_logged = True
    return providers


def _try_activate_mtlflashattn() -> str:
    """Try to install the mtlflashattn SDPA shim. Returns a status string.

    Never raises. Returns one of:
      * 'not-installed' - package not importable, nothing to do.
      * 'off'           - user disabled via MTLFLASHATTN_SHIM=off.
      * 'activated'     - shim installed successfully.
      * 'error:<msg>'   - import worked but install() blew up.
    """
    if os.environ.get("MTLFLASHATTN_SHIM", "").strip().lower() == "off":
        return "off"
    try:
        # The `mtlflashattn` PyPI distribution ships no top-level
        # `mtlflashattn` module — only `metal_flash_attn` (the actual
        # kernel) and a private `.pth`-activated finder. Checking for
        # `mtlflashattn` itself always raises ModuleNotFoundError even
        # when the shim is correctly installed and active.
        import metal_flash_attn  # noqa: F401
    except Exception:
        return "not-installed"
    # The package normally auto-activates via a .pth entry, but we still
    # try the explicit install path for robustness (some users pip-install
    # into virtualenvs where .pth doesn't fire during dev-mode imports).
    try:
        from metal_flash_attn import sdpa as _mfa_sdpa  # type: ignore

        if hasattr(_mfa_sdpa, "install"):
            _mfa_sdpa.install()
        return "activated"
    except Exception as exc:
        # Auto-activation via .pth may already have kicked in even if the
        # explicit path failed. Treat this as best-effort.
        return f"activated-implicit ({exc.__class__.__name__})"


def _marker_dir() -> str:
    """Location for install-attempt markers. Chosen to live next to the
    package so it survives across launches and is trivially inspectable.
    Falls back to the repo root if we're running from a source tree.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(module_dir)
    return repo_root


_MTLFLASHATTN_MARKER = ".mtlflashattn_install_attempted"


def _current_fooocus_version() -> str:
    """Best-effort read of the FoocusRX version string. Empty on failure."""
    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(module_dir)
        ns: dict = {}
        with open(os.path.join(repo_root, "fooocus_version.py")) as fh:
            exec(fh.read(), ns)  # noqa: S102 - trusted local file
        return str(ns.get("version", ""))
    except Exception:
        return ""


def _marker_is_stale(marker_path: str) -> bool:
    """Return True if the marker was written by a different FoocusRX version
    than the one currently running. This lets a version bump automatically
    retry the mtlflashattn install once — critical because new chip generations
    (e.g. M5) may lack wheels at first ship but gain them within days, and we
    don't want users stuck on 'previously-tried' forever.
    """
    current = _current_fooocus_version()
    if not current:
        return False
    try:
        with open(marker_path) as fh:
            content = fh.read()
    except OSError:
        return False
    # Marker format is 'attempted <version>\n' as of 2.6.4; older markers
    # (just 'attempted\n') are treated as stale so they get one retry on
    # first upgrade.
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("attempted"):
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1] == current:
                return False
            return True
    return True


def auto_install_mtlflashattn(*, log=print) -> str:
    """Attempt a one-shot install of mtlflashattn on the first launch.

    Contract
    --------
    Returns a status string. Never raises. Idempotent — subsequent calls
    check a marker file and short-circuit instead of re-invoking pip.

    Guards (short-circuit paths, all return without calling pip):

      * ``'not-apple-silicon'`` - not running on Darwin/arm64.
      * ``'gen-too-low'``       - detected chip is < M4. On M1-M3 the shim
                                  offers only a marginal speedup and
                                  the correctness fix past 4k tokens
                                  isn't hit by SDXL's typical shapes
                                  there; skip silently. M4+ still
                                  benefits from the fp32-accumulate
                                  correctness fix even without the
                                  M5 Neural Accelerator speedup.
      * ``'already-installed'`` - ``mtlflashattn`` is importable.
      * ``'opt-out'``           - user set
                                  ``FOOOCUS_AUTO_INSTALL_MTLFLASHATTN=0``.
      * ``'previously-tried'``  - marker file exists; we've already made
                                  one install attempt this venv. User can
                                  ``rm .mtlflashattn_install_attempted`` to
                                  retry.
      * ``'no-torch'``          - torch is not importable yet
                                  (mtlflashattn's build depends on
                                  torch.mps.compile_shader; retry once
                                  torch is installed).
      * ``'torch-too-old'``     - torch < 2.5 (missing compile_shader).

    Install path (calls pip):

      * ``'installed'``   - pip returned 0 and the package now imports.
      * ``'install-import-failed:<msg>'`` - pip returned 0 but import
                                            still fails; unusual.
      * ``'install-failed:<code>``` - pip returned non-zero; message goes
                                      to the log.
    """
    if os.environ.get("FOOOCUS_AUTO_INSTALL_MTLFLASHATTN", "").strip() == "0":
        return "opt-out"

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return "not-apple-silicon"

    # Cheap import check first — avoids ever writing the marker if
    # someone already pip-installed mtlflashattn out-of-band. See the
    # comment in _try_activate_mtlflashattn: the importable module is
    # `metal_flash_attn`, not `mtlflashattn`.
    try:
        import metal_flash_attn  # noqa: F401
        return "already-installed"
    except Exception:
        pass

    gen = detect_chip_generation(detect_chip_brand())
    if gen and gen < 4:
        return "gen-too-low"

    marker = os.path.join(_marker_dir(), _MTLFLASHATTN_MARKER)
    if os.path.exists(marker):
        if _marker_is_stale(marker):
            # New FoocusRX version since the last attempt — retry once. This
            # covers the common case where a user upgrades and a wheel is
            # now available on PyPI for their chip.
            try:
                os.remove(marker)
            except OSError:
                return "previously-tried"
        else:
            return "previously-tried"

    # Torch must be present with mps.compile_shader before we can install
    # mtlflashattn — its build imports torch and compiles a Metal shader.
    try:
        import torch  # noqa: F401
    except Exception:
        return "no-torch"

    try:
        torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except Exception:
        torch_version = (0, 0)
    if torch_version < (2, 5):
        return f"torch-too-old:{'.'.join(map(str, torch_version))}"

    # Write the marker BEFORE invoking pip so a hard crash mid-install
    # doesn't put us in an infinite retry loop next launch. If pip
    # succeeds, the marker just signifies 'we've made an attempt'.
    try:
        with open(marker, "w") as fh:
            fh.write(f"attempted {_current_fooocus_version()}\n")
    except OSError:
        # Read-only filesystem or permissions issue — don't loop, but
        # don't try to install either.
        return "marker-write-failed"

    log("[apple_silicon] auto-installing mtlflashattn (one-time)...")
    import subprocess as _sp
    import sys as _sys
    try:
        completed = _sp.run(
            [_sys.executable, "-m", "pip", "install", "--quiet", "mtlflashattn"],
            check=False,
        )
    except Exception as exc:
        log(f"[apple_silicon] mtlflashattn install raised: {exc!r}")
        return f"install-failed:exception:{exc.__class__.__name__}"

    if completed.returncode != 0:
        log(
            "[apple_silicon] mtlflashattn install failed (pip exit "
            f"{completed.returncode}). Continuing without it. To retry "
            "later: `rm .mtlflashattn_install_attempted` and relaunch."
        )
        return f"install-failed:{completed.returncode}"

    # Confirm the package is importable in the same process. If pip
    # succeeded but the shim is still not importable (e.g. a namespace
    # collision), report that explicitly so the log is actionable.
    try:
        # Reload sys.path caches so a freshly-installed package is found.
        import importlib
        import site
        importlib.reload(site)
        import metal_flash_attn  # noqa: F401
    except Exception as exc:
        return f"install-import-failed:{exc.__class__.__name__}"

    log("[apple_silicon] mtlflashattn installed.")
    return "installed"


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

    # PYTORCH_MPS_LOW_WATERMARK_RATIO must also be set explicitly. On
    # recent torch (2.5+), when the user or FoocusRX sets *only* the high
    # ratio, torch derives the low ratio via a multiplier that on shared-
    # memory Apple Silicon can compute a value > 1.0 (observed: 1.4 with
    # high=0.85). Torch then rejects it at first `.to('mps')` with
    # `RuntimeError: invalid low watermark ratio 1.4`. Pin it below high.
    # The low ratio controls when MPS starts *proactively* releasing
    # cached blocks; a value slightly below high gives the allocator a
    # small headroom band to work in.
    if "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in os.environ:
        high_str = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
        try:
            high = float(high_str)
        except ValueError:
            high = 0.85
        # high=0.0 is the "disable cap" sentinel; pair it with a low that
        # is also disabled (0.0) so torch doesn't try to derive one.
        if high <= 0.0:
            low = 0.0
        else:
            low = max(0.0, min(high - 0.05, high * 0.9))
        low_str = f"{low:.2f}"
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = low_str
        applied["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = low_str

    # MPS CPU fallback for ops the backend doesn't implement yet.
    # Upstream sets this too; we set it via setdefault to avoid clobbering
    # a user override.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    applied.setdefault(
        "PYTORCH_ENABLE_MPS_FALLBACK",
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
    )

    # torch>=2.5 on Darwin: hint the backend to prefer the Metal
    # implementation of ops that have both a Metal and a MPSGraph path.
    # Harmless on older torch (env var is simply ignored).
    os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")
    applied["PYTORCH_MPS_PREFER_METAL"] = os.environ["PYTORCH_MPS_PREFER_METAL"]

    # Cap CPU thread count to the number of performance cores. macOS
    # will happily oversubscribe E-cores with BLAS threads, which
    # actively hurts throughput on the P-core-heavy Pro/Max chips.
    p_cores = detect_performance_cores()
    info["perf_cores"] = p_cores
    if p_cores > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(p_cores))
        os.environ.setdefault("MKL_NUM_THREADS", str(p_cores))
        applied["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
        applied["MKL_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"]

    # Activate mtlflashattn on M4+. M5+ hits the fastest tier via the
    # Neural Accelerators; M4 still gets the correctness fix (fp32
    # softmax/output accumulation vs. stock MPS SDPA's fp16 which is
    # silently wrong past ~4k tokens). Safe on any Mac — returns
    # 'not-installed' if the package isn't present.
    if gen >= 4:
        status = _try_activate_mtlflashattn()
        info["mtlflashattn"] = status
        applied["mtlflashattn"] = status

    if verbose:
        _log(info)
    return info


def _log(info: dict) -> None:
    brand = info["brand"] or "unknown"
    gen = info["generation"]
    mem = info["memory_gb"]
    p_cores = info.get("perf_cores", 0)
    gen_desc = f"M{gen}" if gen else "unknown generation"
    core_desc = f", {p_cores} P-cores" if p_cores else ""
    print(f"[apple_silicon] detected: {brand} ({gen_desc}, {mem} GB unified memory{core_desc})")
    for k, v in info["applied"].items():
        print(f"[apple_silicon]   {k}={v}")
    if gen >= 4 and info.get("mtlflashattn") == "not-installed":
        # After PR #16 the launcher auto-installs mtlflashattn on M4+,
        # so hitting this branch means either the auto-install was
        # opted out, previously failed, or torch was too old at that
        # time. Point the user at the marker file so a retry is
        # obvious.
        print(
            "[apple_silicon] mtlflashattn not active. Delete "
            ".mtlflashattn_install_attempted and relaunch to retry "
            "the auto-install, or `pip install mtlflashattn` manually."
        )
    if gen >= 5 and not os.environ.get("FOOOCUS_TORCH_COMPILE"):
        print(
            "[apple_silicon] M5+ detected: set FOOOCUS_TORCH_COMPILE=1 for "
            "an additional 10-30% throughput (30-120s one-time warmup)."
        )


if __name__ == "__main__":
    # Handy for debugging: `python -m modules.apple_silicon`
    import json
    print(json.dumps(configure(verbose=False), indent=2))
