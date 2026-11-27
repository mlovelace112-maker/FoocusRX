"""Auto upscale: enlarge every finished generation right after it is made.

This module is the plain-Python half (no torch): settings, which upscaler
models exist, how big each stage gets, and what the upscaled file is called.
The worker (modules/async_worker.py, run_auto_upscale) does the GPU work.

Two methods:

* Detailed (default): an upscaler model enlarges the image, then a light
  diffusion pass with the same model, prompt and seed redraws fine detail,
  the way Input Image > "Upscale (2x)" does (SDXL: denoise 0.382; FLUX.1:
  tiled refine). The diffusion pass runs at most at DETAIL_MAX_SHAPE (about
  2x of a normal SDXL image); anything above that is reached with the
  upscaler model alone, so 4x = model 2x -> detail pass -> model 2x. That
  keeps memory and time sane on Apple Silicon; a full 4x diffusion pass
  would need ~16x the compute of a normal image.
* Fast: the upscaler model only, no diffusion. Seconds, not minutes.

Krea 2 / MOSS (sd.cpp) images always use Fast: the detail pass needs an
image-to-image mode that FoocusRX doesn't drive for sd.cpp yet.

The upscaled copy is saved next to the original as ``<name>_up4x.<ext>``
(``_up1.5x``, ``_up2x`` ...), with the original's metadata plus the upscale
settings, and is added to the gallery and to History.
"""
import math
import os
import re
from dataclasses import dataclass

METHOD_DETAILED = 'Detailed (slower, adds detail)'
METHOD_FAST = 'Fast (upscaler model only)'
METHODS = (METHOD_DETAILED, METHOD_FAST)

SCALES = (1.5, 2.0, 3.0, 4.0)
SCALE_LABELS = ('1.5x', '2x', '3x', '4x')

BUILT_IN_UPSCALER = 'Built-in (Fooocus ESRGAN)'
BUILT_IN_FILE = 'fooocus_upscaler_s409985e5.bin'
UPSCALER_EXTENSIONS = ('.pth', '.pt', '.safetensors')

# Fooocus skips its diffusion upscale above this shape ceil (sqrt of the
# pixel count, rounded up to 64): an SDXL pass at 4K is slow and no better.
DETAIL_MAX_SHAPE = 2800
# ...and never runs it above 2x: that's what Upscale (2x) does, and each step
# past it costs a lot more time (the pass's cost grows faster than its area).
DETAIL_MAX_FACTOR = 2.0
# A detail pass that would enlarge less than this is skipped (image is
# already near the cap); the model upscale still runs.
MIN_DETAIL_GAIN = 1.25
# Largest side of any output, so a 4x of a big image can't exhaust memory.
MAX_OUTPUT_SIDE = 8192

SDXL_DETAIL_DENOISE = 0.382  # async_worker.apply_upscale

FAMILY_SDXL = 'sdxl'
FAMILY_FLUX = 'flux'
FAMILY_SDCPP = 'sdcpp'
DETAIL_FAMILIES = (FAMILY_SDXL, FAMILY_FLUX)

_SUFFIX_RE = re.compile(r'_up\d+(?:\.\d+)?x$')


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    scale: float = 4.0
    method: str = METHOD_DETAILED
    upscaler: str = BUILT_IN_UPSCALER
    keep_original: bool = True


def parse_scale(value, default: float = 4.0) -> float:
    """'4x', '1.5x', 4, '4' -> float; anything unusable -> default."""
    try:
        scale = float(str(value).strip().lower().rstrip('x'))
    except (TypeError, ValueError):
        return default
    return scale if 1.0 < scale <= 8.0 else default


def scale_label(scale: float) -> str:
    return f'{scale:g}x'


def make_settings(enabled=True, scale=4.0, method=METHOD_DETAILED, upscaler=BUILT_IN_UPSCALER,
                  keep_original=True) -> Settings:
    """Settings from UI values, falling back to the defaults for junk."""
    return Settings(
        enabled=bool(enabled),
        scale=parse_scale(scale),
        method=method if method in METHODS else METHOD_DETAILED,
        upscaler=str(upscaler) if upscaler else BUILT_IN_UPSCALER,
        keep_original=bool(keep_original),
    )


# --- Upscaler models -----------------------------------------------------------

def list_upscalers(folders) -> list:
    """Built-in first, then every model file found in ``folders`` (a path or
    a list of paths; subfolders included), shown by its path relative to its
    folder. The built-in Fooocus file itself is not listed twice."""
    if isinstance(folders, str):
        folders = [folders]
    found = []
    for folder in folders or []:
        if not folder or not os.path.isdir(folder):
            continue
        for root, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            for name in sorted(files):
                if name.startswith('.') or not name.lower().endswith(UPSCALER_EXTENSIONS):
                    continue
                if name == BUILT_IN_FILE:
                    continue
                rel = os.path.relpath(os.path.join(root, name), folder).replace(os.sep, '/')
                if rel not in found:
                    found.append(rel)
    return [BUILT_IN_UPSCALER] + sorted(found, key=str.casefold)


def resolve_upscaler(name: str, folders) -> str | None:
    """Full path of a listed upscaler, or None for the built-in (and for a
    name that isn't in ``folders`` any more, so a stale pick falls back to
    the built-in instead of failing the run). Refuses names that climb out
    of the folder."""
    if not name or name == BUILT_IN_UPSCALER:
        return None
    if isinstance(folders, str):
        folders = [folders]
    for folder in folders or []:
        if not folder:
            continue
        root = os.path.realpath(folder)
        path = os.path.realpath(os.path.join(folder, name))
        if os.path.commonpath([root, path]) == root and os.path.isfile(path):
            return path
    return None


# --- Sizes ---------------------------------------------------------------------

def shape_ceil(width: float, height: float) -> int:
    """modules.util.get_shape_ceil without the numpy import."""
    return int(math.ceil(math.sqrt(width * height) / 64.0) * 64)


def target_size(width: int, height: int, scale: float, max_side: int = MAX_OUTPUT_SIDE) -> tuple:
    """Final (width, height) for ``scale``, capped so the long side is at
    most ``max_side``; even numbers."""
    factor = min(float(scale), max_side / float(max(width, height)))
    factor = max(factor, 1.0)
    return 2 * round(width * factor / 2), 2 * round(height * factor / 2)


@dataclass(frozen=True)
class Plan:
    final: tuple                 # (width, height) of the saved upscale
    detail: tuple | None         # size the diffusion pass runs at, or None
    note: str = ''               # why there is no detail pass, if relevant


def plan(width: int, height: int, settings: Settings, family: str) -> Plan:
    """How to get from (width, height) to the upscale: an optional detail
    pass at ``detail`` size (reached with the upscaler model), then the
    upscaler model again up to ``final``."""
    final = target_size(width, height, settings.scale)
    if settings.method != METHOD_DETAILED:
        return Plan(final, None)
    if family not in DETAIL_FAMILIES:
        return Plan(final, None, 'detail pass not available for this model family yet; used the upscaler model only')
    # Largest factor (<= the asked scale, <= 2x) whose shape ceil stays under the cap.
    limit = (DETAIL_MAX_SHAPE - 64) / math.sqrt(width * height)
    factor = min(settings.scale, DETAIL_MAX_FACTOR, limit, final[0] / float(width))
    if factor < MIN_DETAIL_GAIN:
        return Plan(final, None, 'image already near the detail-pass size limit; used the upscaler model only')
    # Multiples of 16 fit SDXL's 8-px latent grid and FLUX's 16-px one.
    detail = (max(64, 16 * int(width * factor // 16)), max(64, 16 * int(height * factor // 16)))
    detail = (min(detail[0], final[0]), min(detail[1], final[1]))
    return Plan(final, detail)


def model_passes(current: int, target: int, model_scale: int, max_passes: int = 3) -> int:
    """How many upscaler-model passes take the long side from ``current`` to
    at least ``target`` (the result is then resized down to the exact size)."""
    passes = 0
    model_scale = max(int(model_scale or 1), 1)
    if model_scale == 1:
        return 1 if target > current else 0
    while current < target and passes < max_passes:
        current *= model_scale
        passes += 1
    return passes


# --- Files ---------------------------------------------------------------------

def upscaled_path(original: str, scale: float) -> str:
    """``/o/2026-11-27/a.png`` + 4 -> ``/o/2026-11-27/a_up4x.png``. Adds
    `` (2)`` etc. if that name is taken by something else."""
    stem, ext = os.path.splitext(original)
    candidate = f'{stem}_up{scale_label(scale)}{ext}'
    n = 2
    while os.path.exists(candidate):
        candidate = f'{stem}_up{scale_label(scale)} ({n}){ext}'
        n += 1
    return candidate


def is_upscale(path: str) -> bool:
    """True for files this pipeline wrote (``*_up4x.png``, ``*_up2x (2).png``)."""
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    stem = re.sub(r' \(\d+\)$', '', stem)
    return bool(_SUFFIX_RE.search(stem))


def should_upscale(path, temp_dir: str | None = None) -> bool:
    """Only saved image files, never an upscale itself, never a throwaway
    intermediate in the temp folder (Enhance with "save final image only")."""
    if not isinstance(path, str) or not os.path.isfile(path):
        return False
    if not path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        return False
    if is_upscale(path):
        return False
    if temp_dir:
        try:
            if os.path.commonpath([os.path.realpath(temp_dir), os.path.realpath(path)]) == os.path.realpath(temp_dir):
                return False
        except ValueError:  # different drives on Windows
            pass
    return True


def upscale_metadata(metadata, settings: Settings, final: tuple, upscaler_label: str,
                     detail_label: str | None, note: str, original_name: str) -> list:
    """The original's (label, key, value) rows plus the upscale settings, for
    log.html and the embedded metadata. "Resolution" stays the generation
    size, so "Load these settings" on an upscale reproduces the original."""
    rows = list(metadata or [])
    rows.append(('Upscaled size', 'upscaled_size', str(tuple(final))))
    rows.append(('Auto upscale', 'auto_upscale', f'{scale_label(settings.scale)} · {upscaler_label}'
                 + (f' · {detail_label}' if detail_label else '')))
    if note:
        rows.append(('Auto upscale note', 'auto_upscale_note', note))
    rows.append(('Upscaled from', 'upscaled_from', original_name))
    return rows
