"""Which model family a LoRA file belongs to, from its tensor names.

Reads only the .safetensors header (no weights), cached by path, size and
mtime. Anything unrecognised counts as SDXL, so existing SDXL LoRAs in any
naming style (kohya, LyCORIS, diffusers, Civitai) keep showing for SDXL.
"""
import os
import re

from modules.model_profiles import read_header
from modules.sdcpp.registry import KREA2, MOSS_IMAGE

SDXL = 'SDXL'
FLUX = 'FLUX.1'

# (family, pattern on any tensor name) - first family with a match wins.
_RULES = (
    (MOSS_IMAGE.family, re.compile(r'(^|\.)layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)\.')),
    # Krea 2 blocks: gated attention and a gate/up/down FFN (FLUX has neither).
    (KREA2.family, re.compile(r'text_fusion\.|transformer\.final_layer\.|time_mod_proj|'
                              r'\.attn\.to_gate\.|\.ff\.(gate|up|down)\.')),
    (FLUX, re.compile(r'double_blocks|single_blocks|single_transformer_blocks|'
                      r'\.attn\.add_[qkv]_proj\.|\.ff_context\.|\.norm1_context\.')),
)

_cache = {}


def family_from_keys(keys) -> str:
    for family, pattern in _RULES:
        if any(pattern.search(k) for k in keys):
            return family
    return SDXL


def family_of_lora(path: str) -> str:
    """Model family a LoRA file is for (SDXL when unknown or unreadable)."""
    try:
        st = os.stat(path)
    except OSError:
        return SDXL
    key = (path, st.st_size, st.st_mtime)
    if key not in _cache:
        _meta, names = read_header(path)
        _cache[key] = family_from_keys(names)
    return _cache[key]


def loras_for(family: str, names: list, resolve) -> list:
    """The LoRA file names (as listed in the UI) that fit ``family``.
    ``resolve(name)`` returns the file's path, or None."""
    out = []
    for name in names:
        path = resolve(name)
        if path and family_of_lora(path) == family:
            out.append(name)
    return out


def split_by_family(loras: list, family: str, resolve):
    """Split [(name, weight), ...] into (usable, skipped) for ``family``."""
    usable, skipped = [], []
    for name, weight in loras:
        path = resolve(name)
        (usable if path and family_of_lora(path) == family else skipped).append((name, weight))
    return usable, skipped
