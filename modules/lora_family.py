"""Which model family a LoRA file belongs to, from its tensor names.

Reads only the file's header (no weights), .safetensors or .gguf, through
modules.model_files, cached by path, size and mtime. Tensor names are matched
in every common spelling: kohya (``lora_unet_blocks_0_attn_wq``), PEFT /
diffusers (``transformer.transformer_blocks.0.attn.to_q.lora_A``) and
ComfyUI / native (``diffusion_model.blocks.0.attn.wq.lora_A``).

Files that can't be read (.ckpt, .pt, a broken header) count as SDXL, as
before. A readable LoRA made for a model this app doesn't run (Qwen-Image,
SD3, Wan, ...) is OTHER: it is listed for no family, and the LoRA panel says
so instead of the LoRA silently doing nothing.
"""
import os

from modules import model_files

SDXL = model_files.SDXL
FLUX = model_files.FLUX
KREA2 = model_files.KREA2
MOSS = model_files.MOSS
OTHER = model_files.OTHER
FAMILIES = (SDXL, FLUX, KREA2, MOSS)


def family_from_keys(keys) -> str:
    """Family of a LoRA with these tensor names. SDXL when nothing matches,
    except that LoRA-shaped names matching no family are OTHER."""
    keys = list(keys)
    family = model_files.family_from_names(keys)
    if family:
        return family
    return OTHER if model_files.classify_names(keys).kind == model_files.LORA else SDXL


def family_of_lora(path: str) -> str:
    """Model family a LoRA file is for (SDXL when unknown or unreadable)."""
    if not path or not os.path.isfile(path):
        return SDXL
    info = model_files.inspect(path)
    if info.kind == model_files.LORA:
        return info.family or OTHER
    if info.kind == model_files.MODEL:
        # A whole model in the LoRA folder: never a LoRA for any family.
        return OTHER
    return SDXL


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


def group_by_family(names: list, resolve) -> dict:
    """{family: [names]} for every LoRA in ``names`` (OTHER included)."""
    groups = {}
    for name in names:
        path = resolve(name)
        groups.setdefault(family_of_lora(path) if path else SDXL, []).append(name)
    return groups


def _short(names, limit=4) -> str:
    shown = ', '.join(os.path.splitext(os.path.basename(n))[0] for n in names[:limit])
    return shown + (f' and {len(names) - limit} more' if len(names) > limit else '')


def panel_note(family: str, names: list, resolve) -> str:
    """One line for the LoRA panel: how many LoRAs fit ``family`` and which
    ones in the LoRA folders are for something else, so a LoRA that doesn't
    show is explained instead of silently missing."""
    groups = group_by_family(names, resolve)
    usable = len(groups.get(family, []))
    if not names:
        return ('No LoRAs found. Put .safetensors LoRAs in the LoRA folder (or next to your models), '
                'then click Refresh All Files.')
    parts = []
    for other in FAMILIES:
        if other != family and groups.get(other):
            parts.append(f'{len(groups[other])} for {other} ({_short(groups[other])})')
    if groups.get(OTHER):
        parts.append(f'{len(groups[OTHER])} for a model this app can\'t run ({_short(groups[OTHER])})')
    head = f'{usable} LoRA{"" if usable == 1 else "s"} fit {family}.'
    return head + (' Not shown because they are made for another model: ' + '; '.join(parts) + '.' if parts else '')
