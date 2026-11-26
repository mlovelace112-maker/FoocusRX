"""Glue between the FoocusRX app and the stable-diffusion.cpp models (Krea 2,
MOSS-Image).

A model shows up in the Base Model list under its diffusion GGUF's file name
when the engine (``path_sdcpp``/sd-cli) and that model's support files are in
place; scripts/download_sdcpp.py fetches both. Text-to-image only.
"""
import json
import os
import random

from modules.sdcpp import registry
from modules.sdcpp.registry import ENGINE_BINARY, MODELS

_SKIP_STYLES = ('Fooocus V2',)
EXTRA_ARGS_ENV = 'FOOOCUS_SDCPP_ARGS'  # extra sd-cli flags, e.g. "--offload-to-cpu --vae-tiling"


def _config():
    import modules.config as config
    return config


def engine_dir() -> str:
    return getattr(_config(), 'path_sdcpp', '') or ''


def engine_path(directory: str | None = None) -> str:
    return os.path.join(engine_dir() if directory is None else directory, ENGINE_BINARY)


def engine_installed(directory: str | None = None) -> bool:
    path = engine_path(directory)
    return os.path.isfile(path) and os.access(path, os.X_OK)


def model_dir(spec) -> str:
    return getattr(_config(), spec.config_key, '') or ''


def missing_files(spec, directory: str) -> list:
    return [f.name for f in spec.support_files() if not os.path.isfile(os.path.join(directory, f.name))]


def diffusion_models(spec, directory: str) -> list:
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(n for n in os.listdir(directory) if spec.matches(n) and not n.endswith('.partial'))


def model_choices(engine: str | None = None, dirs: dict | None = None) -> list:
    """Base model names for every sd.cpp model that is ready to run."""
    if not engine_installed(engine):
        return []
    names = []
    for spec in MODELS:
        directory = (dirs or {}).get(spec.key) if dirs is not None else model_dir(spec)
        if directory and os.path.isdir(directory) and not missing_files(spec, directory):
            names += diffusion_models(spec, directory)
    return names


def is_sdcpp(model_name) -> bool:
    return registry.spec_for(model_name) is not None


def is_json_prompt(prompt: str) -> bool:
    text = prompt.strip()
    if not text.startswith('{'):
        return False
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def build_prompt(prompt: str, styles: list, styles_table: dict, rng: random.Random) -> str:
    """Apply styles as plain text, keeping the prompt's line breaks (MOSS-Image
    renders multi-line text). A JSON layout prompt is returned unchanged."""
    from modules.sdxl_styles import get_random_style, random_style_name
    if is_json_prompt(prompt):
        return prompt.strip()
    text = prompt.strip()
    extra = []
    for style in styles:
        if style in _SKIP_STYLES:
            continue
        if style == random_style_name:
            style = get_random_style(rng)
        if style not in styles_table:
            continue
        positive, _negative = styles_table[style]
        if '{prompt}' in positive:
            text = positive.replace('{prompt}', text).strip()
        elif positive.strip():
            extra.append(positive.strip())
    return ', '.join([text] + extra) if extra else text


def snap16(value: int) -> int:
    return max(256, 16 * int(round(value / 16)))


def extra_args() -> list:
    import shlex
    return shlex.split(os.environ.get(EXTRA_ARGS_ENV, ''))
