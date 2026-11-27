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


def support_names(spec) -> set:
    return {f.name.lower() for f in spec.support_files()}


def _belongs(spec, name: str, path: str, in_own_folder: bool) -> bool:
    """Is the file at ``path`` a diffusion model of ``spec``'s family?

    Decided from the file's header (any name, .gguf or .safetensors). A file
    whose header can't be read counts only when it sits in the family's own
    folder and its name matches the family (the original rule)."""
    from modules import model_files
    if os.path.basename(name).lower() in support_names(spec) or name.endswith('.partial'):
        return False
    info = model_files.inspect(path)
    if info.kind == model_files.MODEL:
        return info.family == spec.family
    if info.kind == model_files.UNKNOWN:
        return in_own_folder and spec.matches(name)
    return False  # a LoRA, text encoder or VAE


def diffusion_models(spec, directory: str) -> list:
    """Names (relative to ``directory``, subfolders included) of this family's
    diffusion models in ``directory``."""
    from modules import model_files
    return [name for name, path in model_files.scan_folder(directory)
            if _belongs(spec, name, path, in_own_folder=True)]


def discover(engine: str | None = None, dirs: dict | None = None, extra_dirs=()) -> dict:
    """{base model name: (spec, path)} for every sd.cpp model that is ready to
    run: the engine and the family's support files are installed. Looks in each
    family's folder and in ``extra_dirs`` (the checkpoints folders, where a
    downloaded model often ends up)."""
    from modules import model_files
    if not engine_installed(engine):
        return {}
    found = {}
    ready = []
    for spec in MODELS:
        directory = (dirs or {}).get(spec.key) if dirs is not None else model_dir(spec)
        if directory and os.path.isdir(directory) and not missing_files(spec, directory):
            ready.append((spec, directory))
    for spec, directory in ready:
        for name, path in model_files.scan_folder(directory):
            if _belongs(spec, name, path, in_own_folder=True):
                found.setdefault(name, (spec, path))
    for folder in extra_dirs or ():
        for name, path in model_files.scan_folder(folder):
            if name in found:
                continue
            for spec, _directory in ready:
                if _belongs(spec, name, path, in_own_folder=False):
                    found[name] = (spec, path)
                    break
    return found


def model_choices(engine: str | None = None, dirs: dict | None = None, extra_dirs=()) -> list:
    """Base model names for every sd.cpp model that is ready to run, in
    family order. Also records where each one is (registry.installed_path)."""
    found = discover(engine, dirs, extra_dirs)
    registry.register_installed(found)
    order = {spec.key: i for i, spec in enumerate(MODELS)}
    return sorted(found, key=lambda n: (order[found[n][0].key], n.casefold()))


def diffusion_path(model_name: str) -> str:
    """Where a base model's diffusion weights are: the path found on disk, or
    (not scanned yet) the model name inside its family's folder."""
    path = registry.installed_path(model_name)
    if path:
        return path
    spec = registry.spec_for(model_name)
    return os.path.join(model_dir(spec), model_name)


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


def extract_prompt_loras(prompt: str, lora_filenames: list):
    """(prompt without <lora:NAME:W> tags, [(file name, weight), ...]).

    Unlike util.parse_lora_references_from_prompt this keeps the rest of the
    prompt byte for byte: MOSS-Image prompts carry line breaks and JSON."""
    from modules.util import LORAS_PROMPT_PATTERN, get_filname_by_stem
    found = []
    for whole, stem, weight in LORAS_PROMPT_PATTERN.findall(prompt):
        name = get_filname_by_stem(stem, lora_filenames)
        if name is not None:
            found.append((name, float(weight)))
        prompt = prompt.replace(whole, '', 1)
    return prompt.strip(), found


def merge_loras(selected: list, from_prompt: list, limit: int) -> list:
    """Selected LoRAs first, then prompt ones not already selected; 'None' and
    zero weights dropped; at most ``limit``."""
    out, seen = [], set()
    for name, weight in list(selected) + list(from_prompt):
        if name == 'None' or name in seen or float(weight) == 0:
            continue
        seen.add(name)
        out.append((name, float(weight)))
    return out[:limit]


def stage_loras(loras: list, directory: str, resolve) -> tuple:
    """Link the LoRA files into ``directory`` as l0, l1, ... so the tags stay
    valid whatever the file names are (spaces, ':', '>', subfolders, several
    LoRA folders). Returns (lora dir or None, prompt tag suffix)."""
    if not loras:
        return None, ''
    os.makedirs(directory, exist_ok=True)
    tags = []
    for i, (name, weight) in enumerate(loras):
        src = resolve(name)
        ext = os.path.splitext(src)[1] or '.safetensors'
        os.symlink(src, os.path.join(directory, f'l{i}{ext}'))
        tags.append(f'<lora:l{i}:{weight:g}>')
    return directory, ''.join(tags)


def snap16(value: int) -> int:
    return max(256, 16 * int(round(value / 16)))


def default_args(platform: str | None = None) -> list:
    """sd-cli flags added to every run, for Apple Silicon (unified memory):

    * --auto-fit off keeps all weights on Metal. sd-cli's auto-fit (default
      on) misjudges unified memory and sometimes spills part of the model to a
      zero-sized "RAM" budget ("RAM graph capacity exceeded ... diffusion
      compute failed", seen with Krea 2 on a 24 GB M5 Pro).
    * --mmap maps the weights from disk instead of copying them: Metal uses the
      mapped pages directly, so they are clean file pages macOS can drop under
      pressure instead of swapping. Measured with Krea 2: peak footprint
      13.8 GB -> 0.5 GB, load 8-33 s -> 0.2 s, same sampling speed. Without it
      a second app-driven run failed "during weight preparation" with swap full.
    * --flash-attn on: auto picks off on Metal, but on shrinks the Krea 2
      compute buffer at 1024x1024 from 3.8 GB to 1.2 GB and samples ~10%
      faster, with the same image (mean pixel difference 1.3/255). sd.cpp
      checks free + inactive RAM before each graph, so that 2.6 GB decides
      whether Krea 2 fits next to the app on a 24 GB Mac.
    """
    import sys
    return ['--auto-fit', 'off', '--mmap', '--flash-attn', 'on'] if (platform or sys.platform) == 'darwin' else []


_MEMORY_ERRORS = ('capacity exceeded', 'failed during weight preparation', 'diffusion compute failed',
                  'failed to allocate', 'out of memory')


def explain_error(message: str, family: str) -> str:
    """A short message for the UI; memory failures get advice."""
    if any(m in message.lower() for m in _MEMORY_ERRORS):
        return (f'{family}: not enough free memory for this image. Close other apps (browsers, other '
                f'models) and try again, or pick a smaller size.')
    last = [line for line in message.splitlines() if line.strip()][-1:] or ['sd-cli failed']
    return f'{family}: {last[0].strip()[:200]}'


def extra_args() -> list:
    """default_args() plus FOOOCUS_SDCPP_ARGS (later flags win in sd-cli)."""
    import shlex
    return default_args() + shlex.split(os.environ.get(EXTRA_ARGS_ENV, ''))
