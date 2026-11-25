"""Glue between the FoocusRX app and the FLUX.1 schnell pipeline.

Flux shows up in the Base Model list as FLUX_MODEL_NAME when its four files
are in ``path_flux`` and the optional ``gguf`` package is installed. Only
text-to-image is supported; see UNSUPPORTED_NOTICE.
"""
import gc
import importlib.util
import os
import random

FLUX_MODEL_NAME = 'flux1-schnell-Q4_K_S.gguf'
UNSUPPORTED_NOTICE = (
    'FLUX.1 schnell: 4-step text-to-image. Uses your prompt, styles (as text), wildcards, aspect ratio, '
    'seed and image number. Not used with Flux: negative prompt, guidance scale/CFG, sharpness, sampler, '
    'performance, LoRAs, refiner, Fooocus V2 expansion, image prompt, inpaint/outpaint, vary/upscale, enhance.'
)
_SKIP_STYLES = ('Fooocus V2',)

_pipeline = None


def flux_dir() -> str:
    import modules.config as config
    return getattr(config, 'path_flux', '') or ''


def dependencies_installed() -> bool:
    return importlib.util.find_spec('gguf') is not None and importlib.util.find_spec('sentencepiece') is not None


def available(model_dir: str | None = None) -> bool:
    from modules.flux.pipeline import missing_files
    model_dir = flux_dir() if model_dir is None else model_dir
    return bool(model_dir) and os.path.isdir(model_dir) and not missing_files(model_dir) and dependencies_installed()


def model_choices(model_dir: str | None = None) -> list:
    return [FLUX_MODEL_NAME] if available(model_dir) else []


def is_flux(model_name) -> bool:
    return model_name == FLUX_MODEL_NAME


def build_prompt(prompt: str, styles: list, styles_table: dict, rng: random.Random) -> str:
    """Apply styles to the prompt as plain text, the way Fooocus builds the
    positive prompt (``{prompt}`` placeholders, otherwise appended)."""
    from modules.sdxl_styles import get_random_style, random_style_name
    parts = []
    placeholder_used = False
    for style in styles:
        if style in _SKIP_STYLES:
            continue
        if style == random_style_name:
            style = get_random_style(rng)
        if style not in styles_table:
            continue
        positive, _negative = styles_table[style]
        if '{prompt}' in positive:
            placeholder_used = True
        parts += [p.strip() for p in positive.replace('{prompt}', prompt).splitlines() if p.strip()]
    if not placeholder_used:
        parts = [prompt.strip()] + parts
    return ', '.join(p for p in parts if p)


def free_sdxl():
    """Release the SDXL models for real. On Apple Silicon "offloading" only
    moves weights within the same unified memory, so SDXL (~12 GB) and Flux
    together would push a 24 GB Mac into swap. The next SDXL generation
    reloads its model from disk through refresh_everything()."""
    import modules.core as core
    import modules.default_pipeline as pipeline
    import ldm_patched.modules.model_management as mm
    mm.unload_all_models()
    pipeline.model_base = core.StableDiffusionModel()
    pipeline.model_refiner = core.StableDiffusionModel()
    pipeline.final_unet = pipeline.final_clip = pipeline.final_vae = None
    pipeline.final_refiner_unet = pipeline.final_refiner_vae = None
    pipeline.loaded_ControlNets = {}
    gc.collect()
    mm.soft_empty_cache(force=True)


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from modules.flux.pipeline import FluxSchnell
        _pipeline = FluxSchnell(flux_dir())
    return _pipeline


def unload():
    """Free Flux before an SDXL generation (same reason as free_sdxl)."""
    global _pipeline
    if _pipeline is not None:
        _pipeline.unload()
        _pipeline = None
        gc.collect()
