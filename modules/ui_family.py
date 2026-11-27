"""Model family switch (SDXL / FLUX.1 / Krea 2 / MOSS-Image) for the main UI.

The family dropdown filters the base model list; picking a model of another
family (from History, a preset, or loaded parameters) switches the family to
match. Each family remembers the last model used, so switching back restores
it. Pure functions here; webui.py turns the results into Gradio updates.
"""
from modules.sdcpp import registry as sdcpp_registry

SDXL = 'SDXL'
FLUX = 'FLUX.1'
KREA2 = sdcpp_registry.KREA2.family
MOSS = sdcpp_registry.MOSS_IMAGE.family
FAMILIES = [SDXL, FLUX, KREA2, MOSS]
SDCPP_FAMILIES = (KREA2, MOSS)

FLUX_NOTICE = ('**FLUX.1 schnell**: 4 steps, natural-language prompts. Uses your prompt, styles, aspect ratio, '
               'seed and image number; **Input Image > Upscale or Variation** does Vary and Upscale. SDXL-only '
               'controls (negative prompt, performance, CFG, sampler, refiner, image prompt, inpaint, '
               'enhance) are hidden. LoRAs made for FLUX.1 work (Advanced > Models).')
FLUX_MISSING = ('FLUX.1 is not installed. Download it (~12 GB) with: '
                'venv/bin/python scripts/download_flux.py, then click Refresh All Files.')
NOTICES = {FLUX: FLUX_NOTICE, KREA2: sdcpp_registry.KREA2.notice, MOSS: sdcpp_registry.MOSS_IMAGE.notice}
MISSING = {
    SDXL: 'No SDXL checkpoints found in the checkpoints folder.',
    FLUX: FLUX_MISSING,
    KREA2: ('Krea 2 is not installed. Download it (~13 GB, Krea 2 Community License) with: '
            'venv/bin/python scripts/download_sdcpp.py krea2, then click Refresh All Files.'),
    MOSS: ('MOSS-Image is not installed. Download it (~12 GB) with: '
           'venv/bin/python scripts/download_sdcpp.py moss_image, then click Refresh All Files.'),
}


def family_of(model_name) -> str:
    spec = sdcpp_registry.spec_for(model_name)
    if spec is not None:
        return spec.family
    return FLUX if str(model_name or '').lower().endswith('.gguf') else SDXL


def is_sdxl(model_name) -> bool:
    return family_of(model_name) == SDXL


def uses_negative_prompt(family: str, model=None) -> bool:
    """Whether the negative prompt applies: SDXL, and sd.cpp checkpoints that
    sample with CFG (MOSS-Image SFT, Krea 2 Raw/Base; not the distilled ones)."""
    spec = sdcpp_registry.spec_by_family(family)
    if family == SDXL:
        return True
    if spec is None:
        return False
    return spec.settings_for(model)['uses_negative'] if model else spec.uses_negative


def models_for(family: str, model_filenames: list) -> list:
    return [m for m in model_filenames if family_of(m) == family]


def initial_memory(current_model, model_filenames: list) -> dict:
    memory = {}
    for family in FAMILIES:
        options = models_for(family, model_filenames)
        if options:
            memory[family] = options[0]
    if current_model:
        memory[family_of(current_model)] = current_model
    return memory


def select_family(family: str, memory: dict, model_filenames: list):
    """User picked a family. Returns (family, choices, model, warning).

    ``model`` is None when nothing should change (no models in that family;
    ``warning`` then says why and ``family`` is the fallback).
    """
    options = models_for(family, model_filenames)
    if not options:
        fallback = SDXL if family != SDXL else next(
            (f for f in FAMILIES if f != SDXL and models_for(f, model_filenames)), FLUX)
        return fallback, None, None, MISSING.get(family, f'No {family} models found.')
    remembered = (memory or {}).get(family)
    return family, options, remembered if remembered in options else options[0], None


def model_changed(model, shown_family: str, memory: dict, model_filenames: list):
    """The base model changed (any source). Returns (family, memory, choices).

    ``choices`` is None when the dropdown already lists this family's models.
    """
    family = family_of(model)
    memory = dict(memory or {})
    if model:
        memory[family] = model
    choices = models_for(family, model_filenames) if family != shown_family else None
    return family, memory, choices
