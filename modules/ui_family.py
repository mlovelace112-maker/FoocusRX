"""Model family switch (SDXL / FLUX.1) for the main UI.

The family dropdown filters the base model list; picking a model of the other
family (from History, a preset, or loaded parameters) switches the family to
match. Each family remembers the last model used, so switching back restores
it. Pure functions here; webui.py turns the results into Gradio updates.
"""
SDXL = 'SDXL'
FLUX = 'FLUX.1'
FAMILIES = [SDXL, FLUX]

FLUX_NOTICE = ('**FLUX.1 schnell**: 4 steps, natural-language prompts. Uses your prompt, styles, aspect ratio, '
               'seed and image number; **Input Image > Upscale or Variation** does Vary and Upscale. SDXL-only '
               'controls (negative prompt, performance, CFG, sampler, LoRAs, refiner, image prompt, inpaint, '
               'enhance) are hidden.')
FLUX_MISSING = ('FLUX.1 is not installed. Download it (~12 GB) with: '
                'venv/bin/python scripts/download_flux.py, then click Refresh All Files.')


def family_of(model_name) -> str:
    return FLUX if str(model_name or '').lower().endswith('.gguf') else SDXL


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
        other = SDXL if family == FLUX else FLUX
        warning = FLUX_MISSING if family == FLUX else 'No SDXL checkpoints found in the checkpoints folder.'
        return other, None, None, warning
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
