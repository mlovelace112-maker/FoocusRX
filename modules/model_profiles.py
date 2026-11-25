"""Best settings for the selected checkpoint.

Profiles come from the presets that ship with Fooocus (each pairs a model
with tuned settings). A checkpoint is matched to one:
  1. exactly, when it is a preset's default_model;
  2. by family, from keywords in its file name, its embedded title
     (modelspec.title) and any source checkpoints recorded by merge tools;
  3. otherwise the default preset (Fooocus's general SDXL tuning).
Distilled few-step merges (Lightning/Turbo/Hyper/LCM/DMD) get the settings
the app itself uses for its Lightning mode, without adding the LoRA.
Only quality settings are applied; prompts, styles and LoRAs are untouched.
"""
import json
import os
import re
import struct
from dataclasses import dataclass, field

# Preset keys -> settings this feature manages.
PRESET_KEYS = {
    'default_performance': 'performance',
    'default_cfg_scale': 'guidance_scale',
    'default_sample_sharpness': 'sharpness',
    'default_sampler': 'sampler_name',
    'default_scheduler': 'scheduler_name',
    'default_overwrite_step': 'overwrite_step',
    'default_aspect_ratio': 'aspect_ratio',
    'default_vae': 'vae_name',
    'default_clip_skip': 'clip_skip',
    'default_cfg_tsnr': 'adaptive_cfg',
}
SETTING_LABELS = {
    'performance': 'Performance', 'guidance_scale': 'Guidance scale', 'sharpness': 'Sharpness',
    'sampler_name': 'Sampler', 'scheduler_name': 'Scheduler', 'overwrite_step': 'Steps override',
    'aspect_ratio': 'Aspect ratio', 'vae_name': 'VAE', 'clip_skip': 'CLIP skip', 'adaptive_cfg': 'Adaptive CFG',
}
# Presets that set a speed mode rather than tune a model.
SPEED_PRESETS = frozenset({'lcm'})
FAMILY_PATTERNS = (  # (preset or 'distilled', regex) — first match wins
    ('distilled', r'lightning|turbo|hyper[-_ ]?sd|\blcm\b|_lcm|dmd2|\b[48][-_ ]?steps?\b'),
    ('pony_v6', r'pony'),
    ('playground_v2.5', r'playground'),
    ('anime', r'illustrious|noobai|animagine|anime'),
    ('realistic', r'realvis|realistic|photoreal'),
    ('sai', r'sd_xl_base|sdxl_base'),
)
# Mirrors the Lightning performance settings in async_worker, minus the LoRA.
DISTILLED_SETTINGS = {
    'performance': 'Speed', 'overwrite_step': 8, 'sampler_name': 'euler', 'scheduler_name': 'sgm_uniform',
    'guidance_scale': 1.0, 'sharpness': 0.0, 'adaptive_cfg': 1.0,
}
SDXL_ARCHITECTURE = 'stable-diffusion-xl-v1-base'


@dataclass
class Profile:
    name: str                  # preset name, 'distilled' or 'default'
    reason: str                # why it was chosen, shown to the user
    settings: dict = field(default_factory=dict)
    supported: bool = True     # False for non-SDXL base checkpoints


def read_header(path: str) -> tuple[dict, list]:
    """(__metadata__, tensor names) from a .safetensors header, without
    reading any weights. ({}, []) for other formats or on error."""
    if not path or not path.lower().endswith('.safetensors'):
        return {}, []
    try:
        with open(path, 'rb') as fh:
            size = struct.unpack('<Q', fh.read(8))[0]
            if size > 100 * 1024 * 1024:  # corrupt or not safetensors
                return {}, []
            header = json.loads(fh.read(size))
    except (OSError, ValueError, struct.error):
        return {}, []
    meta = header.pop('__metadata__', None) or {}
    return (meta if isinstance(meta, dict) else {}), list(header)


def _source_names(meta: dict) -> list:
    """Models that went into a merge: the A1111 merge recipe, and the
    checkpoints/LoRAs in the graph ComfyUI actually executed ("prompt").
    ComfyUI's "workflow" field is the UI layout and can include unused or
    bypassed nodes, so it is ignored."""
    names = []
    for key in ('sd_merge_models', 'sd_merge_recipe'):
        raw = meta.get(key)
        if raw:
            names += re.findall(r'"(?:name|model_a|model_b|model_c)"\s*:\s*"([^"]+)"', str(raw))
    try:
        graph = json.loads(meta.get('prompt') or '{}')
    except (TypeError, ValueError):
        graph = {}
    for node in graph.values() if isinstance(graph, dict) else []:
        inputs = node.get('inputs', {}) if isinstance(node, dict) else {}
        if isinstance(inputs.get('ckpt_name'), str):
            names.append(inputs['ckpt_name'])
        if isinstance(inputs.get('lora_name'), str):
            strength = inputs.get('strength_model', 1.0)
            if not isinstance(strength, (int, float)) or strength != 0:
                names.append(inputs['lora_name'])
    return names


def is_sdxl_base(meta: dict, tensor_names: list) -> bool | None:
    """True/False when the header says, None when it can't tell."""
    arch = str(meta.get('modelspec.architecture', ''))
    if arch:
        return arch.split('/')[0] == SDXL_ARCHITECTURE
    if tensor_names:
        return any(n.startswith('conditioner.embedders.1') for n in tensor_names)
    return None


def load_presets(preset_dir: str) -> dict:
    presets = {}
    for name in sorted(os.listdir(preset_dir)) if os.path.isdir(preset_dir) else []:
        if name.endswith('.json'):
            try:
                with open(os.path.join(preset_dir, name), encoding='utf-8') as fh:
                    presets[name[:-5]] = json.load(fh)
            except (OSError, ValueError):
                continue
    return presets


def preset_settings(preset: dict) -> dict:
    return {setting: preset[key] for key, setting in PRESET_KEYS.items() if preset.get(key) is not None}


def choose_profile(model_filename: str, model_path: str | None, presets: dict) -> Profile:
    if str(model_filename).lower().endswith('.gguf'):
        # e.g. FLUX.1 schnell, which ignores the SDXL quality settings.
        return Profile('none', 'GGUF model such as FLUX.1; SDXL settings left unchanged', supported=False)
    meta, tensors = read_header(model_path)
    if is_sdxl_base(meta, tensors) is False:
        return Profile('none', 'not an SDXL base model; settings left unchanged', supported=False)

    base = os.path.basename(model_filename)
    for name, preset in presets.items():
        if name not in SPEED_PRESETS and preset.get('default_model') == base:
            return Profile(name, f'matches the "{name}" preset', preset_settings(preset))

    haystack = ' '.join([base, str(meta.get('modelspec.title', ''))] + _source_names(meta)).lower()
    for family, pattern in FAMILY_PATTERNS:
        match = re.search(pattern, haystack)
        if match is None:
            continue
        hit = match.group(0)
        if family == 'distilled':
            return Profile('distilled', f'looks like a few-step distilled model ("{hit}")', dict(DISTILLED_SETTINGS))
        if family in presets:
            return Profile(family, f'looks like a {family} model ("{hit}")', preset_settings(presets[family]))

    if 'default' in presets:
        return Profile('default', 'general SDXL model: Fooocus default tuning', preset_settings(presets['default']))
    return Profile('none', 'no profile available', supported=False)


def resolve_for_ui(settings: dict, *, vae_filenames, aspect_ratio_labels, add_ratio, default_vae) -> dict:
    """Turn profile settings into values the UI components accept, dropping
    anything that isn't available here (e.g. a VAE that isn't installed)."""
    out = {}
    for key, value in settings.items():
        if key == 'aspect_ratio':
            label = add_ratio(str(value))
            if label in aspect_ratio_labels:
                out[key] = label
        elif key == 'vae_name':
            if value == default_vae or value in vae_filenames:
                out[key] = value
            else:
                out[key] = default_vae  # profile's VAE isn't installed
        else:
            out[key] = value
    return out


def describe_changes(current: dict, new: dict) -> list:
    """Human-readable 'Setting: old -> new' lines for values that differ."""
    def fmt(v):
        if isinstance(v, float):
            return f'{v:g}'
        text = str(v)
        return text.split(' <span')[0]  # aspect-ratio labels carry HTML for the ratio hint

    lines = []
    for key, value in new.items():
        old = current.get(key)
        if old != value and not (isinstance(old, (int, float)) and isinstance(value, (int, float)) and float(old) == float(value)):
            lines.append(f'{SETTING_LABELS.get(key, key)}: {fmt(old)} → {fmt(value)}')
    return lines
