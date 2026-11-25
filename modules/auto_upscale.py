"""Detailed 2x upscaling of finished images, for scripts/auto_upscale.py.

Mirrors the app's "Upscale (2x)" (async_worker.apply_upscale): an ESRGAN
2x pass, then an SDXL diffusion pass at denoise 0.382 that redraws fine
detail, using the original image's prompt, styles, model and sampler (read
from its History record). Output: ``<name>_2x.png`` next to the original.

The GPU part loads its own copy of the model, so it must not run while the
app is open (two ~12 GB models don't fit in 24 GB); see app_running().
"""
import ast
import datetime
import os
import re
import socket
import subprocess

from modules.history import IMAGE_EXTENSIONS

SUFFIX = '_2x'
DEFAULT_PORTS = (7865, 7866)
_DATE_DIR_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_APP_PROCESS_RE = re.compile(r'(launch|entry_with_update|webui)\.py')


def upscaled_path(path: str) -> str:
    stem, _ext = os.path.splitext(path)
    return f'{stem}{SUFFIX}.png'


def is_upscaled(path: str) -> bool:
    return os.path.splitext(os.path.basename(path))[0].endswith(SUFFIX)


def find_pending(root: str, since: datetime.date | None = None) -> list:
    """Images under root/<YYYY-MM-DD>/ (on or after ``since``) that are not
    upscales themselves and have no upscale yet, oldest first."""
    pending = []
    if not root or not os.path.isdir(root):
        return pending
    for day in sorted(d for d in os.listdir(root) if _DATE_DIR_RE.match(d)):
        if since is not None and datetime.date.fromisoformat(day) < since:
            continue
        day_dir = os.path.join(root, day)
        for name in sorted(os.listdir(day_dir)):
            path = os.path.join(day_dir, name)
            if (name.lower().endswith(IMAGE_EXTENSIONS) and not is_upscaled(path)
                    and not os.path.exists(upscaled_path(path))):
                pending.append(path)
    return pending


def port_in_use(port: int, host: str = '127.0.0.1') -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def app_process_running(ps_output: str | None = None, own_pid: int | None = None) -> bool:
    """True if a FoocusRX app process (launch.py / entry_with_update.py /
    webui.py) appears in ``ps`` output."""
    if ps_output is None:
        try:
            ps_output = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True,
                                       text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return False
    own_pid = os.getpid() if own_pid is None else own_pid
    for line in ps_output.splitlines():
        pid, _, command = line.strip().partition(' ')
        if pid.isdigit() and int(pid) == own_pid:
            continue
        if 'python' in command.lower() and _APP_PROCESS_RE.search(command) and 'auto_upscale' not in command:
            return True
    return False


def app_running(ports=DEFAULT_PORTS) -> bool:
    extra = os.environ.get('GRADIO_SERVER_PORT')
    ports = tuple(ports) + ((int(extra),) if extra and extra.isdigit() else ())
    return any(port_in_use(p) for p in ports) or app_process_running()


def _literal(value, default):
    if isinstance(value, (list, tuple)):
        return value
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return default


def build_workloads(meta: dict, styles_table: dict, expansion_style: str = 'Fooocus V2'):
    """Positive/negative prompt lists and pool_top_k, built the way
    async_worker.process_prompt builds them for a task."""
    prompt = str(meta.get('prompt') or '')
    negative = str(meta.get('negative_prompt') or '')
    styles = [s for s in _literal(meta.get('styles', '[]'), []) if isinstance(s, str)]
    use_expansion = expansion_style in styles
    positive, negatives = [], []
    placeholder_replaced = False
    for style in styles:
        if style == expansion_style or style not in styles_table:
            continue
        p, n = styles_table[style]
        if '{prompt}' in p:
            placeholder_replaced = True
        positive += p.replace('{prompt}', prompt).splitlines()
        negatives += n.splitlines()
    if not placeholder_replaced:
        positive = [prompt] + positive
    negatives.append(negative)
    positive = [p for p in positive if p.strip()] or [prompt]
    negatives = [n for n in negatives if n.strip()] or [negative]
    pool_top_k = len(positive)
    expansion = str(meta.get('prompt_expansion') or '')
    if use_expansion and expansion.strip():
        positive = positive + [expansion]
    return positive, negatives, pool_top_k


def unsupported_reason(meta: dict) -> str | None:
    """Why an image can't be upscaled here, or None. The detailed upscale
    redraws with an SDXL model; images made with a GGUF model (FLUX.1)
    would be redrawn by the wrong model."""
    if str(meta.get('base_model', '')).lower().endswith('.gguf'):
        return 'made with a GGUF model such as FLUX.1; the detailed upscale uses SDXL'
    return None


def settings_from_metadata(meta: dict, steps_for_performance) -> dict:
    """Generation settings to reuse for the diffusion pass, with the app's
    defaults where the metadata is missing a value."""
    adm = _literal(meta.get('adm_guidance', '(1.5, 0.8, 0.3)'), (1.5, 0.8, 0.3))
    loras = []
    for key in sorted(k for k in meta if k.startswith('lora_combined_')):
        parts = str(meta[key]).split(' : ')
        if len(parts) == 3 and parts[0] != 'True':
            continue
        name, weight = (parts[1], parts[2]) if len(parts) == 3 else (parts + ['1.0'])[:2]
        if name != 'None':
            loras.append((name, float(weight)))
    def number(key, default, cast=float):
        value = meta.get(key)
        if value in (None, '', 'None'):
            return default
        try:
            return cast(value)
        except (TypeError, ValueError):
            pass
        try:  # e.g. clip_skip "2.0"; never used for seeds, which are always whole numbers
            return cast(float(value))
        except (TypeError, ValueError):
            return default

    performance = str(meta.get('performance') or 'Speed')
    return {
        'performance': performance,
        'steps': steps_for_performance(performance),
        'sampler': str(meta.get('sampler') or 'dpmpp_2m_sde_gpu'),
        'scheduler': str(meta.get('scheduler') or 'karras'),
        'cfg_scale': number('guidance_scale', 4.0),
        'sharpness': number('sharpness', 2.0),
        'adm_guidance': tuple(float(x) for x in adm),
        'adaptive_cfg': number('adaptive_cfg', 7.0),
        'clip_skip': number('clip_skip', 2, int),
        'seed': number('seed', 0, int),
        'base_model': meta.get('base_model'),
        'vae': meta.get('vae'),
        'loras': loras,
    }


class DetailedUpscaler:
    """GPU part. Importing this loads the default model, like the app does."""

    def __init__(self):
        import modules.patch as patch
        patch.patch_all()  # the worker applies these before loading models
        import modules.core as core
        import modules.default_pipeline as pipeline
        import modules.sdxl_styles as sdxl_styles
        from modules.flags import Performance
        self.patch, self.core, self.pipeline = patch, core, pipeline
        self.styles = sdxl_styles.styles
        self.Performance = Performance

    def steps_for(self, performance: str) -> int:
        try:
            return self.Performance(performance).steps_uov() or 18
        except ValueError:
            return 18

    def upscale(self, image_path: str, meta: dict, out_path: str) -> str:
        import json
        import numpy as np
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        import modules.config as config
        from modules.history import to_load_parameters
        from modules.upscaler import perform_upscale
        from modules.util import HWC3, get_shape_ceil, resample_image, set_image_shape_ceil

        s = settings_from_metadata(to_load_parameters(meta, config.model_filenames, config.lora_filenames,
                                                      config.vae_filenames), self.steps_for)
        base_model = s['base_model'] or config.default_base_model_name
        self.pipeline.refresh_everything(refiner_model_name='None', base_model_name=base_model,
                                         loras=s['loras'], vae_name=s['vae'] or config.default_vae)
        self.pipeline.set_clip_skip(s['clip_skip'])
        p, n, e = s['adm_guidance']
        self.patch.patch_settings[os.getpid()] = self.patch.PatchSettings(
            s['sharpness'], e, p, n, 0.25, s['adaptive_cfg'])

        positive, negative, pool_top_k = build_workloads(meta, self.styles)
        c = self.pipeline.clip_encode(texts=positive, pool_top_k=pool_top_k)
        uc = self.pipeline.clip_encode(texts=negative, pool_top_k=len(negative))

        # Same steps as async_worker.apply_upscale for 'Upscale (2x)'.
        image = HWC3(np.asarray(Image.open(image_path).convert('RGB')))
        H, W, _ = image.shape
        config.downloading_upscale_model()
        image = perform_upscale(image)
        shape_ceil = get_shape_ceil(H * 2, W * 2)
        if shape_ceil < 1024:
            image = set_image_shape_ceil(image, 1024)
            shape_ceil = 1024
        else:
            image = resample_image(image, width=W * 2, height=H * 2)
        if shape_ceil > 2800:  # the app returns the ESRGAN result directly here too
            result = image
        else:
            denoise = 0.382
            vae, _ = self.pipeline.get_candidate_vae(steps=s['steps'], switch=s['steps'], denoise=denoise)
            latent = self.core.encode_vae(vae=vae, pixels=self.core.numpy_to_pytorch(image), tiled=True)
            _, _, lh, lw = latent['samples'].shape
            result = self.pipeline.process_diffusion(
                positive_cond=c, negative_cond=uc, steps=s['steps'], switch=s['steps'],
                width=lw * 8, height=lh * 8, image_seed=s['seed'], callback=lambda *a: None,
                sampler_name=s['sampler'], scheduler_name=s['scheduler'], latent=latent, denoise=denoise,
                tiled=True, cfg_scale=s['cfg_scale'], refiner_swap_method='joint', disable_preview=True)[0]

        info = PngInfo()
        record = dict(meta, upscale='Upscale (2x)', upscaled_from=os.path.basename(image_path))
        info.add_text('parameters', json.dumps(record))
        info.add_text('fooocus_scheme', 'fooocus')
        tmp = out_path + '.partial'  # no image extension, so find_pending never picks it up
        Image.fromarray(result).save(tmp, format='PNG', pnginfo=info)
        os.replace(tmp, out_path)  # never leave a half-written *_2x.png
        return out_path
