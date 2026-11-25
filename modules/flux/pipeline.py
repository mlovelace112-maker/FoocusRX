"""FLUX.1 [schnell] text-to-image, image-to-image and tiled refine.

Sampling follows black-forest-labs/flux @ 802fb47 (Apache-2.0):
src/flux/sampling.py (get_noise, prepare, get_schedule, denoise, unpack) with
the schnell settings from src/flux/cli.py: 4 steps, no timestep shift, T5
max_length 256, no guidance embedding.

Memory on a 24 GB Mac: the text encoders run first and T5 (~10 GB) is freed
before the quantized transformer (~7 GB resident) is used. Text encodings are
cached per prompt so repeated prompts skip the T5 load.
"""
import gc
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange, repeat

DEFAULT_STEPS = 4
FILES = {
    'transformer': 'flux1-schnell-Q4_K_S.gguf',
    't5': 't5xxl_fp8_e4m3fn.safetensors',
    'clip': 'clip_l.safetensors',
    'vae': 'ae.safetensors',
}


@dataclass
class FluxTimings:
    text_s: float = 0.0
    load_s: float = 0.0
    sample_s: float = 0.0
    decode_s: float = 0.0


def missing_files(model_dir: str) -> list:
    return [name for name in FILES.values() if not os.path.exists(os.path.join(model_dir, name))]


def _free():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# --- sampling.py (reference) ------------------------------------------------

def get_noise(num_samples: int, height: int, width: int, device, dtype, seed: int) -> torch.Tensor:
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device)


def prepare(img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor) -> dict:
    bs, c, h, w = img.shape
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)
    return {"img": img, "img_ids": img_ids.to(img.device), "txt": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device), "vec": vec.to(img.device)}


def get_schedule(num_steps: int, image_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15,
                 shift: bool = True) -> list:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        m = (max_shift - base_shift) / (4096 - 256)
        mu = m * image_seq_len + (base_shift - m * 256)
        timesteps = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1) ** 1.0)
    return timesteps.tolist()


def denoise(model, img, img_ids, txt, txt_ids, vec, timesteps: list, callback=None):
    for step, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, y=vec, timesteps=t_vec, guidance=None)
        img = img + (t_prev - t_curr) * pred
        if callback is not None:
            callback(step, len(timesteps) - 1)
    return img


def img2img_schedule(num_steps: int, strength: float) -> list:
    """Flow-matching image-to-image: start at t=strength on the straight path
    x_t = t * noise + (1 - t) * image and integrate to 0, using the share of
    the schnell schedule that strength covers (at least one step)."""
    strength = min(max(float(strength), 0.05), 1.0)
    steps = max(1, math.ceil(num_steps * strength))
    return torch.linspace(strength, 0, steps + 1).tolist()


def feather_weights(height: int, width: int, overlap: int, top: bool, bottom: bool, left: bool,
                    right: bool) -> np.ndarray:
    """HxW blend weights for a tile: linear ramps over ``overlap`` px on the
    sides that overlap a neighbour, 1 elsewhere (never 0, so edges count)."""
    def ramp(n, start, end):
        w = np.ones(n, dtype=np.float32)
        r = min(overlap, n // 2)
        if r > 0:
            edge = (np.arange(r, dtype=np.float32) + 1) / (r + 1)
            if start:
                w[:r] = edge
            if end:
                w[-r:] = edge[::-1]
        return w
    return np.outer(ramp(height, top, bottom), ramp(width, left, right))


def tile_starts(size: int, tile: int, overlap: int) -> list:
    """Start offsets covering ``size`` with tiles of ``tile`` overlapping by >= ``overlap``."""
    if size <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, size - tile, stride)) + [size - tile]
    return sorted(set(starts))


def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                     h=math.ceil(height / 16), w=math.ceil(width / 16), ph=2, pw=2)


# --- pipeline ---------------------------------------------------------------

class FluxSchnell:
    def __init__(self, model_dir: str, device='mps', dtype=torch.bfloat16):
        missing = missing_files(model_dir)
        if missing:
            raise FileNotFoundError(f'FLUX.1 schnell files missing from {model_dir}: {", ".join(missing)}')
        self.dir, self.device, self.dtype = model_dir, device, dtype
        self.transformer = None
        self.vae = None
        self._text_cache = {}  # prompt -> (txt, vec) on CPU
        self.timings = FluxTimings()

    def _path(self, key):
        return os.path.join(self.dir, FILES[key])

    def encode(self, prompts: list) -> None:
        """Encode any prompts not cached yet. Frees the transformer first if
        T5 is needed (T5 + transformer don't fit together on 24 GB)."""
        todo = [p for p in dict.fromkeys(prompts) if p not in self._text_cache]
        if not todo:
            return
        from modules.flux.text_encoders import ClipEncoder, T5Encoder
        t = time.perf_counter()
        self.unload_transformer()
        t5 = T5Encoder(self._path('t5'), self.device, self.dtype)
        txt = t5(todo).cpu()
        del t5
        _free()
        clip = ClipEncoder(self._path('clip'), self.device, self.dtype)
        vec = clip(todo).cpu()
        del clip
        _free()
        for i, prompt in enumerate(todo):
            self._text_cache[prompt] = (txt[i:i + 1], vec[i:i + 1])
        self.timings.text_s += time.perf_counter() - t

    def load(self):
        if self.transformer is None:
            from modules.flux import gguf_ops
            from modules.flux.model import Flux
            t = time.perf_counter()
            with torch.device('meta'):
                model = Flux()
            self.transformer = gguf_ops.load_into(model, gguf_ops.read_gguf(self._path('transformer')),
                                                  self.device, self.dtype)
            self.timings.load_s += time.perf_counter() - t
        if self.vae is None:
            from safetensors.torch import load_file
            from modules.flux.autoencoder import AutoEncoder
            vae = AutoEncoder()
            vae.load_state_dict(load_file(self._path('vae')), strict=True)
            self.vae = vae.to(self.device, dtype=self.dtype).eval().requires_grad_(False)

    def unload_transformer(self):
        if self.transformer is not None:
            self.transformer = None
            _free()

    def unload(self):
        self.transformer = None
        self.vae = None
        self._text_cache.clear()
        _free()

    @torch.no_grad()
    def generate(self, prompt: str, width: int, height: int, seed: int, steps: int = DEFAULT_STEPS,
                 callback=None, init_image: np.ndarray | None = None, strength: float = 1.0) -> np.ndarray:
        """One image as an HxWx3 uint8 array. Width/height are rounded down to
        multiples of 16. With ``init_image`` (HxWx3 uint8, resized to fit) and
        strength < 1 this is image-to-image: lower strength keeps more of it."""
        width, height = 16 * (width // 16), 16 * (height // 16)
        self.encode([prompt])
        self.load()
        txt, vec = self._text_cache[prompt]

        t = time.perf_counter()
        x = get_noise(1, height, width, self.device, self.dtype, seed)
        if init_image is not None and strength < 1.0:
            timesteps = img2img_schedule(steps, strength)
            x0 = self.encode_image(init_image, width, height)
            x = timesteps[0] * x + (1.0 - timesteps[0]) * x0
        else:
            timesteps = None
        inp = prepare(x, txt.to(self.device, self.dtype), vec.to(self.device, self.dtype))
        if timesteps is None:
            timesteps = get_schedule(steps, inp['img'].shape[1], shift=False)  # schnell: no shift
        x = denoise(self.transformer, **inp, timesteps=timesteps, callback=callback)
        x = unpack(x.float(), height, width).to(self.dtype)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.timings.sample_s += time.perf_counter() - t

        t = time.perf_counter()
        image = self.vae.decode(x).float().clamp(-1, 1)
        image = ((image[0] + 1.0) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        self.timings.decode_s += time.perf_counter() - t
        return image

    @torch.no_grad()
    def encode_image(self, image: np.ndarray, width: int, height: int) -> torch.Tensor:
        """HxWx3 uint8 -> latent (1x16xH/8xW/8) at width x height."""
        from PIL import Image
        if image.shape[1] != width or image.shape[0] != height:
            image = np.array(Image.fromarray(image).resize((width, height), Image.LANCZOS))
        x = torch.from_numpy(image).to(self.device, torch.float32).permute(2, 0, 1)[None] / 127.5 - 1.0
        return self.vae.encode(x.to(self.dtype))

    @torch.no_grad()
    def refine_tiled(self, prompt: str, image: np.ndarray, seed: int, strength: float = 0.35, tile: int = 1024,
                     overlap: int = 128, steps: int = DEFAULT_STEPS, callback=None) -> np.ndarray:
        """Image-to-image over overlapping tiles (keeps memory flat at large
        sizes), blended with feathered seams. Returns an image of the same size
        (rounded down to multiples of 16)."""
        height, width = 16 * (image.shape[0] // 16), 16 * (image.shape[1] // 16)
        image = image[:height, :width]
        tile_h, tile_w = min(tile, height), min(tile, width)
        ys, xs = tile_starts(height, tile_h, overlap), tile_starts(width, tile_w, overlap)
        acc = np.zeros((height, width, 3), dtype=np.float32)
        weight = np.zeros((height, width, 1), dtype=np.float32)
        total, done = len(ys) * len(xs), 0
        for y in ys:
            for x in xs:
                def on_step(step, steps_, done=done):
                    if callback is not None:
                        callback(done * steps_ + step, total * steps_)
                out = self.generate(prompt, tile_w, tile_h, seed + done, steps=steps, callback=on_step,
                                    init_image=image[y:y + tile_h, x:x + tile_w], strength=strength)
                w = feather_weights(tile_h, tile_w, overlap, y > 0, y + tile_h < height, x > 0,
                                    x + tile_w < width)[..., None]
                acc[y:y + tile_h, x:x + tile_w] += out.astype(np.float32) * w
                weight[y:y + tile_h, x:x + tile_w] += w
                done += 1
        return np.clip(acc / np.maximum(weight, 1e-6), 0, 255).round().astype(np.uint8)
