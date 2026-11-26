"""LoRAs for the FLUX.1 transformer (modules/flux/model.py).

The transformer's weights stay GGUF-quantized, so LoRAs are not merged into
them. Each targeted Linear gets a forward hook that adds the low-rank term
``scale * (x @ A.T) @ B.T``, optionally into a slice of the output: diffusers
LoRAs train q, k and v (and the single blocks' MLP input) as separate layers,
while this model fuses them into one Linear.

Formats read:
  * kohya / ComfyUI:   lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight (+ .lora_up, .alpha)
  * BFL / ComfyUI PEFT: [diffusion_model.]double_blocks.0.img_attn.qkv.lora_A.weight (+ lora_B, alpha)
  * diffusers:         [transformer.]transformer_blocks.0.attn.to_q.lora_A.weight (+ lora_B)
"""
import os
import re
from dataclasses import dataclass

import torch

HIDDEN = 3072
MLP = 12288


@dataclass
class LoraTerm:
    down: torch.Tensor          # A: (rank, in_features)
    up: torch.Tensor            # B: (out_slice, rank)
    scale: float                # alpha / rank (1 when no alpha is stored)
    start: int = 0              # where the term lands in the module's output
    end: int | None = None


def _diffusers_map(name: str):
    """diffusers module name (prefix stripped) -> (model module, out start, out end, swap halves)."""
    m = re.match(r'transformer_blocks\.(\d+)\.(.+)$', name)
    if m:
        b, rest = f'double_blocks.{m[1]}', m[2]
        qkv = {'attn.to_q': 0, 'attn.to_k': 1, 'attn.to_v': 2}
        add_qkv = {'attn.add_q_proj': 0, 'attn.add_k_proj': 1, 'attn.add_v_proj': 2}
        if rest in qkv:
            i = qkv[rest]
            return f'{b}.img_attn.qkv', i * HIDDEN, (i + 1) * HIDDEN, False
        if rest in add_qkv:
            i = add_qkv[rest]
            return f'{b}.txt_attn.qkv', i * HIDDEN, (i + 1) * HIDDEN, False
        simple = {'attn.to_out.0': 'img_attn.proj', 'attn.to_add_out': 'txt_attn.proj',
                  'ff.net.0.proj': 'img_mlp.0', 'ff.net.2': 'img_mlp.2',
                  'ff_context.net.0.proj': 'txt_mlp.0', 'ff_context.net.2': 'txt_mlp.2',
                  'norm1.linear': 'img_mod.lin', 'norm1_context.linear': 'txt_mod.lin'}
        if rest in simple:
            return f'{b}.{simple[rest]}', 0, None, False
        return None
    m = re.match(r'single_transformer_blocks\.(\d+)\.(.+)$', name)
    if m:
        b, rest = f'single_blocks.{m[1]}', m[2]
        slices = {'attn.to_q': (0, HIDDEN), 'attn.to_k': (HIDDEN, 2 * HIDDEN), 'attn.to_v': (2 * HIDDEN, 3 * HIDDEN),
                  'proj_mlp': (3 * HIDDEN, 3 * HIDDEN + MLP)}
        if rest in slices:
            return f'{b}.linear1', *slices[rest], False
        if rest == 'proj_out':
            return f'{b}.linear2', 0, None, False
        if rest == 'norm.linear':
            return f'{b}.modulation.lin', 0, None, False
        return None
    top = {'x_embedder': 'img_in', 'context_embedder': 'txt_in', 'proj_out': 'final_layer.linear',
           'time_text_embed.timestep_embedder.linear_1': 'time_in.in_layer',
           'time_text_embed.timestep_embedder.linear_2': 'time_in.out_layer',
           'time_text_embed.text_embedder.linear_1': 'vector_in.in_layer',
           'time_text_embed.text_embedder.linear_2': 'vector_in.out_layer'}
    if name in top:
        return top[name], 0, None, False
    if name == 'norm_out.linear':  # diffusers orders (scale, shift); this model (shift, scale)
        return 'final_layer.adaLN_modulation.1', 0, None, True
    return None


def _split_key(key: str):
    """'<module>.<kind>' -> (module, kind) with kind in down/up/alpha, else None."""
    for suffix, kind in (('.lora_down.weight', 'down'), ('.lora_up.weight', 'up'), ('.lora_A.weight', 'down'),
                         ('.lora_B.weight', 'up'), ('.lora_A', 'down'), ('.lora_B', 'up'), ('.alpha', 'alpha')):
        if key.endswith(suffix):
            return key[:-len(suffix)], kind
    return None


def parse(state: dict, module_names: set) -> tuple[dict, list]:
    """LoRA state dict -> ({model module: [LoraTerm, ...]}, [unused LoRA module names])."""
    kohya = {'lora_unet_' + n.replace('.', '_'): n for n in module_names}
    groups = {}
    for key, value in state.items():
        split = _split_key(key)
        if split:
            groups.setdefault(split[0], {})[split[1]] = value
    terms, unused = {}, []
    for lora_name, parts in groups.items():
        if 'down' not in parts or 'up' not in parts:
            unused.append(lora_name)
            continue
        start, end, swap = 0, None, False
        if lora_name in kohya:
            target = kohya[lora_name]
        else:
            bare = re.sub(r'^(diffusion_model|transformer|model\.diffusion_model)\.', '', lora_name)
            if bare in module_names:
                target = bare
            else:
                mapped = _diffusers_map(bare)
                if mapped is None or mapped[0] not in module_names:
                    unused.append(lora_name)
                    continue
                target, start, end, swap = mapped
        down, up = parts['down'].float(), parts['up'].float()
        if down.dim() > 2:  # conv-style 1x1 LoRA saved with spatial dims
            down, up = down.flatten(1), up.flatten(1)
        if swap:
            half = up.shape[0] // 2
            up = torch.cat([up[half:], up[:half]])
        rank = down.shape[0]
        scale = float(parts['alpha']) / rank if 'alpha' in parts else 1.0
        terms.setdefault(target, []).append(LoraTerm(down, up, scale, start, end))
    return terms, unused


_cache = {}
CACHE_FILES = 6  # parsed LoRAs kept on the CPU (one per LoRA slot is enough)


def load(path: str, module_names: set) -> tuple[dict, list]:
    """Parsed LoRA file (CPU, float32), cached by path, size and mtime."""
    from safetensors.torch import load_file
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime)
    if key not in _cache:
        while len(_cache) >= CACHE_FILES:
            _cache.pop(next(iter(_cache)))
        _cache[key] = parse(load_file(path), module_names)
    return _cache[key]


class LoraHooks:
    """Forward hooks on a model that add LoRA terms; remove() undoes them."""

    def __init__(self):
        self.handles = []
        self.matched = 0

    def apply(self, model, loras: list, device, dtype, log=print):
        """``loras``: [(path, weight), ...]. Returns per-file matched module counts."""
        modules = dict(model.named_modules())
        names = {n for n, m in modules.items() if hasattr(m, 'out_features') or isinstance(m, torch.nn.Linear)}
        combined = {}
        report = []
        for path, weight in loras:
            terms, unused = load(path, names)
            report.append((os.path.basename(path), len(terms), len(unused)))
            if not terms:
                log(f'[FLUX LoRA] {os.path.basename(path)}: no layers matched FLUX.1 ({len(unused)} unknown); skipped')
                continue
            if unused:
                log(f'[FLUX LoRA] {os.path.basename(path)}: {len(unused)} layer(s) not used, e.g. {unused[0]}')
            for target, items in terms.items():
                for t in items:
                    combined.setdefault(target, []).append(LoraTerm(
                        t.down.to(device, dtype), t.up.to(device, dtype), t.scale * float(weight), t.start, t.end))
        for target, items in combined.items():
            self.handles.append(modules[target].register_forward_hook(_make_hook(items)))
        self.matched = len(combined)
        return report

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.matched = 0


def _make_hook(items):
    def hook(_module, inputs, output):
        x = inputs[0]
        for t in items:
            delta = (x @ t.down.T) @ t.up.T
            if t.scale != 1.0:
                delta = delta * t.scale
            if t.start == 0 and (t.end is None or t.end == output.shape[-1]):
                output = output + delta
            else:  # inference only (no autograd), so in place is safe
                output[..., t.start:t.end] += delta
        return output
    return hook
