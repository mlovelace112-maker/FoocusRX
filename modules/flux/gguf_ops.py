# Dequantization ported from city96/ComfyUI-GGUF @ 6ea2651e7df66d7585f6ffee804b20e92fb38b8a
# (dequant.py, loader.py). (c) City96, Apache-2.0 (see modules/flux/NOTICE.md).
# Changes: only the quant types FLUX.1-schnell Q4_K_S uses (Q4_K, F16, F32), a
# plain nn.Module instead of a tensor subclass, and loading for modules/flux.
"""GGUF weights for FLUX.1, kept quantized in memory.

The 12B transformer is ~24 GB in bf16 but ~6.8 GB as Q4_K. GGUFLinear keeps
its weight as the raw Q4_K blocks on the device and dequantizes it only for
the duration of its own forward pass.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

QK_K = 256
K_SCALE_SIZE = 12


def _qtypes():
    import gguf
    return gguf.GGMLQuantizationType


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))


def dequantize_blocks_Q4_K(blocks, dtype):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize(raw: torch.Tensor, qtype, shape, dtype=torch.bfloat16) -> torch.Tensor:
    """Raw GGUF tensor data -> weight of ``shape`` (torch order) in ``dtype``."""
    Q = _qtypes()
    if qtype == Q.Q4_K:
        import gguf
        _block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        rows = raw.reshape((-1, raw.shape[-1])).view(torch.uint8)
        blocks = rows.reshape((rows.numel() // type_size, type_size))
        return dequantize_blocks_Q4_K(blocks, dtype).reshape(shape)
    if qtype in (Q.F32, Q.F16):
        return raw.reshape(shape).to(dtype)
    raise NotImplementedError(f'GGUF quant type {getattr(qtype, "name", qtype)} is not supported')


class GGUFLinear(nn.Module):
    """nn.Linear whose weight stays quantized until forward()."""

    def __init__(self, raw: torch.Tensor, qtype, out_features: int, in_features: int, bias: torch.Tensor | None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.qtype = qtype
        self.register_buffer('raw', raw, persistent=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x):
        weight = dequantize(self.raw, self.qtype, (self.out_features, self.in_features), dtype=x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, qtype={self.qtype.name}'


def read_gguf(path: str) -> dict:
    """name -> (raw tensor on CPU (memory-mapped), qtype, torch-order shape)."""
    import gguf
    reader = gguf.GGUFReader(path)
    out = {}
    for t in reader.tensors:
        shape = tuple(int(v) for v in reversed(t.shape))
        raw = torch.from_numpy(np.asarray(t.data))
        out[t.name] = (raw, t.tensor_type, shape)
    return out


def load_into(model: nn.Module, tensors: dict, device, dtype=torch.bfloat16) -> nn.Module:
    """Fill a meta-device ``model`` from GGUF ``tensors``: quantized Linear
    weights become GGUFLinear (kept quantized); everything else is loaded
    as ``dtype``. Raises if any weight is missing or left over."""
    Q = _qtypes()
    used = set()
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        key = f'{name}.weight'
        raw, qtype, shape = tensors[key]
        if qtype in (Q.F32, Q.F16):
            continue  # plain weight: loaded below with the other parameters
        bias = None
        if module.bias is not None:
            b_raw, b_type, b_shape = tensors[f'{name}.bias']
            bias = dequantize(b_raw, b_type, b_shape, dtype).to(device)
            used.add(f'{name}.bias')
        replacement = GGUFLinear(raw.to(device), qtype, shape[0], shape[1], bias)
        parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
        setattr(parent, name.rsplit('.', 1)[-1], replacement)
        used.add(key)

    state = {}
    for key, (raw, qtype, shape) in tensors.items():
        if key in used:
            continue
        state[key] = dequantize(raw, qtype, shape, dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # GGUFLinear biases were loaded when the layer was replaced (in ``used``).
    missing = [k for k in missing if k not in used and not k.endswith('.raw')]
    if missing or unexpected:
        raise RuntimeError(f'GGUF weights do not match the model: missing={missing[:5]} unexpected={unexpected[:5]}')
    return model.to(device).eval().requires_grad_(False)
