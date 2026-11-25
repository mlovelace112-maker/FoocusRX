"""FLUX.1 text encoders: T5-XXL (sequence context) and CLIP-L (pooled vector).

Mirrors HFEmbedder in black-forest-labs/flux (src/flux/modules/conditioner.py):
T5 last_hidden_state with max_length 256 for schnell, CLIP pooler_output with
max_length 77, both padded to max length with no attention mask.

Everything loads from local files: weights from models/flux/, the CLIP
tokenizer/config bundled in ldm_patched, the T5 tokenizer in t5_tokenizer/.
T5 ships as fp8 (unsupported on MPS) and is upcast to bf16 (fp16 overflows
in T5). Models are built on the meta device and filled with assign=True, so
loading never holds an extra fp32 copy (~19 GB for T5-XXL).
"""
import json
import os

import torch
from safetensors import safe_open

_HERE = os.path.dirname(os.path.abspath(__file__))
_LDM = os.path.join(os.path.dirname(_HERE), '..', 'ldm_patched', 'modules')
T5_MAX_LENGTH_SCHNELL = 256
CLIP_MAX_LENGTH = 77

# google/t5-v1_1-xxl encoder (same values as ComfyUI's t5_config_xxl.json).
T5_XXL_CONFIG = dict(
    vocab_size=32128, d_model=4096, d_kv=64, d_ff=10240, num_layers=24, num_heads=64,
    relative_attention_num_buckets=32, relative_attention_max_distance=128,
    layer_norm_epsilon=1e-6, feed_forward_proj='gated-gelu', dropout_rate=0.0,
    is_encoder_decoder=False, use_cache=False, pad_token_id=0, eos_token_id=1,
)


def _load_weights(path: str, dtype: torch.dtype) -> dict:
    """Read a .safetensors file, converting each tensor to ``dtype`` as it is
    read so only one full-precision copy ever exists."""
    out = {}
    with safe_open(path, framework='pt', device='cpu') as fh:
        for key in fh.keys():
            out[key] = fh.get_tensor(key).to(dtype)
    return out


def _assign(model: torch.nn.Module, state: dict, device) -> torch.nn.Module:
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # Non-persistent buffers (e.g. CLIP position_ids) aren't in checkpoints.
    missing = [k for k in missing if not k.endswith('position_ids')]
    if missing or unexpected:
        raise RuntimeError(f'weights do not match the model: missing={missing[:5]} unexpected={unexpected[:5]}')
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:  # recreate buffers the checkpoint doesn't carry
            module = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
            leaf = name.rsplit('.', 1)[-1]
            if leaf == 'position_ids':
                module.register_buffer(leaf, torch.arange(buf.shape[-1]).expand(buf.shape), persistent=False)
    return model.to(device).eval().requires_grad_(False)


class T5Encoder:
    def __init__(self, weights_path: str, device, dtype=torch.bfloat16, max_length=T5_MAX_LENGTH_SCHNELL):
        from transformers import T5Config, T5EncoderModel, T5TokenizerFast
        self.max_length = max_length
        self.device = device
        self.tokenizer = T5TokenizerFast(tokenizer_file=os.path.join(_HERE, 't5_tokenizer', 'tokenizer.json'),
                                         eos_token='</s>', pad_token='<pad>', unk_token='<unk>', extra_ids=0)
        with torch.device('meta'):
            model = T5EncoderModel(T5Config(**T5_XXL_CONFIG))
        self.model = _assign(model, _load_weights(weights_path, dtype), device)

    @torch.no_grad()
    def __call__(self, prompts: list[str]) -> torch.Tensor:
        ids = self.tokenizer(prompts, truncation=True, max_length=self.max_length, padding='max_length',
                             return_tensors='pt')['input_ids'].to(self.device)
        return self.model(input_ids=ids, attention_mask=None, output_hidden_states=False).last_hidden_state


class ClipEncoder:
    def __init__(self, weights_path: str, device, dtype=torch.bfloat16):
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
        self.device = device
        tok_dir = os.path.join(_LDM, 'sd1_tokenizer')
        self.tokenizer = CLIPTokenizer(os.path.join(tok_dir, 'vocab.json'), os.path.join(tok_dir, 'merges.txt'))
        with open(os.path.join(_LDM, 'sd1_clip_config.json')) as fh:
            config = CLIPTextConfig(**{k: v for k, v in json.load(fh).items() if k not in ('architectures', '_name_or_path')})
        with torch.device('meta'):
            model = CLIPTextModel(config)
        self.model = _assign(model, _load_weights(weights_path, dtype), device)

    @torch.no_grad()
    def __call__(self, prompts: list[str]) -> torch.Tensor:
        ids = self.tokenizer(prompts, truncation=True, max_length=CLIP_MAX_LENGTH, padding='max_length',
                             return_tensors='pt')['input_ids'].to(self.device)
        return self.model(input_ids=ids, attention_mask=None, output_hidden_states=False).pooler_output
