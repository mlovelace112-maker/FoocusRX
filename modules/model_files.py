"""What a model file is, from its header: base model, LoRA, text encoder or VAE,
and which model family (SDXL, FLUX.1, Krea 2, MOSS-Image) it belongs to.

Reads only the header of a .safetensors or .gguf file (tensor names and, for
GGUF, the metadata), never the weights, and caches the answer by path, size
and modification time. This lets the app list a model or LoRA wherever it was
saved and whatever it is called, instead of going by folder and file name:

  * a Krea 2 or MOSS-Image model with any name, in its folder or a subfolder,
    as .gguf or .safetensors;
  * a LoRA saved next to the models instead of in the LoRA folder;
  * LoRAs in every naming style (kohya ``lora_unet_...``, PEFT/diffusers,
    ComfyUI/native ``diffusion_model....``), as .safetensors or .gguf.

Pure Python, no torch: modules.config imports it while listing files.
"""
import json
import os
import re
import struct
from dataclasses import dataclass

SDXL = 'SDXL'
FLUX = 'FLUX.1'
KREA2 = 'Krea 2'
MOSS = 'MOSS-Image'
OTHER = 'Other'  # a model or LoRA for an architecture this app doesn't run (Qwen-Image, SD3, Wan, ...)

MODEL = 'model'
LORA = 'lora'
TEXT_ENCODER = 'text_encoder'
VAE = 'vae'
UNKNOWN = 'unknown'

SCAN_EXTENSIONS = ('.safetensors', '.gguf')

# Tensor names are matched with every '_' turned into '.', so one pattern covers
# native (``blocks.0.attn.wq``), kohya (``lora_unet_blocks_0_attn_wq``) and
# diffusers/PEFT (``transformer.transformer_blocks.0.attn.to_q``) spellings.
_LORA_KEY = re.compile(r'(^|\.)lora\.(down|up|mid|a|b)(\.|$)|(^|\.)hada\.w[12]\.[ab](\.|$)|'
                       r'(^|\.)lokr\.w[12](\.|$)|(^|\.)lora\.linear(\.|$)', re.IGNORECASE)

# (family, pattern) - first family with a match wins. Shared by models and LoRAs.
_FAMILY_RULES = (
    # MOSS-Image: layers.N.proj.{q,k,v,qkv,o,ffn_up_gate,ffn_down}
    (MOSS, re.compile(r'(^|\.)layers\.\d+\.proj\.(q|k|v|qkv|o|ffn\.up\.gate|ffn\.down)(\.|$)')),
    # Krea 2, native/ComfyUI names (sd.cpp's own) and diffusers names: gated
    # attention (wq/wk/wv/wo + gate) and a gate/up/down MLP; the text fusion stack.
    (KREA2, re.compile(r'(^|\.)txtfusion\.|(^|\.)text\.fusion\.|(^|\.)tproj\.|(^|\.)txtmlp\.|time\.mod\.proj|'
                       r'(^|\.)blocks\.\d+\.attn\.(wq|wk|wv|wo|gate)(\.|$)|'
                       r'(^|\.)blocks\.\d+\.mlp\.(gate|up|down)(\.|$)|'
                       r'\.attn\.to\.gate(\.|$)|\.ff\.(gate|up|down)(\.|$)')),
    (FLUX, re.compile(r'double\.blocks\.|single\.blocks\.|single\.transformer\.blocks\.')),
    # Qwen-Image (and edit models) share FLUX's add_*_proj names but use
    # img_mod/txt_mod/img_mlp; this app can't run them.
    (OTHER, re.compile(r'(^|\.)(img|txt)\.mod\.|(^|\.)(img|txt)\.mlp\.')),
    (FLUX, re.compile(r'\.attn\.add\.[qkv]\.proj(\.|$)|\.ff\.context\.|\.norm1\.context\.')),
    # SDXL / SD1.5 UNet and CLIP, in LDM, diffusers, kohya and LyCORIS spellings.
    (SDXL, re.compile(r'(^|\.)(input|output)\.blocks\.|(^|\.)middle\.block\.|(^|\.)(down|up)\.blocks\.|'
                      r'(^|\.)mid\.block\.|(^|\.)te[12]?\.|text\.encoder|conditioner\.embedders|'
                      r'(^|\.)text\.model\.encoder')),
)

_TEXT_ENCODER = re.compile(r'(^|\.)blk\.\d+\.|token\.embd|embed\.tokens|(^|\.)model\.layers\.\d+\.|'
                           r'(^|\.)text\.model\.|(^|\.)encoder\.block\.\d+|(^|\.)v\.blk\.|(^|\.)visual\.|'
                           r'(^|\.)shared\.weight')
_VAE = re.compile(r'^(first\.stage\.model\.|vae\.)?(encoder|decoder|quant\.conv|post\.quant\.conv)(\.|$)')

# GGUF general.architecture values of language models (text encoders here).
_LLM_ARCHS = re.compile(r'^(llama|mistral|qwen|gemma|t5|clip|phi|gpt|mllama|glm|internlm|deepseek|smollm)',
                        re.IGNORECASE)
_DIFFUSION_ARCHS = {'krea2': KREA2, 'moss_image': MOSS, 'moss-image': MOSS, 'flux': FLUX, 'sdxl': SDXL}


@dataclass(frozen=True)
class FileInfo:
    kind: str           # MODEL | LORA | TEXT_ENCODER | VAE | UNKNOWN
    family: str = ''    # SDXL | FLUX | KREA2 | MOSS | OTHER for MODEL/LORA, else ''


def _dotted(name: str) -> str:
    return name.replace('_', '.').lower()


def family_from_names(names) -> str:
    """Model family from tensor names ('' when no rule matches)."""
    dotted = [_dotted(n) for n in names]
    for family, pattern in _FAMILY_RULES:
        if any(pattern.search(n) for n in dotted):
            return family
    return ''


def classify_names(names, architecture: str = '') -> FileInfo:
    """What a file is, from its tensor names (and GGUF architecture, if any)."""
    names = list(names)
    if not names and not architecture:
        return FileInfo(UNKNOWN)
    arch = (architecture or '').lower()
    if any(_LORA_KEY.search(_dotted(n)) for n in names):
        return FileInfo(LORA, family_from_names(names) or OTHER)
    if arch and _LLM_ARCHS.match(arch):
        return FileInfo(TEXT_ENCODER)
    family = family_from_names(names)
    if family:
        # A full SDXL checkpoint also holds CLIP and VAE tensors: the UNet decides.
        return FileInfo(MODEL, family)
    if arch in _DIFFUSION_ARCHS:
        return FileInfo(MODEL, _DIFFUSION_ARCHS[arch])
    dotted = [_dotted(n) for n in names]
    if dotted and all(_VAE.search(n) for n in dotted):
        return FileInfo(VAE)
    if any(_TEXT_ENCODER.search(n) for n in dotted):
        return FileInfo(TEXT_ENCODER)
    return FileInfo(UNKNOWN)


# ---------------------------------------------------------------- headers

MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024
MAX_GGUF_METADATA = 256 * 1024 * 1024


def read_safetensors_names(path: str) -> list:
    try:
        with open(path, 'rb') as fh:
            size = struct.unpack('<Q', fh.read(8))[0]
            if size > MAX_SAFETENSORS_HEADER:
                return []
            header = json.loads(fh.read(size))
    except (OSError, ValueError, struct.error):
        return []
    if not isinstance(header, dict):
        return []
    header.pop('__metadata__', None)
    return list(header)


_GGUF_SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
_GGUF_STRING, _GGUF_ARRAY = 8, 9


class _GGUFReader:
    def __init__(self, fh, version: int):
        self.fh = fh
        self.wide = version >= 2  # v1 used 32-bit counts and lengths
        self.read_total = 0

    def _read(self, n: int) -> bytes:
        self.read_total += n
        if self.read_total > MAX_GGUF_METADATA:
            raise ValueError('GGUF metadata too large')
        data = self.fh.read(n)
        if len(data) != n:
            raise ValueError('truncated GGUF')
        return data

    def count(self) -> int:
        return struct.unpack('<Q' if self.wide else '<I', self._read(8 if self.wide else 4))[0]

    def u32(self) -> int:
        return struct.unpack('<I', self._read(4))[0]

    def string(self) -> str:
        return self._read(self.count()).decode('utf-8', errors='replace')

    def skip_string(self):
        n = self.count()
        self.read_total += n
        if self.read_total > MAX_GGUF_METADATA:
            raise ValueError('GGUF metadata too large')
        self.fh.seek(n, os.SEEK_CUR)

    def value(self, kind: int, keep: bool):
        if kind in _GGUF_SCALARS:
            fmt = '<' + _GGUF_SCALARS[kind]
            return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]
        if kind == _GGUF_STRING:
            if keep:
                return self.string()
            self.skip_string()
            return None
        if kind == _GGUF_ARRAY:
            item_kind, n = self.u32(), self.count()
            if item_kind in _GGUF_SCALARS:  # skip fixed-size items in one seek
                size = struct.calcsize('<' + _GGUF_SCALARS[item_kind]) * n
                self.read_total += size
                if self.read_total > MAX_GGUF_METADATA:
                    raise ValueError('GGUF metadata too large')
                self.fh.seek(size, os.SEEK_CUR)
            else:
                for _ in range(n):
                    self.value(item_kind, keep=False)
            return None
        raise ValueError(f'unknown GGUF value type {kind}')


def read_gguf_header(path: str):
    """(general.architecture, tensor names) of a GGUF file; ('', []) on error.

    Stops after the metadata for a language model (its architecture says
    enough) so a text encoder's large tokenizer tables are skipped, not parsed."""
    try:
        with open(path, 'rb') as fh:
            if fh.read(4) != b'GGUF':
                return '', []
            version = struct.unpack('<I', fh.read(4))[0]
            reader = _GGUFReader(fh, version)
            n_tensors, n_kv = reader.count(), reader.count()
            arch = ''
            for _ in range(n_kv):
                key = reader.string()
                kind = reader.u32()
                value = reader.value(kind, keep=(key == 'general.architecture'))
                if key == 'general.architecture' and isinstance(value, str):
                    arch = value
                    if _LLM_ARCHS.match(arch):
                        return arch, []
            names = []
            for _ in range(n_tensors):
                names.append(reader.string())
                n_dims = reader.u32()
                reader._read(8 * n_dims + 4 + 8)  # dims, type, offset
            return arch, names
    except (OSError, ValueError, struct.error, MemoryError):
        return '', []


_cache = {}


def inspect(path: str) -> FileInfo:
    """What the file at ``path`` is (FileInfo(UNKNOWN) when unreadable)."""
    try:
        st = os.stat(path)
    except OSError:
        return FileInfo(UNKNOWN)
    key = (os.path.realpath(path), st.st_size, st.st_mtime)
    if key not in _cache:
        lower = path.lower()
        if lower.endswith('.safetensors'):
            info = classify_names(read_safetensors_names(path))
        elif lower.endswith('.gguf'):
            arch, names = read_gguf_header(path)
            info = classify_names(names, arch)
        else:
            info = FileInfo(UNKNOWN)
        _cache[key] = info
    return _cache[key]


def scan_folder(folder: str, extensions=SCAN_EXTENSIONS) -> list:
    """[(name relative to ``folder``, absolute path), ...] for every file with
    one of ``extensions`` in ``folder`` and its subfolders, sorted by name.
    Unfinished downloads (.partial, .part, .tmp) and hidden files are skipped."""
    out = []
    if not folder or not os.path.isdir(folder):
        return out
    seen = set()
    for root, dirs, files in os.walk(folder, followlinks=True):
        real = os.path.realpath(root)
        if real in seen:  # a symlinked folder that loops back
            dirs[:] = []
            continue
        seen.add(real)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for name in files:
            if name.startswith('.') or not name.lower().endswith(tuple(extensions)):
                continue
            path = os.path.join(root, name)
            out.append((os.path.relpath(path, folder), path))
    return sorted(out, key=lambda item: item[0].casefold())
