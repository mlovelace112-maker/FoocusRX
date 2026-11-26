"""Models that run through stable-diffusion.cpp (sd-cli), and the engine itself.

Pure data, no heavy imports: ui_family, the downloader and the tests read it.

Each model is a diffusion transformer GGUF plus a GGUF LLM text encoder, a VAE
and (MOSS-Image) a tokenizer. Sampling defaults follow the upstream references:
  * Krea 2 Turbo: 8 steps, CFG 1 (distilled), euler + simple scheduler
    (Comfy-Org template image_krea2_turbo_t2i.json).
  * MOSS-Image SFT: 50 steps, CFG 3, flow-matching euler, shift 5, negative
    branch skips layer 22 (OpenMOSS model card; sd-cli applies shift 5 and the
    layer skip itself for this family).
"""
import os
import re
from dataclasses import dataclass, field

# Pinned engine build: https://github.com/leejet/stable-diffusion.cpp/releases
# (Krea2 since master-721, MOSS-Image since master-975). SHA-256 is the digest
# GitHub publishes for the release asset.
ENGINE_VERSION = 'master-1133-52491bd'
ENGINE_URL = ('https://github.com/leejet/stable-diffusion.cpp/releases/download/master-1133-52491bd/'
              'sd-master-52491bd-bin-Darwin-macOS-26.7.2-arm64.zip')
ENGINE_SIZE = 39900245
ENGINE_SHA256 = '083968d8cd095c221041ad5fce7a940b0fddf3e9ddb785bbecf67dabe28c2fec'
ENGINE_BINARY = 'sd-cli'


@dataclass(frozen=True)
class FileSpec:
    role: str        # 'diffusion' | 'llm' | 'vae' | 'tokenizer'
    url: str
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    key: str                     # short id, used in metadata and the downloader
    family: str                  # name shown in the Model family dropdown
    config_key: str              # modules.config path key
    default_dir: str
    name_pattern: str            # a diffusion GGUF in default_dir whose name matches belongs here
    files: tuple
    steps: int
    cfg: float
    sampler: str = 'euler'
    scheduler: str = ''          # '' = sd-cli's model default
    uses_negative: bool = False  # only meaningful with CFG > 1
    license_name: str = ''
    license_url: str = ''
    license_requires_acceptance: bool = False
    notice: str = ''
    extra_args: tuple = field(default_factory=tuple)

    def file(self, role: str):
        return next((f for f in self.files if f.role == role), None)

    def support_files(self) -> list:
        return [f for f in self.files if f.role != 'diffusion']

    def matches(self, model_name) -> bool:
        name = os.path.basename(str(model_name or ''))
        return name.lower().endswith('.gguf') and re.search(self.name_pattern, name, re.IGNORECASE) is not None

    @property
    def download_size(self) -> int:
        return sum(f.size for f in self.files)


KREA2 = ModelSpec(
    key='krea2',
    family='Krea 2',
    config_key='path_krea2',
    default_dir='../models/krea2/',
    name_pattern=r'^krea[-_ ]?2',
    files=(
        FileSpec('diffusion', 'https://huggingface.co/realrebelai/KREA-2_GGUFs/resolve/main/TURBO/Krea-2-Turbo-Q5_K_S.gguf',
                 'Krea-2-Turbo-Q5_K_S.gguf', 8819266656,
                 '2d9a6bfb1b9ef512b040af72b59ce8c4a564f834a083747f3a6e7d3781e8b6dd'),
        FileSpec('llm', 'https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/Qwen3VL-4B-Instruct-Q8_0.gguf',
                 'Qwen3VL-4B-Instruct-Q8_0.gguf', 4280406144,
                 '054721f478bc5fa6beffb7f38eae575d45298f88cbb8d2f83ef675a727863eb1'),
        FileSpec('vae', 'https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors',
                 'qwen_image_vae.safetensors', 253806246,
                 'a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f'),
    ),
    steps=8, cfg=1.0, sampler='euler', scheduler='simple', uses_negative=False,
    license_name='Krea 2 Community License',
    license_url='https://huggingface.co/krea/Krea-2-Turbo/blob/main/LICENSE.pdf',
    license_requires_acceptance=True,
    notice=('**Krea 2 Turbo** (12.9B, 8 steps): natural-language prompts; describe the scene in full sentences. '
            'Uses your prompt, styles (as text), wildcards, aspect ratio, seed and image number. SDXL-only controls '
            '(negative prompt, performance, CFG, sampler, refiner, input image, enhance) are hidden. LoRAs made for Krea 2 work (Advanced > Models).'),
)

MOSS_IMAGE = ModelSpec(
    key='moss_image',
    family='MOSS-Image',
    config_key='path_moss_image',
    default_dir='../models/moss_image/',
    name_pattern=r'^moss[-_ ]?image',
    files=(
        FileSpec('diffusion', 'https://huggingface.co/unsloth/MOSS-Image-SFT-GGUF/resolve/main/MOSS-Image-SFT-Q8_0.gguf',
                 'MOSS-Image-SFT-Q8_0.gguf', 5482213536,
                 'f18beeb8c6101a3edaff379acce2bdf9fcc1272dfb4c40530c9070f7a0650458'),
        FileSpec('llm', 'https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q5_K_M.gguf',
                 'Qwen3.5-9B-Q5_K_M.gguf', 6577841376,
                 'dc2a39aef291f91a9116ad214058da0d86eb648743a124bd8c333787c4b9c91c'),
        FileSpec('vae', 'https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae.safetensors',
                 'flux2-vae.safetensors', 336211292,
                 '868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3'),
        FileSpec('tokenizer', 'https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/resolve/main/tokenizer/tokenizer.json',
                 'tokenizer.json', 19989325,
                 '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'),
    ),
    steps=50, cfg=3.0, sampler='euler', scheduler='', uses_negative=True,
    # EasyCache reuses the transformer output on steps where it barely changes:
    # 27/50 steps skipped, sampling 493 s -> 230 s at 1024x1024 on an M5 Pro,
    # with the multi-line sign text still exact and the image nearly identical.
    extra_args=('--cache-mode', 'easycache'),
    license_name='Apache-2.0',
    license_url='https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/main/LICENSE',
    notice=('**MOSS-Image SFT** (5B, 50 steps, CFG 3): strong at on-image text and layout. Line breaks in the prompt '
            'are kept, so multi-line sign or poster text works, and a JSON layout prompt is passed through unchanged '
            '(styles are not applied to it). Uses your prompt, negative prompt, styles, wildcards, aspect ratio, seed '
            'and image number; Debug > "Forced Overwrite of Sampling Step" changes the step count. About 5 min per image '
            '(two transformer passes per step; about half the steps are reused from a cache).'),
)

MODELS = (KREA2, MOSS_IMAGE)
FAMILIES = tuple(m.family for m in MODELS)


def spec_for(model_name):
    """The ModelSpec a base model name belongs to, or None."""
    for spec in MODELS:
        if spec.matches(model_name):
            return spec
    return None


def spec_by_key(key):
    return next((m for m in MODELS if m.key == key), None)


def spec_by_family(family):
    return next((m for m in MODELS if m.family == family), None)
