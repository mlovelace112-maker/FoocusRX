#!/usr/bin/env python3
"""Download the FLUX.1 [schnell] Q4 files (~12.3 GB) into path_flux.

    venv/bin/python scripts/download_flux.py

Uses the app's downloader: resumes interrupted downloads and verifies each
file's size and SHA-256 (as published by Hugging Face) before accepting it.
Files that are already present and verified are skipped. Restart FoocusRX
afterwards; "flux1-schnell-Q4_K_S.gguf" then appears under Base Model.

Licences: FLUX.1 [schnell] weights and the VAE are Apache-2.0 (Black Forest
Labs); T5-XXL is Apache-2.0 (Google); CLIP-L is MIT (OpenAI). The VAE comes
from a public mirror because the official repository now requires a
Hugging Face login; its size matches the official listing and its SHA-256
is the one published for the FLUX.1 VAE.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [  # (url, file name, size in bytes, sha256)
    ('https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_K_S.gguf',
     'flux1-schnell-Q4_K_S.gguf', 6783943712,
     '4fd16477b3a5296d0cf722c4b92a9fd7f30d09ac7495826e4465d8de9c9fd973'),
    ('https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors',
     't5xxl_fp8_e4m3fn.safetensors', 4893934904,
     '7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09'),
    ('https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors',
     'clip_l.safetensors', 246144152,
     '660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd'),
    ('https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors',
     'ae.safetensors', 335304388,
     'afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38'),
]


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.argv = sys.argv[:1]  # args_manager parses sys.argv on import
    import modules.config as config
    from modules.model_loader import load_file_from_url

    target = config.path_flux
    print(f'Downloading FLUX.1 schnell (~12.3 GB) into {target}')
    for url, name, size, sha in FILES:
        path = load_file_from_url(url, model_dir=target, file_name=name, expected_size=size, expected_sha256=sha)
        print(f'OK  {os.path.basename(path)}')
    print('Done. Restart FoocusRX and pick "flux1-schnell-Q4_K_S.gguf" as the Base Model.')
    try:
        import gguf  # noqa: F401
        import sentencepiece  # noqa: F401
    except ImportError:
        print('Note: install the Python packages too: venv/bin/python -m pip install gguf==0.10.0 sentencepiece==0.2.0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
