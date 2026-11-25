#!/usr/bin/env python3
"""Download the AI prompt rewriter's language model (~3.1 GB) into path_prompt_model.

    venv/bin/python scripts/download_prompt_model.py

Qwen2.5-1.5B-Instruct (Apache-2.0, Alibaba Cloud), into
<path_prompt_model>/Qwen2.5-1.5B-Instruct/. Uses the app's downloader, which
resumes interrupted downloads; the weights are verified against the SHA-256
Hugging Face publishes. Already-verified files are skipped.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/'
MODEL_DIR_NAME = 'Qwen2.5-1.5B-Instruct'
FILES = [  # (file, size, sha256 or None for small config files)
    ('model.safetensors', 3087467144, 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'),
    ('config.json', None, None),
    ('generation_config.json', None, None),
    ('tokenizer.json', None, None),
    ('tokenizer_config.json', None, None),
    ('vocab.json', None, None),
    ('merges.txt', None, None),
    ('LICENSE', None, None),
]


def main():
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.argv = sys.argv[:1]  # args_manager parses sys.argv on import
    import modules.config as config
    from modules.model_loader import load_file_from_url

    target = os.path.join(config.path_prompt_model, MODEL_DIR_NAME)
    print(f'Downloading {MODEL_DIR_NAME} (~3.1 GB) into {target}')
    for name, size, sha in FILES:
        path = load_file_from_url(REPO + name, model_dir=target, file_name=name,
                                  expected_size=size, expected_sha256=sha)
        print(f'OK  {os.path.basename(path)}')
    print('Done. Restart FoocusRX to use "Rewrite prompt".')
    return 0


if __name__ == '__main__':
    sys.exit(main())
