#!/usr/bin/env python3
"""Download Krea 2 Turbo and/or MOSS-Image SFT, plus the stable-diffusion.cpp
engine that runs them (Apple Silicon build).

    venv/bin/python scripts/download_sdcpp.py krea2        # ~13.4 GB
    venv/bin/python scripts/download_sdcpp.py moss_image   # ~12.4 GB
    venv/bin/python scripts/download_sdcpp.py all
    venv/bin/python scripts/download_sdcpp.py engine       # engine only (~40 MB)

Uses the app's downloader: resumes nothing half-done (a .partial is only
renamed once complete) and verifies each file's size and SHA-256 (as
published by Hugging Face / GitHub) before accepting it. Files already present
and verified are skipped. Afterwards click Refresh All Files (or restart) and
pick the model under Model family.

Krea 2 is released under the Krea 2 Community License, which you must accept
before downloading (the script shows the link and asks; pass
--accept-krea2-license to accept non-interactively). The GGUF comes from a
community quantisation of the official weights, so Hugging Face's own gate on
krea/Krea-2-Turbo does not ask for you. MOSS-Image, Qwen3-VL and Qwen3.5 are
Apache-2.0; stable-diffusion.cpp is MIT.
"""
import argparse
import os
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gb(n):
    return f'{n / 1e9:.1f} GB'


def install_engine(config, load_file_from_url, registry):
    from modules.sdcpp.app import engine_installed, engine_path
    target = config.path_sdcpp
    stamp = os.path.join(target, 'VERSION')
    if engine_installed(target) and os.path.isfile(stamp) and open(stamp).read().strip() == registry.ENGINE_VERSION:
        print(f'OK  stable-diffusion.cpp {registry.ENGINE_VERSION} (already installed)')
        return
    if sys.platform != 'darwin':
        raise SystemExit('The bundled engine is the macOS arm64 build. On other systems, build or download '
                         f'stable-diffusion.cpp {registry.ENGINE_VERSION} and put sd-cli (and its libraries) in {target}.')
    print(f'Downloading stable-diffusion.cpp {registry.ENGINE_VERSION} into {target}')
    with tempfile.TemporaryDirectory() as tmp:
        archive = load_file_from_url(registry.ENGINE_URL, model_dir=tmp, file_name='sd-cpp.zip',
                                     expected_size=registry.ENGINE_SIZE, expected_sha256=registry.ENGINE_SHA256)
        unpacked = os.path.join(tmp, 'unpacked')
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():  # refuse absolute paths / ".." entries
                dest = os.path.realpath(os.path.join(unpacked, member))
                if not dest.startswith(os.path.realpath(unpacked) + os.sep):
                    raise SystemExit(f'Unsafe path in engine archive: {member}')
            zf.extractall(unpacked)
        os.makedirs(target, exist_ok=True)
        for name in os.listdir(unpacked):
            src = os.path.join(unpacked, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(target, name))
    for name in (registry.ENGINE_BINARY, 'sd-server'):
        path = os.path.join(target, name)
        if os.path.isfile(path):
            os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    with open(stamp, 'w') as fh:
        fh.write(registry.ENGINE_VERSION + '\n')
    print(f'OK  {engine_path(target)}')


def accept_license(spec, assume_yes: bool) -> bool:
    if not spec.license_requires_acceptance:
        return True
    print(f'\n{spec.family} is licensed under the {spec.license_name}:\n  {spec.license_url}\n'
          'Read it before continuing. Among other terms it sets limits on commercial use and asks deployers '
          'to filter content.')
    if assume_yes:
        print('Accepted with --accept-krea2-license.')
        return True
    if not sys.stdin.isatty():
        print('Not accepted (no terminal to ask). Re-run with --accept-krea2-license to accept.')
        return False
    return input('Do you accept this license? [y/N] ').strip().lower() in ('y', 'yes')


def download_model(spec, config, load_file_from_url):
    target = getattr(config, spec.config_key)
    print(f'\nDownloading {spec.family} (~{gb(spec.download_size)}) into {target}')
    for f in spec.files:
        path = load_file_from_url(f.url, model_dir=target, file_name=f.name, expected_size=f.size,
                                  expected_sha256=f.sha256)
        print(f'OK  {os.path.basename(path)}')
    print(f'{spec.family} ready: pick "{spec.file("diffusion").name}" under Model family "{spec.family}".')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('what', choices=['krea2', 'moss_image', 'all', 'engine'])
    parser.add_argument('--accept-krea2-license', action='store_true')
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.argv = sys.argv[:1]  # args_manager parses sys.argv on import
    import modules.config as config
    from modules.model_loader import load_file_from_url
    from modules.sdcpp import registry

    specs = list(registry.MODELS) if args.what == 'all' else \
        [] if args.what == 'engine' else [registry.spec_by_key(args.what)]
    specs = [s for s in specs if accept_license(s, args.accept_krea2_license)]
    if args.what not in ('engine',) and not specs:
        return 1
    install_engine(config, load_file_from_url, registry)
    for spec in specs:
        download_model(spec, config, load_file_from_url)
    print('\nDone. Click Refresh All Files in FoocusRX (or restart it).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
