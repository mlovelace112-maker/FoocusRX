"""Model manager: list installed model files and download new ones by URL.

Downloads go through modules.model_loader.load_file_from_url, so they
resume after interruptions, are verified against an optional SHA-256 and
land atomically (a half-finished file never appears under its final name).
Only .safetensors files can be downloaded: pickle-based formats (.ckpt,
.pt, .bin) can run arbitrary code when loaded.
"""
import datetime
import os
import re
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

MODEL_EXTENSIONS = ('.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf')
DOWNLOAD_EXTENSIONS = ('.safetensors',)
KINDS = ('Checkpoint', 'LoRA', 'VAE', 'Embedding')
_SHA256_RE = re.compile(r'^[0-9a-fA-F]{64}$')
_SAFE_NAME_RE = re.compile(r'^[\w.\- ()+\[\]]+$')


@dataclass
class DownloadRequest:
    url: str
    kind: str
    model_dir: str
    file_name: str
    sha256: str | None


def default_dirs() -> dict:
    """Where each kind of model is installed (first configured folder)."""
    import modules.config as config
    return {
        'Checkpoint': config.paths_checkpoints[0],
        'LoRA': config.paths_loras[0],
        'VAE': config.path_vae,
        'Embedding': config.path_embeddings,
    }


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return f'{size:.0f} {unit}' if unit == 'B' else f'{size:.1f} {unit}'
        size /= 1024


def list_installed(dirs: dict) -> list:
    """Rows of [kind, file (relative to its folder), size, modified], sorted."""
    rows = []
    for kind in KINDS:
        root = dirs.get(kind)
        if not root or not os.path.isdir(root):
            continue
        for current, _subdirs, files in os.walk(root):
            for name in files:
                if not name.lower().endswith(MODEL_EXTENSIONS):
                    continue
                path = os.path.join(current, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                modified = datetime.datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M')
                rows.append([kind, os.path.relpath(path, root), human_size(st.st_size), modified])
    rows.sort(key=lambda r: (KINDS.index(r[0]), r[1].lower()))
    return rows


def prepare_download(url: str, kind: str, file_name: str | None, sha256: str | None, dirs: dict) -> DownloadRequest:
    """Validate a download request from the UI. Raises ValueError with a
    message suitable for showing to the user."""
    url = (url or '').strip()
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.netloc:
        raise ValueError('Enter a full https:// URL.')
    if kind not in KINDS:
        raise ValueError(f'Choose a model type: {", ".join(KINDS)}.')

    name = (file_name or '').strip() or unquote(os.path.basename(parsed.path))
    if not name:
        raise ValueError('This URL has no file name in it; enter one in "Save as".')
    if os.path.basename(name) != name or name in ('.', '..') or not _SAFE_NAME_RE.match(name):
        raise ValueError('"Save as" must be a plain file name (letters, numbers, spaces, . _ - ( ) + [ ]).')
    if not name.lower().endswith(DOWNLOAD_EXTENSIONS):
        raise ValueError('Only .safetensors files can be downloaded (other formats can run code when loaded). '
                         'If the URL has no extension, enter a name ending in .safetensors in "Save as".')

    sha = (sha256 or '').strip() or None
    if sha is not None and not _SHA256_RE.match(sha):
        raise ValueError('SHA-256 must be 64 hexadecimal characters (or leave it empty).')

    model_dir = dirs.get(kind)
    if not model_dir:
        raise ValueError(f'No folder is configured for {kind} files.')
    if os.path.exists(os.path.join(model_dir, name)):
        raise ValueError(f'{name} is already installed.')
    return DownloadRequest(url, kind, model_dir, name, sha.lower() if sha else None)


def download(request: DownloadRequest, progress_callback=None) -> str:
    from modules.model_loader import load_file_from_url
    return load_file_from_url(request.url, model_dir=request.model_dir, file_name=request.file_name,
                              expected_sha256=request.sha256, progress=False,
                              progress_callback=progress_callback)


def progress_text(file_name: str, done: int, total: int | None) -> str:
    if total:
        pct = min(100.0, 100.0 * done / total)
        if done >= total:
            return f'Verifying {file_name} …'
        return f'Downloading {file_name}: {human_size(done)} of {human_size(total)} ({pct:.0f}%)'
    return f'Downloading {file_name}: {human_size(done)}'
