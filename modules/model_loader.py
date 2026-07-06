import os
import hashlib
from urllib.parse import urlparse
from typing import Optional


def _sha256(path: str, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        expected_size: Optional[int] = None,
) -> str:
    """Download a file from ``url`` into ``model_dir``, using the existing
    file if it looks intact.

    FoocusRX changes vs upstream:

    * Downloads land in ``<target>.partial`` and are only renamed to the final
      path after the download completes. Interrupted downloads (Ctrl-C, Wi-Fi
      drop, laptop sleep) no longer leave a corrupt file in the model dir that
      every subsequent launch happily loads.
    * If ``expected_sha256`` or ``expected_size`` is supplied, the file is
      validated before being accepted. On mismatch, the file is quarantined
      as ``<target>.corrupt`` and a fresh download is attempted.
    * If a stale ``.partial`` from a previous run exists on entry, it's
      removed so we don't attempt to reuse it (torch.hub doesn't support
      resume anyway).

    Returns the path to the downloaded file.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip("/")
    url = url.replace("https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)

    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)

    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    partial_file = cached_file + ".partial"

    def _validate(path: str) -> Optional[str]:
        """Return an error string if ``path`` fails validation, else None."""
        if expected_size is not None:
            actual_size = os.path.getsize(path)
            if actual_size != expected_size:
                return (
                    f"size mismatch (expected {expected_size} bytes, "
                    f"got {actual_size})"
                )
        if expected_sha256 is not None:
            actual_sha = _sha256(path)
            if actual_sha.lower() != expected_sha256.lower():
                return f"sha256 mismatch (expected {expected_sha256}, got {actual_sha})"
        return None

    # If the final file already exists, validate it. If it fails, quarantine
    # and re-download; if there's no expected hash/size we trust it (matches
    # upstream behavior for the common case).
    if os.path.exists(cached_file):
        err = _validate(cached_file)
        if err is None:
            return cached_file
        quarantine = cached_file + ".corrupt"
        print(
            f'Cached file failed validation ({err}); '
            f'moving to "{quarantine}" and re-downloading.'
        )
        try:
            if os.path.exists(quarantine):
                os.remove(quarantine)
            os.rename(cached_file, quarantine)
        except OSError as e:
            print(f"  (couldn't quarantine, deleting instead: {e})")
            os.remove(cached_file)

    # Clean up any leftover .partial from a previous interrupted run.
    if os.path.exists(partial_file):
        try:
            os.remove(partial_file)
        except OSError:
            pass

    print(f'Downloading: "{url}" to {cached_file}')
    from torch.hub import download_url_to_file
    download_url_to_file(url, partial_file, progress=progress)

    err = _validate(partial_file)
    if err is not None:
        quarantine = cached_file + ".corrupt"
        try:
            os.rename(partial_file, quarantine)
        except OSError:
            pass
        raise RuntimeError(
            f'Downloaded file failed validation: {err}. '
            f'Quarantined at "{quarantine}".'
        )

    # Atomic rename onto the final path. os.replace is atomic on POSIX and
    # Windows (Python 3.3+), which prevents a partial file from ever being
    # visible under its final name.
    os.replace(partial_file, cached_file)
    return cached_file
