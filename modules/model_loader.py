import json
import os
import hashlib
from urllib.parse import urlparse
from typing import Optional

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REGISTRY_PATH = os.path.abspath(os.path.join(_MODULE_DIR, "..", "models", "hashes.json"))


def _load_registry() -> dict:
    """Read models/hashes.json, returning ``{}`` on any failure.

    The registry maps download URLs to ``{"sha256": ..., "size": ...}``.
    Anything missing or malformed degrades to "no known hash" — we
    never want a bad JSON file to block model loading.
    """
    try:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


_HASH_REGISTRY = _load_registry()


def _sha256(path: str, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _stamp_path(cached_file: str) -> str:
    """Sidecar path recording that a given (size, mtime) pair validated.

    Rehashing a 6 GB checkpoint every launch is slow. Once we've validated
    a file we drop a tiny sidecar so subsequent launches can skip the hash
    if size + mtime are unchanged. Any external modification invalidates
    the stamp automatically.
    """
    return cached_file + ".sha256.ok"


def _read_stamp(path: str) -> Optional[dict]:
    try:
        with open(_stamp_path(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_stamp(path: str, sha: str) -> None:
    try:
        st = os.stat(path)
        with open(_stamp_path(path), "w", encoding="utf-8") as f:
            json.dump({"sha256": sha, "size": st.st_size, "mtime": st.st_mtime}, f)
    except OSError:
        # Best effort. Missing stamp just means we rehash next time.
        pass


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

    * If the caller doesn't pass ``expected_sha256``/``expected_size``, the
      URL is looked up in ``models/hashes.json`` (shipped with the repo).
      That gives all ~18 downloaders in ``modules/config.py`` free
      checksum verification without touching each call site. On repeat
      launches, a ``.sha256.ok`` stamp lets us skip rehashing files whose
      size + mtime haven't changed. Set ``FOOOCUS_SKIP_HASH_CHECK=1`` to
      disable hash checks globally (size checks still run).

    Returns the path to the downloaded file.
    """
    # If the caller didn't supply expected values, consult the shipped
    # registry keyed by URL. See docstring above.
    if expected_sha256 is None or expected_size is None:
        entry = _HASH_REGISTRY.get(url)
        if entry:
            if expected_sha256 is None:
                expected_sha256 = entry.get("sha256")
            if expected_size is None:
                expected_size = entry.get("size")

    if os.environ.get("FOOOCUS_SKIP_HASH_CHECK", "").strip() == "1":
        expected_sha256 = None

    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip("/")
    url = url.replace("https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)

    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)

    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    partial_file = cached_file + ".partial"

    def _validate(path: str, use_stamp: bool = False) -> Optional[str]:
        """Return an error string if ``path`` fails validation, else None.

        When ``use_stamp`` is True we consult the sidecar stamp first and
        skip the expensive sha256 read if size + mtime are unchanged.
        """
        actual_size = os.path.getsize(path)
        if expected_size is not None and actual_size != expected_size:
            return (
                f"size mismatch (expected {expected_size} bytes, "
                f"got {actual_size})"
            )
        if expected_sha256 is None:
            return None
        if use_stamp:
            stamp = _read_stamp(path)
            if stamp is not None:
                try:
                    st = os.stat(path)
                except OSError:
                    stamp = None
                else:
                    if (
                        stamp.get("size") == st.st_size
                        and stamp.get("mtime") == st.st_mtime
                        and (stamp.get("sha256") or "").lower() == expected_sha256.lower()
                    ):
                        return None  # trusted cache
        actual_sha = _sha256(path)
        if actual_sha.lower() != expected_sha256.lower():
            return f"sha256 mismatch (expected {expected_sha256}, got {actual_sha})"
        _write_stamp(path, actual_sha)
        return None

    # If the final file already exists, validate it. If it fails, quarantine
    # and re-download; if there's no expected hash/size we trust it (matches
    # upstream behavior for the common case). ``use_stamp=True`` lets us
    # skip the full sha256 pass on repeat launches when the file hasn't
    # changed since we last verified it.
    if os.path.exists(cached_file):
        err = _validate(cached_file, use_stamp=True)
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
