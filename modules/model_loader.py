import json
import os
import hashlib
import urllib.error
import urllib.request
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


def _partial_meta_path(partial_file: str) -> str:
    """Sidecar next to a ``.partial`` file that records what it's a partial *of*."""
    return partial_file + ".meta"


def _read_partial_meta(partial_file: str) -> Optional[dict]:
    try:
        with open(_partial_meta_path(partial_file), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_partial_meta(partial_file: str, url: str, expected_size: Optional[int]) -> None:
    try:
        with open(_partial_meta_path(partial_file), "w", encoding="utf-8") as f:
            json.dump({"url": url, "expected_size": expected_size}, f)
    except OSError:
        pass


def _remove_partial(partial_file: str) -> None:
    """Delete a ``.partial`` and its ``.meta`` sidecar, ignoring errors."""
    for p in (partial_file, _partial_meta_path(partial_file)):
        try:
            os.remove(p)
        except OSError:
            pass


def _tqdm_bar(total: Optional[int], initial: int, desc: str):
    """Return a ``tqdm`` progress bar if the library is available, else a
    duck-typed no-op with ``.update()`` / ``.close()``.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        class _NullBar:
            def update(self, n): pass
            def close(self): pass
        return _NullBar()
    return tqdm(
        total=total,
        initial=initial,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
        dynamic_ncols=True,
    )


def _download_with_resume(
    url: str,
    partial_file: str,
    expected_size: Optional[int] = None,
    progress: bool = True,
    chunk_size: int = 1 << 20,  # 1 MiB
) -> None:
    """Download ``url`` into ``partial_file``, resuming an interrupted
    ``.partial`` when possible.

    Behavior:

    * If ``partial_file`` exists and its ``.meta`` sidecar matches ``url``
      and ``expected_size``, send ``Range: bytes=<len>-`` and append.
    * If the server returns HTTP 206 Partial Content with a sensible
      ``Content-Range``, we resume. Anything else (200 OK, wrong range,
      HTTP error) falls back to a full restart: truncate and re-download
      from byte 0.
    * Progress bar reflects the true cumulative position when resuming.
    * Set ``FOOOCUS_DOWNLOAD_RESUME=0`` to disable resume entirely.

    Raises on unrecoverable network errors after resume fallback.
    """
    resume_enabled = os.environ.get("FOOOCUS_DOWNLOAD_RESUME", "1").strip() != "0"

    existing_bytes = 0
    if resume_enabled and os.path.exists(partial_file):
        meta = _read_partial_meta(partial_file)
        if (
            meta
            and meta.get("url") == url
            and meta.get("expected_size") == expected_size
        ):
            existing_bytes = os.path.getsize(partial_file)
            # Guard against a .partial that's already at or past the
            # expected size (would ask for an empty/invalid range).
            if expected_size is not None and existing_bytes >= expected_size:
                existing_bytes = 0
        else:
            # Stale .partial from a different URL/size. Discard.
            _remove_partial(partial_file)

    if not os.path.exists(partial_file):
        _write_partial_meta(partial_file, url, expected_size)

    req = urllib.request.Request(url, headers={"User-Agent": "FoocusRX/1.0"})
    open_mode = "wb"
    resumed = False
    if existing_bytes > 0:
        req.add_header("Range", f"bytes={existing_bytes}-")

    try:
        with urllib.request.urlopen(req) as resp:
            status = getattr(resp, "status", 200)
            content_range = resp.headers.get("Content-Range")
            content_length = resp.headers.get("Content-Length")

            if existing_bytes > 0 and status == 206 and content_range:
                # Sanity-check the returned range: must start at existing_bytes.
                # Format: "bytes 12345-99999/100000"
                try:
                    start_str = content_range.split()[1].split("-")[0]
                    if int(start_str) == existing_bytes:
                        open_mode = "ab"
                        resumed = True
                        print(
                            f"Resuming download from byte {existing_bytes:,}: \"{url}\""
                        )
                except (IndexError, ValueError):
                    pass

            if not resumed:
                # Server ignored our Range or gave a wrong offset. Start over.
                if existing_bytes > 0:
                    print(
                        f"Server declined resume (status={status}, range={content_range!r}); "
                        f"restarting from byte 0."
                    )
                existing_bytes = 0
                open_mode = "wb"

            # Total size for the progress bar: prefer expected_size, else
            # derive from headers (Content-Range total or Content-Length).
            total = expected_size
            if total is None and content_range:
                try:
                    total = int(content_range.split("/")[-1])
                except ValueError:
                    total = None
            if total is None and content_length:
                try:
                    total = int(content_length) + existing_bytes
                except ValueError:
                    total = None

            bar = _tqdm_bar(total, existing_bytes, os.path.basename(partial_file)) if progress else None
            try:
                with open(partial_file, open_mode) as out:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out.write(chunk)
                        if bar is not None:
                            bar.update(len(chunk))
            finally:
                if bar is not None:
                    bar.close()
    except urllib.error.HTTPError as e:
        # Server rejected Range; retry once without it.
        if existing_bytes > 0 and e.code in (416, 400):
            print(
                f"Server rejected Range request (HTTP {e.code}); "
                f"restarting from byte 0."
            )
            _remove_partial(partial_file)
            return _download_with_resume(
                url, partial_file, expected_size, progress, chunk_size
            )
        raise


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

    # Any leftover .partial is now potentially resumable (see
    # _download_with_resume). Only remove it if the .meta sidecar shows
    # it belongs to a different URL/size, which the resume helper does
    # for us.
    print(f'Downloading: "{url}" to {cached_file}')
    _download_with_resume(
        url, partial_file, expected_size=expected_size, progress=progress
    )

    err = _validate(partial_file)
    if err is not None:
        quarantine = cached_file + ".corrupt"
        try:
            os.rename(partial_file, quarantine)
        except OSError:
            pass
        # Discard the .meta sidecar - the resume metadata is only valid
        # while the .partial is still being appended to.
        try:
            os.remove(_partial_meta_path(partial_file))
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
    # Clean up the .meta sidecar now that the .partial is gone.
    try:
        os.remove(_partial_meta_path(partial_file))
    except OSError:
        pass
    return cached_file
