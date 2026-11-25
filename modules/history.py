"""In-app history: find past generations and the settings that made them.

Images live in ``<path_outputs>/<YYYY-MM-DD>/``. Settings come from that
day's ``log.html`` (every entry carries a URL-encoded JSON copy of the
image's metadata for its "Copy to Clipboard" button), falling back to
metadata embedded in the image itself, which is off by default.
"""
import html
import json
import os
import re
import urllib.parse
from dataclasses import dataclass, field

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
SEARCH_FIELDS = ('prompt', 'negative_prompt', 'prompt_expansion', 'styles', 'base_model', 'seed')
DETAIL_FIELDS = (
    ('Prompt', 'prompt'),
    ('Negative prompt', 'negative_prompt'),
    ('Styles', 'styles'),
    ('Model', 'base_model'),
    ('Performance', 'performance'),
    ('Resolution', 'resolution'),
    ('Seed', 'seed'),
    ('Sampler', 'sampler'),
    ('Scheduler', 'scheduler'),
    ('Guidance scale', 'guidance_scale'),
)

_DATE_DIR_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_LOG_ITEM_RE = re.compile(
    r'<div id="([^"]+)" class="image-container">.*?to_clipboard\(\'([^\']*)\'\)', re.S)


@dataclass
class HistoryItem:
    path: str
    filename: str
    date: str
    mtime: float
    metadata: dict = field(default_factory=dict)
    source: str = 'none'  # 'log', 'image' or 'none'


def _div_id_to_filename(div_id: str) -> str:
    """log.html ids are the image filename with '.' replaced by '_'."""
    base, sep, ext = html.unescape(div_id).rpartition('_')
    return f'{base}.{ext}' if sep else div_id


def parse_log_html(text: str) -> dict:
    """Map image filename -> metadata dict for every entry in a log.html."""
    entries = {}
    for div_id, payload in _LOG_ITEM_RE.findall(text):
        try:
            data = json.loads(urllib.parse.unquote(payload))
        except ValueError:
            continue
        if isinstance(data, dict):
            entries[_div_id_to_filename(div_id)] = data
    return entries


def read_embedded_metadata(path: str) -> dict | None:
    """Metadata saved inside the image file, as a Fooocus-style dict."""
    from PIL import Image
    from modules.meta_parser import A1111MetadataParser, read_info_from_image

    with Image.open(path) as image:
        parameters, _scheme = read_info_from_image(image)
    if isinstance(parameters, dict):
        return parameters
    if isinstance(parameters, str):
        try:
            return A1111MetadataParser().to_json(parameters)
        except Exception:
            return {'prompt': parameters.split('\n', 1)[0].strip()}
    return None


class HistoryIndex:
    """Scans an outputs folder, caching parsed logs and embedded metadata by
    file modification time so repeated refreshes stay cheap."""

    def __init__(self, embedded_reader=read_embedded_metadata):
        self._embedded_reader = embedded_reader
        self._logs = {}       # log path -> (mtime, {filename: metadata})
        self._embedded = {}   # image path -> (mtime, metadata or None)

    def _log_metadata(self, log_path: str) -> dict:
        try:
            mtime = os.path.getmtime(log_path)
        except OSError:
            return {}
        cached = self._logs.get(log_path)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            with open(log_path, encoding='utf-8') as fh:
                parsed = parse_log_html(fh.read())
        except OSError:
            parsed = {}
        self._logs[log_path] = (mtime, parsed)
        return parsed

    def _embedded_metadata(self, path: str, mtime: float):
        cached = self._embedded.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            meta = self._embedded_reader(path)
        except Exception:
            meta = None
        self._embedded[path] = (mtime, meta)
        return meta

    def scan(self, root: str) -> list:
        """All images under ``root``, newest first."""
        if not root or not os.path.isdir(root):
            return []
        items = []
        days = sorted((d for d in os.listdir(root) if _DATE_DIR_RE.match(d)), reverse=True)
        for day in days:
            day_dir = os.path.join(root, day)
            if not os.path.isdir(day_dir):
                continue
            log_meta = self._log_metadata(os.path.join(day_dir, 'log.html'))
            files = []
            for name in os.listdir(day_dir):
                if not name.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                path = os.path.join(day_dir, name)
                try:
                    files.append((os.path.getmtime(path), name, path))
                except OSError:
                    continue
            for mtime, name, path in sorted(files, reverse=True):
                meta, source = log_meta.get(name), 'log'
                if meta is None:
                    meta = self._embedded_metadata(path, mtime)
                    source = 'image' if meta else 'none'
                items.append(HistoryItem(path, name, day, mtime, meta or {}, source))
        return items


def search(items: list, query: str | None) -> list:
    """Items matching every whitespace-separated term (case-insensitive)."""
    terms = (query or '').lower().split()
    if not terms:
        return list(items)

    def haystack(item):
        values = [item.filename] + [str(item.metadata.get(k, '')) for k in SEARCH_FIELDS]
        return ' '.join(values).lower()

    return [item for item in items if all(term in haystack(item) for term in terms)]


def describe(item: HistoryItem) -> str:
    """Plain-text summary for the details box (never rendered as HTML)."""
    lines = [f'File: {item.filename}', f'Date: {item.date}']
    for label, key in DETAIL_FIELDS:
        value = item.metadata.get(key)
        if value not in (None, ''):
            lines.append(f'{label}: {value}')
    if item.source == 'none':
        lines.append('No saved settings were found for this image.')
    return '\n'.join(lines)


def _resolve_filename(value, filenames):
    """Map a model name (with or without extension) to an installed filename.
    Returns None when it isn't installed, so the UI keeps its current choice."""
    if value in (None, '', 'None'):
        return value
    if value in filenames:
        return value
    for filename in filenames:
        if os.path.splitext(os.path.basename(filename))[0] == value:
            return filename
    return None


def to_load_parameters(metadata: dict, model_filenames, lora_filenames, vae_filenames) -> dict:
    """Metadata ready for meta_parser.load_parameter_button_click."""
    params = dict(metadata)
    for key in ('base_model', 'refiner_model'):
        if key in params:
            params[key] = _resolve_filename(params[key], model_filenames)
    # Drop choices this app can't show (e.g. records from FLUX.1 or another
    # tool): an unknown Performance would break the next Generate.
    from modules.flags import Performance, sampler_list, scheduler_list
    for key, valid in (('performance', Performance.values()), ('sampler', sampler_list),
                       ('scheduler', scheduler_list)):
        if key in params and params[key] not in valid:
            del params[key]
    # A GGUF model's step count isn't an SDXL step override (it would force
    # e.g. 4 steps on the next SDXL image).
    if str(metadata.get('base_model', '')).lower().endswith('.gguf'):
        params.pop('steps', None)
    from modules.flags import default_vae
    if 'vae' in params and params['vae'] != default_vae:
        params['vae'] = _resolve_filename(params['vae'], vae_filenames)
    for key in [k for k in params if k.startswith('lora_combined_')]:
        parts = str(params[key]).split(' : ')
        name_index = 1 if len(parts) == 3 else 0
        if len(parts) < 2:
            del params[key]
            continue
        resolved = _resolve_filename(parts[name_index], lora_filenames)
        if resolved is None:
            del params[key]
        else:
            parts[name_index] = resolved
            params[key] = ' : '.join(parts)
    return params
