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


# --- Deleting images (History tab) -------------------------------------------
#
# Deleting never unlinks: images move to the macOS Trash (~/.Trash) when it is
# available, otherwise to <outputs>/trash/, so a mistake can be undone in the
# app (restore_items) or from the Finder. Only image files inside a dated
# folder of the outputs root can be moved.

def default_trash_dir(outputs_root: str) -> str:
    override = os.environ.get('FOOOCUS_TRASH_DIR')  # tests: never touch the real Trash
    if override:
        return override
    mac_trash = os.path.expanduser('~/.Trash')
    if os.path.isdir(mac_trash) and os.access(mac_trash, os.W_OK):
        return mac_trash
    return os.path.join(outputs_root, 'trash')


def is_deletable(path: str, outputs_root: str) -> bool:
    """True for an existing image file directly inside <outputs_root>/<YYYY-MM-DD>/."""
    if not path or not outputs_root or not str(path).lower().endswith(IMAGE_EXTENSIONS):
        return False
    real = os.path.realpath(path)
    root = os.path.realpath(outputs_root)
    day_dir = os.path.dirname(real)
    return (os.path.isfile(real) and os.path.dirname(day_dir) == root
            and bool(_DATE_DIR_RE.match(os.path.basename(day_dir))))


def _unique_path(directory: str, name: str) -> str:
    base, ext = os.path.splitext(name)
    candidate, n = os.path.join(directory, name), 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f'{base} ({n}){ext}')
        n += 1
    return candidate


def delete_items(paths, outputs_root: str, trash_dir: str | None = None, log_remover=None):
    """Move images to the trash. Returns (moves, skipped).

    ``moves`` is a list of (original_path, trashed_path, log_block) for
    restore_items; ``log_block`` is the image's removed log.html entry (or
    None). ``log_remover(html_name, filenames) -> {filename: block}`` drops
    entries from a day's log (private_logger.remove_log_entries). Each image's
    cached thumbnail is removed too. ``skipped`` lists refused/failed paths.
    """
    import shutil

    trash_dir = trash_dir or default_trash_dir(outputs_root)
    moved, skipped = [], []
    for path in dict.fromkeys(paths or []):  # unique, order kept
        if not is_deletable(path, outputs_root):
            skipped.append(path)
            continue
        try:
            os.makedirs(trash_dir, exist_ok=True)
            target = _unique_path(trash_dir, os.path.basename(path))
            shutil.move(os.path.realpath(path), target)
            moved.append((os.path.realpath(path), target))
        except OSError:
            skipped.append(path)
    blocks = {}
    if log_remover is not None:
        by_day = {}
        for original, _ in moved:
            by_day.setdefault(os.path.dirname(original), []).append(os.path.basename(original))
        for day_dir, names in by_day.items():
            try:
                for name, block in (log_remover(os.path.join(day_dir, 'log.html'), names) or {}).items():
                    blocks[os.path.join(day_dir, name)] = block
            except Exception:
                pass  # the images are already safe in the trash; the log is cosmetic
    for original, _ in moved:
        try:
            os.remove(thumbnail_path(original, outputs_root))
        except OSError:
            pass
    return [(original, target, blocks.get(original)) for original, target in moved], skipped


def restore_items(moves, log_restorer=None) -> int:
    """Undo delete_items (also puts log entries back via ``log_restorer``).
    Files whose original name is taken are left in the trash."""
    import shutil

    restored, blocks = 0, {}
    for move in moves or []:
        original, trashed = move[0], move[1]
        block = move[2] if len(move) > 2 else None
        if os.path.exists(trashed) and not os.path.exists(original):
            try:
                os.makedirs(os.path.dirname(original), exist_ok=True)
                shutil.move(trashed, original)
                restored += 1
                if block:
                    blocks.setdefault(os.path.dirname(original), []).append(block)
            except OSError:
                pass
    if log_restorer is not None:
        for day_dir, day_blocks in blocks.items():
            try:
                log_restorer(os.path.join(day_dir, 'log.html'), day_blocks)
            except Exception:
                pass
    return restored


# --- Thumbnails ----------------------------------------------------------------
#
# Small JPEGs under <outputs>/.thumbs/<day>/ (outside the dated folders, so the
# scan and Delete never see them), regenerated when the image is newer.

THUMB_DIR = '.thumbs'
THUMB_SIZE = 384


def thumbnail_path(path: str, outputs_root: str) -> str:
    day = os.path.basename(os.path.dirname(path))
    return os.path.join(outputs_root, THUMB_DIR, day, os.path.basename(path) + '.jpg')


def ensure_thumbnail(path: str, outputs_root: str, size: int = THUMB_SIZE) -> str:
    """Path of a cached thumbnail for ``path`` (created if missing or stale);
    the original path if a thumbnail can't be made."""
    thumb = thumbnail_path(path, outputs_root)
    try:
        if os.path.getmtime(thumb) >= os.path.getmtime(path):
            return thumb
    except OSError:
        pass
    try:
        from PIL import Image
        os.makedirs(os.path.dirname(thumb), exist_ok=True)
        with Image.open(path) as image:
            image.thumbnail((size, size))
            tmp = thumb + '.tmp'
            image.convert('RGB').save(tmp, 'JPEG', quality=85)
        os.replace(tmp, thumb)
        return thumb
    except Exception:
        return path


# --- Favourites and ratings ------------------------------------------------------
#
# Stored in <outputs>/.history_ratings.json keyed by "<day>/<filename>", so they
# survive Delete + Undo (same path) and never touch the images.

RATINGS_FILE = '.history_ratings.json'


class RatingsStore:
    def __init__(self, outputs_root: str):
        import threading
        self.path = os.path.join(outputs_root, RATINGS_FILE)
        self._lock = threading.Lock()

    @staticmethod
    def key(path: str) -> str:
        return f'{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}'

    def load(self) -> dict:
        try:
            with open(self.path, encoding='utf-8') as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def get(self, path: str) -> dict:
        entry = self.load().get(self.key(path)) or {}
        return {'favourite': bool(entry.get('favourite')), 'stars': int(entry.get('stars') or 0)}

    def update(self, path: str, favourite=None, stars=None) -> dict:
        with self._lock:
            data = self.load()
            entry = dict(data.get(self.key(path)) or {})
            if favourite is not None:
                entry['favourite'] = bool(favourite)
            if stars is not None:
                entry['stars'] = max(0, min(5, int(stars)))
            if entry.get('favourite') or entry.get('stars'):
                data[self.key(path)] = entry
            else:
                data.pop(self.key(path), None)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
        return {'favourite': bool(entry.get('favourite')), 'stars': int(entry.get('stars') or 0)}


# --- Filters and paging --------------------------------------------------------

ALL_MODELS = 'All models'
ALL_DAYS = 'All days'


def filter_choices(items: list):
    """(model choices, day choices) for the filter dropdowns."""
    models = sorted({str(i.metadata.get('base_model')) for i in items if i.metadata.get('base_model')})
    days = sorted({i.date for i in items}, reverse=True)
    return [ALL_MODELS] + models, [ALL_DAYS] + days


def filter_items(items: list, model=None, day=None, favourites_only=False, min_stars=0, ratings=None) -> list:
    ratings = ratings or {}
    out = []
    for item in items:
        if model and model != ALL_MODELS and str(item.metadata.get('base_model')) != model:
            continue
        if day and day != ALL_DAYS and item.date != day:
            continue
        if favourites_only or min_stars:
            entry = ratings.get(RatingsStore.key(item.path)) or {}
            if favourites_only and not entry.get('favourite'):
                continue
            if min_stars and int(entry.get('stars') or 0) < min_stars:
                continue
        out.append(item)
    return out


def caption(item, rating: dict | None) -> str:
    rating = rating or {}
    marks = ('\u2665 ' if rating.get('favourite') else '') + ('\u2605' * int(rating.get('stars') or 0))
    text = f"{item.date} \u00b7 seed {item.metadata.get('seed', '?')}"
    return f'{marks} {text}'.strip() if marks else text
