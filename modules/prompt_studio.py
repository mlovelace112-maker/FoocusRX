"""Prompt Studio: turn an idea and/or a reference image into a ready-to-use
prompt (positive, negative, suggested Fooocus styles and aspect ratio).

Engines
-------
* Offline (default): a local Ollama server. A vision model reads the
  reference image, a text model writes the prompt. Nothing leaves the Mac.
* Claude (optional, online): the Anthropic API with the user's own key
  (ANTHROPIC_API_KEY, an ``ant auth login`` profile, or a key typed into the
  Studio settings for this session only; keys are never written to disk).
  Needs the optional ``anthropic`` package.

When the chosen engine can't be reached the studio degrades instead of
failing: Claude falls back to Ollama, and Ollama falls back to the built-in
prompt rewriter (text) and Fooocus' Describe interrogator (images) when the
caller supplies them.

Ollama keeps a model in memory for ``keep_alive`` after a request; release()
asks it to unload them so they never sit next to the image model while it
renders (webui calls it when a generation starts).

Nothing here imports torch or gradio, so it is cheap to import and to test.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field, fields

from modules import prompt_rewriter

# --- Choices shown in the UI -------------------------------------------------

ENGINE_OLLAMA = 'Offline (Ollama)'
ENGINE_CLAUDE = 'Claude (online)'
ENGINES = [ENGINE_OLLAMA, ENGINE_CLAUDE]

TARGET_AUTO = 'Auto (from base model)'
TARGETS = {
    TARGET_AUTO: None,
    'SDXL phrases': 'sdxl',
    'Anime tags (Pony/Illustrious)': 'tags',
    'FLUX sentences': 'flux',
}

# Label -> Fooocus Image Prompt type (None: the reference is only described).
REF_DESCRIBE_ONLY = 'Describe only'
REFERENCE_MODES = {
    REF_DESCRIBE_ONLY: None,
    'Overall look (Image Prompt)': 'ImagePrompt',
    'Pose and outlines (PyraCanny)': 'PyraCanny',
    'Depth and structure (CPDS)': 'CPDS',
    'Face (FaceSwap)': 'FaceSwap',
}
DEFAULT_REFERENCE_MODE = 'Overall look (Image Prompt)'

REFERENCE_INSTRUCTIONS = {
    None: ("The reference is only used for its description: the picture is generated from the prompt alone, "
           "so the prompt must carry everything worth keeping from the reference, changed as the idea asks."),
    'ImagePrompt': ("Fooocus Image Prompt will also feed the reference into generation to carry its overall look. "
                    "Match the reference's subject and style, changing only what the idea asks for."),
    'PyraCanny': ("PyraCanny will make the picture follow the reference's outlines, pose and layout. Keep the "
                  "subject and composition consistent with the reference's shapes and pose; let the idea change "
                  "materials, style, colours and setting freely."),
    'CPDS': ("CPDS will make the picture follow the reference's depth and structure. Keep the arrangement and "
             "perspective consistent with the reference; let the idea change surface detail, style and colours."),
    'FaceSwap': ("FaceSwap will carry the face from the reference into the new picture. Describe the person's "
                 "general look (hair, apparent age range, build) without naming them, and put them in the scene "
                 "the idea describes."),
}

ASPECTS = {'square': 1.0, 'portrait': 0.75, 'tall': 9 / 16, 'landscape': 4 / 3, 'wide': 16 / 9}

# Styles the model may suggest (only those present in this install are offered).
CURATED_STYLES = [
    'Fooocus Enhance', 'Fooocus Sharp', 'Fooocus Photograph', 'Fooocus Cinematic', 'Fooocus Semi Realistic',
    'Fooocus Masterpiece', 'SAI Photographic', 'SAI Cinematic', 'SAI Analog Film', 'SAI Anime', 'SAI Digital Art',
    'SAI Fantasy Art', 'SAI Comic Book', 'SAI 3D Model', 'SAI Line Art', 'SAI Neonpunk', 'SAI Pixel Art',
    'SAI Isometric', 'SAI Origami', 'SAI Craft Clay', 'SAI Lowpoly', 'SAI Texture', 'Watercolor 2',
    'Dark Moody Atmosphere', 'Light Cheery Atmosphere', 'Volumetric Lighting', 'Vibrant Rim Light',
    'Faded Polaroid Photo', 'Macro Photography', 'Pencil Sketch Drawing', 'Japanese Ink Drawing', 'Art Nouveau',
    'Art Deco', 'Steampunk 2', 'Surrealism', 'Minimalism', 'Dark Fantasy', 'Cel Shaded Art', 'Flat 2D Art',
    'Simple Vector Art', 'Impressionism', 'Pop Art 2', 'MK Van Gogh', 'Double Exposure', 'High Fashion',
]
MAX_STYLES = 3

# --- Defaults (override with environment variables) --------------------------
# gemma3:4b reads images and writes text, so one ~3.3 GB download covers both
# roles and it fits comfortably next to SDXL on an Apple Silicon Mac.
DEFAULT_OLLAMA_URL = os.environ.get('FOOOCUS_OLLAMA_URL', 'http://127.0.0.1:11434')
DEFAULT_TEXT_MODEL = os.environ.get('FOOOCUS_STUDIO_TEXT_MODEL', 'gemma3:4b')
DEFAULT_VISION_MODEL = os.environ.get('FOOOCUS_STUDIO_VISION_MODEL', 'gemma3:4b')
DEFAULT_CLAUDE_MODEL = os.environ.get('FOOOCUS_STUDIO_CLAUDE_MODEL', 'claude-opus-5-5')
# How long Ollama keeps a model loaded after a request (Ollama duration string).
DEFAULT_KEEP_ALIVE = os.environ.get('FOOOCUS_STUDIO_KEEP_ALIVE', '2m')
OLLAMA_TIMEOUT = float(os.environ.get('FOOOCUS_STUDIO_OLLAMA_TIMEOUT', '300'))
MAX_IMAGE_SIDE = 1024

# Server-side refusal fallback is offered for these Claude models only.
_CLAUDE_FALLBACK_MODELS = {'claude-fable-5-2', 'claude-fable-5-1', 'claude-opus-5-5', 'claude-opus-5',
                           'claude-sonnet-5-5'}

SETTINGS_FILE = '.prompt_studio.json'
HISTORY_FILE = '.prompt_studio_history.json'
HISTORY_LIMIT = 50

OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'positive': {'type': 'string'},
        'negative': {'type': 'string'},
        'styles': {'type': 'array', 'items': {'type': 'string'}},
        'aspect': {'type': 'string', 'enum': list(ASPECTS)},
        'notes': {'type': 'string'},
    },
    'required': ['positive', 'negative', 'styles', 'aspect', 'notes'],
    'additionalProperties': False,
}

DESCRIBE_PROMPT = (
    "Describe this image so it can be recreated with an image generator. In 80 to 120 words of plain prose, "
    "cover: the main subject and its appearance, pose and expression; clothing; setting and background; "
    "composition, framing and camera angle; lighting; colour palette; medium or art style (photo, painting, "
    "3D render, anime ...); mood. Do not guess who anyone is or give names. Reply with the description only.")


class StudioError(Exception):
    """A problem the user can act on; the message is shown in the UI."""


class OllamaUnavailable(StudioError):
    pass


class ClaudeUnavailable(StudioError):
    pass


# --- Settings and history (stored next to the images, never in the repo) ----

@dataclass
class StudioSettings:
    engine: str = ENGINE_OLLAMA
    ollama_url: str = DEFAULT_OLLAMA_URL
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    claude_model: str = DEFAULT_CLAUDE_MODEL
    keep_alive: str = DEFAULT_KEEP_ALIVE
    # Session only: never saved (see save_settings).
    claude_api_key: str = field(default='', repr=False)

    def clean(self) -> 'StudioSettings':
        self.engine = self.engine if self.engine in ENGINES else ENGINE_OLLAMA
        self.ollama_url = (self.ollama_url or DEFAULT_OLLAMA_URL).strip().rstrip('/')
        self.text_model = (self.text_model or DEFAULT_TEXT_MODEL).strip()
        self.vision_model = (self.vision_model or DEFAULT_VISION_MODEL).strip()
        self.claude_model = (self.claude_model or DEFAULT_CLAUDE_MODEL).strip()
        self.claude_api_key = (self.claude_api_key or '').strip()
        return self


_file_lock = threading.Lock()


def _read_json(path, default):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _write_json(path, data):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def load_settings(root: str) -> StudioSettings:
    data = _read_json(os.path.join(root, SETTINGS_FILE), {})
    known = {f.name for f in fields(StudioSettings)} - {'claude_api_key'}
    data = {k: v for k, v in data.items() if k in known and isinstance(v, str)} if isinstance(data, dict) else {}
    return StudioSettings(**data).clean()


def save_settings(root: str, settings: StudioSettings):
    data = asdict(settings)
    data.pop('claude_api_key', None)  # never persist secrets
    with _file_lock:
        _write_json(os.path.join(root, SETTINGS_FILE), data)


def load_history(root: str) -> list:
    data = _read_json(os.path.join(root, HISTORY_FILE), [])
    return [e for e in data if isinstance(e, dict) and e.get('id') and isinstance(e.get('positive'), str)] \
        if isinstance(data, list) else []


def add_history(root: str, entry: dict) -> dict:
    entry = dict(entry, id=uuid.uuid4().hex[:12], created=time.strftime('%Y-%m-%d %H:%M'))
    with _file_lock:
        entries = [entry] + load_history(root)
        _write_json(os.path.join(root, HISTORY_FILE), entries[:HISTORY_LIMIT])
    return entry


def get_history(root: str, entry_id: str):
    return next((e for e in load_history(root) if e['id'] == entry_id), None)


def history_label(entry: dict) -> str:
    title = (entry.get('idea') or entry.get('positive') or '').strip().replace('\n', ' ')
    title = title[:60] + ('…' if len(title) > 60 else '')
    image = ' 🖼' if entry.get('had_image') else ''
    return f"{entry.get('created', '')} · {title}{image}"


# --- Helpers ---------------------------------------------------------------

def resolve_target(target_label: str, base_model: str | None) -> str:
    style = TARGETS.get(target_label)
    return style or prompt_rewriter.style_for_model(base_model)


def available_styles(legal_style_names) -> list:
    legal = set(legal_style_names or [])
    return [s for s in CURATED_STYLES if s in legal]


def pick_styles(suggested, legal_style_names, limit: int = MAX_STYLES) -> list:
    """Keep suggested names that exist (case-insensitive), in order, without duplicates."""
    by_lower = {name.lower(): name for name in legal_style_names or []}
    out = []
    for name in suggested or []:
        match = by_lower.get(str(name).strip().lower())
        if match and match not in out:
            out.append(match)
    return out[:limit]


def _label_ratio(label: str):
    m = re.match(r'\s*(\d+)\s*[×x*]\s*(\d+)', label or '')
    return int(m.group(1)) / int(m.group(2)) if m else None


def pick_aspect(kind: str, labels) -> str | None:
    """The available aspect-ratio label closest to ``kind`` (square, portrait ...)."""
    target = ASPECTS.get(str(kind or '').strip().lower())
    best, best_err = None, None
    for label in labels or []:
        ratio = _label_ratio(label)
        if target is None or ratio is None:
            continue
        err = abs(ratio - target) / target
        if best_err is None or err < best_err - 1e-9:
            best, best_err = label, err
    return best


def aspect_display(label: str) -> str:
    """'1152×896 <span ...> ∣ 9:7</span>' -> '1152×896 (9:7)'."""
    text = re.sub(r'<[^>]+>', '', label or '').replace('∣', '').split()
    return f'{text[0]} ({text[1]})' if len(text) >= 2 else ' '.join(text)


def image_to_png_b64(image, max_side: int = MAX_IMAGE_SIDE) -> str:
    """numpy HxWxC array (or PIL image) -> base64 PNG, downscaled to max_side."""
    from PIL import Image
    if not isinstance(image, Image.Image):
        import numpy as np
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
    image = image.convert('RGB')
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return base64.standard_b64encode(buf.getvalue()).decode('ascii')


def parse_json_reply(text: str) -> dict:
    """Best-effort parse of a model reply that should be one JSON object."""
    text = (text or '').strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r'\{.*\}', text, flags=re.S)
        data = None
        if m:
            try:
                data = json.loads(m.group(0))
            except ValueError:
                data = None
    if not isinstance(data, dict):
        # Not JSON at all: treat the whole reply as the prompt.
        return {'positive': text, 'negative': '', 'styles': [], 'aspect': '', 'notes': ''}
    return data


def clean_negative(text: str) -> str:
    parts = [p.strip(' .;"\'') for p in re.split(r'[,\n]', str(text or ''))]
    out = []
    for part in parts:
        if part and part.lower() not in (o.lower() for o in out):
            out.append(part)
    return ', '.join(out)


# --- Prompts -----------------------------------------------------------------

def build_system_prompt(style: str, style_choices) -> str:
    spec = prompt_rewriter.STYLES.get(style, prompt_rewriter.STYLES['sdxl'])
    example_in, example_out = spec['example']
    if style == 'flux':
        negative_rule = 'an empty string (FLUX does not use a negative prompt).'
    else:
        negative_rule = ('5 to 15 comma-separated things to avoid that fit this picture (for a photo, e.g. '
                         '"illustration, cartoon, plastic skin"; for a drawing, e.g. "photo, 3d render"). '
                         'Skip generic quality words such as "low quality" or "worst quality": the app adds those.')
    styles = ', '.join(style_choices) if style_choices else '(none available: return an empty list)'
    return (
        f"You are Prompt Studio, an expert prompt writer for the text-to-image model {spec['target']} "
        "running in Fooocus. Turn the user's idea, and their reference image if there is one, into one "
        "ready-to-use image prompt.\n\n"
        "Reply with a JSON object with these keys:\n"
        f"- positive: the image prompt. {spec['format']} Put the main subject first. Keep every detail the "
        "user gave and add concrete visual details: appearance, setting, lighting, composition, camera or "
        "medium, colour palette and mood. No text, captions, watermarks or signatures; no real people's or "
        "artists' names.\n"
        f"- negative: {negative_rule}\n"
        f"- styles: up to {MAX_STYLES} Fooocus style names that suit the picture, chosen only from this list, "
        f"or an empty list: {styles}\n"
        f"- aspect: the best frame for the picture, one of: {', '.join(ASPECTS)}.\n"
        "- notes: one short sentence for the user about a choice you made.\n\n"
        f"Example of the prompt style. Idea: \"{example_in}\" -> positive: \"{example_out}\"")


def build_user_prompt(idea: str, ip_type: str | None, description: str | None, image_attached: bool) -> str:
    idea = (idea or '').strip()
    lines = [f"Idea: {idea}" if idea else
             "Idea: (none given: build the prompt from the reference image)"]
    if image_attached:
        lines.append("Reference image: attached.")
    elif description:
        lines.append(f"Reference image, as described by a vision model: {description.strip()}")
    if image_attached or description:
        lines.append(f"How the reference is used: {REFERENCE_INSTRUCTIONS.get(ip_type, REFERENCE_INSTRUCTIONS[None])}")
    lines.append("Reply with the JSON object only.")
    return '\n'.join(lines)


# --- Ollama ------------------------------------------------------------------

def _is_loopback(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or '').lower()
    return host in ('localhost', '127.0.0.1', '::1')


def _http_json(url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'},
                                     method='POST' if data is not None else 'GET')
    # Loopback traffic never goes through a proxy (mac_launch.sh clears them too).
    handlers = [urllib.request.ProxyHandler({})] if _is_loopback(url) else []
    with urllib.request.build_opener(*handlers).open(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8') or '{}')


def _ollama_error(exc: Exception, url: str, model: str | None = None) -> StudioError:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = json.loads(exc.read().decode('utf-8')).get('error', '')
        except Exception:
            detail = ''
        if exc.code == 404 or 'not found' in detail.lower():
            return StudioError(f"The Ollama model “{model}” isn't installed. In Terminal, run: ollama pull {model}")
        return StudioError(f"Ollama returned an error ({exc.code}): {detail or exc.reason}")
    return OllamaUnavailable(f"Ollama isn't running at {url}. Open the Ollama app (or run “ollama serve”), "
                             "then try again.")


def ollama_models(url: str, timeout: float = 3.0) -> list:
    """Names of the installed Ollama models; raises OllamaUnavailable."""
    try:
        data = _http_json(url.rstrip('/') + '/api/tags', timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise _ollama_error(exc, url) from exc
    return sorted(m.get('name', '') for m in data.get('models', []) if m.get('name'))


def _model_installed(name: str, installed) -> bool:
    name = name.strip()
    return name in installed or (':' not in name and f'{name}:latest' in installed)


def ollama_status(settings: StudioSettings) -> str:
    """One Markdown line for the settings panel."""
    try:
        installed = ollama_models(settings.ollama_url)
    except StudioError as exc:
        return f'⚠️ {exc}'
    missing = [m for m in dict.fromkeys([settings.text_model, settings.vision_model])
               if not _model_installed(m, installed)]
    if missing:
        pulls = '; '.join(f'`ollama pull {m}`' for m in missing)
        return f'⚠️ Ollama is running, but these models are missing. In Terminal, run {pulls}'
    return f'✅ Ollama is running with {settings.text_model}' + (
        f' and {settings.vision_model}.' if settings.vision_model != settings.text_model else '.')


_used_ollama = {}   # model -> url, so release() knows what to unload
_used_lock = threading.Lock()


def ollama_chat(settings: StudioSettings, model: str, messages: list, schema: dict | None = None,
                temperature: float = 0.7, timeout: float = OLLAMA_TIMEOUT) -> str:
    url = settings.ollama_url.rstrip('/') + '/api/chat'
    payload = {'model': model, 'messages': messages, 'stream': False, 'keep_alive': settings.keep_alive,
               'options': {'temperature': temperature, 'num_ctx': 4096}}
    if schema is not None:
        payload['format'] = schema
    try:
        try:
            data = _http_json(url, payload, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # Ollama older than 0.5 only knows format="json".
            if schema is None or exc.code not in (400, 500):
                raise
            payload['format'] = 'json'
            data = _http_json(url, payload, timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise _ollama_error(exc, settings.ollama_url, model) from exc
    with _used_lock:
        _used_ollama[model] = settings.ollama_url
    if data.get('error'):
        raise StudioError(f"Ollama: {data['error']}")
    return (data.get('message') or {}).get('content', '')


def release(wait: bool = False):
    """Ask Ollama to unload the studio's models now (non-blocking by default)."""
    with _used_lock:
        used = dict(_used_ollama)
        _used_ollama.clear()
    if not used:
        return False

    def unload():
        for model, url in used.items():
            try:
                _http_json(url.rstrip('/') + '/api/generate', {'model': model, 'keep_alive': 0}, timeout=5)
            except Exception:
                pass

    if wait:
        unload()
    else:
        threading.Thread(target=unload, daemon=True, name='prompt-studio-release').start()
    return True


# --- Claude ------------------------------------------------------------------

def _claude_client(settings: StudioSettings):
    try:
        import anthropic
    except ImportError as exc:
        raise ClaudeUnavailable("Claude mode needs the anthropic package. In Terminal, from the FoocusRX folder, "
                                "run: venv/bin/pip install anthropic") from exc
    # No key typed: the SDK reads ANTHROPIC_API_KEY or an `ant auth login` profile.
    return anthropic.Anthropic(api_key=settings.claude_api_key) if settings.claude_api_key else anthropic.Anthropic()


def claude_compose(settings: StudioSettings, system: str, user_text: str, image_b64: str | None,
                   client=None) -> str:
    """One Claude request that reads the image (if any) and returns the JSON text."""
    try:
        import anthropic
    except ImportError:
        anthropic = None
    content = []
    if image_b64:
        content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': image_b64}})
    content.append({'type': 'text', 'text': user_text})
    kwargs = dict(model=settings.claude_model, max_tokens=8000, system=system,
                  messages=[{'role': 'user', 'content': content}],
                  output_config={'effort': 'low', 'format': {'type': 'json_schema', 'schema': OUTPUT_SCHEMA}})
    try:
        if client is None:
            client = _claude_client(settings)
        if settings.claude_model in _CLAUDE_FALLBACK_MODELS:
            # If a safety classifier declines, the API retries on a fallback model.
            response = client.beta.messages.create(betas=['server-side-fallback-2026-07-01'], fallbacks='default',
                                                   **kwargs)
        else:
            response = client.messages.create(**kwargs)
    except StudioError:
        raise
    except Exception as exc:
        raise _claude_error(exc, anthropic) from exc
    if response.stop_reason == 'refusal':
        raise StudioError("Claude declined to write this prompt. Try rewording the idea, or use the offline engine.")
    text = next((b.text for b in response.content if getattr(b, 'type', '') == 'text'), '')
    if not text.strip():
        raise StudioError('Claude returned an empty reply; try again.')
    return text


def _claude_error(exc: Exception, anthropic) -> Exception:
    if isinstance(exc, TypeError) and ('api_key' in str(exc) or 'authentication' in str(exc).lower()):
        return StudioError("No Claude API key found. Set ANTHROPIC_API_KEY before launching, or paste a key in "
                           "Studio settings.")
    if anthropic is None:
        return exc
    if isinstance(exc, anthropic.AuthenticationError):
        return StudioError("Claude rejected the API key. Set ANTHROPIC_API_KEY before launching, or paste a key in "
                           "Studio settings.")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return StudioError("This API key can't use that Claude model. Check the model name in Studio settings.")
    if isinstance(exc, anthropic.NotFoundError):
        return StudioError("Claude doesn't know that model name. Check it in Studio settings.")
    if isinstance(exc, anthropic.RateLimitError):
        return StudioError("Claude is rate-limiting this key right now; wait a minute or use the offline engine.")
    if isinstance(exc, anthropic.APIConnectionError):
        return ClaudeUnavailable("Couldn't reach Claude (no internet connection?).")
    if isinstance(exc, anthropic.APIStatusError):
        return StudioError(f"Claude returned an error ({exc.status_code}): {exc.message}")
    return exc


# --- The studio --------------------------------------------------------------

@dataclass
class StudioResult:
    positive: str
    negative: str = ''
    styles: list = field(default_factory=list)
    aspect: str | None = None          # an aspect-ratio label from the app, or None
    notes: str = ''
    description: str = ''              # what the vision model saw in the reference
    engine_used: str = ''
    ip_type: str | None = None         # Image Prompt type to send the reference as


def build_prompt(idea: str, image, reference_mode: str, settings: StudioSettings, *, style: str = 'sdxl',
                 legal_style_names=(), aspect_labels=(), fallback_rewrite=None, fallback_describe=None,
                 claude_client=None) -> StudioResult:
    """Build a prompt from an idea and/or a reference image.

    fallback_rewrite(idea, style) -> str and fallback_describe(image) -> str are
    used when Ollama isn't running (the built-in rewriter / Describe).
    """
    idea = (idea or '').strip()
    has_image = image is not None
    if not idea and not has_image:
        raise StudioError('Type an idea or add a reference image first.')
    settings = settings.clean()
    ip_type = REFERENCE_MODES.get(reference_mode) if has_image else None
    style_choices = available_styles(legal_style_names)
    system = build_system_prompt(style, style_choices)
    image_b64 = image_to_png_b64(image) if has_image else None
    notes, description, reply, engine_used = [], '', None, settings.engine

    if settings.engine == ENGINE_CLAUDE:
        try:
            reply = claude_compose(settings, system, build_user_prompt(idea, ip_type, None, has_image), image_b64,
                                   client=claude_client)
        except ClaudeUnavailable as exc:
            notes.append(f'{exc} Used the offline engine instead.')
            engine_used = ENGINE_OLLAMA

    if reply is None:
        try:
            if has_image:
                description = ollama_chat(settings, settings.vision_model,
                                          [{'role': 'user', 'content': DESCRIBE_PROMPT, 'images': [image_b64]}],
                                          temperature=0.2).strip()
            reply = ollama_chat(settings, settings.text_model, [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': build_user_prompt(idea, ip_type, description, False)},
            ], schema=OUTPUT_SCHEMA)
        except OllamaUnavailable as exc:
            return _degraded(idea, image, ip_type, style, str(exc), notes, fallback_rewrite, fallback_describe)

    data = parse_json_reply(reply)
    positive = prompt_rewriter.clean_output(str(data.get('positive') or ''), style, max_chars=900)
    if not positive:
        raise StudioError('The model returned an empty prompt; try again.')
    if data.get('notes'):
        notes.insert(0, str(data['notes']).strip())
    return StudioResult(positive=positive,
                        negative='' if style == 'flux' else clean_negative(data.get('negative')),
                        styles=pick_styles(data.get('styles'), style_choices),
                        aspect=pick_aspect(data.get('aspect'), aspect_labels),
                        notes=' '.join(n for n in notes if n), description=description,
                        engine_used=engine_used, ip_type=ip_type)


def _degraded(idea, image, ip_type, style, reason, notes, fallback_rewrite, fallback_describe) -> StudioResult:
    """Ollama is down: use the built-in rewriter / Describe if the caller offers them."""
    parts = []
    description = ''
    if image is not None and fallback_describe is not None:
        try:
            description = (fallback_describe(image) or '').strip()
        except Exception as exc:  # the interrogator needs its own model download
            print(f'[Prompt Studio] Describe fallback failed: {exc}')
    if description:
        parts.append(description)
    if idea:
        parts.insert(0, idea)
    seed_text = ', '.join(parts)
    positive = ''
    if seed_text and fallback_rewrite is not None:
        try:
            positive = (fallback_rewrite(seed_text, style) or '').strip()
        except Exception as exc:
            print(f'[Prompt Studio] Rewriter fallback failed: {exc}')
    if not positive and description:
        positive = seed_text
    if not positive:
        raise OllamaUnavailable(reason)
    notes = notes + [f'{reason} Used the built-in tools instead, so there is no negative prompt or style '
                     'suggestion this time.']
    return StudioResult(positive=positive, notes=' '.join(notes), description=description,
                        engine_used='Built-in', ip_type=ip_type)
