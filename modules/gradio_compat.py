"""
Gradio 3/4 compatibility shim.

Purpose
-------
All Gradio-3-vs-4 differences are localized here so webui.py's call
sites stay unchanged when Gradio bumps a major version. As of
Gradio 4.44.1 (the currently pinned version) the shim returns 4.x
kwargs; the 3.x branches are retained for completeness and to make the
git history readable next to the historical pin.

What this module intentionally does NOT do
------------------------------------------
- Import from Gradio internals (`gradio_client.serializing`,
  `gradio.deprecation`, `gradio.components.base._Keywords`, etc.). Those
  were deleted in Gradio 4.
- Subclass any Gradio component.

Usage
-----
    from modules import gradio_compat as gc

    # event-listener JS kwarg (Gradio 3 uses `_js`, Gradio 4 uses `js`)
    stop_button.click(fn, **gc.js_kwarg('cancelGenerateForever'), ...)

    # image source kwarg (Gradio 3: source='upload', 4: sources=['upload'])
    gr.Image(**gc.image_source('upload'), type='numpy', ...)

    # image-editor kwargs for sketch canvases (compat wrapper)
    canvas = gc.sketch_image(label='Image', color='#FFFFFF', height=500,
                             elem_id='inpaint_canvas', show_label=False)
    # In the handler, use extract_sketch_mask to unpack:
    img, mask = gc.extract_sketch_mask(canvas_value)
"""

from __future__ import annotations

try:  # pragma: no cover - environmental
    import gradio as gr
except ImportError:  # pragma: no cover - environmental
    gr = None  # type: ignore[assignment]

# Gradio-4 flag. Detect at import time; only affects the kwarg names we
# emit. Currently pinned to Gradio 4.44.1, so this resolves to True at
# runtime; the False branches are kept for readability against the
# historical Gradio 3 pin.
if gr is None:
    _MAJOR = 3
else:
    try:
        _MAJOR = int(str(gr.__version__).split('.', 1)[0])
    except (AttributeError, ValueError):  # pragma: no cover - defensive
        _MAJOR = 3

IS_GRADIO_4 = _MAJOR >= 4


def _patch_api_info_bool_schema() -> None:
    """Work around gradio-app/gradio#8237 in gradio-client 1.3.x.

    Symptom, seen live at first HTTP hit to the loopback health probe:

        File ".../gradio_client/utils.py", line 863, in get_type
            if "const" in schema:
        TypeError: argument of type 'bool' is not iterable

    Root cause: `get_api_info()` recurses through the OpenAPI schema. When a
    component emits `additionalProperties: true` (a bool, not a dict) — which
    happens for e.g. `gr.Number(precision=0)` and several others in our UI —
    `_json_schema_to_python_type()` forwards that bool into `get_type()`, and
    `get_type()` blindly does `"const" in schema`, which raises on bool.

    The knock-on effect is nastier than it looks: the health probe crashes,
    Gradio's loopback readiness check declares the server unreachable, and
    then throws the misleading:

        ValueError: When localhost is not accessible, a shareable link must be
        created. Please set share=True ...

    So the user thinks it's a networking / proxy issue when it's actually a
    schema-serialization crash. This blocks the app from ever finishing
    startup on Gradio 4.44.1 with the components we use.

    Fix: monkeypatch `_json_schema_to_python_type` so a bool schema returns
    early with the JSON-Schema-correct answer:

      * `additionalProperties: true`  →  any value allowed  →  "Any"
      * `additionalProperties: false` →  no extra properties →  "None"

    Everything else falls through to the original implementation. This is the
    community-standard workaround while Gradio 4.44.x stays pinned; upstream
    fixed it in 4.45.x but the fastapi/pydantic/starlette pin quartet we rely
    on won't lift until we do a separate migration.
    """
    try:
        from gradio_client import utils as _gcu  # type: ignore
    except Exception:
        # No gradio_client in this environment (unit-test or Gradio 3
        # fallback). Nothing to patch, and nothing broken to patch it for.
        return

    if getattr(_gcu, "_foocusrx_bool_schema_patch", False):
        return  # idempotent

    _orig = getattr(_gcu, "_json_schema_to_python_type", None)
    if _orig is None:
        return  # different gradio_client layout; leave it alone.

    def _patched(schema, defs):
        # Bool schemas are legal per JSON Schema; the shipped 4.44.1
        # implementation doesn't handle them. Handle them here and
        # delegate everything else to the original function.
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return _orig(schema, defs)

    _gcu._json_schema_to_python_type = _patched
    _gcu._foocusrx_bool_schema_patch = True


# Apply the patch at import time so that any downstream module (in
# particular webui.py, which imports gradio and then registers routes
# during startup) sees the patched function before Gradio's health
# probe fires. Cheap and idempotent.
_patch_api_info_bool_schema()


def js_kwarg(js_code: str | None) -> dict:
    """Return the correct kwarg dict for the event-listener JS hook.

    Gradio 3.x: ``_js='...'``
    Gradio 4.x: ``js='...'``
    """
    if js_code is None:
        return {}
    return {'js': js_code} if IS_GRADIO_4 else {'_js': js_code}


def image_source(source: str | list[str]) -> dict:
    """Return the correct kwarg dict for gr.Image source selection.

    Gradio 3.x: ``source='upload'``
    Gradio 4.x: ``sources=['upload']``
    """
    if IS_GRADIO_4:
        if isinstance(source, str):
            source = [source]
        return {'sources': source}
    # Gradio 3.x only accepts a single string.
    if isinstance(source, list):
        source = source[0]
    return {'source': source}


def sketch_image(*, label: str, color: str, height: int | None = None,
                 elem_id: str | None = None, show_label: bool = True,
                 image_type: str = 'numpy'):
    """Return an image component configured as a sketch/mask canvas.

    Uses ``gr.ImageEditor`` on Gradio 4+ with a fixed-color brush.
    The value returned to handlers is
    ``{'background': np.ndarray, 'layers': [np.ndarray], 'composite': np.ndarray}``
    with ``layers[0]`` carrying the brush strokes; call
    :func:`extract_sketch_mask` to unpack.

    Kept as a factory so the two sketch call sites stay symmetric — if
    a future Gradio release changes the ImageEditor API again, this is
    the only place to touch.
    """
    if gr is None:
        raise RuntimeError('gradio is not installed')
    kwargs = dict(
        label=label,
        sources=['upload'],
        type=image_type,
        brush=gr.Brush(colors=[color], color_mode='fixed'),
        layers=False,
    )
    if height is not None:
        kwargs['height'] = height
    if elem_id is not None:
        kwargs['elem_id'] = elem_id
    if not show_label:
        kwargs['show_label'] = False
    return gr.ImageEditor(**kwargs)


def update_sketch_brush_color(color: str):
    """Return a ``gr.update`` payload that changes the brush color on a
    sketch canvas across Gradio versions.

    On Gradio 3 the ``Image(tool='sketch')`` component accepted a
    ``brush_color=`` kwarg via ``gr.update(brush_color=...)``. On Gradio
    4's ``ImageEditor`` the brush is a nested object; the equivalent
    update swaps the whole ``brush=`` value.
    """
    if gr is None:
        raise RuntimeError('gradio is not installed')
    if IS_GRADIO_4:
        return gr.update(brush=gr.Brush(colors=[color], color_mode='fixed'))
    return gr.update(brush_color=color)


def _mask_to_2d(mask):
    """Return a single-channel 2D mask from a numpy array of arbitrary shape.

    Gradio 3's ``Image(tool='sketch')`` mask was ``(H, W, 3)`` uint8 with
    identical channels; taking channel 0 was the historical unpack.

    Gradio 4's ``ImageEditor(layers=False)`` returns a single ``(H, W, 4)``
    RGBA layer: the alpha channel carries the actual strokes and everywhere
    the user did not paint is transparent. Taking channel 0 there would give
    the RGB white value everywhere the layer contains anything, which is
    wrong. We prefer alpha when present.
    """
    if mask is None:
        return None
    import numpy as np
    if not isinstance(mask, np.ndarray):
        return mask
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            return mask[:, :, 3]  # alpha
        return mask[:, :, 0]
    return mask


def extract_sketch_mask(value):
    """Normalize a sketch-image value to ``(image, mask)`` numpy arrays.

    Handles both value shapes:

    * Gradio 3.x ``gr.Image(tool='sketch')`` returns
      ``{'image': np.ndarray, 'mask': np.ndarray}``.
    * Gradio 4.x ``gr.ImageEditor(layers=False)`` returns
      ``{'background': np.ndarray, 'layers': [np.ndarray, ...],
      'composite': np.ndarray}`` where ``layers[0]`` carries the brush
      strokes.

    Returns ``(None, None)`` when ``value`` is not a dict, matching the
    calling code's ``isinstance(..., dict)`` guard.
    """
    if not isinstance(value, dict):
        return None, None
    # Gradio 3 shape
    if 'image' in value and 'mask' in value:
        return value['image'], value['mask']
    # Gradio 4 shape (ImageEditor with layers=False)
    if 'background' in value and 'layers' in value:
        import numpy as np
        background = value.get('background')
        # Strip alpha from background if the ImageEditor returned RGBA — the
        # rest of the pipeline expects (H, W, 3) uint8.
        if isinstance(background, np.ndarray) and background.ndim == 3 and background.shape[2] == 4:
            background = background[:, :, :3]
        layers = value.get('layers') or []
        mask = layers[0] if layers else None
        return background, mask
    return None, None
