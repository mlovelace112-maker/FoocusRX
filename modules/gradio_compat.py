"""
Gradio 3/4 compatibility shim.

Purpose
-------
Prep work for a future Gradio 4 migration. All call sites in webui.py go
through the helpers here so the eventual pin bump only has to flip a
single constant. Behavior on the current pinned Gradio (3.41.2) is
unchanged.

What this module intentionally does NOT do
------------------------------------------
- Import from Gradio internals (`gradio_client.serializing`,
  `gradio.deprecation`, `gradio.components.base._Keywords`, etc.). Those
  were deleted in Gradio 4 and would break the pin bump.
- Subclass any Gradio component. `modules/gradio_hijack.py` still does
  that on 3.41.2; this shim complements it.

Usage
-----
    from modules import gradio_compat as gc

    # event-listener JS kwarg (Gradio 3 uses `_js`, Gradio 4 uses `js`)
    stop_button.click(fn, **gc.js_kwarg('cancelGenerateForever'), ...)

    # image source kwarg (Gradio 3: source='upload', 4: sources=['upload'])
    gr.Image(**gc.image_source('upload'), type='numpy', ...)

    # image-editor kwargs for sketch canvases (compat wrapper)
    gc.sketch_image(label='Image', color='#FFFFFF', height=500,
                    elem_id='inpaint_canvas', show_label=False)
"""

from __future__ import annotations

try:  # pragma: no cover - environmental
    import gradio as gr
except ImportError:  # pragma: no cover - environmental
    gr = None  # type: ignore[assignment]

# Gradio-4 flag. Detect at import time; only affects the kwarg names we
# emit. Stays False on Gradio 3.x — the currently pinned version.
if gr is None:
    _MAJOR = 3
else:
    try:
        _MAJOR = int(str(gr.__version__).split('.', 1)[0])
    except (AttributeError, ValueError):  # pragma: no cover - defensive
        _MAJOR = 3

IS_GRADIO_4 = _MAJOR >= 4


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
                 mask_opacity: float | None = None, image_type: str = 'numpy'):
    """Return an image component configured as a sketch/mask canvas.

    On Gradio 3.x this returns a ``gradio_hijack.Image`` with the
    ``tool='sketch'`` + ``brush_color=...`` combo that Fooocus has always
    used. On Gradio 4.x this will return a ``gr.ImageEditor`` with the
    equivalent ``brush=gr.Brush(colors=[color], color_mode='fixed')``
    configuration. The value returned to handlers changes shape between
    the two lines; ``modules/async_worker.py`` handles both via
    ``extract_mask()``.

    Kept as a factory so the two sketch call sites stay symmetric and
    the migration touches one file, not seven.
    """
    if IS_GRADIO_4:
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

    # Gradio 3.x path — imported lazily to avoid a circular import
    # (gradio_hijack imports this module transitively via webui.py).
    from modules import gradio_hijack as grh
    kwargs = dict(
        label=label,
        source='upload',
        type=image_type,
        tool='sketch',
        brush_color=color,
    )
    if height is not None:
        kwargs['height'] = height
    if elem_id is not None:
        kwargs['elem_id'] = elem_id
    if not show_label:
        kwargs['show_label'] = False
    if mask_opacity is not None:
        kwargs['mask_opacity'] = mask_opacity
    return grh.Image(**kwargs)


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
        background = value.get('background')
        layers = value.get('layers') or []
        mask = layers[0] if layers else None
        return background, mask
    return None, None
