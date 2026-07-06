"""
Small runtime patches applied on top of Gradio.

Historically this module also shipped a full copy of Gradio 3's
``gr.Image`` component with extra ``brush_color`` / ``mask_opacity``
sketch kwargs. Gradio 4 replaced sketch mode with a first-class
``gr.ImageEditor`` component, and the Gradio-3 subclass depended on
internals that no longer exist (``gradio_client.serializing``,
``gradio.components.base._Keywords``, the ``Editable/Streamable/...``
mixin family, etc.). That copy has been deleted; call sites use
``modules.gradio_compat.sketch_image()`` instead.

What remains here:

1. ``Block.__init__`` monkey-patch that appends every constructed Block
   into ``all_components``. Fooocus's config dumper
   (``dump_english_config``) walks this list to pick up all UI labels
   for i18n. Compatible with both Gradio 3 and 4.

2. ``asyncio.wait_for`` patch inside ``gradio.routes``. Gradio's queue
   worker used to call ``asyncio.wait_for(fut, timeout=<small>)`` on
   long-running predictions, which caused Fooocus's multi-minute
   diffusion jobs to be cancelled. The patch swallows the timeout and
   uses a very large one instead. We keep it behind a
   ``hasattr(gradio.routes.asyncio, 'wait_for')`` guard so we degrade
   silently if a future Gradio removes the choke point.
"""

from __future__ import annotations

import importlib

import gradio
import gradio.routes
from gradio.blocks import Block


# --- Component tracker ------------------------------------------------------

all_components: list[Block] = []

if not hasattr(Block, '_foocusrx_original_init'):
    Block._foocusrx_original_init = Block.__init__

    def _tracking_init(self, *args, **kwargs):
        all_components.append(self)
        return Block._foocusrx_original_init(self, *args, **kwargs)

    Block.__init__ = _tracking_init


# --- asyncio.wait_for lockout ----------------------------------------------

# Best-effort: only patch if the attribute is actually present on the
# module Gradio's queue uses. On Gradio 4 the choke point still exists in
# ``gradio.routes`` for the same reasons it did on 3.x, but if a future
# release removes it, we degrade to a no-op instead of raising at import.

try:
    gradio.routes.asyncio = importlib.reload(gradio.routes.asyncio)
except Exception:  # pragma: no cover - environmental
    pass

if hasattr(gradio.routes, 'asyncio') and hasattr(gradio.routes.asyncio, 'wait_for'):
    if not hasattr(gradio.routes.asyncio, '_foocusrx_original_wait_for'):
        gradio.routes.asyncio._foocusrx_original_wait_for = gradio.routes.asyncio.wait_for

        def _patched_wait_for(fut, timeout):
            del timeout  # ignore caller's timeout; use a huge one instead
            return gradio.routes.asyncio._foocusrx_original_wait_for(fut, timeout=65535)

        gradio.routes.asyncio.wait_for = _patched_wait_for
