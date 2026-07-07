# FoocusRX

A fork of [Fooocus](https://github.com/lllyasviel/Fooocus) by lllyasviel,
focused on Apple Silicon quality of life, reliability, and modern
dependency compatibility.

Upstream commit at fork time: `ae05379`.
Current version: see [`fooocus_version.py`](./fooocus_version.py).

## Install on macOS (recommended)

Grab the latest `FoocusRX-<version>-macOS.zip` from the
[releases page](https://github.com/mlovelace112-maker/FoocusRX/releases),
unzip, and double-click **Install FoocusRX.command**. See
[`docs/mac-installer.md`](./docs/mac-installer.md) for details.

If you already have the repo cloned:

```bash
./scripts/install_mac.command      # one-shot: venv + deps + launch
# or, once installed:
./scripts/mac_launch.sh
```

On M4 or newer, the launcher auto-installs `mtlflashattn` on first run.
See [`docs/apple-silicon.md`](./docs/apple-silicon.md).

## What's different from upstream

- **PR #5** — Removed global SSL bypass in `launch.py`; pinned the
  `gradio` / `starlette` / `fastapi` / `jinja2` / `pydantic` quartet.
- **PR #6** — Auto-updater is opt-in and refuses to overwrite a dirty
  working tree; `scripts/mac_launch.sh` clears proxies and sets
  `NO_PROXY` for the Gradio loopback health probe.
- **PR #7** — Apple Silicon chip detection with M5+ tuning; atomic
  model downloads; `torch>=2.5` on Darwin.
- **PR #8** — `FOOOCUS_PREFERRED_DTYPE` wiring, MPS SDPA path, and the
  `mtlflashattn` shim.
- **PR #9** — `ast.literal_eval` for metadata parsing, event-driven
  async worker, `channels_last` UNet layout.
- **PR #10** — Model checksum verification with a shipped hash registry.
- **PR #11** — Opt-in `torch.compile` for UNet on MPS/CUDA.
- **PR #12** — Resume interrupted model downloads via HTTP `Range`.
- **PR #13** — `gradio_compat` shim; route `_js` / `source` kwargs
  through it.
- **PR #14** — Migrate UI to Gradio 4.44.1 (net –371 LOC).
- **PR #15** — Post-Gradio-4 docs cleanup.
- **PR #16** — Auto-install `mtlflashattn` on first launch (M4+) with
  marker-file idempotency and `FOOOCUS_AUTO_INSTALL_MTLFLASHATTN=0`
  opt-out.
- **PR #17** — macOS installer (`scripts/install_mac.command`) and
  releasable installer bundle (`scripts/build_mac_installer.sh`).
- **PR #18** — Installer rejects Python 3.13+ and offers to
  `brew install python@3.12` instead.
- **PR #19** — Dropped a leftover `gr.Button(label=...)` from Gradio 3.
- **PR #20** — Patched a `gradio_client` bool-schema crash and pinned
  `setuptools<82`.
- **PR #21** — Fixed `gr.Dataset.update`, tuned the MPS low watermark,
  and fixed the `mtlflashattn` install marker.
- Cast `bfloat16` tensors to `float32` before `.numpy()` in
  `modules/core.py`; NumPy has no native bfloat16 support, so both the
  live preview and the final image decode crashed with
  `TypeError: Got unsupported ScalarType BFloat16` on Apple Silicon.

## Upstream README

The original Fooocus documentation lives in
[`docs/upstream-readme.md`](./docs/upstream-readme.md). It was
previously tracked as `readme.md`, which on case-insensitive
filesystems (macOS/Windows) collided with `README.md` on disk — the
last one git touched would silently overwrite the other's content.
