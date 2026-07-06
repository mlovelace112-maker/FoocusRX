# FoocusRX

A fork of [Fooocus](https://github.com/lllyasviel/Fooocus) by lllyasviel, with
Mac Silicon quality-of-life and reliability improvements in progress.

Upstream commit at fork time: `ae05379`

## Changes vs upstream

Shipped:

- Removed global SSL verification bypass in `launch.py` (PR #5).
- Pinned the `gradio` / `starlette` / `fastapi` / `jinja2` / `pydantic`
  quartet to a known-good set (fixes upstream
  [#4180](https://github.com/lllyasviel/Fooocus/issues/4180)
  "unhashable type: 'dict'") — PR #5.
- Auto-updater in `entry_with_update.py` is now opt-in and refuses to
  overwrite a dirty working tree.
- `scripts/mac_launch.sh`: launch wrapper that clears proxy env vars
  and sets `NO_PROXY` so Gradio's loopback health probe stops failing
  with "When localhost is not accessible" on macOS.

Planned:

- Detect Apple Silicon at install time and install torch >= 2.4 (MPS)
  instead of the hard-coded CUDA 12.1 wheel.
- Atomic downloads + size/SHA verification in `modules/model_loader.py`.

## Quickstart on Apple Silicon

```bash
conda env create -f environment.yaml
conda activate fooocus
pip install -r requirements_versions.txt
./scripts/mac_launch.sh
```

## Upstream README

The original Fooocus documentation lives in [`readme.md`](./readme.md).
