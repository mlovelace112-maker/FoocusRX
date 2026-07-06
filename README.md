# FoocusRX

A fork of [Fooocus](https://github.com/lllyasviel/Fooocus) by lllyasviel, with
Mac Silicon quality-of-life and reliability improvements in progress.

Upstream commit at fork time: `ae05379`

## Planned changes

- Remove global SSL verification bypass in `launch.py`.
- Guard the auto-updater in `entry_with_update.py` against dirty working trees.
- Pin the `gradio` / `starlette` / `fastapi` / `jinja2` quartet to a
  known-good set (fixes upstream #4180 "unhashable type: 'dict'").
- Bypass proxy env vars for loopback so Gradio's health probe stops failing
  with "When localhost is not accessible" on macOS.
- Detect Apple Silicon at install time and install torch >= 2.4 (MPS) instead
  of the hard-coded CUDA 12.1 wheel.
- Atomic downloads + size/SHA verification in `modules/model_loader.py`.

## Upstream README

The original Fooocus documentation lives in [`readme.md`](./readme.md).
