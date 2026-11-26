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

### FoocusRX app (Dock launcher)

```bash
venv/bin/python scripts/foocusrx.py build-app --install   # -> ~/Applications/FoocusRX.app
```

Drag **FoocusRX** from `~/Applications` to the Dock. Clicking it starts
FoocusRX in the background (port 7865) and opens it in your browser; clicking
it again while it runs offers **Open / Stop / Update / View log**. Logs go to
`~/Library/Logs/FoocusRX/`. If an update is needed (packages don't match
`requirements_versions.txt`) it offers to run the update instead of
installing silently. From a terminal: `scripts/foocusrx.py start|stop|status|log`.

### Updating

Double-click **Update FoocusRX.command** (or `scripts/foocusrx.py update`).
It fetches `origin/main`, fast-forwards, installs changed packages, then runs
the unit tests and a smoke launch (the app must start and report the pinned
Gradio version). If anything fails, the code (`git reset --keep`) and the
virtualenv (restored from an instant APFS copy) go back to where they were.
It only fast-forwards, never discards local changes (your files under
`models/` are fine), and keeps one `venv.bak-*` (`scripts/foocusrx.py
prune-backups --keep 0` removes it). `--check` reports without changing
anything; `--e2e` also runs the browser tests.

### Upgrading dependencies

Double-click **Upgrade Dependencies.command**, or:

```bash
scripts/foocusrx.py upgrade check                   # newer versions of the pinned packages
scripts/foocusrx.py upgrade test gradio==5.50.0     # try it in venv-candidate (your venv is untouched)
scripts/foocusrx.py upgrade apply [--dry-run]       # pull request with the new pins
scripts/foocusrx.py update                          # after merging: swaps the tested venv-candidate in
```

`test` clones the virtualenv, installs the versions there, and runs the unit
tests plus a smoke launch against the copy. torch/torchvision are not pinned
in `requirements_versions.txt` and are out of scope.

### Login when sharing on a network

With `--listen` on a non-loopback address or `--share`, add a login: copy
`auth-example.json` to `auth.json` and set your own user and password. Without
one, History delete and model downloads are disabled.

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
- **PR #22** — HTML-escape prompts and metadata in the History Log
  (`log.html`), closing a stored-XSS hole; `unittest discover` works again.
- **PR #23** — Loading settings from an A1111/Civitai image keeps its
  prompt and infers Performance from the step count; metadata tests.
- **PR #24** — Document why the generation worker's catch-all must stay
  broad (`except BaseException`).
- **PR #25** — Ignore `.claude/` local state.
- **PR #26** — Read the version with `ast` instead of `exec()`.
- **PR #27** — Tests for config loading and precedence (default preset <
  `config.txt` < `--preset` < env var).
- **PR #28** — Run the UNet natively in bf16 on M4+ instead of up-casting
  every layer to fp32: ~2.5x faster per SDXL image (0.9 s vs 2.7 s per
  step on an M5 Pro). `FOOOCUS_MPS_FP32_COMPUTE=1` restores the old path.
- **PR #29** — Prompt queue: one generation per prompt line.
- **PR #30** — History tab: browse, search and reuse past generations.
- **PR #31** — Model manager: list installed models and download
  `.safetensors` by URL with progress and checksum verification.
- **PR #32** — Draft mode: 12-step DPM++ 2M previews (~2.5x faster), then
  "Render full quality" for the one you pick.
- **PR #33** — Auto-apply the best settings for the selected checkpoint,
  based on the shipped presets and the model's family.
- **PR #34** — `scripts/auto_upscale.py` and `Upscale Images.command`:
  detailed 2x upscales of finished images while the app is closed.
- **PR #35** — FLUX.1 [schnell] (Q4 GGUF) text-to-image as a base model;
  download the files with `scripts/download_flux.py`.
- **PR #36** — Removed the `FOOOCUS_TORCH_COMPILE` option from PR #11 and
  its startup hint: it never took effect, and compiling the UNet itself
  fails on MPS today. See `docs/apple-silicon.md`.
- **PR #37** — Ignore `*.gguf` models and the model downloader's sidecar
  files.
- Krea 2 Turbo and MOSS-Image SFT as model families, run through
  stable-diffusion.cpp; download with `scripts/download_sdcpp.py`. See
  [`docs/krea2-moss-image.md`](./docs/krea2-moss-image.md).

## Upstream README

The original Fooocus documentation lives in
[`docs/upstream-readme.md`](./docs/upstream-readme.md). It was
previously tracked as `readme.md`, which on case-insensitive
filesystems (macOS/Windows) collided with `README.md` on disk — the
last one git touched would silently overwrite the other's content.
