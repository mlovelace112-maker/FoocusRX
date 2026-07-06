# Apple Silicon on FoocusRX

FoocusRX auto-detects Apple Silicon chip generation at launch and tunes
the MPS backend accordingly. This document explains what it does and
which optional packages give the biggest speedups.

## What auto-tuning does

On `arm64 Darwin`, `modules/apple_silicon.configure()` runs before
anything imports torch and sets a handful of environment variables:

| Env var                              | When set                            | Value                                     |
|--------------------------------------|-------------------------------------|-------------------------------------------|
| `FOOOCUS_PREFERRED_DTYPE`            | M4 or newer (BF16 hardware present) | `bfloat16`                                |
| `FOOOCUS_HAS_NEURAL_ACCELERATORS`    | M5 or newer                         | `1`                                       |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO`   | Unless user has set it              | `0.0` (≥32 GB), `0.85` (16–31), `0.7` (<16) |
| `PYTORCH_ENABLE_MPS_FALLBACK`        | Unless user has set it              | `1`                                       |
| `PYTORCH_MPS_PREFER_METAL`           | Always (Darwin arm64)               | `1` (ignored by torch < 2.5)              |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS`| P-core count detected               | e.g. `6` on M5 Pro                        |

`FOOOCUS_PREFERRED_DTYPE` is now wired into the model-management stack:

- `ldm_patched/modules/model_management.py` — `unet_dtype`,
  `text_encoder_dtype`, `should_use_fp16`, and the `VAE_DTYPE` selection
  all honor the hint when running on MPS. Upstream returned `float32`
  for every one of these on MPS, wasting ~2× memory.
- `modules/patch_clip.py` and `modules/core.py` — CLIP-vision and
  VAE-approx models pick up bf16 too.
- Attention: `ENABLE_PYTORCH_ATTENTION` is now set on MPS whenever
  torch ≥ 2. This routes attention through `F.scaled_dot_product_attention`
  (the shim point mtlflashattn hooks) instead of the O(N²) sub-quadratic
  fallback.

The chip is read from `sysctl machdep.cpu.brand_string`, e.g. `Apple M5 Pro`,
and the generation is parsed from the M-number.

Every value is set via `setdefault` — if you've already exported something,
FoocusRX will not override it.

To see the detected chip and applied settings without launching:

```bash
python -m modules.apple_silicon
```

## Big wins on M5+

The M5 series introduced **per-GPU-core Neural Accelerators** —
dedicated matmul hardware inside every GPU core. Stock PyTorch MPS
doesn't fully exploit these yet, but community kernels do. Two are
worth installing:

### `mtlflashattn` — attention (highly recommended)

```bash
pip install -r requirements_mac.txt   # or:
pip install mtlflashattn
```

FoocusRX detects `mtlflashattn` at launch and prints `activated` or
`not-installed` in the `[apple_silicon]` boot log. On M5+ chips, when
the package is missing, it also prints an install hint.

- **Speed:** 3–11× faster than stock fused MPS SDPA on M5, driven by
  the Neural Accelerators via TensorOps `matmul2d`.
- **Correctness:** the stock fused MPS SDPA is silently numerically
  wrong past ~4k tokens (per-element errors up to ~28 on real DiT q/k/v,
  producing grid artifacts / variance collapse). `mtlflashattn`
  accumulates softmax/output state in fp32 and matches a chunked-fp32
  reference to ~1e-4.
- **Zero-config:** installs a `.pth` file that transparently replaces
  `flash_attn` on MPS. Kill it with `MTLFLASHATTN_SHIM=off`.
- **Requirements:** `torch >= 2.5`, macOS 26+ for the fast tier (v2),
  macOS 27+ for the fastest tier (v2r).

### `torch.compile` — UNet graph compilation (opt-in)

```bash
FOOOCUS_TORCH_COMPILE=1 ./scripts/mac_launch.sh
```

When set, the loaded UNet is passed through `torch.compile` for another
10–30% throughput on M5-class Macs and modern CUDA. Uses
`mode="reduce-overhead"` by default; override with
`FOOOCUS_TORCH_COMPILE_MODE=default|reduce-overhead|max-autotune`.

**Warmup cost.** The first inference after model load pays a one-time
compile cost — typically 30–120s on M5, longer on CUDA with
`max-autotune`. Every subsequent inference benefits.

**Fallback is silent.** If the UNet doesn't trace cleanly (rare custom
ops in a third-party LoRA, older torch, etc.) the loader logs
`torch.compile skipped` and falls back to eager mode. Nothing breaks.

**Requires** `torch >= 2.5`. FoocusRX pins `torch>=2.5,<3` on Darwin
arm64 already.

### `fp4-fp8-for-torch-mps` — sub-byte weights (optional)

```bash
pip install fp4-fp8-for-torch-mps
```

Adds FP8/FP4 sub-byte dtype support to PyTorch MPS via Metal shaders.
Not currently needed by SDXL (Fooocus's default models are fp16/bf16),
but useful if you swap in an FP8-quantized checkpoint from ComfyUI.

## Memory notes

- Upstream Fooocus sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
  unconditionally, which disables MPS's OOM guard. On a 64+ GB M5 Max
  that's fine; on a 16 GB M-series machine it can push the whole
  laptop into swap and lock up the UI. FoocusRX picks the value based
  on `hw.memsize`.
- You can override at any time:
  `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9`
- To see current MPS memory pressure:
  `python -c "import torch; print(torch.mps.current_allocated_memory()/1e9, 'GB')"`

## Torch install

`launch.py` now picks the right torch install command per-platform:

- **Apple Silicon:** `pip install --upgrade 'torch>=2.5,<3'` from PyPI
  (MPS-enabled arm64 wheel; no `--extra-index-url` needed).
- **Intel Mac:** same command; PyPI serves the CPU wheel.
- **Linux/Windows:** unchanged — `torch==2.1.0 torchvision==0.16.0`
  from the CUDA 12.1 index. Override with `TORCH_COMMAND` /
  `TORCH_INDEX_URL` env vars.

## Troubleshooting

**"When localhost is not accessible, a shareable link must be created"**

Use `scripts/mac_launch.sh` — it clears proxy env vars and sets
`NO_PROXY=localhost,127.0.0.1,::1` so Gradio's httpx-based health probe
stops routing loopback through a proxy. See
[gradio-app/gradio#4046](https://github.com/gradio-app/gradio/issues/4046).

**mtlflashattn imports fine but produces artifacts**

Set `MTLFLASHATTN_SHIM=off` to fall back to stock MPS SDPA. Please
report the failure to the mtlflashattn maintainers with your macOS and
torch versions — the shim aims for full correctness parity.

**Model download interrupted by Wi-Fi drop or laptop sleep**

FoocusRX resumes interrupted downloads using HTTP `Range` requests.
Partial files are stored as `<name>.partial` alongside a small
`<name>.partial.meta` sidecar that pins the source URL and expected
size; on relaunch the loader validates the sidecar before sending
`Range: bytes=<offset>-` and falls back to a full restart if the server
responds `200` (Range ignored) or the offset is inconsistent. Servers
that return `416`/`400` on the Range request trigger one automatic
retry from byte 0. Disable resume entirely with
`FOOOCUS_DOWNLOAD_RESUME=0` (falls back to atomic full-file downloads).

**Model file appears corrupted after Wi-Fi drop or laptop sleep**

FoocusRX ships SHA256 hashes for all first-party model downloads in
`models/hashes.json`. On launch the loader validates the on-disk file
against the shipped hash and quarantines corrupted files as
`<name>.corrupt` before triggering a fresh download. A `.sha256.ok`
sidecar caches the last successful verification so we don't rehash
unchanged multi-gigabyte checkpoints every launch. Disable with
`FOOOCUS_SKIP_HASH_CHECK=1` (size checks still run).

**Black images at resolutions above 2048px**

Known upstream MPS bug in some torch versions. If it hits you,
`ComfyUI-AppleSilicon-FP8` is a community node that includes a fix for
this specific issue; it can be adapted.

**`Trying to convert Float8_e4m3fn to the MPS backend`**

MPS can't natively hold float8 tensors. Either install
`fp4-fp8-for-torch-mps` (adds it via Metal shaders) or use a bf16
checkpoint instead. Fooocus's default SDXL checkpoints are fp16, so
this only bites if you swap in an FP8-quantized model.
