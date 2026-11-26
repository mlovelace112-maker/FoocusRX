# Krea 2 and MOSS-Image

FoocusRX runs two newer open-weight text-to-image models through
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
(`sd-cli`, Metal on Apple Silicon), started as a subprocess for each image:

| Family | Model | Files | Defaults |
| --- | --- | --- | --- |
| Krea 2 | Krea 2 Turbo, 12.9B DiT (Q5_K_S GGUF) + Qwen3-VL-4B (Q8_0) + Qwen-Image VAE | ~13.4 GB | 8 steps, CFG 1, euler / simple |
| MOSS-Image | MOSS-Image SFT, 5B DiT (Q8_0 GGUF) + Qwen3.5-9B (Q5_K_M) + FLUX.2 VAE + tokenizer | ~12.4 GB | 50 steps, CFG 3, euler, shift 5 |

## Install

```sh
venv/bin/python scripts/download_sdcpp.py krea2        # asks you to accept the Krea 2 Community License
venv/bin/python scripts/download_sdcpp.py moss_image
```

The script also installs the pinned engine (`master-1133-52491bd`, ~40 MB)
into `models/sdcpp/`. Every file is checked against its published size and
SHA-256. Then click **Refresh All Files** and pick the family under
**Model family**.

## What works

- Text-to-image with your prompt, styles (applied as text), wildcards,
  aspect ratio, seed and image number; Stop and Skip end the subprocess.
- MOSS-Image also uses the negative prompt, keeps line breaks (multi-line
  sign or poster text), and passes a JSON layout prompt through unchanged.
- Debug > "Forced Overwrite of Sampling Step" overrides the step count.
- Not available for these families: input image (vary / upscale /
  inpaint), image prompt, LoRAs, refiner, enhance.

## Tuning

On macOS every run passes `--auto-fit off --mmap --flash-attn on`
(`modules/sdcpp/app.py`, `default_args`). Without them, Krea 2 failed
through the app on a 24 GB Mac with "RAM graph capacity exceeded ...
diffusion compute failed". sd.cpp checks free and inactive RAM before each
graph, and these flags keep the weights as memory-mapped, reclaimable pages
and shrink the Krea 2 compute buffer from 3.8 GB to 1.2 GB. If a run still
fails, the app says there is not enough free memory; close other apps and
retry. `FOOOCUS_SDCPP_ARGS` adds more `sd-cli` flags after the defaults.
`--vae-tiling` saves memory at very large sizes but made the Krea 2 decode
3x slower in testing (280 s vs 92 s at 1024²).

Measured on an M5 Pro (24 GB), one 1024² image through the app, including
model loading: Krea 2 Turbo ~180 s (8 steps ~85 s, VAE decode ~92 s);
MOSS-Image SFT ~10 min at 50 steps (~137 s at 12 steps). The sd-cli
process's own footprint stays under 0.5 GB because the weights are mapped
from disk. Folders can be moved with `path_sdcpp`, `path_krea2` and
`path_moss_image` in `config.txt`. Other quantisations work too: put a GGUF
whose name starts with `Krea-2` / `MOSS-Image` next to the support files.

## Licences

Krea 2: Krea 2 Community License
(<https://huggingface.co/krea/Krea-2-Turbo/blob/main/LICENSE.pdf>).
MOSS-Image, Qwen3-VL and Qwen3.5: Apache-2.0. stable-diffusion.cpp: MIT.
