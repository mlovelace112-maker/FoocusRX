# Auto upscale

With **Auto upscale** on (below Image Number; on by default), every image a
run saves gets an upscaled copy when the run finishes:
`2026-11-27/2026-11-27_10-15-02_1234.png` ->
`2026-11-27/2026-11-27_10-15-02_1234_up4x.png`. The copy shows up in the
gallery right after its original, and in History with the original's
settings plus "Auto upscale", "Upscaled size" and "Upscaled from".

## Settings (Auto upscale settings, under the checkbox)

| Setting | Default | |
|---|---|---|
| Scale | 4x | 1.5x, 2x, 3x or 4x. The long side is capped at 8192 px. |
| Method | Detailed | See below. |
| Upscaler model | Built-in (Fooocus ESRGAN) | Any model in `models/upscale_models`. |
| Keep original | On | Off moves the original to the Trash once its upscale is saved. |

Change the defaults in `config.txt`: `default_auto_upscale`,
`default_auto_upscale_scale` (`"4x"`), `default_auto_upscale_method`,
`default_auto_upscale_model`, `default_auto_upscale_keep_original`.

## Methods

**Detailed (slower, adds detail).** What Input Image > Upscale (2x) does,
applied automatically:

1. The upscaler model enlarges the image to at most 2x.
2. A light diffusion pass redraws fine detail with the image's own model,
   LoRAs, prompt and seed (SDXL: denoise 0.382, or your "Forced Overwrite
   of Denoising Strength of Upscale"; FLUX.1: a tiled refine at strength
   0.35).
3. The upscaler model takes it the rest of the way (4x = 2x detail pass,
   then 2x more).

The diffusion pass stays at 2x because its cost grows much faster than the
image: a full-size 4x pass would take many times longer and need far more
memory, for little gain. Expect a minute or more per image on an M5 at 4x.

**Fast (upscaler model only).** No diffusion pass; seconds per image.

Krea 2 and MOSS-Image (stable-diffusion.cpp) images always use the upscaler
model only for now, and say so in their "Auto upscale note".

## Not upscaled

- Draft mode previews (use "Render full quality", which is upscaled).
- Results of Input Image > Upscale (they're upscales already).
- Enhance steps saved only to the temp folder ("Save only final enhanced
  image").
- Anything after you press Stop. Skip skips the image being upscaled.

## More upscaler models

Put `.pth`, `.pt` or `.safetensors` upscalers (ESRGAN, Real-ESRGAN, SwinIR,
e.g. `4x-UltraSharp.pth`, `RealESRGAN_x4plus.pth`) in
`models/upscale_models` (or your `path_upscale_models`), then press the
refresh button next to "Upscaler model". A 4x model used for a 2x step
upscales 4x and is then resized down, which is normal.

`Upscale Images.command` (detailed 2x while the app is closed) skips images
that already have an auto upscale.
