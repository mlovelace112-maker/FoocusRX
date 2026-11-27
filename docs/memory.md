# Memory: loading, unloading and swap

FoocusRX runs several kinds of models: SDXL (base, refiner, LoRAs, VAE,
CLIP, ControlNets), FLUX.1, Krea 2 / MOSS-Image through stable-diffusion.cpp,
upscaler models for auto upscale, and Ollama models for Prompt Studio. On
Apple Silicon they all share one pool of unified memory, so whatever one
stage leaves loaded is memory the next stage can't have.

## What gets unloaded, and when

| When | What happens |
|------|--------------|
| You switch SDXL checkpoints | The old checkpoint (and its LoRA-patched copies, CLIP and VAE) is unloaded *before* the new one loads, so the two are never in memory together. A synthetic refiner built from the old checkpoint goes with it. |
| You switch refiners | Same for the old refiner (unless it shares weights with the base). |
| You switch model family | SDXL ↔ FLUX.1 ↔ Krea 2 / MOSS: the other families are freed first. Before sd.cpp runs, everything this process holds is freed: SDXL, FLUX.1, upscalers, the prompt rewriter and Prompt Studio's Ollama models. |
| You click Generate | Prompt Studio asks Ollama to unload its models (`keep_alive: 0`); the generation now waits (up to 15 s) for that to finish before loading the image model. |
| Auto upscale finishes | The upscaler model is dropped and the cache released. |
| An upscale pass runs | Room is made for the upscaler first (it isn't tracked by the SDXL model manager); afterwards it goes back to the CPU and the cache is released, even if the pass failed. |
| No generation for 10 minutes | Loaded SDXL models move off the GPU (they stay in RAM, where macOS can compress or swap them), FLUX.1 is unloaded, Ollama is asked to unload, and the cache is released. The next generation moves them back. `FOOOCUS_IDLE_UNLOAD_MINUTES` changes the delay; `0` turns it off. |

"Released the cache" means: Python's garbage collector runs (model graphs
often sit in reference cycles), queued GPU work finishes, and PyTorch hands
its cached blocks back to the system (`torch.mps.empty_cache()`).

`.pth` / `.pt` / `.ckpt` / `.bin` files saved in PyTorch's zip format are
now memory-mapped when loaded, as `.safetensors` files already were: the
weights are read from the file as needed, and macOS can drop those pages
under pressure instead of writing them to swap.

## Running out of memory

On MPS an out-of-memory error is a plain `RuntimeError("MPS backend out of
memory ...")`. The upstream fallbacks only caught CUDA's error class, so on a
Mac they never ran. Now:

- **Sampling**: if a diffusion run runs out of memory, every model is moved
  off the GPU, the cache is released, and the run is tried once more with the
  same seed.
- **VAE decode/encode**: falls back to tiled decoding/encoding.
- **Upscaler model**: the first time, every other model is unloaded and the
  same tile size is retried; after that the tiles are halved (512 → 256 →
  128 px) until it fits.
- **Auto upscale, Detailed**: if the detail pass still runs out of memory,
  that image is redone with the upscaler model only (tiled), and the upscaled
  file's metadata says so.
- **Anything else**: the generation stops with a message in the terminal, and
  everything is unloaded so the next generation starts clean.

## Swap ("allow swap", on by default)

macOS decides what goes to swap; an app can't send memory there itself. What
FoocusRX controls is when PyTorch gives up, through
`PYTORCH_MPS_HIGH_WATERMARK_RATIO` (a multiple of Metal's recommended working
set, about 3/4 of physical memory), set at launch before torch loads:

| Unified memory | Allow swap on (default) | Allow swap off |
|----------------|-------------------------|----------------|
| 32 GB or more  | `0.0` (no cap)          | `1.0`          |
| under 32 GB    | `1.7` (about 1.25× physical memory, PyTorch's own default) | `0.85` (16–31 GB), `0.7` (under 16 GB) |

With it on, a generation that needs a little more than the Mac has keeps
going and macOS swaps the overflow. FoocusRX's own planning still aims to stay
inside physical memory (it unloads models as if the cap were 1.0), so swap
only absorbs peaks.

**Heavy swapping is slow**, often several times slower than a run that fits,
and the Mac can feel sluggish meanwhile. It also **writes to the SSD**, and
SSDs wear with writes. Occasional swapping is fine; if every generation
swaps a lot, use a smaller size or fewer images per run, or turn it off.

To turn it off, either:

- put `"allow_swap": false` in `config.txt`, or
- start FoocusRX with `FOOOCUS_ALLOW_SWAP=0`.

It takes effect on the next launch. With it off, PyTorch raises out-of-memory
before going past physical memory, and the recovery steps above take over.
An explicit `PYTORCH_MPS_HIGH_WATERMARK_RATIO` always wins over both.

## The `[memory]` lines

The terminal prints one line before and after each generation, after
loading a checkpoint, before sd.cpp runs and after auto upscale:

```
[memory] after generation: MPS 6.1 GB in use (8.2 GB held) · RAM 18.4 GB of 24.0 GB · swap 0.3 GB
```

"In use" is what tensors occupy; "held" includes PyTorch's cache. A growing
"held" with nothing loading is the sign of a leak; a growing "swap" means
macOS is swapping. `FOOOCUS_MEMORY_LOG=0` turns the lines off.
