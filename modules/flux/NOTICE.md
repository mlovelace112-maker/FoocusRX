# Third-party code in `modules/flux`

| File(s) | Ported from | Licence |
|---|---|---|
| `autoencoder.py`, `model.py`, `layers.py` (incl. RoPE/attention), sampling in `pipeline.py` | [black-forest-labs/flux](https://github.com/black-forest-labs/flux) @ `802fb4713906133fcbd0d8dc5351620ca4773036` (`src/flux/`) | Apache-2.0 |
| `gguf_ops.py` (dequantization, on-the-fly quantized linear) | [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) @ `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` | Apache-2.0 |
| `t5_tokenizer/` (tokenizer files), T5-XXL config | [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) @ `1f13e29b267f051b84e5c5476ee7a65badb28288` (`comfy/text_encoders/`) | GPL-3.0 (same as FoocusRX) |

Changes from the originals are limited to: removing unused features (LoRA,
img2img/conditioning variants, ONNX/TensorRT paths), adapting imports, and
the MPS memory handling described in each file. Model weights are not part
of this repository; FLUX.1 [schnell] weights are released under Apache-2.0
by Black Forest Labs.
