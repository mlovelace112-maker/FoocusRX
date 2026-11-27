# https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py 

import os
from ldm_patched.pfn import model_loading
from ldm_patched.modules import model_management
import torch
import ldm_patched.modules.utils
import ldm_patched.utils.path_utils

class UpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (ldm_patched.utils.path_utils.get_filename_list("upscale_models"), ),
                             }}
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name):
        model_path = ldm_patched.utils.path_utils.get_full_path("upscale_models", model_name)
        sd = ldm_patched.modules.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {"module.":""})
        out = model_loading.load_state_dict(sd).eval()
        return (out, )


class ImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "image": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        device = model_management.get_torch_device()
        # Make room first: the upscaler isn't tracked by model_management, so
        # without this an SDXL UNet left on the GPU competes with it.
        model_bytes = sum(p.numel() * p.element_size() for p in upscale_model.parameters())
        model_management.free_memory(model_bytes * 1.3 + model_management.minimum_inference_memory(), device)

        tile = 512
        overlap = 32
        freed_models = False
        s = None
        in_img = None
        try:
            upscale_model.to(device)
            in_img = image.movedim(-1,-3).to(device)
            while s is None:
                oom = False
                try:
                    steps = in_img.shape[0] * ldm_patched.modules.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = ldm_patched.modules.utils.ProgressBar(steps)
                    s = ldm_patched.modules.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                except Exception as e:
                    # MPS raises a plain RuntimeError for out-of-memory.
                    if not model_management.is_oom_error(e) or (tile // 2 < 128 and freed_models):
                        raise
                    oom = True
                if oom:
                    # Outside the except block, so the failed tile's tensors are gone.
                    if not freed_models:
                        # First time: unload every other model, then retry the same tile size.
                        print(f"[upscale] out of memory at {tile}px tiles; unloading other models and retrying.")
                        model_management.unload_all_models()
                        freed_models = True
                    else:
                        model_management.soft_empty_cache(force=True)
                        tile //= 2
                        print(f"[upscale] out of memory again; retrying with {tile}px tiles.")
        finally:
            upscale_model.cpu()
            in_img = None
            model_management.soft_empty_cache(force=True)
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return (s,)

NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoader": UpscaleModelLoader,
    "ImageUpscaleWithModel": ImageUpscaleWithModel
}
