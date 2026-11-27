from collections import OrderedDict

import modules.core as core
import torch
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from modules.config import downloading_upscale_model

opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None              # the built-in Fooocus ESRGAN, loaded on first use
_extra = (None, None)     # (path, model): the last user model (models/upscale_models), one kept at a time


def _built_in():
    global model
    if model is None:
        model_filename = downloading_upscale_model()
        sd = torch.load(model_filename, weights_only=True)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()
    return model


def load_model(path=None):
    """The built-in model for ``path=None``, otherwise a user upscaler
    (ESRGAN, Real-ESRGAN, SwinIR, ... any architecture ldm_patched.pfn
    recognises) from a .pth / .pt / .safetensors file. Kept on the CPU
    between uses; ImageUpscaleWithModel moves it to the GPU per call."""
    global _extra
    if path is None:
        return _built_in()
    if _extra[0] == path:
        return _extra[1]
    import ldm_patched.modules.utils
    from ldm_patched.pfn import model_loading
    _extra = (None, None)  # free the previous one before loading the next
    sd = ldm_patched.modules.utils.load_torch_file(path, safe_load=True)
    if 'module.layers.0.residual_group.blocks.0.norm1.weight' in sd:
        sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {'module.': ''})
    loaded = model_loading.load_state_dict(sd).eval()
    loaded.cpu()
    _extra = (path, loaded)
    return loaded


def model_scale(path=None) -> int:
    return int(getattr(load_model(path), 'scale', 4) or 4)


def perform_upscale(img, model_path=None):
    """One pass of the upscaler model over an HWC uint8 image (tiled, so
    memory stays flat). ``model_path=None`` is the built-in Fooocus model."""
    print(f'Upscaling image with shape {str(img.shape)} ...')
    upscale_model = load_model(model_path)
    img = core.numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(upscale_model, img)[0]
    img = core.pytorch_to_numpy(img)[0]

    return img
