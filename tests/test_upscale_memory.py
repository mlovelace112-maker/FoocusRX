import sys

# args_manager parses sys.argv at import time; drop the test runner's own arguments.
sys.argv = sys.argv[:1]

import importlib
import unittest
from unittest import mock

from modules import memory_manager

try:
    import torch
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - CI installs CPU torch
    torch = None
    _IMPORT_ERROR = exc


def fake_model_management():
    fake = mock.Mock()
    fake.get_torch_device.return_value = torch.device('cpu')
    fake.minimum_inference_memory.return_value = 1024
    fake.is_oom_error.side_effect = memory_manager.is_oom_error
    return fake


_MM = 'ldm_patched.modules.model_management'
_NODE = 'ldm_patched.contrib.external_upscale_model'


def load_node(fake_mm):
    """A fresh ImageUpscaleWithModel whose model_management is ``fake_mm``
    (the real one picks a GPU at import, which CI doesn't have). Only the two
    modules involved are swapped in sys.modules, and both are put back."""
    saved = {name: sys.modules.get(name) for name in (_MM, _NODE)}
    sys.modules[_MM] = fake_mm
    sys.modules.pop(_NODE, None)
    try:
        module = importlib.import_module(_NODE)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    module.model_management = fake_mm
    return module.ImageUpscaleWithModel()


if torch is not None:
    class NearestUpscaler(torch.nn.Module):
        """2x nearest-neighbour 'upscaler' that runs out of memory on tiles
        larger than ``max_tile`` px, like an MPS model would on a big tile."""
        scale = 2

        def __init__(self, max_tile=10_000, error=None):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.max_tile = max_tile
            self.error = error
            self.sizes = []

        def forward(self, x):
            self.sizes.append(x.shape[-1])
            if self.error is not None:
                raise self.error
            if x.shape[-1] > self.max_tile:
                raise RuntimeError('MPS backend out of memory (MPS allocated: 17.10 GB, other allocations: '
                                   '1.20 GB, max allowed: 18.13 GB). Tried to allocate 1.00 GB on private pool.')
            return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest') * self.weight


@unittest.skipIf(torch is None, f'torch unavailable: {_IMPORT_ERROR!r}')
class TestUpscaleOutOfMemory(unittest.TestCase):
    def image(self, size=600):
        return torch.rand(1, size, size, 3)

    def test_makes_room_before_upscaling(self):
        fake = fake_model_management()
        node = load_node(fake)
        out, = node.upscale(NearestUpscaler(), self.image(64))
        self.assertEqual(tuple(out.shape), (1, 128, 128, 3))
        fake.free_memory.assert_called_once()
        fake.unload_all_models.assert_not_called()

    def test_out_of_memory_unloads_other_models_then_shrinks_tiles(self):
        fake = fake_model_management()
        node = load_node(fake)
        model = NearestUpscaler(max_tile=256)
        with mock.patch('builtins.print'):
            out, = node.upscale(model, self.image(600))
        self.assertEqual(tuple(out.shape), (1, 1200, 1200, 3))
        fake.unload_all_models.assert_called_once()  # first OOM: free the other models
        self.assertIn(256, model.sizes)              # then 512 -> 256 px tiles
        self.assertEqual(model.weight.device.type, 'cpu')

    def test_gives_up_below_128_px_tiles(self):
        fake = fake_model_management()
        node = load_node(fake)
        model = NearestUpscaler(max_tile=64)
        with mock.patch('builtins.print'), self.assertRaises(RuntimeError):
            node.upscale(model, self.image(600))
        self.assertEqual(min(model.sizes), 128)

    def test_other_errors_are_not_swallowed_and_model_goes_back_to_cpu(self):
        fake = fake_model_management()
        node = load_node(fake)
        model = NearestUpscaler(error=ValueError('broken model'))
        with self.assertRaises(ValueError):
            node.upscale(model, self.image(64))
        fake.unload_all_models.assert_not_called()
        fake.soft_empty_cache.assert_called()  # cache released on the way out
        self.assertEqual(model.weight.device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
