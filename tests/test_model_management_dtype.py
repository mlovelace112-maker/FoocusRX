import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import os
import unittest
from unittest import mock

import torch

import args_manager  # noqa: F401  (sets up ldm_patched args before model_management)

# model_management probes the GPU at import time and raises on machines with
# neither MPS nor CUDA available (e.g. CI, or a sandbox that hides Metal).
try:
    import ldm_patched.modules.model_management as mm
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the host
    mm = None
    _IMPORT_ERROR = exc

MPS = torch.device('mps')
CPU = torch.device('cpu')


@unittest.skipIf(mm is None, f'model_management unavailable on this host: {_IMPORT_ERROR!r}')
class TestUnetManualCast(unittest.TestCase):
    def cast(self, weight_dtype, device, **env):
        clean = {k: v for k, v in os.environ.items()
                 if k not in ('FOOOCUS_PREFERRED_DTYPE', 'FOOOCUS_MPS_FP32_COMPUTE')}
        with mock.patch.dict(os.environ, {**clean, **env}, clear=True):
            return mm.unet_manual_cast(weight_dtype, device)

    def test_mps_bf16_hint_computes_natively(self):
        self.assertIsNone(self.cast(torch.bfloat16, MPS, FOOOCUS_PREFERRED_DTYPE='bfloat16'))

    def test_mps_fp16_hint_computes_natively(self):
        self.assertIsNone(self.cast(torch.float16, MPS, FOOOCUS_PREFERRED_DTYPE='float16'))

    def test_opt_out_restores_fp32_compute(self):
        self.assertEqual(
            self.cast(torch.bfloat16, MPS, FOOOCUS_PREFERRED_DTYPE='bfloat16', FOOOCUS_MPS_FP32_COMPUTE='1'),
            torch.float32)

    def test_mps_without_hint_is_unchanged(self):
        # M1-M3 get no hint; a bf16 checkpoint there still computes in fp32.
        self.assertEqual(self.cast(torch.bfloat16, MPS), torch.float32)

    def test_weight_dtype_not_matching_hint_is_unchanged(self):
        self.assertEqual(self.cast(torch.float16, MPS, FOOOCUS_PREFERRED_DTYPE='bfloat16'), torch.float32)

    def test_fp32_weights_never_cast(self):
        self.assertIsNone(self.cast(torch.float32, MPS, FOOOCUS_PREFERRED_DTYPE='bfloat16'))

    def test_cpu_is_unaffected_by_hint(self):
        self.assertEqual(self.cast(torch.bfloat16, CPU, FOOOCUS_PREFERRED_DTYPE='bfloat16'), torch.float32)


if __name__ == '__main__':
    unittest.main()
