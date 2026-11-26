import sys

sys.argv = sys.argv[:1]  # args_manager parses sys.argv at import time

import os
import unittest
from unittest import mock

import args_manager

try:
    import ldm_patched.modules.model_management as mm
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the host
    mm = None
    _IMPORT_ERROR = exc

GB = 1024 ** 3


class TestOffloadDefault(unittest.TestCase):
    def test_unified_memory_never_offloads_by_default(self):
        self.assertFalse(args_manager.always_offload(False, False, unified_memory=True))

    def test_other_hosts_keep_the_upstream_default(self):
        self.assertTrue(args_manager.always_offload(False, False, unified_memory=False))
        self.assertFalse(args_manager.always_offload(False, True, unified_memory=False))

    def test_explicit_flag_still_forces_offload(self):
        self.assertTrue(args_manager.always_offload(True, False, unified_memory=True))


@unittest.skipIf(mm is None, f'model_management unavailable on this host: {_IMPORT_ERROR!r}')
class TestMpsFreeMemory(unittest.TestCase):
    def free(self, recommended, driver, **env):
        clean = {k: v for k, v in os.environ.items()
                 if k not in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'FOOOCUS_MPS_PSUTIL_FREE_MEMORY')}
        with mock.patch.dict(os.environ, {**clean, **env}, clear=True), \
                mock.patch.object(mm.torch.mps, 'recommended_max_memory', return_value=recommended), \
                mock.patch.object(mm.torch.mps, 'driver_allocated_memory', return_value=driver):
            return mm._mps_free_memory()

    def test_budget_is_watermark_minus_allocator(self):
        self.assertEqual(self.free(20 * GB, 5 * GB, PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.85'), 12 * GB)

    def test_no_ratio_uses_recommended_size(self):
        self.assertEqual(self.free(20 * GB, 5 * GB), 15 * GB)

    def test_ratio_zero_or_above_one_never_exceeds_recommended(self):
        self.assertEqual(self.free(20 * GB, 5 * GB, PYTORCH_MPS_HIGH_WATERMARK_RATIO='0'), 15 * GB)
        self.assertEqual(self.free(20 * GB, 5 * GB, PYTORCH_MPS_HIGH_WATERMARK_RATIO='1.7'), 15 * GB)

    def test_never_negative(self):
        self.assertEqual(self.free(10 * GB, 12 * GB), 0)

    def test_psutil_opt_out(self):
        mps = mm.torch.device('mps')
        with mock.patch.dict(os.environ, {'FOOOCUS_MPS_PSUTIL_FREE_MEMORY': '1'}), \
                mock.patch.object(mm.psutil, 'virtual_memory', return_value=mock.Mock(available=3 * GB)):
            self.assertEqual(mm.get_free_memory(mps), 3 * GB)


if __name__ == '__main__':
    unittest.main()
