import sys

# args_manager parses sys.argv at import time; drop the test runner's own arguments.
sys.argv = sys.argv[:1]

import json
import os
import tempfile
import unittest
from unittest import mock

from modules import apple_silicon
from modules import memory_manager as memmgr

try:
    import torch
except Exception:  # pragma: no cover - CI installs CPU torch
    torch = None

GB = 1024 ** 3


class OutOfMemoryError(RuntimeError):
    """Same name as torch's (torch.OutOfMemoryError / torch.cuda.OutOfMemoryError)."""


class InterruptProcessingException(Exception):
    pass


class TestIsOomError(unittest.TestCase):
    def test_mps_out_of_memory_is_a_plain_runtime_error(self):
        exc = RuntimeError('MPS backend out of memory (MPS allocated: 17.10 GB, other allocations: 1.20 GB, '
                           'max allowed: 18.13 GB). Tried to allocate 256.00 MB on private pool.')
        self.assertTrue(memmgr.is_oom_error(exc))

    def test_other_out_of_memory_errors(self):
        self.assertTrue(memmgr.is_oom_error(OutOfMemoryError('CUDA out of memory')))
        self.assertTrue(memmgr.is_oom_error(MemoryError()))
        self.assertTrue(memmgr.is_oom_error(RuntimeError('[enforce fail] DefaultCPUAllocator: not enough memory')))

    def test_other_errors_are_not_out_of_memory(self):
        self.assertFalse(memmgr.is_oom_error(RuntimeError('mat1 and mat2 shapes cannot be multiplied')))
        self.assertFalse(memmgr.is_oom_error(ValueError('out of memory')))  # not a RuntimeError
        self.assertFalse(memmgr.is_oom_error(InterruptProcessingException()))
        self.assertFalse(memmgr.is_oom_error(None))

    @unittest.skipIf(torch is None, 'torch not installed')
    def test_torch_out_of_memory_class(self):
        self.assertTrue(memmgr.is_oom_error(torch.cuda.OutOfMemoryError('CUDA out of memory')))


class TestRunWithOomRetry(unittest.TestCase):
    def test_frees_memory_and_retries_once(self):
        calls, freed = [], []

        def step(x):
            calls.append(x)
            if len(calls) == 1:
                raise RuntimeError('MPS backend out of memory')
            return x * 2

        def free(reason):
            # The failed attempt's traceback must be gone by now, so its tensors can be freed.
            self.assertEqual(sys.exc_info(), (None, None, None))
            freed.append(reason)

        self.assertEqual(memmgr.run_with_oom_retry(step, 21, label='sampling', free=free), 42)
        self.assertEqual(calls, [21, 21])
        self.assertEqual(len(freed), 1)
        self.assertIn('sampling', freed[0])

    def test_other_errors_are_raised_without_freeing(self):
        free = mock.Mock()
        with self.assertRaises(ValueError):
            memmgr.run_with_oom_retry(mock.Mock(side_effect=ValueError('bad')), free=free)
        free.assert_not_called()

    def test_gives_up_after_the_retry(self):
        free = mock.Mock()
        step = mock.Mock(side_effect=RuntimeError('MPS backend out of memory'))
        with self.assertRaises(RuntimeError):
            memmgr.run_with_oom_retry(step, free=free, retries=1)
        self.assertEqual(step.call_count, 2)
        free.assert_called_once()

    def test_stop_and_skip_are_not_retried(self):
        free = mock.Mock()
        with self.assertRaises(InterruptProcessingException):
            memmgr.run_with_oom_retry(mock.Mock(side_effect=InterruptProcessingException()), free=free)
        free.assert_not_called()


class TestFreeing(unittest.TestCase):
    def test_free_everything_keeps_going_when_one_part_fails(self):
        with mock.patch.object(memmgr, '_release_ollama', side_effect=OSError('no ollama')), \
                mock.patch.object(memmgr, 'release_cached_memory') as release, \
                mock.patch.dict(sys.modules, {
                    'modules.flux.app': mock.Mock(),
                    'modules.upscaler': mock.Mock(),
                    'modules.prompt_rewriter': mock.Mock(),
                }):
            memmgr.free_everything('test')
            flux = sys.modules['modules.flux.app']
            flux.unload.assert_called_once()
            flux.free_sdxl.assert_called_once()
            sys.modules['modules.upscaler'].release.assert_called_once()
            sys.modules['modules.prompt_rewriter'].release.assert_called_once()
        release.assert_called_once()

    def test_free_gpu_memory_keeps_the_pipeline_models(self):
        mm = mock.Mock()
        flux = mock.Mock()
        with mock.patch.object(memmgr, '_release_ollama'), \
                mock.patch.object(memmgr, 'release_cached_memory') as release, \
                mock.patch.dict(sys.modules, {
                    'ldm_patched.modules.model_management': mm,
                    'modules.flux.app': flux,
                    'modules.upscaler': mock.Mock(),
                    'modules.prompt_rewriter': mock.Mock(),
                }):
            memmgr.free_gpu_memory('retry')
        mm.unload_all_models.assert_called_once()
        flux.free_sdxl.assert_not_called()  # the retry needs SDXL's models back
        release.assert_called_once()

    def test_release_cached_memory_without_torch(self):
        with mock.patch.object(memmgr, '_torch', return_value=None):
            memmgr.release_cached_memory()  # must not raise


class TestReadout(unittest.TestCase):
    def test_format(self):
        line = memmgr.format_memory_line('after generation', {
            'gpu_name': 'MPS', 'gpu_allocated': 6 * GB, 'gpu_reserved': 8 * GB,
            'ram_used': 18 * GB, 'ram_total': 24 * GB, 'swap_used': GB // 2})
        self.assertEqual(line, '[memory] after generation: MPS 6.0 GB in use (8.0 GB held) · '
                               'RAM 18.0 GB of 24.0 GB · swap 0.5 GB')

    def test_format_with_nothing_known(self):
        self.assertEqual(memmgr.format_memory_line('x', {}), '[memory] x: unavailable')

    def test_log_can_be_turned_off(self):
        with mock.patch.dict(os.environ, {'FOOOCUS_MEMORY_LOG': '0'}):
            self.assertIsNone(memmgr.log_memory('x'))
        with mock.patch.dict(os.environ, {'FOOOCUS_MEMORY_LOG': '1'}), \
                mock.patch.object(memmgr, 'memory_snapshot', return_value={}), \
                mock.patch('builtins.print'):
            self.assertEqual(memmgr.log_memory('x'), '[memory] x: unavailable')

    def test_snapshot_has_ram_and_swap(self):
        snap = memmgr.memory_snapshot()
        try:
            import psutil  # noqa: F401
        except ImportError:
            self.skipTest('psutil not installed')
        self.assertGreater(snap['ram_total'], 0)
        self.assertIn('swap_used', snap)


class TestIdleUnload(unittest.TestCase):
    def test_minutes_setting(self):
        for raw, seconds in (('10', 600.0), ('0', 0.0), ('-3', 0.0), ('junk', 600.0), ('0.5', 30.0)):
            with mock.patch.dict(os.environ, {'FOOOCUS_IDLE_UNLOAD_MINUTES': raw}):
                self.assertEqual(memmgr.idle_unload_seconds(), seconds, raw)

    def test_activity_resets_the_clock(self):
        memmgr.mark_activity()
        self.assertLess(memmgr.idle_for(), 5)

    def test_unload_idle_models(self):
        mm = mock.Mock(current_loaded_models=[object(), object()])
        flux = mock.Mock(_pipeline=object())
        with mock.patch.dict(sys.modules, {'ldm_patched.modules.model_management': mm,
                                           'modules.flux.app': flux,
                                           'modules.prompt_studio': mock.Mock()}), \
                mock.patch.object(memmgr, 'release_cached_memory'), \
                mock.patch.object(memmgr, 'log_memory'):
            self.assertTrue(memmgr.unload_idle_models())
        mm.unload_all_models.assert_called_once()
        flux.unload.assert_called_once()


class TestSwapSetting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config = os.path.join(self.tmp, 'config.txt')

    def env(self, **values):
        clean = {k: v for k, v in os.environ.items() if k not in ('FOOOCUS_ALLOW_SWAP', 'config_path')}
        clean['config_path'] = self.config
        clean.update(values)
        return mock.patch.dict(os.environ, clean, clear=True)

    def write_config(self, data):
        with open(self.config, 'w', encoding='utf-8') as fh:
            json.dump(data, fh)

    def test_on_by_default(self):
        with self.env():
            self.assertTrue(apple_silicon.allow_swap_enabled())

    def test_config_txt_can_turn_it_off(self):
        self.write_config({'allow_swap': False})
        with self.env():
            self.assertFalse(apple_silicon.allow_swap_enabled())

    def test_environment_wins(self):
        self.write_config({'allow_swap': False})
        with self.env(FOOOCUS_ALLOW_SWAP='1'):
            self.assertTrue(apple_silicon.allow_swap_enabled())
        self.write_config({'allow_swap': True})
        with self.env(FOOOCUS_ALLOW_SWAP='off'):
            self.assertFalse(apple_silicon.allow_swap_enabled())

    def test_unreadable_config_means_default(self):
        with open(self.config, 'w') as fh:
            fh.write('{not json')
        with self.env():
            self.assertTrue(apple_silicon.allow_swap_enabled())


class TestWatermarks(unittest.TestCase):
    def test_allow_swap(self):
        self.assertEqual(apple_silicon.watermark_ratios(16, True), ('1.7', '1.4'))
        self.assertEqual(apple_silicon.watermark_ratios(24, True), ('1.7', '1.4'))
        self.assertEqual(apple_silicon.watermark_ratios(8, True), ('1.7', '1.4'))
        self.assertEqual(apple_silicon.watermark_ratios(64, True), ('0.0', '0.0'))

    def test_no_swap_stays_inside_physical_memory(self):
        self.assertEqual(apple_silicon.watermark_ratios(64, False)[0], '1.0')
        self.assertEqual(apple_silicon.watermark_ratios(24, False)[0], '0.85')
        self.assertEqual(apple_silicon.watermark_ratios(8, False)[0], '0.7')

    def test_low_never_above_high_and_high_within_torch_limit(self):
        for mem in (0, 8, 16, 24, 32, 48, 128):
            for allow in (True, False):
                high, low = (float(v) for v in apple_silicon.watermark_ratios(mem, allow))
                self.assertLessEqual(high, 2.0)  # PyTorch rejects a high ratio above 2.0
                if high > 0:
                    self.assertLessEqual(low, high)
                else:
                    self.assertEqual(low, 0.0)

    def configure(self, mem_gb, **env):
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith('PYTORCH_MPS_') and k not in ('FOOOCUS_ALLOW_SWAP', 'config_path')}
        clean['config_path'] = os.path.join(tempfile.mkdtemp(), 'missing.txt')
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True), \
                mock.patch.object(apple_silicon.platform, 'system', return_value='Darwin'), \
                mock.patch.object(apple_silicon.platform, 'machine', return_value='arm64'), \
                mock.patch.object(apple_silicon, 'detect_chip_brand', return_value='Apple M5'), \
                mock.patch.object(apple_silicon, 'detect_memory_gb', return_value=mem_gb), \
                mock.patch.object(apple_silicon, 'detect_performance_cores', return_value=0), \
                mock.patch.object(apple_silicon, '_try_activate_mtlflashattn', return_value='not-installed'):
            info = apple_silicon.configure(verbose=False)
            return info, os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO'), \
                os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO')

    def test_configure_allows_swap_by_default(self):
        info, high, low = self.configure(24)
        self.assertTrue(info['allow_swap'])
        self.assertEqual((high, low), ('1.7', '1.4'))

    def test_configure_with_swap_off(self):
        info, high, low = self.configure(24, FOOOCUS_ALLOW_SWAP='0')
        self.assertFalse(info['allow_swap'])
        self.assertEqual(high, '0.85')
        self.assertLessEqual(float(low), 0.85)

    def test_explicit_ratio_is_kept_and_low_follows_it(self):
        _, high, low = self.configure(24, PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.9')
        self.assertEqual(high, '0.9')
        self.assertLessEqual(float(low), 0.9)


@unittest.skipIf(torch is None, 'torch not installed')
class TestTorchLoadMmap(unittest.TestCase):
    def setUp(self):
        import ldm_patched.modules.utils as utils
        self.utils = utils
        self.tmp = tempfile.mkdtemp()

    def test_zip_format_loads(self):
        path = os.path.join(self.tmp, 'w.pth')
        torch.save({'w': torch.arange(6.0)}, path)
        sd = self.utils.torch_load_mmap(path, weights_only=True)
        self.assertTrue(torch.equal(sd['w'], torch.arange(6.0)))

    def test_legacy_format_falls_back(self):
        path = os.path.join(self.tmp, 'legacy.pth')
        torch.save({'w': torch.ones(3)}, path, _use_new_zipfile_serialization=False)
        sd = self.utils.torch_load_mmap(path, weights_only=True)
        self.assertTrue(torch.equal(sd['w'], torch.ones(3)))

    def test_load_torch_file_uses_it(self):
        path = os.path.join(self.tmp, 'ckpt.pt')
        torch.save({'state_dict': {'a': torch.zeros(2)}}, path)
        self.assertTrue(torch.equal(self.utils.load_torch_file(path, safe_load=True)['a'], torch.zeros(2)))


if __name__ == '__main__':
    unittest.main()
