"""VAE encode/decode fall back to tiles when the GPU runs out of memory,
including on MPS, where out-of-memory is a plain RuntimeError.

Runs in a subprocess with --always-cpu: model_management picks its device
when it is imported, and CI has no GPU."""
import os
import subprocess
import sys
import textwrap
import unittest

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:  # pragma: no cover - CI installs CPU torch
    _HAS_TORCH = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = textwrap.dedent('''
    import sys
    sys.argv = ['test', '--always-cpu']
    import args_manager  # parses --always-cpu before model_management picks a device
    import torch
    import ldm_patched.modules.model_management as mm
    import ldm_patched.modules.sd as sd

    assert mm.is_oom_error(RuntimeError('MPS backend out of memory (MPS allocated: 1 GB)'))
    assert not mm.is_oom_error(RuntimeError('mat1 and mat2 shapes cannot be multiplied'))
    assert not mm.is_oom_error(mm.InterruptProcessingException())
    mm.unload_all_models()  # nothing loaded: must still run cleanly

    class Stage(torch.nn.Module):
        def decode(self, z):
            if z.shape[-1] > 8:
                raise RuntimeError('MPS backend out of memory (MPS allocated: 17 GB)')
            return torch.zeros(z.shape[0], 3, z.shape[2] * 8, z.shape[3] * 8)

        def encode(self, x):
            if x.shape[-1] > 64:
                raise RuntimeError('MPS backend out of memory (MPS allocated: 17 GB)')
            return torch.zeros(x.shape[0], 4, x.shape[2] // 8, x.shape[3] // 8)

    vae = sd.VAE.__new__(sd.VAE)
    vae.first_stage_model = Stage()
    vae.vae_dtype = torch.float32
    vae.device = vae.output_device = torch.device('cpu')
    vae.downscale_ratio = 8
    vae.latent_channels = 4
    vae.patcher = None
    vae.memory_used_decode = vae.memory_used_encode = lambda shape, dtype: 1
    mm.load_models_gpu = lambda *a, **k: None
    tiled = []
    vae.decode_tiled_ = lambda s, *a, **k: tiled.append('decode') or torch.zeros(s.shape[0], 3, s.shape[2] * 8, s.shape[3] * 8)
    vae.encode_tiled_ = lambda p, *a, **k: tiled.append('encode') or torch.zeros(p.shape[0], 4, p.shape[2] // 8, p.shape[3] // 8)

    assert tuple(vae.decode(torch.zeros(1, 4, 16, 16)).shape) == (1, 128, 128, 3)
    assert tuple(vae.encode(torch.zeros(1, 128, 128, 3)).shape) == (1, 4, 16, 16)
    assert tiled == ['decode', 'encode'], tiled
    assert tuple(vae.decode(torch.zeros(1, 4, 8, 8)).shape) == (1, 64, 64, 3)  # fits: no tiling
    assert tiled == ['decode', 'encode'], tiled

    class Broken(torch.nn.Module):
        def decode(self, z):
            raise ValueError('a real bug')
    vae.first_stage_model = Broken()
    try:
        vae.decode(torch.zeros(1, 4, 4, 4))
    except ValueError:
        pass
    else:
        raise AssertionError('a non-memory error was swallowed')
    print('VAE-OOM-OK')
''')


@unittest.skipUnless(_HAS_TORCH, 'torch not installed')
class TestVaeOutOfMemory(unittest.TestCase):
    def test_tiled_fallback(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', FOOOCUS_MEMORY_LOG='0')
        result = subprocess.run([sys.executable, '-c', SCRIPT], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=300)
        self.assertIn('VAE-OOM-OK', result.stdout, result.stdout[-2000:] + result.stderr[-4000:])


if __name__ == '__main__':
    unittest.main()
