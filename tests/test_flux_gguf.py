import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import os
import unittest

import torch

try:
    import gguf
    from modules.flux import gguf_ops
except ImportError as exc:  # optional dependency
    gguf = None
    _IMPORT_ERROR = exc

# Set FOOOCUS_FLUX_GGUF to test a file outside the repo's models/flux folder.
PATH = os.environ.get('FOOOCUS_FLUX_GGUF') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'flux', 'flux1-schnell-Q4_K_S.gguf')


@unittest.skipIf(gguf is None, 'gguf package not installed')
@unittest.skipUnless(os.path.exists(PATH), 'FLUX.1 schnell Q4_K_S file not present')
class TestDequantMatchesReference(unittest.TestCase):
    """Our torch Q4_K dequant must equal gguf's own NumPy implementation."""

    @classmethod
    def setUpClass(cls):
        cls.reader = gguf.GGUFReader(PATH)
        cls.by_type = {}
        for t in cls.reader.tensors:
            cls.by_type.setdefault(t.tensor_type, []).append(t)

    def check(self, tensor):
        expected = torch.from_numpy(gguf.quants.dequantize(tensor.data, tensor.tensor_type)).float()
        shape = tuple(int(v) for v in reversed(tensor.shape))
        got = gguf_ops.dequantize(torch.from_numpy(tensor.data.copy()), tensor.tensor_type, shape, torch.float32)
        self.assertEqual(tuple(got.shape), shape)
        self.assertEqual(tuple(expected.reshape(shape).shape), shape)
        torch.testing.assert_close(got, expected.reshape(shape), rtol=0, atol=1e-6)

    def test_q4_k(self):
        tensors = self.by_type[gguf.GGMLQuantizationType.Q4_K]
        for t in (tensors[0], tensors[len(tensors) // 2], tensors[-1]):
            with self.subTest(t.name):
                self.check(t)

    def test_f16_and_f32(self):
        for qtype in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
            with self.subTest(qtype.name):
                self.check(self.by_type[qtype][0])

    def test_only_supported_types_in_file(self):
        supported = {gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32}
        self.assertLessEqual(set(self.by_type), supported)


if __name__ == '__main__':
    unittest.main()
