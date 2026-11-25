import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import os
import random
import shutil
import tempfile
import unittest
from unittest import mock

from modules.flux import app as flux_app
from modules.flux import pipeline

STYLES = {
    'Wrap': ('cinematic photo of {prompt}, film grain', 'cartoon'),
    'Extra': ('high detail\nsharp focus', 'blurry'),
}


class TestBuildPrompt(unittest.TestCase):
    def build(self, prompt, styles):
        return flux_app.build_prompt(prompt, styles, STYLES, random.Random(0))

    def test_placeholder_style_wraps_prompt(self):
        self.assertEqual(self.build('a fox', ['Fooocus V2', 'Wrap', 'Extra']),
                         'cinematic photo of a fox, film grain, high detail, sharp focus')

    def test_prompt_first_without_placeholder(self):
        self.assertEqual(self.build('a fox', ['Extra', 'Unknown']), 'a fox, high detail, sharp focus')
        self.assertEqual(self.build('a fox', []), 'a fox')


class TestAvailability(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def test_needs_all_four_files(self):
        with mock.patch.object(flux_app, 'dependencies_installed', return_value=True):
            self.assertEqual(flux_app.model_choices(self.dir), [])
            for name in pipeline.FILES.values():
                open(os.path.join(self.dir, name), 'w').close()
            self.assertEqual(flux_app.model_choices(self.dir), [flux_app.FLUX_MODEL_NAME])
            os.remove(os.path.join(self.dir, pipeline.FILES['t5']))
            self.assertEqual(flux_app.model_choices(self.dir), [])
            self.assertEqual(pipeline.missing_files(self.dir), [pipeline.FILES['t5']])

    def test_needs_packages(self):
        for name in pipeline.FILES.values():
            open(os.path.join(self.dir, name), 'w').close()
        with mock.patch.object(flux_app, 'dependencies_installed', return_value=False):
            self.assertEqual(flux_app.model_choices(self.dir), [])

    def test_is_flux(self):
        self.assertTrue(flux_app.is_flux(flux_app.FLUX_MODEL_NAME))
        self.assertFalse(flux_app.is_flux('juggernautXL_v8Rundiffusion.safetensors'))


class TestSamplingHelpers(unittest.TestCase):
    """The reference schnell settings: linear 1 -> 0 schedule, no shift."""

    def test_schnell_schedule(self):
        self.assertEqual(pipeline.get_schedule(4, 4096, shift=False), [1.0, 0.75, 0.5, 0.25, 0.0])

    def test_shifted_schedule_favours_high_noise(self):
        shifted = pipeline.get_schedule(4, 4096, shift=True)
        self.assertEqual((shifted[0], shifted[-1]), (1.0, 0.0))
        self.assertGreater(shifted[2], 0.5)

    def test_noise_is_seeded_and_packed_shape(self):
        import torch
        a = pipeline.get_noise(1, 896, 1152, 'cpu', torch.float32, 7)
        b = pipeline.get_noise(1, 896, 1152, 'cpu', torch.float32, 7)
        self.assertEqual(tuple(a.shape), (1, 16, 112, 144))
        self.assertTrue(torch.equal(a, b))
        packed = pipeline.prepare(a, torch.zeros(1, 256, 4096), torch.zeros(1, 768))
        self.assertEqual(tuple(packed['img'].shape), (1, 56 * 72, 64))
        self.assertEqual(tuple(packed['img_ids'].shape), (1, 56 * 72, 3))
        self.assertEqual(tuple(pipeline.unpack(packed['img'], 896, 1152).shape), (1, 16, 112, 144))
        torch.testing.assert_close(pipeline.unpack(packed['img'], 896, 1152), a)


if __name__ == '__main__':
    unittest.main()
