import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import os
import shutil
import tempfile
import unittest

from modules import upscale_pipeline as up


class TestSettings(unittest.TestCase):
    def test_defaults_are_4x_detailed_keep_original(self):
        s = up.Settings()
        self.assertEqual((s.scale, s.method, s.upscaler, s.keep_original),
                         (4.0, up.METHOD_DETAILED, up.BUILT_IN_UPSCALER, True))

    def test_parse_scale(self):
        self.assertEqual(up.parse_scale('4x'), 4.0)
        self.assertEqual(up.parse_scale('1.5x'), 1.5)
        self.assertEqual(up.parse_scale(2), 2.0)
        self.assertEqual(up.parse_scale('junk'), 4.0)
        self.assertEqual(up.parse_scale('1x'), 4.0)   # not an upscale
        self.assertEqual(up.parse_scale(None, 2.0), 2.0)

    def test_make_settings_cleans_ui_values(self):
        s = up.make_settings(enabled=1, scale='2x', method='nonsense', upscaler='', keep_original=0)
        self.assertEqual(s, up.Settings(True, 2.0, up.METHOD_DETAILED, up.BUILT_IN_UPSCALER, False))

    def test_scale_labels_round_trip(self):
        for label in up.SCALE_LABELS:
            self.assertEqual(up.scale_label(up.parse_scale(label)), label)


class TestUpscalerList(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def touch(self, rel):
        path = os.path.join(self.dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w').close()
        return path

    def test_built_in_first_then_model_files(self):
        self.touch(up.BUILT_IN_FILE)
        self.touch('4x-UltraSharp.pth')
        self.touch('RealESRGAN_x4plus.safetensors')
        self.touch('extra/2x_Sharp.pt')
        self.touch('readme.txt')
        self.touch('.hidden.pth')
        self.assertEqual(up.list_upscalers(self.dir),
                         [up.BUILT_IN_UPSCALER, '4x-UltraSharp.pth', 'extra/2x_Sharp.pt',
                          'RealESRGAN_x4plus.safetensors'])

    def test_missing_folder_lists_only_built_in(self):
        self.assertEqual(up.list_upscalers(os.path.join(self.dir, 'nope')), [up.BUILT_IN_UPSCALER])

    def test_resolve(self):
        path = self.touch('4x-UltraSharp.pth')
        self.assertEqual(up.resolve_upscaler('4x-UltraSharp.pth', self.dir), os.path.realpath(path))
        self.assertIsNone(up.resolve_upscaler(up.BUILT_IN_UPSCALER, self.dir))
        self.assertIsNone(up.resolve_upscaler('gone.pth', self.dir))           # falls back to built-in
        outside = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        self.addCleanup(os.remove, outside.name)
        self.assertIsNone(up.resolve_upscaler(os.path.join('..', os.path.basename(outside.name)), self.dir))


class TestPlan(unittest.TestCase):
    def test_4x_detailed_sdxl_details_at_2x_then_upscales(self):
        p = up.plan(1024, 1024, up.Settings(scale=4.0), 'sdxl')
        self.assertEqual(p.final, (4096, 4096))
        self.assertEqual(p.detail, (2048, 2048))
        self.assertLessEqual(up.shape_ceil(*p.detail), up.DETAIL_MAX_SHAPE)

    def test_landscape_keeps_aspect(self):
        p = up.plan(1152, 896, up.Settings(scale=4.0), 'sdxl')
        self.assertEqual(p.final, (4608, 3584))
        self.assertEqual(p.detail, (2304, 1792))

    def test_2x_details_at_the_final_size(self):
        p = up.plan(1024, 1024, up.Settings(scale=2.0), 'flux')
        self.assertEqual(p.detail, p.final)

    def test_1_5x(self):
        p = up.plan(1024, 1024, up.Settings(scale=1.5), 'sdxl')
        self.assertEqual(p.final, (1536, 1536))
        self.assertEqual(p.detail, (1536, 1536))

    def test_fast_has_no_detail_pass(self):
        p = up.plan(1024, 1024, up.Settings(method=up.METHOD_FAST), 'sdxl')
        self.assertIsNone(p.detail)
        self.assertEqual(p.note, '')

    def test_sdcpp_falls_back_to_fast_with_a_note(self):
        p = up.plan(1024, 1024, up.Settings(), 'sdcpp')
        self.assertIsNone(p.detail)
        self.assertIn('family', p.note)

    def test_big_image_skips_detail(self):
        p = up.plan(2600, 2600, up.Settings(scale=2.0), 'sdxl')
        self.assertIsNone(p.detail)
        self.assertIn('limit', p.note)

    def test_output_side_is_capped(self):
        p = up.plan(3000, 2000, up.Settings(scale=4.0, method=up.METHOD_FAST), 'sdxl')
        self.assertEqual(max(p.final), up.MAX_OUTPUT_SIDE)
        self.assertAlmostEqual(p.final[0] / p.final[1], 1.5, places=2)

    def test_detail_sizes_fit_the_latent_grid(self):
        for w, h in ((832, 1216), (896, 1152), (1344, 768), (1000, 700)):
            p = up.plan(w, h, up.Settings(), 'sdxl')
            self.assertEqual(p.detail[0] % 16, 0)
            self.assertEqual(p.detail[1] % 16, 0)
            self.assertLessEqual(up.shape_ceil(*p.detail), up.DETAIL_MAX_SHAPE)


class TestModelPasses(unittest.TestCase):
    def test_passes(self):
        self.assertEqual(up.model_passes(1024, 4096, 4), 1)
        self.assertEqual(up.model_passes(1024, 4096, 2), 2)
        self.assertEqual(up.model_passes(2048, 4096, 2), 1)
        self.assertEqual(up.model_passes(4096, 4096, 2), 0)
        self.assertEqual(up.model_passes(100, 10 ** 6, 2), 3)   # capped
        self.assertEqual(up.model_passes(1024, 2048, 1), 1)     # 1x "restoration" models still run once


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def test_upscaled_path_and_detection(self):
        original = os.path.join(self.dir, 'a.png')
        target = up.upscaled_path(original, 4.0)
        self.assertEqual(os.path.basename(target), 'a_up4x.png')
        self.assertEqual(os.path.basename(up.upscaled_path(os.path.join(self.dir, 'b.webp'), 1.5)), 'b_up1.5x.webp')
        open(target, 'w').close()
        self.assertEqual(os.path.basename(up.upscaled_path(original, 4.0)), 'a_up4x (2).png')
        self.assertTrue(up.is_upscale(target))
        self.assertTrue(up.is_upscale('a_up4x (2).png'))
        self.assertFalse(up.is_upscale(original))
        self.assertFalse(up.is_upscale('setup4x.png'))

    def test_should_upscale(self):
        temp = os.path.join(self.dir, 'temp')
        os.makedirs(temp)
        keep = os.path.join(self.dir, 'a.png')
        throwaway = os.path.join(temp, 'b.png')
        for path in (keep, throwaway, os.path.join(self.dir, 'a_up4x.png'), os.path.join(self.dir, 'n.txt')):
            open(path, 'w').close()
        self.assertTrue(up.should_upscale(keep, temp))
        self.assertFalse(up.should_upscale(throwaway, temp))
        self.assertTrue(up.should_upscale(throwaway, None))
        self.assertFalse(up.should_upscale(os.path.join(self.dir, 'a_up4x.png'), temp))
        self.assertFalse(up.should_upscale(os.path.join(self.dir, 'n.txt'), temp))
        self.assertFalse(up.should_upscale(os.path.join(self.dir, 'missing.png'), temp))
        self.assertFalse(up.should_upscale(None, temp))   # an image wall is a numpy array, not a path

    def test_metadata_keeps_generation_resolution(self):
        rows = up.upscale_metadata([('Resolution', 'resolution', '(1024, 1024)'), ('Seed', 'seed', '1')],
                                   up.Settings(), (4096, 4096), up.BUILT_IN_UPSCALER, 'detail pass', '', 'a.png')
        data = {k: v for _, k, v in rows}
        self.assertEqual(data['resolution'], '(1024, 1024)')
        self.assertEqual(data['upscaled_size'], '(4096, 4096)')
        self.assertEqual(data['upscaled_from'], 'a.png')
        self.assertIn('4x', data['auto_upscale'])
        self.assertNotIn('auto_upscale_note', data)


if __name__ == '__main__':
    unittest.main()
