import unittest

import numpy as np

from modules.flux import app as flux_app
from modules.flux import pipeline as fp


class TestSchedule(unittest.TestCase):
    def test_img2img_schedule_covers_strength(self):
        self.assertEqual(fp.img2img_schedule(4, 0.5), [0.5, 0.25, 0.0])
        self.assertEqual(len(fp.img2img_schedule(4, 0.85)), 5)       # ceil(3.4) = 4 steps
        self.assertAlmostEqual(fp.img2img_schedule(4, 0.35)[0], 0.35, places=5)
        self.assertEqual(len(fp.img2img_schedule(4, 0.01)), 2)       # clamped, at least one step
        self.assertEqual(fp.img2img_schedule(4, 2.0)[0], 1.0)


class TestTiles(unittest.TestCase):
    def test_tile_starts_cover_with_overlap(self):
        for size, tile, overlap in ((2304, 1024, 128), (1792, 1024, 128), (1024, 1024, 128), (600, 1024, 128)):
            starts = fp.tile_starts(size, min(tile, size), overlap)
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1] + min(tile, size), size)
            for a, b in zip(starts, starts[1:]):
                self.assertGreaterEqual(a + min(tile, size) - b, overlap)

    def test_feather_weights(self):
        w = fp.feather_weights(64, 64, 16, top=True, bottom=False, left=False, right=True)
        self.assertTrue((w > 0).all())
        self.assertLess(w[0, 0], 0.1)       # ramps up from the top edge
        self.assertEqual(w[-1, 0], 1.0)     # no ramp at bottom/left
        self.assertLess(w[32, -1], 0.1)     # ramps down to the right edge

    def test_blend_of_identical_tiles_is_identity(self):
        # Weighted average of tiles that all return the input must return the input.
        image = np.random.default_rng(0).integers(0, 255, (96, 160, 3)).astype(np.uint8)
        acc = np.zeros(image.shape, np.float32)
        weight = np.zeros(image.shape[:2] + (1,), np.float32)
        for y in fp.tile_starts(96, 64, 16):
            for x in fp.tile_starts(160, 64, 16):
                w = fp.feather_weights(64, 64, 16, y > 0, y + 64 < 96, x > 0, x + 64 < 160)[..., None]
                acc[y:y + 64, x:x + 64] += image[y:y + 64, x:x + 64] * w
                weight[y:y + 64, x:x + 64] += w
        out = np.clip(acc / weight, 0, 255).round().astype(np.uint8)
        self.assertTrue(np.array_equal(out, image))


class TestImageMode(unittest.TestCase):
    def test_modes(self):
        img = np.zeros((8, 8, 3), np.uint8)
        self.assertEqual(flux_app.image_mode(True, 'uov', 'Vary (Subtle)', img), ('vary', 0.5))
        self.assertEqual(flux_app.image_mode(True, 'uov', 'Vary (Strong)', img), ('vary', 0.85))
        self.assertEqual(flux_app.image_mode(True, 'uov', 'Upscale (2x)', img), ('upscale', 2.0))
        self.assertEqual(flux_app.image_mode(True, 'uov', 'Upscale (1.5x)', img), ('upscale', 1.5))
        self.assertEqual(flux_app.image_mode(True, 'uov', 'Upscale (Fast 2x)', img), ('upscale_fast', 2.0))
        self.assertIsNone(flux_app.image_mode(True, 'uov', 'Disabled', img))
        self.assertIsNone(flux_app.image_mode(False, 'uov', 'Vary (Subtle)', img))
        self.assertIsNone(flux_app.image_mode(True, 'inpaint', 'Vary (Subtle)', img))
        self.assertIsNone(flux_app.image_mode(True, 'uov', 'Vary (Subtle)', None))

    def test_vary_size(self):
        self.assertEqual(flux_app.vary_size(896, 1152), (1152, 896))           # already in budget
        w, h = flux_app.vary_size(3000, 4000)
        self.assertLessEqual(w * h, flux_app.MAX_VARY_PIXELS)
        self.assertEqual((w % 16, h % 16), (0, 0))
        self.assertAlmostEqual(w / h, 4000 / 3000, delta=0.02)


if __name__ == '__main__':
    unittest.main()
