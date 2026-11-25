import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import os
import unittest

from modules import model_profiles
from modules import quality_prompts as qp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESETS = model_profiles.load_presets(os.path.join(REPO, 'presets'))


class TestFamily(unittest.TestCase):
    def family(self, name):
        return qp.family_for_model(name, None, PRESETS)

    def test_families(self):
        self.assertEqual(self.family('flux1-schnell-Q4_K_S.gguf'), 'flux')
        self.assertEqual(self.family('ponyDiffusionV6XL.safetensors'), 'pony')
        self.assertEqual(self.family('illustriousXL_v01.safetensors'), 'anime')
        self.assertEqual(self.family('realisticStockPhoto_v20.safetensors'), 'photo')
        self.assertEqual(self.family('dreamshaperXL_lightningDPMSDE.safetensors'), 'distilled')
        self.assertEqual(self.family('juggernautXL_v8Rundiffusion.safetensors'), 'general')
        self.assertEqual(self.family('mystery.safetensors'), 'general')


class TestAdditions(unittest.TestCase):
    def test_general_adds_positive_and_negative(self):
        pos, neg = qp.additions('general', 'a fox in snow', '')
        self.assertIn('highly detailed', pos)
        self.assertIn('watermark', neg)

    def test_skips_terms_already_written(self):
        pos, neg = qp.additions('general', 'a fox, Highly Detailed, sharp focus', 'watermark, blurry')
        self.assertNotIn('highly detailed', pos.lower())
        self.assertNotIn('sharp focus', pos)
        self.assertNotIn('watermark', neg)
        self.assertNotIn('blurry', neg.split(', '))
        self.assertIn('intricate details', pos)

    def test_pony_score_tags(self):
        pos, neg = qp.additions('pony', 'a fox', '')
        self.assertTrue(pos.startswith('score_9'))
        self.assertIn('score_4', neg)

    def test_flux_sentence_and_no_negative(self):
        pos, neg = qp.additions('flux', 'a fox in snow', 'ignored')
        self.assertTrue(pos.endswith('.'))
        self.assertEqual(neg, '')
        again, _ = qp.additions('flux', f'a fox in snow. {pos}', '')
        self.assertEqual(again, '')

    def test_distilled_has_no_negative(self):
        self.assertEqual(qp.additions('distilled', 'a fox')[1], '')

    def test_unknown_family_falls_back(self):
        self.assertEqual(qp.additions('???', 'a fox'), qp.additions('general', 'a fox'))


if __name__ == '__main__':
    unittest.main()
