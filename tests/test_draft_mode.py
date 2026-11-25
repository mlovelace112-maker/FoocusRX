import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import types
import unittest

from modules import draft_mode
from modules.constants import MAX_SEED


def values(**overrides):
    v = {'performance': 'Speed', 'overwrite_step': -1, 'sampler_name': 'dpmpp_2m_sde_gpu',
         'scheduler_name': 'karras', 'image_seed': 111, 'image_number': 4}
    v.update(overrides)
    return v


class TestApplyDraft(unittest.TestCase):
    def test_draft(self):
        v = values()
        self.assertTrue(draft_mode.apply_draft(v, 'draft'))
        self.assertEqual((v['overwrite_step'], v['sampler_name'], v['scheduler_name']), (12, 'dpmpp_2m', 'karras'))
        self.assertEqual((v['image_seed'], v['image_number']), (111, 4))  # untouched

    def test_full_render(self):
        v = values(overwrite_step=12)
        self.assertFalse(draft_mode.apply_draft(v, 'full', seed=999))
        self.assertEqual((v['overwrite_step'], v['sampler_name'], v['image_seed'], v['image_number']),
                         (-1, 'dpmpp_2m', 999, 1))

    def test_off_and_fast_presets_are_untouched(self):
        for mode, perf in ((None, 'Speed'), ('draft', 'Lightning'), ('full', 'Hyper-SD'), ('draft', 'Extreme Speed')):
            v = values(performance=perf)
            self.assertFalse(draft_mode.apply_draft(v, mode))
            self.assertEqual(v, values(performance=perf))

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            draft_mode.apply_draft(values(), 'bogus')


def task(**overrides):
    t = types.SimpleNamespace(draft_mode=True, prompt='a fox', prompt_queue=False, should_enhance=False,
                              image_number=4, seed=100, disable_seed_increment=False)
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


class TestSeedForResult(unittest.TestCase):
    def test_seed_increments_per_image(self):
        self.assertEqual([draft_mode.seed_for_result(task(), i) for i in range(4)], [100, 101, 102, 103])

    def test_wraps_and_disable_increment(self):
        self.assertEqual(draft_mode.seed_for_result(task(seed=MAX_SEED), 1), 0)
        self.assertEqual(draft_mode.seed_for_result(task(disable_seed_increment=True), 3), 100)

    def test_errors(self):
        cases = [
            (task(draft_mode=False), 0, 'Draft mode on'),
            (task(), None, 'Click one'),
            (task(), 4, 'not the grid'),
            (task(prompt_queue=True, prompt='a\nb'), 0, 'prompt-queue'),
            (task(should_enhance=True), 0, 'Enhance'),
        ]
        for t, index, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                draft_mode.seed_for_result(t, index)

    def test_single_line_queue_is_fine(self):
        self.assertEqual(draft_mode.seed_for_result(task(prompt_queue=True, prompt='a fox\n\n'), 2), 102)


if __name__ == '__main__':
    unittest.main()
