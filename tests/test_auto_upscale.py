import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import datetime
import os
import shutil
import tempfile
import unittest

from modules import auto_upscale as au


class TestNaming(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(au.upscaled_path('/o/2026-11-24/a.png'), '/o/2026-11-24/a_2x.png')
        self.assertEqual(au.upscaled_path('/o/x/b.jpeg'), '/o/x/b_2x.png')
        self.assertTrue(au.is_upscaled('/o/a_2x.png'))
        self.assertFalse(au.is_upscaled('/o/a.png'))


class TestFindPending(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)

    def touch(self, rel):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w').close()
        return path

    def test_finds_only_images_without_an_upscale(self):
        a = self.touch('2026-11-23/a.png')
        b = self.touch('2026-11-24/b.png')
        self.touch('2026-11-24/c.png')
        self.touch('2026-11-24/c_2x.png')              # c already done
        self.touch('2026-11-24/d_2x.png.partial')       # interrupted write: ignored
        self.touch('2026-11-24/log.html')
        self.touch('notadate/e.png')
        self.assertEqual(au.find_pending(self.root), [a, b])

    def test_since_filter(self):
        self.touch('2026-11-23/a.png')
        b = self.touch('2026-11-24/b.png')
        self.assertEqual(au.find_pending(self.root, datetime.date(2026, 11, 24)), [b])

    def test_missing_root(self):
        self.assertEqual(au.find_pending(os.path.join(self.root, 'nope')), [])


class TestAppDetection(unittest.TestCase):
    def test_process_list(self):
        ps = '\n'.join([
            ' 100 /usr/bin/zsh',
            ' 200 /Users/me/FoocusRX/venv/bin/python launch.py --port 7865',
            ' 300 /Users/me/FoocusRX/venv/bin/python scripts/auto_upscale.py --once',
        ])
        self.assertTrue(au.app_process_running(ps, own_pid=300))
        self.assertFalse(au.app_process_running(ps.replace('launch.py', 'other.py'), own_pid=300))
        self.assertFalse(au.app_process_running(' 200 grep launch.py', own_pid=1))  # not python
        self.assertFalse(au.app_process_running(' 200 /x/python launch.py', own_pid=200))  # ourselves

    def test_closed_port(self):
        self.assertFalse(au.port_in_use(1))


STYLES = {
    'Wrap': ('cinematic photo of {prompt}, film grain', 'cartoon, painting'),
    'Extra': ('high detail', 'blurry'),
}


class TestBuildWorkloads(unittest.TestCase):
    def test_like_the_worker(self):
        meta = {'prompt': 'a fox', 'negative_prompt': 'ugly', 'styles': "['Fooocus V2', 'Wrap', 'Extra', 'Missing']",
                'prompt_expansion': 'a fox, golden light'}
        positive, negative, top_k = au.build_workloads(meta, STYLES)
        self.assertEqual(positive, ['cinematic photo of a fox, film grain', 'high detail', 'a fox, golden light'])
        self.assertEqual(top_k, 2)  # the expansion isn't pooled
        self.assertEqual(negative, ['cartoon, painting', 'blurry', 'ugly'])

    def test_prompt_first_when_no_placeholder_and_no_expansion(self):
        positive, negative, top_k = au.build_workloads(
            {'prompt': 'a fox', 'styles': "['Extra']", 'prompt_expansion': 'unused'}, STYLES)
        self.assertEqual((positive, top_k), (['a fox', 'high detail'], 2))
        self.assertEqual(negative, ['blurry'])

    def test_empty_metadata(self):
        self.assertEqual(au.build_workloads({}, STYLES), ([''], [''], 1))


class TestSettingsFromMetadata(unittest.TestCase):
    def test_parses_log_style_strings(self):
        s = au.settings_from_metadata({
            'performance': 'Quality', 'sampler': 'dpmpp_2m', 'scheduler': 'karras', 'guidance_scale': '7.5',
            'sharpness': '0.0', 'adm_guidance': '(1.5, 0.8, 0.3)', 'clip_skip': '2', 'seed': '3966941723112853047',
            'base_model': 'm.safetensors', 'vae': 'Default (model)',
            'lora_combined_1': 'a.safetensors : 0.5', 'lora_combined_2': 'False : b.safetensors : 1.0',
            'lora_combined_3': 'True : c.safetensors : 0.25', 'lora_combined_4': 'None : 1.0',
        }, steps_for_performance=lambda p: {'Quality': 36}.get(p, 18))
        self.assertEqual(s['steps'], 36)
        self.assertEqual((s['cfg_scale'], s['sharpness'], s['clip_skip']), (7.5, 0.0, 2))
        self.assertEqual(s['seed'], 3966941723112853047)
        self.assertEqual(s['adm_guidance'], (1.5, 0.8, 0.3))
        self.assertEqual(s['loras'], [('a.safetensors', 0.5), ('c.safetensors', 0.25)])

    def test_defaults(self):
        s = au.settings_from_metadata({}, steps_for_performance=lambda p: 18)
        self.assertEqual((s['performance'], s['steps'], s['sampler'], s['cfg_scale'], s['seed']),
                         ('Speed', 18, 'dpmpp_2m_sde_gpu', 4.0, 0))


if __name__ == '__main__':
    unittest.main()
