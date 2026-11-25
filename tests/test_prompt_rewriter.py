import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import os
import shutil
import tempfile
import unittest

from modules import prompt_rewriter as pr

# Set FOOOCUS_PROMPT_MODEL_DIR to the folder holding Qwen2.5-1.5B-Instruct/ to
# run the real-model test (it loads ~3 GB).
REAL_BASE = os.environ.get('FOOOCUS_PROMPT_MODEL_DIR')


class TestStyleAndMessages(unittest.TestCase):
    def test_style_for_model(self):
        self.assertEqual(pr.style_for_model('flux1-schnell-Q4_K_S.gguf'), 'flux')
        self.assertEqual(pr.style_for_model('ponyDiffusionV6XL.safetensors'), 'tags')
        self.assertEqual(pr.style_for_model('illustriousXL_v01.safetensors'), 'tags')
        self.assertEqual(pr.style_for_model('juggernautXL_v8Rundiffusion.safetensors'), 'sdxl')
        self.assertEqual(pr.style_for_model(None), 'sdxl')

    def test_messages_have_system_example_and_idea(self):
        msgs = pr.build_messages('  a fox  ', 'flux')
        self.assertEqual([m['role'] for m in msgs], ['system', 'user', 'assistant', 'user'])
        self.assertIn('FLUX.1', msgs[0]['content'])
        self.assertIn('sentences', msgs[0]['content'])
        self.assertEqual(msgs[-1]['content'], 'a fox')
        self.assertIn('tags', pr.build_messages('x', 'tags')[0]['content'])


class TestCleanOutput(unittest.TestCase):
    def test_strips_labels_and_quotes(self):
        self.assertEqual(pr.clean_output('Prompt: "a fox, snow, golden light"', 'sdxl'), 'a fox, snow, golden light')
        self.assertEqual(pr.clean_output('**Rewritten prompt:** a fox, snow', 'sdxl'), 'a fox, snow')

    def test_joins_lines(self):
        self.assertEqual(pr.clean_output('- a fox\n- snow,  golden light\n', 'sdxl'), 'a fox, snow, golden light')
        self.assertEqual(pr.clean_output('A fox sits.\nSnow falls.', 'flux'), 'A fox sits. Snow falls.')

    def test_length_limit_cuts_at_a_boundary(self):
        text = ', '.join(f'term{i}' for i in range(300))
        out = pr.clean_output(text, 'sdxl', max_chars=100)
        self.assertLessEqual(len(out), 100)
        self.assertFalse(out.endswith(','))
        self.assertTrue(out.split(', ')[-1].startswith('term'))


class TestAvailability(unittest.TestCase):
    def test_missing_files(self):
        base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, base)
        self.assertFalse(pr.available(base))
        os.makedirs(pr.model_dir(base))
        for name in pr.REQUIRED_FILES:
            open(os.path.join(pr.model_dir(base), name), 'w').close()
        self.assertTrue(pr.available(base))

    def test_empty_idea_and_missing_model(self):
        with self.assertRaisesRegex(ValueError, 'Type a short idea'):
            pr.rewrite('   ', 'sdxl', base='/nope')
        with self.assertRaisesRegex(FileNotFoundError, 'download_prompt_model'):
            pr.rewrite('a fox', 'sdxl', base='/nope')


@unittest.skipUnless(REAL_BASE and pr.available(REAL_BASE), 'prompt model not available')
class TestRealModel(unittest.TestCase):
    def test_rewrites_in_each_style(self):
        for style in ('sdxl', 'tags', 'flux'):
            with self.subTest(style):
                out = pr.rewrite('a red fox in the snow', style, seed=1, base=REAL_BASE)
                self.assertIn('fox', out.lower())
                self.assertGreater(len(out.split()), 8)
                if style != 'flux':
                    self.assertNotIn('\n', out)


if __name__ == '__main__':
    unittest.main()
