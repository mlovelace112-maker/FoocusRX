import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import json
import os
import shutil
import tempfile
import time
import unittest

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from modules import history
from modules.private_logger import build_log_item

MODEL = 'juggernautXL_v8Rundiffusion.safetensors'
LORA = 'sd_xl_offset_example-lora_1.0.safetensors'


def log_page(*items):
    """A log.html shaped like private_logger writes it."""
    return '<html><body><!--fooocus-log-split-->' + ''.join(items) + '<!--fooocus-log-split--></body></html>'


def entry(filename, **meta):
    return build_log_item(filename, [(k, k, v) for k, v in meta.items()])


class TestParseLogHtml(unittest.TestCase):
    def test_reads_every_entry_written_by_the_logger(self):
        page = log_page(
            entry('2026-11-24_10-00-00_1111.png', prompt='a fox', seed='1'),
            entry('2026-11-24_09-00-00_2222.jpeg', prompt='<script>x</script>', seed='2'),
        )
        parsed = history.parse_log_html(page)
        self.assertEqual(parsed['2026-11-24_10-00-00_1111.png']['prompt'], 'a fox')
        # Escaped in the HTML, recovered intact from the JSON payload.
        self.assertEqual(parsed['2026-11-24_09-00-00_2222.jpeg']['prompt'], '<script>x</script>')

    def test_ignores_garbage(self):
        self.assertEqual(history.parse_log_html('<html>nothing here</html>'), {})
        bad = '<div id="a_png" class="image-container"> to_clipboard(\'%7Bnot json\')'
        self.assertEqual(history.parse_log_html(bad), {})


class TestScan(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)

    def image(self, day, name, mtime):
        os.makedirs(os.path.join(self.root, day), exist_ok=True)
        path = os.path.join(self.root, day, name)
        Image.new('RGB', (4, 4)).save(path)
        os.utime(path, (mtime, mtime))
        return path

    def write_log(self, day, *items):
        with open(os.path.join(self.root, day, 'log.html'), 'w', encoding='utf-8') as fh:
            fh.write(log_page(*items))

    def test_newest_first_with_log_and_embedded_sources(self):
        now = time.time()
        self.image('2026-11-23', 'old.png', now - 100)
        self.image('2026-11-24', 'b.png', now - 10)
        self.image('2026-11-24', 'a.png', now - 20)
        self.image('2026-11-24', 'nolog.png', now - 5)
        self.write_log('2026-11-24', entry('a.png', prompt='first'), entry('b.png', prompt='second'))
        os.makedirs(os.path.join(self.root, 'not-a-date'))
        reads = []

        def fake_reader(path):
            reads.append(os.path.basename(path))
            return {'prompt': 'embedded'} if path.endswith('nolog.png') else None

        index = history.HistoryIndex(embedded_reader=fake_reader)
        items = index.scan(self.root)
        self.assertEqual([i.filename for i in items], ['nolog.png', 'b.png', 'a.png', 'old.png'])
        self.assertEqual([i.source for i in items], ['image', 'log', 'log', 'none'])
        self.assertEqual(items[0].metadata['prompt'], 'embedded')
        self.assertEqual(items[1].metadata['prompt'], 'second')

        index.scan(self.root)
        self.assertEqual(sorted(reads), ['nolog.png', 'old.png'])  # cached on the second scan

    def test_log_is_reread_when_it_changes(self):
        self.image('2026-11-24', 'a.png', time.time())
        self.write_log('2026-11-24', entry('a.png', prompt='before'))
        index = history.HistoryIndex(embedded_reader=lambda p: None)
        self.assertEqual(index.scan(self.root)[0].metadata['prompt'], 'before')
        log_path = os.path.join(self.root, '2026-11-24', 'log.html')
        self.write_log('2026-11-24', entry('a.png', prompt='after'))
        os.utime(log_path, (time.time() + 5, time.time() + 5))
        self.assertEqual(index.scan(self.root)[0].metadata['prompt'], 'after')

    def test_missing_root(self):
        self.assertEqual(history.HistoryIndex().scan(os.path.join(self.root, 'nope')), [])


class TestReadEmbeddedMetadata(unittest.TestCase):
    def png(self, parameters, scheme):
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        self.addCleanup(os.remove, path)
        info = PngInfo()
        info.add_text('parameters', parameters)
        info.add_text('fooocus_scheme', scheme)
        Image.new('RGB', (4, 4)).save(path, pnginfo=info)
        return path

    def test_fooocus_json(self):
        path = self.png(json.dumps({'prompt': 'a fox', 'seed': '5'}), 'fooocus')
        self.assertEqual(history.read_embedded_metadata(path)['prompt'], 'a fox')

    def test_a1111_text(self):
        path = self.png('a fox, snow\nNegative prompt: blurry\nSteps: 30, Seed: 7, CFG scale: 4', 'a1111')
        meta = history.read_embedded_metadata(path)
        self.assertEqual(meta['prompt'], 'a fox, snow')
        self.assertEqual(meta['seed'], '7')

    def test_no_metadata(self):
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        self.addCleanup(os.remove, path)
        Image.new('RGB', (4, 4)).save(path)
        self.assertIsNone(history.read_embedded_metadata(path))


class TestSearchAndDescribe(unittest.TestCase):
    items = [
        history.HistoryItem('/x/a.png', 'a.png', '2026-11-24', 1.0,
                            {'prompt': 'A Red Fox in snow', 'seed': '123', 'base_model': MODEL}, 'log'),
        history.HistoryItem('/x/b.png', 'b.png', '2026-11-24', 2.0,
                            {'prompt': 'a lighthouse', 'styles': "['Fooocus Sharp']"}, 'log'),
        history.HistoryItem('/x/c.png', 'c.png', '2026-11-23', 3.0, {}, 'none'),
    ]

    def names(self, query):
        return [i.filename for i in history.search(self.items, query)]

    def test_all_terms_case_insensitive(self):
        self.assertEqual(self.names('fox SNOW'), ['a.png'])
        self.assertEqual(self.names('fox lighthouse'), [])

    def test_searches_seed_model_styles_and_filename(self):
        self.assertEqual(self.names('123'), ['a.png'])
        self.assertEqual(self.names('juggernaut'), ['a.png'])
        self.assertEqual(self.names('sharp'), ['b.png'])
        self.assertEqual(self.names('c.png'), ['c.png'])

    def test_empty_query_returns_everything(self):
        self.assertEqual(self.names('  '), ['a.png', 'b.png', 'c.png'])

    def test_describe(self):
        text = history.describe(self.items[0])
        self.assertIn('Prompt: A Red Fox in snow', text)
        self.assertIn('Seed: 123', text)
        self.assertNotIn('Negative prompt', text)
        self.assertIn('No saved settings', history.describe(self.items[2]))


class TestToLoadParameters(unittest.TestCase):
    def convert(self, **meta):
        return history.to_load_parameters(meta, [MODEL, 'sub/other.safetensors'], [LORA], ['myvae.safetensors'])

    def test_model_names(self):
        p = self.convert(base_model='juggernautXL_v8Rundiffusion', refiner_model='None')
        self.assertEqual((p['base_model'], p['refiner_model']), (MODEL, 'None'))
        self.assertEqual(self.convert(base_model=MODEL)['base_model'], MODEL)
        self.assertEqual(self.convert(base_model='other')['base_model'], 'sub/other.safetensors')
        self.assertIsNone(self.convert(base_model='not_installed')['base_model'])

    def test_vae(self):
        self.assertEqual(self.convert(vae='Default (model)')['vae'], 'Default (model)')
        self.assertEqual(self.convert(vae='myvae')['vae'], 'myvae.safetensors')

    def test_loras(self):
        p = self.convert(lora_combined_1='sd_xl_offset_example-lora_1.0 : 0.1',
                         lora_combined_2=f'True : {LORA} : 0.5',
                         lora_combined_3='missing : 1.0',
                         lora_combined_4='garbage')
        self.assertEqual(p['lora_combined_1'], f'{LORA} : 0.1')
        self.assertEqual(p['lora_combined_2'], f'True : {LORA} : 0.5')
        self.assertNotIn('lora_combined_3', p)
        self.assertNotIn('lora_combined_4', p)

    def test_unknown_choices_are_dropped(self):
        p = self.convert(performance='FLUX.1 schnell', sampler='flow-matching euler', scheduler='nope')
        self.assertNotIn('performance', p)
        self.assertNotIn('sampler', p)
        self.assertNotIn('scheduler', p)
        p = self.convert(performance='Speed', sampler='dpmpp_2m_sde_gpu', scheduler='karras')
        self.assertEqual((p['performance'], p['sampler'], p['scheduler']), ('Speed', 'dpmpp_2m_sde_gpu', 'karras'))

    def test_gguf_step_count_is_not_an_sdxl_override(self):
        self.assertNotIn('steps', self.convert(base_model='flux1-schnell-Q4_K_S.gguf', steps=4))
        self.assertEqual(self.convert(base_model=MODEL, steps=30)['steps'], 30)

    def test_does_not_mutate_input(self):
        meta = {'base_model': 'juggernautXL_v8Rundiffusion'}
        history.to_load_parameters(meta, [MODEL], [], [])
        self.assertEqual(meta['base_model'], 'juggernautXL_v8Rundiffusion')


if __name__ == '__main__':
    unittest.main()
