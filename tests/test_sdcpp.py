import os
import random
import stat
import sys
import tempfile
import textwrap
import threading
import time
import unittest

from modules.sdcpp import app, registry, runner
from modules.sdcpp.registry import KREA2, MOSS_IMAGE

STYLES = {'Cinematic': ('cinematic still {prompt}, film grain', ''), 'Sharp': ('sharp focus', '')}


class TestRegistry(unittest.TestCase):
    def test_spec_for(self):
        self.assertIs(registry.spec_for('Krea-2-Turbo-Q5_K_S.gguf'), KREA2)
        self.assertIs(registry.spec_for('krea2_turbo-Q4_K_M.gguf'), KREA2)
        self.assertIs(registry.spec_for('MOSS-Image-SFT-Q8_0.gguf'), MOSS_IMAGE)
        self.assertIs(registry.spec_for('moss_image_sft-UD-Q4_K_XL.gguf'), MOSS_IMAGE)
        for other in ('flux1-schnell-Q4_K_S.gguf', 'Qwen3VL-4B-Instruct-Q8_0.gguf', 'Qwen3.5-9B-Q5_K_M.gguf',
                      'krea2.safetensors', 'juggernautXL_v8Rundiffusion.safetensors', None, ''):
            self.assertIsNone(registry.spec_for(other), other)

    def test_files_are_complete(self):
        for spec in registry.MODELS:
            roles = {f.role for f in spec.files}
            self.assertTrue({'diffusion', 'llm', 'vae'} <= roles)
            self.assertTrue(spec.matches(spec.file('diffusion').name))
            for f in spec.files:
                self.assertTrue(f.url.startswith('https://huggingface.co/'))
                self.assertEqual(len(f.sha256), 64)
                self.assertGreater(f.size, 0)
                if f.role != 'diffusion':
                    self.assertFalse(spec.matches(f.name), f.name)  # support files never listed as models
        self.assertIsNotNone(MOSS_IMAGE.file('tokenizer'))  # Qwen3.5 needs the external tokenizer.json
        self.assertTrue(KREA2.license_requires_acceptance)


class TestArgv(unittest.TestCase):
    def test_krea2(self):
        argv = runner.build_argv('/e/sd-cli', KREA2, '/m', 'Krea-2-Turbo-Q5_K_S.gguf', prompt='a cat',
                                 negative_prompt='blurry', width=1024, height=768, seed=7, steps=8, cfg=1.0,
                                 output='/t/out.png')
        self.assertEqual(argv[0], '/e/sd-cli')
        opts = dict(zip(argv[1::2], argv[2::2]))
        self.assertEqual(opts['--diffusion-model'], '/m/Krea-2-Turbo-Q5_K_S.gguf')
        self.assertEqual(opts['--llm'], '/m/Qwen3VL-4B-Instruct-Q8_0.gguf')
        self.assertEqual(opts['--vae'], '/m/qwen_image_vae.safetensors')
        self.assertEqual((opts['-p'], opts['--steps'], opts['--cfg-scale']), ('a cat', '8', '1'))
        self.assertEqual((opts['--sampling-method'], opts['--scheduler']), ('euler', 'simple'))
        self.assertEqual((opts['-W'], opts['-H'], opts['-s'], opts['-o']), ('1024', '768', '7', '/t/out.png'))
        self.assertNotIn('-n', argv)          # CFG 1: no negative branch
        self.assertNotIn('--tokenizer', argv)

    def test_moss_keeps_newlines_and_negative(self):
        argv = runner.build_argv('sd-cli', MOSS_IMAGE, '/m', 'MOSS-Image-SFT-Q8_0.gguf',
                                 prompt='A sign reads:\n"OPEN\nlate"', negative_prompt='lowres', width=1024,
                                 height=1024, seed=1, steps=50, cfg=3.0, output='o.png', extra_args=['--vae-tiling'])
        self.assertEqual(argv[argv.index('-p') + 1], 'A sign reads:\n"OPEN\nlate"')
        self.assertEqual(argv[argv.index('-n') + 1], 'lowres')
        self.assertEqual(argv[argv.index('--tokenizer') + 1], '/m/tokenizer.json')
        self.assertNotIn('--scheduler', argv)  # model default (flow shift 5)
        self.assertEqual(argv[-1], '--vae-tiling')
        no_neg = runner.build_argv('sd-cli', MOSS_IMAGE, '/m', 'x.gguf', prompt='p', negative_prompt='  ',
                                   width=512, height=512, seed=1, steps=4, cfg=3.0, output='o.png')
        self.assertNotIn('-n', no_neg)


class TestProgress(unittest.TestCase):
    def test_parse_sampling_steps_only(self):
        line = '\r  |=====>                     |  3/8 - 1.23s/it - ETA 00:07\x1b[K'
        self.assertEqual(runner.parse_steps(line), [(3, 8)])
        self.assertEqual(runner.parse_steps('  |=====>    | 12/50 - 4.10it/s'), [(12, 50)])
        self.assertEqual(runner.parse_steps('  |=====>    | 120/400 - 512.3MB/s'), [])  # weight loading
        self.assertEqual(runner.parse_steps('[INFO ] loading model'), [])

    def test_split_updates(self):
        pieces, rest = runner.split_updates('a\rb\nc')
        self.assertEqual((pieces, rest), (['a', 'b'], 'c'))


FAKE_CLI = textwrap.dedent('''\
    import sys, time
    args = sys.argv[1:]
    out = args[args.index('-o') + 1]
    steps = int(args[args.index('--steps') + 1])
    mode = args[args.index('-p') + 1]
    print('[INFO ] loading weights', flush=True)
    if mode == 'fail':
        print('[ERROR] cannot load model', flush=True)
        sys.exit(3)
    for i in range(1, steps + 1):
        sys.stdout.write(f'\\r  |===>   | {i}/{steps} - 0.01s/it\\x1b[K' + ('\\n' if i == steps else ''))
        sys.stdout.flush()
        time.sleep(0.5 if mode == 'slow' else 0.01)
    open(out, 'wb').write(b'png')
    print('[INFO ] save result image to ' + out, flush=True)
''')


class TestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = os.path.join(self.tmp.name, 'fake_cli.py')
        with open(self.script, 'w') as fh:
            fh.write(FAKE_CLI)
        self.out = os.path.join(self.tmp.name, 'out.png')

    def tearDown(self):
        self.tmp.cleanup()

    def argv(self, prompt, steps=4):
        return [sys.executable, self.script, '-p', prompt, '--steps', str(steps), '-o', self.out]

    def test_progress_and_output(self):
        steps, logs = [], []
        runner.run(self.argv('ok'), on_step=lambda s, n: steps.append((s, n)), log=logs.append)
        self.assertEqual(steps, [(1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertTrue(os.path.isfile(self.out))
        self.assertTrue(any('loading weights' in line for line in logs))
        self.assertFalse(any('|' in line for line in logs))

    def test_failure_reports_tail(self):
        with self.assertRaises(runner.SdcppError) as ctx:
            runner.run(self.argv('fail'), log=lambda _: None)
        self.assertIn('code 3', str(ctx.exception))
        self.assertIn('cannot load model', str(ctx.exception))

    def test_stop_terminates(self):
        stop = threading.Event()
        started = time.perf_counter()
        threading.Timer(0.6, stop.set).start()
        with self.assertRaises(runner.Stopped):
            runner.run(self.argv('slow', steps=40), should_stop=stop.is_set, log=lambda _: None)
        self.assertLess(time.perf_counter() - started, 10)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(runner._active, set())


class TestApp(unittest.TestCase):
    def test_build_prompt_keeps_newlines(self):
        text = app.build_prompt('A poster that says:\n"HELLO\nWORLD"', ['Fooocus V2', 'Sharp'], STYLES,
                                random.Random(0))
        self.assertEqual(text, 'A poster that says:\n"HELLO\nWORLD", sharp focus')
        self.assertEqual(app.build_prompt('a cat', ['Cinematic'], STYLES, random.Random(0)),
                         'cinematic still a cat, film grain')

    def test_json_layout_is_untouched(self):
        layout = '{"summary": {"caption": "a shop"}, "items": []}'
        self.assertTrue(app.is_json_prompt(layout))
        self.assertEqual(app.build_prompt('  ' + layout + '\n', ['Sharp'], STYLES, random.Random(0)), layout)
        self.assertFalse(app.is_json_prompt('{not json'))
        self.assertFalse(app.is_json_prompt('[1, 2]'))

    def test_default_args(self):
        self.assertEqual(app.default_args('darwin'), ['--auto-fit', 'off', '--mmap'])
        self.assertEqual(app.default_args('linux'), [])
        os.environ[app.EXTRA_ARGS_ENV] = '--vae-tiling --max-vram 8'
        try:
            self.assertEqual(app.extra_args()[-3:], ['--vae-tiling', '--max-vram', '8'])
        finally:
            del os.environ[app.EXTRA_ARGS_ENV]

    def test_snap16(self):
        self.assertEqual(app.snap16(1152), 1152)
        self.assertEqual(app.snap16(1003), 1008)
        self.assertEqual(app.snap16(996), 992)
        self.assertEqual(app.snap16(10), 256)

    def test_model_choices(self):
        with tempfile.TemporaryDirectory() as root:
            engine, krea, moss = (os.path.join(root, d) for d in ('engine', 'krea', 'moss'))
            for d in (engine, krea, moss):
                os.makedirs(d)
            dirs = {'krea2': krea, 'moss_image': moss}
            self.assertEqual(app.model_choices(engine, dirs), [])  # no engine
            cli = os.path.join(engine, 'sd-cli')
            open(cli, 'w').close()
            os.chmod(cli, os.stat(cli).st_mode | stat.S_IXUSR)
            for f in KREA2.files:
                open(os.path.join(krea, f.name), 'w').close()
            open(os.path.join(krea, 'Krea-2-Turbo-Q8_0.gguf.partial'), 'w').close()
            open(os.path.join(moss, MOSS_IMAGE.file('diffusion').name), 'w').close()  # support files missing
            self.assertEqual(app.model_choices(engine, dirs), ['Krea-2-Turbo-Q5_K_S.gguf'])
            for f in MOSS_IMAGE.support_files():
                open(os.path.join(moss, f.name), 'w').close()
            self.assertEqual(app.model_choices(engine, dirs), ['Krea-2-Turbo-Q5_K_S.gguf', 'MOSS-Image-SFT-Q8_0.gguf'])
            self.assertEqual(app.missing_files(MOSS_IMAGE, moss), [])


if __name__ == '__main__':
    unittest.main()
