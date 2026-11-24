import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

import modules.config as config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IsolatedConfigState(unittest.TestCase):
    """Give each test its own config_dict / key lists and a temp directory."""

    def setUp(self):
        for name in ('config_dict', 'visited_keys', 'always_save_keys'):
            patcher = mock.patch.object(config, name, type(getattr(config, name))())
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: subprocess.run(['rm', '-rf', self.tmp]))

    def env(self, **values):
        patcher = mock.patch.dict(os.environ, values)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestGetConfigPath(unittest.TestCase):
    def test_env_var_wins(self):
        with mock.patch.dict(os.environ, {'fooocus_test_cfg': '/x/y.txt'}):
            self.assertEqual(config.get_config_path('fooocus_test_cfg', './config.txt'), '/x/y.txt')

    def test_default_is_made_absolute(self):
        os.environ.pop('fooocus_test_cfg', None)
        self.assertEqual(config.get_config_path('fooocus_test_cfg', 'rel.txt'), os.path.abspath('rel.txt'))


class TestGetConfigItemOrSetDefault(IsolatedConfigState):
    def get(self, key, default, validator=lambda x: isinstance(x, (int, float)), **kw):
        return config.get_config_item_or_set_default(key, default, validator, **kw)

    def test_missing_key_uses_and_stores_default(self):
        self.assertEqual(self.get('k', 4.0), 4.0)
        self.assertEqual(config.config_dict['k'], 4.0)
        self.assertIn('k', config.visited_keys)

    def test_valid_value_is_returned(self):
        config.config_dict['k'] = 7.5
        self.assertEqual(self.get('k', 4.0), 7.5)

    def test_invalid_value_falls_back_and_is_replaced(self):
        config.config_dict['k'] = 'not a number'
        self.assertEqual(self.get('k', 4.0), 4.0)
        self.assertEqual(config.config_dict['k'], 4.0)

    def test_empty_values_become_string_none(self):
        config.config_dict['k'] = ''
        self.assertEqual(self.get('k', 'x', validator=lambda v: v == 'None'), 'None')
        config.config_dict['k2'] = ''
        self.assertEqual(self.get('k2', 'x', validator=lambda v: v == '', disable_empty_as_none=True), '')

    def test_env_var_overrides_and_is_typed(self):
        config.config_dict['k'] = 7.5
        self.env(k='9.0')
        self.assertEqual(self.get('k', 4.0, expected_type=float), 9.0)

    def test_env_var_bool_is_case_insensitive(self):
        self.env(flag='false')
        self.assertIs(self.get('flag', True, validator=lambda x: isinstance(x, bool), expected_type=bool), False)

    def test_env_var_of_wrong_type_is_rejected_by_validator(self):
        self.env(k='hello')
        self.assertEqual(self.get('k', 4.0, expected_type=float), 4.0)


class TestGetDirOrSetDefault(IsolatedConfigState):
    def test_existing_dir_is_used(self):
        config.config_dict['p'] = self.tmp
        self.assertEqual(config.get_dir_or_set_default('p', '/unused'), self.tmp)
        self.assertEqual(config.get_dir_or_set_default('p', '/unused', as_array=True), [self.tmp])
        self.assertIn('p', config.always_save_keys)

    def test_list_of_existing_dirs(self):
        other = os.path.join(self.tmp, 'b')
        os.makedirs(other)
        config.config_dict['p'] = [self.tmp, other]
        self.assertEqual(config.get_dir_or_set_default('p', ['/unused']), [self.tmp, other])

    def test_missing_dir_falls_back_to_default_and_creates_it(self):
        config.config_dict['p'] = os.path.join(self.tmp, 'does-not-exist')
        fallback = os.path.join(self.tmp, 'fallback')
        self.assertEqual(config.get_dir_or_set_default('p', fallback), fallback)
        self.assertTrue(os.path.isdir(fallback))
        self.assertEqual(config.config_dict['p'], fallback)

    def test_make_directory_creates_configured_dir(self):
        wanted = os.path.join(self.tmp, 'made')
        config.config_dict['p'] = wanted
        self.assertEqual(config.get_dir_or_set_default('p', '/unused', make_directory=True), wanted)
        self.assertTrue(os.path.isdir(wanted))

    def test_env_var_overrides_config(self):
        config.config_dict['p'] = '/nope'
        self.env(p=self.tmp)
        self.assertEqual(config.get_dir_or_set_default('p', '/unused'), self.tmp)


class TestPresets(unittest.TestCase):
    def test_get_presets_lists_initial_and_files(self):
        presets = config.get_presets()
        self.assertEqual(presets[0], 'initial')
        self.assertIn('default', presets)

    def test_try_get_preset_content(self):
        self.assertEqual(config.try_get_preset_content('default')['default_model'],
                         'juggernautXL_v8Rundiffusion.safetensors')
        self.assertEqual(config.try_get_preset_content('does-not-exist'), {})
        self.assertEqual(config.try_get_preset_content(None), {})


class TestHelpers(IsolatedConfigState):
    def test_add_ratio(self):
        self.assertEqual(config.add_ratio('1024*1024'), '1024×1024 <span style="color: grey;"> ∣ 1:1</span>')
        self.assertTrue(config.add_ratio('1152*896').endswith('9:7</span>'))

    def test_init_temp_path(self):
        default = os.path.join(self.tmp, 'default')
        custom = os.path.join(self.tmp, 'custom')
        with mock.patch.object(config.args_manager.args, 'temp_path', None):
            self.assertEqual(config.init_temp_path(custom, default), custom)
            self.assertTrue(os.path.isdir(custom))
            self.assertEqual(config.init_temp_path('', default), default)
            self.assertTrue(os.path.isdir(default))

    def test_get_model_filenames(self):
        for name in ('a.safetensors', 'b.ckpt', 'notes.txt'):
            open(os.path.join(self.tmp, name), 'w').close()
        self.assertEqual(sorted(config.get_model_filenames(self.tmp)), ['a.safetensors', 'b.ckpt'])


class TestConfigLoadingEndToEnd(unittest.TestCase):
    """Import modules.config in a fresh process against a temp config.txt."""

    PROBE = ('default_model', 'default_cfg_scale', 'default_sample_sharpness', 'path_outputs')

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: subprocess.run(['rm', '-rf', self.tmp]))
        self.config_path = os.path.join(self.tmp, 'config.txt')
        self.tutorial_path = os.path.join(self.tmp, 'tutorial.txt')
        self.outputs = os.path.join(self.tmp, 'outputs')

    def write_config(self, data):
        with open(self.config_path, 'w', encoding='utf-8') as fh:
            fh.write(data if isinstance(data, str) else json.dumps(data))

    def load(self, argv=(), **env):
        code = (
            'import sys, json\n'
            f'sys.argv = ["x", *{list(argv)!r}]\n'
            'import modules.config as c\n'
            f'print("RESULT=" + json.dumps({{k: getattr(c, k) for k in {list(self.PROBE)!r}}}))\n'
        )
        full_env = {k: v for k, v in os.environ.items() if not k.startswith('default_')}
        full_env.update(config_path=self.config_path, config_example_path=self.tutorial_path, **env)
        proc = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT, env=full_env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        line = next(l for l in proc.stdout.splitlines() if l.startswith('RESULT='))
        return json.loads(line[len('RESULT='):])

    def test_config_overrides_default_preset_and_invalid_values_fall_back(self):
        self.write_config({'default_cfg_scale': 7.5, 'default_sample_sharpness': 'bad',
                           'path_outputs': self.outputs})
        r = self.load()
        self.assertEqual(r['default_model'], 'juggernautXL_v8Rundiffusion.safetensors')  # from default preset
        self.assertEqual(r['default_cfg_scale'], 7.5)        # from config.txt
        self.assertEqual(r['default_sample_sharpness'], 2.0)  # invalid -> default
        self.assertEqual(r['path_outputs'], self.outputs)
        self.assertTrue(os.path.isdir(self.outputs))

    def test_preset_argument_overrides_config(self):
        self.write_config({'default_model': 'mine.safetensors', 'path_outputs': self.outputs})
        self.assertEqual(self.load(['--preset', 'anime'])['default_model'], 'animaPencilXL_v500.safetensors')

    def test_env_var_overrides_everything(self):
        self.write_config({'default_cfg_scale': 7.5, 'path_outputs': self.outputs})
        self.assertEqual(self.load(['--preset', 'anime'], default_cfg_scale='9.0')['default_cfg_scale'], 9.0)

    def test_first_launch_writes_config_and_tutorial(self):
        self.load(path_outputs=self.outputs)
        with open(self.config_path, encoding='utf-8') as fh:
            written = json.load(fh)
        self.assertIn('path_checkpoints', written)
        self.assertEqual(written['path_outputs'], self.outputs)
        with open(self.tutorial_path, encoding='utf-8') as fh:
            self.assertIn('default_cfg_scale', fh.read())

    def test_existing_config_is_not_rewritten(self):
        original = json.dumps({'path_outputs': self.outputs, 'default_cfg_scale': 7.5})
        self.write_config(original)
        self.load()
        with open(self.config_path, encoding='utf-8') as fh:
            self.assertEqual(fh.read(), original)

    def test_malformed_config_loads_defaults_and_is_left_alone(self):
        broken = '{"default_cfg_scale": 7.5,}'
        self.write_config(broken)
        r = self.load(path_outputs=self.outputs)
        self.assertEqual(r['default_cfg_scale'], 4.0)
        with open(self.config_path, encoding='utf-8') as fh:
            self.assertEqual(fh.read(), broken)


if __name__ == '__main__':
    unittest.main()
