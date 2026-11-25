import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from modules import app_service, updater, upgrader

REQS = """torchsde==0.2.6
# comment line
transformers==4.42.4
gradio==5.50.0
# Web-stack pins
fastapi>=0.115.0,<0.116
Pillow==10.4.0  # trailing comment
"""


class TestPins(unittest.TestCase):
    def test_parse_and_replace(self):
        pins = upgrader.parse_pins(REQS)
        self.assertEqual(pins['gradio'], '5.50.0')
        self.assertEqual(pins['pillow'], '10.4.0')
        self.assertNotIn('fastapi', pins)  # ranges are not pins
        out = upgrader.replace_pins(REQS, {'Gradio': '6.0.0', 'pillow': '11.0.0'})
        self.assertIn('gradio==6.0.0\n', out)
        self.assertIn('Pillow==11.0.0  # trailing comment\n', out)
        self.assertIn('transformers==4.42.4\n', out)
        self.assertIn('# Web-stack pins\nfastapi>=0.115.0,<0.116\n', out)  # untouched byte for byte
        self.assertEqual(upgrader.replace_pins(REQS, {}), REQS)

    def test_parse_request(self):
        self.assertEqual(upgrader.parse_request('gradio==5.50.0'), ('gradio', '5.50.0'))
        self.assertEqual(upgrader.parse_request('gradio'), ('gradio', None))
        with self.assertRaises(updater.UpdateError):
            upgrader.parse_request('gradio; rm -rf /')

    def test_pinned_version(self):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as fh:
            fh.write(REQS)
        self.addCleanup(os.remove, fh.name)
        self.assertEqual(updater.pinned_version(fh.name, 'gradio'), '5.50.0')
        self.assertIsNone(updater.pinned_version(fh.name, 'fastapi'))
        self.assertIsNone(updater.pinned_version('/nope', 'gradio'))


def sh(cwd, *args):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@unittest.skipUnless(shutil.which('git'), 'git not installed')
class TestUpdatePreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.origin = os.path.join(self.tmp, 'origin.git')
        self.repo = os.path.join(self.tmp, 'repo')
        sh(self.tmp, 'git', 'init', '-q', '--bare', '-b', 'main', self.origin)
        sh(self.tmp, 'git', 'clone', '-q', self.origin, self.repo)
        for k, v in (('user.email', 't@example.invalid'), ('user.name', 'T')):
            sh(self.repo, 'git', 'config', k, v)
        os.makedirs(os.path.join(self.repo, 'models'))
        for name in ('README.md', 'models/keep.txt'):
            open(os.path.join(self.repo, name), 'w').write('x\n')
        sh(self.repo, 'git', 'add', '-A')
        sh(self.repo, 'git', 'commit', '-qm', 'init')
        sh(self.repo, 'git', 'push', '-q', 'origin', 'main')
        os.makedirs(os.path.join(self.repo, 'venv', 'bin'))
        os.symlink(sys.executable, os.path.join(self.repo, 'venv', 'bin', 'python'))

    def test_model_changes_are_not_blocking_but_code_changes_are(self):
        os.remove(os.path.join(self.repo, 'models', 'keep.txt'))
        self.assertEqual(updater.blocking_changes(self.repo), [])
        open(os.path.join(self.repo, 'README.md'), 'w').write('edited\n')
        self.assertEqual(updater.blocking_changes(self.repo), ['README.md'])
        with self.assertRaisesRegex(updater.UpdateError, 'README.md'):
            updater.update(self.repo, log=lambda *a: None)

    def test_wrong_branch_refused(self):
        sh(self.repo, 'git', 'switch', '-q', '-c', 'feature')
        with self.assertRaisesRegex(updater.UpdateError, 'not "main"'):
            updater.update(self.repo, log=lambda *a: None)

    def test_diverged_refused(self):
        other = os.path.join(self.tmp, 'other')
        sh(self.tmp, 'git', 'clone', '-q', self.origin, other)
        for k, v in (('user.email', 't@example.invalid'), ('user.name', 'T')):
            sh(other, 'git', 'config', k, v)
        open(os.path.join(other, 'a.txt'), 'w').write('a')
        sh(other, 'git', 'add', '-A')
        sh(other, 'git', 'commit', '-qm', 'remote change')
        sh(other, 'git', 'push', '-q', 'origin', 'main')
        open(os.path.join(self.repo, 'b.txt'), 'w').write('b')
        sh(self.repo, 'git', 'add', '-A')
        sh(self.repo, 'git', 'commit', '-qm', 'local change')
        with self.assertRaisesRegex(updater.UpdateError, 'only fast-forwards'):
            updater.update(self.repo, log=lambda *a: None)

    def test_prune_backups_keeps_newest(self):
        for stamp in ('20260101-000000', '20260102-000000', '20260103-000000'):
            os.makedirs(os.path.join(self.repo, f'venv.bak-{stamp}'))
        removed = updater.prune_backups(self.repo, keep=1, log=lambda *a: None)
        self.assertEqual(removed, ['venv.bak-20260101-000000', 'venv.bak-20260102-000000'])
        self.assertTrue(os.path.isdir(os.path.join(self.repo, 'venv.bak-20260103-000000')))

    def test_fix_script_paths(self):
        old = os.path.join(self.tmp, 'venv-candidate')
        new = os.path.join(self.tmp, 'venvx')
        os.makedirs(os.path.join(new, 'bin'))
        with open(os.path.join(new, 'bin', 'pip'), 'w') as fh:
            fh.write(f'#!{old}/bin/python\nimport pip\n')
        updater.fix_script_paths(new, old)
        with open(os.path.join(new, 'bin', 'pip')) as fh:
            self.assertEqual(fh.readline().strip(), f'#!{new}/bin/python')


class TestService(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.svc = app_service.Service(self.tmp, port=1, state_dir=self.tmp, log_dir=self.tmp)

    def test_status_when_nothing_runs(self):
        st = self.svc.status()
        self.assertFalse(st['running'] or st['responding'] or st['port_taken_by_other'])
        self.assertFalse(self.svc.stop())

    def test_never_stops_a_process_that_is_not_ours(self):
        proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(proc.kill)
        with open(self.svc.pidfile, 'w') as fh:
            fh.write(str(proc.pid))
        self.assertTrue(self.svc.pid_alive(proc.pid))
        self.assertFalse(self.svc.pid_is_ours(proc.pid))
        self.assertFalse(self.svc.stop())
        self.assertIsNone(proc.poll())  # still running
        self.assertFalse(os.path.exists(self.svc.pidfile))  # stale pidfile cleared

    def test_start_refuses_during_update_and_without_venv(self):
        open(self.svc.update_lock, 'w').close()
        with self.assertRaisesRegex(app_service.ServiceError, 'update is in progress'):
            self.svc.start()
        os.remove(self.svc.update_lock)
        with self.assertRaises(app_service.NeedsInstall):
            self.svc.start()

    def test_clean_env(self):
        os.environ['HTTPS_PROXY'] = 'http://proxy.invalid:1'
        try:
            env = app_service.clean_env({'X': '1'})
        finally:
            del os.environ['HTTPS_PROXY']
        self.assertNotIn('HTTPS_PROXY', env)
        self.assertEqual(env['NO_PROXY'], 'localhost,127.0.0.1,::1')
        self.assertEqual(env['X'], '1')


if __name__ == '__main__':
    unittest.main()
