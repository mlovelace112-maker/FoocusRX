import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import hashlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

from modules import model_loader, model_manager

GOOD_URL = 'https://huggingface.co/org/repo/resolve/main/cool-lora.safetensors'
SHA = 'a' * 64


class TempDirs(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.dirs = {k: os.path.join(self.root, k.lower()) for k in model_manager.KINDS}
        for d in self.dirs.values():
            os.makedirs(d)


class TestPrepareDownload(TempDirs):
    def prep(self, url=GOOD_URL, kind='LoRA', name='', sha=''):
        return model_manager.prepare_download(url, kind, name, sha, self.dirs)

    def test_valid_request_uses_url_file_name(self):
        r = self.prep(sha=SHA.upper())
        self.assertEqual((r.file_name, r.model_dir, r.sha256), ('cool-lora.safetensors', self.dirs['LoRA'], SHA))

    def test_save_as_overrides_and_is_required_without_extension(self):
        url = 'https://civitai.com/api/download/models/12345'
        self.assertEqual(self.prep(url=url, name='my style.safetensors').file_name, 'my style.safetensors')
        with self.assertRaisesRegex(ValueError, 'Only .safetensors'):
            self.prep(url=url)

    def test_rejects_non_https(self):
        for url in ('http://example.com/a.safetensors', 'file:///etc/passwd', 'example.com/a.safetensors', ''):
            with self.assertRaisesRegex(ValueError, 'https'):
                self.prep(url=url)

    def test_rejects_pickle_formats(self):
        for ext in ('ckpt', 'pt', 'bin', 'pth'):
            with self.assertRaisesRegex(ValueError, 'Only .safetensors'):
                self.prep(url=f'https://example.com/model.{ext}')

    def test_rejects_path_tricks_in_name(self):
        for name in ('../evil.safetensors', 'sub/dir.safetensors', '..', 'a;rm.safetensors'):
            with self.assertRaisesRegex(ValueError, 'plain file name|Only .safetensors'):
                self.prep(name=name)

    def test_url_encoded_traversal_in_url_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'plain file name'):
            self.prep(url='https://example.com/x/..%2F..%2Fevil.safetensors')

    def test_bad_sha_kind_and_existing_file(self):
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.prep(sha='abc')
        with self.assertRaisesRegex(ValueError, 'model type'):
            self.prep(kind='Upscaler')
        open(os.path.join(self.dirs['LoRA'], 'cool-lora.safetensors'), 'w').close()
        with self.assertRaisesRegex(ValueError, 'already installed'):
            self.prep()


class TestListInstalled(TempDirs):
    def test_lists_model_files_with_sizes(self):
        with open(os.path.join(self.dirs['Checkpoint'], 'b.safetensors'), 'wb') as fh:
            fh.write(b'x' * 2048)
        os.makedirs(os.path.join(self.dirs['LoRA'], 'styles'))
        open(os.path.join(self.dirs['LoRA'], 'styles', 'a.safetensors'), 'w').close()
        for junk in ('notes.txt', 'b.safetensors.partial', 'b.safetensors.sha256.ok'):
            open(os.path.join(self.dirs['Checkpoint'], junk), 'w').close()
        rows = model_manager.list_installed(self.dirs)
        self.assertEqual([(r[0], r[1], r[2]) for r in rows],
                         [('Checkpoint', 'b.safetensors', '2.0 KB'),
                          ('LoRA', os.path.join('styles', 'a.safetensors'), '0 B')])

    def test_missing_folders_are_skipped(self):
        self.assertEqual(model_manager.list_installed({'Checkpoint': os.path.join(self.root, 'nope')}), [])


class TestHelpers(unittest.TestCase):
    def test_human_size(self):
        self.assertEqual(model_manager.human_size(512), '512 B')
        self.assertEqual(model_manager.human_size(1536), '1.5 KB')
        self.assertEqual(model_manager.human_size(7105348592), '6.6 GB')

    def test_progress_text(self):
        self.assertIn('50%', model_manager.progress_text('m.safetensors', 512, 1024))
        self.assertTrue(model_manager.progress_text('m.safetensors', 1024, 1024).startswith('Verifying'))
        self.assertIn('1.0 KB', model_manager.progress_text('m.safetensors', 1024, None))


class FakeResponse(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.status = 200
        self.headers = {'Content-Length': str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestDownloadWithProgress(TempDirs):
    """End to end through load_file_from_url with the network faked."""

    def fetch(self, data, sha):
        req = model_manager.prepare_download(GOOD_URL, 'LoRA', '', sha, self.dirs)
        calls = []
        with mock.patch.object(model_loader.urllib.request, 'urlopen', return_value=FakeResponse(data)), \
                mock.patch.dict(os.environ, {'HF_MIRROR': 'https://huggingface.co'}):
            path = model_manager.download(req, progress_callback=lambda d, t: calls.append((d, t)))
        return path, calls

    def test_download_reports_progress_and_verifies_hash(self):
        data = os.urandom(3 * (1 << 20) + 17)  # spans several 1 MiB chunks
        path, calls = self.fetch(data, hashlib.sha256(data).hexdigest())
        with open(path, 'rb') as fh:
            self.assertEqual(fh.read(), data)
        self.assertEqual(calls[0], (0, len(data)))
        self.assertEqual(calls[-1], (len(data), len(data)))
        self.assertEqual([d for d, _ in calls], sorted(d for d, _ in calls))
        self.assertFalse(os.path.exists(path + '.partial'))
        # The checksum stamp follows the file (no orphan .partial stamp),
        # so the next load trusts it without re-hashing.
        self.assertTrue(os.path.exists(path + '.sha256.ok'))
        self.assertFalse(os.path.exists(path + '.partial.sha256.ok'))
        with mock.patch.object(model_loader, '_sha256', side_effect=AssertionError('re-hashed')):
            again = model_loader.load_file_from_url(GOOD_URL, model_dir=self.dirs['LoRA'],
                                                    expected_sha256=hashlib.sha256(data).hexdigest())
        self.assertEqual(again, path)

    def test_hash_mismatch_is_quarantined_not_installed(self):
        with self.assertRaisesRegex(RuntimeError, 'sha256 mismatch'):
            self.fetch(b'not the model', SHA)
        final = os.path.join(self.dirs['LoRA'], 'cool-lora.safetensors')
        self.assertFalse(os.path.exists(final))
        self.assertTrue(os.path.exists(final + '.corrupt'))


if __name__ == '__main__':
    unittest.main()
