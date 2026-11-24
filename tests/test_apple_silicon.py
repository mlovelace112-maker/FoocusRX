import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import os
import tempfile
import unittest
from unittest import mock

from modules import apple_silicon


class TestReadVersionString(unittest.TestCase):
    def write(self, text):
        fd, path = tempfile.mkstemp(suffix='.py')
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_reads_plain_assignment(self):
        self.assertEqual(apple_silicon._read_version_string(self.write("version = '2.6.4'")), '2.6.4')

    def test_ignores_other_names_and_finds_version(self):
        path = self.write("# comment\nname = 'x'\nversion = \"3.0.0\"\n")
        self.assertEqual(apple_silicon._read_version_string(path), '3.0.0')

    def test_missing_or_non_string_version_is_empty(self):
        self.assertEqual(apple_silicon._read_version_string(self.write("name = 'x'")), '')
        self.assertEqual(apple_silicon._read_version_string(self.write("version = 2")), '')
        self.assertEqual(apple_silicon._read_version_string(self.write("version = '1' + '2'")), '')

    def test_file_contents_are_never_executed(self):
        marker = os.path.join(tempfile.gettempdir(), f'fooocus_exec_marker_{os.getpid()}')
        self.addCleanup(lambda: os.path.exists(marker) and os.remove(marker))
        path = self.write(f"open({marker!r}, 'w').close()\nversion = '9.9.9'\n")
        self.assertEqual(apple_silicon._read_version_string(path), '9.9.9')
        self.assertFalse(os.path.exists(marker))

    def test_current_version_matches_repo_file(self):
        import fooocus_version
        self.assertEqual(apple_silicon._current_fooocus_version(), fooocus_version.version)

    def test_current_version_is_empty_on_unreadable_file(self):
        with mock.patch.object(apple_silicon, '_read_version_string', side_effect=OSError):
            self.assertEqual(apple_silicon._current_fooocus_version(), '')


if __name__ == '__main__':
    unittest.main()
