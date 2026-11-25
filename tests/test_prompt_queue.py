import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import unittest
from unittest import mock

from modules.prompt_queue import queue_progress_title, split_prompt_queue

# modules.async_worker imports model_management, which probes the GPU at
# import time and raises on hosts with neither MPS nor CUDA.
try:
    import modules.async_worker as async_worker
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the host
    async_worker = None
    _IMPORT_ERROR = exc


class TestSplitPromptQueue(unittest.TestCase):
    def test_one_prompt_per_non_empty_line(self):
        self.assertEqual(split_prompt_queue('a cat\n\n  a dog  \n\t\na fox'), ['a cat', 'a dog', 'a fox'])

    def test_empty_and_none(self):
        self.assertEqual(split_prompt_queue(''), [])
        self.assertEqual(split_prompt_queue(None), [])
        self.assertEqual(split_prompt_queue('\n \n'), [])

    def test_windows_line_endings(self):
        self.assertEqual(split_prompt_queue('a\r\nb\r\n'), ['a', 'b'])


class TestQueueProgressTitle(unittest.TestCase):
    def test_short_prompt(self):
        self.assertEqual(queue_progress_title(2, 5, 'a cat'), 'Prompt 2/5: a cat')

    def test_long_prompt_is_truncated(self):
        title = queue_progress_title(1, 1, 'x' * 200, max_len=10)
        self.assertEqual(title, 'Prompt 1/1: ' + 'x' * 9 + '…')


@unittest.skipIf(async_worker is None, f'async_worker unavailable on this host: {_IMPORT_ERROR!r}')
class TestAsyncTaskQueueSupport(unittest.TestCase):
    def test_default_is_off_without_the_control(self):
        self.assertFalse(getattr(async_worker.AsyncTask([]), 'prompt_queue', False))

    def test_child_forwards_stop_and_skip_to_parent(self):
        parent = async_worker.AsyncTask([])
        child = async_worker.AsyncTask([])
        child.parent = parent
        parent.last_stop = 'stop'          # Stop button sets it on the parent
        self.assertEqual(child.last_stop, 'stop')
        child.last_stop = False            # handler clears it after a skip
        self.assertIs(parent.last_stop, False)

    def test_make_queue_child_swaps_only_the_prompt(self):
        parent = async_worker.AsyncTask([])
        parent.args = [True, 'a cat\na dog', 'blurry', ['Fooocus V2'], 'other']
        captured = {}

        class FakeTask:
            def __init__(self, args):
                captured['args'] = args
                self.parent = None

        # make_queue_child builds the child via the module-level AsyncTask name.
        with mock.patch.object(async_worker, 'AsyncTask', FakeTask):
            child = parent.make_queue_child('a dog')

        self.assertEqual(captured['args'], [True, 'a dog', 'blurry', ['Fooocus V2'], 'other'])
        self.assertEqual(parent.args[1], 'a cat\na dog')  # parent's inputs untouched
        self.assertIs(child.parent, parent)
        self.assertIs(child.yields, parent.yields)
        self.assertIs(child.results, parent.results)


if __name__ == '__main__':
    unittest.main()
