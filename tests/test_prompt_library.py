import json
import os
import shutil
import tempfile
import unittest

from modules.prompt_library import PromptLibrary, label


class TestPromptLibrary(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.lib = PromptLibrary(self.root)

    def test_save_search_delete(self):
        fox = self.lib.save('a red fox in snow', 'blurry', name='Fox', tags='Animals, winter, animals')
        self.assertEqual(fox['tags'], ['animals', 'winter'])
        cabin = self.lib.save('a cozy cabin at night', tags='scenes')
        self.assertTrue(cabin['name'].startswith('a cozy cabin'))
        self.assertEqual([e['name'] for e in self.lib.search()], [cabin['name'], 'Fox'])  # newest first
        self.assertEqual([e['id'] for e in self.lib.search('fox snow')], [fox['id']])
        self.assertEqual([e['id'] for e in self.lib.search(tag='scenes')], [cabin['id']])
        self.assertEqual(self.lib.search('blurry')[0]['id'], fox['id'])  # negative is searchable
        self.assertEqual(self.lib.all_tags(), ['animals', 'scenes', 'winter'])
        self.assertTrue(self.lib.delete(fox['id']))
        self.assertFalse(self.lib.delete(fox['id']))
        self.assertIsNone(self.lib.get(fox['id']))
        self.assertIn('[scenes]', label(cabin))

    def test_saving_same_prompt_updates_instead_of_duplicating(self):
        first = self.lib.save('a lighthouse', 'text', tags='sea')
        again = self.lib.save('a lighthouse', 'text', name='Lighthouse', tags='')
        self.assertEqual(first['id'], again['id'])
        self.assertEqual(len(self.lib.load()), 1)
        self.assertEqual(self.lib.get(first['id'])['name'], 'Lighthouse')
        self.assertEqual(self.lib.get(first['id'])['tags'], ['sea'])  # empty tags keep the old ones
        self.lib.save('a lighthouse', 'different negative')
        self.assertEqual(len(self.lib.load()), 2)

    def test_empty_prompt_rejected_and_bad_file_ignored(self):
        with self.assertRaises(ValueError):
            self.lib.save('   ')
        with open(os.path.join(self.root, '.prompt_library.json'), 'w') as f:
            f.write('{not json')
        self.assertEqual(self.lib.load(), [])
        with open(os.path.join(self.root, '.prompt_library.json'), 'w') as f:
            json.dump([{'id': 'x', 'prompt': 'ok'}, {'nope': 1}, 'junk'], f)
        self.assertEqual([e['id'] for e in self.lib.load()], ['x'])


if __name__ == '__main__':
    unittest.main()
