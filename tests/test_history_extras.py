import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import os
import shutil
import tempfile
import unittest

from PIL import Image

from modules import history


class TestThumbnailsRatingsFilters(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.day = os.path.join(self.root, '2026-11-25')
        os.makedirs(self.day)
        self.image = os.path.join(self.day, 'a.png')
        Image.new('RGB', (1152, 896), 'red').save(self.image)

    def test_thumbnail_cached_outside_dated_folders(self):
        thumb = history.ensure_thumbnail(self.image, self.root)
        self.assertTrue(thumb.startswith(os.path.join(self.root, '.thumbs', '2026-11-25')))
        with Image.open(thumb) as im:
            self.assertLessEqual(max(im.size), history.THUMB_SIZE)
        mtime = os.path.getmtime(thumb)
        self.assertEqual(history.ensure_thumbnail(self.image, self.root), thumb)
        self.assertEqual(os.path.getmtime(thumb), mtime)  # reused, not rebuilt
        names = [i.filename for i in history.HistoryIndex(embedded_reader=lambda p: None).scan(self.root)]
        self.assertEqual(names, ['a.png'])  # thumbnails are never listed as images
        self.assertFalse(history.is_deletable(thumb, self.root))

    def test_thumbnail_falls_back_to_original(self):
        bad = os.path.join(self.day, 'broken.png')
        open(bad, 'wb').write(b'not an image')
        self.assertEqual(history.ensure_thumbnail(bad, self.root), bad)

    def test_delete_removes_thumbnail(self):
        thumb = history.ensure_thumbnail(self.image, self.root)
        history.delete_items([self.image], self.root, os.path.join(self.root, 'trash'))
        self.assertFalse(os.path.exists(thumb))

    def test_ratings_round_trip_and_survive_delete_undo(self):
        store = history.RatingsStore(self.root)
        self.assertEqual(store.get(self.image), {'favourite': False, 'stars': 0})
        store.update(self.image, favourite=True)
        store.update(self.image, stars=9)  # clamped
        self.assertEqual(store.get(self.image), {'favourite': True, 'stars': 5})
        moves, _ = history.delete_items([self.image], self.root, os.path.join(self.root, 'trash'))
        history.restore_items(moves)
        self.assertEqual(store.get(self.image), {'favourite': True, 'stars': 5})
        store.update(self.image, favourite=False, stars=0)
        self.assertEqual(store.load(), {})  # empty entries are dropped

    def test_filters(self):
        def item(name, day, model):
            return history.HistoryItem(os.path.join(self.root, day, name), name, day, 0, {'base_model': model})
        items = [item('a.png', '2026-11-25', 'juggernaut'), item('b.png', '2026-11-25', 'flux1.gguf'),
                 item('c.png', '2026-11-24', 'juggernaut')]
        models, days = history.filter_choices(items)
        self.assertEqual(models, [history.ALL_MODELS, 'flux1.gguf', 'juggernaut'])
        self.assertEqual(days, [history.ALL_DAYS, '2026-11-25', '2026-11-24'])

        def names(xs):
            return [x.filename for x in xs]
        self.assertEqual(names(history.filter_items(items, model='juggernaut')), ['a.png', 'c.png'])
        self.assertEqual(names(history.filter_items(items, day='2026-11-24')), ['c.png'])
        ratings = {'2026-11-25/b.png': {'favourite': True, 'stars': 2}, '2026-11-24/c.png': {'stars': 4}}
        self.assertEqual(names(history.filter_items(items, favourites_only=True, ratings=ratings)), ['b.png'])
        self.assertEqual(names(history.filter_items(items, min_stars=3, ratings=ratings)), ['c.png'])
        self.assertEqual(names(history.filter_items(items, history.ALL_MODELS, history.ALL_DAYS)), names(items))
        self.assertIn('♥', history.caption(items[1], ratings['2026-11-25/b.png']))


class TestLogTidy(unittest.TestCase):
    def setUp(self):
        from modules import private_logger
        self.pl = private_logger
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.log = os.path.join(self.dir, 'log.html')
        names = ('2026-11-25_10-00-03_1.png', '2026-11-25_10-00-02_2.png', '2026-11-25_10-00-01_3.png')
        items = ''.join(private_logger.build_log_item(n, [('Prompt', 'prompt', n)]) for n in names)
        with open(self.log, 'w', encoding='utf-8') as f:
            f.write('<html><body>head' + private_logger.LOG_SPLIT + '\n\n' + items + private_logger.LOG_SPLIT
                    + '</body></html>')

    def tearDown(self):
        self.pl.log_cache.pop(self.log, None)

    def entries(self):
        with open(self.log, encoding='utf-8') as f:
            return list(history.parse_log_html(f.read()))

    def test_remove_and_restore_in_order(self):
        removed = self.pl.remove_log_entries(self.log, ['2026-11-25_10-00-02_2.png'])
        self.assertEqual(list(removed), ['2026-11-25_10-00-02_2.png'])
        self.assertEqual(self.entries(), ['2026-11-25_10-00-03_1.png', '2026-11-25_10-00-01_3.png'])
        with open(self.log, encoding='utf-8') as f:
            text = f.read()
        self.assertTrue(text.startswith('<html><body>head') and text.endswith('</body></html>'))
        self.assertEqual(self.pl.restore_log_entries(self.log, list(removed.values())), 1)
        self.assertEqual(self.entries(), ['2026-11-25_10-00-03_1.png', '2026-11-25_10-00-02_2.png',
                                          '2026-11-25_10-00-01_3.png'])
        self.assertEqual(self.pl.restore_log_entries(self.log, list(removed.values())), 0)  # no duplicates

    def test_cache_is_updated_so_next_image_does_not_resurrect_entries(self):
        with open(self.log, encoding='utf-8') as f:
            self.pl.log_cache[self.log] = f.read().split(self.pl.LOG_SPLIT)[1]
        self.pl.remove_log_entries(self.log, ['2026-11-25_10-00-03_1.png'])
        self.assertNotIn('10-00-03_1', self.pl.log_cache[self.log])

    def test_delete_items_tidies_log_and_undo_restores_it(self):
        day = os.path.join(self.dir, 'out', '2026-11-25')
        os.makedirs(day)
        shutil.move(self.log, os.path.join(day, 'log.html'))
        self.log = os.path.join(day, 'log.html')
        image = os.path.join(day, '2026-11-25_10-00-02_2.png')
        open(image, 'wb').write(b'x')
        moves, _ = history.delete_items([image], os.path.join(self.dir, 'out'), os.path.join(self.dir, 'trash'),
                                        log_remover=self.pl.remove_log_entries)
        self.assertIsNotNone(moves[0][2])
        self.assertNotIn('2026-11-25_10-00-02_2.png', self.entries())
        history.restore_items(moves, log_restorer=self.pl.restore_log_entries)
        self.assertIn('2026-11-25_10-00-02_2.png', self.entries())

    def test_missing_or_unsplit_log_is_left_alone(self):
        self.assertEqual(self.pl.remove_log_entries(os.path.join(self.dir, 'nope.html'), ['x.png']), {})
        plain = os.path.join(self.dir, 'plain.html')
        with open(plain, 'w') as f:
            f.write('<html>no markers</html>')
        self.assertEqual(self.pl.remove_log_entries(plain, ['x.png']), {})
        with open(plain) as f:
            self.assertEqual(f.read(), '<html>no markers</html>')


if __name__ == '__main__':
    unittest.main()
