import unittest

from modules import auto_negative as an


def terms(text):
    return [t.strip() for t in text.split(',') if t.strip()]


class TestBuild(unittest.TestCase):
    def test_photo_prompt_gets_anti_illustration_terms(self):
        neg = terms(an.build('raw photo of an old fisherman on a boat at dawn, 35mm'))
        for t in ('illustration', 'cartoon', '3d render', 'deformed face', 'duplicate', 'watermark'):
            self.assertIn(t, neg)
        self.assertNotIn('photo', neg)

    def test_anime_prompt_gets_anti_photo_terms(self):
        neg = terms(an.build('anime illustration of a girl reading in a library'))
        self.assertIn('photorealistic', neg)
        self.assertNotIn('illustration', neg)
        self.assertNotIn('anime', neg)

    def test_never_suppresses_what_the_prompt_asks_for(self):
        cases = {
            'portrait photo, shallow depth of field, bokeh': 'blurry',
            'a neon sign that reads OPEN above a bar': 'text',
            'black and white photo of a street': 'monochrome',
            'a crowd of people at a festival': 'crowd',
            'macro close-up of a dew drop': 'cropped',
            'a dark moody forest at night': 'dark',
            'pixel art castle, 16-bit': 'lowres',
            'a polaroid photo in a frame': 'frame',
            'a spider on its web': 'extra legs',
            'oil painting of a lighthouse': 'painting',
        }
        for prompt, blocked in cases.items():
            with self.subTest(prompt):
                self.assertNotIn(blocked, terms(an.build(prompt)))

    def test_adherence_rules(self):
        self.assertIn('daylight', terms(an.build('a lighthouse in a storm at night')))
        self.assertIn('night', terms(an.build('a sunny beach at noon')))
        self.assertIn('monochrome', terms(an.build('a vibrant colorful market')))
        self.assertIn('color', terms(an.build('black and white portrait of a man')))
        self.assertIn('full body', terms(an.build('close-up portrait of a woman')))
        self.assertIn('cropped', terms(an.build('full body shot of a knight standing')))
        self.assertIn('extra legs', terms(an.build('a red fox sitting in fresh snow')))

    def test_skips_existing_terms_and_caps_length(self):
        neg = an.build('photo of a man', existing_negative='Watermark, cartoon, (blurry:1.2)')
        self.assertNotIn('watermark', terms(neg))
        self.assertNotIn('cartoon', terms(neg))
        self.assertNotIn('blurry', terms(neg))
        long = an.build('raw photo portrait of a man and a cat in a city at night, close-up', max_terms=10)
        self.assertEqual(len(terms(long)), 10)
        self.assertEqual(len(terms(long)), len(set(terms(long))))

    def test_empty_prompt_still_safe(self):
        self.assertIn('watermark', terms(an.build('')))


class TestConflictsAndApplies(unittest.TestCase):
    def test_drop_conflicts_cleans_other_negatives(self):
        quality_negative = 'lowres, blurry, text, illustration, cartoon, deformed'
        cleaned = terms(an.drop_conflicts(quality_negative, 'watercolor illustration of a sign with text, bokeh'))
        self.assertEqual(cleaned, ['lowres', 'cartoon', 'deformed'])

    def test_applies_only_with_guidance(self):
        self.assertTrue(an.applies(4.0))
        self.assertFalse(an.applies(1.0))
        self.assertFalse(an.applies(7.0, family='flux'))
        self.assertFalse(an.applies(None))


if __name__ == '__main__':
    unittest.main()
