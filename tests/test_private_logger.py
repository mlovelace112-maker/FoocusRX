import sys

# args_manager parses sys.argv at import time; drop the test runner's own
# arguments (e.g. `unittest discover tests`) so it doesn't exit with code 2.
sys.argv = sys.argv[:1]

import unittest

from modules.private_logger import build_log_item


class TestBuildLogItem(unittest.TestCase):
    def test_escapes_metadata_values_and_labels(self):
        metadata = [
            ('Prompt', 'prompt', '<script>alert(1)</script>'),
            ('<b>Label</b>', 'x', 'ok'),
        ]
        item = build_log_item('a.png', metadata)
        self.assertNotIn('<script>alert(1)</script>', item)
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', item)
        self.assertNotIn('<b>Label</b>', item)

    def test_newlines_still_become_line_breaks(self):
        item = build_log_item('a.png', [('Prompt', 'prompt', 'line1\nline2')])
        self.assertIn('line1 </br> line2', item)

    def test_escapes_full_raw_prompt(self):
        task = {'positive': ['<img src=x onerror=alert(1)>'], 'negative': ['"><svg onload=alert(2)>']}
        item = build_log_item('a.png', [], task)
        self.assertNotIn('<img src=x', item)
        self.assertNotIn('<svg', item)
        self.assertIn('&lt;img src=x onerror=alert(1)&gt;', item)

    def test_clipboard_payload_cannot_break_out_of_onclick(self):
        item = build_log_item('a.png', [('Prompt', 'prompt', "');alert(1);//\"<")])
        onclick = item.split('onclick="', 1)[1].split('"', 1)[0]
        payload = onclick[len("to_clipboard('"):-len("')")]
        self.assertNotIn("'", payload)
        self.assertNotIn('<', payload)

    def test_filename_is_escaped_in_attributes(self):
        item = build_log_item('a"b.png', [])
        self.assertNotIn('a"b.png', item)
        self.assertIn('a&quot;b.png', item)


if __name__ == '__main__':
    unittest.main()
