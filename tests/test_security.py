import unittest

from modules import security


class TestSecurity(unittest.TestCase):
    def test_loopback(self):
        for host in ('127.0.0.1', 'localhost', '::1', '[::1]', '127.0.0.2', ''):
            self.assertTrue(security.is_loopback(host), host)
        for host in ('0.0.0.0', '192.168.1.20', '::', 'myhost.local', '10.0.0.5'):
            self.assertFalse(security.is_loopback(host), host)

    def test_locked_only_when_exposed_without_auth(self):
        self.assertFalse(security.file_actions_locked('127.0.0.1', False, False, env={}))
        self.assertTrue(security.file_actions_locked('0.0.0.0', False, False, env={}))
        self.assertTrue(security.file_actions_locked('127.0.0.1', True, False, env={}))  # --share
        self.assertFalse(security.file_actions_locked('0.0.0.0', False, True, env={}))   # auth.json
        self.assertFalse(security.file_actions_locked('0.0.0.0', False, False,
                                                      env={'FOOOCUS_ALLOW_REMOTE_FILE_ACTIONS': '1'}))
        self.assertTrue(security.file_actions_locked('127.0.0.1', False, True,
                                                     env={'FOOOCUS_FILE_ACTIONS_LOCKED': '1'}))

    def test_warning(self):
        self.assertIsNone(security.startup_warning('127.0.0.1', False, False))
        self.assertIsNone(security.startup_warning('0.0.0.0', False, True))
        self.assertIn('--listen 0.0.0.0', security.startup_warning('0.0.0.0', False, False))
        self.assertIn('share link', security.startup_warning('127.0.0.1', True, False))


if __name__ == '__main__':
    unittest.main()
