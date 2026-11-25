"""Safety when the app is reachable from other machines without a login.

With --listen on a non-loopback address (or --share) and no auth.json, anyone
who can reach the page could delete images (History) or make the server
download files (Model manager). In that case those actions are switched off,
in the UI and in their handlers, and a warning is printed at startup.
FOOOCUS_ALLOW_REMOTE_FILE_ACTIONS=1 turns them back on.
"""
import ipaddress
import os

LOOPBACK_NAMES = {'localhost', 'localhost.localdomain', ''}
LOCKED_MESSAGE = ('Disabled: the app is reachable from other machines without a login. Copy auth-example.json '
                  'to auth.json, set your own user and password, and restart (or set '
                  'FOOOCUS_ALLOW_REMOTE_FILE_ACTIONS=1).')


def is_loopback(host) -> bool:
    host = str(host or '').strip().strip('[]').lower()
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def exposed(listen, share=False) -> bool:
    """True when the server accepts connections from other machines."""
    return bool(share) or not is_loopback(listen)


def file_actions_locked(listen, share, auth_enabled, env=None) -> bool:
    env = os.environ if env is None else env
    if env.get('FOOOCUS_FILE_ACTIONS_LOCKED') == '1':  # force the locked state (tests, kiosks)
        return True
    if env.get('FOOOCUS_ALLOW_REMOTE_FILE_ACTIONS') == '1':
        return False
    return exposed(listen, share) and not auth_enabled


def startup_warning(listen, share, auth_enabled) -> str | None:
    if not exposed(listen, share) or auth_enabled:
        return None
    where = 'a public Gradio share link' if share else f'--listen {listen}'
    return ('\n' + '!' * 78 + f'\n[Security] FoocusRX is reachable from other machines ({where}) without a login.\n'
            '[Security] Anyone who can open the page can generate images and read your outputs.\n'
            '[Security] History delete and model downloads are disabled until you add a login:\n'
            '[Security] copy auth-example.json to auth.json and set your own user and password\n'
            '[Security] (or set FOOOCUS_ALLOW_REMOTE_FILE_ACTIONS=1).\n' + '!' * 78)
