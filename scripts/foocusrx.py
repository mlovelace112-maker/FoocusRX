#!/usr/bin/env python3
"""FoocusRX control: launcher, update and dependency-upgrade pipelines.

    scripts/foocusrx.py gui                     what FoocusRX.app runs (dialogs)
    scripts/foocusrx.py start|stop|status|open|log
    scripts/foocusrx.py update [--check] [--e2e] [--no-smoke]
    scripts/foocusrx.py upgrade check
    scripts/foocusrx.py upgrade test PKG[==VER] ... [--e2e]
    scripts/foocusrx.py upgrade apply [--dry-run]
    scripts/foocusrx.py prune-backups [--keep N]
    scripts/foocusrx.py build-app [--root DIR] [--out DIR] [--install]

Run it with the project's venv python (the .app and .command files do).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.app_service import DEFAULT_PORT, NeedsInstall, NeedsUpdate, Service, ServiceError  # noqa: E402
from modules import updater, upgrader  # noqa: E402

APP_NAME = 'FoocusRX'


# --- macOS dialogs (only used by `gui`) ----------------------------------------

def _osa(script: str) -> str:
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ''


def _q(text: str) -> str:
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def dialog(message: str, buttons=('OK',), default=None, icon='note') -> str:
    items = ', '.join(_q(b) for b in buttons)
    out = _osa(f'display dialog {_q(message)} with title {_q(APP_NAME)} buttons {{{items}}} '
               f'default button {_q(default or buttons[-1])} with icon {icon}')
    return out.split('button returned:')[-1].strip() if 'button returned:' in out else ''


def choose(prompt: str, options) -> str:
    items = ', '.join(_q(o) for o in options)
    out = _osa(f'choose from list {{{items}}} with title {_q(APP_NAME)} with prompt {_q(prompt)} '
               f'default items {{{_q(options[0])}}}')
    return '' if out in ('', 'false') else out


def notify(message: str):
    _osa(f'display notification {_q(message)} with title {_q(APP_NAME)}')


def open_path(target: str):
    subprocess.run(['open', target])


def open_in_terminal(command_file: str):
    subprocess.run(['open', '-a', 'Terminal', command_file])


def gui(service: Service) -> int:
    status = service.status()
    update_cmd = os.path.join(ROOT, 'Update FoocusRX.command')
    if status['updating']:
        dialog('FoocusRX is being updated. Try again when the update has finished.')
        return 0
    if status['running']:
        pick = choose('FoocusRX is running.', ['Open FoocusRX', 'Stop FoocusRX', 'Update FoocusRX', 'View log'])
        if pick == 'Open FoocusRX':
            open_path(service.url)
        elif pick == 'Stop FoocusRX':
            service.stop()
            notify('FoocusRX stopped.')
        elif pick == 'Update FoocusRX':
            service.stop()
            open_in_terminal(update_cmd)
        elif pick == 'View log':
            open_path(service.logfile)
        return 0
    if status['port_taken_by_other']:
        dialog(f'Another program is using port {service.port}, so FoocusRX cannot start.', icon='caution')
        return 1
    notify('Starting FoocusRX ...')
    try:
        service.start()
    except NeedsInstall:
        if dialog('FoocusRX is not installed yet (no virtualenv). Run the installer now?',
                  buttons=('Cancel', 'Install')) == 'Install':
            open_in_terminal(os.path.join(ROOT, 'scripts', 'install_mac.command'))
        return 1
    except NeedsUpdate:
        if dialog('FoocusRX needs to install updated packages before it can start. Run the update now?',
                  buttons=('Cancel', 'Update')) == 'Update':
            open_in_terminal(update_cmd)
        return 1
    except ServiceError as exc:
        tail = str(exc)[-1200:]
        if dialog(f'FoocusRX could not start.\n\n{tail}', buttons=('View log', 'OK'), icon='stop') == 'View log':
            open_path(service.logfile)
        return 1
    open_path(service.url)
    return 0


# --- .app bundle ---------------------------------------------------------------

INFO_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>FoocusRX</string>
  <key>CFBundleDisplayName</key><string>FoocusRX</string>
  <key>CFBundleIdentifier</key><string>local.foocusrx.launcher</string>
  <key>CFBundleVersion</key><string>{version}</string>
  <key>CFBundleShortVersionString</key><string>{version}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>FoocusRX</string>
  <key>CFBundleIconFile</key><string>FoocusRX</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>LSUIElement</key><false/>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
"""

LAUNCHER_SH = """#!/bin/bash
# FoocusRX.app launcher (generated by scripts/foocusrx.py build-app).
ROOT={root}
PY="$ROOT/venv/bin/python"
if [ ! -x "$PY" ]; then
  /usr/bin/osascript -e 'display dialog "FoocusRX is not installed yet. The installer will open in Terminal." with title "FoocusRX" buttons {{"OK"}} default button "OK"'
  /usr/bin/open -a Terminal "$ROOT/scripts/install_mac.command"
  exit 0
fi
exec "$PY" "$ROOT/scripts/foocusrx.py" gui
"""


def make_icon(resources: str) -> bool:
    """FoocusRX.icns: the header's "RX" mark on the violet->indigo gradient."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    import shutil
    import tempfile
    size = 1024
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    grad = Image.new('RGBA', (size, size))
    top, bottom = (139, 92, 246), (79, 70, 229)  # violet-500 -> indigo-600
    px = grad.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * size)
            px[x, y] = tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,)
    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((80, 80, size - 80, size - 80), radius=200, fill=255)
    img.paste(grad, (0, 0), mask)
    draw = ImageDraw.Draw(img)
    font = None
    for path in ('/System/Library/Fonts/SFNSRounded.ttf', '/System/Library/Fonts/SFNS.ttf',
                 '/System/Library/Fonts/Helvetica.ttc'):
        try:
            font = ImageFont.truetype(path, 420)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    box = draw.textbbox((0, 0), 'RX', font=font)
    draw.text(((size - (box[2] - box[0])) / 2 - box[0], (size - (box[3] - box[1])) / 2 - box[1]), 'RX',
              fill='white', font=font)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, 'FoocusRX.iconset')
        os.makedirs(iconset)
        for s in (16, 32, 64, 128, 256, 512):
            img.resize((s, s), Image.LANCZOS).save(os.path.join(iconset, f'icon_{s}x{s}.png'))
            img.resize((s * 2, s * 2), Image.LANCZOS).save(os.path.join(iconset, f'icon_{s}x{s}@2x.png'))
        out = os.path.join(resources, 'FoocusRX.icns')
        result = subprocess.run(['iconutil', '-c', 'icns', iconset, '-o', out], capture_output=True)
        if result.returncode != 0:
            img.resize((512, 512), Image.LANCZOS).save(os.path.join(resources, 'FoocusRX.png'))
            return False
    return True


def build_app(root: str, out_dir: str, install: bool = False) -> str:
    import shlex
    import shutil
    root = os.path.abspath(root)
    if not os.path.exists(os.path.join(root, 'fooocus_version.py')):
        raise SystemExit(f'{root} is not a FoocusRX checkout')
    version = '0'
    try:
        with open(os.path.join(root, 'fooocus_version.py')) as fh:
            version = fh.read().split("'")[1]
    except (OSError, IndexError):
        pass
    app = os.path.join(os.path.abspath(out_dir), 'FoocusRX.app')
    if os.path.exists(app):
        shutil.rmtree(app)
    macos, resources = os.path.join(app, 'Contents', 'MacOS'), os.path.join(app, 'Contents', 'Resources')
    os.makedirs(macos)
    os.makedirs(resources)
    with open(os.path.join(app, 'Contents', 'Info.plist'), 'w') as fh:
        fh.write(INFO_PLIST.format(version=version))
    exe = os.path.join(macos, 'FoocusRX')
    with open(exe, 'w') as fh:
        fh.write(LAUNCHER_SH.format(root=shlex.quote(root)))
    os.chmod(exe, 0o755)
    icon = make_icon(resources)
    print(f'Built {app}{"" if icon else " (no icon: iconutil/Pillow unavailable)"} for {root}')
    if install:
        dest = os.path.expanduser('~/Applications/FoocusRX.app')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(app, dest, symlinks=True)
        print(f'Installed {dest} (drag it to the Dock to keep it there)')
        return dest
    return app


# --- CLI -----------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='foocusrx', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=ROOT)
    p.add_argument('--venv', default=None)
    p.add_argument('--port', type=int, default=int(os.environ.get('FOOOCUS_PORT', DEFAULT_PORT)))
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('gui', 'start', 'stop', 'status', 'open', 'log'):
        sub.add_parser(name)
    u = sub.add_parser('update')
    u.add_argument('--check', action='store_true', help='only report what would change')
    u.add_argument('--no-smoke', action='store_true', help='skip the smoke launch')
    u.add_argument('--e2e', action='store_true', help='also run the browser tests (needs Chrome)')
    u.add_argument('--branch', default='main')
    g = sub.add_parser('upgrade')
    gsub = g.add_subparsers(dest='action', required=True)
    gsub.add_parser('check')
    t = gsub.add_parser('test')
    t.add_argument('packages', nargs='+')
    t.add_argument('--e2e', action='store_true')
    a = gsub.add_parser('apply')
    a.add_argument('--dry-run', action='store_true')
    pb = sub.add_parser('prune-backups')
    pb.add_argument('--keep', type=int, default=1)
    b = sub.add_parser('build-app')
    b.add_argument('--out', default=None, help='output folder (default: <root>/dist)')
    b.add_argument('--install', action='store_true', help='also copy to ~/Applications')
    args = p.parse_args(argv)

    service = Service(args.root, venv=args.venv, port=args.port)
    try:
        if args.cmd == 'gui':
            return gui(service)
        if args.cmd == 'start':
            info = service.start()
            print(('Started' if info['started'] else 'Already running') + f": {info['url']} (pid {info['pid']})")
        elif args.cmd == 'stop':
            print('Stopped.' if service.stop() else 'FoocusRX was not running.')
        elif args.cmd == 'status':
            st = service.status()
            state = 'running' if st['running'] else ('port used by another program' if st['port_taken_by_other']
                                                     else 'stopped')
            print(f"{state}{' (update in progress)' if st['updating'] else ''} - {st['url']}  log: {st['log']}")
            return 0 if st['running'] else 3
        elif args.cmd == 'open':
            open_path(service.url)
        elif args.cmd == 'log':
            print(service.log_tail(80))
        elif args.cmd == 'update':
            updater.update(args.root, venv=args.venv, branch=args.branch, check_only=args.check,
                           smoke=not args.no_smoke, e2e=args.e2e)
        elif args.cmd == 'upgrade':
            if args.action == 'check':
                upgrader.check(args.root, venv=args.venv)
            elif args.action == 'test':
                return 0 if upgrader.test(args.root, args.packages, venv=args.venv, e2e=args.e2e)['passed'] else 1
            else:
                upgrader.apply(args.root, dry_run=args.dry_run)
        elif args.cmd == 'prune-backups':
            updater.prune_backups(args.root, keep=args.keep)
        elif args.cmd == 'build-app':
            build_app(args.root, args.out or os.path.join(args.root, 'dist'), install=args.install)
    except NeedsUpdate as exc:
        print(f'{exc}\nRun: scripts/foocusrx.py update', file=sys.stderr)
        return 2
    except (ServiceError, updater.UpdateError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
