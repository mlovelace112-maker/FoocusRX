"""Update pipeline: bring this checkout to origin/main, safely.

    venv/bin/python scripts/foocusrx.py update

Steps (any failure after step 5 rolls everything back):
 1. Preflight: on the release branch, no merge/rebase in progress, no local
    changes to tracked files outside models/ (your own model files are fine).
 2. Lock out the launcher and stop the app if it is running.
 3. Fetch; refuse anything but a fast-forward.
 4. Nothing to do when already current *and* the venv matches the pins.
 5. Back up the venv (APFS clone: instant, exact, no network needed to undo).
 6. Fast-forward the code.
 7. If the pins changed or aren't met: swap in a tested candidate venv when it
    matches (scripts/foocusrx.py upgrade test), otherwise pip install.
 8. Verify: unit tests, then a smoke launch on a spare port that must answer
    and report the pinned Gradio version (unit tests alone passed while
    Gradio 5 broke the UI), and optionally the browser tests.
 9. On failure: `git reset --keep` to the old commit and put the old venv
    back. Never `reset --hard`, never stash.
"""
import os
import re
import shutil
import subprocess
import sys
import time

from modules.app_service import Service, ServiceError

PROTECTED_OK_PREFIXES = ('models/',)  # local changes here are the user's data
SMOKE_PORT = 7871


class UpdateError(RuntimeError):
    pass


def git(root, *args, check=True, capture=True):
    result = subprocess.run(['git', *args], cwd=root, capture_output=capture, text=True)
    if check and result.returncode != 0:
        raise UpdateError(f'git {" ".join(args)} failed:\n{(result.stderr or result.stdout).strip()}')
    return result.stdout.strip() if capture else ''


def pinned_version(reqs_file: str, package: str) -> str | None:
    pattern = re.compile(rf'^\s*{re.escape(package)}\s*==\s*([^\s;#]+)', re.I)
    try:
        with open(reqs_file, encoding='utf-8') as fh:
            for line in fh:
                match = pattern.match(line)
                if match:
                    return match.group(1)
    except OSError:
        pass
    return None


def blocking_changes(root) -> list:
    """Tracked files with local changes, excluding the user's model folders."""
    out = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, capture_output=True,
                         text=True, check=True).stdout  # not stripped: column 1 may be a space
    paths = [line[3:].split(' -> ')[-1].strip('"') for line in out.splitlines() if len(line) > 3]
    return [p for p in paths if not p.startswith(PROTECTED_OK_PREFIXES)]


def clone_dir(src, dst):
    """APFS copy-on-write clone (instant); plain copy where unsupported."""
    result = subprocess.run(['cp', '-cR', src, dst], capture_output=True, text=True)
    if result.returncode != 0:
        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True)


def prune_backups(root, keep: int = 1, log=print) -> list:
    backups = sorted(d for d in os.listdir(root) if d.startswith('venv.bak-'))
    removed = []
    for name in backups[:-keep] if keep else backups:
        shutil.rmtree(os.path.join(root, name), ignore_errors=True)
        removed.append(name)
        log(f'Removed old backup {name}')
    return removed


def fix_script_paths(venv, old_venv):
    """After renaming a venv, point its console scripts and activate files at the new path."""
    old_b, new_b = os.fsencode(os.path.abspath(old_venv)), os.fsencode(os.path.abspath(venv))
    bindir = os.path.join(venv, 'bin')
    for name in os.listdir(bindir):
        path = os.path.join(bindir, name)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        try:
            with open(path, 'rb') as fh:
                data = fh.read(1 << 20)
            if old_b in data and b'\0' not in data:
                with open(path, 'wb') as fh:
                    fh.write(data.replace(old_b, new_b))
        except OSError:
            pass


def candidate_marker(venv) -> str:
    return os.path.join(venv, '.foocusrx-candidate.json')


def run_unit_tests(python, root, log=print) -> bool:
    log('Running unit tests ...')
    result = subprocess.run([python, '-m', 'unittest', 'discover', 'tests'], cwd=root, capture_output=True,
                            text=True, env={k: v for k, v in os.environ.items() if not k.startswith('FOOOCUS_E2E')})
    summary = [line for line in result.stderr.splitlines() if line.startswith(('Ran ', 'OK', 'FAILED'))]
    log('  ' + ' | '.join(summary))
    if result.returncode != 0:
        log(result.stderr[-3000:])
    return result.returncode == 0


def smoke_test(root, venv, reqs_file, port=SMOKE_PORT, e2e=False, log=print, log_dir=None) -> bool:
    """Start the app on a spare port: it must answer and run the pinned Gradio."""
    service = Service(root, venv=venv, port=port, reqs_file=reqs_file, name='smoke', log_dir=log_dir)
    expected = pinned_version(reqs_file, 'gradio')
    log(f'Smoke test: starting the app on port {port} ...')
    try:
        service.start(wait=300, env_extra={'FOOOCUS_IGNORE_UPDATE_LOCK': '1'})
        running = service.gradio_version()
        log(f'  answered; Gradio {running} (pinned {expected})')
        if expected and running != expected:
            log('  FAILED: the app is not running the pinned Gradio version')
            return False
        if e2e:
            log('  running browser tests ...')
            env = dict(os.environ, FOOOCUS_E2E_URL=service.url)
            result = subprocess.run([service.python, '-m', 'unittest', 'tests.e2e.test_ui'], cwd=root,
                                    capture_output=True, text=True, env=env)
            log('  ' + ' | '.join(l for l in result.stderr.splitlines() if l.startswith(('Ran ', 'OK', 'FAILED'))))
            if result.returncode != 0:
                log(result.stderr[-3000:])
                return False
        return True
    except ServiceError as exc:
        log(f'  FAILED: {exc}')
        return False
    finally:
        service.stop()


def update(root, venv=None, remote='origin', branch='main', check_only=False, smoke=True, e2e=False,
           stop_app=True, log=print) -> dict:
    root = os.path.abspath(root)
    venv = os.path.abspath(venv or os.path.join(root, 'venv'))
    app = Service(root, venv=venv)
    reqs = os.path.join(root, 'requirements_versions.txt')

    # 1. Preflight
    current_branch = git(root, 'rev-parse', '--abbrev-ref', 'HEAD')
    if current_branch != branch:
        raise UpdateError(f'This checkout is on "{current_branch}", not "{branch}". Switch to {branch} first.')
    gitdir = git(root, 'rev-parse', '--git-dir')
    gitdir = gitdir if os.path.isabs(gitdir) else os.path.join(root, gitdir)
    for marker in ('MERGE_HEAD', 'rebase-merge', 'rebase-apply', 'CHERRY_PICK_HEAD'):
        if os.path.exists(os.path.join(gitdir, marker)):
            raise UpdateError('A merge/rebase is in progress in this checkout; finish or abort it first.')
    blocked = blocking_changes(root)
    if blocked:
        raise UpdateError('Local changes to tracked files would be affected; commit or revert them first:\n  '
                          + '\n  '.join(blocked[:15]))
    if not os.path.exists(os.path.join(venv, 'bin', 'python')):
        raise UpdateError(f'No virtualenv at {venv}. Run scripts/install_mac.command first.')

    # 3. What's new?
    log(f'Fetching {remote}/{branch} ...')
    git(root, 'fetch', remote, branch)
    old = git(root, 'rev-parse', 'HEAD')
    new = git(root, 'rev-parse', f'{remote}/{branch}')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', old, new], cwd=root).returncode != 0:
        raise UpdateError(f'Your {branch} has commits that are not on {remote}/{branch}; '
                          'resolve that manually (this tool only fast-forwards).')
    incoming = git(root, 'log', '--oneline', '--no-decorate', f'{old}..{new}')
    pins_changed = bool(git(root, 'diff', '--name-only', old, new, '--', 'requirements_versions.txt'))
    reqs_met_now = app.requirements_ok()
    result = {'old': old, 'new': new, 'incoming': incoming.splitlines(), 'updated': False}
    if incoming:
        log(f'{len(incoming.splitlines())} new commit(s):\n  ' + incoming.replace('\n', '\n  '))
    else:
        log(f'Code is already at {remote}/{branch} ({old[:8]}).')
    if not incoming and reqs_met_now:
        log('Packages match requirements_versions.txt. Nothing to do.')
        return result
    if not incoming:
        log('Installed packages do not match requirements_versions.txt; they will be installed.')
    if check_only:
        result['would_install'] = pins_changed or not reqs_met_now
        return result

    # 2. Lock and stop the app
    os.makedirs(app.state_dir, exist_ok=True)
    if os.path.exists(app.update_lock):
        raise UpdateError(f'Another update seems to be running (remove {app.update_lock} if it is not).')
    with open(app.update_lock, 'w') as fh:
        fh.write(str(os.getpid()))
    backup = None
    moved = False
    try:
        status = app.status()
        if status['running']:
            if not stop_app:
                raise UpdateError('FoocusRX is running; stop it first (or pass --stop-app).')
            log('Stopping FoocusRX ...')
            app.stop()

        # 5. Back up the venv
        backup = os.path.join(root, f'venv.bak-{time.strftime("%Y%m%d-%H%M%S")}')
        log(f'Backing up the virtualenv to {os.path.basename(backup)} ...')
        clone_dir(venv, backup)

        # 6. Fast-forward
        if incoming:
            log('Updating the code ...')
            git(root, 'merge', '--ff-only', new)
            moved = True

        # 7. Packages
        if not app.requirements_ok():
            candidate = os.path.join(root, 'venv-candidate')
            cand = Service(root, venv=candidate)
            if os.path.exists(candidate_marker(candidate)) and cand.requirements_ok():
                log('Using the tested candidate virtualenv (venv-candidate) ...')
                shutil.rmtree(venv)
                os.rename(candidate, venv)
                fix_script_paths(venv, candidate)
            else:
                log('Installing packages from requirements_versions.txt (this can take a few minutes) ...')
                pip = subprocess.run([os.path.join(venv, 'bin', 'python'), '-m', 'pip', 'install', '-r', reqs],
                                     cwd=root, capture_output=True, text=True)
                if pip.returncode != 0:
                    raise UpdateError('pip install failed:\n' + (pip.stderr or pip.stdout)[-3000:])
            if not app.requirements_ok():
                raise UpdateError('Packages still do not match requirements_versions.txt after installing.')

        # 8. Verify
        python = os.path.join(venv, 'bin', 'python')
        if not run_unit_tests(python, root, log=log):
            raise UpdateError('Unit tests failed.')
        if smoke and not smoke_test(root, venv, reqs, e2e=e2e, log=log):
            raise UpdateError('The app did not start correctly.')
    except BaseException as exc:
        log(f'\nUpdate failed: {exc}\nRolling back ...')
        if moved:
            git(root, 'reset', '--keep', old, check=False)
            log(f'  code restored to {old[:8]}')
        if backup and os.path.exists(backup):
            failed = venv + f'.failed-{time.strftime("%Y%m%d-%H%M%S")}'
            if os.path.exists(venv):
                os.rename(venv, failed)
            os.rename(backup, venv)
            shutil.rmtree(failed, ignore_errors=True)
            log('  virtualenv restored')
        raise
    finally:
        try:
            os.remove(app.update_lock)
        except OSError:
            pass

    prune_backups(root, keep=1, log=log)
    result['updated'] = True
    result['backup'] = os.path.basename(backup)
    log(f'\nUpdated to {new[:8]}. Previous virtualenv kept as {os.path.basename(backup)} '
        '(remove it with: scripts/foocusrx.py prune-backups --keep 0).')
    return result
