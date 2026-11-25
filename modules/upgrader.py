"""Dependency upgrade pipeline: test upgrades in a venv copy, then ship them.

    scripts/foocusrx.py upgrade check                 # what's outdated (pinned packages)
    scripts/foocusrx.py upgrade test gradio==5.50.0   # try it in venv-candidate
    scripts/foocusrx.py upgrade apply [--dry-run]     # PR with the new pins
    scripts/foocusrx.py update                        # after merging: swaps venv-candidate in

`test` clones the venv (APFS, instant), installs the requested versions into
the clone, writes a candidate pins file, and runs the unit tests plus a smoke
launch against the clone with REQS_FILE pointing at the candidate pins (so
launch.py can't quietly reinstall the old versions) - the app must report
the candidate Gradio version. The real venv is never touched.

`apply` opens a pull request with the new pins from a temporary worktree
(never your checkout). After it is merged, `update` sees that the tested
candidate matches the new pins and swaps it in instead of reinstalling.

torch / torchvision are not pinned in requirements_versions.txt (launch.py
installs them separately) and are out of scope here.
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile

from modules.updater import UpdateError, candidate_marker, clone_dir, git, run_unit_tests, smoke_test

PIN_RE = re.compile(r'^(\s*)([A-Za-z0-9_.\-]+)(\s*==\s*)([^\s;#]+)(.*)$')


def parse_pins(reqs_text: str) -> dict:
    pins = {}
    for line in reqs_text.splitlines():
        match = PIN_RE.match(line)
        if match:
            pins[match.group(2).lower()] = match.group(4)
    return pins


def replace_pins(reqs_text: str, new_pins: dict) -> str:
    """Rewrite ``name==version`` lines for the packages in ``new_pins``
    (case-insensitive names); other lines are kept byte for byte."""
    wanted = {k.lower(): v for k, v in new_pins.items()}
    out = []
    for line in reqs_text.splitlines(keepends=True):
        body = line.rstrip('\r\n')
        ending = line[len(body):]
        match = PIN_RE.match(body)
        if match and match.group(2).lower() in wanted:
            body = f'{match.group(1)}{match.group(2)}{match.group(3)}{wanted[match.group(2).lower()]}{match.group(5)}'
        out.append(body + ending)
    return ''.join(out)


def major(version: str) -> str:
    return version.split('.')[0]


def parse_request(spec: str):
    """'gradio==5.50.0' -> ('gradio', '5.50.0'); 'gradio' -> ('gradio', None)."""
    name, _, version = spec.partition('==')
    if not re.fullmatch(r'[A-Za-z0-9_.\-]+', name.strip()):
        raise UpdateError(f'Not a package name: {spec!r}')
    return name.strip(), (version.strip() or None)


def check(root, venv=None, log=print) -> list:
    """Pinned packages with newer releases: [(name, pinned, latest, is_major)]."""
    venv = venv or os.path.join(root, 'venv')
    with open(os.path.join(root, 'requirements_versions.txt'), encoding='utf-8') as fh:
        pins = parse_pins(fh.read())
    log('Checking PyPI for newer versions of the pinned packages ...')
    result = subprocess.run([os.path.join(venv, 'bin', 'python'), '-m', 'pip', 'list', '--outdated',
                             '--format=json'], capture_output=True, text=True)
    if result.returncode != 0:
        raise UpdateError('pip list --outdated failed:\n' + result.stderr[-2000:])
    latest = {p['name'].lower(): p['latest_version'] for p in json.loads(result.stdout or '[]')}
    rows = []
    for name, pinned in sorted(pins.items()):
        new = latest.get(name) or latest.get(name.replace('_', '-'))
        if new and new != pinned:
            rows.append((name, pinned, new, major(new) != major(pinned)))
    if not rows:
        log('All pinned packages are current.')
    for name, pinned, new, is_major in rows:
        log(f'  {name:28} {pinned:>12} -> {new:<12}{"  MAJOR - review release notes" if is_major else ""}')
    if rows:
        log('\nTry one with:  scripts/foocusrx.py upgrade test <package>==<version>')
    return rows


def test(root, requests, venv=None, e2e=False, smoke=True, log=print) -> dict:
    """Install ``requests`` into venv-candidate (a clone of venv) and verify it."""
    root = os.path.abspath(root)
    venv = os.path.abspath(venv or os.path.join(root, 'venv'))
    candidate = os.path.join(root, 'venv-candidate')
    reqs = os.path.join(root, 'requirements_versions.txt')
    wanted = [parse_request(r) for r in requests]
    if not wanted:
        raise UpdateError('Name at least one package, e.g. gradio==5.50.0')

    if os.path.exists(candidate):
        log('Removing the previous venv-candidate ...')
        shutil.rmtree(candidate)
    log('Cloning the virtualenv to venv-candidate ...')
    clone_dir(venv, candidate)
    python = os.path.join(candidate, 'bin', 'python')
    specs = [f'{n}=={v}' if v else n for n, v in wanted]
    log(f'Installing {" ".join(specs)} into venv-candidate ...')
    pip = subprocess.run([python, '-m', 'pip', 'install', '--upgrade', *specs], cwd=root, capture_output=True,
                         text=True)
    if pip.returncode != 0:
        raise UpdateError('pip install failed:\n' + (pip.stderr or pip.stdout)[-3000:])

    # The versions actually installed become the candidate pins.
    code = ('import importlib.metadata as m, json, sys; '
            'print(json.dumps({n: m.version(n) for n in sys.argv[1:]}))')
    versions = json.loads(subprocess.run([python, '-c', code, *[n for n, _ in wanted]], capture_output=True,
                                         text=True, check=True).stdout)
    with open(reqs, encoding='utf-8') as fh:
        current = fh.read()
    pinned_now = parse_pins(current)
    missing = [n for n in versions if n.lower() not in pinned_now]
    if missing:
        log(f'Note: not pinned in requirements_versions.txt (left unpinned): {", ".join(missing)}')
    candidate_reqs = os.path.join(candidate, 'requirements_candidate.txt')
    with open(candidate_reqs, 'w', encoding='utf-8') as fh:
        fh.write(replace_pins(current, versions))

    report = {'packages': versions, 'unit_tests': None, 'smoke': None, 'passed': False}
    report['unit_tests'] = run_unit_tests(python, root, log=log)
    if report['unit_tests'] and smoke:
        report['smoke'] = smoke_test(root, candidate, candidate_reqs, port=7872, e2e=e2e, log=log)
    report['passed'] = bool(report['unit_tests']) and (report['smoke'] is not False)
    report['tested_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    with open(candidate_marker(candidate) if report['passed'] else candidate_marker(candidate) + '.failed',
              'w') as fh:
        json.dump(report, fh, indent=1)
    if report['passed']:
        log('\nCandidate PASSED: ' + ', '.join(f'{k} {v}' for k, v in versions.items())
            + '\nNext: scripts/foocusrx.py upgrade apply   (opens a PR with the new pins)')
    else:
        log('\nCandidate FAILED; your venv is untouched. venv-candidate is kept for inspection.')
    return report


def apply(root, remote='origin', base='main', dry_run=False, log=print) -> dict:
    """Open a PR with the tested candidate's pins (from a temporary worktree)."""
    root = os.path.abspath(root)
    candidate = os.path.join(root, 'venv-candidate')
    try:
        with open(candidate_marker(candidate)) as fh:
            report = json.load(fh)
    except OSError:
        raise UpdateError('No passing candidate. Run: scripts/foocusrx.py upgrade test <package>==<version>')
    versions = report['packages']
    slug = '-'.join(f'{n}-{v}' for n, v in sorted(versions.items()))[:60]
    branch = f'deps/{slug}'
    title = 'deps: ' + ', '.join(f'{n} {v}' for n, v in sorted(versions.items()))

    git(root, 'fetch', remote, base)
    with tempfile.TemporaryDirectory(prefix='foocusrx-deps-') as tmp:
        work = os.path.join(tmp, 'wt')
        git(root, 'worktree', 'add', '--quiet', '-b', branch, work, f'{remote}/{base}')
        try:
            reqs = os.path.join(work, 'requirements_versions.txt')
            with open(reqs, encoding='utf-8') as fh:
                old = fh.read()
            new = replace_pins(old, versions)
            if new == old:
                raise UpdateError(f'{remote}/{base} already has these pins.')
            with open(reqs, 'w', encoding='utf-8') as fh:
                fh.write(new)
            diff = git(work, 'diff', '--', 'requirements_versions.txt')
            log(diff)
            if dry_run:
                log('\n--dry-run: nothing committed or pushed.')
                return {'branch': branch, 'diff': diff, 'pushed': False}
            git(work, 'commit', '-q', '-am', f'{title}\n\nTested in venv-candidate: unit tests and a smoke launch '
                                             f'passed ({report["tested_at"]}).')
            git(work, 'push', '-q', remote, f'HEAD:refs/heads/{branch}')
            body = (f'Upgrade tested with `scripts/foocusrx.py upgrade test` on {report["tested_at"]}:\n'
                    f'- unit tests: {"pass" if report["unit_tests"] else "fail"}\n'
                    f'- smoke launch on the candidate venv: {report["smoke"]}\n\n'
                    'After merging, `scripts/foocusrx.py update` swaps the tested venv-candidate in.')
            pr = subprocess.run(['gh', 'pr', 'create', '--base', base, '--head', branch, '--title', title,
                                 '--body', body], cwd=work, capture_output=True, text=True)
            url = pr.stdout.strip() if pr.returncode == 0 else None
            log(url or f'Pushed {branch}; open the PR on GitHub (gh: {pr.stderr.strip()[:200]})')
            return {'branch': branch, 'diff': diff, 'pushed': True, 'pr': url}
        finally:
            git(root, 'worktree', 'remove', '--force', work, check=False)
            if dry_run:
                git(root, 'branch', '-D', branch, check=False)
