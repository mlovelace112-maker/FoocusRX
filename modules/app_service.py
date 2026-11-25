"""Run FoocusRX as a background service: start, stop, status, logs.

Used by the Dock app (scripts/foocusrx.py gui), the update pipeline (stop
before updating, smoke-test after) and the upgrade pipeline (smoke-test a
candidate venv). Standard library only, so it works from any interpreter
and never imports torch or gradio itself.

Safety: only a process whose command line is this checkout's launch.py is
ever stopped; a port held by anything else is reported, never killed.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

DEFAULT_PORT = 7865


class ServiceError(RuntimeError):
    pass


class NeedsUpdate(ServiceError):
    """Installed packages don't match requirements_versions.txt."""


class NeedsInstall(ServiceError):
    """No usable virtualenv."""


def default_log_dir() -> str:
    override = os.environ.get('FOOOCUS_LOG_DIR')
    if override:
        return override
    if sys.platform == 'darwin':
        return os.path.expanduser('~/Library/Logs/FoocusRX')
    return os.path.join(os.path.expanduser('~'), '.foocusrx', 'logs')


def clean_env(extra=None) -> dict:
    """Environment for the app: no inherited proxies (Gradio's localhost
    health probe must not go through a VPN/proxy), loopback bypass set."""
    env = dict(os.environ)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1,::1'
    env['PYTHONUNBUFFERED'] = '1'
    env.update(extra or {})
    return env


def http_get(url: str, timeout: float = 2.0):
    """(status, body bytes) or (None, b'') if unreachable. Never uses a proxy."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b''
    except Exception:
        return None, b''


class Service:
    def __init__(self, root: str, venv: str | None = None, port: int = DEFAULT_PORT, state_dir: str | None = None,
                 log_dir: str | None = None, reqs_file: str | None = None, name: str = 'foocusrx'):
        self.root = os.path.abspath(root)
        self.venv = os.path.abspath(venv or os.path.join(self.root, 'venv'))
        self.port = int(port)
        self.state_dir = state_dir or os.path.join(self.root, '.foocusrx')
        self.log_dir = log_dir or default_log_dir()
        self.reqs_file = reqs_file or os.path.join(self.root, 'requirements_versions.txt')
        self.name = name

    # --- paths ---------------------------------------------------------------
    @property
    def python(self) -> str:
        return os.path.join(self.venv, 'bin', 'python')

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.port}/'

    @property
    def pidfile(self) -> str:
        return os.path.join(self.state_dir, f'{self.name}-{self.port}.pid')

    @property
    def update_lock(self) -> str:
        return os.path.join(self.state_dir, 'update.lock')

    @property
    def logfile(self) -> str:
        return os.path.join(self.log_dir, f'{self.name}.log' if self.port == DEFAULT_PORT
                            else f'{self.name}-{self.port}.log')

    # --- environment checks ----------------------------------------------------
    def venv_ready(self) -> bool:
        if not os.path.exists(self.python):
            return False
        result = subprocess.run([self.python, '-c', 'import gradio, numpy, PIL'], cwd=self.root,
                                capture_output=True, timeout=120)
        return result.returncode == 0

    def requirements_ok(self, reqs_file: str | None = None) -> bool:
        """Same check launch.py makes, run inside this venv."""
        code = ('import sys; sys.path.insert(0, sys.argv[1]); from modules.launch_util import requirements_met; '
                'sys.exit(0 if requirements_met(sys.argv[2]) else 1)')
        result = subprocess.run([self.python, '-c', code, self.root, reqs_file or self.reqs_file],
                                cwd=self.root, capture_output=True, timeout=120)
        return result.returncode == 0

    def installed_version(self, package: str) -> str | None:
        code = 'import importlib.metadata as m, sys; print(m.version(sys.argv[1]))'
        result = subprocess.run([self.python, '-c', code, package], capture_output=True, text=True, timeout=60)
        return result.stdout.strip() if result.returncode == 0 else None

    # --- process ---------------------------------------------------------------
    def read_pid(self) -> int | None:
        try:
            with open(self.pidfile) as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def pid_alive(pid) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def pid_is_ours(self, pid) -> bool:
        """True only if ``pid`` runs this checkout's launch.py."""
        if not self.pid_alive(pid):
            return False
        try:
            result = subprocess.run(['ps', '-o', 'command=', '-p', str(pid)], capture_output=True, text=True)
        except OSError:
            return False  # can't inspect it: never treat it as ours
        command = result.stdout.strip()
        if 'launch.py' not in command:
            return False
        try:
            result = subprocess.run(['lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn'], capture_output=True,
                                    text=True)
        except OSError:
            return self.root in command
        cwd = next((line[1:] for line in result.stdout.splitlines() if line.startswith('n')), '')
        return os.path.realpath(cwd) == os.path.realpath(self.root) if cwd else self.root in command

    def port_in_use(self) -> bool:
        """Anything at all listening on the port (not just FoocusRX)."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            return sock.connect_ex(('127.0.0.1', self.port)) == 0

    def responding(self) -> bool:
        status, _ = http_get(self.url + 'config', timeout=2)
        return status == 200

    def gradio_version(self) -> str | None:
        status, body = http_get(self.url + 'config', timeout=5)
        if status != 200:
            return None
        try:
            return json.loads(body).get('version')
        except ValueError:
            return None

    def status(self) -> dict:
        pid = self.read_pid()
        ours = self.pid_is_ours(pid)
        responding = self.responding()
        return {'running': ours, 'pid': pid if ours else None, 'responding': responding,
                'port_taken_by_other': not ours and (responding or self.port_in_use()), 'url': self.url,
                'updating': os.path.exists(self.update_lock), 'log': self.logfile}

    def log_tail(self, lines: int = 25) -> str:
        try:
            with open(self.logfile, encoding='utf-8', errors='replace') as fh:
                return ''.join(fh.readlines()[-lines:])
        except OSError:
            return ''

    def start(self, wait: float = 300, extra_args=(), check_requirements: bool = True, env_extra=None,
              on_wait=None) -> dict:
        """Start in the background and wait until it answers. Raises
        NeedsInstall / NeedsUpdate / ServiceError (with the log tail)."""
        if os.path.exists(self.update_lock) and not (env_extra or {}).get('FOOOCUS_IGNORE_UPDATE_LOCK'):
            raise ServiceError('An update is in progress. Try again when it has finished.')
        current = self.status()
        if current['running'] and current['responding']:
            return {'started': False, **current}
        if current['port_taken_by_other']:
            raise ServiceError(f'Port {self.port} is already used by another program; FoocusRX was not started.')
        if not os.path.exists(self.python):
            raise NeedsInstall(f'No virtualenv at {self.venv}.')
        if check_requirements and not self.requirements_ok():
            raise NeedsUpdate('Installed packages do not match requirements_versions.txt; run the update first.')

        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        log = open(self.logfile, 'a', encoding='utf-8')
        log.write(f'\n===== {time.strftime("%Y-%m-%d %H:%M:%S")} starting FoocusRX on port {self.port} =====\n')
        log.flush()
        extra = dict(env_extra or {})
        if os.path.abspath(self.reqs_file) != os.path.join(self.root, 'requirements_versions.txt'):
            extra['REQS_FILE'] = self.reqs_file
        cmd = [self.python, 'launch.py', '--listen', '127.0.0.1', '--port', str(self.port), '--disable-in-browser',
               *extra_args]
        proc = subprocess.Popen(cmd, cwd=self.root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=clean_env(extra), start_new_session=True)
        log.close()
        with open(self.pidfile, 'w') as fh:
            fh.write(str(proc.pid))

        deadline = time.time() + wait
        while time.time() < deadline:
            if proc.poll() is not None:
                self._forget_pid()
                raise ServiceError(f'FoocusRX exited during startup (code {proc.returncode}).\n\n{self.log_tail()}')
            if self.responding():
                return {'started': True, **self.status()}
            if on_wait:
                on_wait()
            time.sleep(1.0)
        self.stop()
        raise ServiceError(f'FoocusRX did not answer within {int(wait)} s.\n\n{self.log_tail()}')

    def stop(self, timeout: float = 20) -> bool:
        """Stop our server (SIGTERM, then SIGKILL). False if nothing of ours runs."""
        pid = self.read_pid()
        if not self.pid_is_ours(pid):
            self._forget_pid()
            return False
        for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 5)):
            try:
                os.killpg(pid, sig)  # started in its own session: stops its children too
            except ProcessLookupError:
                break
            except PermissionError:
                os.kill(pid, sig)
            end = time.time() + wait
            while time.time() < end and self.pid_alive(pid):
                try:  # reap it if we started it (a zombie still "exists")
                    if os.waitpid(pid, os.WNOHANG)[0] == pid:
                        break
                except ChildProcessError:
                    pass
                time.sleep(0.2)
            if not self.pid_alive(pid):
                break
        self._forget_pid()
        return True

    def _forget_pid(self):
        try:
            os.remove(self.pidfile)
        except OSError:
            pass
