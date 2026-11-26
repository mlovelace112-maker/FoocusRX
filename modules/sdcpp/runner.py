"""Run one stable-diffusion.cpp generation as a subprocess.

A subprocess (instead of in-process weights) keeps these models' dependencies
(new GGUF architectures, Qwen3-VL / Qwen3.5 text encoders) out of the app's
Python environment, and returns all of their memory to the system when the
image is done, which matters on a 24 GB Mac.
"""
import atexit
import os
import queue
import re
import subprocess
import threading
import time

# Sampling progress line printed by sd-cli (src/core/util.cpp, print_progress_line):
#   "\r  |=====>      |  3/8 - 1.23s/it - ETA 00:07\033[K"
# Weight loading uses the same bar with a byte rate (e.g. "MB/s"), so only
# lines whose speed is in s/it or it/s count as sampling steps.
_STEP_RE = re.compile(r'\|\s*(\d+)/(\d+) - +[\d.]+(?:s/it|it/s)')
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
TAIL_LINES = 40
STOP_GRACE_S = 5.0

_active = set()
_active_lock = threading.Lock()


class SdcppError(RuntimeError):
    """sd-cli failed; the message includes the end of its output."""


class Stopped(Exception):
    """The run was cancelled through should_stop()."""


def build_argv(binary: str, spec, model_dir: str, diffusion_name: str, *, prompt: str, negative_prompt: str,
               width: int, height: int, seed: int, steps: int, cfg: float, output: str,
               extra_args=()) -> list:
    """argv for one text-to-image run (a list, never a shell string)."""
    def path(role):
        return os.path.join(model_dir, spec.file(role).name)

    argv = [binary,
            '--diffusion-model', os.path.join(model_dir, diffusion_name),
            '--llm', path('llm'),
            '--vae', path('vae')]
    if spec.file('tokenizer') is not None:
        argv += ['--tokenizer', path('tokenizer')]
    argv += ['-p', prompt]
    if spec.uses_negative and cfg > 1.0 and negative_prompt.strip():
        argv += ['-n', negative_prompt]
    argv += ['--cfg-scale', f'{cfg:g}',
             '--steps', str(int(steps)),
             '--sampling-method', spec.sampler]
    if spec.scheduler:
        argv += ['--scheduler', spec.scheduler]
    argv += ['-W', str(int(width)), '-H', str(int(height)),
             '-s', str(int(seed)),
             '-o', output]
    argv += list(spec.extra_args) + list(extra_args)
    return argv


def parse_steps(text: str) -> list:
    """(step, steps) for each sampling progress update in ``text``."""
    return [(int(a), int(b)) for a, b in _STEP_RE.findall(_ANSI_RE.sub('', text))]


def split_updates(buffer: str):
    """Split raw output on both \\r and \\n (the progress bar redraws with \\r).
    Returns (complete pieces, unfinished remainder)."""
    pieces = re.split(r'[\r\n]', buffer)
    return pieces[:-1], pieces[-1]


def _kill_all():
    with _active_lock:
        procs = list(_active)
    for proc in procs:
        _terminate(proc)


atexit.register(_kill_all)  # never leave a ~13 GB sd-cli running after the app exits


def _terminate(proc):
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except OSError:
        pass


def run(argv: list, *, on_step=None, should_stop=None, env=None, cwd=None, log=print) -> float:
    """Run sd-cli. Calls on_step(step, steps) per sampling step, polls
    should_stop() (truthy -> terminate and raise Stopped). Returns seconds taken;
    raises SdcppError on failure."""
    started = time.perf_counter()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=env, cwd=cwd, bufsize=0)
    with _active_lock:
        _active.add(proc)
    chunks = queue.Queue()

    def reader():
        while True:
            data = proc.stdout.read(4096)
            if not data:
                break
            chunks.put(data)
        chunks.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    tail, buffer, done = [], '', False
    try:
        while not done:
            if should_stop is not None and should_stop():
                _terminate(proc)
                raise Stopped()
            try:
                data = chunks.get(timeout=0.2)
            except queue.Empty:
                continue
            if data is None:
                done = True
                pieces, buffer = [buffer], ''
            else:
                pieces, buffer = split_updates(buffer + data.decode('utf-8', errors='replace'))
            for piece in pieces:
                clean = _ANSI_RE.sub('', piece).strip()
                if not clean:
                    continue
                steps = parse_steps(clean)
                if steps:
                    if on_step is not None:
                        on_step(*steps[-1])
                    continue
                tail = (tail + [clean])[-TAIL_LINES:]
                if '|' not in clean:  # skip load-progress redraws in the app log
                    log(f'[sd.cpp] {clean}')
        code = proc.wait()
    finally:
        if proc.poll() is None:
            _terminate(proc)
        with _active_lock:
            _active.discard(proc)
    if code != 0:
        raise SdcppError(f'sd-cli exited with code {code}:\n' + '\n'.join(tail[-15:]))
    return time.perf_counter() - started
