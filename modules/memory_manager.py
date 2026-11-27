"""Clean model unloading, out-of-memory recovery and a memory readout.

FoocusRX runs several model families (SDXL, FLUX.1, Krea 2 / MOSS through
sd.cpp), upscaler models and, through Prompt Studio, Ollama models. On Apple
Silicon all of them share one pool of unified memory, so anything left loaded
by one stage is memory the next stage can't have. This module is the one place
that knows how to:

* free everything (``free_everything``): SDXL, FLUX.1, the upscaler models, the
  prompt rewriter and Prompt Studio's Ollama models, then PyTorch's cache;
* return PyTorch's cached memory to the system (``release_cached_memory``);
* recognise an out-of-memory error on any backend (``is_oom_error``). On MPS it
  is a plain ``RuntimeError("MPS backend out of memory ...")``, which the
  upstream ``except OOM_EXCEPTION`` handlers (CUDA's class) never caught;
* run a step, and if it runs out of memory, free everything and try once more
  (``run_with_oom_retry``);
* print a one-line memory readout (``log_memory``).

Swap: macOS decides what goes to swap; an app can't direct memory there. What
FoocusRX controls is when PyTorch gives up. With "allow swap" on (the default,
see modules/apple_silicon.py) PyTorch may allocate past the physical memory
budget and macOS swaps; with it off PyTorch raises out-of-memory first.

Importing this module does not import torch, so it is cheap to import and test.
"""
from __future__ import annotations

import gc
import importlib
import os
import threading
import time

GB = 1024 ** 3

_OOM_MESSAGES = (
    'out of memory',                 # CUDA, MPS ("MPS backend out of memory"), XPU, DirectML
    'insufficient memory',           # Metal command buffer errors surfaced as RuntimeError
    'failed to allocate',            # some allocators phrase it this way
    'not enough memory',             # Windows / DirectML
)


def is_oom_error(exc: BaseException) -> bool:
    """True for an out-of-memory error from PyTorch on CUDA, MPS, XPU or the
    CPU, whatever class it was raised as. Never true for Stop/Skip."""
    if exc is None:
        return False
    names = {cls.__name__ for cls in type(exc).__mro__}
    if 'InterruptProcessingException' in names:
        return False
    if 'OutOfMemoryError' in names or isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return any(text in message for text in _OOM_MESSAGES)
    return False


# --- Releasing memory ------------------------------------------------------------

def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def release_cached_memory():
    """Collect unreachable Python objects (model graphs often sit in reference
    cycles), wait for queued GPU work, and hand PyTorch's cached blocks back to
    the system. Safe to call any time; does nothing without torch."""
    gc.collect()
    torch = _torch()
    if torch is None:
        return
    try:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:  # never let cleanup fail a generation
        print(f'[memory] cache release skipped: {exc!r}')


def free_everything(reason: str = '', wait_for_ollama: bool = True):
    """Unload every model FoocusRX holds: SDXL (base, refiner, ControlNets),
    FLUX.1, the upscaler models, the prompt rewriter and Prompt Studio's Ollama
    models, then release the cache. The next step reloads what it needs."""
    if reason:
        print(f'[memory] freeing all models: {reason}')
    steps = (
        ('FLUX.1', lambda: importlib.import_module('modules.flux.app').unload()),
        ('SDXL', lambda: importlib.import_module('modules.flux.app').free_sdxl()),
        ('upscaler', lambda: importlib.import_module('modules.upscaler').release()),
        ('prompt rewriter', lambda: importlib.import_module('modules.prompt_rewriter').release()),
        ('Prompt Studio', lambda: _release_ollama(wait_for_ollama)),
    )
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            print(f'[memory] could not free {name}: {exc!r}')
    release_cached_memory()


def free_gpu_memory(reason: str = '', wait_for_ollama: bool = True):
    """Lighter than free_everything: move the loaded models off the GPU (the
    pipeline keeps them, so the next step loads them back without reading the
    checkpoint again), drop the upscaler, rewriter and Ollama models, and
    release the cache. Used to retry a step that ran out of memory."""
    if reason:
        print(f'[memory] moving models off the GPU: {reason}')
    steps = (
        ('loaded models', lambda: importlib.import_module(
            'ldm_patched.modules.model_management').unload_all_models()),
        ('upscaler', lambda: importlib.import_module('modules.upscaler').release()),
        ('prompt rewriter', lambda: importlib.import_module('modules.prompt_rewriter').release()),
        ('Prompt Studio', lambda: _release_ollama(wait_for_ollama)),
    )
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            print(f'[memory] could not free {name}: {exc!r}')
    release_cached_memory()


def _release_ollama(wait: bool):
    prompt_studio = importlib.import_module('modules.prompt_studio')
    prompt_studio.release(wait=False)
    if wait:
        prompt_studio.wait_for_release()


def run_with_oom_retry(fn, *args, label: str = 'this step', free=None, retries: int = 1, **kwargs):
    """``fn(*args, **kwargs)``; if it runs out of memory, free memory with
    ``free`` (default: free_everything) and try again, up to ``retries`` times.
    Any other error, and an out-of-memory error on the last try, is raised."""
    free = free_everything if free is None else free
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:
            if attempt >= retries or not is_oom_error(exc):
                raise
            attempt += 1
            print(f'[memory] ran out of memory during {label}; freeing memory and retrying '
                  f'({attempt}/{retries}).')
        # Free outside the except block: the error's traceback keeps the failed
        # step's tensors alive until the block ends.
        free(f'out of memory during {label}')


# --- Readout -------------------------------------------------------------------

def _gb(value) -> str:
    return f'{value / GB:.1f} GB'


def memory_snapshot() -> dict:
    """Byte counts for the readout; keys are missing when unknown."""
    snapshot = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        snapshot['ram_total'] = vm.total
        snapshot['ram_used'] = vm.total - vm.available
        swap = psutil.swap_memory()
        snapshot['swap_used'] = swap.used
    except Exception:
        pass
    torch = _torch()
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                snapshot['gpu_name'] = 'MPS'
                snapshot['gpu_allocated'] = torch.mps.current_allocated_memory()
                snapshot['gpu_reserved'] = torch.mps.driver_allocated_memory()
            elif torch.cuda.is_available():
                snapshot['gpu_name'] = 'CUDA'
                snapshot['gpu_allocated'] = torch.cuda.memory_allocated()
                snapshot['gpu_reserved'] = torch.cuda.memory_reserved()
        except Exception:
            pass
    return snapshot


def format_memory_line(label: str, snapshot: dict) -> str:
    parts = []
    if 'gpu_allocated' in snapshot:
        gpu = f"{snapshot.get('gpu_name', 'GPU')} {_gb(snapshot['gpu_allocated'])} in use"
        if 'gpu_reserved' in snapshot:
            gpu += f" ({_gb(snapshot['gpu_reserved'])} held)"
        parts.append(gpu)
    if 'ram_used' in snapshot and 'ram_total' in snapshot:
        parts.append(f"RAM {_gb(snapshot['ram_used'])} of {_gb(snapshot['ram_total'])}")
    if 'swap_used' in snapshot:
        parts.append(f"swap {_gb(snapshot['swap_used'])}")
    body = ' · '.join(parts) if parts else 'unavailable'
    return f'[memory] {label}: {body}'


def memory_log_enabled() -> bool:
    return os.environ.get('FOOOCUS_MEMORY_LOG', '1').strip().lower() not in ('0', 'false', 'off', 'no')


def log_memory(label: str):
    """Print the one-line readout (FOOOCUS_MEMORY_LOG=0 turns it off)."""
    if not memory_log_enabled():
        return None
    line = format_memory_line(label, memory_snapshot())
    print(line)
    return line


# --- Idle unload ---------------------------------------------------------------

def idle_unload_seconds() -> float:
    """Seconds without a generation after which models are moved out of GPU
    memory (FOOOCUS_IDLE_UNLOAD_MINUTES, default 10; 0 turns it off)."""
    raw = os.environ.get('FOOOCUS_IDLE_UNLOAD_MINUTES', '10').strip()
    try:
        minutes = float(raw)
    except ValueError:
        minutes = 10.0
    return max(0.0, minutes) * 60.0


_idle_lock = threading.Lock()
_last_activity = time.monotonic()


def mark_activity():
    global _last_activity
    with _idle_lock:
        _last_activity = time.monotonic()


def idle_for() -> float:
    with _idle_lock:
        return time.monotonic() - _last_activity


def unload_idle_models():
    """Move the loaded SDXL models off the GPU (they stay in RAM, where macOS
    can compress or swap them while unused) and release the GPU cache; the
    next generation moves them back. FLUX.1 is unloaded (it reloads from disk)
    and Ollama is asked to unload Prompt Studio's models."""
    try:
        mm = importlib.import_module('ldm_patched.modules.model_management')
        loaded = len(mm.current_loaded_models)
        mm.unload_all_models()
    except Exception as exc:
        print(f'[memory] idle unload skipped: {exc!r}')
        return False
    try:
        flux_app = importlib.import_module('modules.flux.app')
        if flux_app._pipeline is not None:  # FLUX.1 isn't in model_management's list
            flux_app.unload()
            loaded += 1
    except Exception as exc:
        print(f'[memory] idle FLUX.1 unload skipped: {exc!r}')
    try:
        importlib.import_module('modules.prompt_studio').release(wait=False)
    except Exception:
        pass
    release_cached_memory()
    if loaded:
        log_memory('idle, models moved off the GPU')
    return bool(loaded)
