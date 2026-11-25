#!/usr/bin/env python3
"""Upscale finished images 2x with detail (the app's "Upscale (2x)").

Run it while FoocusRX is closed:

    venv/bin/python scripts/auto_upscale.py              # today's images, then keep watching
    venv/bin/python scripts/auto_upscale.py --once       # today's images, then exit
    venv/bin/python scripts/auto_upscale.py --days 3     # images from the last 3 days
    venv/bin/python scripts/auto_upscale.py --all --once # every image without an upscale

Each image is upscaled with its own prompt, styles, model and sampler (from
its History record) and saved next to it as <name>_2x.png. Images that
already have one are skipped. It stops if the app is running, because both
would load a ~12 GB model and won't fit in memory together.
"""
import argparse
import datetime
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    when = parser.add_mutually_exclusive_group()
    when.add_argument('--days', type=int, default=1, help='upscale images from the last N days (default: 1 = today)')
    when.add_argument('--all', action='store_true', help='upscale every image that has no upscale yet')
    parser.add_argument('--once', action='store_true', help='exit when done instead of watching for new images')
    parser.add_argument('--interval', type=float, default=10.0, help='seconds between checks while watching')
    parser.add_argument('--outputs', help='outputs folder (default: path_outputs from config.txt)')
    return parser.parse_args()


def main():
    opts = parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.argv = [sys.argv[0], '--disable-offload-from-vram', '--disable-analytics']  # for args_manager

    from modules import auto_upscale
    if auto_upscale.app_running():
        print('FoocusRX is running. Close it first: the app and this script would each load a ~12 GB '
              'model, which does not fit in memory together.')
        return 2

    from modules.apple_silicon import configure
    configure(verbose=False)
    import modules.config as config
    from modules.history import HistoryIndex

    outputs = opts.outputs or config.path_outputs
    since = None if opts.all else datetime.date.today() - datetime.timedelta(days=max(opts.days, 1) - 1)
    print(f'Watching {outputs} for images {"(all dates)" if since is None else f"since {since}"} ...')

    upscaler = None
    index = HistoryIndex()
    done = failed = skipped = 0
    passed_over = set()  # failed or unsupported this session: don't retry every pass
    while True:
        pending = [p for p in auto_upscale.find_pending(outputs, since) if p not in passed_over]
        if pending:
            records = {item.path: item for item in index.scan(outputs)}
        for path in pending:
            if auto_upscale.app_running():
                print('FoocusRX was started; stopping so the two don\'t compete for memory.')
                return 3
            item = records.get(path)
            meta = item.metadata if item else {}
            reason = auto_upscale.unsupported_reason(meta)
            if reason:
                skipped += 1
                passed_over.add(path)
                print(f'  {os.path.basename(path)}: skipped ({reason}).')
                continue
            if upscaler is None:
                print('Loading model (first image only) ...')
                upscaler = auto_upscale.DetailedUpscaler()
            if not meta:
                print(f'  {os.path.basename(path)}: no saved settings found; upscaling with default settings.')
            target = auto_upscale.upscaled_path(path)
            started = time.perf_counter()
            try:
                upscaler.upscale(path, meta, target)
                done += 1
                print(f'  {os.path.basename(path)} -> {os.path.basename(target)} '
                      f'({time.perf_counter() - started:.0f} s)')
            except Exception as exc:  # keep going with the next image; don't retry it this session
                failed += 1
                passed_over.add(path)
                print(f'  {os.path.basename(path)}: failed ({exc.__class__.__name__}: {exc})')
        if opts.once:
            print(f'Done: {done} upscaled, {failed} failed, {skipped} skipped.')
            return 0 if failed == 0 else 1
        try:
            time.sleep(opts.interval)
        except KeyboardInterrupt:
            print(f'\nStopped: {done} upscaled, {failed} failed, {skipped} skipped.')
            return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nStopped.')
        sys.exit(130)
