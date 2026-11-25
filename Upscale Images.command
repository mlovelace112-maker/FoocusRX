#!/usr/bin/env bash
# Double-click to upscale today's images 2x with detail, then keep watching
# for new ones (close this window or press Ctrl-C to stop).
# Close FoocusRX first: both would load a ~12 GB model.
# Options: see `venv/bin/python scripts/auto_upscale.py --help`.
set -uo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [[ -f "$DIR/fooocus_version.py" ]]; then
    ROOT="$DIR"
elif [[ -f "$DIR/FoocusRX/fooocus_version.py" ]]; then
    ROOT="$DIR/FoocusRX"
else
    echo "[FoocusRX] Could not find the FoocusRX folder next to this file." >&2
    read -r -p "Press Return to close." _ || true
    exit 1
fi
PY="$ROOT/venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[FoocusRX] FoocusRX isn't installed yet. Run \"Launch FoocusRX.command\" once first." >&2
    read -r -p "Press Return to close." _ || true
    exit 1
fi
cd "$ROOT"
"$PY" scripts/auto_upscale.py "$@"
status=$?
if [[ -t 0 ]]; then
    read -r -p "Finished (exit $status). Press Return to close." _ || true
fi
exit "$status"
