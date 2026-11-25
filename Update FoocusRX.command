#!/usr/bin/env bash
# Double-click to update FoocusRX to the latest version on GitHub (main).
# Fetches, fast-forwards, installs changed packages, runs the tests and a
# smoke launch, and rolls everything back if anything fails.
# Options: --check (report only), --e2e (also run browser tests), --no-smoke
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR" || exit 1
PY="$DIR/venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[FoocusRX] No virtualenv found; running the installer instead."
    exec bash "$DIR/scripts/install_mac.command"
fi
"$PY" "$DIR/scripts/foocusrx.py" update "$@"
status=$?
echo
if [[ $status -eq 0 ]]; then
    echo "[FoocusRX] Done. Start FoocusRX from the FoocusRX app (or Launch FoocusRX.command)."
else
    echo "[FoocusRX] The update did not complete (status $status). Nothing was changed if it rolled back."
fi
if [[ -t 0 ]]; then
    read -r -p "Press Return to close this window." _ || true
fi
exit $status
