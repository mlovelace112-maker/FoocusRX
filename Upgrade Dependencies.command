#!/usr/bin/env bash
# Double-click to see which pinned Python packages have newer versions, and
# optionally test one in a copy of the virtualenv (your real one is untouched).
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR" || exit 1
PY="$DIR/venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[FoocusRX] No virtualenv found; run scripts/install_mac.command first."
    read -r -p "Press Return to close this window." _ || true
    exit 1
fi
"$PY" "$DIR/scripts/foocusrx.py" upgrade check
echo
read -r -p "Package to test (e.g. gradio==5.50.0), or Return to quit: " spec || true
if [[ -n "${spec:-}" ]]; then
    # shellcheck disable=SC2086
    "$PY" "$DIR/scripts/foocusrx.py" upgrade test $spec
    if [[ $? -eq 0 ]]; then
        read -r -p "Open a pull request with the new pins now? [y/N] " yn || true
        if [[ "${yn:-}" =~ ^[Yy] ]]; then
            "$PY" "$DIR/scripts/foocusrx.py" upgrade apply
        fi
    fi
fi
read -r -p "Press Return to close this window." _ || true
