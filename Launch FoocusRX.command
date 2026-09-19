#!/usr/bin/env bash
# One-click FoocusRX launcher. Double-click this file.
# First run installs a local virtualenv; later runs just start the UI.
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# If this file sits at the bundle root, the source is ./FoocusRX.
# If it sits inside the source tree, the source is $DIR itself.
if [[ -f "$DIR/fooocus_version.py" ]]; then
    ROOT="$DIR"
elif [[ -f "$DIR/FoocusRX/fooocus_version.py" ]]; then
    ROOT="$DIR/FoocusRX"
else
    # Bundle root with no extracted FoocusRX/ yet: fall back to the
    # bundled source tarball, same as the installer does on first run.
    TARBALL=""
    shopt -s nullglob
    tarballs=("$DIR"/FoocusRX-*-src.tar.gz)
    shopt -u nullglob
    if [[ ${#tarballs[@]} -gt 0 ]]; then
        TARBALL="${tarballs[0]}"
    fi
    if [[ -z "$TARBALL" ]]; then
        echo "[FoocusRX] Could not find the FoocusRX folder or a source tarball next to this launcher." >&2
        echo "Press Return to close." >&2
        read -r _ || true
        exit 1
    fi
    echo "[FoocusRX] Extracting $TARBALL ..."
    tar -xzf "$TARBALL" -C "$DIR"
    ROOT="$DIR/FoocusRX"
fi

exec bash "$ROOT/scripts/mac_launch.sh" "$@"
