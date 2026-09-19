#!/usr/bin/env bash
# ------------------------------------------------------------------
# scripts/mac_launch.sh
#
# One-click / CLI launcher for FoocusRX on macOS.
#
# What it does:
#   * Resolves the repo from this script's location (works from Finder,
#     USB sticks, paths with spaces, iCloud copies).
#   * Uses the local venv interpreter — never a random `python` on PATH.
#   * If the venv is missing or broken, hands off to install_mac.command.
#   * Clears proxy env vars so Gradio's localhost health probe does not
#     get routed through a corporate VPN / Charles / Proxyman.
#     See https://github.com/gradio-app/gradio/issues/4046.
#
# Usage:
#   ./scripts/mac_launch.sh
#   ./scripts/mac_launch.sh --preset anime
#   ./Launch\ FoocusRX.command
# ------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/mac_common.sh"

fx_bootstrap_path
fx_keep_open_on_error

REPO_ROOT="$(fx_find_repo_root "$SCRIPT_DIR")" || {
    fx_err "Could not find FoocusRX (no fooocus_version.py above $SCRIPT_DIR)."
    exit 1
}

cd "$REPO_ROOT"
fx_warn_if_cloud "$REPO_ROOT"
fx_clear_quarantine \
    "$REPO_ROOT/scripts/mac_launch.sh" \
    "$REPO_ROOT/scripts/install_mac.command" \
    "$REPO_ROOT/Launch FoocusRX.command"

if ! fx_venv_ready "$REPO_ROOT"; then
    if [[ "${FOOOCUS_INSTALLING:-}" == "1" ]]; then
        fx_err "Installer finished but the virtualenv still cannot import"
        fx_err "gradio / numpy / Pillow. Scroll up for pip errors."
        exit 1
    fi
    if fx_venv_present "$REPO_ROOT"; then
        fx_warn "Virtualenv exists but required packages are missing or the"
        fx_warn "interpreter is broken (moved folder, other Mac, iCloud eviction)."
        fx_warn "Running the installer to repair it."
    else
        fx_info "No virtualenv yet — running first-time installer."
    fi
    echo
    exec bash "$REPO_ROOT/scripts/install_mac.command" "$@"
fi

# Clear inherited proxies (only for this process; the user's shell is unchanged).
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

# Belt-and-braces loopback bypass — some libraries read NO_PROXY, some no_proxy.
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

PYTHON="$(fx_venv_python "$REPO_ROOT")"
fx_ok "Launching FoocusRX with $PYTHON"
fx_info "Press Ctrl-C in this window to quit."
echo

# Do not `exec` — if Python exits non-zero we want the trap to keep the
# Terminal window open so the user can read the traceback.
set +e
"$PYTHON" entry_with_update.py --disable-offload-from-vram "$@"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
    fx_err "FoocusRX exited with status $status."
    if [[ -t 0 ]]; then
        fx_err "Press Return to close this window."
        read -r _ || true
    fi
    exit "$status"
fi
