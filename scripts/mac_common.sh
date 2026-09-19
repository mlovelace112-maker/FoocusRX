#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/mac_common.sh
#
# Shared helpers for the macOS installer and one-click launcher.
# Source this file; do not execute it.
# ---------------------------------------------------------------------------

# ---- pretty output --------------------------------------------------------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
YLW=$'\033[33m'; BLU=$'\033[34m'; RST=$'\033[0m'

fx_info()  { printf '%s[FoocusRX]%s %s\n' "$BLU" "$RST" "$*"; }
fx_ok()    { printf '%s[FoocusRX]%s %s\n' "$GRN" "$RST" "$*"; }
fx_warn()  { printf '%s[FoocusRX]%s %s\n' "$YLW" "$RST" "$*"; }
fx_err()   { printf '%s[FoocusRX]%s %s\n' "$RED" "$RST" "$*" >&2; }

# Finder-launched .command files get a stripped PATH that often omits
# Homebrew, python.org, and pyenv. Put those locations back before we
# look for an interpreter.
fx_bootstrap_path() {
    local extras=()
    extras+=(
        /opt/homebrew/bin
        /opt/homebrew/sbin
        /opt/homebrew/opt/python@3.12/bin
        /opt/homebrew/opt/python@3.11/bin
        /opt/homebrew/opt/python@3.10/bin
        /usr/local/bin
        /usr/local/sbin
        /usr/local/opt/python@3.12/bin
        /usr/local/opt/python@3.11/bin
        /usr/local/opt/python@3.10/bin
        /Library/Frameworks/Python.framework/Versions/3.12/bin
        /Library/Frameworks/Python.framework/Versions/3.11/bin
        /Library/Frameworks/Python.framework/Versions/3.10/bin
        "$HOME/.local/bin"
        "$HOME/.pyenv/shims"
        "$HOME/.asdf/shims"
    )
    if [[ -x /opt/homebrew/bin/brew ]]; then
        # shellcheck disable=SC1091
        eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null)" || true
    elif [[ -x /usr/local/bin/brew ]]; then
        # shellcheck disable=SC1091
        eval "$(/usr/local/bin/brew shellenv 2>/dev/null)" || true
    fi
    local p
    for p in "${extras[@]}"; do
        if [[ -d "$p" && ":$PATH:" != *":$p:"* ]]; then
            PATH="$p:$PATH"
        fi
    done
    export PATH
}

# Keep the Terminal window open on failure when the user double-clicked
# from Finder. No-ops when stdin is not a TTY (CI, `ssh`, pipes).
fx_keep_open_on_error() {
    on_error() {
        fx_err "Failed on line $1. Scroll up for details."
        if [[ -t 0 ]]; then
            fx_err "Press Return to close this window."
            read -r _ || true
        fi
    }
    trap 'on_error $LINENO' ERR
}

fx_pause_on_exit() {
    local status=$?
    if [[ "$status" -ne 0 && -t 0 ]]; then
        fx_err "Press Return to close this window."
        read -r _ || true
    fi
}

# Absolute directory of the calling script (not this helper).
fx_caller_dir() {
    cd "$(dirname "${BASH_SOURCE[1]}")" >/dev/null 2>&1 && pwd
}

# Walk up from $1 until we find fooocus_version.py. Empty on failure.
fx_find_repo_root() {
    local here="$1"
    local probe
    while [[ -n "$here" && "$here" != "/" ]]; do
        if [[ -f "$here/fooocus_version.py" ]]; then
            printf '%s\n' "$here"
            return 0
        fi
        # Bundle layout: this folder contains FoocusRX/fooocus_version.py
        probe="$here/FoocusRX"
        if [[ -f "$probe/fooocus_version.py" ]]; then
            printf '%s\n' "$probe"
            return 0
        fi
        here="$(dirname "$here")"
    done
    return 1
}

# True if $1 looks like cloud-synced storage where venv binaries get
# evicted or rewritten as placeholders.
fx_is_cloud_path() {
    local p="${1:-}"
    case "$p" in
        *"Mobile Documents"*|*"CloudDocs"*|*"iCloud"*|*"Dropbox"*|*"Google Drive"*|*"OneDrive"*|*"Box Sync"*)
            return 0
            ;;
    esac
    return 1
}

fx_warn_if_cloud() {
    local root="$1"
    if fx_is_cloud_path "$root"; then
        fx_warn "This copy lives in cloud-synced storage:"
        fx_warn "  $root"
        fx_warn "iCloud / Dropbox / Drive often evict Python binaries, which"
        fx_warn "breaks the virtualenv. For a portable, reliable install:"
        fx_warn "  1. Copy this whole folder to a local disk (e.g. ~/FoocusRX)"
        fx_warn "     or a USB volume that stays mounted."
        fx_warn "  2. Double-click Launch FoocusRX.command from the new location."
        echo
    fi
}

# Is $1 a Python 3.10, 3.11, or 3.12 interpreter?
fx_py_in_range() {
    "$1" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)
PY
}

fx_find_python() {
    local cand _generic
    for cand in python3.12 python3.11 python3.10; do
        if command -v "$cand" >/dev/null 2>&1; then
            cand="$(command -v "$cand")"
            if fx_py_in_range "$cand"; then
                printf '%s\n' "$cand"
                return 0
            fi
        fi
    done
    if command -v python3 >/dev/null 2>&1; then
        _generic="$(command -v python3)"
        if fx_py_in_range "$_generic"; then
            printf '%s\n' "$_generic"
            return 0
        fi
    fi
    return 1
}

fx_venv_python() {
    local repo="$1"
    printf '%s\n' "$repo/venv/bin/python"
}

# A venv is "present" if its interpreter exists and can start.
fx_venv_present() {
    local py
    py="$(fx_venv_python "$1")"
    [[ -x "$py" ]] && "$py" -c "import sys" >/dev/null 2>&1
}

# A venv is "ready" if the packages from requirements_versions.txt
# import. Do NOT check torch here — launch.py installs torch on the
# first run via prepare_environment(). Treating a missing torch as
# "broken" would send the launcher back into the installer forever.
fx_venv_ready() {
    local py
    py="$(fx_venv_python "$1")"
    [[ -x "$py" ]] || return 1
    "$py" -c "import gradio, numpy, PIL, packaging" >/dev/null 2>&1
}

# Strip the Gatekeeper quarantine flag so double-click works after a
# download or USB copy. Best-effort; failures are ignored.
fx_clear_quarantine() {
    local f
    for f in "$@"; do
        [[ -e "$f" ]] || continue
        xattr -d com.apple.quarantine "$f" >/dev/null 2>&1 || true
    done
}
