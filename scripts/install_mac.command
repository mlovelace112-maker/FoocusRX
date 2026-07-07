#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/install_mac.command
#
# Double-clickable macOS installer for FoocusRX.
#
# What it does (idempotently, safe to re-run):
#   1. Verify we are on macOS.
#   2. Verify Xcode Command Line Tools are installed (needs `git`, `python3`).
#   3. Choose an install root:
#        * If this file lives inside a checked-out FoocusRX repo, use that.
#        * Otherwise clone the repo into $HOME/FoocusRX (or update it).
#   4. Create a Python venv at <root>/venv (Python 3.10 or newer required).
#   5. Upgrade pip and install requirements_versions.txt.
#   6. Launch FoocusRX via scripts/mac_launch.sh, which triggers first-run
#      auto-install of mtlflashattn on M4+ (see PR #16).
#
# Nothing here needs sudo. The installer writes only inside the chosen
# install root and the venv it creates. To uninstall, delete that folder.
#
# Design notes:
#   * The file has a .command extension so Finder treats it as executable
#     and opens a Terminal window on double-click.
#   * `set -euo pipefail` catches missing prerequisites early.
#   * We `cd` to the script's own directory to make relative paths sane
#     regardless of how the user launched it.
#   * `trap` keeps the Terminal window open on error so the user can read
#     the diagnostic instead of losing it when the shell exits.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- pretty output --------------------------------------------------------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
YLW=$'\033[33m'; BLU=$'\033[34m'; RST=$'\033[0m'

info()  { printf '%s[FoocusRX]%s %s\n' "$BLU" "$RST" "$*"; }
ok()    { printf '%s[FoocusRX]%s %s\n' "$GRN" "$RST" "$*"; }
warn()  { printf '%s[FoocusRX]%s %s\n' "$YLW" "$RST" "$*"; }
err()   { printf '%s[FoocusRX]%s %s\n' "$RED" "$RST" "$*" >&2; }

# Keep the Terminal window open long enough to read a failure message
# when the user double-clicked from Finder.
on_error() {
    err "Installer failed on line $1. Scroll up for details."
    err "Press Return to close this window."
    read -r _ || true
}
trap 'on_error $LINENO' ERR

# ---- 1. OS check ----------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
    err "This installer targets macOS only. Detected: $(uname -s)."
    err "For Linux, use requirements_versions.txt + python -m venv directly."
    exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
    arm64) info "Detected Apple Silicon (${BOLD}${ARCH}${RST}).";;
    x86_64) warn "Detected Intel Mac. FoocusRX runs but the Metal Performance"
            warn "Shaders backend is much slower without Apple Silicon."
            ;;
    *)     warn "Unfamiliar architecture: $ARCH. Continuing anyway.";;
esac

# ---- 2. prerequisites -----------------------------------------------------
# Xcode Command Line Tools give us /usr/bin/git and /usr/bin/python3. We
# probe for `git` first because `xcode-select -p` returning a path is not
# enough — the tools directory can exist while the binaries are missing
# after certain OS upgrades.
if ! command -v git >/dev/null 2>&1; then
    err "'git' not found. Install Xcode Command Line Tools:"
    err "    xcode-select --install"
    exit 1
fi

# Pick a Python interpreter in the supported range [3.10, 3.12].
#
# Why capped at 3.12: several of our transitive pins (scipy 1.14.0,
# numpy 1.26.4, tokenizers 0.19.1, safetensors 0.4.3, pyyaml 6.0.1)
# do not publish cp313 macOS-arm64 wheels on PyPI. A fresh Mac in 2026
# ships Python 3.13 as /usr/bin/python3, so pip would try to build
# those from source and blow up on scipy's Meson build. Un-pinning
# them cascades into numpy 2.x, which breaks the torch 2.5.x we
# install for MPS. Cap at 3.12 until the pins move to numpy 2.x.

# Helper: is $1 an interpreter in the [3.10, 3.12] range?
py_in_range() {
    "$1" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)
PY
}

PYTHON_BIN=""
for cand in python3.12 python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "$cand")"
        break
    fi
done

# Fall back to /usr/bin/python3 (or similar) only if it happens to be
# in range. This is the common Homebrew / Xcode-CLT case on 3.11-era Macs.
if [[ -z "$PYTHON_BIN" ]] && command -v python3 >/dev/null 2>&1; then
    _generic="$(command -v python3)"
    if py_in_range "$_generic"; then
        PYTHON_BIN="$_generic"
    fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
    err "No supported Python found (need 3.10, 3.11, or 3.12)."
    err ""
    err "Reason: several pinned scientific packages don't publish Python"
    err "3.13 wheels for macOS arm64 yet, so pip would try to build"
    err "scipy / numpy from source and fail."
    err ""
    if command -v brew >/dev/null 2>&1; then
        printf '%s[FoocusRX]%s Install Python 3.12 via Homebrew now? [Y/n] ' "$YLW" "$RST"
        read -r reply || reply="n"
        if [[ "$reply" =~ ^([Yy]|)$ ]]; then
            info "Running: brew install python@3.12"
            brew install python@3.12
            PYTHON_BIN="$(command -v python3.12 || true)"
            if [[ -z "$PYTHON_BIN" ]]; then
                err "brew install completed but python3.12 is still not on PATH."
                err "Try:  brew link --overwrite python@3.12"
                exit 1
            fi
        else
            err "Aborted. Install manually with:  brew install python@3.12"
            exit 1
        fi
    else
        err "Install one of:"
        err "    * Homebrew (https://brew.sh) then:  brew install python@3.12"
        err "    * The python.org installer for Python 3.12"
        exit 1
    fi
fi

# Final sanity check against the selected interpreter — guards against
# an in-PATH but weird python3.12 (e.g. asdf shim that resolves to 3.13).
if ! py_in_range "$PYTHON_BIN"; then
    err "$PYTHON_BIN reports $("$PYTHON_BIN" -V 2>&1) which is outside 3.10-3.12."
    exit 1
fi
ok "Using $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))."

# ---- 3. locate or clone the repo -----------------------------------------
# Resolve this script's own directory. `readlink -f` isn't on stock macOS,
# so use the portable cd/pwd trick.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Are we inside a FoocusRX checkout already? The marker is fooocus_version.py
# one directory above scripts/.
if [[ -f "$SCRIPT_DIR/../fooocus_version.py" && -d "$SCRIPT_DIR/../.git" ]]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
    info "Running from an existing checkout: $REPO_ROOT"
    ( cd "$REPO_ROOT" && git pull --ff-only || warn "git pull skipped (dirty tree or offline)." )
else
    REPO_ROOT="${FOOOCUSRX_HOME:-$HOME/FoocusRX}"
    if [[ -d "$REPO_ROOT/.git" ]]; then
        info "Updating existing clone at $REPO_ROOT"
        ( cd "$REPO_ROOT" && git pull --ff-only || warn "git pull skipped (dirty tree or offline)." )
    else
        info "Cloning FoocusRX into $REPO_ROOT"
        git clone https://github.com/mlovelace112-maker/FoocusRX.git "$REPO_ROOT"
    fi
fi

cd "$REPO_ROOT"

# ---- 4. venv --------------------------------------------------------------
VENV_DIR="$REPO_ROOT/venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtualenv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    info "Reusing existing virtualenv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---- 5. install requirements ---------------------------------------------
info "Upgrading pip / wheel / setuptools inside venv"
# Pin setuptools < 82. Torch 2.12.x for MPS declares setuptools<82 as an
# upper bound, and pip's resolver logs the conflict but still installs
# the newer setuptools, which then breaks torch's own build tools when
# any package touches torch.utils.cpp_extension. Cap it explicitly.
python -m pip install --upgrade --quiet pip wheel "setuptools<82"

info "Installing requirements_versions.txt (this can take several minutes)"
python -m pip install --quiet -r requirements_versions.txt

# On Apple Silicon we do NOT pre-install mtlflashattn here — launch.py
# handles that on first run via auto_install_mtlflashattn() (PR #16).
# Reason: keeping the installer strictly Python-package driven avoids
# duplicating platform-detection logic. Advanced users who want to force
# the pre-install can still do: pip install -r requirements_mac.txt.

ok "Dependencies installed."

# ---- 6. launch ------------------------------------------------------------
info "Launching FoocusRX. First launch on M4+ will auto-install mtlflashattn."
info "Press Ctrl-C in this window to quit."
echo
exec bash scripts/mac_launch.sh "$@"
