#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/install_mac.command
#
# Double-clickable macOS installer for FoocusRX.
#
# What it does (idempotently, safe to re-run):
#   1. Verify we are on macOS.
#   2. Restore a usable PATH (Finder-launched scripts omit Homebrew).
#   3. Verify Xcode Command Line Tools (`git` is optional; `python3` is not).
#   4. Use the FoocusRX folder this script lives in. Never clone into
#      $HOME just because the tree was shipped as a tarball without .git.
#   5. Create a Python venv at <root>/venv (Python 3.10–3.12).
#   6. Upgrade pip and install requirements_versions.txt.
#   7. Write/refresh Launch FoocusRX.command next to the source (and in
#      the parent bundle folder when present).
#   8. Launch FoocusRX via scripts/mac_launch.sh.
#
# Nothing here needs sudo. The installer writes only inside the chosen
# install root and the venv it creates. To uninstall, delete that folder.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/mac_common.sh"

fx_bootstrap_path
fx_keep_open_on_error

# ---- 1. OS check ----------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
    fx_err "This installer targets macOS only. Detected: $(uname -s)."
    fx_err "For Linux, use requirements_versions.txt + python -m venv directly."
    exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
    arm64) fx_info "Detected Apple Silicon (${BOLD}${ARCH}${RST}).";;
    x86_64) fx_warn "Detected Intel Mac. FoocusRX runs but the Metal Performance"
            fx_warn "Shaders backend is much slower without Apple Silicon."
            ;;
    *)     fx_warn "Unfamiliar architecture: $ARCH. Continuing anyway.";;
esac

# ---- 2. Python ------------------------------------------------------------
# Why capped at 3.12: several of our transitive pins (scipy 1.14.0,
# numpy 1.26.4, tokenizers 0.19.1, safetensors 0.4.3, pyyaml 6.0.1)
# do not publish cp313+ macOS-arm64 wheels on PyPI. A fresh Mac in 2026
# ships Python 3.13+ as /usr/bin/python3, so pip would try to build
# those from source and blow up on scipy's Meson build. Un-pinning
# them cascades into numpy 2.x, which breaks the torch 2.5.x we
# install for MPS. Cap at 3.12 until the pins move to numpy 2.x.

# git is nice-to-have (optional updates) but not required for a tarball
# / USB / airgapped install. python3 is required.
if ! command -v git >/dev/null 2>&1; then
    fx_warn "'git' not found. Updates via git pull will be skipped."
    fx_warn "Install Xcode Command Line Tools later with:  xcode-select --install"
fi

PYTHON_BIN="$(fx_find_python || true)"

if [[ -z "${PYTHON_BIN}" ]]; then
    fx_err "No supported Python found (need 3.10, 3.11, or 3.12)."
    fx_err ""
    fx_err "Reason: several pinned scientific packages don't publish Python"
    fx_err "3.13+ wheels for macOS arm64 yet, so pip would try to build"
    fx_err "scipy / numpy from source and fail."
    fx_err ""
    if command -v brew >/dev/null 2>&1; then
        printf '%s[FoocusRX]%s Install Python 3.12 via Homebrew now? [Y/n] ' "$YLW" "$RST"
        read -r reply || reply="n"
        if [[ "$reply" =~ ^([Yy]|)$ ]]; then
            fx_info "Running: brew install python@3.12"
            brew install python@3.12
            fx_bootstrap_path
            PYTHON_BIN="$(fx_find_python || true)"
            if [[ -z "${PYTHON_BIN}" ]]; then
                fx_err "brew install completed but python3.12 is still not on PATH."
                fx_err "Try:  brew link --overwrite python@3.12"
                exit 1
            fi
        else
            fx_err "Aborted. Install manually with:  brew install python@3.12"
            exit 1
        fi
    else
        fx_err "Install one of:"
        fx_err "    * Homebrew (https://brew.sh) then:  brew install python@3.12"
        fx_err "    * The python.org installer for Python 3.12"
        exit 1
    fi
fi

if ! fx_py_in_range "$PYTHON_BIN"; then
    fx_err "$PYTHON_BIN reports $($PYTHON_BIN -V 2>&1) which is outside 3.10-3.12."
    exit 1
fi
fx_ok "Using $PYTHON_BIN ($($PYTHON_BIN -V 2>&1))."

# ---- 3. locate the source tree -------------------------------------------
# Prefer the checkout this script lives in, even when it was extracted
# from a release tarball and has no .git directory. The previous check
# required .git, which made the official macOS zip clone into
# $HOME/FoocusRX and ignore the bundled source.
if [[ -f "$SCRIPT_DIR/../fooocus_version.py" ]]; then
    REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
    fx_info "Installing into local tree: $REPO_ROOT"
    if [[ -d "$REPO_ROOT/.git" ]] && command -v git >/dev/null 2>&1; then
        ( cd "$REPO_ROOT" && git pull --ff-only ) \
            || fx_warn "git pull skipped (dirty tree, offline, or no remote)."
    fi
else
    REPO_ROOT="${FOOOCUSRX_HOME:-$HOME/FoocusRX}"
    if [[ -f "$REPO_ROOT/fooocus_version.py" ]]; then
        fx_info "Using existing tree at $REPO_ROOT"
        if [[ -d "$REPO_ROOT/.git" ]] && command -v git >/dev/null 2>&1; then
            ( cd "$REPO_ROOT" && git pull --ff-only ) \
                || fx_warn "git pull skipped (dirty tree, offline, or no remote)."
        fi
    elif command -v git >/dev/null 2>&1; then
        fx_info "Cloning FoocusRX into $REPO_ROOT"
        git clone https://github.com/mlovelace112-maker/FoocusRX.git "$REPO_ROOT"
    else
        fx_err "No FoocusRX source next to this installer, and git is not available to clone one."
        fx_err "Extract FoocusRX-*-src.tar.gz next to Install FoocusRX.command and re-run."
        exit 1
    fi
fi

cd "$REPO_ROOT"
fx_warn_if_cloud "$REPO_ROOT"

# ---- 4. venv --------------------------------------------------------------
VENV_DIR="$REPO_ROOT/venv"
VENV_PY="$VENV_DIR/bin/python"

# Recreate if missing, or if the interpreter is a dangling symlink
# (folder copied to another Mac / iCloud evicted the binary).
if fx_venv_present "$REPO_ROOT"; then
    fx_info "Reusing existing virtualenv at $VENV_DIR"
else
    if [[ -d "$VENV_DIR" ]]; then
        fx_warn "Existing virtualenv is broken. Recreating at $VENV_DIR"
        rm -rf "$VENV_DIR"
    else
        fx_info "Creating virtualenv at $VENV_DIR"
    fi
    # --copies: real files instead of symlinks, so USB / folder copies
    # on the same Mac keep a working interpreter stub.
    "$PYTHON_BIN" -m venv --copies "$VENV_DIR"
fi

if [[ ! -x "$VENV_PY" ]]; then
    fx_err "venv was created but $VENV_PY is not executable."
    exit 1
fi

# ---- 5. install requirements ---------------------------------------------
fx_info "Upgrading pip / wheel / setuptools inside venv"
# Pin setuptools < 82. Torch 2.12.x for MPS declares setuptools<82 as an
# upper bound, and pip's resolver logs the conflict but still installs
# the newer setuptools, which then breaks torch's own build tools when
# any package touches torch.utils.cpp_extension. Cap it explicitly.
"$VENV_PY" -m pip install --upgrade pip wheel "setuptools<82"

fx_info "Installing requirements_versions.txt (this can take several minutes)"
"$VENV_PY" -m pip install -r "$REPO_ROOT/requirements_versions.txt"

# On Apple Silicon we do NOT pre-install mtlflashattn here — launch.py
# handles that on first run via auto_install_mtlflashattn() (PR #16).
# Reason: keeping the installer strictly Python-package driven avoids
# duplicating platform-detection logic. Advanced users who want to force
# the pre-install can still do: pip install -r requirements_mac.txt.

fx_ok "Dependencies installed."

# ---- 6. one-click launcher ------------------------------------------------
# Write a double-clickable launcher next to the source tree and, when
# this tree lives inside a release bundle (parent has the installer or
# a source tarball), also at the bundle root.
write_launcher() {
    local dest="$1"
    cat > "$dest" <<'LAUNCHER_EOF'
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
LAUNCHER_EOF
    chmod +x "$dest"
    fx_clear_quarantine "$dest"
}

write_launcher "$REPO_ROOT/Launch FoocusRX.command"
PARENT="$(dirname "$REPO_ROOT")"
shopt -s nullglob
_tarballs=("$PARENT"/FoocusRX-*-src.tar.gz)
shopt -u nullglob
if [[ -f "$PARENT/Install FoocusRX.command" || -e "$PARENT/Launch FoocusRX.command" || ${#_tarballs[@]} -gt 0 ]]; then
    write_launcher "$PARENT/Launch FoocusRX.command"
fi
fx_ok "One-click launcher ready: Launch FoocusRX.command"

# ---- 7. launch ------------------------------------------------------------
fx_info "Launching FoocusRX. First launch on M4+ will auto-install mtlflashattn."
fx_info "Press Ctrl-C in this window to quit."
echo
# Tell mac_launch.sh not to bounce back into this installer if something
# is still missing after pip — that would loop forever.
export FOOOCUS_INSTALLING=1
exec bash "$REPO_ROOT/scripts/mac_launch.sh" "$@"
