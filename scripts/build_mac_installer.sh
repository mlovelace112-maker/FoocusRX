#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/build_mac_installer.sh
#
# Produce a distributable macOS installer archive:
#   FoocusRX-<version>-macOS.zip
#     ├── Install FoocusRX.command       (double-clickable installer)
#     ├── README.txt                     (short human-readable primer)
#     └── FoocusRX-<version>-src.tar.gz  (repo source at HEAD)
#
# The user downloads the zip, unzips it, and double-clicks
# "Install FoocusRX.command". The installer extracts the bundled source
# tarball beside itself (or clones fresh if the tarball is missing) and
# then follows the venv + pip + launch flow.
#
# We prefer bundling the source tarball because:
#   * it lets users install FoocusRX from a USB stick / airgapped mac
#     without a fresh `git clone` from GitHub;
#   * the released zip is reproducible from a known commit, not "whatever
#     main happens to be at install time".
#
# Usage:
#   scripts/build_mac_installer.sh                # builds ./dist/FoocusRX-<ver>-macOS.zip
#   scripts/build_mac_installer.sh --out ~/tmp    # write into ~/tmp
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

OUT_DIR="$REPO_ROOT/dist"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

# ---- 1. version -----------------------------------------------------------
VERSION=$(python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from fooocus_version import version; print(version)")
echo "[build] FoocusRX version: $VERSION"

STAGE="$(mktemp -d)"
BUNDLE_DIR="$STAGE/FoocusRX-$VERSION-macOS"
mkdir -p "$BUNDLE_DIR"

# ---- 2. source tarball ----------------------------------------------------
# `git archive` gives us a clean snapshot of HEAD without .git or ignored
# files. Wrap it in a top-level FoocusRX/ directory so extraction is tidy.
SRC_TARBALL="$BUNDLE_DIR/FoocusRX-$VERSION-src.tar.gz"
echo "[build] Creating source tarball via git archive"
( cd "$REPO_ROOT" && git archive --format=tar.gz --prefix="FoocusRX/" -o "$SRC_TARBALL" HEAD )

# ---- 3. installer entry point --------------------------------------------
# We ship a thin wrapper (`Install FoocusRX.command`) that extracts the
# tarball if a checkout isn't already present next to it, then hands off
# to scripts/install_mac.command inside the extracted repo.
cat > "$BUNDLE_DIR/Install FoocusRX.command" <<COMMAND_EOF
#!/usr/bin/env bash
# Double-clickable installer wrapper. Extracts the bundled FoocusRX
# source tarball into ./FoocusRX/ next to this script (if not already
# extracted), then hands off to the repo's own installer.

set -euo pipefail
DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "\$DIR"

if [[ ! -d "FoocusRX/.git" && ! -f "FoocusRX/fooocus_version.py" ]]; then
    TARBALL=\$(ls FoocusRX-*-src.tar.gz 2>/dev/null | head -n1)
    if [[ -z "\$TARBALL" ]]; then
        echo "No source tarball found next to this installer." >&2
        echo "Press Return to close."
        read -r _ || true
        exit 1
    fi
    echo "[FoocusRX] Extracting \$TARBALL ..."
    tar -xzf "\$TARBALL"
fi

exec bash "FoocusRX/scripts/install_mac.command" "\$@"
COMMAND_EOF

chmod +x "$BUNDLE_DIR/Install FoocusRX.command"

# ---- 4. README ------------------------------------------------------------
cat > "$BUNDLE_DIR/README.txt" <<README_EOF
FoocusRX $VERSION — macOS installer
====================================

Quick start
-----------
1. Double-click "Install FoocusRX.command".
   (If macOS blocks it: right-click → Open → Open, or
    System Settings → Privacy & Security → "Open Anyway".)

2. A Terminal window opens and runs the installer. It will:
     • verify Xcode Command Line Tools and Python 3.10+
     • extract the source next to this installer
     • create a virtualenv under FoocusRX/venv
     • install requirements_versions.txt
     • launch the UI in your browser

3. On Apple Silicon M4 or newer, the first launch auto-installs
   mtlflashattn (attention kernel with fp32 accumulation). To
   opt out, run with FOOOCUS_AUTO_INSTALL_MTLFLASHATTN=0.

Re-running / updating
---------------------
Double-clicking the installer again is safe: it reuses the venv,
runs "git pull --ff-only" if possible, and reinstalls requirements.

Uninstall
---------
Delete the FoocusRX/ folder created next to this installer (and
the "\$HOME/FoocusRX" folder if you used the default location).

Troubleshooting
---------------
See FoocusRX/docs/apple-silicon.md and FoocusRX/troubleshoot.md.
README_EOF

# ---- 5. zip it up ---------------------------------------------------------
ZIP_NAME="FoocusRX-$VERSION-macOS.zip"
ZIP_PATH="$OUT_DIR/$ZIP_NAME"
rm -f "$ZIP_PATH"

echo "[build] Zipping bundle → $ZIP_PATH"
# We cd into $STAGE so paths inside the zip start with "FoocusRX-$VERSION-macOS/…"
( cd "$STAGE" && zip -r -q "$ZIP_PATH" "FoocusRX-$VERSION-macOS" )

# ---- 6. cleanup + summary ------------------------------------------------
rm -rf "$STAGE"

SIZE=$(du -h "$ZIP_PATH" | cut -f1)
echo "[build] Done. $ZIP_PATH ($SIZE)"
echo "[build] SHA256:"
if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$ZIP_PATH"
else
    sha256sum "$ZIP_PATH"
fi
