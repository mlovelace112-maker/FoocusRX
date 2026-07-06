# macOS Installer

FoocusRX 2.6+ ships a double-clickable macOS installer via GitHub Releases,
plus a source-tree entry point for people who already have the repo cloned.

## For end users

1. Download `FoocusRX-<version>-macOS.zip` from the
   [releases page](https://github.com/mlovelace112-maker/FoocusRX/releases).
2. Unzip it. Inside you get:
   - `Install FoocusRX.command` — the entry point
   - `README.txt` — quick primer
   - `FoocusRX-<version>-src.tar.gz` — bundled source snapshot
3. Double-click **Install FoocusRX.command**.
   - If Gatekeeper blocks it: right-click → **Open** → **Open**, or go to
     **System Settings → Privacy & Security** and click **Open Anyway**.
4. A Terminal window runs the installer end-to-end: prerequisite check,
   source extraction, virtualenv creation, `pip install`, and finally the
   FoocusRX UI itself.

Re-running the installer is idempotent — it reuses the venv, runs
`git pull --ff-only` where possible, and reinstalls requirements without
duplicating downloads.

### What the installer needs

- macOS with Xcode Command Line Tools (`xcode-select --install`)
- **Python 3.10, 3.11, or 3.12** (see version note below)
- ~15 GB free disk for the venv and default models

### Python version note (why 3.12 max)

Several of our pinned scientific packages (`scipy 1.14.0`, `numpy 1.26.4`,
`tokenizers 0.19.1`, `safetensors 0.4.3`, `pyyaml 6.0.1`) don't publish
Python 3.13 wheels for macOS arm64 yet. On 3.13, pip would try to build
scipy / numpy from source and fail on Meson.

Un-pinning those packages isn't a fix because they cascade into numpy 2.x,
and torch 2.5.x for MPS (which we install) is ABI-incompatible with numpy 2.

Until that ecosystem-wide bump happens, the installer detects
`python3.12` / `python3.11` / `python3.10` (in that priority order) and
refuses to run on 3.13+. If nothing matches and Homebrew is available,
the installer offers to run `brew install python@3.12` for you.

Manual override: install any 3.12 interpreter and re-run the installer
— it will pick up the versioned binary on `PATH`.

### First-launch behavior on Apple Silicon

On M4 or newer, the launcher auto-installs
[`mtlflashattn`](../docs/apple-silicon.md) for fp32-accumulation attention
correctness. A marker file `.mtlflashattn_install_attempted` is written
before pip runs, so a failed install never retries in a loop — delete the
marker to retry. Opt out entirely with:

```bash
FOOOCUS_AUTO_INSTALL_MTLFLASHATTN=0 ./scripts/mac_launch.sh
```

## For developers building the installer

The installer bundle is produced by `scripts/build_mac_installer.sh`,
which runs from any macOS or Linux dev box (only needs `git`, `python3`,
and `zip`).

```bash
# From the repo root:
scripts/build_mac_installer.sh
#   → dist/FoocusRX-<version>-macOS.zip
```

The script:

1. Reads `fooocus_version.py` for the version tag.
2. Runs `git archive HEAD` to snapshot the current commit into a
   `FoocusRX-<ver>-src.tar.gz` — no `.git`, no ignored files.
3. Emits a thin `Install FoocusRX.command` wrapper that extracts the
   tarball and hands off to `scripts/install_mac.command`.
4. Zips everything into `dist/FoocusRX-<ver>-macOS.zip` and prints the
   SHA-256 for the release notes.

The wrapper is intentionally small: all real logic lives in
`scripts/install_mac.command`, which is versioned inside the repo and
therefore ships inside the tarball. That means users always run the
installer logic that matches the source they're about to install — no
skew between wrapper and repo.

## Design notes

- **No `sudo`, no `.pkg`**: an unsigned `.pkg` triggers Gatekeeper's
  "unidentified developer" flow more aggressively than a `.command`
  file. Keeping the installer as a plain script also means no Apple
  Developer ID is required to ship it. Users still see one Gatekeeper
  prompt for the `.command` file on first launch.
- **Venv, not conda**: matches the existing Linux and Windows install
  paths, and avoids assuming the user has Miniconda.
- **Source tarball, not `git clone`**: makes the release reproducible
  and installs work on airgapped machines.
- **Idempotent**: re-running the installer over an existing checkout is
  the supported update path. It's cheaper than teaching users to run
  `git pull` themselves.
