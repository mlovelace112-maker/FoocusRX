#!/usr/bin/env bash
# ------------------------------------------------------------------
# scripts/mac_launch.sh
#
# Launch FoocusRX on Apple Silicon while working around Gradio's
# "When localhost is not accessible, a shareable link must be created"
# false-positive.
#
# Root cause: Gradio's startup health probe hits its own bound URL via
# httpx, which respects HTTP_PROXY / HTTPS_PROXY. If the user has a
# proxy configured (corporate VPN, Charles/Proxyman, etc.) and no
# matching NO_PROXY entry for loopback, httpx routes 127.0.0.1 through
# the proxy, the proxy refuses, and Gradio aborts. See
# https://github.com/gradio-app/gradio/issues/4046.
#
# This wrapper:
#   * Clears the six standard proxy env vars in the launched process.
#   * Exports NO_PROXY / no_proxy including localhost, 127.0.0.1, ::1.
#   * Passes through any extra args to entry_with_update.py.
#
# Usage:
#   ./scripts/mac_launch.sh
#   ./scripts/mac_launch.sh --preset anime
#   ./scripts/mac_launch.sh --auto-update
# ------------------------------------------------------------------

set -euo pipefail

# Clear inherited proxies (only for this process; the user's shell is unchanged).
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

# Belt-and-braces loopback bypass — some libraries read NO_PROXY, some no_proxy.
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

# Recommend --disable-offload-from-vram on Mac (the upstream README calls it out).
# Users can override by passing their own flags after this wrapper.
exec python entry_with_update.py --disable-offload-from-vram "$@"
