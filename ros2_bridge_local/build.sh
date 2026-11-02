#!/usr/bin/env bash
# Build the LD_PRELOAD shim that caps the beamng_ros2 bridge's scalar publish
# rate, so the ORIGINAL upstream .so stays in the game (no patched library).
#
#   ./build.sh
#
# The shim is exported via drift-ai/env-drift-tools (LD_PRELOAD); the game
# inherits it from the launching process.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[build] building LD_PRELOAD shim"
cc -O2 -shared -fPIC -Wall -Wextra \
    -o "${HERE}/preload/libbng_ros2_preload.so" \
    "${HERE}/preload/bng_ros2_preload.c" -ldl
echo "[build] shim: ${HERE}/preload/libbng_ros2_preload.so"
echo "[build] export it via drift-ai/env-drift-tools (LD_PRELOAD)"
echo "[build] done"
