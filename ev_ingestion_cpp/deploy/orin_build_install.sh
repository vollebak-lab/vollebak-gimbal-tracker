#!/usr/bin/env bash
# =============================================================================
# orin_build_install.sh — Predator detector build/test/install on the Jetson Orin.
#
# Single entry point for deployment (Phase 33.1, fixes the "deploy drift" class
# of defects where the running binary was built from a stale header).
#
# Contract:
#   * Sources are expected in  $DEPLOY_ROOT/src  (synced by deploy.ps1 from the
#     repo's ev_ingestion_cpp/ directory — the ONLY source of truth).
#   * Build ID = <git rev>-src<sha256 prefix of all build inputs>; compiled into
#     the binary (banner + /pipeline_stats "build_id").
#   * Unit tests must pass before install. Install keeps one rollback copy.
#   * Service restart uses `sudo -n` (requires a narrow NOPASSWD sudoers rule;
#     no password is ever passed by this script). If unavailable, the script
#     installs and prints the manual restart command.
#
# Usage:  orin_build_install.sh <git_rev> [--no-restart]
# Exit codes: 0 ok, 2 bad args, 3 build failed, 4 tests failed, 5 restart/verify failed
# =============================================================================
set -euo pipefail

DEPLOY_ROOT="${PREDATOR_DEPLOY_ROOT:-$HOME/ev_deploy}"
SRC_DIR="$DEPLOY_ROOT/src"
BUILD_DIR="$DEPLOY_ROOT/build"
BIN_DIR="$DEPLOY_ROOT/bin"
SERVICE="predator-camera.service"
STATS_URL="http://127.0.0.1:8080/pipeline_stats"

log()  { printf '[deploy %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
fail() { local code="$1"; shift; log "ERROR: $*"; exit "$code"; }

# ---- Arguments (validated: rev is interpolated into a compile definition) ----
GIT_REV="${1:-}"
RESTART=1
[[ "${2:-}" == "--no-restart" ]] && RESTART=0
[[ "$GIT_REV" =~ ^[A-Za-z0-9._+-]{1,64}$ ]] || fail 2 "invalid or missing git rev '$GIT_REV'"
[[ -f "$SRC_DIR/CMakeLists.txt" ]] || fail 2 "no CMakeLists.txt in $SRC_DIR"

# ---- Build ID: hash of every build input (sorted, so order-independent) ----
SRC_HASH="$(cd "$SRC_DIR" && find . -maxdepth 1 -type f \
    \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' -o -name 'CMakeLists.txt' \) \
    -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)"
BUILD_ID="${GIT_REV}-src${SRC_HASH}"
log "Build ID: $BUILD_ID"

# ---- Configure + build (Release, same as historical cache) ----
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DPREDATOR_BUILD_ID="$BUILD_ID" >/dev/null \
    || fail 3 "cmake configure failed"
cmake --build "$BUILD_DIR" -j"$(nproc)" --target ev_flicker_detector test_flicker_dsp test_ego_motion test_cuda_flicker \
    test_evt21_decoder test_hot_pixel_mask test_gpu_sieve test_raw_pipeline dump_spectrum extract_real_spectra || fail 3 "build failed"



# ---- Unit tests (CPU DSP, ego-motion math, CUDA core, GPU EVT2.1 decoder, hot pixel mask, GPU sieve, raw pipeline). Must all pass. ----
for t in test_flicker_dsp test_ego_motion test_cuda_flicker test_evt21_decoder test_hot_pixel_mask test_gpu_sieve test_raw_pipeline; do
    log "Running $t"
    if ! "$BUILD_DIR/$t" > "$BUILD_DIR/$t.log" 2>&1; then
        tail -n 40 "$BUILD_DIR/$t.log"
        fail 4 "$t failed (full log: $BUILD_DIR/$t.log)"
    fi
    tail -n 3 "$BUILD_DIR/$t.log"
done

# ---- Install with single rollback copy ----
mkdir -p "$BIN_DIR"
if [[ -f "$BIN_DIR/ev_flicker_detector" ]]; then
    cp -p "$BIN_DIR/ev_flicker_detector" "$BIN_DIR/ev_flicker_detector.old"
fi
install -m 0775 "$BUILD_DIR/ev_flicker_detector" "$BIN_DIR/ev_flicker_detector.new"
mv -f "$BIN_DIR/ev_flicker_detector.new" "$BIN_DIR/ev_flicker_detector"   # atomic replace
log "Installed $BIN_DIR/ev_flicker_detector (rollback: ev_flicker_detector.old)"

if [[ "$RESTART" -eq 0 ]]; then
    log "Skipping restart (--no-restart)."
    exit 0
fi

# ---- Restart (non-interactive sudo only) and verify the live build ID ----
if ! sudo -n systemctl restart "$SERVICE" 2>/dev/null; then
    log "Passwordless restart not permitted. Run manually:  sudo systemctl restart $SERVICE"
    exit 5
fi
for _ in $(seq 1 30); do
    sleep 1
    if live="$(curl -s --max-time 2 "$STATS_URL")" && [[ "$live" == *"\"build_id\": \"$BUILD_ID\""* ]]; then
        log "Live service reports build_id=$BUILD_ID (PID $(systemctl show -p MainPID --value "$SERVICE"))"
        exit 0
    fi
done
fail 5 "service did not report build_id=$BUILD_ID within 30 s (check: journalctl -u $SERVICE -n 50)"
