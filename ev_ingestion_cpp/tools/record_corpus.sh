#!/usr/bin/env bash
# ==============================================================================
# record_corpus.sh - Field Recording Automation Tool (Phase 33.8c)
#
# Automates the capture of ground-truth neuromorphic datasets on Jetson Orin Nano:
# 1. Gracefully stops the background predator-camera service to release USB IMX636.
# 2. Executes evt21_capture to record raw EVT 2.1 (.evt21raw) and decoded CD (.cd).
# 3. Generates structured flight metadata JSON for replay_harness batch scoring.
# 4. Restores predator-camera service on completion or abnormal exit.
#
# Usage:
#   record_corpus.sh <name_or_prefix> [duration_s] [standoff_ft] [platform] [blade_count] [expected_bpf_hz] [ground_truth]
#
# Examples:
#   # 10s capture of DJI Mini 2 at 60ft standoff (hover):
#   record_corpus.sh dji_mini_60ft 10 60 "DJI Mini 2" 2 240.0 true
#
#   # 15s darkroom negative control:
#   record_corpus.sh darkroom_control 15 0 "None" 0 0.0 false
#
#   # 20s foliage clutter negative control:
#   record_corpus.sh foliage_wind_negative 20 0 "None" 0 0.0 false
# ==============================================================================

set -uo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <name_or_prefix> [duration_s] [standoff_ft] [platform] [blade_count] [expected_bpf_hz] [ground_truth]"
    echo "  name_or_prefix  : Dataset label or path (e.g. dji_mini_60ft or /path/to/capture)"
    echo "  duration_s      : Capture duration in seconds (default: 10)"
    echo "  standoff_ft     : Standoff distance in feet (default: 60)"
    echo "  platform        : Target drone model name (default: 'DJI Mini 2')"
    echo "  blade_count     : Number of blades per rotor (default: 2)"
    echo "  expected_bpf_hz : Expected fundamental BPF in Hz (default: 240.0)"
    echo "  ground_truth    : Target present boolean 'true' or 'false' (default: true)"
    exit 2
fi

NAME_INPUT="$1"
DURATION_S="${2:-10}"
STANDOFF_FT="${3:-60}"
PLATFORM="${4:-DJI Mini 2}"
BLADE_COUNT="${5:-2}"
EXPECTED_BPF="${6:-240.0}"
GROUND_TRUTH="${7:-true}"

# Resolve prefix and ensure directory exists
CORPUS_DIR="${HOME}/ev_deploy/corpus"
BIN_DIR="${HOME}/ev_deploy/bin"

if [[ "$NAME_INPUT" == /* ]]; then
    PREFIX="$NAME_INPUT"
else
    mkdir -p "$CORPUS_DIR"
    PREFIX="${CORPUS_DIR}/${NAME_INPUT}"
fi

PREFIX_DIR="$(dirname "$PREFIX")"
mkdir -p "$PREFIX_DIR"
STEM="$(basename "$PREFIX")"

CAPTURE_BIN="${BIN_DIR}/evt21_capture"
if [ ! -x "$CAPTURE_BIN" ]; then
    CAPTURE_BIN="${HOME}/ev_deploy/build/evt21_capture"
fi

if [ ! -x "$CAPTURE_BIN" ]; then
    echo "[ERROR] evt21_capture binary not found at $CAPTURE_BIN"
    exit 3
fi

# Ensure camera service is restored upon exit (success, error, or Ctrl+C)
restore_service() {
    local exit_code=$?
    echo ""
    echo "[INFO] Restoring predator-camera.service..."
    sudo -n systemctl start predator-camera.service || echo "[WARN] Failed to restart predator-camera.service"
    if [ $exit_code -eq 0 ]; then
        echo "[INFO] Corpus recording completed successfully."
    else
        echo "[ERROR] Corpus recording exited with code $exit_code"
    fi
}
trap restore_service EXIT

echo "======================================================================"
echo " PREDATOR FIELD RECORDING AUTOMATION (Phase 33.8c)"
echo "======================================================================"
echo " Prefix          : ${PREFIX}"
echo " Duration        : ${DURATION_S} seconds"
echo " Standoff        : ${STANDOFF_FT} ft"
echo " Platform        : ${PLATFORM}"
echo " Blade Count     : ${BLADE_COUNT}"
echo " Expected BPF    : ${EXPECTED_BPF} Hz"
echo " Ground Truth    : ${GROUND_TRUTH}"
echo "======================================================================"

# Gracefully stop camera service
echo "[INFO] Stopping predator-camera.service..."
sudo -n systemctl stop predator-camera.service
sleep 2

# Execute raw EVT2.1 capture
echo "[INFO] Commencing raw IMX636 hardware capture (${DURATION_S}s)..."
TIMESTAMP_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

"$CAPTURE_BIN" "$PREFIX" "$DURATION_S"
CAPTURE_RET=$?

if [ $CAPTURE_RET -ne 0 ]; then
    echo "[ERROR] evt21_capture failed with return code $CAPTURE_RET"
    exit $CAPTURE_RET
fi

# Generate structured JSON flight metadata
META_JSON="${PREFIX}.json"
cat > "$META_JSON" <<EOF
{
  "name": "${STEM}",
  "timestamp_utc": "${TIMESTAMP_UTC}",
  "platform": "${PLATFORM}",
  "standoff_ft": ${STANDOFF_FT},
  "blade_count": ${BLADE_COUNT},
  "expected_bpf_hz": ${EXPECTED_BPF},
  "ground_truth_target": ${GROUND_TRUTH},
  "duration_s": ${DURATION_S},
  "target_start_s": 0.0,
  "target_end_s": ${DURATION_S},
  "raw_file": "${STEM}.evt21raw",
  "cd_file": "${STEM}.cd"
}
EOF

echo ""
echo "[INFO] Generated flight metadata: ${META_JSON}"
cat "$META_JSON"
echo ""
echo "[INFO] Output files:"
ls -lh "${PREFIX}".*
