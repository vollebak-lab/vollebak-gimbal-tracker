#!/usr/bin/env bash
# Locate OpenEB 5.2 format-selection and EVT2.1 legacy/non-legacy decoder selection logic.
set -uo pipefail
cd ~/openeb
echo "--- DeviceConfig format API"
grep -n -e 'format' hal/cpp/include/metavision/hal/utils/device_config.h | head -20
echo "--- EVT21 decoder selection"
grep -rn -e 'Legacy' -e 'make_evt21' -e '"EVT21"' -e 'EVT21;' hal/cpp/src hal/cpp/include hal_psee_plugins/ 2>/dev/null | grep -v -e '/test' | head -30
echo "--- format strings in psee plugin"
grep -rn -e 'EVT21' hal_psee_plugins/src 2>/dev/null | grep -v test | head -20
echo "--- raw file writer / record APIs"
grep -rn -e 'log_raw_data' -e 'start_recording' sdk/modules/stream/cpp/include/metavision/sdk/stream/camera.h | head
