#!/usr/bin/env bash
# Probe the camera's raw event format and the installed OpenEB decoder/link surface.
# Briefly stops predator-camera.service (camera is exclusive), always restarts it on exit.
set -uo pipefail
restart() { sudo -n systemctl start predator-camera.service; }
trap restart EXIT

echo "--- installed EVT3 headers / libs"
ls /usr/local/include/metavision/hal/decoders/evt3/ 2>&1
ls /usr/local/lib/libmetavision_hal*.so* /usr/local/lib/libmetavision_sdk_stream*.so* 2>&1 | head
ls /usr/local/lib/metavision/hal/plugins/ 2>&1 | head

echo "--- platform info (service stopped)"
sudo -n systemctl stop predator-camera.service
sleep 2
if command -v metavision_platform_info >/dev/null; then
  timeout 20 metavision_platform_info 2>&1 | grep -i -e format -e evt -e 'sensor' -e 'plugin' -e 'integrator' -e 'serial' | head -20
else
  echo "metavision_platform_info not installed"
fi
