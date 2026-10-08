#!/usr/bin/env bash
# Runs hot_pixel_survey with predator-camera.service safely stopped, always restarting it on exit.
# Usage: ./run_survey.sh [options to hot_pixel_survey]
set -uo pipefail

trap 'sudo -n systemctl start predator-camera.service' EXIT
sudo -n systemctl stop predator-camera.service
sleep 2

cd ~/ev_deploy/build || exit 3
./hot_pixel_survey "$@"
rc=$?

echo "survey exit=$rc"
if [ -f hot_pixels.txt ]; then
    echo "=== hot_pixels.txt contents ==="
    cat hot_pixels.txt
    echo "==============================="
    # Copy to deployment dir as well
    cp -f hot_pixels.txt ~/ev_deploy/hot_pixels.txt
fi
exit $rc
