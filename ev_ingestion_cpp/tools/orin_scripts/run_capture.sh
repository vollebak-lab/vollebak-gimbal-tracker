#!/usr/bin/env bash
# Builds and runs evt21_capture with predator-camera.service stopped; always restarts the service.
# Usage: bash run_capture.sh <seconds> [prefix=~/ev_deploy/fixtures/darkroom_evt21]
set -uo pipefail
secs="${1:-10}"
prefix="${2:-$HOME/ev_deploy/fixtures/darkroom_evt21}"
mkdir -p "$(dirname "$prefix")"
cd ~/ev_deploy/build || exit 3
cmake -S ~/ev_deploy/src -B ~/ev_deploy/build > /tmp/cfg.log 2>&1 || { tail -20 /tmp/cfg.log; exit 3; }
cmake --build . --target evt21_capture -j6 2>&1 | grep -i -e 'error' -e 'warning' | head -20
[ -x ./evt21_capture ] || { echo "build failed"; exit 3; }
trap 'sudo -n systemctl start predator-camera.service' EXIT
sudo -n systemctl stop predator-camera.service
sleep 2
./evt21_capture "$prefix" "$secs"
echo "capture exit=$?"
ls -la "$prefix".*
