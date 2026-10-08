#!/usr/bin/env bash
# Runs bench_ingest with predator-camera.service stopped (exclusive GPU, quiet CPU); always restarts it.
set -uo pipefail
trap 'sudo -n systemctl start predator-camera.service' EXIT
sudo -n systemctl stop predator-camera.service
sleep 2
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) cur_khz: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)"
~/ev_deploy/build/bench_ingest "$@"
echo "bench exit=$?"
