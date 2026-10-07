#!/usr/bin/env bash
set -euo pipefail

finish() {
    echo "[INFO] Cleaning up test process and restarting predator-camera.service..."
    if [ -n "${TEST_PID:-}" ] && kill -0 "$TEST_PID" 2>/dev/null; then
        kill "$TEST_PID" 2>/dev/null || true
        wait "$TEST_PID" 2>/dev/null || true
    fi
    sudo systemctl start predator-camera.service
    echo "[INFO] predator-camera.service restarted."
}
trap finish EXIT

echo "[INFO] Stopping predator-camera.service..."
sudo systemctl stop predator-camera.service
sleep 1

echo "[INFO] Launching ev_flicker_detector as child process..."
/home/orin/ev_deploy/bin/ev_flicker_detector 8080 > /tmp/ev_diag.log 2>&1 &
TEST_PID=$!
echo "[INFO] Child PID: $TEST_PID"

echo "[INFO] Waiting 6 seconds for pipeline threads to initialize..."
sleep 6

echo "=== Child Process Threads (ps -T) ==="
ps -T -p "$TEST_PID" -o pid,tid,comm,%cpu,time | sort -k4 -nr

echo ""
echo "=== Top 2-second Sample ==="
top -b -n 2 -d 1 -H -p "$TEST_PID" | tail -n +7 | head -n 15 || true

echo ""
echo "=== GDB Backtrace of All Child Threads ==="
gdb -batch -ex "thread apply all bt 10" -p "$TEST_PID" || echo "[WARN] GDB attach failed"

echo ""
echo "=== Tail of ev_flicker_detector Log ==="
head -n 25 /tmp/ev_diag.log
