#!/usr/bin/env bash
set -euo pipefail

PID=$(pgrep -f ev_flicker_detector | head -n 1 || true)
if [ -z "$PID" ]; then
    echo "ev_flicker_detector not running"
    exit 1
fi

echo "=== Target PID: $PID ==="
echo "=== Overall Process Status ==="
ps -p "$PID" -o pid,user,%cpu,%mem,etime,cmd

echo ""
echo "=== Thread CPU Usage (ps -T) ==="
ps -T -p "$PID" -o pid,tid,comm,%cpu,time | sort -k4 -nr

echo ""
echo "=== Task Names and TIDs in /proc/$PID/task/ ==="
for t in /proc/"$PID"/task/*; do
    tid=$(basename "$t")
    comm=$(cat "$t/comm" 2>/dev/null || echo "unknown")
    stat=$(cat "$t/stat" 2>/dev/null | awk '{print $14, $15}' || echo "0 0")
    echo "TID $tid: comm='$comm' (utime stime: $stat)"
done

echo ""
echo "=== Sampling top threads for 3 seconds ==="
top -b -n 2 -d 1.5 -H -p "$PID" | tail -n +7 | head -n 20 || true

echo ""
echo "=== Telemetry /pipeline_stats ui_encoder ==="
python3 - << 'EOF'
import urllib.request, json
try:
    with urllib.request.urlopen("http://127.0.0.1:8080/pipeline_stats", timeout=3) as resp:
        data = json.loads(resp.read().decode())
        print(json.dumps(data.get("ui_encoder", {}), indent=2))
        print("Total tracks:", len(data.get("tracks", [])))
        print("Build ID:", data.get("build_id"))
except Exception as e:
    print("Telemetry query failed:", e)
EOF

