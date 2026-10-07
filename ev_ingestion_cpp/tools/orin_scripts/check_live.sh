#!/usr/bin/env bash
# Live-service health snapshot for predator-camera.service (Phase 33 checks).
# Usage: bash check_live.sh   (run on the Orin)
set -uo pipefail
curl -s --max-time 3 localhost:8080/pipeline_stats > /tmp/ps.json || { echo "pipeline_stats unreachable"; exit 1; }
echo "--- telemetry"
grep -o -e '"build_id"[^,]*' -e '"flags"[^}]*}' -e '"cfar"[^}]*}' -e '"total_raw_events"[^,]*' /tmp/ps.json
echo "candidate_entries=$(grep -o -e '"freq_hz"' /tmp/ps.json | wc -l)"
echo "--- service banner (journal, current boot of unit)"
journalctl -u predator-camera.service --since "-10 min" --no-pager -o cat 2>/dev/null \
  | grep -a -e 'CFAR' -e 'Ingestion Mode' -e 'biases' -e 'Build' -e 'EVT' -e 'format' -e 'ERROR' -e 'WARN' | tail -12
echo "--- process"
pid=$(pgrep -f ev_deploy/bin/ev_flicker_detector | head -1)
echo "pid=${pid:-none}"
[ -n "${pid:-}" ] && ps -o pid,pcpu,rss,nlwp,etime -p "$pid"
