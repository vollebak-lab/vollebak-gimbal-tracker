#!/usr/bin/env bash
# Phase 33.5 live false-alarm check: samples /pipeline_stats once per second for N seconds and
# reports how often confirmed targets / tracks appeared. Any non-empty target record is saved raw.
# Usage: bash monitor_fa.sh [seconds=600]   -> summary on stdout, raw hits in ~/fa_hits_<ts>.jsonl
set -uo pipefail
dur="${1:-600}"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
hits="$HOME/fa_hits_${ts}.jsonl"
samples=0; target_samples=0; track_samples=0; max_targets=0; unreachable=0
build=""
for ((i = 0; i < dur; i++)); do
  j="$(curl -s --max-time 1 localhost:8080/pipeline_stats)" || j=""
  if [ -z "$j" ]; then unreachable=$((unreachable + 1)); sleep 1; continue; fi
  samples=$((samples + 1))
  [ -z "$build" ] && build="$(grep -o -e '"build_id": "[^"]*"' <<<"$j")"
  nt="$(grep -o -e '"num_targets": [0-9]*' <<<"$j" | grep -o -e '[0-9]*$')"
  nk="$(grep -o -e '"num_tracks": [0-9]*' <<<"$j" | grep -o -e '[0-9]*$')"
  nt="${nt:-0}"; nk="${nk:-0}"
  if [ "$nt" -gt 0 ]; then
    target_samples=$((target_samples + 1))
    [ "$nt" -gt "$max_targets" ] && max_targets="$nt"
    printf '%s\n' "$j" | tr -d '\n' >> "$hits"; echo >> "$hits"
  fi
  [ "$nk" -gt 0 ] && track_samples=$((track_samples + 1))
  sleep 1
done
echo "window_s=$dur samples=$samples unreachable=$unreachable $build"
echo "samples_with_confirmed_targets=$target_samples max_simultaneous_targets=$max_targets"
echo "samples_with_tentative_or_confirmed_tracks=$track_samples"
echo "raw_hits_file=$( [ -s "$hits" ] && echo "$hits" || echo none)"
