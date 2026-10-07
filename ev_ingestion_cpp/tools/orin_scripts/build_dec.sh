#!/bin/bash
# Reconfigure and build the 33.4b.b decoder test plus the targets touched by the gpu_event.hpp move.
set -o pipefail
cd ~/ev_deploy/build || exit 1
cmake -S ~/ev_deploy/src -B ~/ev_deploy/build > /tmp/cmake_cfg.log 2>&1 || { tail -30 /tmp/cmake_cfg.log; exit 1; }
for t in test_evt21_decoder test_cuda_flicker ev_flicker_detector bench_ingest; do
  echo "=== building $t"
  cmake --build . --target "$t" -j6 2>&1 | grep -E "error|warning|Error|Built target" | head -40
  rc=${PIPESTATUS[0]}
  echo "=== $t rc=$rc"
done
