#!/usr/bin/env bash
# One-time (Phase 33.1): move stale top-level source duplicates out of ~/ev_deploy.
# Non-destructive: files are moved to an attic directory, not deleted.
set -euo pipefail
cd "$HOME/ev_deploy"
ATTIC="attic_phase33_$(date -u +%Y%m%d)"
mkdir -p "$ATTIC"
shopt -s nullglob
moved=0
for f in *.cpp *.hpp *.cu *.cuh CMakeLists.txt; do
    mv -v "$f" "$ATTIC/"
    moved=$((moved + 1))
done
echo "Moved $moved stale files to ~/ev_deploy/$ATTIC"
