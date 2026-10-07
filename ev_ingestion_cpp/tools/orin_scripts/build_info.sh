#!/usr/bin/env bash
# Read-only: inspect existing build configuration on the Orin.
set -u
grep -E 'CMAKE_BUILD_TYPE|CMAKE_CXX_FLAGS:|CMAKE_CUDA_FLAGS:' "$HOME/ev_deploy/build/CMakeCache.txt"
ls -la "$HOME/ev_deploy/build" | head -30
nproc
df -h "$HOME" | tail -1
