#!/usr/bin/env bash
# Read-only: md5 inventory of Predator sources on the Orin (deploy drift audit).
set -u
cd "$HOME/ev_deploy" || exit 1
echo "== src/ =="
( cd src && md5sum *.cpp *.hpp *.cu *.cuh CMakeLists.txt 2>/dev/null )
echo "== top-level =="
md5sum *.cpp *.hpp *.cu *.cuh CMakeLists.txt 2>/dev/null
echo "== build dirs =="
ls -d */ 2>/dev/null
find . -maxdepth 3 -name CMakeCache.txt -exec grep -H "CMAKE_HOME_DIRECTORY" {} \; 2>/dev/null
echo "== binary =="
ls -la bin/ 2>/dev/null
echo "== service =="
systemctl cat predator-camera.service 2>/dev/null | grep -E "ExecStart|Environment|WorkingDirectory"
echo "== models =="
ls -la models/ 2>/dev/null
