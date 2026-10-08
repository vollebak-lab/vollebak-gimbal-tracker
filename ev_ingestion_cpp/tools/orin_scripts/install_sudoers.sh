#!/usr/bin/env bash
# Validates and installs the predator-deploy sudoers drop-in. Must run as root (via sudo -S).
set -euo pipefail
src="$1"
dst=/etc/sudoers.d/predator-deploy
tr -d '\r' < "$src" > "$src.lf"
visudo -cf "$src.lf"
install -m 0440 -o root -g root "$src.lf" "$dst"
visudo -c
rm -f "$src" "$src.lf"
echo "installed $dst"
