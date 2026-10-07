#!/usr/bin/env bash
# Read-only inventory of Metavision/OpenEB tools relevant to lens focusing.
set -u
echo "== binaries =="
ls /usr/bin /usr/local/bin 2>/dev/null | grep -i -E 'metavision|focus|blink' || echo "none"
echo "== packages =="
dpkg -l 2>/dev/null | grep -i -E 'metavision|openeb' | awk '{print $2, $3}' || echo "none"
echo "== source trees =="
find "$HOME" -maxdepth 3 -type d -iname '*openeb*' 2>/dev/null
