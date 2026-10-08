#!/usr/bin/env bash
set -euo pipefail

prefix="${OPENEB_PREFIX:-${HOME}/metavision-5.2-ids}"
detector="${PREDATOR_EVENT_DETECTOR:-${HOME}/predator-event-build/ev_flicker_detector}"
port="${PREDATOR_EVENT_PORT:-8081}"
serial="${PREDATOR_EVENT_CAMERA_SERIAL:-Prophesee:hal_plugin_prophesee:4110044085}"
event_rate_limit="${PREDATOR_EVENT_RATE_LIMIT:-10000000}"

export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MV_HAL_PLUGIN_PATH="${prefix}/lib/metavision/hal/plugins"
export MV_HAL_PLUGIN_SEARCH_MODE="PLUGIN_PATH_ONLY"
export PREDATOR_EVENT_CAMERA_SERIAL="${serial}"
export PREDATOR_EVENT_RATE_LIMIT="${event_rate_limit}"

exec "${detector}" "${port}" "${serial}"
