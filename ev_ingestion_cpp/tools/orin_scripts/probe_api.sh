#!/usr/bin/env bash
# Print the OpenEB 5.2 API surface needed by bench_ingest / GPU EVT3 parity tests.
set -uo pipefail
H=/usr/local/include/metavision
echo "--- I_EventDecoder"
grep -n -e 'class I_EventDecoder' -e 'add_event_buffer_callback' -e 'I_EventDecoder(' -e 'using EventBufferCallback' $H/hal/facilities/i_event_decoder.h
echo "--- I_EventsStreamDecoder"
grep -n -e 'void decode' -e 'using RawData' -e 'get_raw_event_size_bytes' -e 'I_EventsStreamDecoder(' $H/hal/facilities/i_events_stream_decoder.h | head
echo "--- PeriodicFrameGenerationAlgorithm"
grep -rn -e 'PeriodicFrameGenerationAlgorithm(' -e 'process_events' $H/sdk/core/algorithms/periodic_frame_generation_algorithm.h | head -5
echo "--- DataTransfer buffer"
grep -rn -e 'using BufferPtr' -e 'class BufferPtr' -e 'using Data ' $H/hal/utils/data_transfer.h 2>/dev/null | head
ls $H/hal/utils/ | head -30
