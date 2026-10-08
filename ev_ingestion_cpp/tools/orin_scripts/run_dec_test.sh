#!/bin/bash
# Run the GPU EVT2.1 decoder parity test. Usage: run_dec_test.sh [--fixture <prefix> [--batch N]]
cd ~/ev_deploy/build || exit 1
/usr/bin/time -v ./test_evt21_decoder "$@" > /tmp/dec_test.log 2>&1
rc=$?
grep -v 'TimeHigh discrepancy' /tmp/dec_test.log | grep -E 'TEST|INFO|mismatch|count|ERROR|Elapsed|Maximum resident|exception'
echo "discrepancy log lines: $(grep -c 'TimeHigh discrepancy' /tmp/dec_test.log)"
echo "rc=$rc"
