#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
date -u +%FT%TZ > build-started.txt
timeout --signal=TERM --kill-after=60s 7200 bash mini/pytrio_backend/pace_cloud_build.sh > build.log 2>&1
build_status=$?
printf '%s\n' "$build_status" > build-exit-code.txt
date -u +%FT%TZ > build-finished.txt
exit "$build_status"
