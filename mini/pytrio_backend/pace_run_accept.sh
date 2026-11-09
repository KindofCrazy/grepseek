#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
while [[ ! -f build-exit-code.txt ]]; do sleep 5; done
if [[ $(cat build-exit-code.txt) != 0 ]]; then
  echo 125 > cloud-accept-exit-code.txt
  exit 125
fi
date -u +%FT%TZ > cloud-accept-started.txt
timeout --signal=TERM --kill-after=30s 1200 .venv-pace/bin/python \
  -m mini.pytrio_backend.pace_cloud_accept --commands pace-replay-commands.json \
  >cloud-accept.log 2>&1
accept_status=$?
printf '%s\n' "$accept_status" > cloud-accept-exit-code.txt
date -u +%FT%TZ > cloud-accept-finished.txt
exit "$accept_status"
