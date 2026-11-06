#!/usr/bin/env bash
# Use the official runtime with the local source tree mounted over /workspace.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p .cache
exec docker run --rm --gpus "${DOCKER_GPUS:-all}" --ipc=host \
  -v "${PWD}:/workspace" -v "${PWD}/.cache:/cache" \
  -e HF_HOME=/cache -e HF_TOKEN -w /workspace \
  --entrypoint /bin/bash "${GREPSEEK_IMAGE:-ghcr.io/alirezasalemi7/grepseek:v1-slim}" \
  -lc 'export PATH=/opt/envs/grepseek/bin:$PATH; exec "$@"' _ "$@"
