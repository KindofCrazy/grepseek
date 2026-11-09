#!/usr/bin/env bash
# Run in the extracted job root on a temporary 64GB Linux CPU machine.
# Requires python3 >=3.12 with venv, a C compiler, curl and normal build headers.
# Example Ubuntu prerequisite: sudo apt-get install python3-venv build-essential curl
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
from mini.pytrio_backend.pace_artifacts import available_memory
for name,expected in json.loads(Path('job_manifest.json').read_text()).items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Job source hash mismatch: '+name)
if available_memory() < 28*1024**3:
    raise SystemExit('Use a high-memory CPU host (64GB recommended); refusing a low-memory full build.')
PY
python3 -m venv .venv-pace
.venv-pace/bin/python -m pip install \
  --index-url "${PIP_INDEX_URL:-https://mirrors.cernet.edu.cn/pypi/web/simple}" \
  'numpy==2.5.4' 'tqdm==4.70.1'
bash inference/pace/_native/build.sh
mkdir -p data/wiki_18_corpus
archive=data/wiki_18_corpus/wiki-18.jsonl.gz
if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 5 --connect-timeout 30 --output "$archive" \
    'https://hf-mirror.com/datasets/PeterJinGo/wiki-18-corpus/resolve/69c1c00ffe7c5554c68d8548355cb22e46aabc51/wiki-18.jsonl.gz?download=true'
fi
.venv-pace/bin/python mini/environment/prepare_corpus.py \
  --metadata mini/environment/corpus-source.json --archive "$archive"
.venv-pace/bin/python -m mini.pytrio_backend.pace_artifacts build --out data/pace-full
cp job_manifest.json data/pace-full/job_manifest.json
.venv-pace/bin/python -m pip freeze > data/pace-full/build-requirements.txt
tar -czf pace-full.tar.gz -C data pace-full
sha256sum pace-full.tar.gz > pace-full.tar.gz.sha256
echo 'CPU build finished: copy pace-full.tar.gz and its SHA256 file back to the prepared local Ubuntu.'
