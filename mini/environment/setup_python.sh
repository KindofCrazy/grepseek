#!/usr/bin/env bash
# Run as the unprivileged grepseek user, from either the Windows or Linux copy.
set -euo pipefail
source_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
target_repo=/home/grepseek/grepseek
if [[ $(id -un) != grepseek ]]; then
  echo 'Run this script as the grepseek WSL user.' >&2
  exit 1
fi
if [[ "$source_repo" != "$target_repo" ]]; then
  rsync -a --exclude='.git/' --exclude='.wsl/' --exclude='.venv-mini/' \
    --exclude='.venv-pytrio/' --exclude='.venv/' --exclude='__pycache__/' \
    --exclude='.pytest_cache/' --exclude='/data/' "$source_repo/" "$target_repo/"
fi
cd "$target_repo"
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check \
  --index-url "${PIP_INDEX_URL:-https://mirrors.cernet.edu.cn/pypi/web/simple}" \
  -r mini/environment/requirements.txt
mkdir -p mini/reports/environment
.venv/bin/python -m pip freeze > mini/reports/environment/python-freeze.txt
MINI_TOKENIZER=mini/.cache/tokenizer .venv/bin/python -m pytest mini/tests -q -rs \
  --junitxml=mini/reports/environment/pytest.xml
echo "Python environment ready: $target_repo/.venv/bin/python"
