#!/usr/bin/env bash
set -euo pipefail
cd /home/grepseek/grepseek
mkdir -p data/wiki_18_corpus
archive=data/wiki_18_corpus/wiki-18.jsonl.gz
# Metadata is from the official Hugging Face API. This is a transport mirror;
# prepare_corpus checks the official LFS digest before extracting anything.
if [[ ! -f "$archive" ]] || [[ $(stat -c %s "$archive") != 5123307260 ]]; then
  curl --fail --location --continue-at - --retry 5 --retry-delay 3 \
    --connect-timeout 30 --output "$archive" \
    'https://hf-mirror.com/datasets/PeterJinGo/wiki-18-corpus/resolve/69c1c00ffe7c5554c68d8548355cb22e46aabc51/wiki-18.jsonl.gz?download=true'
fi
python3 mini/environment/prepare_corpus.py --metadata mini/environment/corpus-source.json --archive "$archive"
