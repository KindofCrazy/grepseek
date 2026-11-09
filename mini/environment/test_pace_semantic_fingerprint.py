import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mini.environment.pace_semantic_fingerprint import fingerprint


def test_order_independent_fingerprint_and_changed_posting(tmp_path):
    if os.name!='posix':pytest.skip('Linux native acceptance utility')
    corpus=tmp_path/'corpus.jsonl'
    corpus.write_text('alpha beta gamma delta\nalpha beta\nbeta gamma delta\nepsilon zeta eta theta\n')
    indexes=[]
    for seed in (1,2):
        index=tmp_path/str(seed)
        code='from inference.pace.core import build_pace;import sys;build_pace(sys.argv[1],sys.argv[2],progress=False)'
        subprocess.run([sys.executable,'-c',code,str(corpus),str(index)],
                       env={**os.environ,'PYTHONHASHSEED':str(seed)},check=True)
        indexes.append(index)
    source=Path(__file__).with_name('pace_semantic_fingerprint.c')
    binary=tmp_path/'fingerprint'
    subprocess.run(['cc','-O3',str(source),'-ldl','-o',str(binary)],check=True)
    assert (indexes[0]/'vocab.bin').read_bytes()!=(indexes[1]/'vocab.bin').read_bytes()
    assert fingerprint(indexes[0],binary)==fingerprint(indexes[1],binary)
    path=indexes[1]/'singles.bin'
    changed=bytearray(path.read_bytes());changed[:4]=(0).to_bytes(4,'little');path.write_bytes(changed)
    assert fingerprint(indexes[0],binary)!=fingerprint(indexes[1],binary)
