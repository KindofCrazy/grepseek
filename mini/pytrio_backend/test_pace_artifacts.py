import json
from pathlib import Path

import pytest

from mini.pytrio_backend import pace_artifacts as artifacts


def test_low_memory_build_does_not_touch_output(tmp_path,monkeypatch):
    monkeypatch.setattr(artifacts,'available_memory',lambda:2*1024**3)
    with pytest.raises(RuntimeError,match='available RAM'):
        artifacts.build(tmp_path/'not-downloaded',tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_verified_binding_and_data_corruption(tmp_path,monkeypatch):
    corpus = tmp_path/'corpus.jsonl'
    corpus.write_text('fixture corpus\n')
    monkeypatch.setattr(artifacts,'CORPUS_SHA256',artifacts.digest(corpus))
    index = tmp_path/'index'
    index.mkdir()
    (index/'data.bin').write_bytes(b'fixture binary')
    metadata = {'corpus_path':'/old-machine/corpus.jsonl','corpus_mtime':0,
                'corpus_size':corpus.stat().st_size,'format_version':4,'num_lines':1}
    (index/'meta.json').write_text(json.dumps(metadata))
    manifest = {'corpus_sha256':artifacts.digest(corpus),'metadata_filename':'meta.json',
                'original_metadata':metadata,'data_sha256':{'data.bin':artifacts.digest(index/'data.bin')}}
    (index/'artifact_manifest.json').write_text(json.dumps(manifest))
    artifacts.bind(index,corpus)
    assert json.loads((index/'meta.json').read_text())['corpus_path']==str(corpus.resolve())
    artifacts.verify(index,corpus)
    (index/'data.bin').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='differs from'):
        artifacts.verify(index,corpus)


def test_structural_metadata_and_wrong_corpus_rejected(tmp_path,monkeypatch):
    corpus = tmp_path/'corpus'
    corpus.write_bytes(b'expected')
    monkeypatch.setattr(artifacts,'CORPUS_SHA256',artifacts.digest(corpus))
    (tmp_path/'meta.json').write_text(json.dumps({'num_lines':999}))
    (tmp_path/'artifact_manifest.json').write_text(json.dumps({
        'corpus_sha256':artifacts.digest(corpus),'metadata_filename':'meta.json',
        'original_metadata':{'num_lines':1},'data_sha256':{}}))
    with pytest.raises(ValueError,match='structurally'):
        artifacts.verify(tmp_path,corpus)
    corpus.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='Wikipedia snapshot'):
        artifacts.verify(tmp_path,corpus)
