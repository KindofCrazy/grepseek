"""Small native-library unit fixtures; full-corpus acceptance is a separate replay."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from mini.pytrio_backend import pace_artifacts, pace_search


@pytest.fixture(scope='module')
def native():
    if os.name!='posix' or not Path('inference/pace/_native/materialize.so').exists():
        pytest.skip('Compile the PACE native library in Ubuntu first')
    from inference.pace.core import build_pace,META_FILENAME
    return build_pace,META_FILENAME


@pytest.fixture
def fixture_index(tmp_path,monkeypatch,native):
    corpus = tmp_path/'wiki_corpus.jsonl'
    corpus.write_text('alpha first\nbeta second\nALPHA third\nalpha fourth\n',encoding='utf-8')
    expected_hash = pace_artifacts.digest(corpus)
    monkeypatch.setattr(pace_artifacts,'CORPUS_SHA256',expected_hash)
    monkeypatch.setattr(pace_search,'CORPUS_SHA256',expected_hash)
    build,meta_name = native
    index = tmp_path/'pace'
    metadata = build(corpus,index,progress=False)
    manifest = {'corpus_sha256':expected_hash,'metadata_filename':meta_name,
                'original_metadata':json.loads(metadata.to_json()),
                'data_sha256':{p.name:pace_artifacts.digest(p) for p in index.iterdir() if p.name!=meta_name}}
    (index/'artifact_manifest.json').write_text(json.dumps(manifest))
    pace_artifacts.bind(index,corpus)
    return pace_search.PaceSearch(index,corpus),corpus


@pytest.mark.parametrize('command',[
    'rg -F "alpha" corpus.jsonl | head -n 2',
    'rg -iF "alpha" corpus.jsonl | wc -l',
    'rg -F "absent" corpus.jsonl | head -n 3',
    'rg -iF "alpha" corpus.jsonl | grep -F "third" | head -n 3',
])
def test_native_pace_matches_real_pipeline(fixture_index,command):
    search,corpus = fixture_index
    original = subprocess.run(command.replace('corpus.jsonl','wiki_corpus.jsonl'),shell=True,
                              executable='/bin/bash',cwd=corpus.parent,capture_output=True)
    result = search.execute(command,1048576)
    assert result is not None
    assert result==(original.stdout,original.returncode)


def test_unsupported_and_large_capture_fall_back(fixture_index):
    search,_ = fixture_index
    assert search.execute('rg -nF "alpha" corpus.jsonl | head -n 3',1048576) is None
    assert search.execute('rg -F "alpha" corpus.jsonl | head -n 3',2) is None
    assert search.execute('rg -F "`whoami`" corpus.jsonl | head -n 3',1048576) is None
    assert search.execute('rg -F "$HOME" corpus.jsonl | head -n 3',1048576) is None


def test_expanded_head_budget_keeps_count_budget_and_native_semantics(fixture_index):
    from dataclasses import replace
    search,corpus=fixture_index
    search.base_budget=replace(search.base_budget,max_candidate_lines=1)
    search.head_budget=replace(search.base_budget,max_candidate_lines=100)
    command='rg -F "alpha" corpus.jsonl | head -n 2'
    expected=subprocess.run(command.replace('corpus.jsonl','wiki_corpus.jsonl'),shell=True,
                            executable='/bin/bash',cwd=corpus.parent,capture_output=True)
    assert search.execute(command,1048576)==(expected.stdout,expected.returncode)
    assert search.execute('rg -F "alpha" corpus.jsonl | wc -l',1048576) is None
