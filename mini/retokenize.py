"""Revalidate frozen samples with a different local tokenizer. CPU only.

Preserves selection and evaluation IDs. Does not resample, truncate, execute
commands, load model weights, or call any LLM/training service.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import shutil

from mini.common import ROOT, question_key, read_jsonl, validate_sft, write_jsonl
from mini.curation import token_accounting
from mini.curation_utils import file_hash, json_file, quantiles, records
from mini.template import turn_tokens


def verify_files(directory, hashes):
    directory = Path(directory).resolve()
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'Input hash mismatch or unsafe file: {name}')


def verify_tokenizer(directory, provenance):
    directory = Path(directory)
    hashes = {}
    for name, expected in provenance['files'].items():
        data = (directory / name).read_bytes()
        if len(data) != expected['bytes']:
            raise ValueError(f'Tokenizer size mismatch: {name}')
        if 'sha256' in expected:
            actual = hashlib.sha256(data).hexdigest()
            target = expected['sha256']
        else:
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            target = expected['git_blob_sha1']
        if actual != target:
            raise ValueError(f'Tokenizer differs from official metadata: {name}')
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def measure_sft(tokenizer, row, budget):
    messages = validate_sft(row)
    ids, mask = [], []
    for i, message in enumerate(messages):
        segment, weights = turn_tokens(tokenizer, messages, i)
        if len(segment) != len(weights) or (message['role'] != 'assistant' and any(weights)):
            raise ValueError(f'Invalid loss mask: {row["id"]}, turn {i}')
        if message['role'] == 'assistant' and not any(weights):
            raise ValueError(f'No trainable assistant target: {row["id"]}, turn {i}')
        ids.extend(segment)
        mask.extend(weights)
    full = tokenizer.apply_chat_template(messages, tools=[], enable_thinking=True,
                                        tokenize=True, return_dict=False, add_generation_prompt=False)
    if ids != full or len(ids) > budget:
        raise ValueError(f'Token assembly or context budget failure: {row["id"]} ({len(ids)} tokens)')
    accounting = token_accounting(tokenizer, messages)
    if accounting['trainable_assistant_tokens'] != sum(mask):
        raise ValueError(f'Audit and actual loss mask disagree: {row["id"]}')
    # Exactly the one-token autoregressive shift expected by the future backend.
    target_mask = mask[1:]
    if len(ids[:-1]) != len(ids[1:]) or len(target_mask) != len(ids[:-1]) or not any(target_mask):
        raise ValueError(f'Invalid shifted SFT targets: {row["id"]}')
    return accounting, ids, mask


def initial_prompt(question):
    # This is the same system prompt as inference and the RL prompt asset.
    from inference.agent import build_system_prompt
    return [{'role': 'system', 'content': build_system_prompt()},
            {'role': 'user', 'content': 'Query: ' + question}]


def adapt(source, benchmark, tokenizer_path, old_tokenizer_path, provenance_path, out,
          sft_budget=12288, prompt_budget=1024, response_budget=8192):
    source, benchmark, out = map(Path, (source, benchmark, out))
    if out.exists():
        raise ValueError('Snapshot output must not exist; use a fresh directory')
    if not Path(tokenizer_path).is_dir() or not Path(old_tokenizer_path).is_dir():
        raise ValueError('Both tokenizers must be local directories; network fallback is forbidden')
    parent = json.loads((source / 'sft_manifest.json').read_text(encoding='utf-8'))
    provenance = json.loads(Path(provenance_path).read_text(encoding='utf-8'))
    verify_files(source, parent['files_sha256'])
    old_hashes = {name: file_hash(Path(old_tokenizer_path) / name) for name in parent['tokenizer']}
    if old_hashes != parent['tokenizer']:
        raise ValueError('Original tokenizer differs from the frozen parent manifest')
    tokenizer_hashes = verify_tokenizer(tokenizer_path, provenance)
    benchmark_manifest = json.loads((benchmark / 'benchmark_manifest.json').read_text(encoding='utf-8'))
    benchmark_hash = file_hash(benchmark / 'test.jsonl')
    if benchmark_hash != benchmark_manifest['sample_sha256']:
        raise ValueError('Frozen benchmark hash mismatch')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    old_tokenizer = AutoTokenizer.from_pretrained(old_tokenizer_path, local_files_only=True)
    sft = {name: read_jsonl(source / filename) for name, filename in
           [('train', 'sft_selected.jsonl'), ('val', 'sft_validation.jsonl'),
            ('candidate_300', 'sft_candidate_300.jsonl'), ('candidate_800', 'sft_candidate_800.jsonl'),
            ('candidate_1200', 'sft_candidate_1200.jsonl')]}
    if [r['id'] for r in sft['train']] != parent['train_ids'] or [r['id'] for r in sft['val']] != parent['validation_ids']:
        raise ValueError('Frozen SFT selection differs from parent manifest')
    for small, large in [('candidate_300', 'candidate_800'), ('candidate_800', 'candidate_1200')]:
        if sft[small] != sft[large][:len(sft[small])]:
            raise ValueError('Nested SFT candidates changed')
    if sft['train'] != sft['candidate_800']:
        raise ValueError('Primary SFT training set must match frozen 800 candidate set')
    # The parquet schema contains only id/messages; verify it actually matches JSONL.
    for split, filename in [('train', 'sft_train.parquet'), ('val', 'sft_val.parquet')]:
        expected = [{'id': r['id'], 'messages': r['messages']} for r in sft[split]]
        if list(records(source / filename)) != expected:
            raise ValueError(f'Parquet/JSONL mismatch: {split}')
    unique = {}
    for row in itertools.chain.from_iterable(sft.values()):
        if row['id'] in unique and unique[row['id']] != row:
            raise ValueError('Different SFT rows share an ID')
        unique[row['id']] = row
    measurements = {}
    print(f'Checking {len(unique)} frozen SFT trajectories with both actual tokenizers...', flush=True)
    for i, row in enumerate(unique.values(), 1):
        actual, ids, mask = measure_sft(tokenizer, row, sft_budget)
        previous, old_ids, old_mask = measure_sft(old_tokenizer, row, parent['config']['context_budget'])
        measurements[row['id']] = {'id': row['id'], 'source': row['source'], 'tokens': actual,
                                   'previous_tokens': previous, 'token_ids_identical': ids == old_ids,
                                   'loss_mask_identical': mask == old_mask}
        if i % 100 == 0:
            print(f'  verified {i}/{len(unique)}', flush=True)
    qa = {name: read_jsonl(source / f'rl_{name}.jsonl') for name in ('candidates', 'dev', 'test', 'uniform')}
    if [r['id'] for r in qa['candidates']] != parent['rl_candidate_ids']:
        raise ValueError('Frozen RL candidates changed')
    qa['benchmark30'] = read_jsonl(benchmark / 'test.jsonl')
    if [r['id'] for r in qa['benchmark30']] != benchmark_manifest['sample_ids']:
        raise ValueError('Frozen pilot selection changed')
    groups = {'sft_train': sft['train'], 'sft_val': sft['val'],
              **{name: qa[name] for name in ('candidates', 'dev', 'test', 'benchmark30')}}
    keys = {name: {question_key(r['question']) for r in rows} for name, rows in groups.items()}
    if any(len(keys[name]) != len(rows) or len({r['id'] for r in rows}) != len(rows) for name, rows in groups.items()):
        raise ValueError('Duplicate question/ID within a frozen split')
    overlaps = {f'{a}__{b}': len(keys[a] & keys[b]) for a, b in itertools.combinations(groups, 2)}
    if any(overlaps.values()):
        raise ValueError(f'Forbidden frozen split overlap: {overlaps}')
    candidate_keys = {question_key(r['question']) for r in sft['candidate_1200']}
    if candidate_keys & (keys['sft_val'] | keys['dev'] | keys['test'] | keys['benchmark30']):
        raise ValueError('SFT scale candidate leaked into held-out data')
    prompt_audit = []
    for split, rows in qa.items():
        for row in rows:
            ids = tokenizer.apply_chat_template(initial_prompt(row['question']), tools=[], enable_thinking=True,
                                                 tokenize=True, return_dict=False, add_generation_prompt=True)
            if len(ids) > prompt_budget:
                raise ValueError(f'Initial prompt exceeds budget: {split}/{row["id"]}')
            prompt_audit.append({'id': row['id'], 'split': split, 'prompt_tokens': len(ids)})
    # All checks finish before creating any training-ready snapshot.
    out.mkdir(parents=True)
    copied = ['sft_selected.jsonl', 'sft_validation.jsonl', 'sft_train.parquet', 'sft_val.parquet',
              'manual_review.jsonl', 'rl_sampling_manifest.json', 'rl_profile_report.json',
              'rl_profile.jsonl', 'rl_selected.jsonl']
    copied += [f'sft_candidate_{n}.jsonl' for n in (300, 800, 1200)]
    for filename in copied:
        shutil.copyfile(source / filename, out / filename)
    for n in (300, 800, 1200):
        config = json.loads((source / f'sft_candidate_{n}_config.json').read_text())
        config.update(model=provenance['model'], model_revision=provenance['revision'], context_budget=sft_budget)
        json_file(out / f'sft_candidate_{n}_config.json', config)
    for name in ('candidates', 'dev', 'test', 'uniform'):
        for row in qa[name]:
            if 'question_tokens' in row:
                row['question_tokens'] = len(tokenizer.encode(row['question'], add_special_tokens=False))
        write_jsonl(out / f'rl_{name}.jsonl', qa[name])
    (out / 'benchmark30').mkdir()
    # Keep test bytes, golds, IDs and old manifest frozen; target token counts are separate.
    shutil.copyfile(benchmark / 'test.jsonl', out / 'benchmark30/test.jsonl')
    shutil.copyfile(benchmark / 'benchmark_manifest.json', out / 'benchmark30/benchmark_manifest.json')
    write_jsonl(out / 'token_audit.jsonl', measurements.values())
    write_jsonl(out / 'prompt_audit.jsonl', prompt_audit)
    split_stats = {name: {key: quantiles([measurements[r['id']]['tokens'][key] for r in rows])
                          for key in next(iter(measurements.values()))['tokens']}
                   for name, rows in sft.items()}
    report = {'status': 'passed', 'scope': 'frozen_selection_cpu_retokenization',
              'model': provenance['model'], 'model_revision': provenance['revision'],
              'sft_counts': {name: len(rows) for name, rows in sft.items()},
              'unique_sft_trajectories_checked': len(measurements), 'token_statistics': split_stats,
              'token_ids_changed': sum(not r['token_ids_identical'] for r in measurements.values()),
              'loss_masks_changed': sum(not r['loss_mask_identical'] for r in measurements.values()),
              'tokenizer_files_identical_to_parent': {name: value == old_hashes[name] for name, value in tokenizer_hashes.items()},
              'qa_counts': {name: len(rows) for name, rows in qa.items()},
              'prompt_statistics': {name: quantiles([r['prompt_tokens'] for r in prompt_audit if r['split'] == name]) for name in qa},
              'exact_question_overlaps': overlaps,
              'sft_1200_vs_rl_candidates_overlap': len(candidate_keys & keys['candidates']),
              'budget': {'sft_total': sft_budget, 'rl_prompt': prompt_budget, 'rl_response_including_tools': response_budget,
                         'rl_total': prompt_budget + response_budget},
              'template_settings': {'tools': [], 'enable_thinking': True, 'assistant_header_loss': False},
              'selection_changed': False, 'truncated_or_dropped_rows': 0,
              'answer_correctness': 'unknown; this checks serialization, protocol and masks, not factual correctness',
              'difficulty_profile': 'pending frozen SFT model and actual rollouts',
              'paid_model_calls': 0, 'model_weights_downloaded': False}
    json_file(out / 'validation_report.json', report)
    output_hashes = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob('*')) if p.is_file()}
    config = dict(parent['config'], model=str(tokenizer_path), model_revision=provenance['revision'],
                  out=str(out), context_budget=sft_budget)
    manifest = dict(parent, sources=dict(parent['sources'], model=provenance['model']), config=config,
                    tokenizer=tokenizer_hashes, files_sha256=output_hashes)
    manifest.update(parent_manifest_sha256=file_hash(source / 'sft_manifest.json'),
                    adaptation='Retokenization of the existing frozen selections; original source provenance inherited',
                    model_provenance=provenance, benchmark_sha256=benchmark_hash,
                    code_sha256={name: file_hash(ROOT / 'mini' / name) for name in
                                 ('retokenize.py', 'template.py', 'curation.py', 'common.py')})
    json_file(out / 'sft_manifest.json', manifest)
    # The manifest is the completion marker; incomplete outputs cannot pass validation.
    print(json.dumps({k: report[k] for k in ('status', 'model', 'sft_counts', 'unique_sft_trajectories_checked',
                                            'token_ids_changed', 'loss_masks_changed', 'qa_counts', 'paid_model_calls')}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', default='mini/curated')
    p.add_argument('--benchmark', default='mini/benchmark30')
    p.add_argument('--tokenizer', default='mini/.cache/qwen35-4b-tokenizer')
    p.add_argument('--old-tokenizer', default='mini/.cache/tokenizer')
    p.add_argument('--provenance', default='mini/qwen35_4b_source.json')
    p.add_argument('--out', default='mini/curated-qwen35-4b')
    p.add_argument('--sft-budget', type=int, default=12288)
    p.add_argument('--prompt-budget', type=int, default=1024)
    p.add_argument('--response-budget', type=int, default=8192)
    a = p.parse_args()
    if min(a.sft_budget, a.prompt_budget, a.response_budget) < 1:
        p.error('All context budgets must be positive')
    adapt(a.source, a.benchmark, a.tokenizer, a.old_tokenizer, a.provenance, a.out,
          a.sft_budget, a.prompt_budget, a.response_budget)


if __name__ == '__main__':
    main()
