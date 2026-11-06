"""Deterministic samples; dev/test are disjoint partitions of FlashRAG dev."""
import argparse
import json
from pathlib import Path

from mini.common import digest, question_key, read_jsonl, validate_sft, write_jsonl


def ranked(rows, seed):
    return sorted(rows, key=lambda r: digest([seed, question_key(r['question']), str(r.get('id', ''))]))


def unique(rows, excluded=()):
    seen = set(excluded)
    result = []
    for row in sorted(rows, key=digest):
        key = question_key(row['question'])
        if not key or key in seen:
            continue
        if not isinstance(row.get('golden_answers'), list) or not row['golden_answers'] or not all(isinstance(g, str) and g.strip() for g in row['golden_answers']):
            raise ValueError('QA requires non-empty golden_answers strings')
        seen.add(key)
        result.append(row)
    return result


def take(rows, n, label):
    if n < 1 or len(rows) < n:
        raise ValueError(f'{label}: requested {n}, available {len(rows)}')
    return rows[:n]


def build(sft_rows, qa, seed=42, sft_count=400, train_count=300, dev_count=100, test_count=200, tokenizer=None, max_length=9216):
    if any(n % 2 for n in (train_count, dev_count, test_count)):
        raise ValueError('QA counts must be even for equal NQ/HotpotQA allocation')
    result = {'train': [], 'dev': [], 'test': []}
    heldout = set()
    for source in ('nq', 'hotpotqa'):
        dev = ranked(unique(qa[source]['dev'], heldout), seed)
        hold = take(dev, (dev_count + test_count) // 2, source + ' dev/test')
        result['dev'] += hold[:dev_count // 2]
        result['test'] += hold[dev_count // 2:]
        # Exclude the entire original dev split, including unselected rows.
        heldout.update(question_key(r['question']) for r in dev)
    accepted, rejected, seen = [], [], set()
    for raw in sorted(sft_rows, key=digest):
        try:
            row = dict(raw)
            msgs = validate_sft(row)
            question = (row.get('question') or msgs[1]['content']).removeprefix('Query: ').strip()
            key = question_key(question)
            if key in heldout or key in seen:
                continue
            row.update(question=question, id=str(row.get('id') or digest(msgs)))
            if tokenizer is not None:
                # Upstream SFT concatenates separately rendered turns: conservatively
                # count ALL rendered tokens, including repeated template prefixes.
                length = len(tokenizer.apply_chat_template(msgs, tools=[], tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=True))
                if length > max_length:
                    raise ValueError(f'overlength {length}>{max_length}')
            seen.add(key)
            accepted.append(row)
        except (ValueError, KeyError, TypeError) as e:
            rejected.append({'id': str(raw.get('id', '')), 'reason': str(e)})
    selected = take(ranked(accepted, seed), sft_count, 'SFT')
    sft_keys = {question_key(r['question']) for r in selected}
    used_train = set(heldout) | sft_keys
    for source in ('nq', 'hotpotqa'):
        candidates = unique(qa[source]['train'], used_train)
        chosen = take(ranked(candidates, seed), train_count // 2, source + ' train')
        result['train'] += chosen
        used_train.update(question_key(r['question']) for r in chosen)
    for split in result:
        result[split] = ranked(result[split], seed)
    return selected, result, rejected


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', default='mini/data')
    p.add_argument('--local-input', help='Offline directory: sft.jsonl, {nq,hotpotqa}_{train,dev}.jsonl')
    p.add_argument('--model', default='Qwen/Qwen3.5-2B')
    p.add_argument('--model-revision', default='main')
    p.add_argument('--sft-revision', default='main')
    p.add_argument('--qa-revision', default='main')
    for name, default in [('seed', 42), ('sft-count', 400), ('train-count', 300), ('dev-count', 100), ('test-count', 200), ('max-length', 9216)]:
        p.add_argument('--' + name, type=int, default=default)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.model_revision)
    if a.local_input:
        src = Path(a.local_input)
        if (src / 'sft.jsonl').exists():
            sft = read_jsonl(src / 'sft.jsonl')
        else:
            import pandas as pd
            sft = json.loads(pd.read_parquet(src / 'sft-train.parquet').to_json(orient='records'))
        def local_qa(source, split):
            rows = []
            with (src / f'{source}_{split}.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    rows.append({'id': f'{source}:{split}:{r["id"]}', 'source': source, 'question': r['question'], 'golden_answers': r['golden_answers']})
            return rows
        qa = {s: {k: local_qa(s, k) for k in ('train', 'dev')} for s in ('nq', 'hotpotqa')}
        revisions = {'local_input': str(src.resolve())}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        api = HfApi()
        revisions = {repo: api.dataset_info(repo, revision=rev).sha for repo, rev in [('alireza7/GrepSeek-ColdStart-SFT-10k', a.sft_revision), ('RUC-NLPIR/FlashRAG_datasets', a.qa_revision)]}
        sft = list(load_dataset('alireza7/GrepSeek-ColdStart-SFT-10k', revision=revisions['alireza7/GrepSeek-ColdStart-SFT-10k'], split='train'))
        qa = {}
        for source, folder in [('nq', 'nq'), ('hotpotqa', 'hotpotqa')]:
            qa[source] = {}
            for split in ('train', 'dev'):
                ds = load_dataset('RUC-NLPIR/FlashRAG_datasets', revision=revisions['RUC-NLPIR/FlashRAG_datasets'], data_files={split: f'{folder}/{split}.jsonl'}, split=split)
                qa[source][split] = [dict(r, id=f'{source}:{split}:{r["id"]}', source=source) for r in ds]
    selected, qa_out, rejected = build(sft, qa, a.seed, a.sft_count, a.train_count, a.dev_count, a.test_count, tok, a.max_length)
    out = Path(a.out)
    if (out / 'manifest.json').exists():
        raise SystemExit('refusing to overwrite an existing prepared experiment; use a new --out')
    write_jsonl(out / 'sft.jsonl', selected)
    import pandas as pd
    # No tools column: schema is embedded in system prose, matching RL tools=[].
    pd.DataFrame([{'id': r['id'], 'messages': r['messages']} for r in selected]).to_parquet(out / 'sft.parquet', index=False)
    for split, rows in qa_out.items():
        write_jsonl(out / f'{split}.jsonl', rows)
    write_jsonl(out / 'sft_rejected.jsonl', rejected)
    manifest = {'arguments': vars(a), 'revisions': revisions, 'model_commit': tok.init_kwargs.get('_commit_hash'),
                'inputs_sha256': {'sft': digest(sft), 'qa': digest(qa)},
                'counts': {'sft': len(selected), **{s: len(r) for s, r in qa_out.items()}},
                'test_policy': 'Frozen held-out partition of FlashRAG dev; select checkpoints using dev.jsonl only.',
                'files': {f.name: __import__('hashlib').sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest['counts']))


if __name__ == '__main__':
    main()
