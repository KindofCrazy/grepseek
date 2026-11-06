"""Freeze a 5 x 30 question pilot benchmark, without running a model."""
import argparse
from collections import Counter
import json
from pathlib import Path

from mini.common import ROOT, digest, question_key, read_jsonl, write_jsonl
from mini.curation_utils import NearIndex, file_hash, json_file, records, safe_output

SPLITS = {'nq':'test', 'hotpotqa':'dev', 'triviaqa':'test', '2wikimultihopqa':'dev', 'bamboogle':'test'}


def normalize(row, source, split):
    if not isinstance(row, dict) or not isinstance(row.get('question'), str) or not question_key(row['question']):
        return None
    gold = row.get('golden_answers')
    if not isinstance(gold, list) or not gold or not all(isinstance(g, str) and g.strip() for g in gold):
        return None
    rid = row.get('id')
    if not isinstance(rid, (str, int)) or str(rid).strip() == '':
        return None
    return {'id':f'{source}:{split}:{rid}', 'original_id':rid, 'question':row['question'].strip(),
            'golden_answers':gold, 'source':source, 'original_split':split}


def select(pools, exclusions, count=30, seed=42, threshold=.90):
    if count < 1:
        raise ValueError('positive count required')
    forbidden = {question_key(r['question']) for r in exclusions}
    near = NearIndex(threshold)
    for key in sorted(forbidden):
        near.add(key, {'question_key':key, 'role':'known_training_or_selection_data'})
    selected = []
    used = set()
    cross = NearIndex(threshold)
    audit, reviews = {}, []
    for source in SPLITS:
        normalized = [normalize(r, source, SPLITS[source]) for r in pools[source]]
        good = sorted([r for r in normalized if r is not None], key=lambda r:digest([seed, source, r['id'], question_key(r['question'])]))
        reasons = Counter()
        chosen = []
        examined = 0
        for row in good:
            examined += 1
            key = question_key(row['question'])
            if key in forbidden:
                reasons['exact_training_or_selection_overlap'] += 1
                continue
            if key in used:
                reasons['duplicate_question_within_evaluation'] += 1
                continue
            matches = near.query(row['question']) + cross.query(row['question'])
            if matches:
                reasons['lexical_near_duplicate_quarantine'] += 1
                reviews += [{'id':row['id'],'question':row['question'],**match} for match in matches]
                continue
            chosen.append(row)
            used.add(key)
            cross.add(row['question'], {'id':row['id'],'role':'selected_evaluation_question'})
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f'{source}: only {len(chosen)} isolated eligible questions, requested {count}; no duplication/fake fallback')
        selected += chosen
        audit[source] = {'split':SPLITS[source], 'raw_count':len(pools[source]), 'valid_count':len(good),
                         'invalid_basic_count':sum(r is None for r in normalized), 'selected':len(chosen),
                         'ranked_candidates_examined':examined, 'examined_rejections':dict(reasons)}
    return selected, audit, reviews


def load_exclusions(sft_file, other_files):
    rows = []
    # All available cold-start supervision is reserved, including future scale ablations.
    for i, raw in enumerate(records(sft_file)):
        question = raw.get('question')
        if not isinstance(question,str):
            messages=raw.get('messages',[])
            users=[m.get('content') for m in messages if isinstance(m,dict) and m.get('role')=='user' and isinstance(m.get('content'),str)]
            question=users[0].removeprefix('Query: ').strip() if users else None
        if isinstance(question,str) and question_key(question):
            rows.append({'id':str(raw.get('id',i)), 'question':question, 'role':'available_sft_supervision'})
    for filename in other_files:
        for row in read_jsonl(filename):
            if isinstance(row.get('question'),str):
                rows.append({'id':row['id'],'question':row['question'],'role':str(filename)})
    return rows


def check_training_files(benchmark_file, training_files):
    forbidden={question_key(r['question']) for r in read_jsonl(benchmark_file)}
    for filename in training_files:
        for row in records(filename):
            question=row.get('question')
            if question is None:
                users=[m.get('content','') for m in row.get('messages',[]) if m.get('role')=='user']
                question=users[0].removeprefix('Query: ').strip() if users else ''
            if question_key(question) in forbidden:
                raise ValueError(f'Frozen benchmark question in training file {filename}: {row.get("id")}')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input-dir',default='mini/.cache')
    p.add_argument('--sft-file',default='mini/.cache/sft-train.parquet')
    p.add_argument('--exclude',nargs='*',default=['mini/curated/rl_candidates.jsonl','mini/curated/sft_validation.jsonl','mini/curated/rl_dev.jsonl','mini/curated/rl_test.jsonl'])
    p.add_argument('--out',default='mini/benchmark30')
    p.add_argument('--count',type=int,default=30)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--near-threshold',type=float,default=.90)
    p.add_argument('--model',default='Qwen/Qwen3.5-2B')
    p.add_argument('--check-training',nargs='+',help='Only check given training files against an existing frozen --out/test.jsonl')
    a=p.parse_args()
    if a.check_training:
        check_training_files(Path(a.out)/'test.jsonl',a.check_training)
        print('No normalized-question overlap with frozen benchmark')
        return
    source=Path(a.input_dir)
    inputs={name:source/f'{name}_{split}.jsonl' for name,split in SPLITS.items()}
    hashes={str(f.resolve()):file_hash(f) for f in [*inputs.values(), Path(a.sft_file), *map(Path,a.exclude)]}
    out=safe_output(a.out,{'arguments':vars(a),'input_sha256':hashes})
    pools={name:list(records(path)) for name,path in inputs.items()}
    exclusions=load_exclusions(a.sft_file,a.exclude)
    selected,audit,reviews=select(pools,exclusions,a.count,a.seed,a.near_threshold)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    system=(ROOT/'rl/prompts/system_prompt.txt').read_text(encoding='utf-8').strip()
    for row in selected:
        prompt=[{'role':'system','content':system},{'role':'user','content':'Query: '+row['question']}]
        row['prompt_tokens']=len(tokenizer.apply_chat_template(prompt,tools=[],enable_thinking=True,tokenize=True,return_dict=False,add_generation_prompt=True))
    write_jsonl(out/'test.jsonl',selected)
    for name in SPLITS:
        write_jsonl(out/f'{name}.jsonl',[r for r in selected if r['source']==name])
    write_jsonl(out/'near_duplicate_review.jsonl',reviews)
    samples={name:[{'original_fields':list(r),'id':r.get('id'),'question':r.get('question'),'golden_answers':r.get('golden_answers')} for r in pools[name][:3]] for name in SPLITS}
    json_file(out/'source_audit.json',{'source':'RUC-NLPIR/FlashRAG_datasets','datasets':audit,'actual_examples':samples})
    manifest={'purpose':'frozen_final_pilot_5x30; never select checkpoints on this set',
              'config':vars(a),'input_sha256':hashes,'source_revision':'unknown_for_mirror_cache; byte hashes recorded',
              'sample_ids':[r['id'] for r in selected],'sample_sha256':file_hash(out/'test.jsonl'),
              'code_sha256':{name:file_hash(ROOT/'mini'/name) for name in ('benchmark30.py','common.py','curation_utils.py')},
              'count_by_source':dict(Counter(r['source'] for r in selected)),
              'excluded_known_questions':len({question_key(r['question']) for r in exclusions}),
              'near_duplicate_check':'MinHash candidate search + exact word Jaccard; semantic paraphrase absence is not certified',
              'max_prompt_tokens':max(r['prompt_tokens'] for r in selected),
              'sampling_uses':['seed','dataset','question_id','normalized_question'],
              'sampling_never_uses':['model_output','reward','answer_correctness','difficulty_profile'],
              'model_runs_executed':0,'limitations':['30 per source is a pilot with wide sampling uncertainty','Wikipedia-2018 corpus coverage is unverified; do not replace unanswered test questions after rollout']}
    json_file(out/'benchmark_manifest.json',manifest)
    print(json.dumps({'samples':len(selected),'sources':manifest['count_by_source'],'max_prompt_tokens':manifest['max_prompt_tokens'],'model_runs_executed':0}))


if __name__=='__main__':
    main()
