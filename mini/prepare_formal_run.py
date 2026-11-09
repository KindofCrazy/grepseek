"""Freeze a 500-question benchmark and export the already selected curriculum; no model calls."""
import json
from pathlib import Path
import shutil

from mini.benchmark30 import SPLITS, normalize
from mini.common import digest, question_key, read_jsonl, write_jsonl
from mini.curation_utils import NearIndex, file_hash, json_file, records


def main():
    source = Path('mini/curated-curriculum-candidates-20261107')
    out = Path('mini/curated-formal-20261107')
    report = Path('mini/reports/formal-20261107')
    if out.exists(): raise ValueError('Formal snapshot already exists')
    manifest = json.loads((source/'candidate_manifest.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        if file_hash(source/name)!=expected: raise ValueError('Candidate artifact changed')
    a = read_jsonl(source/'stage_a_2000_candidates.jsonl')
    b = read_jsonl(source/'stage_b_3000_candidates.jsonl')
    validation = read_jsonl(source/'validation_200_candidates.jsonl')
    old = read_jsonl('mini/benchmark30/test.jsonl')
    # Exclude the whole original SFT question pool, not only the selected 5k.
    exclusions = read_jsonl('mini/curated/sft_audit.jsonl')
    for name in ('rl_candidates.jsonl','rl_dev.jsonl','rl_test.jsonl'):
        exclusions += read_jsonl(Path('mini/curated')/name)
    forbidden = {question_key(r['question']) for r in exclusions+validation}
    near = NearIndex(.90)
    for text in sorted(forbidden): near.add(text,{'role':'excluded_training_or_dev'})
    cross = NearIndex(.90)
    used = set()
    for row in old:
        used.add(question_key(row['question']));cross.add(row['question'],{'id':row['id']})
    additional = []
    input_hashes = {}
    for source_name,split in SPLITS.items():
        path = Path('mini/.cache')/f'{source_name}_{split}.jsonl'
        input_hashes[str(path)] = file_hash(path)
        rows = [normalize(r,source_name,split) for r in records(path)]
        rows = sorted((r for r in rows if r),key=lambda r:digest([42,'benchmark500',r['id'],question_key(r['question'])]))
        count = 0
        for row in rows:
            text = question_key(row['question'])
            if text in used or text in forbidden or near.query(text) or cross.query(text): continue
            additional.append(row);used.add(text);cross.add(text,{'id':row['id']});count+=1
            if count==70: break
        if count!=70: raise ValueError(f'Insufficient isolated questions: {source_name} {count}/70')
    benchmark = old+additional  # old serials/seeds stay unchanged for 150 reusable base outputs
    if len(benchmark)!=500 or len({r['id'] for r in benchmark})!=500: raise ValueError('Invalid benchmark')
    out.mkdir(parents=True);report.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/'sft_selected.jsonl',a+b)
    write_jsonl(out/'sft_validation.jsonl',validation)
    json_file(out/'curriculum.json',{'stages':[{'name':'a','ids':[r['id'] for r in a]},
                                                            {'name':'b','ids':[r['id'] for r in b]}]})
    write_jsonl(out/'benchmark500.jsonl',benchmark)
    write_jsonl(out/'benchmark500_additional.jsonl',additional)
    for name in ('rl_candidates.jsonl','rl_dev.jsonl','rl_test.jsonl'):
        shutil.copyfile(Path('mini/curated')/name,out/name)
    (out/'benchmark30').mkdir()
    shutil.copyfile('mini/benchmark30/test.jsonl',out/'benchmark30/test.jsonl')
    snapshot = {'sources':{'model':manifest['model'],'sft':'alireza7/GrepSeek-ColdStart-SFT-10k'},
                'model_provenance':manifest['model_provenance'],
                'quality_status':'user_authorized_candidate_training; tool replay and semantic review not certified',
                'candidate_manifest_sha256':file_hash(source/'candidate_manifest.json'),
                'benchmark_input_sha256':input_hashes,
                'files_sha256':{str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file()}}
    json_file(out/'sft_manifest.json',snapshot)
    json_file(report/'snapshot.json',{'data':str(out),'stage_a':2000,'stage_b':3000,'validation':200,
                                    'benchmark':500,'old_benchmark_preserved':150,'new_questions':350,
                                    'count_per_dataset':100,'manifest_sha256':file_hash(out/'sft_manifest.json'),
                                    'model_api_calls':0})
    config = {'stage':'sft','data':str(out),'steps':625,'curriculum':'curriculum.json','lr_schedule':'cosine',
              'min_lr_ratio':.1,'lr':5e-5,'warmup_ratio':.05,'sft_batch':8,'rank':32,'seed':42,
              'val_every':100,'val_count':200,'validation_batch':8,'reload_count':16,'save_every':20,
              'max_train_tokens':30000000,'max_prefill_tokens':200000,'max_sample_tokens':20000,
              'max_estimated_cny':140.,'max_wall_seconds':28800,'swanlab':'offline'}
    json_file(report/'sft-config.json',config)
    print(json.dumps({'data':str(out),'benchmark':500,'new':350,'sft_steps':625,'estimated_cap_cny':140}))


if __name__=='__main__': main()
