"""Validate exported real curation artifacts on CPU, including actual loss masks."""
import argparse
import json
from pathlib import Path

from mini.common import question_key, read_jsonl, validate_sft
from mini.curation_utils import file_hash, json_file
from mini.template import turn_tokens


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data-dir',default='mini/curated')
    p.add_argument('--model',default='Qwen/Qwen3.5-2B')
    p.add_argument('--out',default='mini/reports/CURATION_CPU_CHECKS.json')
    a=p.parse_args()
    directory=Path(a.data_dir)
    manifest=json.loads((directory/'sft_manifest.json').read_text(encoding='utf-8'))
    for filename,expected in manifest['files_sha256'].items():
        if file_hash(directory/filename)!=expected:
            raise ValueError(f'curation artifact hash mismatch: {filename}')
    splits={k:read_jsonl(directory/f) for k,f in [('sft_train','sft_selected.jsonl'),('sft_val','sft_validation.jsonl'),('rl_train_candidates','rl_candidates.jsonl'),('rl_dev','rl_dev.jsonl'),('final_test','rl_test.jsonl')]}
    sets={k:{question_key(r['question']) for r in rows} for k,rows in splits.items()}
    for key,rows in splits.items():
        if len(sets[key])!=len(rows) or len({r['id'] for r in rows})!=len(rows):
            raise ValueError(f'duplicate question/ID in {key}')
    overlaps={}
    for i,left in enumerate(sets):
        for right in list(sets)[i+1:]:
            count=len(sets[left]&sets[right]);overlaps[f'{left}__{right}']=count
            permitted=manifest['config']['allow_sft_rl_overlap'] and {left,right}=={'sft_train','rl_train_candidates'}
            if count and not permitted:
                raise ValueError(f'forbidden exact overlap: {left}/{right}')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model)
    checked=0
    for split in ('sft_train','sft_val'):
        for row in splits[split]:
            messages=validate_sft(row);ids=[];masks=[]
            for i,message in enumerate(messages):
                tokens,mask=turn_tokens(tok,messages,i)
                if message['role']!='assistant' and any(mask):
                    raise ValueError('input/observation has nonzero loss mask')
                ids+=tokens;masks+=mask
            full=tok.apply_chat_template(messages,tools=[],enable_thinking=True,tokenize=True,return_dict=False,add_generation_prompt=False)
            if ids!=full or not any(masks) or len(ids)>manifest['config']['context_budget']:
                raise ValueError('invalid real SFT token assembly/mask/budget')
            checked+=1
    sizes={n:read_jsonl(directory/f'sft_candidate_{n}.jsonl') for n in (300,800,1200)}
    for small,large in [(300,800),(800,1200)]:
        if sizes[small]!=sizes[large][:len(sizes[small])]:
            raise ValueError('scale-ablation subsets must be nested')
    if any(question_key(r['question']) in sets['sft_val'] for rows in sizes.values() for r in rows):
        raise ValueError('SFT validation leaked into an ablation training subset')
    report={'status':'passed','scope':'actual_cpu_data_only','counts':{k:len(v) for k,v in splits.items()},
            'checked_sft_actual_token_masks':checked,'exact_question_overlaps':overlaps,'hash_verification':'all exported artifact hashes passed',
            'nested_ablation_sizes':{n:len(r) for n,r in sizes.items()},'gpu_or_paid_api_executed':False,
            'semantic_equivalence':'not certified; see manual_review.jsonl','difficulty_profile':'not executed; pending frozen SFT checkpoint'}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    json_file(a.out,report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
