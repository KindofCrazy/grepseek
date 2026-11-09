"""Prepare a larger, frozen QA pool without model calls or difficulty sampling."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path, PureWindowsPath
import shutil

from mini.benchmark30 import normalize
from mini.common import digest, question_key, read_jsonl, write_jsonl
from mini.curation_utils import file_hash, records


class ExactNearIndex:
    """Exact Jaccard >= 9/10 search, with a lossless rare-token prefix filter.

    A matching set must contain at least ceil(0.9*len(query)) query tokens.
    Therefore it must intersect any len(query)-ceil(0.9*len(query))+1 prefix.
    All stored tokens are indexed; choosing rare tokens changes cost, not recall.
    """
    def __init__(self):
        self.items=[]
        self.postings=defaultdict(set)

    def add(self,text,label):
        words=frozenset(question_key(text).split())
        if not words:
            raise ValueError('Empty question')
        index=len(self.items)
        self.items.append((words,label))
        for word in words:
            self.postings[word].add(index)

    def query(self,text):
        words=frozenset(question_key(text).split())
        if not words:
            return None
        prefix_size=len(words)-(9*len(words)+9)//10+1
        prefix=sorted(words,key=lambda w:(len(self.postings.get(w,())),w))[:prefix_size]
        candidates=set().union(*(self.postings.get(w,()) for w in prefix))
        for i in sorted(candidates):
            other,label=self.items[i]
            if 10*min(len(words),len(other)) < 9*max(len(words),len(other)):
                continue
            intersection=len(words&other)
            union=len(words|other)
            if 10*intersection >= 9*union:
                return {'other':label,'word_jaccard':intersection/union}
        return None


def main():
    from transformers import AutoTokenizer
    from inference.agent import build_system_prompt
    from mini.pytrio_backend.rollout import prompt_tokens
    from mini.retokenize import verify_files, verify_tokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('mini/curated-formal-20261107'))
    parser.add_argument('--inputs',type=Path,default=Path('mini/.cache'))
    parser.add_argument('--tokenizer',type=Path,default=Path('mini/.cache/qwen35-4b-tokenizer'))
    parser.add_argument('--out',type=Path,default=Path('mini/curated-rl10k-20261108'))
    parser.add_argument('--report',type=Path,default=Path('mini/reports/rl-pool10k-20261108'))
    parser.add_argument('--per-source',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.out.exists() or args.report.exists():
        raise ValueError('Use fresh output/report directories; snapshots are immutable')
    manifest=json.loads((args.base/'sft_manifest.json').read_text())
    verify_files(args.base,manifest['files_sha256'])
    verify_tokenizer(args.tokenizer,manifest['model_provenance'])
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    prior=json.loads(Path('mini/curated/input_config_signature.json').read_text())
    expected={PureWindowsPath(k).name:v for k,v in prior['input_hashes'].items()}
    input_hashes={}
    pools={}
    for source in ('nq','hotpotqa'):
        for split in ('train','dev'):
            name=f'{source}_{split}.jsonl'
            path=args.inputs/name
            actual=file_hash(path)
            if actual!=expected[name]:
                raise ValueError('Original source changed: '+name)
            input_hashes[name]=actual
            pools[source,split]=[r for raw in records(path) if (r:=normalize(raw,source,split)) is not None]
    print(json.dumps({'source_rows':{f'{s}/{p}':len(v) for (s,p),v in pools.items()}}),flush=True)
    held=ExactNearIndex()
    held_keys=set()
    held_counts={}
    def add_held(rows,label):
        held_counts[label]=len(rows)
        for row in rows:
            key=question_key(row['question'])
            if key not in held_keys:
                held.add(row['question'],{'id':row['id'],'split':label})
                held_keys.add(key)
    for name in ('benchmark500.jsonl','rl_dev.jsonl','rl_test.jsonl','sft_validation.jsonl','benchmark30/test.jsonl'):
        add_held(read_jsonl(args.base/name),name)
    for source in ('nq','hotpotqa'):
        add_held(pools[source,'dev'],source+'_official_dev')
    seed_rows=read_jsonl(args.base/'rl_candidates.jsonl')
    seed_counts=Counter(r['source'] for r in seed_rows)
    if any(args.per_source<seed_counts[s] for s in ('nq','hotpotqa')):
        raise ValueError('Expanded quotas must preserve all previous candidates')
    selected=list(seed_rows)
    chosen=ExactNearIndex()
    chosen_ids=set()
    prompt_audit=[]
    def prompt_length(row):
        messages=[{'role':'system','content':build_system_prompt()},
                  {'role':'user','content':'Query: '+row['question']}]
        return len(prompt_tokens(tok,messages))
    for row in seed_rows:
        if row['original_split']!='train' or held.query(row['question']) or chosen.query(row['question']):
            raise ValueError('Preserved seed fails expanded isolation: '+row['id'])
        if row['id'] in chosen_ids:
            raise ValueError('Duplicate seed ID')
        n=prompt_length(row)
        if n>1024:
            raise ValueError('Preserved seed prompt exceeds budget')
        prompt_audit.append({'id':row['id'],'prompt_tokens':n,'preserved_seed':True})
        chosen.add(row['question'],{'id':row['id']});chosen_ids.add(row['id'])
    rejected=Counter()
    for source in ('nq','hotpotqa'):
        by_id={r['id']:r for r in pools[source,'train']}
        for row in seed_rows:
            if row['source']==source:
                raw=by_id.get(row['id'])
                if raw is None or question_key(raw['question'])!=question_key(row['question']) or raw['golden_answers']!=row['golden_answers']:
                    raise ValueError('Preserved seed does not match original train source')
        ranked=sorted(pools[source,'train'],key=lambda r:(digest([args.seed,'expanded-rl-pool',r['id']]),r['id']))
        count=seed_counts[source]
        for row in ranked:
            if count>=args.per_source:
                break
            if row['id'] in chosen_ids:
                continue
            reason=('heldout_word_jaccard_090' if held.query(row['question']) else
                    'pool_word_jaccard_090' if chosen.query(row['question']) else None)
            if reason:
                rejected[source+'/'+reason]+=1
                continue
            n=prompt_length(row)
            if n>1024:
                rejected[source+'/prompt_over_1024']+=1
                continue
            selected.append(dict(row,retrievable='unknown',selected_candidate=True,
                                 original_split_role='training_candidate',question_tokens=len(tok.encode(row['question'],add_special_tokens=False))))
            chosen.add(row['question'],{'id':row['id']});chosen_ids.add(row['id'])
            prompt_audit.append({'id':row['id'],'prompt_tokens':n,'preserved_seed':False})
            count+=1
        if count!=args.per_source:
            raise ValueError(f'Insufficient eligible {source}: {count}')
        print(json.dumps({'source':source,'selected':count,'rejections':dict(rejected)}),flush=True)
    selected.sort(key=lambda r:(digest([args.seed,'rl-pool-file-order',r['id']]),r['id']))
    assert len(selected)==2*args.per_source and len(chosen_ids)==len(selected)
    sft_keys={question_key(r['question']) for r in read_jsonl(args.base/'sft_selected.jsonl')}
    args.out.mkdir();args.report.mkdir(parents=True)
    # Copy only manifest-listed data. Keep every evaluation/SFT byte unchanged.
    for name in manifest['files_sha256']:
        if name=='rl_candidates.jsonl':
            continue
        target=args.out/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(args.base/name,target)
    shutil.copy2(args.base/'rl_candidates.jsonl',args.out/'rl_seed600.jsonl')
    write_jsonl(args.out/'rl_candidates.jsonl',selected)
    write_jsonl(args.report/'prompt-audit.jsonl',prompt_audit)
    pool_manifest={'model_api_requests':0,'optimizer_updates':0,'difficulty_sampling':False,
        'selected_questions':len(selected),'source_counts':dict(Counter(r['source'] for r in selected)),
        'preserved_seed_questions':len(seed_rows),'preserved_seed_sha256':file_hash(args.out/'rl_seed600.jsonl'),
        'parent_manifest_sha256':file_hash(args.base/'sft_manifest.json'),'input_sha256':input_hashes,
        'seed':args.seed,'selection':'balanced sources; deterministic hash order; preserve prior600; no answer correctness or difficulty ranking',
        'near_check':'lossless rare-token prefix candidates + exact word-set Jaccard >= 9/10',
        'heldout_counts':held_counts,'distinct_heldout_keys':len(held_keys),'rejections':dict(rejected),
        'prompt_tokens_max':max(r['prompt_tokens'] for r in prompt_audit),
        'sft_training_question_overlap':sum(question_key(r['question']) in sft_keys for r in selected),
        'sft_rl_overlap_allowed':True,'retrievability_verified':False,
        'limitations':['Lexical isolation does not certify semantic or base-pretraining isolation.',
                      'No corpus searches or difficulty model calls were made. Official answer aliases retained.'],
        'builder_sha256':file_hash(__file__)}
    (args.out/'rl_pool_manifest.json').write_text(json.dumps(pool_manifest,indent=2)+'\n',encoding='utf-8')
    snapshot=dict(manifest)
    snapshot['rl_pool']={'manifest':'rl_pool_manifest.json','parent_data':str(args.base),'questions':len(selected)}
    snapshot['files_sha256']={p.relative_to(args.out).as_posix():file_hash(p) for p in sorted(args.out.rglob('*')) if p.is_file()}
    (args.out/'sft_manifest.json').write_text(json.dumps(snapshot,indent=2)+'\n',encoding='utf-8')
    verify_files(args.out,snapshot['files_sha256'])
    unchanged={name:file_hash(args.out/name)==value for name,value in manifest['files_sha256'].items() if name!='rl_candidates.jsonl'}
    assert all(unchanged.values())
    report={'status':'prepared_not_started','data':str(args.out),'manifest_sha256':file_hash(args.out/'sft_manifest.json'),
            'unchanged_parent_files':unchanged,**pool_manifest}
    (args.report/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'selected_questions':len(selected),'prompt_tokens_max':pool_manifest['prompt_tokens_max'],'data':str(args.out)}),flush=True)


if __name__=='__main__':main()
