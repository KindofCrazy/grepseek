"""Frozen-SFT coarse profiling and QA-only sampling. Default command is a CPU plan.

Run mode is an explicit future opt-in, restricted to an already-running local
server and a local exported SFT checkpoint. This module never launches a GPU.
"""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlparse

from mini.common import digest, read_jsonl, reward, write_jsonl
from mini.curation_utils import file_hash, json_file, safe_output
from mini.diagnostics import average, sample_std, trajectory_metrics


BUCKETS = ('easy', 'medium', 'hard')


def profile_question(candidate, rollouts):
    if len(rollouts) != 4 or {r['sample_index'] for r in rollouts} != {0,1,2,3}:
        raise ValueError('exactly four unique independent samples (indices 0..3) required')
    rollouts = sorted(rollouts, key=lambda r:r['sample_index'])
    for row in rollouts:
        if row['id'] != candidate['id']:
            raise ValueError('question/rollout ID mismatch')
        for key in ('raw_em','raw_f1','final_reward','format_pass','tool_call_attempts','invalid_tool_calls','tool_calls','response_tokens','failed_rollout','truncated'):
            if key not in row or row[key] is None:
                raise ValueError(f'profiling requires observed {key}')
        if not all(math.isfinite(float(row[k])) for k in ('raw_em','raw_f1','final_reward','response_tokens')) or row['raw_em'] not in (0,1):
            raise ValueError('invalid/nonfinite rollout metrics')
    hits = sum(r['raw_em'] == 1 for r in rollouts)
    rewards = [r['final_reward'] for r in rollouts]
    f1 = [r['raw_f1'] for r in rollouts]
    formats = average([r['format_pass'] for r in rollouts])
    attempts = sum(r['tool_call_attempts'] for r in rollouts)
    invalid = sum(r['invalid_tool_calls'] for r in rollouts)
    valid_rate = 1-invalid/attempts if attempts else None
    protocol_failure = formats < .5 or (valid_rate is not None and valid_rate < .5)
    clean_protocol = formats == 1 and invalid == 0
    retrievable = candidate.get('retrievable') is True and bool(candidate.get('retrievability_evidence'))
    if protocol_failure:
        bucket = 'protocol_failure'
    elif hits == 4:
        bucket = 'easy'
    elif 1 <= hits <= 3:
        bucket = 'medium'
    elif clean_protocol and retrievable:
        bucket = 'hard'
    else:
        bucket = 'hard_pending_retrievability_or_protocol_review'
    return {'id':candidate['id'], 'source':candidate['source'], 'question':candidate['question'],
            'bucket':bucket, 'em_hits':hits, 'estimated_em_rate':hits/4,
            'mean_f1':average(f1), 'f1_std':sample_std(f1),
            'mean_reward':average(rewards), 'reward_std':sample_std(rewards),
            'format_pass_rate':formats, 'valid_tool_call_rate':valid_rate,
            'mean_tool_calls':average([r['tool_calls'] for r in rollouts]),
            'mean_response_tokens':average([r['response_tokens'] for r in rollouts]),
            'failed_rollouts':sum(bool(r['failed_rollout']) for r in rollouts),
            'truncated_rollouts':sum(bool(r['truncated']) for r in rollouts),
            'flat_reward_group':max(rewards)-min(rewards) <= 1e-12,
            'retrievable':candidate.get('retrievable', 'unknown'), 'retrievability_evidence':candidate.get('retrievability_evidence'),
            'rollouts':rollouts, 'scope':'difficulty_profiling',
            'statistical_caveat':'Four stochastic samples provide a coarse estimate only; not a precise success probability.',
            'reward_source':'frozen_sft_profiling_reward_not_GRPO_training_reward'}


def sample_profiles(candidates, profiles, count=300, seed=42, ratios=(.15,.7,.15)):
    if count < 1 or len(ratios)!=3 or any(r<0 for r in ratios) or sum(ratios)<=0:
        raise ValueError('positive count and nonnegative easy/medium/hard ratios required')
    candidates_by_id = {r['id']:r for r in candidates}
    if len(candidates_by_id)!=len(candidates) or len({r['id'] for r in profiles})!=len(profiles):
        raise ValueError('duplicate candidate/profile IDs')
    if any(p['id'] not in candidates_by_id for p in profiles):
        raise ValueError('profile IDs must come from the frozen candidate pool')
    pools = {b:sorted([p for p in profiles if p['bucket']==b], key=lambda r:digest([seed,b,r['id']])) for b in BUCKETS}
    normalized = [r/sum(ratios) for r in ratios]
    targets = [math.floor(count*r) for r in normalized]
    remainder_order = sorted(range(3), key=lambda i:(-(count*normalized[i]-targets[i]), i))
    for i in remainder_order[:count-sum(targets)]:
        targets[i] += 1
    selected = []
    for b,n in zip(BUCKETS,targets):
        selected += pools[b][:n]
    # Fill shortage once from remaining eligible profiles; never repeat samples.
    chosen_ids = {r['id'] for r in selected}
    extras = sorted([p for b in BUCKETS for p in pools[b] if p['id'] not in chosen_ids], key=lambda r:digest([seed,'borrow',r['id']]))
    selected += extras[:max(0,count-len(selected))]
    selected.sort(key=lambda r:digest([seed,'final',r['id']]))
    # Uniform baseline uses the SAME eligible profiled QA population (fair ablation).
    eligible = [candidates_by_id[p['id']] for p in profiles if p['bucket'] in BUCKETS]
    uniform = sorted(eligible,key=lambda r:digest([seed,'uniform',r['id']]))[:count]
    counts = Counter(r['bucket'] for r in selected)
    report = {'seed':seed, 'target_count':count, 'actual_count':len(selected), 'shortfall':max(0,count-len(selected)),
              'requested_ratios':dict(zip(BUCKETS,normalized)), 'target_counts':dict(zip(BUCKETS,targets)),
              'available_counts':{b:len(pools[b]) for b in BUCKETS}, 'actual_counts':dict(counts),
              'actual_ratios':{b:counts[b]/len(selected) if selected else None for b in BUCKETS},
              'excluded_buckets':dict(Counter(p['bucket'] for p in profiles if p['bucket'] not in BUCKETS)),
              'baseline_population':'same eligible profiled question pool; initial unprofiled uniform is a separate candidate baseline',
              'shortage_policy':'borrow unused eligible questions without replacement; report changed ratios and remaining shortfall',
              'training_contract':'Outputs contain QA questions/golds only. GRPO must perform fresh on-policy rollouts; cached profile traces are forbidden training inputs.'}
    def qa_only(r):
        return {key:r[key] for key in ('id','original_id','question','golden_answers','source','retrievable','retrievability_evidence','supervised_overlap_ids') if key in r}
    selected_qa = [dict(qa_only(candidates_by_id[r['id']]), difficulty_bucket=r['bucket']) for r in selected]
    uniform = [qa_only(r) for r in uniform]
    report['actual_source_counts']=dict(Counter(r['source'] for r in selected_qa))
    report['uniform_source_counts']=dict(Counter(r['source'] for r in uniform))
    return selected_qa, uniform, report


def plan(candidates, limit, seed):
    if limit < 1:
        raise ValueError('positive profiling limit required')
    by_source = {s:sorted([r for r in candidates if r['source']==s],key=lambda r:digest([seed,r['id']])) for s in sorted({r['source'] for r in candidates})}
    result = []
    while len(result)<limit and any(by_source.values()):
        for source in by_source:
            if by_source[source] and len(result)<limit:
                result.append(by_source[source].pop(0))
    return result


def run_local(candidates, a, out):
    if not a.execute_local_rollouts:
        raise ValueError('run is disabled without --execute-local-rollouts; use plan for CPU-only work')
    if a.limit>100 and not a.allow_expand:
        raise ValueError('initial profiling is capped at 100; expansion requires --allow-expand after cost review')
    checkpoint = Path(a.checkpoint or '')
    if not a.checkpoint or not checkpoint.is_dir() or not (list(checkpoint.glob('model*.safetensors'))+list(checkpoint.glob('pytorch_model*.bin'))):
        raise ValueError('--checkpoint must be a local exported frozen SFT checkpoint with weights')
    endpoint = urlparse(a.base_url)
    if endpoint.hostname not in ('localhost','127.0.0.1','::1'):
        raise ValueError('only an already-running loopback model server is allowed; no paid/remote API')
    if not (Path(a.corpus)/'wiki_corpus.jsonl').is_file():
        raise ValueError('missing official wiki_corpus.jsonl')
    from transformers import AutoTokenizer
    from openai import OpenAI
    from inference.agent import run_agent_on_example
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    client = OpenAI(base_url=a.base_url,api_key='EMPTY',timeout=a.api_timeout,max_retries=0)
    served = client.models.list().data
    model_name = str(checkpoint.resolve())
    if model_name not in {m.id for m in served}:
        raise ValueError('endpoint must serve the exact absolute checkpoint path as its model name')
    weight_files=sorted(checkpoint.glob('*.safetensors'))+sorted(checkpoint.glob('pytorch_model*.bin'))
    initial_weight_hashes={f.name:file_hash(f) for f in weight_files}
    original = client.chat.completions.create
    result = []
    raw_path = out / 'profiling_rollouts.jsonl'
    if raw_path.exists():
        raise ValueError('refusing to mix fresh samples with an existing profiling run; use a new --out')
    with raw_path.open('w',encoding='utf-8') as f:
        for question in candidates:
            for sample_index in range(4):
                seed = int(digest([a.seed,question['id'],sample_index])[:8],16)
                def create(*args, _seed=seed, **kwargs):
                    kwargs.update(seed=_seed, extra_body={'chat_template_kwargs':{'enable_thinking':True}})
                    return original(*args,**kwargs)
                client.chat.completions.create=create
                started=time.perf_counter()
                record=run_agent_on_example(question,client=client,model=model_name,tokenizer=tokenizer,corpus_dir=a.corpus,
                    max_assistant_turns=6,max_tokens_per_turn=0,temperature=a.temperature,tool_timeout=180).to_dict()
                if record['finish_reason']=='api_error':
                    # Infrastructure failures are never converted into "hard" tasks.
                    raise RuntimeError(f'Profiling API error for {question["id"]}: {record.get("error")}')
                messages=[m for m in record['turns'] if m['role'] in ('assistant','tool')]
                text='\n'.join(m['content'] if m['role']=='assistant' else '<tool_response>'+m['content']+'</tool_response>' for m in messages)
                tokens=sum(len(tokenizer.encode(m['content'],add_special_tokens=False)) for m in messages)
                truncated=any(m.get('finish_reason')=='length' for m in messages)
                # Profiling uses official inference's retry policy; exact official
                # reward is evaluated with its explicit invalid-request flag.
                extra={'response_token_count':tokens,'hit_max_context':truncated,
                       'invalid_tool_request':any(m.get('parse',{}).get('error') and '<tool_call' in m['content'] for m in messages if m['role']=='assistant')}
                official=reward.grepseek_reward(data_source=question['id'],solution_str=text,ground_truth={'golden_answers':question['golden_answers']},extra_info=extra,
                    enable_length_decay=False,reward_metric='f1',penalty_mode='additive_clamped')
                stats=trajectory_metrics(text,question['golden_answers'],messages=messages,training_reward=official['score'],official=official,response_tokens=tokens,truncated=truncated)
                row={'id':question['id'],'sample_index':sample_index,'sample_seed':seed,**stats,'wall_s':time.perf_counter()-started,
                     'scope':'difficulty_profiling','token_count_definition':'sum of assistant+tool content BPE tokens, excluding chat headers; NOT exact RL response_ids length',
                     'agent_semantics':'official inference agent, with one parse-error retry; differs from RL terminal-invalid policy',
                     'raw_record':record}
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();result.append(row)
    weight_hashes={f.name:file_hash(f) for f in weight_files}
    if weight_hashes!=initial_weight_hashes:
        raise RuntimeError('SFT checkpoint files changed during profiling; refusing to treat results as frozen-model evidence')
    json_file(out/'frozen_checkpoint.json',{'path':model_name,'weights_sha256':weight_hashes,'no_optimizer_or_model_update':True})
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=['plan','run','analyze','sample'])
    p.add_argument('--candidates',default='mini/curated/rl_candidates.jsonl')
    p.add_argument('--profiles',help='Existing rl_profile.jsonl for sample mode')
    p.add_argument('--rollouts',help='Actual measured profiling_rollouts.jsonl for analyze mode')
    p.add_argument('--out',default='mini/profile-plan')
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--count',type=int,default=300)
    p.add_argument('--ratios',type=float,nargs=3,default=[.15,.7,.15])
    p.add_argument('--checkpoint')
    p.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--corpus',default='data/wiki_18_corpus')
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--api-timeout',type=float,default=600)
    p.add_argument('--execute-local-rollouts',action='store_true')
    p.add_argument('--allow-expand',action='store_true')
    a=p.parse_args()
    if a.limit>100 and not a.allow_expand:
        p.error('initial plan limit is 100; use --allow-expand only after cost review')
    candidates=read_jsonl(a.candidates)
    if len({r['id'] for r in candidates})!=len(candidates):
        raise ValueError('duplicate candidate IDs')
    out=safe_output(a.out,{'args':vars(a),'candidate_hash':file_hash(a.candidates)})
    if a.mode=='plan':
        subset=plan(candidates,a.limit,a.seed)
        write_jsonl(out/'profiling_questions.jsonl',subset)
        json_file(out/'plan.json',{'status':'not_executed','question_count':len(subset),'rollouts_per_question':4,'total_rollouts':len(subset)*4,'seed':a.seed,'checkpoint_required':True,'budget_rmb_total_user_stated':200,'cost_estimate':'unknown_until_first_measured_local_run; no rented GPU or paid API invoked'})
        print(f'CPU plan only: {len(subset)} questions × 4; no rollouts executed')
        return
    if a.mode in ('run','analyze'):
        measured=run_local(plan(candidates,a.limit,a.seed),a,out) if a.mode=='run' else read_jsonl(a.rollouts or '')
        grouped={}
        for r in measured:
            if r.get('scope')!='difficulty_profiling':
                raise ValueError('profiling requires explicitly scoped measured traces')
            grouped.setdefault(r['id'],[]).append(r)
        by_id={r['id']:r for r in candidates}
        if any(rid not in by_id for rid in grouped):
            raise ValueError('measured question ID absent from frozen pool')
        profiles=[profile_question(by_id[rid],rows) for rid,rows in sorted(grouped.items())]
        write_jsonl(out/'rl_profile.jsonl',profiles)
        json_file(out/'rl_profile_report.json',{'status':'observed_profiles','profiled_questions':len(profiles),
            'buckets':dict(Counter(r['bucket'] for r in profiles)),'mean_reward_std':average([r['reward_std'] for r in profiles]),
            'mean_f1_std':average([r['f1_std'] for r in profiles]),'flat_reward_group_rate':average([r['flat_reward_group'] for r in profiles]),
            'profiling_wall_s':sum(r.get('wall_s',0) for r in measured), 'caveat':'4 samples per question are coarse; do not use final-test data or treat them as exact probabilities'})
    else:
        profiles=read_jsonl(a.profiles or '')
    selected,uniform,report=sample_profiles(candidates,profiles,a.count,a.seed,a.ratios)
    write_jsonl(out/'rl_selected.jsonl',selected)
    write_jsonl(out/'rl_uniform.jsonl',uniform)
    json_file(out/'rl_sampling_manifest.json',{**report,'candidate_sha256':file_hash(a.candidates),'selected_ids':[r['id'] for r in selected],'uniform_ids':[r['id'] for r in uniform]})


if __name__=='__main__':
    main()
