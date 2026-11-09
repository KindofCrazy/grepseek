"""Summarize an existing RL trial without loading models or submitting requests."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def distribution(values):
    if not values:return {'n':0}
    values=sorted(values)
    return {'n':len(values),'mean':statistics.mean(values),'p50':statistics.median(values),
            'p95':values[max(0,math.ceil(.95*len(values))-1)],'max':values[-1]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    config=json.loads((args.run/'config.json').read_text())
    groups=[];recovered_groups=[];updates=[];errors=[];audits=[];trajectory_count=0;alignment_failures=0
    starting_step=0
    if config.get('resume'):
        resume_path=Path(config['resume'])
        if resume_path.exists():starting_step=json.loads(resume_path.read_text())['step']
    completion_counts=Counter();outcomes=Counter();operations=defaultdict(list)
    group_policy={};tool_by_policy=defaultdict(list)
    latencies=[];last_budget=None;last_routes={};checkpoint_events=0
    for line in (args.run/'events.jsonl').open():
        try:event=json.loads(line)
        except json.JSONDecodeError:
            continue
        kind=event['kind']
        if kind=='dynamic_group_started':
            group_policy[event['group_serial']]=event['policy_version']
        elif kind=='dynamic_group':
            groups.append({k:event[k] for k in ('question_id','group_serial','policy_version','round','rewards','accepted','drop_reason')})
        elif kind=='dynamic_recovered_group':
            recovered_groups.append(event)
        elif kind=='rl_update':
            updates.append({'step':event['step'],'metrics':event['metrics'],
                            'accepted_groups':event['accepted_groups'],
                            'sampling':{k:event['sampling'][k] for k in ('candidate_groups','effective_groups',
                                'refill_rounds','acceptance_rate','overlapped_refill_groups','wall_s')}})
        elif kind=='trajectory':
            trajectory_count+=1;t=event['trajectory']
            mask=t['mask'];old=t['old_logprobs'];tokens=t['tokens']
            valid=(len(mask)==len(old)==len(tokens) and all(math.isfinite(x) for x in old)
                   and all(x==0 for x in mask[:t['initial_length']])
                   and all(lp==0 for m,lp in zip(mask,old) if m==0))
            alignment_failures+=int(not valid)
            outcomes['reward_positive']+=int(t['reward']>0)
            outcomes['strict_success']+=int(t['metrics'].get('strict_success',False))
            completion_counts[t['finish_reason']]+=1
        elif kind=='request_finished':
            operations[event['operation']].append(event['elapsed_s'])
            last_budget=event['budget']
        elif kind=='tool_call':
            latencies.append(event['elapsed_s'])
            tool_by_policy[str(group_policy.get(event['group_serial'],'unknown'))].append(event['elapsed_s'])
        elif kind=='search_execution':last_routes=event['totals']
        elif kind in ('pace_native_audit','sharded_native_audit'):
            audits.append({'kind':kind,'equal':event['equal']})
        elif kind in ('request_error','group_aborted'):
            errors.append({'kind':kind,'error_type':event.get('error_type'),'code':event.get('code')})
        elif kind=='checkpoint_saved':checkpoint_events+=1
    final_path=args.run/'result.json'
    final=json.loads(final_path.read_text()) if final_path.exists() else None
    saved=args.run/'checkpoint.json'
    checkpoint=json.loads(saved.read_text()) if saved.exists() else None
    row={'status':final['status'] if final else 'in_progress','run':str(args.run),
         'stop_reason':final.get('stop_reason') if final else None,
         'starting_step':starting_step,'new_optimizer_updates':len(updates),
         'optimizer_updates':final.get('optimizer_updates',starting_step) if final else
                             (updates[-1]['step'] if updates else starting_step),
         'config':config,'budget':final['budget'] if final else last_budget,
         'groups':{'candidate':len(groups),'accepted':sum(g['accepted'] for g in groups),
                   'recovered_candidate':len(recovered_groups),
                   'recovered_accepted':sum(g.get('accepted',False) for g in recovered_groups),
                   'trained':sum(len(u['accepted_groups']) for u in updates),
                   'drop_reasons':dict(Counter(g['drop_reason'] for g in groups if not g['accepted'])),
                   'unique_questions':len({g['question_id'] for g in groups})},
         'trajectory_count':trajectory_count,'alignment_failures':alignment_failures,
         'finish_reasons':dict(completion_counts),'outcomes':dict(outcomes),
         'tool_latency_s_including_queue':distribution(latencies),
         'tool_latency_by_policy_s':{k:distribution(v) for k,v in tool_by_policy.items()},
         'operation_latency_s':{k:distribution(v) for k,v in operations.items()},
         'search_counts':last_routes,'native_audits':len(audits),
         'native_audit_mismatches':sum(not r['equal'] for r in audits),
         'errors':errors,'checkpoint_events':checkpoint_events,'updates':updates,
         'checkpoint':None if checkpoint is None else {k:checkpoint[k] for k in ('step','resume_safe','train_state','sampler_weights')},
         'quality_improvement_claimed':False,
         'limitations':['Training candidates and temperature-one rollouts; not a heldout benchmark.',
                        'Token fees are uncached estimates, not a provider invoice.',
                        'Incomplete trailing events during an active run are ignored in this read-only report.']}
    if checkpoint:
        row['checkpoint_cursor_draws']=sum(checkpoint['cursor']['draws'].values())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:row[k] for k in ('status','stop_reason','optimizer_updates','groups','trajectory_count',
                    'alignment_failures','native_audits','native_audit_mismatches','errors')},ensure_ascii=False))


if __name__=='__main__':main()
