"""Produce compact evidence without copying raw training passages into Git."""
import argparse
import json
from pathlib import Path

from mini.curation_utils import file_hash
from mini.pytrio_backend.tracking import json_write


def summarize(root):
    result = json.loads((root/'result.json').read_text())
    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    groups = [r for r in events if r['kind']=='dynamic_group']
    updates = [r for r in events if r['kind'] in ('sft_update','rl_update')]
    summary = {'run':str(root),'status':result['status'],'config':json.loads((root/'config.json').read_text()),
               'provenance':json.loads((root/'provenance.json').read_text()),
               'optimizer_updates':result.get('optimizer_updates',len(updates)),
               'new_optimizer_updates':len(updates),
               'budget':result['budget'],'quality_improvement_claimed':False,
               'candidate_groups':len(groups),'accepted_groups':sum(r['accepted'] for r in groups),
               'groups':[{k:r[k] for k in ('question_id','group_serial','policy_version','rewards',
                                          'reward_sample_std','accepted','drop_reason')} for r in groups],
               'updates':[{'step':r['step'],'metrics':r['metrics']} for r in updates],
               'files_sha256':{name:file_hash(root/name) for name in
                               ('result.json','config.json','provenance.json','events.jsonl')}}
    for key in ('stop_reason','error_type','error_code','reload_nll_abs_delta','validation_before',
                'validation_after','validation_reloaded','sampler_generation_verified','metrics','split'):
        if key in result:
            summary[key] = result[key]
    if 'checkpoint' in result:
        summary['checkpoint'] = {k:result['checkpoint'][k] for k in
                                 ('train_state','sampler_weights','step','resume_safe')}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rows = [summarize(root) for root in args.runs]
    summary = {'runs':rows,'estimated_uncached_cny':sum(r['budget'].get('estimated_incremental_cny',r['budget']['estimated_uncached_cny']) for r in rows),
               'new_optimizer_updates':{stage:sum(r['new_optimizer_updates'] for r in rows if r['config']['stage']==stage)
                                        for stage in ('sft','rl')},
               'actual_bill_cny':None,'quality_improvement_claimed':False}
    json_write(args.out/'summary.json',summary)
    print(json.dumps({'runs':[(r['run'],r['status'],r['optimizer_updates']) for r in rows],
                      'estimated_uncached_cny':summary['estimated_uncached_cny']},ensure_ascii=False))


if __name__ == '__main__':
    main()
