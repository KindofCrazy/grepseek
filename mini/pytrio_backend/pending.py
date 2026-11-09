"""Hash-bound, same-policy recovery of paid rollouts and sampler-limit migrations."""
from copy import deepcopy
import json
import math
from pathlib import Path

from mini.common import ROOT, read_jsonl
from mini.curation_utils import file_hash
from mini.pytrio_backend.tracking import json_write

PROTOCOL_FILES=('mini/pytrio_backend/rollout.py','mini/pytrio_backend/rewards.py','mini/pytrio_backend/loss.py',
                'inference/agent.py','rl/grepseek/trainer/verl_integration/reward_function.py')


def protocol_hashes():
    return {name:file_hash(ROOT/name) for name in PROTOCOL_FILES}


def verify_migration(config,bundle,log):
    if not config.resume_migration:
        raise ValueError('Sampling limit change requires a validated resume_migration receipt')
    path=Path(config.resume_migration)
    receipt=json.loads(path.read_text())
    before=bundle['config']['sampling_rounds']
    if (receipt.get('kind')!='sampling_round_limit_increase'
        or receipt['checkpoint_sha256']!=file_hash(config.resume)
        or receipt['train_state']!=bundle['train_state']
        or receipt['checkpoint_step']!=bundle['step']
        or receipt['sampling_rounds']!={'before':before,'after':config.sampling_rounds}
        or config.sampling_rounds<=before
        or receipt['max_candidate_groups']!=config.max_candidate_groups
        or receipt['data_manifest_sha256']!=file_hash(Path(config.data)/'sft_manifest.json')
        or receipt['protocol_sha256']!=protocol_hashes()):
        raise ValueError('Resume migration does not bind this checkpoint, data, protocol and limit change')
    log.event('resume_config_migration',receipt=str(path),receipt_sha256=file_hash(path),
              checkpoint_step=bundle['step'],sampling_rounds=receipt['sampling_rounds'],
              optimizer_preserved=True)
    return receipt


def pending_path(log,policy_version):
    return log.out/f'pending-policy-{policy_version}.json' if hasattr(log,'out') else None


def save_pending(log,config,policy_version,groups,serial,wall_s):
    path=pending_path(log,policy_version)
    if path is None:return
    value={'version':1,'policy_version':policy_version,'groups':groups,'last_group_serial':serial,
           'prior_wall_s':wall_s,'data_manifest_sha256':file_hash(Path(config.data)/'sft_manifest.json'),
           'group_size':config.group,'effective_groups':config.effective_groups,
           'protocol_sha256':protocol_hashes()}
    json_write(path,value)
    log.event('pending_batch_saved',policy_version=policy_version,path=str(path),
              sha256=file_hash(path),candidate_groups=len(groups))


def load_pending(config,bundle,rows,log):
    if not bundle:return None
    descriptor=bundle.get('pending_sampling')
    if descriptor is None and bundle['config']['sampling_rounds']!=config.sampling_rounds:
        descriptor=verify_migration(config,bundle,log).get('pending_sampling')
    if descriptor is None:return None
    path=Path(descriptor['path']).resolve()
    if not path.is_relative_to(ROOT/'mini') or file_hash(path)!=descriptor['sha256']:
        raise ValueError('Pending rollout artifact hash/path mismatch')
    value=json.loads(path.read_text())
    if (value['version']!=1 or value['policy_version']!=bundle['step']
        or value['data_manifest_sha256']!=file_hash(Path(config.data)/'sft_manifest.json')
        or value['group_size']!=config.group or value['effective_groups']!=config.effective_groups
        or value['protocol_sha256']!=protocol_hashes()
        or value['last_group_serial']!=bundle['last_group_serial']
        or (config.dynamic_sampling_mode=='bounded' and len(value['groups'])>config.max_candidate_groups)):
        raise ValueError('Pending rollouts do not match the restored policy/data/candidate limits')
    from mini.pytrio_backend.rollout import Trajectory,score
    lookup={r['id']:r for r in rows};serials=set();ordered_serials=[]
    for group in value['groups']:
        qid=group['question_id'];serial=group['group_serial']
        if qid not in lookup or serial in serials or serial>value['last_group_serial']:
            raise ValueError('Invalid pending question/serial')
        serials.add(serial);ordered_serials.append(serial);row=lookup[qid]
        trajectories=[Trajectory(**raw) for raw in group['trajectories']]
        if len(trajectories)!=config.group or [t.sample_index for t in trajectories]!=list(range(config.group)):
            raise ValueError('Pending group is incomplete')
        for t in trajectories:
            if (t.question_id!=qid or t.group_serial!=serial or t.policy_version!=bundle['step']
                or t.question!=row['question'] or t.gold!=row['golden_answers'] or t.source!=row['source']
                or len(t.tokens)!=len(t.mask) or len(t.tokens)!=len(t.old_logprobs)
                or not 0<t.initial_length<=len(t.tokens)<=config.context
                or any(m!=0 for m in t.mask[:t.initial_length])
                or any(m not in (0,1) for m in t.mask)
                or any(not math.isfinite(lp) or (m==0 and lp!=0) for m,lp in zip(t.mask,t.old_logprobs))):
                raise ValueError('Pending trajectory policy/token alignment mismatch')
            recomputed=score(deepcopy(t),config)
            if abs(recomputed.reward-t.reward)>1e-12:
                raise ValueError('Pending reward differs under current scorer')
    if ordered_serials!=list(range(value['last_group_serial']-len(ordered_serials)+1,value['last_group_serial']+1)):
        raise ValueError('Pending groups do not preserve the original serial order')
    log.event('pending_batch_loaded',policy_version=bundle['step'],candidate_groups=len(value['groups']),
              source=str(path),sha256=descriptor['sha256'],new_sampling_requests=0)
    return value


def import_stopped_batch(source,checkpoint,out):
    """Offline import only: prove no optimizer change after the source batch."""
    bundle=json.loads(checkpoint.read_text());config=bundle['config'];policy=bundle['step']
    result=json.loads((source/'result.json').read_text())
    marker=json.loads((source/'optimizer_status.json').read_text())
    if (not bundle['resume_safe'] or result['stop_reason']!='insufficient_nonflat_groups'
        or result['optimizer_updates']!=policy or marker!={'step':policy,'status':'completed'}):
        raise ValueError('Source run is not a safe stopped same-policy batch')
    events=[json.loads(s) for s in (source/'events.jsonl').read_text().splitlines()]
    submitted=[e['step'] for e in events if e['kind']=='optimizer_submitted']
    completed=[e['step'] for e in events if e['kind']=='optimizer_completed']
    if submitted!=completed or max(completed,default=0)!=policy:
        raise ValueError('Uncertain or newer optimizer update exists')
    provenance=json.loads((source/'provenance.json').read_text())
    hashes={**provenance['code_sha256'],**provenance['shared_source_sha256']}
    if any(hashes.get(name)!=value for name,value in protocol_hashes().items()):
        raise ValueError('Source scorer/rollout protocol differs')
    creator=next(e for e in events if e['kind']=='trainer_created')
    if creator['training_run_id']!=bundle['training_run_id']:
        raise ValueError('Checkpoint and source trainer differ')
    groups=[e for e in events if e['kind']=='dynamic_group' and e['policy_version']==policy]
    raws={}
    for e in events:
        if e['kind']=='trajectory' and e['trajectory']['policy_version']==policy:
            raw=e['trajectory'];raws.setdefault(raw['group_serial'],[]).append(raw)
    restored=[]
    for e in groups:
        trajectories=sorted(raws[e['group_serial']],key=lambda t:t['sample_index'])
        if len(trajectories)!=config['group'] or [t['reward'] for t in trajectories]!=e['rewards']:
            raise ValueError('Source group records are incomplete or changed')
        restored.append({'question_id':e['question_id'],'group_serial':e['group_serial'],
                         'round':e['round'],'trajectories':trajectories})
    stats=[e for e in events if e['kind']=='dynamic_sampling' and e['policy_version']==policy][-1]
    if len(restored)!=stats['candidate_groups'] or bundle['last_group_serial']!=stats['last_group_serial']:
        raise ValueError('Source cursor/group accounting mismatch')
    value={'version':1,'policy_version':policy,'groups':restored,
           'last_group_serial':bundle['last_group_serial'],'prior_wall_s':stats['wall_s'],
           'data_manifest_sha256':bundle['data_manifest_sha256'],'group_size':config['group'],
           'effective_groups':config['effective_groups'],'protocol_sha256':protocol_hashes(),
           'import_proof':{'events_sha256':file_hash(source/'events.jsonl'),
                           'checkpoint_sha256':file_hash(checkpoint),'optimizer_updates_after_policy':0}}
    json_write(out,value)
    return value
