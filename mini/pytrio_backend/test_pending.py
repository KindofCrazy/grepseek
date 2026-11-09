from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mini.curation_utils import file_hash
from mini.pytrio_backend.config import Config
from mini.pytrio_backend.data import Cursor
from mini.pytrio_backend.dynamic import collect
from mini.pytrio_backend.pending import load_pending,protocol_hashes,verify_migration
from mini.pytrio_backend.rollout import Trajectory,score
from mini.pytrio_backend.scheduling import Scheduler
from mini.pytrio_backend.tracking import BudgetStop


class Log:
    def __init__(self,out=None):
        self.events=[]
        if out is not None:self.out=out
    def event(self,k,**v):self.events.append((k,v))
    def metrics(self,*args):pass


def group(row,serial,policy,flat=False):
    result=[]
    for i in range(4):
        answer='wrong' if flat or i%2 else 'right'
        t=Trajectory(row['id'],row['source'],row['question'],row['golden_answers'],serial,i,policy,
            tokens=[1,2],mask=[0,1],old_logprobs=[0,-1.],initial_length=1,finish_reason='answer',
            messages=[{'role':'assistant','content':f'<think>reason</think><answer>{answer}</answer>'}],
            turns=[{'stop_reason':'stop','text':f'<think>reason</think><answer>{answer}</answer>'}])
        score(t,Config());result.append(t)
    return result


@pytest.fixture
def recovery(tmp_path):
    data=tmp_path/'data';data.mkdir();(data/'sft_manifest.json').write_text('{}')
    cfg=Config(stage='rl',data=str(data),steps=100,lr=1e-6,sampling_rounds=20,max_candidate_groups=20,
               initial_checkpoint='sft')
    rows=[{'id':str(i),'source':'nq','question':f'Question {i}?','golden_answers':['right']} for i in range(100)]
    cursor=Cursor(rows,42);drawn=cursor.take(28)
    groups=[]
    for serial,row in enumerate(drawn[19:],20):
        rounds={20:1,21:1,22:1,23:1,24:2,25:2,26:3,27:4,28:5}
        groups.append({'question_id':row['id'],'group_serial':serial,'round':rounds[serial],
                       'trajectories':[t.record() for t in group(row,serial,3,flat=serial not in (20,21,24))]})
    # The last failed lineage reached old round 5 at serial 28.
    for item in groups:item['round']=min(item['round'],5)
    pending={'version':1,'policy_version':3,'last_group_serial':28,'groups':groups,'prior_wall_s':100.,
             'data_manifest_sha256':file_hash(data/'sft_manifest.json'),'group_size':4,'effective_groups':4,
             'protocol_sha256':protocol_hashes()}
    return cfg,rows,cursor,pending


def test_recovered_three_groups_need_only_one_new_group(recovery):
    cfg,rows,cursor,pending=recovery
    calls=[]
    def generate(row,serial):calls.append(serial);return group(row,serial,3)
    log=Log()
    selected,serial,stats=collect(cursor,cfg,log,generate,3,28,Scheduler(cfg,log),pending)
    assert calls==[29] and serial==29 and cursor.offset==29
    assert [t.group_serial for t in selected]==[20]*4+[21]*4+[24]*4+[29]*4
    assert stats['candidate_groups']==10 and stats['recovered_candidate_groups']==9
    assert stats['new_candidate_groups']==1 and stats['refill_rounds']==6
    assert all(abs(t.advantage)>0 for t in selected)


def test_candidate_cap_counts_paid_recovered_groups_and_persists(recovery,tmp_path):
    cfg,rows,cursor,pending=recovery
    out=tmp_path/'run';out.mkdir();log=Log(out);calls=[]
    def generate(row,serial):calls.append(serial);return group(row,serial,3,flat=True)
    with pytest.raises(BudgetStop,match='insufficient_nonflat_groups'):
        collect(cursor,cfg,log,generate,3,28,Scheduler(cfg,log),pending)
    assert calls==list(range(29,40))
    saved=json.loads((out/'pending-policy-3.json').read_text())
    assert len(saved['groups'])==20 and saved['last_group_serial']==39
    assert cursor.offset==39


@pytest.mark.parametrize('mutation',['policy','mask','score','data','protocol','order','hash'])
def test_pending_policy_and_token_proof_rejects_mismatch(recovery,tmp_path,monkeypatch,mutation):
    from mini.pytrio_backend import pending as module
    cfg,rows,cursor,value=recovery
    folder=tmp_path/'mini'/'run';folder.mkdir(parents=True)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'protocol_hashes',lambda:value['protocol_sha256'])
    altered=deepcopy(value)
    if mutation=='policy':altered['policy_version']=2
    if mutation=='mask':altered['groups'][0]['trajectories'][0]['mask'][0]=1
    if mutation=='score':altered['groups'][0]['trajectories'][0]['reward']=.123
    if mutation=='data':altered['data_manifest_sha256']='wrong'
    if mutation=='protocol':altered['protocol_sha256']={}
    if mutation=='order':altered['groups'].reverse()
    path=folder/'pending.json';path.write_text(json.dumps(altered))
    descriptor={'path':str(path),'sha256':file_hash(path)}
    if mutation=='hash':descriptor['sha256']='wrong'
    bundle={'step':3,'last_group_serial':28,'pending_sampling':descriptor,'config':cfg.to_dict()}
    with pytest.raises(ValueError):load_pending(cfg,bundle,rows,Log())


def test_valid_pending_is_loaded_without_sampling(recovery,tmp_path,monkeypatch):
    from mini.pytrio_backend import pending as module
    cfg,rows,cursor,value=recovery
    folder=tmp_path/'mini'/'run';folder.mkdir(parents=True)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'protocol_hashes',lambda:value['protocol_sha256'])
    path=folder/'pending.json';path.write_text(json.dumps(value))
    bundle={'step':3,'last_group_serial':28,'pending_sampling':{'path':str(path),'sha256':file_hash(path)},'config':cfg.to_dict()}
    result=load_pending(cfg,bundle,rows,Log())
    assert result==value


def test_limit_migration_requires_exact_checkpoint_and_preserves_optimizer(recovery,tmp_path):
    from mini.pytrio_backend.runtime import trainer
    cfg,rows,cursor,value=recovery
    cfg.initial_checkpoint='';cfg.resume=str(tmp_path/'checkpoint.json')
    previous=cfg.to_dict();previous['sampling_rounds']=5
    bundle={'step':3,'train_state':'state3','resume_safe':True,'config':previous,
            'data_manifest_sha256':value['data_manifest_sha256']}
    Path(cfg.resume).write_text(json.dumps(bundle))
    receipt={'kind':'sampling_round_limit_increase','checkpoint_sha256':file_hash(cfg.resume),
             'train_state':'state3','checkpoint_step':3,'sampling_rounds':{'before':5,'after':20},
             'max_candidate_groups':20,'data_manifest_sha256':value['data_manifest_sha256'],
             'protocol_sha256':protocol_hashes()}
    with pytest.raises(ValueError,match='requires a validated'):
        verify_migration(cfg,bundle,Log())
    cfg.resume_migration=str(tmp_path/'migration.json')
    Path(cfg.resume_migration).write_text(json.dumps(receipt))
    @dataclass
    class Lora:
        rank:int=32
        train_attn:bool=True
        train_mlp:bool=True
        train_unembed:bool=False
    calls=[]
    client=SimpleNamespace(model_id=cfg.model,lora=Lora(),training_run_cuid='restored')
    service=SimpleNamespace(create_training_client_from_state_with_optimizer=lambda path:(calls.append(path) or client))
    assert trainer(service,None,cfg,Log(),bundle) is client and calls==['state3']
    cfg.lr=2e-6
    with pytest.raises(ValueError,match='algorithm parameter'):trainer(service,None,cfg,Log(),bundle)
    cfg.lr=1e-6;receipt['checkpoint_sha256']='wrong';Path(cfg.resume_migration).write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='does not bind'):verify_migration(cfg,bundle,Log())
