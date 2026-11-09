"""Operational short-run boundary must stop before the next rollout, preserving resume."""
from types import SimpleNamespace
from dataclasses import dataclass

import pytest

from mini.pytrio_backend.config import Config
from mini.pytrio_backend import rl


class Log:
    def event(self,*args,**kwargs): pass
    def metrics(self,*args,**kwargs): pass


@pytest.mark.parametrize('start,stop,expected',[(0,5,5),(3,5,5),(5,5,5),(0,0,100)])
def test_stop_is_absolute_and_saves_before_another_rollout(monkeypatch,start,stop,expected):
    cfg=Config(stage='rl',steps=100,stop_after_step=stop,initial_checkpoint='sft')
    cfg.validate()
    sampled=[];optimized=[];saved=[]
    client=SimpleNamespace(save_weights_and_get_sampling_client=lambda:sampled.append(True),
        forward_backward_custom=lambda *args:SimpleNamespace(result=lambda **kwargs:SimpleNamespace(metrics={})))
    monkeypatch.setattr(rl,'read_jsonl',lambda path:[{'id':'q'}])
    monkeypatch.setattr(rl,'Search',lambda *args,**kwargs:SimpleNamespace(stats=lambda:{}))
    monkeypatch.setattr(rl,'trainer',lambda *args:client)
    def collect(cursor,config,log,generate,policy_version,serial,scheduler):
        return [SimpleNamespace(reward=1.,mask=[0,1],group_serial=policy_version+1)],serial+1,{'candidate_groups':1}
    monkeypatch.setattr(rl,'collect',collect)
    monkeypatch.setattr(rl,'ppo_batch',lambda *args:([SimpleNamespace(model_input=[1,2])],None))
    monkeypatch.setattr(rl,'adam',lambda *args:SimpleNamespace(learning_rate=1e-6))
    def update(client,params,step,config,log):
        optimized.append(step);return SimpleNamespace(metrics={})
    monkeypatch.setattr(rl,'optimizer_step',update)
    def save(client,config,log,budget,cursor,step,**kwargs):
        saved.append(step);return {'step':step,'config':config.to_dict(),'resume_safe':True}
    monkeypatch.setattr(rl,'save',save)
    budget=SimpleNamespace(scheduler=SimpleNamespace(snapshot=lambda:{}),
        snapshot=lambda:{'estimated_uncached_cny':0.,'tokens':{}},
        call=lambda operation,work,fn:fn())
    bundle={'step':start,'cursor':{'epoch':0,'offset':0,'draws':{}},'last_group_serial':start,
            'config':cfg.to_dict()} if start else None
    result=rl.run(cfg,None,None,Log(),budget,bundle)
    assert optimized==list(range(start+1,expected+1))
    assert len(sampled)==expected-start and saved[-1]==expected
    assert result['checkpoint']['config']['steps']==100
    assert result['status']==('stopped' if expected<100 else 'completed')
    assert result['stop_reason']==('planned_step_limit' if expected<100 else None)


def test_operational_stop_does_not_change_lr_or_resume_algorithm_fields():
    from mini.pytrio_backend.runtime import adam,trainer
    cfg=Config(stage='rl',steps=100,stop_after_step=5,initial_checkpoint='sft')
    assert adam(cfg,5).learning_rate==adam(Config(stage='rl',steps=100,stop_after_step=0),5).learning_rate
    @dataclass
    class Lora:
        rank: int = 32
        train_attn: bool = True
        train_mlp: bool = True
        train_unembed: bool = False
    lora=Lora()
    client=SimpleNamespace(model_id=cfg.model,lora=lora,training_run_cuid='mock')
    service=SimpleNamespace(create_training_client_from_state_with_optimizer=lambda path:client)
    resumed=Config(stage='rl',steps=100,stop_after_step=20,resume='saved')
    from mini.curation_utils import file_hash
    from pathlib import Path
    bundle={'resume_safe':True,'step':5,'config':cfg.to_dict(),'train_state':'state',
            'data_manifest_sha256':file_hash(Path(cfg.data)/'sft_manifest.json')}
    assert trainer(service,None,resumed,Log(),bundle) is client


@pytest.mark.parametrize('cfg',[Config(stage='rl',steps=100,stop_after_step=101),
                               Config(stage='sft',stop_after_step=1),Config(stop_after_step=-1)])
def test_invalid_stop_boundaries_are_rejected(cfg):
    with pytest.raises(ValueError,match='stop_after_step'):
        cfg.validate()
