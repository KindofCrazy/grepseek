from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mini.curation_utils import file_hash
from mini.pytrio_backend.config import Config
from mini.pytrio_backend.pending import protocol_hashes
from mini.pytrio_backend.runtime import trainer, adam


class Log:
    def __init__(self):self.events=[]
    def event(self,k,**v):self.events.append((k,v))


@pytest.fixture
def extension(tmp_path):
    data=tmp_path/'data';data.mkdir();(data/'sft_manifest.json').write_text('{}')
    old=Config(stage='rl',steps=50,lr=1e-6,data=str(data),initial_checkpoint='sft',
               reward_version='format_f1_v2',dynamic_sampling_mode='continuous',advantage_std_floor=.1)
    bundle={'step':50,'resume_safe':True,'train_state':'state50','config':old.to_dict(),
            'data_manifest_sha256':file_hash(data/'sft_manifest.json')}
    checkpoint=tmp_path/'checkpoint50.json';checkpoint.write_text(json.dumps(bundle))
    values=old.to_dict();values.update(steps=100,initial_checkpoint='',resume=str(checkpoint),
                                     resume_migration=str(tmp_path/'migration.json'))
    cfg=Config(**values).validate()
    receipt={'kind':'rl_step_limit_extension','checkpoint_sha256':file_hash(checkpoint),
             'checkpoint_step':50,'train_state':'state50','steps':{'before':50,'after':100},
             'data_manifest_sha256':file_hash(data/'sft_manifest.json'),'protocol_sha256':protocol_hashes()}
    Path(cfg.resume_migration).write_text(json.dumps(receipt))
    @dataclass
    class Lora:
        rank:int=32
        train_attn:bool=True
        train_mlp:bool=True
        train_unembed:bool=False
    calls=[];client=SimpleNamespace(model_id=cfg.model,lora=Lora(),training_run_cuid='restored')
    svc=SimpleNamespace(create_training_client_from_state_with_optimizer=lambda path:(calls.append(path) or client))
    return cfg,bundle,receipt,svc,client,calls


def test_extension_keeps_optimizer_and_constant_lr(extension):
    cfg,bundle,receipt,svc,client,calls=extension;log=Log()
    assert trainer(svc,None,cfg,log,bundle) is client
    assert calls==['state50'] and log.optimizer_updates==50
    assert adam(cfg,51).learning_rate==adam(Config(**bundle['config']),50).learning_rate==1e-6
    assert any(k=='resume_step_limit_extension' and v['optimizer_preserved'] for k,v in log.events)


@pytest.mark.parametrize('mutation',['missing','hash','protocol','steps','partial','cosine','reward','unsafe'])
def test_extension_rejects_unbound_or_algorithm_changes(extension,mutation):
    cfg,bundle,receipt,svc,client,calls=extension
    if mutation=='missing':cfg.resume_migration=''
    elif mutation=='hash':receipt['checkpoint_sha256']='wrong'
    elif mutation=='protocol':receipt['protocol_sha256']={}
    elif mutation=='steps':cfg.steps=40
    elif mutation=='partial':bundle['step']=40
    elif mutation=='cosine':cfg.lr_schedule='cosine'
    elif mutation=='reward':cfg.format_reward_weight=.3
    elif mutation=='unsafe':bundle['resume_safe']=False
    if cfg.resume_migration:Path(cfg.resume_migration).write_text(json.dumps(receipt))
    with pytest.raises(ValueError):trainer(svc,None,cfg,Log(),bundle)
    assert calls==[]


def test_explicit_restore_loads_full_state_before_returning_client(extension):
    cfg,bundle,receipt,svc,client,calls=extension;cfg.resume_load_mode='explicit'
    actions=[]
    def create(*args,**kwargs):
        actions.append(('create',args,kwargs));return client
    client.load_state_with_optimizer=lambda path:(actions.append(('load_optimizer',path)) or
        SimpleNamespace(result=lambda **kwargs:actions.append(('load_completed',kwargs))))
    service=SimpleNamespace(create_lora_training_client=create)
    log=Log();assert trainer(service,None,cfg,log,bundle) is client
    assert [x[0] for x in actions]==['create','load_optimizer','load_completed']
    assert actions[0][2]=={'rank':32,'seed':42,'train_attn':True,'train_mlp':True,'train_unembed':False}
    assert actions[1][1]=='state50' and calls==[]
    assert next(v for k,v in log.events if k=='trainer_created')['optimizer_restored'] is True


def test_explicit_load_failure_never_reports_a_ready_trainer(extension):
    cfg,bundle,receipt,svc,client,calls=extension;cfg.resume_load_mode='explicit'
    def fail(**kwargs):raise RuntimeError('state could not be loaded')
    client.load_state_with_optimizer=lambda path:SimpleNamespace(result=fail)
    service=SimpleNamespace(create_lora_training_client=lambda *args,**kwargs:client);log=Log()
    with pytest.raises(RuntimeError):trainer(service,None,cfg,log,bundle)
    assert not any(k in ('trainer_created','optimizer_state_loaded') for k,v in log.events)


@pytest.mark.parametrize('cfg',[Config(stage='sft',resume='state',resume_load_mode='explicit'),
                              Config(stage='rl',initial_checkpoint='state',resume_load_mode='explicit'),
                              Config(resume_load_mode='unknown')])
def test_explicit_restore_requires_an_rl_resume(cfg):
    with pytest.raises(ValueError):cfg.validate()
