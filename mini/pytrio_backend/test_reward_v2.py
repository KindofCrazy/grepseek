"""Exercise reward semantics, continuous refill, and inherited paid-token limits."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from mini.pytrio_backend.config import Config
from mini.pytrio_backend.data import Cursor
from mini.pytrio_backend.dynamic import classify, collect
from mini.pytrio_backend.rollout import Trajectory, score
from mini.pytrio_backend.scheduling import Scheduler
from mini.pytrio_backend.tracking import Budget, BudgetStop, budget_parent_tokens


class Log:
    def __init__(self):self.events=[]
    def event(self,kind,**values):self.events.append((kind,values))
    def metrics(self,*args):pass


def trajectory(answer='right',finish='answer',invalid=False):
    text='<think>reason</think>'+('' if answer is None else f'<answer>{answer}</answer>')
    return Trajectory('q','nq','Question?',['right'],1,0,0,tokens=[1,2],mask=[0,1],
                      old_logprobs=[0,-1.],initial_length=1,finish_reason=finish,
                      invalid_request=invalid,messages=[{'role':'assistant','content':text}],
                      turns=[{'stop_reason':'stop','text':text}])


@pytest.mark.parametrize('answer,finish,invalid,expected',[
    ('right','answer',False,1.2),('wrong','answer',False,.2),
    (None,'turn_budget',False,0.),(None,'observation_budget_overflow',False,0.),
    ('right','answer',True,0.),('','answer',False,0.),
])
def test_actual_format_and_answer_f1_components(answer,finish,invalid,expected):
    cfg=Config(reward_version='format_f1_v2')
    t=score(trajectory(answer,finish,invalid),cfg)
    assert t.reward==pytest.approx(expected)
    assert t.reward==pytest.approx(t.metrics['reward_format_component']+t.metrics['reward_f1_component'])
    assert 0<=t.reward<=1.2
    baseline=score(trajectory(answer,finish,invalid),Config())
    assert t.metrics['raw_f1']==baseline.metrics['raw_f1']
    assert t.metrics['raw_em']==baseline.metrics['raw_em']


def test_format_gate_still_rejects_malformed_answer_and_partial_f1_is_continuous():
    cfg=Config(reward_version='format_f1_v2')
    t=trajectory('right extra');score(t,cfg)
    assert t.reward==pytest.approx(.2+2/3)
    t=trajectory();t.messages[0]['content']='<answer>right</answer>'
    score(t,cfg)
    assert t.reward==0 and t.metrics['raw_f1']==1


def test_auxiliary_only_group_is_not_amplified_to_unit_advantage():
    cfg=Config(reward_version='format_f1_v2',format_reward_weight=.05,advantage_std_floor=.1)
    group=[score(trajectory('wrong'),cfg) for _ in range(2)]+[
        score(trajectory(None,'turn_budget'),cfg) for _ in range(2)]
    assert classify(group,cfg)[1] is None
    assert sum(t.advantage for t in group)==pytest.approx(0)
    assert max(abs(t.advantage) for t in group)<.251
    assert classify([score(trajectory('wrong'),cfg) for _ in range(4)],cfg)[1] is not None


def make_group(serial,flat=False):
    group=[score(trajectory('wrong' if flat or i%2 else 'right'),Config()) for i in range(4)]
    for i,t in enumerate(group):t.group_serial=serial;t.sample_index=i
    return group


@pytest.mark.parametrize('width',[1,4])
def test_continuous_refill_retains_three_groups_past_soft_candidate_and_round_limits(width):
    cfg=Config(effective_groups=4,dynamic_sampling_mode='continuous',max_candidate_groups=20,
               sampling_rounds=1,question_concurrency=width)
    cursor=Cursor([{'id':str(i)} for i in range(100)],42);log=Log()
    selected,serial,stats=collect(cursor,cfg,log,lambda row,s:make_group(s,s not in (1,2,3,25)),0,
                                  scheduler=Scheduler(cfg,log))
    assert serial==25 and stats['candidate_groups']==25
    assert [t.group_serial for t in selected]==[s for s in (1,2,3,25) for _ in range(4)]
    assert stats['effective_groups']==4 and stats['soft_limit_warnings']==1
    assert sum(cursor.draws.values())==25
    assert [e['action'] for k,e in log.events if k=='dynamic_soft_limit']==['continue_same_policy']


def test_continuous_sampling_still_honors_real_budget_stop():
    cfg=Config(dynamic_sampling_mode='continuous',sampling_rounds=1,max_candidate_groups=20)
    cursor=Cursor([{'id':str(i)} for i in range(100)],42);log=Log()
    calls=[]
    def generate(row,serial):
        calls.append(serial)
        if serial==24:raise BudgetStop('estimated_cost_limit')
        return make_group(serial,True)
    with pytest.raises(BudgetStop,match='estimated_cost_limit'):
        collect(cursor,cfg,log,generate,0)
    assert calls==list(range(1,25))
    assert 24<=sum(cursor.draws.values())<=24+cfg.effective_groups-1


def test_fresh_optimizer_still_inherits_prior_cost_and_reservations(tmp_path):
    path=tmp_path/'result.json'
    parent={'status':'stopped','budget':{'tokens':{'prefill':6593545,'sample':374356,'train':2448834},'reserved':{}}}
    path.write_text(json.dumps(parent));cfg=Config(budget_parent=str(path),max_estimated_cny=23.,
        max_prefill_tokens=10000000,max_sample_tokens=10000000,max_train_tokens=10000000)
    log=Log();tokens=budget_parent_tokens(cfg,log);budget=Budget(cfg,log,tokens)
    assert budget.snapshot()['estimated_uncached_cny']==pytest.approx(22.64166465)
    submitted=[]
    with pytest.raises(BudgetStop,match='estimated_cost_limit'):
        budget.call('sample',{'sample':100000},lambda:submitted.append(True))
    assert submitted==[] and log.events[0][1]['optimizer_inherited'] is False
    parent['budget']['reserved']={'sample':1};path.write_text(json.dumps(parent))
    with pytest.raises(ValueError,match='unsettled'):budget_parent_tokens(cfg,log)


@pytest.mark.parametrize('kwargs',[
    {'reward_version':'unknown'},{'format_reward_weight':1.5},{'advantage_std_floor':float('nan')},
    {'f1_reward_weight':float('nan')},{'f1_reward_weight':0.},
    {'dynamic_sampling_mode':'unknown'},{'stage':'eval','reward_version':'format_f1_v2'},
    {'budget_parent':'parent','resume':'saved'},
])
def test_invalid_reward_or_budget_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):Config(**kwargs).validate()


@pytest.mark.parametrize('name,value',[
    ('reward_version','format_f1_v2'),('format_reward_weight',.1),('f1_reward_weight',.9),
    ('advantage_std_floor',.1),('dynamic_sampling_mode','continuous'),
])
def test_exact_resume_rejects_silent_reward_or_advantage_changes(name,value):
    from mini.pytrio_backend.runtime import trainer
    cfg=Config(stage='rl',resume='checkpoint')
    previous=cfg.to_dict();setattr(cfg,name,value)
    bundle={'step':15,'resume_safe':True,'config':previous}
    with pytest.raises(ValueError,match='configuration'):
        trainer(None,None,cfg,Log(),bundle)


def test_two_reward_weights_are_independent_and_not_renormalized():
    cfg=Config(reward_version='format_f1_v2',format_reward_weight=.1,f1_reward_weight=.7)
    assert score(trajectory('right'),cfg).reward==pytest.approx(.8)
    assert score(trajectory('wrong'),cfg).reward==pytest.approx(.1)
    assert score(trajectory('right extra'),cfg).reward==pytest.approx(.1+.7*2/3)
