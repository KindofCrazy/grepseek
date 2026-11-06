import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mini.common import question_key, write_jsonl
from mini.curation import audit_sft, token_accounting, stratum
from mini.curation_utils import NearIndex, records, stratified_order, quantiles
from mini.difficulty import profile_question, sample_profiles, plan, run_local
from mini.diagnostics import grpo_metrics, protocol_metrics, trajectory_metrics, optimization_metrics
from mini.tests.test_mini import trajectory


@pytest.fixture
def tokenizer():
    path = os.environ.get('MINI_TOKENIZER')
    if not path:
        pytest.skip('set MINI_TOKENIZER to real Qwen3.5 tokenizer')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def test_real_audit_provenance_and_unknown_correctness(tokenizer, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('curation must not execute commands')
    monkeypatch.setattr('subprocess.run', forbidden)
    raw = trajectory()
    raw.pop('question')
    audit=audit_sft(raw, 0, tokenizer, {}, 12288)
    assert audit['eligible'] and audit['normal_termination']
    assert audit['original_question'] is None
    assert audit['question_provenance']=='parsed_initial_user_message'
    assert audit['answer_correctness']=='unknown' and audit['task_success']=='unknown'
    assert audit['source']=='unknown' and audit['tool_calls']==1


def test_role_token_accounting_matches_actual_masks(tokenizer):
    from mini.template import turn_tokens
    messages=trajectory()['messages']
    totals=token_accounting(tokenizer,messages)
    ids=[]; mask=[]
    for i in range(len(messages)):
        t,m=turn_tokens(tokenizer,messages,i); ids+=t; mask+=m
    assert totals['total_sequence_tokens']==len(ids)
    assert totals['trainable_assistant_tokens']==sum(mask)
    assert totals['masked_tokens']+sum(mask)==len(ids)
    assert sum(totals[k] for k in ('prompt_tokens','assistant_tokens','tool_observation_tokens'))==len(ids)


@pytest.mark.parametrize('change,reason', [
    (lambda r:r['messages'].pop(), 'no_final_answer'),
    (lambda r:r['messages'][2].update(content='<tool_call>bad</tool_call>'), 'protocol_or_format_error'),
    (lambda r:r['messages'][2].update(content='<think>x</think><tool_call>{"name":"python","arguments":{}}</tool_call>'), 'protocol_or_format_error'),
    (lambda r:r['messages'][3].update(content='{}'), 'protocol_or_format_error'),
    (lambda r:r['messages'][1].update(content=''), 'missing_question'),
    (lambda r:r['messages'][2].update(content='<|im_start|>assistant\n<answer>x</answer>'), 'embedded_chat_control_token'),
])
def test_invalid_sft_filter_reasons(tokenizer,change,reason):
    raw=trajectory(); raw.pop('question',None); change(raw)
    a=audit_sft(raw,0,tokenizer,{},12288)
    assert not a['eligible'] and reason in a['filter_reasons']


def test_context_and_leakage(tokenizer):
    raw=trajectory()
    lookup={question_key(raw['question']):[{'id':'nq:dev:x','source':'nq','original_split':'dev'}]}
    a=audit_sft(raw,0,tokenizer,lookup,1)
    assert {'context_budget_exceeded','heldout_dev_question_overlap'} <= set(a['filter_reasons'])
    assert a['source']=='nq' and a['qa_matches'][0]['split']=='dev'


def test_unknown_malformed_record(tokenizer):
    assert 'invalid_messages' in audit_sft({'id':'x','messages':None},0,tokenizer,{},100)['filter_reasons']


def test_observation_truncation_is_not_context_truncation(tokenizer):
    raw=trajectory(); payload=json.loads(raw['messages'][3]['content']);payload['truncated']=True
    raw['messages'][3]['content']=json.dumps(payload)
    a=audit_sft(raw,0,tokenizer,{},12288)
    assert a['eligible'] and not a['obvious_trajectory_truncation'] and a['observation_truncation_count']==1
    b=audit_sft(raw,0,tokenizer,{},12288,True)
    assert 'observation_truncated_by_policy' in b['filter_reasons']


def test_bad_json_loader_audits_instead_of_faking(tmp_path):
    p=tmp_path/'bad.jsonl';p.write_text('{invalid}\n{"id":"ok"}\n')
    rows=list(records(p))
    assert '_parse_error' in rows[0] and rows[1]['id']=='ok'
    with pytest.raises(FileNotFoundError,match='Missing real input'):
        list(records(tmp_path/'missing.jsonl'))


def test_near_duplicates_word_order_and_determinism():
    index=NearIndex(.8)
    index.add('Who wrote the novel War and Peace?',{'id':'x'})
    assert index.query('War and Peace the novel who wrote')[0]['other']['id']=='x'
    assert index.query('Which planet has the largest diameter?')==[]


def test_stratified_order_is_nested_unique_and_stable():
    rows=[{'id':str(i),'source':'nq' if i%2 else 'hotpotqa','bin':i%3} for i in range(50)]
    f=lambda r:(r['source'],r['bin'])
    a=stratified_order(rows,42,f);b=stratified_order(list(reversed(rows)),42,f)
    assert a==b and len({r['id'] for r in a})==50
    assert a[:30][:10]==a[:10]
    assert len({r['source'] for r in a[:4]})==2


def measured(i,em=0,reward_value=0,valid=True):
    # Explicit test-only fixtures, never used for actual data reports.
    return {'id':'q','sample_index':i,'raw_em':em,'raw_f1':float(em), 'final_reward':reward_value,
            'format_pass':valid,'tool_call_attempts':1,'invalid_tool_calls':int(not valid),'tool_calls':int(valid),
            'response_tokens':20,'failed_rollout':False,'truncated':False}


@pytest.mark.parametrize('hits,bucket',[(4,'easy'),(2,'medium'),(0,'hard_pending_retrievability_or_protocol_review')])
def test_difficulty_buckets_and_unknown_retrievability(hits,bucket):
    p=profile_question({'id':'q','source':'nq','question':'q'},[measured(i,i<hits,float(i)) for i in range(4)])
    assert p['bucket']==bucket and p['reward_std']>0 and not p['flat_reward_group']


def test_hard_needs_evidence_and_protocol_failures_not_difficulty():
    c={'id':'q','source':'nq','question':'q','retrievable':True,'retrievability_evidence':'manually verified passage id 123'}
    assert profile_question(c,[measured(i) for i in range(4)])['bucket']=='hard'
    assert profile_question(c,[measured(i,valid=False) for i in range(4)])['bucket']=='protocol_failure'


def test_profile_exact_four_and_finite():
    c={'id':'q','source':'nq','question':'q'}
    with pytest.raises(ValueError):profile_question(c,[measured(0)]*4)
    bad=[measured(i) for i in range(4)];bad[2]['final_reward']=float('nan')
    with pytest.raises(ValueError):profile_question(c,bad)


def test_profiling_plan_100_and_no_execution_default(tmp_path):
    rows=[{'id':str(i),'source':'nq' if i%2 else 'hotpotqa'} for i in range(600)]
    p=plan(rows,100,42)
    assert len(p)==100 and len({r['id'] for r in p})==100
    with pytest.raises(ValueError,match='disabled'):
        run_local([],SimpleNamespace(execute_local_rollouts=False),tmp_path)


def test_sampling_shortage_no_duplicates_and_qa_only():
    c=[{'id':str(i),'question':'q '+str(i),'source':'nq','golden_answers':['x'],'raw_record':{'fixture':True}} for i in range(20)]
    profiles=[{'id':str(i),'bucket':'medium' if i<18 else 'protocol_failure'} for i in range(20)]
    selected,uniform,r=sample_profiles(c,profiles,count=30)
    assert len(selected)==18 and len(uniform)==18 and r['shortfall']==12
    assert r['actual_ratios']['medium']==1 and len({row['id'] for row in selected})==18
    assert all('rollouts' not in row and 'raw_record' not in row for row in selected+uniform)


def test_flat_training_reward_is_not_em_flatness_and_masks_padding():
    stats=grpo_metrics([0.,.4],[[1.,999],[-1.,999]],['same','same'],masks=[[1,0],[1,0]],em=[0.,0.])
    assert stats['flat_reward_group_rate']==0 and stats['all_zero_reward_group_rate']==0
    assert stats['advantage_mean']==0 and stats['advantage_positive_ratio']==.5
    assert stats['advantage_std']==pytest.approx(2**.5)
    flat=grpo_metrics([0.,0.],[[0],[0]],['a','a'],em=[0,1])
    assert flat['flat_reward_group_rate']==1 and flat['all_success_em_group_rate']==0


def test_group_uid_reordering_and_unknown_em():
    a=grpo_metrics([0,1,0,1],[[0],[0],[0],[0]],['a','b','a','b'])
    assert a['groups']==2 and a['flat_reward_group_rate']==1 and a['all_success_em_group_rate'] is None


def test_unavailable_metrics_and_disabled_kl_placeholder():
    m=optimization_metrics({'actor/kl_loss':0.,'actor/pg_loss':.2})
    assert m['kl']['value'] is None and m['policy_loss']['value']==.2
    assert m['clip_fraction']['status']=='unavailable'


def test_behavior_count_and_masked_observations_not_success():
    messages=trajectory()['messages'][2:]
    p=protocol_metrics(messages)
    assert p['tool_calls']==1 and p['invalid_tool_calls']==0 and p['tool_execution_wall_s'] is None
    t=trajectory_metrics('<think>x</think><answer>Paris</answer>',['Paris'],messages=messages)
    assert t['raw_em']==1 and t['final_reward'] is None and t['failed_rollout'] is None


def test_runtime_tracking_preserves_original_call_and_separates_scopes(monkeypatch):
    from mini.observability import patch
    events=[]; calls=[]
    monkeypatch.setattr('mini.observability.emit',lambda kind,**data:events.append((kind,data)))
    class Tracking:
        def log(self,data,step,*a,**kw):
            calls.append((data,step)); return 'original-result'
    patch(SimpleNamespace(__name__='verl.utils.tracking',Tracking=Tracking))
    metrics={'actor/pg_loss':.3,'actor/kl_loss':0.,'val-core/em':.4}
    assert Tracking().log(metrics,2)=='original-result'
    assert calls==[(metrics,2)]
    assert events[0][1]['scope']=='training_optimization'
    assert events[0][1]['optimization']['kl']['value'] is None
    assert events[1][1]['scope']=='validation'
    assert metrics=={'actor/pg_loss':.3,'actor/kl_loss':0.,'val-core/em':.4}


def test_runtime_agent_observer_preserves_outputs_and_measured_latency(monkeypatch):
    from mini.observability import patch
    events=[]
    monkeypatch.setattr('mini.observability.emit',lambda kind,**data:events.append((kind,data)))
    sentinel=object()
    class Worker:
        tokenizer=SimpleNamespace(decode=lambda *a,**kw:'<think>x</think><answer>Paris</answer>')
        async def _agent_loop_postprocess(self,output,validate,**kw):return sentinel
    patch(SimpleNamespace(__name__='verl.experimental.agent_loop.agent_loop',AgentLoopWorker=Worker))
    output=SimpleNamespace(response_ids=[1,2],response_mask=[1,0],reward_score=.5,
        extra_fields={'hit_max_context':False},metrics=SimpleNamespace(tool_calls=.12,generate_sequences=.8))
    coroutine=Worker()._agent_loop_postprocess(output,True,reward_model={'ground_truth':{'golden_answers':['Paris']}})
    # Both coroutines complete synchronously; avoid Windows asyncio socket-pair
    # creation in the restricted CPU sandbox.
    with pytest.raises(StopIteration) as completed:
        coroutine.send(None)
    result=completed.value.value
    assert result is sentinel and events[0][1]['scope']=='validation'
    assert events[0][1]['metrics']['tool_execution_wall_s']==.12
    assert output.response_ids==[1,2] and output.response_mask==[1,0]
