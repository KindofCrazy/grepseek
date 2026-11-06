from pathlib import Path
from types import SimpleNamespace
import copy

import pytest

from mini.benchmark30 import SPLITS,select,check_training_files
from mini.common import write_jsonl
from mini.benchmark_metrics import aggregate,report,trajectory,wilson,BudgetedCompletion


def pools(n=40):
    # Synthetic CPU test fixtures only, never exported as actual benchmark data.
    return {s:[{'id':str(i),'question':s+' '+str(i)+' unique'+str(i),'golden_answers':['x']} for i in range(n)] for s in SPLITS}


def record(source='nq', answer='Paris', generated=100, growth=300):
    return {'id':'fixture','source':source,'prediction':answer,'gold_answers':['Paris'],'n_tool_calls':0,
            'finish_reason':'answer','turns':[{'role':'assistant','content':'<think>Known.</think><answer>'+answer+'</answer>','finish_reason':'stop'}],
            'generated_completion_tokens':generated,'context_growth_tokens':growth,'assistant_content_tokens':20}


def test_exact_5x30_stable_and_no_gold_based_sampling():
    original=pools()
    a,stats,_=select(original,[],count=30)
    reversed_pools={k:list(reversed(v)) for k,v in original.items()}
    for values in reversed_pools.values():
        for row in values:row['golden_answers']=['different gold']
    b,_,_=select(reversed_pools,[],count=30)
    assert [r['id'] for r in a]==[r['id'] for r in b]
    assert len(a)==150 and len({r['id'] for r in a})==150
    assert all(r['selected']==30 for r in stats.values())


def test_benchmark_excludes_training_and_does_not_duplicate_fill():
    p=pools(3)
    forbidden=[{'question':r['question']} for r in p['nq']]
    with pytest.raises(ValueError,match='no duplication'):
        select(p,forbidden,count=1)


def test_future_training_guard(tmp_path):
    test=tmp_path/'test.jsonl';train=tmp_path/'train.jsonl'
    write_jsonl(test,[{'question':'Who won?'}]);write_jsonl(train,[{'id':'x','question':'WHO won!'}])
    with pytest.raises(ValueError,match='Frozen benchmark'):
        check_training_files(test,[train])


def test_requested_metrics_and_conditioned_lengths():
    a=record(generated=100,growth=500); b=record(answer='London',generated=900,growth=1200)
    m=aggregate([a,b])
    assert m['exact_match']==.5 and m['token_f1']==.5 and m['format_pass_rate']==1
    assert m['success_generated_tokens']['mean']==100
    assert m['success_context_growth_tokens']['mean']==500 and m['success_count']==1
    assert m['success_generated_tokens']['observed_n']==1


def test_zero_success_and_missing_usage_are_not_zero_length():
    m=aggregate([record(answer='London')])
    assert m['success_count']==0 and m['success_generated_tokens']['mean'] is None
    m=aggregate([record(generated=None)])
    assert m['success_count']==1 and m['success_generated_tokens']['observed_n']==0
    assert m['success_generated_tokens']['mean'] is None


def test_union_abnormality_and_all_questions_in_denominator():
    a=record();bad=record(answer='')
    bad['prediction']=None;bad['finish_reason']='api_error';bad['error']='maximum context length exceeded'
    bad['turns']=[]
    m=aggregate([a,bad])
    assert m['n']==2 and m['exact_match']==.5
    assert m['abnormal_trajectory_rate']==.5 and m['api_error_rate']==.5
    assert m['no_final_answer_rate']==.5 and m['context_truncation_rate']==.5


def test_invalid_command_not_double_counted_on_exit_minus_two():
    r=record();r['n_tool_calls']=1
    r['turns']=[{'role':'assistant','content':'<think>x</think><tool_call>{"name":"shell","arguments":{"command":"rm corpus.jsonl"}}</tool_call>'},
        {'role':'tool','content':'{}','tool_result':{'exit_code':-2,'timed_out':False}},r['turns'][0]]
    assert aggregate([r])['invalid_tool_calls']==1


def test_macro_micro_and_pooled_success_length():
    r=report([record('nq',generated=10),record('hotpotqa',generated=30),record('hotpotqa',answer='London')])
    assert r['macro_equal_dataset_weight']['exact_match']==.75
    assert r['overall']['exact_match']==pytest.approx(2/3)
    assert r['overall']['success_generated_tokens']['mean']==20


def test_wilson_small_sample_uncertainty():
    low,high=wilson(15,30)
    assert .33<low<.34 and .66<high<.67
    assert wilson(0,0) is None


def test_cumulative_response_budget_counts_observations():
    calls=[]
    def api(*args,**kwargs):
        calls.append(kwargs);return SimpleNamespace(usage=SimpleNamespace(completion_tokens=10))
    tokenizer=SimpleNamespace(apply_chat_template=lambda messages,**kwargs:list(range(messages[-1]['length'])))
    meter=BudgetedCompletion(api,tokenizer,128,256,384,42)
    meter.reset('q')
    meter(messages=[{'length':100}],max_tokens=None)
    meter(messages=[{'length':300}],max_tokens=None)
    assert calls[0]['max_tokens']==256 and calls[1]['max_tokens']==56
    assert meter.usage()['completion_tokens']==20
    with pytest.raises(RuntimeError,match='budget exhausted'):
        meter(messages=[{'length':400}],max_tokens=None)
    assert len(calls)==2 and meter.budget_exhausted


def test_missing_api_usage_is_unavailable():
    tokenizer=SimpleNamespace(apply_chat_template=lambda *args,**kwargs:[0])
    meter=BudgetedCompletion(lambda **kwargs:SimpleNamespace(usage=None),tokenizer)
    meter(messages=[{}])
    assert meter.usage()['complete'] is False
