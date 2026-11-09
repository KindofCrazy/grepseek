import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mini.common import read_jsonl
from mini.pytrio_backend.config import Config
from mini.pytrio_backend.data import Cursor, preflight, supervised
from mini.pytrio_backend.dynamic import classify, collect
from mini.pytrio_backend.loss import ppo_batch
from mini.pytrio_backend.rollout import observation_suffix, prompt_tokens
from mini.pytrio_backend.tracking import Budget, BudgetStop


class Log:
    def __init__(self): self.events = []
    def event(self, kind, **values): self.events.append((kind,values))
    def metrics(self, *args): pass


@pytest.fixture(scope='module')
def tokenizer():
    return preflight(Config())[0]


def test_sft_shift_and_environment_mask(tokenizer):
    from mini.retokenize import measure_sft
    row = read_jsonl(Path(Config().data)/'sft_selected.jsonl')[0]
    _, tokens, mask = measure_sft(tokenizer,row,12288)
    datum = supervised(tokenizer,row,12288)
    assert datum.model_input.to_ints() == tokens[:-1]
    assert datum.loss_fn_inputs['target_tokens'].data == tokens[1:]
    assert datum.loss_fn_inputs['weights'].data == mask[1:]
    assert 0 < sum(mask) < len(mask)


def test_template_suffix_preserves_raw_generation(tokenizer):
    prefix = [{'role':'user','content':'a question'}]
    text = '<think>\nsearch\n</think>\n\n<tool_call>{"name":"shell","arguments":{"command":"head -n 1 corpus.jsonl"}}</tool_call>'
    payload = {'stdout':'a passage','exit_code':0}
    initial = prompt_tokens(tokenizer,prefix)
    completion = tokenizer.encode(text.removeprefix('<think>\n')+'<|im_end|>',add_special_tokens=False)
    assembled = initial+completion+observation_suffix(tokenizer,payload,completion)
    canonical = prompt_tokens(tokenizer,prefix+[{'role':'assistant','content':text},
                                                {'role':'tool','content':json.dumps(payload,ensure_ascii=False)}])
    assert tokenizer.decode(assembled) == tokenizer.decode(canonical)
    assert assembled[:len(initial)+len(completion)] == initial+completion


def test_ppo_gradient_mask_and_clipping():
    trajectories = [SimpleNamespace(tokens=[1,2,3,4],mask=[0,0,1,1],old_logprobs=[0,0,-1,-1],advantage=a)
                    for a in [1.,-1.]]
    data,objective = ppo_batch(trajectories,.2)
    current = [torch.tensor([-50.,-.5,-1.],requires_grad=True),
               torch.tensor([-50.,-1.5,-1.],requires_grad=True)]
    loss,metrics = objective(data,current)
    loss.backward()
    assert metrics['loss_tokens'] == 4
    assert metrics['clip_fraction'] == .5
    assert current[0].grad[0] == current[1].grad[0] == 0
    assert current[0].grad[1] == current[1].grad[1] == 0
    assert current[0].grad[2].item() == -.25
    assert current[1].grad[2].item() == .25
    assert set(data[0].loss_fn_inputs) == {'target_tokens'}


def test_dynamic_actual_f1_and_bounded_refill():
    config = Config(effective_groups=2,sampling_rounds=2,max_candidate_groups=4)
    def group(values):
        from mini.diagnostics import trajectory_metrics
        return [SimpleNamespace(reward=v,advantage=0.,mask=[0,1],tokens=[1,2],metrics=trajectory_metrics(
            '<think>x</think><answer>a</answer>',['a'],training_reward=v)) for v in values]
    assert classify(group([0]*4),config)[1] == 'all_zero'
    assert classify(group([.5]*4),config)[1] == 'constant_positive_or_near_flat'
    varied = group([0,.25,.5,.75])
    assert classify(varied,config)[1] is None
    assert abs(sum(t.advantage for t in varied)) < 1e-6
    cursor = Cursor([{'id':str(i)} for i in range(10)],42)
    with pytest.raises(BudgetStop,match='insufficient_nonflat'):
        collect(cursor,config,Log(),lambda row,serial:group([0]*4),0)
    assert sum(cursor.draws.values()) == 4


def test_cursor_resume_and_budget_reservations():
    rows = [{'id':str(i)} for i in range(7)]
    cursor = Cursor(rows,42)
    cursor.take(9)
    restored = Cursor(rows,42,json.loads(json.dumps(cursor.state())))
    assert cursor.take(12) == restored.take(12)
    log = Log()
    budget = Budget(Config(max_sample_tokens=10),log)
    def nested():
        with pytest.raises(BudgetStop):
            budget.call('concurrent',{'sample':5},lambda:None)
        raise RuntimeError('unknown remote outcome')
    with pytest.raises(RuntimeError):
        budget.call('first',{'sample':6},nested)
    assert budget.used['sample'] == 6
    assert not any(budget.reserved.values())
    with pytest.raises(BudgetStop):
        budget.call('next',{'sample':5},lambda:None)


def test_readonly_real_corpus_search(tokenizer):
    from mini.pytrio_backend.search import Search
    if not Path(Config().corpus,'wiki_corpus.jsonl').is_file():
        pytest.skip('Run real corpus integration in the prepared WSL environment')
    search = Search(Config().corpus,tokenizer,timeout=15,max_tokens=128)
    observed = search.execute('head -n 1 corpus.jsonl')
    assert observed['exit_code'] == 0 and 'Evan' in observed['stdout']
    assert search.execute('cat /home/grepseek/.pytrio/config.toml')['exit_code'] != 0
    assert search.execute('head -n 1 /etc/passwd')['exit_code'] != 0
    assert search.execute('curl https://example.com')['exit_code'] == -2
    limited = Search(Config().corpus,tokenizer,timeout=15,max_tokens=32,max_bytes=4096)
    assert limited.execute('cat corpus.jsonl')['truncated']


def test_reward_gate_does_not_repair_nested_answer():
    from mini.pytrio_backend.rollout import Trajectory, score
    text = '<think>found evidence</think><tool_call><answer>Paris</answer></tool_call>'
    traj = Trajectory('q','nq','capital?', ['Paris'],1,0,0,
                      tokens=[1,2],mask=[0,1],old_logprobs=[0,-1],
                      initial_length=1,messages=[{'role':'assistant','content':text}],
                      turns=[{'stop_reason':'stop'}])
    score(traj,Config())
    assert traj.metrics['raw_f1'] == 1.
    assert not traj.metrics['format_pass']
    assert traj.reward == 0.


def test_refill_keeps_full_mixed_groups_and_policy():
    from mini.diagnostics import trajectory_metrics
    config = Config(effective_groups=2,sampling_rounds=3,max_candidate_groups=6)
    seen = []
    def generate(row,serial):
        seen.append((row['id'],serial))
        rewards = [0]*4 if serial <= 2 else [0,.5,1,1]
        return [SimpleNamespace(reward=r,advantage=0.,mask=[0,1],tokens=[1,2],metrics=trajectory_metrics(
            '<think>x</think><answer>a</answer>',['a'],training_reward=r)) for r in rewards]
    log = Log()
    selected,serial,stats = collect(Cursor([{'id':str(i)} for i in range(10)],42),config,log,generate,7)
    assert len(selected) == 8 and serial == 4
    assert stats['policy_version'] == 7 and stats['refill_rounds'] == 2
    assert stats['candidate_groups'] == 4 and stats['effective_groups'] == 2
    assert stats['drop_reasons'] == {'all_zero':2}
    assert all(t.advantage != 0 for t in selected)


def test_uncertain_optimizer_checkpoint_refuses_resume():
    from mini.pytrio_backend.runtime import trainer
    with pytest.raises(ValueError,match='uncertain optimizer'):
        trainer(None,None,Config(),Log(),{'resume_safe':False})
    with pytest.raises(ValueError,match='100 successful'):
        Config(stage='rl',steps=101,initial_checkpoint='checkpoint.json').validate()


def test_resume_counts_requests_after_periodic_checkpoint(tmp_path):
    from mini.pytrio_backend.runtime import read_bundle
    (tmp_path/'checkpoint.json').write_text(json.dumps({'train_state':'trio://test',
                                                       'budget':{'tokens':{'train':100}}}))
    (tmp_path/'result.json').write_text(json.dumps({'budget':{'tokens':{'train':120,'sample':50}}}))
    assert read_bundle(tmp_path/'checkpoint.json')['budget']['tokens'] == {'train':120,'sample':50}


def test_bounded_cache_and_same_command_coalescing(tokenizer,monkeypatch,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from mini.pytrio_backend.search import Search
    import time
    # This is a cache/coalescing unit test: physical execution is replaced below.
    # Stub the preparation metadata rather than requiring a 14GB corpus here.
    corpus=tmp_path/'corpus';corpus.mkdir()
    (corpus/'wiki_corpus.jsonl').write_text('cache test fixture\n')
    (corpus/'corpus_manifest.json').write_text(json.dumps({'synthetic':False,
        'output_lines':21015324,'output_sha256':'43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d',
        'output_bytes':(corpus/'wiki_corpus.jsonl').stat().st_size}))
    search = Search(corpus,tokenizer,cache_entries=1,cache_bytes=2048)
    calls = []
    def execute(command):
        calls.append(command)
        time.sleep(.05)
        return {'command':command,'stdout':'evidence','stderr':'','exit_code':0,
                'truncated':False,'timed_out':False,'information_lines':['evidence']}
    monkeypatch.setattr(search,'_execute',execute)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(search.execute,['same']*4))
    assert len(calls)==1 and len(values)==4
    values[0]['stdout']='mutated'
    assert search.execute('same')['stdout']=='evidence'
    search.execute('different')
    search.execute('same')
    assert len(calls)==3 and search.stats()['cache_entries']==1
    assert search.stats()['coalesced_requests']==3


def test_accelerator_falls_back_for_semantically_unsafe_commands():
    from mini.pytrio_backend.acceleration import eligible
    assert eligible('rg -F "Verdi" corpus.jsonl | head -n 3')
    assert eligible('rg -iF "Verdi" corpus.jsonl | grep -F "film" | wc -l')
    assert not eligible('rg -nF "Verdi" corpus.jsonl | head -n 3')
    assert not eligible('rg -F "Verdi" corpus.jsonl | sort | head -n 3')
    assert not eligible('rg -F "Verdi" other.jsonl | head -n 3')
    assert not eligible('rg -F "Verdi" corpus.jsonl | grep -F "film" other.jsonl | head -n 3')
    assert not eligible('rg "Verdi.*" corpus.jsonl | head -n 3')


def test_operator_stop_file_prevents_submission(tmp_path):
    log = Log()
    log.out = tmp_path
    (tmp_path/'STOP').touch()
    budget = Budget(Config(),log)
    with pytest.raises(BudgetStop,match='operator_stop_file'):
        budget.call('must not submit',{},lambda:pytest.fail('submitted after STOP'))
    assert budget.serial==0


def test_sample_uses_reported_output_work_and_refuses_overrun():
    import pytrio as trio
    log = Log()
    budget = Budget(Config(),log)
    response = SimpleNamespace(input_tokens=2,output_tokens=3,sequences=[SimpleNamespace(tokens=[3,4])])
    sampler = SimpleNamespace(sample=lambda *args,**kwargs:SimpleNamespace(result=lambda **kwargs:response))
    budget.sample(sampler,[1,2],1,trio.SamplingParams(max_tokens=4))
    assert budget.used['sample']==3 and budget.used['prefill']==2
    response.sequences[0].tokens=[3]*5
    with pytest.raises(ValueError,match='reservation'):
        budget.sample(sampler,[1,2],1,trio.SamplingParams(max_tokens=4))
    assert budget.used['sample']==7  # the failed call retains its worst-case reservation


def test_resume_budget_exposes_incremental_cost_without_double_count():
    budget = Budget(Config(),Log(),{'sample':100})
    assert budget.snapshot()['estimated_incremental_cny']==0
    budget.call('next',{'sample':10},lambda:None)
    assert budget.snapshot()['tokens']['sample']==110
    assert budget.snapshot()['incremental_tokens']=={'sample':10}
