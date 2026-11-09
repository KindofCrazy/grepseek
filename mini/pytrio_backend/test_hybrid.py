"""Routing and full sixteen-branch scheduling regressions without model calls."""
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from types import SimpleNamespace

from mini.pytrio_backend.config import Config
from mini.pytrio_backend.rollout import rollout_group
from mini.pytrio_backend.search import Search
from mini.pytrio_backend.tracking import Budget


class Log:
    def event(self,*args,**kwargs): pass
    def metrics(self,*args,**kwargs): pass


def test_pace_fallback_uses_sandboxed_shards_only_for_eligible_commands():
    search=Search.__new__(Search)
    search.backend='pace'
    search.corpus='/full/corpus'
    search.shards='/verified/shards'
    search.timeout=10
    eligible='rg -F "rare title" corpus.jsonl | head -n 3'
    unsupported='rg -nF "rare title" corpus.jsonl | head -n 3'
    accelerated=search.argv(eligible)
    native=search.argv(unsupported)
    assert '/worker.py' in accelerated and '/verified/shards' in accelerated
    assert native[-3:]==['/bin/bash','-c',unsupported.replace('corpus.jsonl','wiki_corpus.jsonl')]
    assert '--clearenv' in accelerated and '--unshare-all' in accelerated
    search.shards=''
    assert '/worker.py' not in search.argv(eligible)


def test_four_groups_can_supply_sixteen_continuation_rpcs():
    from mini.pytrio_backend.data import preflight
    tok,_=preflight(Config())
    cfg=Config(concurrency=4,question_concurrency=4,sample_concurrency=16,search_concurrency=4)
    log=Log(); budget=Budget(cfg,log)
    barrier=threading.Barrier(16)
    tool=tok.encode('<think>search</think><tool_call>{"name":"shell","arguments":{"command":"head -n 1 corpus.jsonl"}}</tool_call>',add_special_tokens=False)
    answer=tok.encode('<think>done</think><answer>a</answer>',add_special_tokens=False)
    class Sampler:
        def sample(self,prompt,num_samples,sampling_params,return_text):
            initial=num_samples==4
            tokens=tool if initial else answer
            def result(**kwargs):
                if not initial:
                    barrier.wait(10)
                return SimpleNamespace(input_tokens=len(prompt),output_tokens=len(tokens)*num_samples,
                    sequences=[SimpleNamespace(tokens=tokens,logprobs=[-1.]*len(tokens),stop_reason='stop')
                               for _ in range(num_samples)])
            return SimpleNamespace(result=result)
    search=SimpleNamespace(log=None,execute=lambda command:{'command':command,'stdout':'evidence',
        'stderr':'','exit_code':0,'timed_out':False,'truncated':False,'information_lines':['evidence']})
    def generate(serial):
        return rollout_group({'id':str(serial),'question':'a?','golden_answers':['a']},serial,0,
                             Sampler(),tok,search,cfg,budget,log)
    groups=list(budget.scheduler.ordered(generate,range(1,5)))
    assert len(groups)==4 and all(len(g)==4 for g in groups)
    assert [t.sample_index for g in groups for t in g]==[0,1,2,3]*4
    assert all(t.finish_reason=='answer' and t.tool_count==1 for g in groups for t in g)
    stats=budget.scheduler.snapshot()
    assert stats['sample_peak']==16 and stats['sample_calls']==20
    assert stats['sample_active']==stats['search_active']==0


def test_periodic_audit_does_not_block_unrelated_accelerated_results(monkeypatch):
    from mini.pytrio_backend import search as module
    search=Search.__new__(Search)
    search.audit_every=2; search.audit_count=1
    search.audit_lock=threading.Lock(); search.audit_gate=threading.BoundedSemaphore(1)
    search.lock=threading.RLock(); search.counts=Counter(); search.log=None
    search.corpus='unused'; search.tokenizer=None
    search.timeout=10; search.max_tokens=128; search.max_bytes=4096
    began=threading.Event(); release=threading.Event()
    payload={'stdout':'same'}
    def compare(command):
        began.set()
        assert release.wait(5)
        return payload
    monkeypatch.setattr(module,'Search',lambda *args,**kwargs:SimpleNamespace(_execute=compare))
    with ThreadPoolExecutor(max_workers=2) as pool:
        audit=pool.submit(search._audit,'audited',payload,'sharded')
        try:
            assert began.wait(2)
            unrelated=pool.submit(search._audit,'not audited',payload,'pace')
            unrelated.result(timeout=2)
        finally:
            release.set()
        audit.result(timeout=2)
    assert search.counts['native_audits']==1 and search.counts['audit_mismatches']==0
