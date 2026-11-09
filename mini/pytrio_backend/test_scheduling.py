from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from mini.pytrio_backend.config import Config
from mini.pytrio_backend.data import Cursor
from mini.pytrio_backend.dynamic import collect
from mini.pytrio_backend.eval_history import history, read_events, recovery
from mini.pytrio_backend.scheduling import Scheduler


class Log:
    def __init__(self): self.events=[]
    def event(self,kind,**values): self.events.append((kind,values))
    def metrics(self,*args): pass


def test_shared_limits_and_order_despite_completion_order():
    scheduler=Scheduler(Config(question_concurrency=4,sample_concurrency=2,search_concurrency=1),Log())
    finished=[]
    def work(index):
        with scheduler.slot('sample'):
            time.sleep(.03 if index==0 else .005)
        with scheduler.slot('search'):
            time.sleep(.005)
        finished.append(index)
        return index
    assert list(scheduler.ordered(work,range(8)))==list(range(8))
    stats=scheduler.snapshot()
    assert stats['sample_peak']==2 and stats['search_peak']==1
    assert stats['sample_active']==stats['search_active']==0
    assert stats['sample_calls']==stats['search_calls']==8
    assert finished.index(1)<finished.index(0)


def test_failure_drains_inflight_and_preserves_completed_work():
    scheduler=Scheduler(Config(question_concurrency=2),Log())
    completed=[]
    ready=threading.Event()
    def work(index):
        if index==1:
            ready.wait(2)
            raise ValueError('infrastructure failure')
        with scheduler.slot('sample'):
            ready.set()
            time.sleep(.03)
        completed.append(index)
        return index
    with pytest.raises(ValueError,match='infrastructure failure'):
        list(scheduler.ordered(work,range(100)))
    assert completed==[0]
    assert scheduler.snapshot()['sample_active']==0


def test_eval_window_replaces_fast_questions_without_waiting_for_straggler():
    scheduler=Scheduler(Config(question_concurrency=2),Log())
    later_started=threading.Event()
    def work(index):
        if index==0:
            assert later_started.wait(2)
        if index==2:
            later_started.set()
        return index
    result=list(scheduler.completed(work,range(5)))
    assert sorted(result)==list(range(5))
    assert result.index(1)<result.index(0)


@pytest.mark.parametrize('method',['ordered','completed','ordered_refill'])
def test_consumer_failure_closes_and_joins_workers(method):
    from contextlib import closing
    scheduler=Scheduler(Config(question_concurrency=2),Log())
    barrier=threading.Barrier(2)
    def work(index):
        with scheduler.slot('sample'):
            barrier.wait(2)
            time.sleep(.005 if index==0 else .03)
        return index
    with pytest.raises(ValueError,match='consumer failed'):
        arguments=(work,[0,1],lambda result:[]) if method=='ordered_refill' else (work,[0,1])
        with closing(getattr(scheduler,method)(*arguments)) as groups:
            next(groups)
            raise ValueError('consumer failed')
    assert scheduler.snapshot()['sample_active']==0


def test_dynamic_acceptance_and_serials_do_not_follow_fastest_group():
    from mini.diagnostics import trajectory_metrics
    config=Config(effective_groups=2,sampling_rounds=3,max_candidate_groups=6,question_concurrency=2)
    log=Log()
    finished=[]
    def generate(row,serial):
        time.sleep(.025 if serial%2 else .001)
        finished.append(serial)
        rewards=[0]*4 if serial==1 else [0,.25,.5,1]
        return [SimpleNamespace(group_serial=serial,reward=r,advantage=0.,mask=[0,1],tokens=[1,2],
            metrics=trajectory_metrics('<think>x</think><answer>a</answer>',['a'],training_reward=r))
            for r in rewards]
    selected,serial,stats=collect(Cursor([{'id':str(i)} for i in range(10)],42),config,log,
                                 generate,7,scheduler=Scheduler(config,log))
    assert finished[:2]==[2,1]
    assert [t.group_serial for t in selected]==[2]*4+[3]*4
    assert serial==3 and stats['candidate_groups']==3
    assert [(e['group_serial'],e['accepted']) for k,e in log.events if k=='dynamic_group']==[(1,False),(2,True),(3,True)]


def fake_group(serial, flat=False):
    from mini.diagnostics import trajectory_metrics
    return [SimpleNamespace(group_serial=serial,reward=r,advantage=0.,mask=[0,1],tokens=[1,2],
            metrics=trajectory_metrics('<think>x</think><answer>a</answer>',['a'],training_reward=r))
            for r in ([0]*4 if flat else [0,.25,.5,1])]


def test_dynamic_refill_starts_before_slow_parent_round_finishes():
    config=Config(effective_groups=2,sampling_rounds=3,max_candidate_groups=6,question_concurrency=2)
    log=Log()
    replacement_started=threading.Event()
    def generate(row,serial):
        if serial==2:
            assert replacement_started.wait(5), 'Refill waited for the slow parent round'
        if serial==3:
            replacement_started.set()
        return fake_group(serial,flat=serial==1)
    selected,serial,stats=collect(Cursor([{'id':str(i)} for i in range(10)],42),config,log,
                                 generate,7,scheduler=Scheduler(config,log))
    assert [t.group_serial for t in selected]==[2]*4+[3]*4
    assert serial==3 and stats['overlapped_refill_groups']==1
    assert stats['allocated_groups']==stats['candidate_groups']==3


@pytest.mark.parametrize('width',[1,2,4])
@pytest.mark.parametrize('flat,cap,expected,rounds',[
    (set(),6,[1,2],[1,1]),
    ({1},6,[2,3],[1,1,2]),
    ({1,2,3},6,[4,5],[1,1,2,2,3]),
    (set(range(1,7)),6,[],[1,1,2,2,3,3]),
    (set(range(1,7)),3,[],[1,1,2]),
])
def test_refill_preserves_round_caps_cursor_and_acceptance(width,flat,cap,expected,rounds):
    from mini.pytrio_backend.tracking import BudgetStop
    config=Config(effective_groups=2,sampling_rounds=3,max_candidate_groups=cap,question_concurrency=width)
    rows=[{'id':str(i)} for i in range(20)]
    cursor=Cursor(rows,42)
    log=Log()
    generate=lambda row,serial:fake_group(serial,serial in flat)
    if expected:
        selected,serial,stats=collect(cursor,config,log,generate,7,scheduler=Scheduler(config,log))
        assert [t.group_serial for t in selected]==[s for s in expected for _ in range(4)]
        assert serial==len(rounds) and stats['candidate_groups']==len(rounds)
    else:
        with pytest.raises(BudgetStop,match='insufficient_nonflat'):
            collect(cursor,config,log,generate,7,scheduler=Scheduler(config,log))
    events=[v for k,v in log.events if k=='dynamic_group']
    assert [e['group_serial'] for e in events]==list(range(1,len(rounds)+1))
    assert [e['round'] for e in events]==rounds
    assert [e['question_id'] for e in events]==[r['id'] for r in Cursor(rows,42).take(len(rounds))]
    reference=Cursor(rows,42); reference.take(len(rounds))
    assert cursor.state()==reference.state()


def test_refill_failure_drains_started_replacement_and_preserves_first_error():
    scheduler=Scheduler(Config(question_concurrency=2),Log())
    replacement_started=threading.Event()
    finished=[]
    def work(index):
        if index==1:
            assert replacement_started.wait(5)
            raise ValueError('refill infrastructure failure')
        if index==2:
            replacement_started.set()
            time.sleep(.02)
        finished.append(index)
        return index
    with pytest.raises(ValueError,match='refill infrastructure failure'):
        list(scheduler.ordered_refill(work,[0,1],lambda result:[2] if result==0 else []))
    assert finished==[0,2]
    assert scheduler.snapshot()['sample_active']==0


def segment(path,events,prior=None):
    from mini.curation_utils import file_hash
    path.mkdir()
    (path/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    (path/'config.json').write_text(json.dumps({'stage':'eval','seed':42}))
    if prior:
        (path/'continuation.json').write_text(json.dumps({'prior':str(prior),
            'prior_events_sha256':file_hash(prior/'events.jsonl')}))
    return path


def test_three_segment_recovery_charges_uncertain_once(tmp_path):
    first=segment(tmp_path/'one',[
        {'kind':'request_submitted','request_id':1,'reserved_work':{'sample':200}},
        {'kind':'request_finished','request_id':1,'budget':{'tokens':{'sample':100}}},
        {'kind':'request_submitted','request_id':2,'reserved_work':{'sample':200}}])
    second=segment(tmp_path/'two',[
        {'kind':'request_finished','request_id':1,'budget':{'tokens':{'sample':310}}}],first)
    third=segment(tmp_path/'three',[],second)
    used={}
    for _,events in history(third): used,_=recovery(events,used)
    assert used=={'sample':310}
    (first/'events.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        history(third)


def test_crash_tail_only_is_tolerated(tmp_path):
    path=tmp_path/'events.jsonl'
    path.write_text('{"kind":"trajectory"}\n{"kind":')
    assert len(read_events(path))==1
    path.write_text('{"kind":\n{"kind":"trajectory"}\n')
    with pytest.raises(ValueError,match='Corrupted'):
        read_events(path)


def test_source_migration_requires_exact_validation_receipt(tmp_path,monkeypatch):
    from mini.curation_utils import file_hash
    from mini.pytrio_backend import continue_eval
    source=tmp_path/'backend.py';source.write_text('old')
    before=file_hash(source)
    prior=tmp_path/'prior';prior.mkdir()
    (prior/'provenance.json').write_text(json.dumps({'code_sha256':{'backend.py':before},'shared_source_sha256':{}}))
    source.write_text('new')
    validation=tmp_path/'validation.json';validation.write_text('{"tests":"passed"}')
    receipt=tmp_path/'migration.json'
    receipt.write_text(json.dumps({'validation_report':str(validation),'validation_sha256':file_hash(validation),
        'changes':{'backend.py':{'before':[before],'after':file_hash(source)}}}))
    monkeypatch.setattr(continue_eval,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='without validated migration'):
        continue_eval.verify_sources(prior)
    assert continue_eval.verify_sources(prior,receipt)['backend.py']['before']==before
    validation.write_text('changed')
    with pytest.raises(ValueError,match='receipt changed'):
        continue_eval.verify_sources(prior,receipt)
