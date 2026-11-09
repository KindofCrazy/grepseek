from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import threading
import time

from mini.common import ROOT
from mini.curation_utils import file_hash


def json_write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    if os.name=='posix':
        descriptor = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class RunLog:
    def __init__(self, out, config):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.lock = threading.RLock()
        self.stream = (self.out / 'events.jsonl').open('a', encoding='utf-8')
        self.swan = None
        self.optimizer_updates = 0
        self.stage = config.stage
        json_write(self.out / 'config.json', config.to_dict())
        versions = {}
        for package in ('pytrio', 'transformers', 'torch', 'swanlab'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        metadata = {'versions': versions, 'code_sha256': {
            str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / 'mini/pytrio_backend').glob('*.py'))},
            'shared_source_sha256': {name:file_hash(ROOT/name) for name in (
                'mini/common.py','mini/template.py','mini/retokenize.py','mini/diagnostics.py',
                'inference/agent.py','inference/scoring.py','rl/prompts/system_prompt.txt',
                'rl/grepseek/trainer/verl_integration/reward_function.py',
                'rl/grepseek/trainer/verl_integration/search_corpus_tool.py')},
            'upstream':json.loads((ROOT/'mini/UPSTREAM.json').read_text()),
            'data_manifest_sha256': file_hash(Path(config.data) / 'sft_manifest.json'),
            'corpus_manifest_sha256': file_hash(Path(config.corpus) / 'corpus_manifest.json')
            if (Path(config.corpus) / 'corpus_manifest.json').exists() else None,
            'actual_bill_cny': None, 'server_gpu_memory': None}
        source = ROOT / 'mini/.cache/source-revision.json'
        if source.exists():
            metadata['source_revision'] = json.loads(source.read_text())
        json_write(self.out / 'provenance.json', metadata)
        if config.swanlab != 'disabled':
            import swanlab
            self.swan = swanlab.init(project='grepseek-mini', experiment_name=self.out.name,
                                    mode='online' if config.swanlab == 'cloud' else config.swanlab,
                                    public=False, on_online_error='offline',
                                    config=config.to_dict(), logdir=str(self.out / 'swanlab'))

    def event(self, kind, **values):
        with self.lock:
            row = {'at_utc': datetime.now(timezone.utc).isoformat(), 'kind': kind, **values}
            self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            self.stream.flush()
            if kind == 'trajectory' and self.stage == 'eval':
                os.fsync(self.stream.fileno())

    def metrics(self, step, values):
        self.event('metrics', step=step, values=values)
        if self.swan:
            numeric = {k: float(v) for k, v in values.items()
                       if isinstance(v, (int, float)) and math.isfinite(v)}
            try:
                import swanlab
                swanlab.log(numeric, step=step)
            except Exception as error:
                self.event('logging_error', error_type=type(error).__name__)

    def finish(self):
        if self.swan:
            self.swan.finish()
        self.stream.close()


class BudgetStop(RuntimeError):
    pass


def budget_parent_tokens(config, log):
    """Carry paid historical requests into a fresh weights-only experiment."""
    if not config.budget_parent:
        return None
    path = Path(config.budget_parent)
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') not in ('stopped','completed'):
        raise ValueError('Budget parent must be a terminal result')
    tokens = result.get('budget',{}).get('tokens',{})
    if set(tokens)!= {'prefill','sample','train'} or any(type(v) is not int or v<0 for v in tokens.values()):
        raise ValueError('Invalid inherited budget token accounting')
    if any(result['budget'].get('reserved',{}).values()):
        raise ValueError('Budget parent has unsettled reservations')
    log.event('budget_parent_loaded',path=str(path),sha256=file_hash(path),tokens=tokens,
              estimated_cny=Budget.estimate(tokens),optimizer_inherited=False)
    return tokens


class Budget:
    """Reserve worst-case request work before submission, including discarded groups.

    Costs are uncached token estimates, never invoices. Custom loss counts both
    forward and surrogate backward input work. In-flight/failed calls are not refunded.
    """
    def __init__(self, config, log, initial=None):
        self.config, self.log = config, log
        self.used = Counter(initial or {})
        self.initial = Counter(initial or {})
        self.reserved = Counter()
        self.lock = threading.RLock()
        self.start = time.monotonic()
        self.serial = 0
        from mini.pytrio_backend.scheduling import Scheduler
        self.scheduler = Scheduler(config,log)

    @staticmethod
    def estimate(counts):
        return (counts.get('prefill', 0) * 1.49 + counts.get('sample', 0) * 4.54 +
                counts.get('train', 0) * 4.54) / 1000000

    def snapshot(self):
        with self.lock:
            return {'tokens': dict(self.used), 'reserved': dict(self.reserved),
                    'initial_tokens':dict(self.initial),'incremental_tokens':dict(self.used-self.initial),
                    'estimated_incremental_cny':self.estimate(self.used-self.initial),
                    'estimated_uncached_cny': self.estimate(self.used), 'actual_bill_cny': None,
                    'price_source': 'https://pytrio.com/', 'price_date': '2026-11-07',
                    'estimate_is_invoice': False, 'wall_s': time.monotonic() - self.start}

    def call(self, operation, work, fn, actual=None):
        with self.lock:
            if hasattr(self.log,'out') and (self.log.out/'STOP').exists():
                raise BudgetStop('operator_stop_file')
            if time.monotonic() - self.start >= self.config.max_wall_seconds:
                raise BudgetStop('wall_clock_limit')
            projected = self.used + self.reserved + Counter(work)
            for key in ('prefill', 'sample', 'train'):
                if projected[key] > getattr(self.config, f'max_{key}_tokens'):
                    raise BudgetStop(f'{key}_token_limit')
            if self.estimate(projected) > self.config.max_estimated_cny:
                raise BudgetStop('estimated_cost_limit')
            self.serial += 1
            request_id = self.serial
            self.reserved.update(work)
        self.log.event('request_submitted', request_id=request_id, operation=operation, reserved_work=work)
        started = time.monotonic()
        consumed, status = work, 'uncertain'
        try:
            result = fn()
            consumed = actual(result) if actual else work
            if any(consumed.get(k, 0) > v for k, v in work.items()):
                raise ValueError('Remote work exceeds request reservation')
            status = 'completed'
            return result
        except BaseException as error:
            self.log.event('request_error', request_id=request_id, operation=operation,
                           error_type=type(error).__name__, code=getattr(error, 'code', None))
            raise
        finally:
            with self.lock:
                self.reserved.subtract(work)
                self.used.update(consumed)
                self.log.event('request_finished', request_id=request_id, operation=operation, status=status,
                               elapsed_s=time.monotonic()-started, work=consumed, budget=self.snapshot())

    def sample(self, sampler, tokens, count, params):
        import pytrio as trio
        def actual(response):
            if (response.input_tokens != len(tokens) or len(response.sequences)!=count or
                    any(len(s.tokens)>params.max_tokens for s in response.sequences)):
                raise ValueError('Sampler violated prompt/group/token reservation')
            returned = sum(len(s.tokens) for s in response.sequences)
            reported = response.output_tokens
            self.log.event('sample_accounting',server_input_tokens=response.input_tokens,
                           server_output_tokens=reported,returned_output_tokens=returned,num_samples=count)
            return {'prefill':len(tokens)*count,'sample':max(returned,reported or 0)}
        with self.scheduler.slot('sample'):
            return self.call('sample', {'prefill': len(tokens)*count, 'sample': params.max_tokens*count},
                             lambda: sampler.sample(trio.ModelInput.from_ints(tokens), num_samples=count,
                                                    sampling_params=params, return_text=False).result(timeout=self.config.request_timeout),
                             actual)
