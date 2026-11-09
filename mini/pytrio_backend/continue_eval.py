"""Resume immutable-policy evaluation across any number of saved segments."""
import argparse
from contextlib import closing
import json
from pathlib import Path

from mini.common import ROOT, read_jsonl
from mini.curation_utils import file_hash
from mini.diagnostics import aggregate_trajectories
from mini.pytrio_backend.benchmark_report import build
from mini.pytrio_backend.config import Config
from mini.pytrio_backend.data import preflight
from mini.pytrio_backend.eval_history import history, recovery
from mini.pytrio_backend.rollout import rollout_group
from mini.pytrio_backend.runtime import read_bundle, service
from mini.pytrio_backend.search import Search, pace_options
from mini.pytrio_backend.tracking import Budget, BudgetStop, RunLog, json_write


def verify_sources(prior, migration=None):
    provenance = json.loads((prior/'provenance.json').read_text())
    receipt = json.loads(migration.read_text()) if migration else {}
    if receipt:
        if file_hash(ROOT/receipt['validation_report']) != receipt['validation_sha256']:
            raise ValueError('Acceleration validation receipt changed')
    changed = {}
    for section in ('code_sha256','shared_source_sha256'):
        for name, digest in provenance[section].items():
            if name == 'mini/pytrio_backend/continue_eval.py':
                continue
            current = file_hash(ROOT/name)
            if current == digest:
                continue
            allowed = receipt.get('changes',{}).get(name,{})
            if (section != 'code_sha256' or digest not in allowed.get('before',[]) or
                    current != allowed.get('after')):
                raise ValueError(f'Original evaluation source changed without validated migration: {name}')
            changed[name] = {'before':digest,'after':current}
    return changed


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--schedule',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--migration',type=Path)
    for field in ('question_concurrency','sample_concurrency','search_concurrency'):
        parser.add_argument('--'+field.replace('_','-'),type=int)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    config = Config(**json.loads((args.prior/'config.json').read_text())).validate()
    if config.stage != 'eval' or config.resume:
        raise ValueError('Only immutable-policy eval runs can use this continuation')
    changed = verify_sources(args.prior,args.migration)
    provenance = json.loads((args.prior/'provenance.json').read_text())
    if file_hash(Path(config.data)/'sft_manifest.json') != provenance['data_manifest_sha256']:
        raise ValueError('Frozen data snapshot changed')
    expected = Path(config.data)/(config.eval_file or 'benchmark30/test.jsonl')
    if file_hash(args.schedule) != file_hash(expected):
        raise ValueError('Schedule differs from original frozen evaluation')
    for field in ('question_concurrency','sample_concurrency','search_concurrency'):
        value = getattr(args,field)
        if value is not None:
            setattr(config,field,value)
    config.validate()
    chain = history(args.prior)
    prior = [e['trajectory'] for _,events in chain for e in events if e['kind']=='trajectory']
    completed = {t['question_id'] for t in prior}
    scheduled = read_jsonl(args.schedule)
    if prior:
        build(prior,[row for row in scheduled if row['id'] in completed])
    used = {}
    uncertain = []
    for path,events in chain:
        used, pending = recovery(events,used)
        uncertain.extend({'run':str(path),'request':row} for row in pending)
    jobs = [(serial,row) for serial,row in enumerate(scheduled,1) if row['id'] not in completed]
    print(json.dumps({'completed':len(prior),'remaining':len(jobs),'prior_tokens':used,
        'uncertain_requests':len(uncertain),'question_concurrency':config.question_concurrency,
        'execute':args.execute}),flush=True)
    if not args.execute:
        return
    tok,_ = preflight(config)
    log = RunLog(args.out,config)
    json_write(args.out/'continuation.json',{
        'prior':str(args.prior),'prior_events_sha256':file_hash(args.prior/'events.jsonl'),
        'schedule_sha256':file_hash(args.schedule),'completed_question_ids':sorted(completed),
        'uncertain_requests_conservatively_charged':uncertain,
        'same_original_serial_seed_and_policy':True,'source_migration':changed,
        'migration_receipt_sha256':file_hash(args.migration) if args.migration else None})
    budget = Budget(config,log,used)
    result = None
    try:
        fresh = []
        weights = read_bundle(config.initial_checkpoint)['sampler_weights'] if config.initial_checkpoint else None
        if jobs:
            svc = service(config,log)
            sampler = svc.create_sampling_client(config.model,model_path=weights)
            search = Search(config.corpus,tok,config.tool_timeout,config.tool_tokens,log=log,
                cache_entries=config.search_cache_entries,cache_bytes=config.search_cache_bytes,
                backend=config.search_backend,shards=config.shards,pace_dir=config.pace_dir,
                audit_every=config.pace_audit_every,pace_head_budget=pace_options(config))
            def generate(item):
                serial,row = item
                return rollout_group(row,serial,0,sampler,tok,search,config,budget,log,
                                     group_size=1,temperature=0.)
            with closing(budget.scheduler.completed(generate,jobs)) as groups:
                for group in groups:
                    fresh.extend(group)
                    log.event('evaluation_progress',completed=len(prior)+len(fresh),total=len(scheduled))
        combined = prior+[t.record() for t in fresh]
        build(combined,scheduled)
        result = {'status':'completed','split':'pilot','sampler_weights':weights,
            'previous_questions':len(prior),'new_questions':len(fresh),
            'metrics':aggregate_trajectories([t['metrics'] for t in combined]),
            'budget':budget.snapshot(),'quality_improvement_claimed':False}
        log.event('evaluation_summary',**result)
    except BaseException as error:
        result = {'status':'stopped' if isinstance(error,(BudgetStop,KeyboardInterrupt)) else 'failed',
                  'error_type':type(error).__name__,'budget':budget.snapshot()}
        log.event('run_stopped',**result)
        raise
    finally:
        if result is not None:
            json_write(args.out/'result.json',result)
        log.finish()


if __name__=='__main__':
    main()
