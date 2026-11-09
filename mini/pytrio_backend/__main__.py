"""Explicit --execute is required for billable requests; otherwise validate and print plan."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import traceback

from mini.pytrio_backend.config import Config
from mini.pytrio_backend.data import preflight
from mini.pytrio_backend.runtime import read_bundle, service
from mini.pytrio_backend.tracking import Budget, BudgetStop, RunLog, json_write, budget_parent_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['sft','rl','eval'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--split',choices=['dev','pilot'],default='dev')
    parser.add_argument('--count',type=int,default=2)
    defaults = Config()
    for field in fields(Config):
        if field.name == 'stage':
            continue
        parser.add_argument('--'+field.name.replace('_','-'),type=type(getattr(defaults,field.name)),default=None)
    args = parser.parse_args()
    values = json.loads(args.config.read_text()) if args.config else {}
    values['stage'] = args.stage
    if args.stage == 'rl' and 'lr' not in values:
        values['lr'] = 1e-6
    if args.stage == 'rl' and 'swanlab' not in values:
        values['swanlab'] = 'online'
    for field in fields(Config):
        if field.name != 'stage' and getattr(args,field.name) is not None:
            values[field.name] = getattr(args,field.name)
    if args.stage=='eval':
        if args.temperature is not None and args.temperature!=0:
            parser.error('Evaluation temperature is fixed to zero')
        values['temperature'] = 0.
    config = Config(**values).validate()
    if args.count < 1:
        parser.error('--count must be positive')
    tok, manifest = preflight(config)
    print(json.dumps({'stage':args.stage,'execute':args.execute,'config':config.to_dict(),
                      'verified_data_files':len(manifest['files_sha256'])},ensure_ascii=False),flush=True)
    if not args.execute:
        return 0
    log = RunLog(args.out,config)
    bundle = read_bundle(config.resume) if config.resume else None
    budget = Budget(config,log,bundle['budget']['tokens'] if bundle else budget_parent_tokens(config,log))
    result = None
    exit_code = 0
    try:
        svc = service(config,log)
        if args.stage == 'eval':
            from mini.pytrio_backend.evaluate import run
            result = run(config,tok,svc,log,budget,args.split,args.count)
        else:
            if args.stage == 'sft':
                from mini.pytrio_backend.sft import run
            else:
                from mini.pytrio_backend.rl import run
            result = run(config,tok,svc,log,budget,bundle)
    except BaseException as error:
        stopped = isinstance(error,(BudgetStop,KeyboardInterrupt))
        result = {'status':'stopped' if stopped else 'failed','error_type':type(error).__name__,
                  'error_code':getattr(error,'code',None),'budget':budget.snapshot(),
                  'optimizer_updates':log.optimizer_updates}
        if isinstance(error,BudgetStop):
            result['stop_reason'] = str(error)
        checkpoint = log.out / 'checkpoint.json'
        if checkpoint.exists():
            result['checkpoint'] = read_bundle(checkpoint)
        # SDK exception messages may contain request URLs. Record stack frames,
        # type/code only; never print configuration files or request credentials.
        log.event('run_stopped',**result,frames=traceback.format_tb(error.__traceback__))
        print(f"{result['status']}: {type(error).__name__} {result.get('stop_reason','')}",flush=True)
        exit_code = 2 if stopped else 1
    finally:
        if result is not None:
            # Validation and reload checks happen after the final model save.
            # Carry their consumption into a later exact resume as well.
            latest = log.out/'checkpoint.json'
            if latest.exists():
                saved = read_bundle(latest)
                saved['budget'] = budget.snapshot()
                json_write(latest,saved)
                json_write(log.out/f"checkpoint-{saved['step']}.json",saved)
                result['checkpoint'] = saved
            json_write(log.out/'result.json',result)
        log.finish()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
