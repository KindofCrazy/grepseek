"""Replay real training commands without invoking any model; compare full tool payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

from mini.pytrio_backend.acceleration import eligible
from mini.pytrio_backend.config import Config
from mini.pytrio_backend.data import preflight
from mini.pytrio_backend.search import Search
from mini.pytrio_backend.tracking import json_write


class Log:
    def __init__(self): self.events=[]
    def event(self,kind,**values): self.events.append({'kind':kind,**values})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pace-dir',default='data/pace-full')
    parser.add_argument('--events',type=Path,default=Path('mini/runs/pytrio-rl-smoke-20261107b/events.jsonl'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--count',type=int,default=12)
    args=parser.parse_args()
    if args.count<1 or args.out.exists():
        raise ValueError('Positive count and fresh report path required')
    cfg=Config()
    tok,_=preflight(cfg)
    commands=[]
    for line in args.events.open():
        row=json.loads(line)
        if row['kind']!='trajectory': continue
        for message in row['trajectory']['messages']:
            if message['role']!='tool': continue
            command=message['tool_result']['command']
            if eligible(command) and command not in commands:
                commands.append(command)
    commands=commands[:args.count]
    commands.extend(['rg -F "__pace_acceptance_no_match_20261107__" corpus.jsonl | head -n 3',
                     'rg -nF "Giuseppe Verdi" corpus.jsonl | head -n 3'])
    log=Log()
    baseline=Search(cfg.corpus,tok,cache_entries=0)
    started=time.monotonic()
    accelerated=Search(cfg.corpus,tok,cache_entries=0,backend='pace',pace_dir=args.pace_dir,log=log)
    result={'status':'running','model_api_requests':0,'cache_enabled':False,
            'source_events_sha256':hashlib.sha256(args.events.read_bytes()).hexdigest(),
            'index_load_and_verify_s':time.monotonic()-started,'cases':[]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for command in commands:
        times=[]
        payloads=[]
        for search in (baseline,accelerated):
            start=time.monotonic()
            payloads.append(search.execute(command))
            times.append(time.monotonic()-start)
        decision=next((r for r in reversed(log.events) if r['kind']=='pace_execution' and r['command']==command),None)
        case={'command':command,'sequential_s':times[0],'pace_s':times[1],
              'byte_exact':payloads[0]==payloads[1], 'pace_decision':decision,
              'payload_sha256':[hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() for p in payloads]}
        result['cases'].append(case)
        result['peak_self_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        json_write(args.out,result)
        print(json.dumps({k:case[k] for k in ('command','sequential_s','pace_s','byte_exact')}),flush=True)
        if not case['byte_exact']:
            result['status']='failed'
            json_write(args.out,result)
            raise AssertionError('PACE changed a real training command observation')
    before=sum(c['sequential_s'] for c in result['cases'])
    after=sum(c['pace_s'] for c in result['cases'])
    result.update(status='passed',all_payloads_byte_exact=True,sequential_total_s=before,
                  pace_total_s=after,speedup=before/after,
                  pace_served=sum(bool(c['pace_decision'] and c['pace_decision']['served']) for c in result['cases']),
                  search_stats=accelerated.stats())
    json_write(args.out,result)
    print(json.dumps({k:result[k] for k in ('status','pace_served','speedup','peak_self_rss_kib')}),flush=True)


if __name__=='__main__': main()
