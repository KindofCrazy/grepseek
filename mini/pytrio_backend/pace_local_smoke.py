"""Brief local load check using three cloud-approved SFT observations."""
import argparse
import json
from pathlib import Path
import resource
import time

from mini.pytrio_backend.config import Config
from mini.pytrio_backend.data import preflight
from mini.pytrio_backend.search import Search
from mini.pytrio_backend.tracking import json_write


class Log:
    def __init__(self): self.events=[]
    def event(self,kind,**values): self.events.append({'kind':kind,**values})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--approved',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--offset',type=int,default=0)
    args=parser.parse_args()
    samples=json.loads(args.samples.read_text())
    approved=json.loads(args.approved.read_text())
    assert approved['status']=='passed'
    lookup={(row['trajectory_id'],row['turn']):row for row in samples['cases']}
    selected=[lookup[(row['trajectory_id'],row['turn'])] for row in approved['cases'] if row['pace_served']][args.offset:args.offset+3]
    cfg=Config()
    tok,_=preflight(cfg)
    log=Log()
    start=time.monotonic()
    search=Search(cfg.corpus,tok,backend='pace',pace_dir='data/pace-full',cache_entries=0,log=log)
    report={'load_and_verify_s':time.monotonic()-start,'model_api_requests':0,'cases':[]}
    for case in selected:
        old=case['recorded']
        start=time.monotonic()
        value=search.execute(old['command'])
        same=all(old.get(key)==v for key,v in value.items())
        row={'trajectory_id':case['trajectory_id'],'turn':case['turn'],'command':old['command'],
             'recorded_fields_equal':same,'wall_s':time.monotonic()-start}
        report['cases'].append(row)
        print(json.dumps(row),flush=True)
        if not same: raise AssertionError('Local PACE differs from cloud-approved SFT observation')
    report.update(status='passed',peak_self_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  stats=search.stats(),events=log.events)
    if report['stats'].get('pace_executions')!=len(selected):
        raise AssertionError('Local smoke unexpectedly fell back; inspect before enabling PACE')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    json_write(args.out,report)
    print(json.dumps({k:report[k] for k in ('status','peak_self_rss_kib','stats')}),flush=True)


if __name__=='__main__': main()
