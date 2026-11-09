"""Cloud-first byte acceptance: real training commands, no tokenizer/model API."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

from mini.pytrio_backend.pace_artifacts import bind
from mini.pytrio_backend.pace_search import PaceSearch


class Log:
    def __init__(self): self.events=[]
    def event(self,kind,**values): self.events.append({'kind':kind,**values})


def raw(command,corpus):
    argv=['prlimit','--cpu=120','--nproc=128','--','bwrap','--die-with-parent','--unshare-all',
          '--ro-bind','/usr','/usr','--symlink','usr/bin','/bin','--symlink','usr/lib','/lib',
          '--symlink','usr/lib64','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp',
          '--ro-bind',str(corpus.parent),'/corpus','--chdir','/corpus','--clearenv',
          '--setenv','PATH','/usr/bin:/bin','--setenv','LANG','C.UTF-8',
          '/bin/bash','-c',command.replace('corpus.jsonl','wiki_corpus.jsonl')]
    result=subprocess.run(argv,capture_output=True,timeout=180)
    if result.stderr.startswith((b'bwrap:',b'prlimit:')):
        raise RuntimeError('Cloud acceptance sandbox could not start')
    return result.stdout,result.stderr,result.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commands',type=Path,required=True)
    parser.add_argument('--index',type=Path,default=Path('data/pace-full'))
    parser.add_argument('--corpus',type=Path,default=Path('data/wiki_18_corpus/wiki_corpus.jsonl'))
    parser.add_argument('--out',type=Path,default=Path('cloud-acceptance.json'))
    args=parser.parse_args()
    corpus=args.corpus.resolve()
    bind(args.index,corpus)
    log=Log()
    start=time.monotonic()
    pace=PaceSearch(args.index,corpus,log)
    report={'status':'running','verification':'raw stdout/stderr/exit-code; no model/tokenizer substitution',
            'model_api_requests':0,'cache_enabled':False,'load_and_verify_s':time.monotonic()-start,
            'cases':[],'source_commands_sha256':hashlib.sha256(args.commands.read_bytes()).hexdigest()}
    for command in json.loads(args.commands.read_text())['commands']:
        start=time.monotonic()
        expected=raw(command,corpus)
        sequential=time.monotonic()-start
        start=time.monotonic()
        accelerated=pace.execute(command,1048576)
        served=accelerated is not None
        observed=(accelerated[0],b'',accelerated[1]) if served else raw(command,corpus)
        seconds=time.monotonic()-start
        trace=next((row for row in reversed(log.events) if row['kind']=='pace_execution'),None)
        hashes=lambda payload:[hashlib.sha256(v).hexdigest() if isinstance(v,bytes) else v for v in payload]
        report['cases'].append({'command':command,'byte_exact':expected==observed,'pace_served':served,
                                'sequential_s':sequential,'pace_s':seconds,'pace_decision':trace,
                                'reference_hashes':hashes(expected),'observed_hashes':hashes(observed)})
        report['peak_self_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.out.write_text(json.dumps(report,indent=2))
        print(json.dumps(report['cases'][-1]),flush=True)
        if observed!=expected:
            report['status']='failed'
            args.out.write_text(json.dumps(report,indent=2))
            raise AssertionError('Cloud PACE differs from sequential execution')
    before=sum(row['sequential_s'] for row in report['cases'])
    after=sum(row['pace_s'] for row in report['cases'])
    report.update(status='passed',all_byte_exact=True,sequential_total_s=before,pace_total_s=after,
                  speedup=before/after,pace_served=sum(row['pace_served'] for row in report['cases']),
                  index_provenance=log.events[0])
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('status','pace_served','speedup','peak_self_rss_kib')}),flush=True)


if __name__=='__main__': main()
