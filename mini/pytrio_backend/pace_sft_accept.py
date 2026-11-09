"""Compare frozen SFT tool observations with real sequential and PACE execution."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from mini.pytrio_backend.pace_cloud_accept import Log,raw
from mini.pytrio_backend.pace_search import PaceSearch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--tool-module',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('sft-acceptance.json'))
    parser.add_argument('--index',type=Path,default=Path('data/pace-full'))
    parser.add_argument('--corpus',type=Path,default=Path('data/wiki_18_corpus/wiki_corpus.jsonl'))
    args=parser.parse_args()
    from tokenizers import Tokenizer
    if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()!='5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42':
        raise ValueError('Not the verified Qwen3.5 tokenizer')
    backend=Tokenizer.from_file(str(args.tokenizer))
    class Codec:
        def __call__(self,text,add_special_tokens=False):
            return {'input_ids':backend.encode(text,add_special_tokens=add_special_tokens).ids}
        def decode(self,ids,skip_special_tokens=False):
            return backend.decode(ids,skip_special_tokens=skip_special_tokens)
    codec=Codec()
    spec=importlib.util.spec_from_file_location('sft_tool_contract',args.tool_module)
    tool=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=tool
    spec.loader.exec_module(tool)
    def payload(command,result):
        stdout,stderr,code=result
        text,truncated=tool._truncate_tokens(stdout.decode('utf-8',errors='replace'),2048,codec)
        if truncated: text+='\n[... output truncated at 2048 tokens]'
        return {'command':command,'stdout':text,'stderr':stderr.decode('utf-8',errors='replace'),
                'exit_code':code,'timed_out':False,'truncated':truncated,
                'information_lines':[line for line in text.split('\n') if line.strip()]}
    source=json.loads(args.input.read_text())
    log=Log()
    pace=PaceSearch(args.index,args.corpus.resolve(),log)
    report={'status':'running','model_api_requests':0,'selection':source['selection'],
            'trajectory_ids':source['trajectory_ids'],'sft_file_sha256':source['sft_file_sha256'],
            'tool_contract_sha256':hashlib.sha256(args.tool_module.read_bytes()).hexdigest(),
            'tokenizer_sha256':hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),'cases':[]}
    for case in source['cases']:
        old=case['recorded']
        command=old['command']
        sequential=payload(command,raw(command,args.corpus.resolve()))
        accelerated=pace.execute(command,1048576)
        served=accelerated is not None
        observed=payload(command,(accelerated[0],b'',accelerated[1])) if served else sequential
        equal={key:old.get(key)==value for key,value in sequential.items()}
        mismatched=[key for key,match in equal.items() if not match]
        row={'trajectory_id':case['trajectory_id'],'source':case['source'],'turn':case['turn'],
             'command':command,'pace_served':served,'pace_vs_sequential':observed==sequential,
             'recorded_field_equal':equal,'recorded_vs_sequential':not mismatched,'mismatched_fields':mismatched}
        if mismatched:
            row['differences']={key:{'recorded':old.get(key),'replayed':sequential[key]} for key in mismatched}
        report['cases'].append(row)
        args.out.write_text(json.dumps(report,indent=2))
        print(json.dumps({k:row[k] for k in ('trajectory_id','turn','pace_vs_sequential','mismatched_fields')}),flush=True)
    report.update(pace_all_equal=all(row['pace_vs_sequential'] for row in report['cases']),
                  recorded_matches=sum(row['recorded_vs_sequential'] for row in report['cases']),
                  pace_served=sum(row['pace_served'] for row in report['cases']))
    report['status']='passed' if report['pace_all_equal'] and report['recorded_matches']==len(report['cases']) else 'differences_found'
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('status','pace_all_equal','recorded_matches','pace_served')}),flush=True)
    if not report['pace_all_equal']: raise AssertionError('PACE differs from sequential SFT tool replay')


if __name__=='__main__': main()
