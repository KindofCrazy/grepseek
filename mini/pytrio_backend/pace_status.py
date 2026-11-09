"""Read-only, secret-free status for the standalone CPU build job."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re


def main():
    root=Path(__file__).resolve().parents[2]
    output={'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    started=root/'build-started.txt'
    if started.exists():
        origin=datetime.fromisoformat(started.read_text().strip().replace('Z','+00:00'))
        output['elapsed_s']=(datetime.now(timezone.utc)-origin).total_seconds()
    log=root/'build.log'
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-8192))
        tail=stream.read().decode('utf-8',errors='replace').replace('\r','\n').splitlines()
    exit_file=root/'build-exit-code.txt'
    corpus=root/'data/wiki_18_corpus/wiki_corpus.jsonl'
    archive=root/'data/wiki_18_corpus/wiki-18.jsonl.gz'
    packed=root/'pace-full.tar.gz'
    index=root/'data/pace-full'
    if exit_file.exists():
        output['exit_code']=int(exit_file.read_text())
        output['stage']='completed' if output['exit_code']==0 else 'failed'
    elif (index/'artifact_manifest.json').exists(): output['stage']='packing'
    elif corpus.exists(): output['stage']='building'
    elif corpus.with_suffix('.jsonl.part').exists(): output['stage']='extracting'
    else: output['stage']='downloading'
    progress=[line for line in tail if 'building PACE structure:' in line]
    output['last_progress']=progress[-1] if progress else (tail[-1] if tail else '')
    for name,path in [('archive_bytes',archive),('corpus_bytes',corpus),('packed_bytes',packed)]:
        if path.exists(): output[name]=path.stat().st_size
    if index.exists(): output['index_bytes']=sum(p.stat().st_size for p in index.iterdir() if p.is_file())
    processes=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit(): continue
        try:
            args=(directory/'cmdline').read_bytes().split(b'\0')
            if b'mini.pytrio_backend.pace_artifacts' not in args: continue
            status=(directory/'status').read_text()
            processes.append({'pid':int(directory.name),**{
                field:int(re.search(r'^'+field+r':\s+(\d+)',status,re.M).group(1))
                for field in ('VmRSS','VmHWM') if re.search(r'^'+field+':',status,re.M)}})
        except (FileNotFoundError,ProcessLookupError,PermissionError): pass
    output['builder_processes']=processes
    print(json.dumps(output,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
