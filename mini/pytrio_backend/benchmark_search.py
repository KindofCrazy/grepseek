"""Offline full-corpus SPCS/cache benchmark. No model API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import re
import time

from mini.pytrio_backend.acceleration import audit_shards, eligible
from mini.pytrio_backend.config import Config
from mini.pytrio_backend.data import preflight
from mini.pytrio_backend.search import Search
from mini.pytrio_backend.tracking import json_write


COMMANDS = [
    'rg -F "Giuseppe Verdi" corpus.jsonl | head -n 3',
    'rg -F "Drive-in theater" corpus.jsonl | head -n 3',
    'rg -F "__grepseek_accel_no_match_20261107__" corpus.jsonl | head -n 3',
    'rg -F "Giuseppe Verdi" corpus.jsonl | wc -l',
    'rg -F "Verdi" corpus.jsonl | grep -i -F "film" | head -n 3',
    'rg -n -F "Giuseppe Verdi" corpus.jsonl | head -n 3',
]


def measured(search,command):
    start = time.monotonic()
    payload = search.execute(command)
    return payload,time.monotonic()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--keep-shards',action='store_true')
    args = parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    config = Config()
    tokenizer,_ = preflight(config)
    corpus = Path(config.corpus).resolve()
    source = json.loads((corpus/'corpus_manifest.json').read_text())
    baseline = Search(corpus,tokenizer,cache_entries=0)
    expected = {}
    measurements = []
    generated = []
    result = {'corpus_sha256':source['output_sha256'],'measurements':measurements,
              'hardware_note':'Prepared Windows/WSL host, 4 CPUs and 2GB WSL limit; results are not paper hardware',
              'billable_model_requests':0,'shards_retained':args.keep_shards}
    try:
        for command in COMMANDS:
            payload,elapsed = measured(baseline,command)
            expected[command] = payload
            measurements.append({'backend':'sequential','command':command,'wall_s':elapsed,
                                 'payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()})
        cached = Search(corpus,tokenizer)
        command = COMMANDS[2]
        first,first_s = measured(cached,command)
        second,second_s = measured(cached,command)
        assert first == second == expected[command]
        result['cache'] = {'cold_s':first_s,'hit_s':second_s,'byte_exact':True,'stats':cached.stats()}
        for count in (2,4):
            volume = Path('/mnt/d') if Path('/mnt/d').exists() else args.work
            if shutil.disk_usage(volume).free < source['output_bytes']+5*1024**3:
                raise RuntimeError('Insufficient physical disk headroom for temporary shards')
            directory = args.work/f'shards-{count}'
            directory.mkdir()
            generated.append(directory)
            from inference.parallel_search.sharder import build_shards
            started = time.monotonic()
            build_shards(str(corpus/'wiki_corpus.jsonl'),str(directory),count)
            audit = audit_shards(directory,source)
            print(f'{count} shards prepared and byte-audited',flush=True)
            search = Search(corpus,tokenizer,backend='sharded',shards=str(directory),cache_entries=0)
            result[f'shards_{count}'] = {'build_and_audit_s':time.monotonic()-started,'audit':audit}
            for command in COMMANDS:
                payload,elapsed = measured(search,command)
                equal = payload==expected[command]
                measurements.append({'backend':f'shards-{count}','command':command,'wall_s':elapsed,
                                     'accelerated':eligible(command),'byte_exact':equal,
                                     'payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()})
                json_write(args.out,result)
                if not equal:
                    raise AssertionError(f'Full-corpus semantic mismatch: {count} shards, {command}')
            if not args.keep_shards:
                # Delete only these freshly generated, verified child files.
                root = args.work.resolve()
                for name in [*audit['files'],'manifest.json','verified.json']:
                    target = (directory/name).resolve()
                    if target.parent != directory.resolve() or not target.is_relative_to(root):
                        raise RuntimeError('Unsafe cleanup path')
                    target.unlink()
                directory.rmdir()
                generated.remove(directory)
        result['status']='passed'
        result['byte_exact_all_cases']=True
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__)
        raise
    finally:
        if not args.keep_shards:
            for directory in generated:
                directory = directory.resolve()
                if directory.parent != args.work.resolve() or directory.name not in ('shards-2','shards-4'):
                    raise RuntimeError('Unsafe generated-directory cleanup path')
                for target in directory.iterdir():
                    resolved = target.resolve()
                    if (resolved.parent != directory or not target.is_file() or
                            not re.fullmatch(r'(?:shard_[0-9]+|manifest\.json|verified\.json)',target.name)):
                        raise RuntimeError('Unexpected file in temporary shard directory; refusing cleanup')
                    target.unlink()
                directory.rmdir()
        json_write(args.out,result)
    print(json.dumps({'status':result['status'],'cache':result['cache'],
                      'mean_s':{kind:sum(r['wall_s'] for r in measurements if r['backend']==kind)/6
                                for kind in ('sequential','shards-2','shards-4')}},indent=2),flush=True)


if __name__ == '__main__':
    main()
