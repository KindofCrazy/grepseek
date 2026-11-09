"""Conservative adapter for the upstream SPCS engine; unsupported commands stay sequential."""
import hashlib
import json
from pathlib import Path

from inference.parallel_search.pipeline import Strategy, classify, parse_pipeline


def eligible(command):
    try:
        stages = parse_pipeline(command)
        plan = classify(stages)
        if plan.strategy not in (Strategy.HEAD,Strategy.COUNT):
            return False
        if plan.head_n is not None and not 1 <= plan.head_n <= 100:
            return False
        for index,stage in enumerate(stages[:-1]):
            if stage.prog not in ('rg','grep'):
                return False
            options = [t for t in stage.tokens[1:] if t.startswith('-')]
            positional = [t for t in stage.tokens[1:] if not t.startswith('-')]
            if index==0:
                if len(positional)!=2 or positional[-1] not in ('corpus.jsonl','wiki_corpus.jsonl'):
                    return False
            elif len(positional)!=1:
                return False
            if not any(t=='--fixed-strings' or (t.startswith('-') and not t.startswith('--') and 'F' in t) for t in options):
                return False
            for option in options:
                if option not in ('--fixed-strings','--ignore-case','--word-regexp') and (
                        option.startswith('--') or not set(option[1:]) <= set('Fiw')):
                    return False
        return True
    except (ValueError,IndexError):
        return False


def audit_shards(directory,source_manifest):
    """Verify byte concatenation against the prepared full corpus, in one pass."""
    directory = Path(directory).resolve()
    path = directory/'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['src_sha256'] != source_manifest['output_sha256']:
        raise ValueError('Shard manifest has a different corpus')
    combined = hashlib.sha256()
    files = {}
    for name in manifest['shards']:
        shard = (directory/name).resolve()
        if shard.parent != directory or not shard.is_file():
            raise ValueError('Unsafe or missing shard file')
        digest = hashlib.sha256()
        with shard.open('rb') as stream:
            for data in iter(lambda:stream.read(8*1024**2),b''):
                combined.update(data)
                digest.update(data)
        stat = shard.stat()
        files[name] = {'sha256':digest.hexdigest(),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    if combined.hexdigest() != source_manifest['output_sha256']:
        raise ValueError('Shards do not reconstruct the byte-exact original corpus')
    result = {'combined_sha256':combined.hexdigest(),'files':files,'byte_exact':True}
    (directory/'verified.json').write_text(json.dumps(result,indent=2))
    return result


def shard_signature(directory,corpus_hash):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    verified = json.loads((directory/'verified.json').read_text())
    if (manifest['src_sha256'] != corpus_hash or verified['combined_sha256'] != corpus_hash or
            manifest['shards'] != list(verified['files']) or
            manifest['n_shards'] != len(manifest['shards'])):
        raise ValueError('Shards were not audited against the current corpus')
    signatures = []
    for name,info in verified['files'].items():
        path = (directory/name).resolve()
        if path.parent != directory:
            raise ValueError('Unsafe shard path')
        stat = path.stat()
        if stat.st_size != info['bytes'] or stat.st_mtime_ns != info['mtime_ns']:
            raise ValueError('Shard changed after its byte audit')
        signatures.append((stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns))
    return tuple(signatures)


def main():
    import argparse
    from inference.parallel_search.sharder import build_shards
    parser = argparse.ArgumentParser(description='Build and byte-audit full-corpus SPCS shards; CPU only')
    parser.add_argument('--corpus',type=Path,default=Path('data/wiki_18_corpus'))
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--shards',type=int,choices=(2,4),default=2)
    args = parser.parse_args()
    manifest = json.loads((args.corpus/'corpus_manifest.json').read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    build_shards(str(args.corpus/'wiki_corpus.jsonl'),str(args.out),args.shards)
    audit = audit_shards(args.out,manifest)
    print(json.dumps({'shards':args.shards,'byte_exact':audit['byte_exact'],
                      'combined_sha256':audit['combined_sha256']}))


if __name__ == '__main__':
    main()
