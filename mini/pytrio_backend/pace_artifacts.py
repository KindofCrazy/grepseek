"""CPU-only PACE build and verified relocation. Never imports PyTRIO or calls a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

CORPUS_SHA256 = '43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)


def available_memory():
    rows = Path('/proc/meminfo').read_text().splitlines()
    return int(next(line.split()[1] for line in rows if line.startswith('MemAvailable:')))*1024


def verify(index,corpus):
    index,corpus = Path(index).resolve(),Path(corpus).resolve()
    manifest = json.loads((index/'artifact_manifest.json').read_text())
    if manifest['corpus_sha256'] != CORPUS_SHA256 or digest(corpus)!=CORPUS_SHA256:
        raise ValueError('Corpus is not the verified full Wikipedia snapshot')
    for name,expected in manifest['data_sha256'].items():
        path = (index/name).resolve()
        if path.parent!=index or digest(path)!=expected:
            raise ValueError(f'PACE artifact differs from the build manifest: {name}')
    metadata_path = index/manifest['metadata_filename']
    if metadata_path.resolve().parent!=index:
        raise ValueError('Unsafe PACE metadata path')
    metadata = json.loads(metadata_path.read_text())
    origin = manifest['original_metadata']
    for key,value in origin.items():
        if key not in ('corpus_path','corpus_mtime') and metadata.get(key)!=value:
            raise ValueError(f'PACE metadata changed structurally: {key}')
    return manifest,metadata_path,metadata


def build(corpus,out):
    # Guard first: a mistaken invocation on the 2GB local WSL must not OOM.
    available = available_memory()
    if available < 28*1024**3:
        raise RuntimeError(f'PACE build requires at least 28 GiB available RAM; got {available/1024**3:.1f} GiB. Use a 64GB CPU host.')
    if digest(corpus)!=CORPUS_SHA256:
        raise ValueError('Full corpus hash mismatch')
    out.mkdir(parents=True,exist_ok=False)
    from inference.pace.core import META_FILENAME, build_pace
    metadata = build_pace(corpus,out)
    manifest = {'corpus_sha256':CORPUS_SHA256,'metadata_filename':META_FILENAME,
                'original_metadata':json.loads(metadata.to_json()),
                'data_sha256':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!=META_FILENAME},
                'build_available_memory_bytes':available,
                'numpy_version':importlib.metadata.version('numpy'),
                'built_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(out/'artifact_manifest.json',manifest)
    print(json.dumps({'status':'built','corpus_sha256':CORPUS_SHA256,
                      'files':len(manifest['data_sha256']),'output':str(out)}),flush=True)


def bind(index,corpus):
    manifest,path,metadata = verify(index,corpus)
    stat = corpus.stat()
    if stat.st_size!=metadata['corpus_size']:
        raise ValueError('PACE corpus size mismatch')
    # Data files stay byte-identical. Only relocate after full content hashes pass.
    metadata.update(corpus_path=str(corpus.resolve()),corpus_mtime=stat.st_mtime)
    write_json(path,metadata)
    write_json(index/'binding.json',{'corpus_sha256':CORPUS_SHA256,'corpus_path':str(corpus.resolve()),
                                   'at_utc':datetime.now(timezone.utc).isoformat(),
                                   'artifact_manifest_sha256':digest(index/'artifact_manifest.json')})
    print(json.dumps({'status':'bound','corpus':str(corpus.resolve())}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('build','bind','verify'))
    parser.add_argument('--corpus',type=Path,default=Path('data/wiki_18_corpus/wiki_corpus.jsonl'))
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='build':
        build(args.corpus,args.out)
    elif args.command=='bind':
        bind(args.out,args.corpus)
    else:
        verify(args.out,args.corpus)
        print('PACE data and full corpus hashes verified',flush=True)


if __name__=='__main__':
    main()
