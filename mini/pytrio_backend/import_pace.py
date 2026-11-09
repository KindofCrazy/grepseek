"""Verify a returned PACE archive, safely extract it, then bind the local corpus."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import uuid

from mini.pytrio_backend.pace_artifacts import bind,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--out',type=Path,default=Path('data/pace-full'))
    parser.add_argument('--corpus',type=Path,default=Path('data/wiki_18_corpus/wiki_corpus.jsonl'))
    args=parser.parse_args()
    if args.out.exists(): raise ValueError('Import destination must be fresh')
    if len(args.sha256)!=64 or digest(args.archive)!=args.sha256.lower():
        raise ValueError('Returned archive SHA256 mismatch')
    staging=args.out.parent/(args.out.name+'-importing-'+uuid.uuid4().hex[:8])
    staging.mkdir(parents=True,exist_ok=False)
    total=0
    with tarfile.open(args.archive,'r:gz') as archive:
        for member in archive:
            parts=PurePosixPath(member.name).parts
            if not parts or parts[0]!='pace-full' or '..' in parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe or unexpected returned archive entry')
            destination=staging.joinpath(*parts[1:])
            if not destination.resolve().is_relative_to(staging.resolve()):
                raise ValueError('Archive entry escapes import directory')
            if member.isdir():
                destination.mkdir(parents=True,exist_ok=True)
            else:
                total+=member.size
                if total>6*1024**3: raise ValueError('Unexpectedly large PACE archive')
                destination.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,destination.open('xb') as dst:
                    shutil.copyfileobj(src,dst,length=8*1024**2)
    bind(staging,args.corpus)
    # Binding describes the corpus, independent of the directory holding arrays.
    staging.rename(args.out)
    print(f'PACE imported and corpus-bound: {args.out}',flush=True)


if __name__=='__main__': main()
