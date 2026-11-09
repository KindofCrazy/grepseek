"""Export only the CPU build sources; exclude training runs, keys, WSL disks and model files."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('mini/exports/pace-build-job.tar.gz'))
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a fresh job archive path')
    root = Path(__file__).resolve().parents[2]
    files = [root/name for name in (
        'mini/__init__.py','mini/pytrio_backend/__init__.py','mini/pytrio_backend/pace_artifacts.py',
        'mini/pytrio_backend/pace_cloud_build.sh','mini/pytrio_backend/PACE.md',
        'mini/environment/prepare_corpus.py','mini/environment/corpus-source.json','mini/UPSTREAM.json',
        'inference/pace/__init__.py','inference/pace/core.py','inference/pace/build.py')]
    files.extend(p for p in (root/'inference/pace/_native').rglob('*')
                 if p.is_file() and (p.suffix in ('.c','.h','.sh') or p.name=='LICENSE'))
    manifest = {str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.out,'w:gz') as archive:
        for path in files:
            archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
        data = json.dumps(manifest,indent=2).encode()
        info = tarfile.TarInfo('job_manifest.json')
        info.size = len(data)
        archive.addfile(info,io.BytesIO(data))
    checksum = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(args.out.suffix+'.sha256').write_text(f'{checksum}  {args.out.name}\n')
    print(json.dumps({'archive':str(args.out),'bytes':args.out.stat().st_size,
                      'sha256':checksum,'source_files':len(files)}))


if __name__=='__main__':
    main()
