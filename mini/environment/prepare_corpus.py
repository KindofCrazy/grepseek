"""Download the real Wikipedia archive at a pinned revision, verify, extract atomically.

No model calls or synthetic corpus. Only the JSONL member is streamed out of
the tar archive; member paths are never used as filesystem destinations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from datetime import datetime, timezone


REPO = 'PeterJinGo/wiki-18-corpus'
FILENAME = 'wiki-18.jsonl.gz'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dest', default='data/wiki_18_corpus')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--metadata', help='Previously verified official source metadata JSON')
    parser.add_argument('--archive', help='Already downloaded archive; SHA256 must match official metadata')
    args = parser.parse_args()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / 'wiki_corpus.jsonl'
    manifest_path = dest / 'corpus_manifest.json'
    if output.exists():
        if not manifest_path.exists():
            raise SystemExit('Existing corpus has no verified manifest; use a fresh destination.')
        manifest = json.loads(manifest_path.read_text())
        if output.stat().st_size != manifest['output_bytes'] or sha256(output) != manifest['output_sha256']:
            raise SystemExit('Existing corpus differs from its manifest; refusing to use it.')
        print('Existing full corpus hash verified.', flush=True)
        return
    if shutil.disk_usage(dest).free < 23 * 1024**3:
        raise SystemExit('At least 23 GiB free space is required for archive + extraction.')
    if args.metadata:
        source = json.loads(Path(args.metadata).read_text())
        if source['repo'] != REPO or source['filename'] != FILENAME:
            raise SystemExit('Source metadata identifies a different corpus.')
        revision, source_size, expected_hash = source['revision'], source['bytes'], source['sha256']
    else:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(REPO, revision=args.revision, files_metadata=True)
        source = next(item for item in info.siblings if item.rfilename == FILENAME)
        revision, source_size = info.sha, source.size
        expected_hash = source.lfs.sha256 if source.lfs else None
    if not expected_hash or not source_size:
        raise SystemExit('Source did not supply LFS SHA256 and byte size; refusing unverified download.')
    print(json.dumps({'repo': REPO, 'revision': revision, 'bytes': source_size,
                      'archive_sha256': expected_hash}), flush=True)
    if args.archive:
        archive = Path(args.archive)
    else:
        from huggingface_hub import hf_hub_download
        archive = Path(hf_hub_download(REPO, FILENAME, repo_type='dataset', revision=revision, local_dir=dest))
    actual_hash = sha256(archive)
    if archive.stat().st_size != source_size or actual_hash != expected_hash:
        raise SystemExit('Downloaded archive size/SHA256 differs from source metadata.')
    print('Archive SHA256 verified. Extracting JSONL...', flush=True)
    partial = dest / 'wiki_corpus.jsonl.part'
    digest = hashlib.sha256()
    size = lines = 0
    last_byte = b''
    with tarfile.open(archive, 'r|gz') as tar:
        member = next((m for m in tar if m.isfile() and m.name.endswith('.jsonl')), None)
        if member is None:
            raise SystemExit('Official archive contains no JSONL member.')
        with tar.extractfile(member) as stream, partial.open('wb') as target:
            while chunk := stream.read(16 * 1024**2):
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                lines += chunk.count(b'\n')
                last_byte = chunk[-1:]
        if size != member.size:
            raise SystemExit('Extracted size differs from tar member size.')
    if last_byte and last_byte != b'\n':
        lines += 1
    if size < 10 * 1024**3 or lines < 20_000_000:
        raise SystemExit('Extracted corpus is unexpectedly small; refusing to mark complete.')
    manifest = {'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                'source_repo': REPO, 'source_file': FILENAME, 'source_revision': revision,
                'archive_bytes': source_size, 'archive_sha256': actual_hash,
                'archive_member': member.name, 'output_bytes': size,
                'output_sha256': digest.hexdigest(), 'output_lines': lines,
                'synthetic': False}
    # Manifest first: an interrupted rename cannot leave a final corpus without metadata.
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    os.replace(partial, output)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
