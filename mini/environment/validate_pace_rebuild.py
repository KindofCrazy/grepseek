"""Validate a rebuilt PACE index without requiring randomized term IDs to match."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(index, original_manifest, old_semantic, new_semantic):
    original=json.loads(Path(original_manifest).read_text())
    rebuilt=json.loads((Path(index)/'artifact_manifest.json').read_text())
    for key in ('corpus_size','num_lines','num_terms','min_term_len','token_fold',
                'format_version','n_singletons','token_class'):
        if original['original_metadata'][key]!=rebuilt['original_metadata'][key]:
            raise ValueError('Index structure differs: '+key)
    if original['corpus_sha256']!=rebuilt['corpus_sha256']:
        raise ValueError('Index corpus hash differs')
    before=json.loads(Path(old_semantic).read_text())
    after=json.loads(Path(new_semantic).read_text())
    if before!=after:
        raise ValueError('Full term-to-posting fingerprints differ')
    for key in ('offsets_base.bin','offsets_delta.bin'):
        if original['data_sha256'][key]!=rebuilt['data_sha256'][key]:
            raise ValueError('Corpus line offsets differ')
    if set(original['data_sha256'])!=set(rebuilt['data_sha256']):
        raise ValueError('Index data file set differs')
    same=[key for key,value in original['data_sha256'].items() if rebuilt['data_sha256'][key]==value]
    return {'status':'passed','binary_identical':len(same)==len(original['data_sha256']),
            'same_binary_files':same,'full_term_posting_fingerprint':before,
            'reason':'Upstream per-line token sets have randomized iteration; term ID assignment is not a canonical serialization.',
            'reference_manifest_sha256':hashlib.sha256(Path(original_manifest).read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('index','original_manifest','old_semantic','new_semantic','out'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=parser.parse_args()
    report=validate(args.index,args.original_manifest,args.old_semantic,args.new_semantic)
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
