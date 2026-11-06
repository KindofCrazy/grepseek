"""Fetch only official CPU data/tokenizer artifacts; never downloads model weights."""
import argparse
from pathlib import Path

from mini.curation_utils import file_hash, json_file


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', default='mini/.cache')
    p.add_argument('--sft-revision', default='main')
    p.add_argument('--qa-revision', default='main')
    p.add_argument('--model-revision', default='main')
    p.add_argument('--include-benchmark',action='store_true',help='Also cache nq/test, triviaqa/test, 2wikimultihopqa/dev and bamboogle/test')
    a = p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = {}
    try:
        qa_files=[(f'{s}/{k}.jsonl', f'{s}_{k}.jsonl') for s in ('nq','hotpotqa') for k in ('train','dev')]
        if a.include_benchmark:
            qa_files += [(f'{s}/{k}.jsonl',f'{s}_{k}.jsonl') for s,k in [('nq','test'),('triviaqa','test'),('2wikimultihopqa','dev'),('bamboogle','test')]]
        for repo, revision, files in [
            ('alireza7/GrepSeek-ColdStart-SFT-10k', a.sft_revision, [('train.parquet','sft-train.parquet')]),
            ('RUC-NLPIR/FlashRAG_datasets', a.qa_revision, qa_files)]:
            commit = api.dataset_info(repo, revision=revision).sha
            manifest[repo] = {'commit':commit, 'files':{}}
            for remote, local in files:
                cached = Path(hf_hub_download(repo, remote, repo_type='dataset', revision=commit))
                destination = out / local
                import shutil
                if not destination.exists() or file_hash(destination) != file_hash(cached):
                    shutil.copyfile(cached, destination)
                manifest[repo]['files'][local] = file_hash(destination)
        commit = api.model_info('Qwen/Qwen3.5-2B', revision=a.model_revision).sha
        snapshot_download('Qwen/Qwen3.5-2B', revision=commit, local_dir=out / 'tokenizer', allow_patterns=['tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json','special_tokens_map.json'])
        manifest['Qwen/Qwen3.5-2B'] = {'commit':commit, 'weights_downloaded':False}
    except Exception as exc:
        raise SystemExit(f'Official data/tokenizer download failed: {type(exc).__name__}: {exc}\nCached complete files remain at {out.resolve()}. Supply --input-dir PATH to mini.curation and --model PATH to an actual tokenizer; no synthetic fallback.')
    json_file(out / 'download_manifest.json', manifest)


if __name__ == '__main__':
    main()
