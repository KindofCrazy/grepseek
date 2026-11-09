"""Exercise the original RL and inference tools against the verified FULL corpus."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import time

from mini.common import shell


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--corpus', default='data/wiki_18_corpus')
    parser.add_argument('--tokenizer', default='mini/.cache/tokenizer')
    parser.add_argument('--out', default='mini/reports/environment/search_checks.json')
    args = parser.parse_args()
    if os.name != 'posix' or not Path('/bin/bash').exists():
        raise SystemExit('Run inside Ubuntu/WSL, not Windows Python.')
    missing = sorted(command for command in shell.ALLOWED_COMMANDS if not shutil.which(command))
    if missing:
        raise SystemExit(f'Missing tools: {missing}')
    root = Path(args.corpus).resolve()
    corpus = root / 'wiki_corpus.jsonl'
    manifest = json.loads((root / 'corpus_manifest.json').read_text())
    if corpus.stat().st_size != manifest['output_bytes'] or manifest['output_lines'] < 20_000_000:
        raise SystemExit('A complete verified corpus is required.')
    with corpus.open(encoding='utf-8') as stream:
        first_line = stream.readline()
    first = json.loads(first_line)
    if not isinstance(first.get('contents'), str) or not isinstance(first.get('id'), str):
        raise SystemExit('Unexpected real corpus schema.')
    needle = re.search(r'"id"\s*:\s*"[^"]+"', first_line).group(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    commands = [f'rg -F -- {shlex.quote(needle)} corpus.jsonl | head -n 3',
                f'grep -F -- {shlex.quote(needle)} corpus.jsonl | head -n 3',
                "rg -F -- '__GREPSEEK_ENV_MISSING_9513d1257c__' corpus.jsonl | head -n 3"]
    checks = []
    for command in commands:
        shell._validate_pipeline(command)
        start = time.monotonic()
        payload = shell._execute_pipeline(command, cwd=str(root), timeout=180., max_tokens=2048, tokenizer=tok)
        if payload['timed_out'] or payload['exit_code'] != 0:
            raise RuntimeError(f'Real corpus search failed: {payload}')
        checks.append({'command': command, 'wall_s': time.monotonic()-start, 'payload': payload})
    if not checks[0]['payload']['stdout'] or checks[0]['payload']['stdout'] != checks[1]['payload']['stdout']:
        raise RuntimeError('rg/grep did not return the same known real passage.')
    if checks[2]['payload']['stdout']:
        raise RuntimeError('No-match scan unexpectedly returned a result.')
    # Both paths must expose the same observation shape and actual passage.
    from inference.tools import run_tool
    inference_result = run_tool(commands[0], corpus_dir=str(root), tokenizer=tok, max_tokens=2048, timeout=180.)
    if inference_result.exit_code != 0 or inference_result.stdout != checks[0]['payload']['stdout']:
        raise RuntimeError('Inference and RL tool paths disagree on the known passage.')
    truncated = shell._execute_pipeline('head -n 8 corpus.jsonl', cwd=str(root), timeout=5., max_tokens=32, tokenizer=tok)
    if not truncated['truncated'] or '[... output truncated at 32 tokens]' not in truncated['stdout']:
        raise RuntimeError('Real passage token truncation failed.')
    timeout = shell._execute_pipeline(commands[2], cwd=str(root), timeout=.001, max_tokens=2048, tokenizer=tok)
    if not timeout['timed_out'] or timeout['exit_code'] != -1:
        raise RuntimeError('Full-corpus forced timeout was not reported.')
    report = {'status': 'passed', 'scope': 'full_real_corpus_cpu_search',
              'platform': platform.platform(), 'corpus_manifest': manifest,
              'first_passage_id': first['id'], 'checks': checks,
              'inference_result': vars(inference_result), 'truncation_check': truncated,
              'forced_timeout_check': timeout, 'paid_model_calls': 0}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'output': str(output),
                      'search_wall_s': [round(c['wall_s'], 3) for c in checks]}))


if __name__ == '__main__':
    main()
