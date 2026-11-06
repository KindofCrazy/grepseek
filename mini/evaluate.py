"""Evaluate frozen Base/SFT/GRPO checkpoints through the official inference agent."""
import argparse
import json
from pathlib import Path

from mini.common import digest, read_jsonl


def metrics(records):
    from mini.benchmark_metrics import aggregate
    return aggregate(records)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoints', required=True, help='JSON list: [{label,model,base_url,tokenizer}, ...], exactly base/sft/grpo')
    p.add_argument('--data-dir', default='mini/data')
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    p.add_argument('--selection', help='Frozen JSON mapping base/sft/grpo to model paths, selected on DEV; required for test')
    p.add_argument('--corpus', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-turns', type=int, default=6)
    p.add_argument('--max-tokens-per-turn', type=int, default=0)
    p.add_argument('--temperature', type=float, default=0.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prompt-length',type=int,default=1024)
    p.add_argument('--response-length',type=int,default=8192)
    p.add_argument('--tool-max-tokens', type=int, default=2048)
    p.add_argument('--tool-timeout', type=float, default=180.)
    p.add_argument('--serve-locally', action='store_true', help='Load/evaluate/unload each checkpoint sequentially with vLLM')
    p.add_argument('--server-port', type=int, default=8000)
    p.add_argument('--server-tp', type=int, default=1)
    p.add_argument('--context-length', type=int, default=None,help='Defaults to prompt-length + response-length')
    p.add_argument('--server-memory', type=float, default=.8)
    p.add_argument('--server-timeout', type=int, default=600)
    a = p.parse_args()
    if a.context_length is None:
        a.context_length=a.prompt_length+a.response_length
    if min(a.prompt_length,a.response_length)<1 or a.context_length<a.prompt_length+a.response_length:
        p.error('context-length must cover positive prompt-length + response-length')
    checkpoints = json.loads(Path(a.checkpoints).read_text(encoding='utf-8'))
    if {c['label'] for c in checkpoints} != {'base', 'sft', 'grpo'} or len(checkpoints) != 3:
        raise ValueError('provide exactly base, sft, grpo')
    selection = json.loads(Path(a.selection).read_text()) if a.selection else None
    if a.split == 'test' and selection != {c['label']: c['model'] for c in checkpoints}:
        raise ValueError('test requires matching frozen DEV checkpoint selection')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh evaluation directory')
    out.mkdir(parents=True, exist_ok=True)
    examples = read_jsonl(Path(a.data_dir) / f'{a.split}.jsonl')
    from transformers import AutoTokenizer
    from openai import OpenAI
    from inference.agent import run_agent_on_example
    result = {}
    for c in checkpoints:
        from mini.eval_server import checkpoint_server
        with checkpoint_server(c, a, out) as endpoint:
            client = OpenAI(base_url=endpoint, api_key='EMPTY')
            available = {m.id for m in client.models.list().data}
            if c['model'] not in available:
                raise ValueError(f'endpoint {c["base_url"]} does not serve {c["model"]}')
            tokenizer = AutoTokenizer.from_pretrained(c.get('tokenizer', c['model']))
            from mini.benchmark_metrics import BudgetedCompletion,attach_token_lengths,report
            meter=BudgetedCompletion(client.chat.completions.create,tokenizer,a.prompt_length,a.response_length,a.context_length,a.seed)
            client.chat.completions.create=meter
            records = []
            with (out / f'{c["label"]}.jsonl').open('w', encoding='utf-8') as f:
                for ex in examples:
                    meter.reset(ex['id'])
                    rec = run_agent_on_example(ex, client=client, model=c['model'], tokenizer=tokenizer, corpus_dir=a.corpus,
                        max_assistant_turns=a.max_turns, max_tokens_per_turn=a.max_tokens_per_turn, temperature=a.temperature,
                        tool_max_tokens=a.tool_max_tokens, tool_timeout=a.tool_timeout).to_dict()
                    rec['source'] = ex.get('source')
                    rec['scope']='final_evaluation' if a.split=='test' else 'validation'
                    rec['response_budget_exhausted']=meter.budget_exhausted
                    if meter.budget_exhausted:
                        rec['finish_reason']='response_budget_exhausted'
                    attach_token_lengths(rec,tokenizer,meter.usage())
                    rec['observation_budget_overflow']=rec.get('context_growth_tokens') is not None and rec['context_growth_tokens']>a.response_length
                    records.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    f.flush()
            result[c['label']] = report(records)
    (out / 'summary.json').write_text(json.dumps({'arguments': vars(a), 'checkpoints': checkpoints, 'data_hash': digest(examples), 'metrics': result}, indent=2))
    from mini.benchmark_metrics import markdown
    (out/'summary.md').write_text(markdown(result),encoding='utf-8')


if __name__ == '__main__':
    main()
