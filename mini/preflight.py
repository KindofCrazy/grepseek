"""CPU checks using the actual tokenizer and model metadata (no model weights)."""
import argparse
import json
from pathlib import Path

from mini.common import read_jsonl, validate_sft
from mini.template import turn_tokens


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--model', default='Qwen/Qwen3.5-2B')
    p.add_argument('--data-dir', default='mini/data')
    p.add_argument('--max-prompt-length', type=int, default=1024)
    p.add_argument('--sft-length', type=int, default=9216)
    a = p.parse_args()
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    config = AutoConfig.from_pretrained(a.model)
    if config.model_type != 'qwen3_5' or 'Qwen3_5ForConditionalGeneration' not in config.architectures:
        raise ValueError('expected Qwen3.5 multimodal container with text decoder')
    # Read the upstream RL prompt asset without importing Linux-only tool executors.
    from mini.common import ROOT
    system_prompt = (ROOT / 'rl/prompts/system_prompt.txt').read_text(encoding='utf-8').strip()
    data = Path(a.data_dir)
    count = 0
    for row in read_jsonl(data / 'sft.jsonl'):
        messages = validate_sft(row)
        ids, mask = [], []
        for i in range(len(messages)):
            turn, loss = turn_tokens(tokenizer, messages, i)
            if messages[i]['role'] != 'assistant' and any(loss):
                raise ValueError('observation/input loss mask must be zero')
            ids += turn
            mask += loss
        full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, tools=[], enable_thinking=True)
        if ids != full or len(ids) > a.sft_length or not any(mask):
            raise ValueError('invalid assembled SFT tokens/mask/length')
        count += 1
    max_prompt = 0
    for split in ('train', 'dev', 'test'):
        for row in read_jsonl(data / f'{split}.jsonl'):
            messages = [{'role':'system', 'content':system_prompt}, {'role':'user', 'content':'Query: ' + row['question']}]
            ids = tokenizer.apply_chat_template(messages, tools=[], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
            max_prompt = max(max_prompt, len(ids))
            if len(ids) > a.max_prompt_length:
                raise ValueError(f'{split}/{row["id"]}: prompt too long: {len(ids)}')
    print(json.dumps({'sft_validated': count, 'max_prompt_tokens': max_prompt, 'model_type': config.model_type, 'gpu_validated': False}))


if __name__ == '__main__':
    main()
