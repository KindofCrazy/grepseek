from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rl'))


def upstream(name, relative):
    # Load pure upstream modules without importing trainer/__init__ (needs CUDA stack).
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reward = upstream('mini_upstream_reward', 'rl/grepseek/trainer/verl_integration/reward_function.py')
shell = upstream('mini_upstream_shell', 'rl/grepseek/trainer/verl_integration/search_corpus_tool.py')


def question_key(question):
    return ' '.join(re.findall(r'\w+', question.casefold()))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def validate_sft(row):
    messages = row['messages']
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError('messages must contain system, user, assistant')
    if [m['role'] for m in messages[:2]] != ['system', 'user']:
        raise ValueError('expected initial system/user')
    pending = False
    for index, msg in enumerate(messages):
        if not isinstance(msg.get('content'), str):
            raise ValueError('text-only messages required')
        role, text = msg['role'], msg['content']
        if index < 2:
            continue
        if role == 'assistant' and not pending:
            calls = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S)
            if calls:
                if len(calls) != 1 or '<answer>' in text:
                    raise ValueError('one action per assistant turn')
                call = json.loads(calls[0])
                args = call.get('arguments')
                if call.get('name') != 'shell' or not isinstance(args, dict):
                    raise ValueError('expected shell arguments object')
                if not isinstance(args.get('command'), str) or not args['command'].strip():
                    raise ValueError('expected non-empty command')
                shell._validate_pipeline(args['command'])
                pending = True
            elif index != len(messages) - 1 or not re.search(r'<answer>.+?</answer>', text, re.S):
                raise ValueError('expected final answer')
        elif role == 'tool' and pending:
            payload = json.loads(text)
            for key in ('command', 'stdout', 'stderr', 'exit_code', 'timed_out', 'truncated', 'information_lines'):
                if key not in payload:
                    raise ValueError(f'missing tool payload {key}')
            pending = False
        else:
            raise ValueError('invalid assistant/tool alternation')
    if pending or messages[-1]['role'] != 'assistant' or '<answer>' not in messages[-1]['content']:
        raise ValueError('unfinished trajectory')
    transcript = '\n'.join(m['content'] if m['role'] == 'assistant' else '<tool_response>' + m['content'] + '</tool_response>' for m in messages[2:])
    ok, reason = reward._is_valid_sequence(transcript)
    if not ok:
        raise ValueError(f'upstream format gate: {reason}')
    return messages


def check_batch(prompt_batch=4, mini_batch=4, group=4, gpus=1, sp=1, micro=1, tp=1):
    if min(prompt_batch, mini_batch, group, gpus, sp, micro, tp) < 1:
        raise ValueError('all batch/parallel sizes must be positive')
    if group < 2:
        raise ValueError('GRPO requires at least two rollouts per prompt')
    if gpus % sp or gpus % tp:
        raise ValueError('SP and TP must divide GPU count')
    dp = gpus // sp
    if prompt_batch % mini_batch or (prompt_batch * group) % gpus:
        raise ValueError('prompt batch must divide into PPO batches; trajectories must divide GPUs')
    if (mini_batch * group) % dp or (mini_batch * group // dp) % micro:
        raise ValueError('normalized PPO batch must divide into per-GPU micro batches')
    return {'trajectories': prompt_batch * group, 'ppo_trajectories_per_dp': mini_batch * group // dp}
