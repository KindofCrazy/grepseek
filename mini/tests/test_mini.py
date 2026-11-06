import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mini.common import ROOT, check_batch, question_key, reward, shell, validate_sft
from mini.launch import parser, build as command
from mini.prepare_data import build
from mini.evaluate import metrics
from mini.observability import group_stats


def trajectory(q='Who won?'):
    return {'id': q, 'question': q, 'messages': [
        {'role': 'system', 'content': 'Use shell to search corpus.jsonl.'},
        {'role': 'user', 'content': 'Query: ' + q},
        {'role': 'assistant', 'content': '<think>Find evidence.</think><tool_call>{"name":"shell","arguments":{"command":"grep -F Paris corpus.jsonl | head -n 3"}}</tool_call>'},
        {'role': 'tool', 'content': json.dumps(dict(command='grep', stdout='Paris', stderr='', exit_code=0, timed_out=False, truncated=False, information_lines=['Paris']))},
        {'role': 'assistant', 'content': '<think>Evidence found.</think><answer>Paris</answer>'}]}


def test_sft_format_and_schema():
    validate_sft(trajectory())
    import jsonschema
    tool = OmegaConf.load(ROOT / 'rl/grepseek/trainer/config/tools/search_tools.yaml').tools[0].tool_schema
    params = OmegaConf.to_container(tool.function.parameters)
    jsonschema.validate({'command': 'grep Paris corpus.jsonl'}, params)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'command': 3}, params)


@pytest.mark.parametrize('mutate', [lambda r: r['messages'].pop(), lambda r: r['messages'][2].update(content='<answer>x</answer>'), lambda r: r['messages'][3].update(content='{}')])
def test_invalid_sft(mutate):
    row = trajectory()
    mutate(row)
    with pytest.raises(ValueError):
        validate_sft(row)


@pytest.mark.parametrize('cmd', ['grep -F Paris corpus.jsonl | head -n 3', 'rg -i Paris corpus.jsonl | wc -l', 'cat corpus.jsonl | tail -n 1'])
def test_legal_shell(cmd):
    shell._validate_pipeline(cmd)


@pytest.mark.parametrize('cmd', ['rm corpus.jsonl', 'cat corpus.jsonl > output', 'grep x corpus.jsonl; ls', 'find . -delete', 'sed -i s/a/b/ corpus.jsonl', 'python -c pass', 'cat corpus.jsonl && ls', 'cat corpus.jsonl\nls', 'cat corpus.jsonl |'])
def test_illegal_shell(cmd):
    with pytest.raises(shell.ToolError):
        shell._validate_pipeline(cmd)


@pytest.mark.skipif(os.name == 'nt', reason='official executor requires /bin/bash; run in Linux container')
def test_real_shell_execution(tmp_path):
    (tmp_path / 'wiki_corpus.jsonl').write_text('Paris\nLondon\n')
    result = shell._execute_pipeline('grep Paris corpus.jsonl', cwd=str(tmp_path), timeout=5, max_tokens=2048, tokenizer=None)
    assert result['exit_code'] == 0 and result['stdout'].strip() == 'Paris'


@pytest.mark.parametrize('text,expected', [('<think>Known.</think><answer>Paris</answer>', 1.), ('Known.</think><answer>Paris</answer>', 1.), ('<think>Known.</think><answer>London</answer>', 0.), ('<answer>Paris</answer>', 0.), ('<think>x</think><answer>Paris</answer>trailing', 0.), ('', 0.)])
def test_reward(text, expected):
    r = reward.grepseek_reward(data_source='test', solution_str=text, ground_truth={'golden_answers': ['Paris']}, extra_info={}, enable_length_decay=False, reward_metric='f1')
    assert r['score'] == expected


def test_reward_partial_and_invalid():
    kwargs = dict(data_source='test', solution_str='<think>x</think><answer>New York</answer>', ground_truth={'golden_answers':['York']}, enable_length_decay=False, reward_metric='f1')
    assert reward.grepseek_reward(**kwargs, extra_info={})['score'] == pytest.approx(2/3)
    assert reward.grepseek_reward(**kwargs, extra_info={'invalid_tool_request': True})['score'] == 0


@pytest.mark.parametrize('gpus,sp', [(1, 1), (2, 1), (2, 2)])
def test_batch(gpus, sp):
    assert check_batch(gpus=gpus, sp=sp)['trajectories'] == 16


@pytest.mark.parametrize('kwargs', [{'gpus':1,'sp':2}, {'group':1}, {'prompt_batch':5}, {'micro':3}, {'gpus':1,'tp':2}])
def test_bad_batch(kwargs):
    with pytest.raises(ValueError):
        check_batch(**kwargs)


@pytest.mark.parametrize('stage', ['sft', 'rl'])
def test_hydra_composition(stage, monkeypatch):
    a = parser().parse_args([stage])
    cmd, env, out = command(a, [])
    for k, v in env.items():
        if k.startswith(('GREPSEEK_', 'VERL_TOOL_')):
            monkeypatch.setenv(k, v)
    config_dir = ROOT / ('verl/verl/trainer/config' if stage == 'sft' else 'rl/grepseek/trainer/config')
    start = next(i for i, x in enumerate(cmd) if x.startswith('data.train'))
    overrides = cmd[start:]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name='sft_trainer_engine' if stage == 'sft' else 'grpo_trainer', overrides=overrides)
        OmegaConf.resolve(cfg)
    assert cfg.trainer.total_training_steps == 2
    if stage == 'rl':
        assert cfg.actor_rollout_ref.rollout.n == 4
        assert cfg.actor_rollout_ref.rollout.max_model_len == 9216
        assert cfg.custom_reward_function.reward_kwargs.L_max == 8192
        assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
        assert cfg.data.apply_chat_template_kwargs.enable_thinking


def test_stable_isolated_samples():
    qa = {s: {split: [{'id':f'{s}-{split}-{i}', 'question':f'{s} {split} question {i}', 'golden_answers':['x'], 'source':s} for i in range(8)] for split in ('train','dev')} for s in ('nq','hotpotqa')}
    sft = [trajectory(f'SFT question {i}') for i in range(8)]
    sft.append(trajectory(qa['nq']['dev'][0]['question']))
    a = build(sft, qa, sft_count=4, train_count=4, dev_count=4, test_count=4)
    b = build(list(reversed(sft)), qa, sft_count=4, train_count=4, dev_count=4, test_count=4)
    assert a[:2] == b[:2]
    sets = [{question_key(r['question']) for r in rows} for rows in [a[0], *a[1].values()]]
    assert all(not x & y for i,x in enumerate(sets) for y in sets[i+1:])


def test_eval_metrics():
    rec = dict(prediction='Paris', gold_answers=['Paris'], n_tool_calls=1, turns=[{'role':'assistant','content':'<tool_call>bad</tool_call>', 'finish_reason':'length'}])
    m = metrics([rec])
    assert m['exact_match'] == 1 and m['token_f1'] == 1
    assert m['invalid_tool_call_rate'] == 1 and m['context_truncation_rate'] == 1


def test_group_metrics():
    m = group_stats([0, 0, 1, 0], [[0], [0], [1], [-1]], ['a','a','b','b'])
    assert m['zero_advantage_group_ratio'] == .5
    assert m['group_reward_std'] > 0


def test_actual_model_chat_template():
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from jinja2 import TemplateError
    template = (ROOT / 'mini/qwen35_chat_template.jinja').read_text(encoding='utf-8')
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def fail(message):
        raise TemplateError(message)
    t = env.from_string(template)
    kwargs = dict(messages=trajectory()['messages'], tools=[], add_generation_prompt=True, enable_thinking=True, raise_exception=fail)
    rendered = t.render(**kwargs)
    assert rendered.endswith('<|im_start|>assistant\n<think>\n')
    assert '<tool_response>' in rendered and '<answer>Paris</answer>' in rendered
    assert '<tools>' not in rendered and '<function=shell>' not in rendered
    with pytest.raises(TemplateError, match='No user query'):
        t.render(**dict(kwargs, messages=[kwargs['messages'][0]]))


def test_cpu_compile_all_mini():
    for p in (ROOT / 'mini').rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'))


@pytest.mark.skipif(not os.environ.get('MINI_TOKENIZER'), reason='set MINI_TOKENIZER to actual HF tokenizer cache/path')
@pytest.mark.parametrize('empty_think', [False, True])
def test_real_tokenizer_loss_mask(empty_think):
    from transformers import AutoTokenizer
    from mini.template import turn_tokens
    tok = AutoTokenizer.from_pretrained(os.environ['MINI_TOKENIZER'], local_files_only=True)
    messages = trajectory()['messages']
    if empty_think:
        messages[-1]['content'] = '<think>\n\n</think>\n<answer>Paris</answer>'
    ids, masks = [], []
    for i, m in enumerate(messages):
        turn, mask = turn_tokens(tok, messages, i)
        assert len(turn) == len(mask)
        assert any(mask) if m['role'] == 'assistant' else not any(mask)
        ids.extend(turn)
        masks.extend(mask)
    assert ids == tok.apply_chat_template(messages, tools=[], enable_thinking=True, tokenize=True, return_dict=False, add_generation_prompt=False)
    assert '<tool_response>' in tok.decode(ids)


def test_smoke_rejects_no_gpu_evidence(tmp_path):
    from mini.gpu_smoke import verify
    with pytest.raises(RuntimeError):
        verify(tmp_path, 'rl', 2)
