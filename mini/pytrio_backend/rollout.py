"""Preserve sampled token IDs; append only template/environment suffixes."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
import json
import math
import re
import time

import pytrio as trio

from mini.common import reward
from mini.diagnostics import trajectory_metrics
from mini.pytrio_backend.tracking import BudgetStop


def prompt_tokens(tok, messages, generation=True):
    return tok.apply_chat_template(messages, tools=[], enable_thinking=True, tokenize=True,
                                   return_dict=False, add_generation_prompt=generation)


def observation_suffix(tok, payload, completion):
    # Qwen renders tool messages as user/<tool_response>. Derive its exact
    # serialization through the actual template with a disposable user prefix.
    dummy = [{'role':'user','content':'boundary'}]
    prefix = prompt_tokens(tok, dummy, generation=False)
    rendered = prompt_tokens(tok, dummy + [{'role':'tool','content':json.dumps(payload, ensure_ascii=False)}])
    if rendered[:len(prefix)] != prefix:
        raise ValueError('Tool template modified its prefix')
    close = tok.encode('<|im_end|>\n', add_special_tokens=False)
    overlap = 0
    for size in range(min(len(close),len(completion)),0,-1):
        if completion[-size:] == close[:size]:
            overlap = size
            break
    return close[overlap:] + rendered[len(prefix):]


@dataclass
class Trajectory:
    question_id: str
    source: str
    question: str
    gold: list
    group_serial: int
    sample_index: int
    policy_version: int
    tokens: list = field(default_factory=list)
    mask: list = field(default_factory=list)
    old_logprobs: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    initial_length: int = 0
    finish_reason: str = ''
    reward: float = 0.
    reward_info: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    advantage: float = 0.
    tool_count: int = 0
    invalid_request: bool = False

    def record(self):
        return asdict(self)


def validate_sequence(sequence):
    if len(sequence.tokens) != len(sequence.logprobs) or not all(math.isfinite(v) for v in sequence.logprobs):
        raise ValueError('Sampled tokens/logprobs are not finite and aligned')


def consume(traj, sequence, tok, search, config, scheduler=None):
    validate_sequence(sequence)
    generated = list(sequence.tokens)
    if not generated:
        traj.finish_reason = 'empty_generation'
        return False
    text = tok.decode(generated, skip_special_tokens=True)
    # The first <think> is already in the generation prefix, not sampled.
    if not text.lstrip().startswith(('<think>', '<reason>')):
        text = '<think>\n' + text
    previous = len(traj.tokens)
    traj.tokens.extend(generated)
    traj.mask.extend([1]*len(generated))
    traj.old_logprobs.extend(sequence.logprobs)
    traj.messages.append({'role':'assistant','content':text})
    traj.turns.append({'prompt_length':previous, 'tokens':generated, 'old_logprobs':sequence.logprobs,
                       'stop_reason':sequence.stop_reason, 'text':text})
    answers = list(re.finditer(r'<answer>\s*(.*?)\s*</answer>',text,re.S))
    calls = list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S))
    if answers and (not calls or answers[0].start() < calls[0].start()):
        traj.finish_reason = 'answer'
        return False
    if not calls:
        traj.invalid_request = '<tool_call' in text or '</tool_call>' in text
        traj.finish_reason = 'invalid_request' if traj.invalid_request else 'no_action'
        return False
    try:
        if len(calls) != 1:
            raise ValueError('one tool call required')
        call = json.loads(calls[0].group(1))
        if call['name'] != 'shell' or not isinstance(call['arguments'],dict):
            raise ValueError('invalid tool name or arguments')
        command = call['arguments']['command']
        if not isinstance(command,str) or not command.strip():
            raise ValueError('invalid command')
    except (ValueError,TypeError,KeyError):
        traj.invalid_request = True
        traj.finish_reason = 'invalid_request'
        return False
    if traj.tool_count >= config.max_tools or len(traj.turns) >= config.max_turns:
        traj.finish_reason = 'turn_budget'
        return False
    started = time.monotonic()
    from contextlib import nullcontext
    with scheduler.slot('search') if scheduler else nullcontext():
        payload = search.execute(command.strip())
    if getattr(search,'log',None):
        from mini.pytrio_backend.agent_metrics import tool_outcome
        search.log.event('tool_call',question_id=traj.question_id,group_serial=traj.group_serial,
                         sample_index=traj.sample_index,tool_index=traj.tool_count+1,command=command.strip(),
                         elapsed_s=time.monotonic()-started,result=payload,outcome=tool_outcome(payload))
    traj.tool_count += 1
    traj.messages.append({'role':'tool','content':json.dumps(payload,ensure_ascii=False),
                          'elapsed_s':time.monotonic()-started,'tool_result':payload})
    suffix = observation_suffix(tok, payload, generated)
    remaining = min(config.context-len(traj.tokens), config.response-(len(traj.tokens)-traj.initial_length))
    if len(suffix) > remaining:
        # Preserve complete observation in logs; no partial observation enters a model request.
        traj.finish_reason = 'observation_budget_overflow'
        return False
    traj.tokens.extend(suffix)
    traj.mask.extend([0]*len(suffix))
    traj.old_logprobs.extend([0.]*len(suffix))
    return True


def score(traj, config):
    transcript = '\n'.join(m['content'] if m['role']=='assistant' else
                            '<tool_response>'+m['content']+'</tool_response>' for m in traj.messages)
    info = {'response_token_count':len(traj.tokens)-traj.initial_length,
            'hit_max_context':traj.finish_reason in ('context_budget','observation_budget_overflow'),
            'invalid_tool_request':traj.invalid_request}
    result = reward.grepseek_reward(data_source=traj.source, solution_str=transcript,
                                    ground_truth={'golden_answers':traj.gold},extra_info=info,
                                    enable_length_decay=False,reward_metric='f1',L_max=config.response)
    from mini.pytrio_backend.rewards import auxiliary_reward
    result = auxiliary_reward(result,traj,config)
    traj.reward, traj.reward_info = result['score'], result
    traj.metrics = trajectory_metrics(transcript,traj.gold,messages=traj.messages,
                                      training_reward=traj.reward,official=result,
                                      response_tokens=len(traj.tokens)-traj.initial_length,
                                      truncated=info['hit_max_context'] or any('length' in t['stop_reason'].lower() for t in traj.turns))
    traj.metrics['executed_tool_calls'] = traj.tool_count
    traj.metrics.update(reward_base_outcome=result['reward/base_outcome'],
                        reward_auxiliary=result['reward/auxiliary'],
                        reward_format_component=result['reward/format_component'],
                        reward_f1_component=result['reward/f1_component'],
                        reward_valid_format=result['reward/valid_format'],
                        completion_failure=bool(result['reward/completion_failure']))
    if not (len(traj.tokens)==len(traj.mask)==len(traj.old_logprobs)):
        raise ValueError('Trajectory token/mask/logprob misalignment')
    return traj


def rollout_group(row, serial, policy_version, sampler, tok, search, config, budget, log,
                  group_size=None, temperature=None):
    from inference.agent import build_system_prompt
    messages = [{'role':'system','content':build_system_prompt()},
                {'role':'user','content':'Query: '+row['question']}]
    initial = prompt_tokens(tok,messages)
    if len(initial)>config.prompt:
        raise ValueError('Initial prompt over budget')
    count = group_size or config.group
    temp = config.temperature if temperature is None else temperature
    seed = (config.seed + serial*1009 + policy_version*1000003) % (2**31-1)
    maximum = min(config.max_turn_tokens,config.response,config.context-len(initial))
    request_started = time.monotonic()
    first = budget.sample(sampler,initial,count,trio.SamplingParams(max_tokens=maximum,seed=seed,temperature=temp,top_p=config.top_p))
    if first.input_tokens != len(initial) or len(first.sequences)!=count:
        raise ValueError('Sampler response prompt/group size mismatch')

    def branch(index_sequence):
        index, sequence = index_sequence
        traj = Trajectory(row['id'],row.get('source','unknown'),row['question'],row['golden_answers'],
                          serial,index,policy_version,tokens=list(initial),mask=[0]*len(initial),
                          old_logprobs=[0.]*len(initial),initial_length=len(initial))
        while True:
            if not consume(traj,sequence,tok,search,config,budget.scheduler):
                break
            maximum = min(config.max_turn_tokens, config.context-len(traj.tokens),
                          config.response-(len(traj.tokens)-traj.initial_length))
            if maximum <= 0:
                traj.finish_reason='context_budget'
                break
            result = budget.sample(sampler,traj.tokens,1,trio.SamplingParams(
                max_tokens=maximum,seed=(seed+index*97+len(traj.turns))%(2**31-1),temperature=temp,top_p=config.top_p))
            if result.input_tokens != len(traj.tokens) or len(result.sequences)!=1:
                raise ValueError('Continuation prompt mismatch')
            sequence = result.sequences[0]
        score(traj,config)
        from mini.pytrio_backend.agent_metrics import trajectory as agent_metrics
        traj.metrics.update(agent_metrics(traj,time.monotonic()-request_started))
        return traj

    try:
        with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
            group = list(pool.map(branch,enumerate(first.sequences)))
    except BaseException as error:
        # A failed RPC/infra group cannot be reinterpreted as an all-wrong question.
        log.event('group_aborted', question_id=row['id'],group_serial=serial,
                  policy_version=policy_version,error_type=type(error).__name__)
        raise
    for traj in group:
        log.event('trajectory',trajectory=traj.record())
    return group
