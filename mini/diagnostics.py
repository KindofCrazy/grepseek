"""Pure read-only metrics; unknown values stay null rather than invented zeroes."""
from collections import Counter, defaultdict
import argparse
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

from inference.scoring import score
from mini.common import read_jsonl, reward, shell, write_jsonl
from mini.curation_utils import json_file


def average(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def sample_std(values):
    return stdev(values) if len(values) > 1 else 0. if values else None


def transcript_messages(text):
    """Split observed tool payloads without generating or executing a command."""
    result = []
    for i, part in enumerate(re.split(r'<tool_response>\s*(.*?)\s*</tool_response>', text, flags=re.S)):
        if part.strip():
            result.append({'role': 'tool' if i % 2 else 'assistant', 'content':part})
    return result


def protocol_metrics(messages):
    attempts = invalid = observations = errors = repairs = 0
    previous_error = False
    commands = []
    latencies = []
    for msg in messages:
        text = msg.get('content', '')
        if not isinstance(text, str):
            continue
        if msg.get('role') == 'assistant':
            # Count malformed openings and orphan closing tags as attempted calls.
            n_attempts = max(text.count('<tool_call'), text.count('</tool_call>'))
            attempts += n_attempts
            blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S)
            invalid += max(0, n_attempts-len(blocks))
            for block in blocks:
                try:
                    obj = json.loads(block)
                    args = obj['arguments']
                    if obj['name'] != 'shell' or not isinstance(args, dict) or set(args) != {'command'} or not isinstance(args['command'], str) or not args['command'].strip():
                        raise ValueError('invalid tool schema')
                    shell._validate_pipeline(args['command'])
                    commands.append(args['command'])
                except (ValueError, TypeError, KeyError, shell.ToolError):
                    invalid += 1
        elif msg.get('role') == 'tool':
            observations += 1
            try:
                obj = json.loads(text)
                if not isinstance(obj, dict) or 'exit_code' not in obj:
                    continue
                failed = obj['exit_code'] != 0 or bool(obj.get('timed_out'))
                errors += int(failed)
                repairs += int(previous_error and not failed)
                previous_error = failed
            except (ValueError, TypeError):
                pass
            # Official inference records measured wall time around run_tool.
            # Includes executor/tokenizer overhead; not just subprocess latency.
            if isinstance(msg.get('elapsed_s'), (int, float)):
                latencies.append(msg['elapsed_s'])
    return {'tool_call_attempts':attempts, 'invalid_tool_calls':invalid,
            'valid_tool_call_rate':(attempts-invalid)/attempts if attempts else None,
            'invalid_tool_call_rate':invalid/attempts if attempts else None,
            'tool_calls':len(commands), 'tool_observations':observations, 'tool_error_count':errors,
            'repair_transitions':repairs, 'tool_execution_wall_s':sum(latencies) if latencies else None,
            'tool_latency_definition':'measured run_tool wall time including executor/tokenizer overhead; null if absent'}


def trajectory_metrics(text, gold, *, messages=None, training_reward=None, official=None, response_tokens=None, truncated=None):
    messages = transcript_messages(text) if messages is None else messages
    extracted = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, re.S)
    answer = extracted[-1].strip() if extracted else None
    valid_text = text if text.lstrip().startswith(('<think>', '<reason>')) else '<think>\n' + text
    format_pass, _ = reward._is_valid_sequence(valid_text)
    raw = score(answer or '', gold) if gold is not None else {'em':None,'f1':None}
    protocol = protocol_metrics(messages)
    official = official or {}
    return {'raw_em':float(raw['em']) if raw['em'] is not None else None, 'raw_f1':raw['f1'],
            'reward_em':official.get('reward/em'), 'reward_f1':official.get('reward/f1'),
            'final_reward':training_reward, 'format_pass':bool(format_pass), 'no_final_answer':not bool(answer),
            'failed_rollout':official.get('reward/failed_rollout'), 'truncated':truncated,
            'response_tokens':response_tokens, 'assistant_turns':sum(m['role']=='assistant' for m in messages),
            **protocol}


def grpo_metrics(scores, advantages, uids, *, masks=None, em=None, flat_tolerance=1e-12):
    """scores MUST be actual token_level_rewards.sum(-1), after KL/corrections.

    No GRPO advantages are computed here. The tensors are produced by upstream.
    """
    if not scores or not (len(scores) == len(advantages) == len(uids)):
        raise ValueError('nonempty matching rewards/advantages/UIDs required')
    if masks is not None and len(masks) != len(advantages):
        raise ValueError('advantage mask length mismatch')
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('nonfinite actual training reward')
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[str(uid)].append(i)
    flat, allzero, success, means, stds, zero_adv = [], [], [], [], [], []
    valid_adv = []
    for indices in groups.values():
        values = [scores[i] for i in indices]
        means.append(mean(values))
        stds.append(sample_std(values))
        flat.append(max(values)-min(values) <= flat_tolerance)
        allzero.append(all(v == 0 for v in values))
        if em is not None and all(em[i] is not None for i in indices):
            success.append(all(em[i] == 1 for i in indices))
        group_adv = []
        for i in indices:
            if masks is not None and len(masks[i]) != len(advantages[i]):
                raise ValueError('per-token advantage/mask length mismatch')
            group_adv += [v for j,v in enumerate(advantages[i]) if masks is None or masks[i][j] != 0]
        valid_adv += group_adv
        zero_adv.append(all(v == 0 for v in group_adv) if group_adv else None)
    return {'reward_mean':mean(scores), 'reward_std':sample_std(scores), 'group_mean_reward':mean(means),
            'group_reward_std':mean(stds), 'flat_reward_group_rate':mean(flat), 'nonflat_group_rate':1-mean(flat),
            'all_zero_reward_group_rate':mean(allzero), 'all_success_em_group_rate':mean(success) if success else None,
            'all_success_em_observed_group_count':len(success), 'groups':len(groups), 'group_sizes':[len(i) for i in groups.values()],
            'all_success_em_source':'official reward/em when supplied; unavailable otherwise',
            'zero_advantage_group_ratio':average(zero_adv), 'advantage_mean':average(valid_adv), 'advantage_std':sample_std(valid_adv),
            'advantage_positive_ratio':average([v>0 for v in valid_adv]), 'advantage_negative_ratio':average([v<0 for v in valid_adv]),
            'advantage_zero_ratio':average([v==0 for v in valid_adv]), 'advantage_observed_tokens':len(valid_adv),
            'flat_tolerance':flat_tolerance, 'group_reward_source':'actual_training_token_level_rewards_sum',
            'std_definition':'sample standard deviation; one observation => 0'}


def aggregate_trajectories(rows):
    if not rows:
        return {'count':0, 'status':'unavailable_no_observations'}
    attempts = sum(r['tool_call_attempts'] for r in rows)
    invalid = sum(r['invalid_tool_calls'] for r in rows)
    return {'count':len(rows), 'mean_reward':average([r['final_reward'] for r in rows]),
            'std_reward':sample_std([r['final_reward'] for r in rows if r['final_reward'] is not None]),
            'em':average([r['raw_em'] for r in rows]), 'token_f1':average([r['raw_f1'] for r in rows]),
            'format_pass_rate':average([r['format_pass'] for r in rows]),
            'invalid_tool_call_rate':invalid/attempts if attempts else None,
            'failed_rollout_rate':average([r['failed_rollout'] for r in rows]),
            'no_final_answer_rate':average([r['no_final_answer'] for r in rows]),
            'context_truncation_rate':average([r['truncated'] for r in rows]),
            'tool_calls_per_trajectory':average([r['tool_calls'] for r in rows]),
            'tool_calls_on_em_success':average([r['tool_calls'] for r in rows if r['raw_em']==1]),
            'tool_calls_on_em_failure':average([r['tool_calls'] for r in rows if r['raw_em']==0]),
            'response_token_count':average([r['response_tokens'] for r in rows]),
            'assistant_turn_count':average([r['assistant_turns'] for r in rows]),
            'tool_error_count':sum(r['tool_error_count'] for r in rows),
            'tool_execution_wall_s':average([r['tool_execution_wall_s'] for r in rows]),
            'metric_observation_counts':{key:sum(r.get(key) is not None for r in rows) for key in ('raw_em','raw_f1','final_reward','response_tokens','truncated','failed_rollout','tool_execution_wall_s')}}


OPTIMIZATION_KEYS = {
    'policy_loss':['actor/pg_loss'], 'grad_norm':['actor/grad_norm','train/grad_norm'],
    'clip_fraction':['actor/pg_clipfrac'], 'learning_rate':['actor/lr','train/lr'],
    # actor/kl_loss is initialized to 0 even when KL loss is disabled. Do not
    # expose that placeholder as an observed KL. PPO KL is a genuine estimate.
    'kl':['actor/ppo_kl'], 'training_step_time':['timing_s/step'],
    'rollout_generation_time':['timing_s/gen'], 'gpu_memory_peak':['perf/max_memory_allocated_gb'],
}


def optimization_metrics(metrics):
    return {name:{'value':next((metrics[k] for k in keys if k in metrics), None),
                  'source_key':next((k for k in keys if k in metrics), None),
                  'status':'observed' if any(k in metrics for k in keys) else 'unavailable'} for name,keys in OPTIMIZATION_KEYS.items()}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', required=True, nargs='+', help='Actual official rollout JSONL / evaluation records')
    p.add_argument('--scope', required=True, choices=['training_rollout','validation','final_evaluation'])
    p.add_argument('--out', required=True)
    a = p.parse_args()
    rows = []
    groups = []
    optimization = []
    unavailable = []
    for filename in a.input:
        for r in read_jsonl(filename):
            if 'event' in r:
                observed_scope=r.get('scope')
                if observed_scope==a.scope and r['event']=='agent_trajectory':
                    rows.append(r['metrics'])
                elif observed_scope==a.scope and r['event']=='grpo_groups':
                    groups.append(r)
                elif a.scope=='training_rollout' and observed_scope=='training_optimization' and r['event']=='training_metrics':
                    optimization.append({'step':r['step'],'scope':'training_optimization','metrics':r.get('optimization',optimization_metrics(r['metrics']))})
                elif observed_scope is None:
                    unavailable.append({'event':r['event'],'reason':'scope_not_available; not merged into task metrics'})
                continue
            if r.get('scope') is not None and r['scope'] != a.scope:
                raise ValueError('refusing to mix scopes')
            if 'output' in r:
                gt = r.get('gts')
                if isinstance(gt, str):
                    try:
                        gt=json.loads(gt)
                    except ValueError:
                        gt=None
                rows.append(trajectory_metrics(r['output'], (gt or {}).get('golden_answers') if isinstance(gt,dict) else None,
                    training_reward=r.get('score'), official=r,
                    response_tokens=r.get('metrics/response_token_count'), truncated=r.get('reward/hit_max_context')))
            elif 'turns' in r:
                messages = [m for m in r['turns'] if m['role'] in ('assistant','tool')]
                text = '\n'.join(m['content'] if m['role']=='assistant' else '<tool_response>'+m['content']+'</tool_response>' for m in messages)
                # API errors unrelated to context are not counted as context truncation.
                truncated = any(t.get('finish_reason')=='length' for t in messages) or 'context' in (r.get('error') or '').lower()
                rows.append(trajectory_metrics(text, r.get('gold_answers'), messages=messages, truncated=truncated))
            else:
                raise ValueError('unsupported diagnostic input schema')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / f'{a.scope}_trajectories.jsonl', rows)
    json_file(out / f'{a.scope}_report.json', {'scope':a.scope, 'summary':aggregate_trajectories(rows),
              'grpo_metric_events':groups if groups else None,
              'optimization_metric_events':optimization if optimization else None,
              'excluded_unscoped_events':unavailable,
              'availability_notes':'null means unavailable. GRPO group metrics require runtime actual rewards/UIDs/advantages/masks, never inferred from EM.'})


if __name__ == '__main__':
    main()
