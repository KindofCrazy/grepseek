"""Agent measurements with explicit denominators; unavailable policy quantities stay null."""
from collections import Counter
import json
import math
import re
import shlex
from statistics import mean


def fraction(n,d): return n/d if d else None


def command_parts(command):
    try:
        lexer = shlex.shlex(command,posix=True,punctuation_chars='|')
        lexer.whitespace_split = True
        segments, current = [], []
        for token in lexer:
            if token == '|':
                segments.append(current); current = []
            else: current.append(token)
        segments.append(current)
        return segments
    except ValueError:
        return []


def tool_outcome(payload):
    parts = command_parts(payload.get('command',''))
    programs = [p[0].split('/')[-1] for p in parts if p]
    search = any(p in ('rg','grep') for p in programs)
    code = payload.get('exit_code')
    timeout = bool(payload.get('timed_out'))
    stdout, stderr = payload.get('stdout',''),payload.get('stderr','')
    # Count-only pipelines are not classified as nonempty retrieval just because they print "0".
    count_only = bool(programs and programs[-1]=='wc')
    empty = not stdout.strip() or (count_only and stdout.strip().isdigit() and int(stdout.strip())==0)
    no_match = code==1 and search and not stderr and empty
    executed = (code==0 or no_match) and not stderr.strip()
    error = timeout or not executed
    return {'execution_success':not error,'search':search,'empty_search':search and not error and empty,
            'nonempty_search':search and not error and not empty and not count_only,
            'count_only':count_only,'error':error,'programs':programs,
            'error_type':'timeout' if timeout else 'invalid_command' if code==-2 else 'execution_error' if error else None}


def trajectory(traj,elapsed_s=None):
    outcomes,commands,latencies = [],[],[]
    for message in traj.messages:
        if message['role']=='tool':
            payload = message.get('tool_result') or json.loads(message['content'])
            outcome = tool_outcome(payload)
            outcomes.append(outcome)
            commands.append(json.dumps(command_parts(payload.get('command','')),ensure_ascii=False))
            if isinstance(message.get('elapsed_s'),(int,float)): latencies.append(message['elapsed_s'])
    decisions = len(traj.turns)
    tool_decisions = sum('<tool_call>' in t['text'] for t in traj.turns)
    answer_decisions = sum('<answer>' in t['text'] for t in traj.turns)
    attempts = traj.metrics['tool_call_attempts']
    invalid = max(traj.metrics['invalid_tool_calls'],int(traj.invalid_request))
    invalid_before_execution = max(0,invalid-sum(o['error_type']=='invalid_command' for o in outcomes))
    generated = sum(len(t['tokens']) for t in traj.turns)
    sample_lps = [p for turn in traj.turns for p in turn['old_logprobs']]
    search_n = sum(o['search'] for o in outcomes)
    nonempty_den = sum(o['search'] and not o['count_only'] for o in outcomes)
    success = bool(traj.metrics['raw_em']==1 and traj.metrics['format_pass'] and not traj.metrics['no_final_answer'])
    return {'generated_completion_tokens':generated,'trajectory_latency_s':elapsed_s,
            'strict_success':success,'normal_termination':traj.finish_reason=='answer',
            'tool_execution_numerator':sum(o['execution_success'] for o in outcomes),'tool_execution_denominator':len(outcomes),
            'tool_execution_success_rate':fraction(sum(o['execution_success'] for o in outcomes),len(outcomes)),
            'tool_error_numerator':sum(o['error'] for o in outcomes)+invalid_before_execution,
            'tool_error_denominator':len(outcomes)+invalid_before_execution,
            'tool_error_rate':fraction(sum(o['error'] for o in outcomes)+invalid_before_execution,len(outcomes)+invalid_before_execution),
            'empty_search_numerator':sum(o['empty_search'] for o in outcomes),'empty_search_denominator':search_n,
            'empty_search_rate':fraction(sum(o['empty_search'] for o in outcomes),search_n),
            'nonempty_result_numerator':sum(o['nonempty_search'] for o in outcomes),'nonempty_result_denominator':nonempty_den,
            'tool_nonempty_result_rate':fraction(sum(o['nonempty_search'] for o in outcomes),nonempty_den),
            'decision_count':decisions,'tool_decision_count':tool_decisions,'answer_decision_count':answer_decisions,
            'tool_use_decision_rate':fraction(tool_decisions,decisions),
            'query_count':len(commands),'unique_query_count':len(set(commands)),
            'repeated_query_rate':fraction(len(commands)-len(set(commands)),len(commands)),
            'query_normalization':'shell token/whitespace exact equivalence only; semantic equivalence unavailable',
            'tool_type_counts':dict(Counter(p for o in outcomes for p in o['programs'])),
            'tool_error_types':dict(Counter(o['error_type'] for o in outcomes if o['error_type'])),
            'tool_latencies_s':latencies,
            'policy_entropy':None,'action_entropy':None,'entropy_status':'SDK returns sampled-token logprob, not full distribution',
            'sampled_token_surprisal_mean':-mean(sample_lps) if sample_lps else None,
            'sampled_surprisal_is_exact_entropy':False,
            'evidence_hit_rate':None,'evidence_status':'no verified evidence labels; answer-string hits are not evidence',
            'reward_total':traj.reward,'reward_answer_f1':traj.metrics['raw_f1'],
            'reward_format':int(traj.metrics['format_pass']),'reward_efficiency':None,
            'efficiency_reward_enabled':False}


def aggregate(trajectories):
    if not trajectories: return {'count':0}
    metrics = [t.metrics for t in trajectories]
    result = {'count':len(metrics),'policy_entropy':None,'action_entropy':None,'gradient_norm':None,
              'entropy_status':'full policy distribution unavailable','evidence_hit_rate':None,
              'normal_termination_rate':mean(m.get('normal_termination',False) for m in metrics),
              'reward_total':mean(t.reward for t in trajectories),
              'reward_answer_f1':mean(m['raw_f1'] for m in metrics),
              'reward_format':mean(m['format_pass'] for m in metrics),
              'reward_zero_rate':mean(t.reward==0 for t in trajectories),
              'reward_positive_rate':mean(t.reward>0 for t in trajectories),
              'reward_negative_rate':mean(t.reward<0 for t in trajectories),
              'reward_base_outcome':mean(m.get('reward_base_outcome',t.reward) for m,t in zip(metrics,trajectories)),
              'reward_auxiliary':mean(m.get('reward_auxiliary',0.) for m in metrics),
              'reward_format_component':mean(m.get('reward_format_component',0.) for m in metrics),
              'reward_f1_component':mean(m.get('reward_f1_component',t.reward) for m,t in zip(metrics,trajectories)),
              'reward_valid_format_rate':mean(m.get('reward_valid_format',m['format_pass']) for m in metrics),
              'completion_failure_rate':mean(m.get('completion_failure',False) for m in metrics),
              'reward_efficiency':None,'efficiency_reward_enabled':False}
    for name,num,den in [('tool_execution_success_rate','tool_execution_numerator','tool_execution_denominator'),
                         ('tool_error_rate','tool_error_numerator','tool_error_denominator'),
                         ('empty_search_rate','empty_search_numerator','empty_search_denominator'),
                         ('tool_nonempty_result_rate','nonempty_result_numerator','nonempty_result_denominator'),
                         ('tool_use_decision_rate','tool_decision_count','decision_count'),
                         ('unique_query_ratio','unique_query_count','query_count')]:
        n,d = sum(m.get(num,0) for m in metrics),sum(m.get(den,0) for m in metrics)
        result[name],result[name+'_numerator'],result[name+'_denominator'] = fraction(n,d),n,d
    def n_calls(t): return getattr(t,'tool_count',t.metrics.get('executed_tool_calls',t.metrics.get('tool_calls',0)))
    calls = sorted({n_calls(t) for t in trajectories})
    result['accuracy_by_tool_count'] = {str(c):{'count':sum(n_calls(t)==c for t in trajectories),
        'em':mean(t.metrics['raw_em'] for t in trajectories if n_calls(t)==c),
        'f1':mean(t.metrics['raw_f1'] for t in trajectories if n_calls(t)==c)} for c in calls}
    result['success_per_tool_call'] = fraction(sum(m.get('strict_success',False) for m in metrics),sum(n_calls(t) for t in trajectories))
    result['repeated_query_rate'] = 1-result['unique_query_ratio'] if result['unique_query_ratio'] is not None else None
    result['answer_decision_rate'] = fraction(sum(m.get('answer_decision_count',0) for m in metrics),sum(m.get('decision_count',0) for m in metrics))
    successes = [m for m in metrics if m.get('strict_success')]
    result['strict_success_count'] = len(successes)
    result['success_generated_tokens_mean'] = mean(m['generated_completion_tokens'] for m in successes) if successes else None
    success_latencies = [m['trajectory_latency_s'] for m in successes if m.get('trajectory_latency_s') is not None]
    result['success_latency_mean_s'] = mean(success_latencies) if success_latencies else None
    result['success_latency_observed_n'] = len(success_latencies)
    result['generated_tokens_per_success_including_failures'] = fraction(sum(m.get('generated_completion_tokens',0) for m in metrics),len(successes))
    latencies = sorted(x for m in metrics for x in m.get('tool_latencies_s',[]))
    result['tool_latency_observed_calls'] = len(latencies)
    for p in (.50,.95):
        result[f'tool_latency_p{int(p*100)}_s'] = latencies[max(0,math.ceil(p*len(latencies))-1)] if latencies else None
    normalized = [re.sub(r'\s+',' ',t.turns[-1]['text']).strip() for t in trajectories if getattr(t,'turns',None)]
    result['response_exact_unique_ratio'] = fraction(len(set(normalized)),len(normalized))
    result['response_diversity_definition'] = 'normalized final assistant text exact uniqueness, not semantic diversity'
    result['tool_type_counts'] = dict(Counter(p for m in metrics for p,n in m.get('tool_type_counts',{}).items() for _ in range(n)))
    result['tool_type_denominator'] = sum(result['tool_type_counts'].values())
    observations = [(m.get('sampled_token_surprisal_mean'),m.get('generated_completion_tokens',0)) for m in metrics]
    observed = [(value,n) for value,n in observations if value is not None and n]
    result['sampled_token_surprisal_mean'] = sum(v*n for v,n in observed)/sum(n for _,n in observed) if observed else None
    result['sampled_surprisal_is_exact_entropy'] = False
    return result
