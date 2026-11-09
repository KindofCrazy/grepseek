from collections import Counter
from contextlib import closing
import math
from statistics import mean, stdev
import time
import threading

from mini.diagnostics import aggregate_trajectories
from mini.pytrio_backend.tracking import BudgetStop


def classify(group, config):
    rewards = [t.reward for t in group]
    if len(group) != config.group or not all(math.isfinite(r) for r in rewards):
        raise ValueError('Dynamic sampling requires a complete finite-reward group')
    deviation = stdev(rewards)
    reason = ('all_zero' if all(r == 0 for r in rewards) else 'constant_positive_or_near_flat') \
        if deviation <= config.min_reward_std else None
    if reason is None:
        center = mean(rewards)
        for t in group:
            t.advantage = (t.reward-center)/(max(deviation,config.advantage_std_floor)+config.advantage_eps)
    return deviation, reason


def collect(cursor, config, log, generate, policy_version, serial=0, scheduler=None, pending=None):
    accepted, candidates, ids, sources = [], [], [], []
    reasons = Counter()
    started = time.monotonic()
    rounds = 0
    deviations = []
    allocated = 0
    refill_groups = 0
    advanced_refill_groups = 0
    allocated_by_round = Counter()
    committed_by_round = Counter()
    finished_by_round = Counter()
    telemetry_lock = threading.Lock()
    overlapped_refill_groups = 0
    group_compute_s = 0.
    group_queue_wait_s = 0.
    recovered_groups = 0
    prior_wall_s = 0.
    records = []
    soft_warnings = 0

    def allocate(round_index):
        nonlocal serial, allocated, soft_warnings
        if config.dynamic_sampling_mode=='bounded' and (round_index > config.sampling_rounds or allocated >= config.max_candidate_groups):
            return []
        if config.dynamic_sampling_mode=='continuous' and allocated and allocated % config.max_candidate_groups==0:
            soft_warnings += 1
            log.event('dynamic_soft_limit',policy_version=policy_version,allocated_groups=allocated,
                      committed_groups=len(ids),retained_effective_groups=len(accepted),action='continue_same_policy')
        row = cursor.take(1)[0]
        serial += 1
        allocated += 1
        with telemetry_lock:
            allocated_by_round[round_index] += 1
        return [(row, serial, round_index, time.monotonic())]

    def run_job(job):
        nonlocal overlapped_refill_groups, group_compute_s, group_queue_wait_s
        row, group_serial, round_index, queued_at = job
        began = time.monotonic()
        with telemetry_lock:
            overlap = (round_index > 1 and
                       finished_by_round[round_index-1] < allocated_by_round[round_index-1])
            overlapped_refill_groups += int(overlap)
            group_queue_wait_s += began-queued_at
        log.event('dynamic_group_started', question_id=row['id'], group_serial=group_serial,
                  policy_version=policy_version, round=round_index, queue_wait_s=began-queued_at,
                  overlaps_parent_round=overlap)
        try:
            group = generate(row, group_serial)
        finally:
            with telemetry_lock:
                finished_by_round[round_index] += 1
                group_compute_s += time.monotonic()-began
        log.event('dynamic_group_finished', question_id=row['id'], group_serial=group_serial,
                  policy_version=policy_version, round=round_index, wall_s=time.monotonic()-began)
        return row, group_serial, round_index, group

    def accept_and_refill(result, recovered=False):
        nonlocal rounds, refill_groups, advanced_refill_groups
        row, group_serial, round_index, group = result
        rounds = max(rounds, round_index)
        deviation, reason = classify(group, config)
        ids.append(row['id'])
        sources.append(row.get('source', 'unknown'))
        deviations.append(deviation)
        candidates.extend(group)
        committed_by_round[round_index] += 1
        if reason:
            reasons[reason] += 1
        else:
            accepted.append(group)
        if hasattr(log,'out'):
            records.append({'question_id':row['id'],'group_serial':group_serial,'round':round_index,
                            'trajectories':[t.record() for t in group]})
        log.event('dynamic_recovered_group' if recovered else 'dynamic_group', question_id=row['id'], group_serial=group_serial,
                  policy_version=policy_version, round=round_index, rewards=[t.reward for t in group],
                  reward_sample_std=deviation, accepted=reason is None, drop_reason=reason,
                  advantages=[t.advantage for t in group],
                  total_question_draws=cursor.draws[row['id']])
        replacement = allocate(round_index+1) if reason and not recovered else []
        if replacement:
            refill_groups += 1
            advanced = committed_by_round[round_index] < allocated_by_round[round_index]
            advanced_refill_groups += int(advanced)
            log.event('dynamic_refill_scheduled', policy_version=policy_version,
                      after_group_serial=group_serial, replacement_group_serial=replacement[0][1],
                      round=round_index+1, before_parent_round_committed=advanced)
        return replacement

    try:
        first_round = 1
        if pending:
            from mini.pytrio_backend.rollout import Trajectory
            if pending['policy_version']!=policy_version or pending['last_group_serial']!=serial:
                raise ValueError('Recovered batch does not match policy/serial')
            lookup={r['id']:r for r in cursor.rows}
            recovered_groups=len(pending['groups'])
            prior_wall_s=pending['prior_wall_s']
            allocated=recovered_groups
            for saved in pending['groups']:
                round_index=saved['round']
                allocated_by_round[round_index]+=1
                finished_by_round[round_index]+=1
                group=[Trajectory(**raw) for raw in saved['trajectories']]
                accept_and_refill((lookup[saved['question_id']],saved['group_serial'],round_index,group),True)
            if len(accepted)>config.effective_groups or (config.dynamic_sampling_mode=='bounded' and allocated>config.max_candidate_groups):
                raise ValueError('Recovered batch exceeds fixed group/candidate target')
            first_round=max((g['round'] for g in pending['groups']),default=0)+1
        jobs = [job for _ in range(config.effective_groups-len(accepted)) for job in allocate(first_round)]
        if scheduler:
            with closing(scheduler.ordered_refill(run_job, jobs, accept_and_refill)) as results:
                for _ in results:
                    pass
        else:
            from collections import deque
            pending = deque(jobs)
            while pending:
                pending.extend(accept_and_refill(run_job(pending.popleft())))
    finally:
        selected = [t for group in accepted for t in group]
        stats = {'policy_version':policy_version, 'candidate_groups':len(ids), 'effective_groups':len(accepted),
                 'refill_rounds':rounds, 'refill_groups':refill_groups,
                 'advanced_refill_groups':advanced_refill_groups, 'allocated_groups':allocated,
                 'overlapped_refill_groups':overlapped_refill_groups,
                 'group_compute_s':group_compute_s, 'group_queue_wait_s':group_queue_wait_s,
                 'acceptance_rate':len(accepted)/len(ids) if ids else 0.,
                 'oversampling_factor':len(ids)/len(accepted) if accepted else None,
                 'drop_reasons':dict(reasons), 'question_ids':ids, 'unique_questions':len(set(ids)),
                 'repeated_questions':len(ids)-len(set(ids)), 'source_counts':dict(Counter(sources)),
                 'reward_sample_stds':deviations, 'wall_s':prior_wall_s+time.monotonic()-started,
                 'new_wall_s':time.monotonic()-started,'recovered_wall_s':prior_wall_s,
                 'soft_limit_warnings':soft_warnings,
                 'auxiliary_only_effective_groups':sum(
                     max(t.metrics.get('reward_base_outcome',t.reward) for t in group)-
                     min(t.metrics.get('reward_base_outcome',t.reward) for t in group)<=config.min_reward_std
                     for group in accepted),
                 'advantage_std_floor':config.advantage_std_floor,
                 'recovered_candidate_groups':recovered_groups,'new_candidate_groups':len(ids)-recovered_groups,
                 'candidate_completion_tokens':sum(sum(t.mask) for t in candidates),
                 'accepted_completion_tokens':sum(sum(t.mask) for t in selected),
                 'candidate_context_tokens':sum(len(t.tokens) for t in candidates),
                 'accepted_context_tokens':sum(len(t.tokens) for t in selected),
                 'executed_searches_all':sum(t.metrics.get('executed_tool_calls',0) for t in candidates),
                 'executed_searches_accepted':sum(t.metrics.get('executed_tool_calls',0) for t in selected),
                 'all_candidates':aggregate_trajectories([t.metrics for t in candidates]),
                 'accepted':aggregate_trajectories([t.metrics for t in selected]), 'last_group_serial':serial}
        log.event('dynamic_sampling', **stats)
        from mini.pytrio_backend.agent_metrics import aggregate as agent_aggregate
        agent = agent_aggregate(candidates)
        advantages = [t.advantage for t in selected]
        agent.update(advantage_std=stdev(advantages) if len(advantages)>1 else None,
                     positive_advantage_ratio=mean(a>0 for a in advantages) if advantages else None,
                     negative_advantage_ratio=mean(a<0 for a in advantages) if advantages else None,
                     zero_variance_group_rate=sum(reasons.values())/len(ids) if ids else None)
        log.event('agent_step_metrics',policy_version=policy_version,metrics=agent)
        log.metrics(policy_version+1,{f'agent/{k}':v for k,v in agent.items() if isinstance(v,(float,int))})
        values = {f'dynamic/{k}':v for k,v in stats.items() if isinstance(v,(int,float))}
        for name in ('all_candidates','accepted'):
            values.update({f'rollout/{name}/{k}':v for k,v in stats[name].items() if isinstance(v,(int,float))})
        log.metrics(policy_version+1, values)
    from mini.pytrio_backend.pending import save_pending
    save_pending(log,config,policy_version,records,serial,stats['wall_s'])
    if len(accepted) != config.effective_groups:
        raise BudgetStop('insufficient_nonflat_groups')
    return selected, serial, stats
