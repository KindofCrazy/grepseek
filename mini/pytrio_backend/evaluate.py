from pathlib import Path
from contextlib import closing

from mini.common import read_jsonl
from mini.diagnostics import aggregate_trajectories
from mini.pytrio_backend.rollout import rollout_group
from mini.pytrio_backend.runtime import read_bundle
from mini.pytrio_backend.search import Search, pace_options


def run(config, tok, svc, log, budget, split='dev', count=2):
    path = config.eval_file or {'dev':'rl_dev.jsonl','pilot':'benchmark30/test.jsonl'}[split]
    rows = read_jsonl(Path(config.data)/path)
    if split == 'pilot' and count != len(rows):
        raise ValueError('Frozen pilot must be evaluated in full; use dev for smoke checks')
    rows = rows[:count]
    weights = None
    if config.initial_checkpoint:
        weights = read_bundle(config.initial_checkpoint)['sampler_weights']
    search = Search(config.corpus,tok,config.tool_timeout,config.tool_tokens,log=log,
                    cache_entries=config.search_cache_entries,cache_bytes=config.search_cache_bytes,
                    backend=config.search_backend,shards=config.shards,pace_dir=config.pace_dir,audit_every=config.pace_audit_every,
                    pace_head_budget=pace_options(config))
    sampler = svc.create_sampling_client(config.model,model_path=weights)
    trajectories = []
    def generate(item):
        serial,row = item
        return rollout_group(row,serial,0,sampler,tok,search,config,budget,log,
                             group_size=1,temperature=0.)
    with closing(budget.scheduler.completed(generate, enumerate(rows,1))) as groups:
        for group in groups:
            trajectories.extend(group)
            log.event('evaluation_progress',completed=len(trajectories),total=len(rows))
    trajectories.sort(key=lambda t:t.group_serial)
    metrics = aggregate_trajectories([t.metrics for t in trajectories])
    from mini.pytrio_backend.agent_metrics import aggregate as agent_aggregate
    agent = agent_aggregate(trajectories)
    success_count = agent['strict_success_count']
    agent['estimated_cny_per_strict_success_including_failures'] = budget.snapshot()['estimated_incremental_cny']/success_count if success_count else None
    agent['cost_definition'] = 'uncached incremental eval estimate across all attempts / strict successes; not an invoice'
    log.event('agent_evaluation',split=split,metrics=agent)
    log.metrics(0,{f'agent_eval/{k}':v for k,v in agent.items() if isinstance(v,(int,float))})
    log.event('evaluation_summary',split=split,temperature=0.,sampler_weights=weights,metrics=metrics)
    log.metrics(0,{f'eval/{split}/{k}':v for k,v in metrics.items() if isinstance(v,(int,float))})
    return {'status':'completed','split':split,'metrics':metrics,
            'sampler_weights':weights,'agent_metrics':agent,'budget':budget.snapshot(),'quality_improvement_claimed':False}
