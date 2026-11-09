from pathlib import Path
import time

from mini.common import read_jsonl
from mini.pytrio_backend.data import Cursor, datum_tokens
from mini.pytrio_backend.dynamic import collect
from mini.pytrio_backend.loss import ppo_batch
from mini.pytrio_backend.rollout import rollout_group
from mini.pytrio_backend.runtime import adam, optimizer_step, save, trainer
from mini.pytrio_backend.search import Search, pace_options


def run(config, tok, svc, log, budget, bundle=None):
    rows = read_jsonl(Path(config.data) / 'rl_candidates.jsonl')
    cursor = Cursor(rows, config.seed, bundle['cursor'] if bundle else None)
    search = Search(config.corpus, tok, config.tool_timeout, config.tool_tokens,log=log,
                    cache_entries=config.search_cache_entries,cache_bytes=config.search_cache_bytes,
                    backend=config.search_backend,shards=config.shards,pace_dir=config.pace_dir,audit_every=config.pace_audit_every,
                    pace_head_budget=pace_options(config))
    client = trainer(svc, tok, config, log, bundle)
    completed = bundle['step'] if bundle else 0
    serial = bundle.get('last_group_serial',sum(cursor.draws.values())) if bundle else 0
    stop_at = min(config.steps, config.stop_after_step or config.steps)
    from mini.pytrio_backend.pending import load_pending
    pending = load_pending(config,bundle,rows,log)
    initialized = False
    try:
        while completed < stop_at:
            started = time.monotonic()
            # All candidate/refill groups use one immutable sampler snapshot.
            sampler = client.save_weights_and_get_sampling_client()
            if not initialized:
                log.event('trainer_ready',checkpoint_step=completed,optimizer_restored=bool(bundle),
                          before_new_sampling=True)
                initialized = True
            sampler_ready = time.monotonic()
            def generate(row, group_serial):
                return rollout_group(row, group_serial, completed, sampler, tok, search, config, budget, log)
            extra = {'pending':pending} if pending else {}
            selected, serial, sampling = collect(cursor, config, log, generate, completed, serial,
                                                 scheduler=budget.scheduler,**extra)
            pending = None
            rollouts_ready = time.monotonic()
            batch, objective = ppo_batch(selected, config.clip)
            batch_ready = time.monotonic()
            # Custom API runs forward then surrogate backward: count both passes.
            result = budget.call('ppo_custom_forward_backward', {'train':2*datum_tokens(batch)},
                                 lambda: client.forward_backward_custom(batch, objective).result(timeout=config.request_timeout))
            backward_ready = time.monotonic()
            step = completed+1
            params = adam(config, step)
            update = optimizer_step(client, params, step, config, log)
            optimizer_ready = time.monotonic()
            completed = step
            consumption = budget.snapshot()
            values = {'rl/mean_reward':sum(t.reward for t in selected)/len(selected),
                      'train/lr':params.learning_rate,'time/step_s':time.monotonic()-started,
                      'time/sampler_refresh_s':sampler_ready-started,
                      'time/rollout_s':rollouts_ready-sampler_ready,
                      'time/batch_prepare_s':batch_ready-rollouts_ready,
                      'time/forward_backward_s':backward_ready-batch_ready,
                      'time/optimizer_s':optimizer_ready-backward_ready,
                      'budget/estimated_cny':consumption['estimated_uncached_cny'],
                      **{f'budget/{k}_tokens_total':v for k,v in consumption['tokens'].items()},
                      **{f'scheduler/{k}':v for k,v in budget.scheduler.snapshot().items()},
                      **{f'search/{k}':v for k,v in search.stats().items()},
                      **{f'ppo/{k}':v for k,v in result.metrics.items()},
                      **{f'optimizer/{k}':v for k,v in update.metrics.items()}}
            log.event('rl_update', step=step, policy_version=completed-1, metrics=values,
                      accepted_groups=sorted(set(t.group_serial for t in selected)),
                      assistant_tokens=sum(sum(t.mask) for t in selected), sampling=sampling,
                      budget=budget.snapshot())
            log.metrics(step, values)
            print(f'RL step {step}: reward={values["rl/mean_reward"]:.4f}, candidate groups={sampling["candidate_groups"]}',flush=True)
            if step % config.save_every == 0:
                save(client,config,log,budget,cursor,completed,last_group_serial=serial)
        saved = save(client,config,log,budget,cursor,completed,last_group_serial=serial)
        planned_stop = completed < config.steps
        return {'status':'stopped' if planned_stop else 'completed',
                'stop_reason':'planned_step_limit' if planned_stop else None,
                'optimizer_updates':completed,'checkpoint':saved,
                'quality_improvement_claimed':False,'budget':budget.snapshot()}
    except BaseException:
        try:
            save(client,config,log,budget,cursor,completed,interrupted=True,
                 last_group_serial=max(serial,sum(cursor.draws.values())))
        except Exception as error:
            log.event('emergency_checkpoint_failed',error_type=type(error).__name__)
        raise
