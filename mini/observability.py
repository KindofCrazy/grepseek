"""Read-only observer for upstream optimizer, metrics and computed advantages."""
import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from pathlib import Path


def emit(kind, **data):
    directory = Path(os.environ['MINI_OBSERVE_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'{os.getpid()}.jsonl').open('a', encoding='utf-8') as f:
        def serialize(value):
            if hasattr(value, 'item'):
                return value.item()
            if hasattr(value, 'tolist'):
                return value.tolist()
            return str(value)
        f.write(json.dumps({'event': kind, 'pid': os.getpid(), 'time': time.time(), **data}, default=serialize) + '\n')


def group_stats(scores, advantages, uids, masks=None, em=None):
    from mini.diagnostics import grpo_metrics
    result = grpo_metrics(scores, advantages, uids, masks=masks, em=em)
    result['rollout_reward'] = result['reward_mean']
    return result


def patch(module):
    if module.__name__ == 'verl.utils.tracking':
        original = module.Tracking.log
        previous = {}
        @functools.wraps(original)
        def log(self, data, step, *args, **kwargs):
            now = time.perf_counter()
            from mini.diagnostics import optimization_metrics
            train = {k:v for k,v in data.items() if not k.startswith(('val/', 'val-', 'validation/'))}
            val = {k:v for k,v in data.items() if k not in train}
            if train:
                emit('training_metrics', scope='training_optimization', step=int(step), metrics=train,
                     optimization=optimization_metrics(train),
                     observed_inter_log_wall_s=now-previous[id(self)] if id(self) in previous else None)
            if val:
                emit('validation_metrics', scope='validation', step=int(step), metrics=val)
            previous[id(self)] = now
            return original(self, data, step, *args, **kwargs)
        module.Tracking.log = log
    elif module.__name__ == 'verl.trainer.ppo.ray_trainer':
        original = module.compute_advantage
        @functools.wraps(original)
        def compute(*args, **kwargs):
            result = original(*args, **kwargs)
            # Inspect actual upstream GRPO output, grouped by UID after DP balancing.
            estimator = kwargs.get('adv_estimator', args[1] if len(args)>1 else None)
            if getattr(estimator, 'value', estimator) == 'grpo':
                scores = result.batch['token_level_rewards'].sum(-1).detach().cpu().tolist()
                adv = result.batch['advantages'].detach().cpu().tolist()
                masks = result.batch['response_mask'].detach().cpu().tolist()
                em = result.non_tensor_batch.get('reward/em')
                emit('grpo_groups', scope='training_rollout', **group_stats(scores, adv, result.non_tensor_batch['uid'], masks=masks, em=em))
            return result
        module.compute_advantage = compute
        original_log_rollout = module.RayPPOTrainer._log_rollout_data
        @functools.wraps(original_log_rollout)
        def log_rollout(self, batch, reward_extra_infos_dict, *args, **kwargs):
            from mini.diagnostics import aggregate_trajectories, trajectory_metrics
            outputs = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            scores = batch.batch['token_level_rewards'].sum(-1).detach().cpu().tolist()
            rows = []
            for i,text in enumerate(outputs):
                gt = batch.non_tensor_batch['reward_model'][i].get('ground_truth', {})
                info = {k:v[i] for k,v in reward_extra_infos_dict.items() if len(v)==len(outputs)}
                rows.append(trajectory_metrics(text, gt.get('golden_answers'), training_reward=scores[i], official=info,
                            response_tokens=info.get('metrics/response_token_count'), truncated=info.get('reward/hit_max_context')))
            emit('rollout_diagnostics', scope='training_rollout', step=int(self.global_steps), summary=aggregate_trajectories(rows))
            return original_log_rollout(self, batch, reward_extra_infos_dict, *args, **kwargs)
        module.RayPPOTrainer._log_rollout_data = log_rollout
    elif module.__name__ == 'verl.experimental.agent_loop.agent_loop':
        original = module.AgentLoopWorker._agent_loop_postprocess
        @functools.wraps(original)
        async def postprocess(self, output, validate, **kwargs):
            result = await original(self, output, validate, **kwargs)
            from mini.diagnostics import trajectory_metrics
            text = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
            gold = kwargs.get('reward_model', {}).get('ground_truth', {}).get('golden_answers')
            info = output.extra_fields.get('reward_extra_info', {})
            stats = trajectory_metrics(text, gold, training_reward=output.reward_score, official=info,
                        response_tokens=len(output.response_ids), truncated=output.extra_fields.get('hit_max_context'))
            stats['tool_execution_wall_s'] = output.metrics.tool_calls
            stats['tool_latency_definition'] = 'official AgentLoopMetrics.tool_calls timer around asyncio.gather; includes scheduler and tool overhead'
            emit('agent_trajectory', scope='validation' if validate else 'training_rollout',
                 metrics=stats, rollout_generation_s=output.metrics.generate_sequences,
                 reward_source='official_rollout_score_before_training_KL; grpo_groups uses actual post-KL training reward',
                 measured_response_mask_trainable_tokens=sum(output.response_mask))
            return result
        module.AgentLoopWorker._agent_loop_postprocess = postprocess


class ObserverFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ('verl.utils.tracking', 'verl.trainer.ppo.ray_trainer', 'verl.experimental.agent_loop.agent_loop'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        loader = spec.loader
        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return loader.create_module(spec)
            def exec_module(self, module):
                loader.exec_module(module)
                patch(module)
        spec.loader = Loader()
        return spec


def install():
    sys.meta_path.insert(0, ObserverFinder())
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    started = time.perf_counter()
    def forward(module, inputs, output):
        def has_cuda(x):
            if isinstance(x, torch.Tensor):
                return x.is_cuda and x.requires_grad
            if isinstance(x, (tuple, list)):
                return any(has_cuda(v) for v in x)
            if isinstance(x, dict):
                return any(has_cuda(v) for v in x.values())
            return False
        if has_cuda(output):
            emit('cuda_forward', module=type(module).__name__)
            forward_handle.remove()
    forward_handle = torch.nn.modules.module.register_module_forward_hook(forward)
    def step(optimizer, args, kwargs):
        params = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        emit('optimizer_step', optimizer=type(optimizer).__name__, gradient_tensors=len(params),
             cuda_available=torch.cuda.is_available(), since_process_start_s=time.perf_counter() - started,
             max_memory_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
             max_memory_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None)
    register_optimizer_step_post_hook(step)

