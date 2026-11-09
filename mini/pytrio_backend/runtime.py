import json
from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
import time

import pytrio as trio

from mini.pytrio_backend.tracking import json_write


def service(config, log):
    from pytrio.config.config import configuration
    configuration.training_retry_timeout_seconds = 120
    configuration.control_retry_timeout_seconds = 120
    client = trio.ServiceClient()
    capabilities = client.get_server_capabilities()
    model = next((m for m in capabilities.supported_models if m.model_name == config.model), None)
    if model is None or not model.sampling.available or (config.stage != 'eval' and not model.training.available):
        raise ValueError('Requested model operation is unavailable')
    limits = [model.max_context_length, model.training.max_seq_len, model.sampling.max_seq_len]
    if any(v is not None and config.context > v for v in limits):
        raise ValueError('Configured context exceeds current server capabilities')
    json_write(log.out / 'capabilities.json', capabilities.model_dump(mode='json'))
    log.event('service_created', service_session_id=client.service_session_cuid)
    return client


def bind_tokenizer(client, tokenizer):
    # Public SDK tokenizer accessor is bound to the already hash-verified local
    # tokenizer. This avoids implicit ModelScope downloads during batch validation.
    client.get_tokenizer = lambda: tokenizer
    return client


def read_bundle(path):
    path = Path(path)
    bundle = json.loads(path.read_text(encoding='utf-8'))
    result_path = path.parent/'result.json'
    if 'train_state' in bundle and result_path.exists():
        # A periodic checkpoint can precede later validation or an interrupted
        # rollout. Those paid requests still count when resuming this run.
        result = json.loads(result_path.read_text(encoding='utf-8'))
        latest = result.get('budget',{}).get('tokens',{})
        previous = bundle.get('budget',{}).get('tokens',{})
        if latest:
            bundle['budget']['tokens'] = {k:max(previous.get(k,0),latest.get(k,0)) for k in previous.keys()|latest.keys()}
    return bundle


def verify_step_extension(config, bundle, log):
    """Extend a completed constant-LR RL run without resetting its optimizer."""
    from mini.curation_utils import file_hash
    from mini.pytrio_backend.pending import protocol_hashes
    before=bundle['config']['steps']
    if not config.resume_migration:
        raise ValueError('RL step extension requires a checkpoint-bound resume_migration receipt')
    receipt=json.loads(Path(config.resume_migration).read_text())
    if (config.stage!='rl' or bundle['config']['stage']!='rl'
        or config.lr_schedule!='constant' or bundle['config'].get('lr_schedule','constant')!='constant'
        or not before < config.steps <= 100 or bundle['step']!=before
        or receipt.get('kind')!='rl_step_limit_extension'
        or receipt.get('checkpoint_sha256')!=file_hash(config.resume)
        or receipt.get('checkpoint_step')!=bundle['step']
        or receipt.get('train_state')!=bundle['train_state']
        or receipt.get('steps')!={'before':before,'after':config.steps}
        or receipt.get('data_manifest_sha256')!=file_hash(Path(config.data)/'sft_manifest.json')
        or receipt.get('protocol_sha256')!=protocol_hashes()):
        raise ValueError('Invalid RL step extension receipt or nonconstant schedule')
    log.event('resume_step_limit_extension',receipt=str(config.resume_migration),
              receipt_sha256=file_hash(config.resume_migration),steps=receipt['steps'],
              checkpoint_step=bundle['step'],optimizer_preserved=True)


def trainer(service_client, tokenizer, config, log, bundle=None):
    restore_verified = False
    if bundle:
        if not bundle.get('resume_safe', True):
            raise ValueError('Checkpoint has an uncertain optimizer outcome; exact resume is unsafe')
        log.optimizer_updates = bundle['step']
        previous = bundle['config']
        fixed = ('stage', 'model', 'rank', 'seed', 'lr', 'context', 'prompt', 'response', 'group',
                 'effective_groups', 'clip', 'beta1', 'beta2', 'eps', 'sft_batch', 'warmup_ratio',
                 'max_turn_tokens', 'max_turns', 'max_tools', 'temperature', 'top_p',
                 'tool_tokens','tool_timeout','max_candidate_groups',
                 'min_reward_std','advantage_eps','concurrency','grad_clip','data','tokenizer','corpus')
        if any(previous[k] != getattr(config, k) for k in fixed):
            raise ValueError('Resume changes an algorithm parameter; use a separate weights-only run')
        if previous['steps']!=config.steps:
            verify_step_extension(config,bundle,log)
        if previous['sampling_rounds'] != config.sampling_rounds:
            from mini.pytrio_backend.pending import verify_migration
            verify_migration(config,bundle,log)
        from mini.pytrio_backend.config import Config
        defaults = Config()
        for name in ('lr_schedule','min_lr_ratio','curriculum','validation_batch','reload_count',
                     'reward_version','format_reward_weight','f1_reward_weight',
                     'advantage_std_floor','dynamic_sampling_mode'):
            if previous.get(name,getattr(defaults,name)) != getattr(config,name):
                raise ValueError('Resume changes curriculum/schedule/validation configuration')
        from mini.curation_utils import file_hash
        if bundle.get('data_manifest_sha256',file_hash(Path(config.data)/'sft_manifest.json')) != file_hash(Path(config.data)/'sft_manifest.json'):
            raise ValueError('Resume dataset differs from the saved snapshot')
        if config.resume_load_mode=='explicit':
            # Some actors reject checkpoint_uri during lazy TRAINER_INIT_RUN.
            # Initialize the matching LoRA shape, then restore ALL state through
            # the public load API before any rollout or optimizer submission.
            client = service_client.create_lora_training_client(config.model,rank=config.rank,seed=config.seed,
                train_attn=True,train_mlp=True,train_unembed=False)
            client.load_state_with_optimizer(bundle['train_state']).result(timeout=config.request_timeout)
            restore_verified = True
            log.event('optimizer_state_loaded',checkpoint_step=bundle['step'],train_state=bundle['train_state'],
                      method='load_state_with_optimizer',optimizer_restored=True)
        else:
            client = service_client.create_training_client_from_state_with_optimizer(bundle['train_state'])
    elif config.initial_checkpoint:
        path = config.initial_checkpoint
        if Path(path).is_file():
            parent = read_bundle(path)
            if parent['config']['model'] != config.model or parent['config']['rank'] != config.rank:
                raise ValueError('SFT checkpoint model/rank mismatch')
            path = parent['train_state']
        client = service_client.create_training_client_from_state(path)
    else:
        client = service_client.create_lora_training_client(config.model, rank=config.rank, seed=config.seed,
                                                           train_attn=True, train_mlp=True, train_unembed=False)
    if client.model_id != config.model:
        raise ValueError('Restored training run has a different model')
    lora = client.lora
    if lora.rank != config.rank or not lora.train_attn or not lora.train_mlp or lora.train_unembed:
        raise ValueError('Restored LoRA target/rank mismatch')
    log.event('trainer_created', training_run_id=client.training_run_cuid,
              model=client.model_id, lora=asdict(lora) if is_dataclass(lora) else lora.model_dump(mode='json'),
              optimizer_restore_requested=bool(bundle),optimizer_restored=restore_verified,
              initialization='ready' if restore_verified else 'pending',tokenizer_binding='verified_local')
    return bind_tokenizer(client, tokenizer)


def adam(config, step):
    warmup = max(1, math.ceil(config.steps * config.warmup_ratio)) if config.stage == 'sft' else 0
    lr = config.lr * min(1., step/warmup) if warmup else config.lr
    if config.stage == 'sft' and config.lr_schedule == 'cosine' and step > warmup:
        progress = min(1.,(step-warmup)/max(1,config.steps-warmup))
        lr = config.lr * (config.min_lr_ratio+(1-config.min_lr_ratio)*(1+math.cos(math.pi*progress))/2)
    return trio.AdamParams(learning_rate=lr, beta1=config.beta1, beta2=config.beta2,
                           eps=config.eps, weight_decay=0., grad_clip_norm=config.grad_clip)


def save(client, config, log, budget, cursor, step, **extra):
    from mini.curation_utils import file_hash
    name = f'{log.out.name}-{config.stage}-{step}-{time.time_ns()}'
    log.event('checkpoint_start', step=step)
    state = client.save_state(name=name+'-state').result(timeout=config.request_timeout)
    weights = client.save_weights_for_sampler(name=name+'-sampler').result(timeout=config.request_timeout)
    marker = log.out / 'optimizer_status.json'
    safe = not marker.exists() or json.loads(marker.read_text())['status'] == 'completed'
    bundle = {'step': step, 'config': config.to_dict(), 'train_state': state.path, 'resume_safe':safe,
              'data_manifest_sha256':file_hash(Path(config.data)/'sft_manifest.json'),
              'sampler_weights': weights.path, 'cursor': cursor.state(),
              'budget': budget.snapshot(), 'training_run_id': client.training_run_cuid, **extra}
    from mini.pytrio_backend.pending import pending_path
    pending = pending_path(log,step)
    if pending is not None and pending.exists():
        bundle['pending_sampling']={'path':str(pending),'sha256':file_hash(pending),'policy_version':step}
    json_write(log.out / f'checkpoint-{step}.json', bundle)
    json_write(log.out / 'checkpoint.json', bundle)
    log.event('checkpoint_saved', step=step, train_state=state.path, sampler_weights=weights.path)
    return bundle


def optimizer_step(client, params, step, config, log):
    # A timeout can happen after the server commits. Never label that checkpoint
    # as an exact resumable step or replay an uncertain optimizer request.
    json_write(log.out / 'optimizer_status.json', {'step':step,'status':'submitted'})
    log.event('optimizer_submitted',step=step)
    result = client.optim_step(params).result(timeout=config.request_timeout)
    log.optimizer_updates = step
    json_write(log.out / 'optimizer_status.json', {'step':step,'status':'completed'})
    log.event('optimizer_completed',step=step,metrics=result.metrics)
    return result
