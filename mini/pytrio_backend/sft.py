from pathlib import Path
import time

import numpy as np
import pytrio as trio

from mini.common import read_jsonl
from mini.pytrio_backend.data import Cursor, CurriculumCursor, datum_tokens, supervised, supervised_metrics
from mini.pytrio_backend.runtime import adam, bind_tokenizer, optimizer_step, save, trainer


def forward(client, datums, config, budget, operation='validation_forward'):
    return budget.call(operation, {'train': datum_tokens(datums)},
                       lambda: client.forward(datums, 'cross_entropy').result(timeout=config.request_timeout))


def validate(client, datums, config, budget, operation='validation_forward'):
    total = count = work = 0
    for offset in range(0,len(datums),config.validation_batch):
        batch = datums[offset:offset+config.validation_batch]
        values = supervised_metrics(forward(client,batch,config,budget,operation),batch)
        count += values['supervised_tokens']
        total += values['assistant_nll']*values['supervised_tokens']
        work += values['computed_input_tokens']
    return {'assistant_nll':total/count,'supervised_tokens':count,'computed_input_tokens':work}


def run(config, tok, svc, log, budget, bundle=None):
    rows = read_jsonl(Path(config.data) / 'sft_selected.jsonl')
    validation = read_jsonl(Path(config.data) / 'sft_validation.jsonl')[:config.val_count]
    if config.curriculum:
        import json
        stages = json.loads((Path(config.data)/config.curriculum).read_text())['stages']
        if sum(len(s['ids']) for s in stages) != config.steps*config.sft_batch:
            raise ValueError('Curriculum exposure must equal steps times batch size')
        cursor = CurriculumCursor(rows,config.seed,stages,bundle['cursor'] if bundle else None)
    else:
        cursor = Cursor(rows, config.seed, bundle['cursor'] if bundle else None)
    client = trainer(svc, tok, config, log, bundle)
    completed = bundle['step'] if bundle else 0
    probe = [supervised(tok, row, config.context) for row in validation]
    before = validate(client, probe, config, budget)
    log.event('sft_validation', step=completed, split='sft_validation', metrics=before)
    log.metrics(completed,{'sft/validation_assistant_nll':before['assistant_nll']})
    values = before
    rollback = None
    try:
        for step in range(completed + 1, config.steps + 1):
            import copy
            rollback = (completed,copy.deepcopy(cursor.state()))
            examples = cursor.take(config.sft_batch)
            batch = [supervised(tok, row, config.context) for row in examples]
            started = time.monotonic()
            result = budget.call('sft_forward_backward', {'train': datum_tokens(batch)},
                                 lambda: client.forward_backward(batch, 'cross_entropy').result(timeout=config.request_timeout))
            params = adam(config, step)
            update = optimizer_step(client, params, step, config, log)
            completed = step
            metrics = supervised_metrics(result, batch)
            metrics.update(step_wall_s=time.monotonic()-started, learning_rate=params.learning_rate,
                           epoch=cursor.epoch, optimizer_metrics=update.metrics,
                           curriculum_phase=getattr(cursor,'phase','single'),
                           source_counts=dict(__import__('collections').Counter(r['source'] for r in examples)),
                           tool_count_distribution=dict(__import__('collections').Counter(r.get('tool_calls',0) for r in examples)),
                           policy_entropy=None,policy_entropy_status='full_policy_logits_not_returned',
                           gradient_norm=next((v for k,v in update.metrics.items() if 'grad' in k and 'norm' in k),None))
            log.event('sft_update', step=step, sample_ids=[r['id'] for r in examples], metrics=metrics)
            log.metrics(step, {'sft/train_assistant_nll': metrics['assistant_nll'],
                               'sft/supervised_tokens': metrics['supervised_tokens'],
                               'sft/computed_input_tokens': metrics['computed_input_tokens'],
                               'train/lr': params.learning_rate, 'time/step_s': metrics['step_wall_s'],
                               **{f'optimizer/{k}': v for k,v in update.metrics.items()}})
            print(f'SFT step {step}: assistant NLL={metrics["assistant_nll"]:.4f}', flush=True)
            phase_end = bool(config.curriculum and cursor.phase_end)
            if (config.val_every and step % config.val_every == 0) or phase_end:
                values = validate(client, probe, config, budget)
                log.event('sft_validation', step=step, split='sft_validation', metrics=values)
                log.metrics(step,{'sft/validation_assistant_nll':values['assistant_nll']})
            if step % config.save_every == 0 or phase_end:
                boundary = save(client, config, log, budget, cursor, step,
                                curriculum_phase=getattr(cursor,'phase','single'))
                if phase_end:
                    check = probe[:config.reload_count]
                    reference = validate(client,check,config,budget)
                    restored = bind_tokenizer(svc.create_training_client_from_state_with_optimizer(boundary['train_state']),tok)
                    observed = validate(restored,check,config,budget,'reload_validation_forward')
                    delta = abs(reference['assistant_nll']-observed['assistant_nll'])
                    log.event('curriculum_boundary',step=step,phase=cursor.phase,reload_nll_abs_delta=delta,
                              checkpoint=boundary,validation=values)
                    if delta > .1:
                        raise ValueError('Curriculum boundary reload mismatch')
        saved = save(client, config, log, budget, cursor, completed)
        after = values if config.curriculum else validate(client,probe,config,budget)
        # Restore through the real public checkpoint API, without taking another step.
        reloaded = bind_tokenizer(svc.create_training_client_from_state_with_optimizer(saved['train_state']), tok)
        reload_values = validate(reloaded,probe[:config.reload_count],config,budget,'reload_validation_forward')
        reference = validate(client,probe[:config.reload_count],config,budget)
        difference = abs(reference['assistant_nll'] - reload_values['assistant_nll'])
        log.event('sft_validation',step=completed,split='sft_validation',metrics=after)
        log.metrics(completed,{'sft/validation_assistant_nll':after['assistant_nll'],
                               'checkpoint/reload_nll_abs_delta':difference})
        if difference > .1:
            raise ValueError(f'Reloaded checkpoint validation NLL mismatch: {difference}')
        sampler = svc.create_sampling_client(config.model, model_path=saved['sampler_weights'])
        from mini.pytrio_backend.rollout import prompt_tokens
        initial = prompt_tokens(tok, [{'role':'system','content':rows[0]['messages'][0]['content']},
                                      rows[0]['messages'][1]])
        response = budget.sample(sampler, initial, 1, trio.SamplingParams(max_tokens=128, temperature=0., seed=config.seed))
        if response.input_tokens != len(initial) or len(response.sequences) != 1 or not response.sequences[0].tokens:
            raise ValueError('Reloaded sampler did not return aligned nonempty generation')
        log.event('reload_generation', prompt_id=rows[0]['id'], tokens=response.sequences[0].tokens,
                  text=tok.decode(response.sequences[0].tokens, skip_special_tokens=True))
        return {'status':'completed', 'optimizer_updates':completed, 'checkpoint':saved,
                'validation_before':before, 'validation_after':after, 'validation_reloaded':reload_values,
                'reload_nll_abs_delta':difference, 'sampler_generation_verified':True,
                'quality_improvement_claimed':False, 'budget':budget.snapshot()}
    except BaseException as error:
        from mini.pytrio_backend.tracking import BudgetStop
        if isinstance(error,BudgetStop) and rollback and rollback[0]==completed:
            cursor = (CurriculumCursor(rows,config.seed,stages,rollback[1]) if config.curriculum
                      else Cursor(rows,config.seed,rollback[1]))
        try:
            save(client, config, log, budget, cursor, completed, interrupted=True)
        except Exception as error:
            log.event('emergency_checkpoint_failed', error_type=type(error).__name__)
        raise
