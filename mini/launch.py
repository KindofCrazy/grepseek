"""Construct official trainer commands. Default is bounded to two real updates."""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from mini.common import ROOT, check_batch


def env(name, default):
    return os.environ.get(name, str(default))


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=['sft', 'rl'])
    p.add_argument('--model', default=env('MODEL_PATH', 'Qwen/Qwen3.5-2B'))
    for name, default in [('gpus', 1), ('sp', 1), ('tp', 1), ('steps', 2), ('prompt-batch', 4), ('ppo-mini-batch', 4), ('group', 4), ('micro', 1), ('prompt-length', 1024), ('response-length', 8192), ('save-freq', 10), ('eval-freq', 10)]:
        p.add_argument('--' + name, type=int, default=int(env(name.upper().replace('-', '_'), default)))
    p.add_argument('--sft-length', type=int, default=int(env('SFT_LENGTH', 9216)))
    p.add_argument('--lr', type=float, default=float(env('LR', 5e-6)))
    p.add_argument('--memory-utilization', type=float, default=float(env('MEMORY_UTILIZATION', .45)))
    p.add_argument('--data-dir', default=env('DATA_DIR', 'mini/data'))
    p.add_argument('--sft-train-file', default=env('SFT_TRAIN_FILE', ''))
    p.add_argument('--sft-val-file', default=env('SFT_VAL_FILE', ''))
    p.add_argument('--rl-train-file', default=env('RL_TRAIN_FILE', ''))
    p.add_argument('--rl-dev-file', default=env('RL_DEV_FILE', ''))
    p.add_argument('--corpus', default=env('CORPUS_DIR', 'data/wiki_18_corpus'))
    p.add_argument('--output', default=env('OUTPUT_DIR', 'mini/runs'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--compose-only', action='store_true')
    return p


def build(a, extra):
    check_batch(a.prompt_batch, a.ppo_mini_batch, a.group, a.gpus, a.sp, a.micro, a.tp)
    if a.steps < 1 or min(a.prompt_length, a.response_length, a.sft_length) < 1:
        raise ValueError('steps and lengths must be positive')
    if a.stage == 'sft' and (a.prompt_batch % (a.gpus // a.sp) or (a.prompt_batch // (a.gpus // a.sp)) % a.micro):
        raise ValueError('SFT global batch must divide DP/micro batch')
    total = a.prompt_length + a.response_length
    data = Path(a.data_dir).resolve()
    out = Path(a.output).resolve() / a.stage
    environ = os.environ.copy()
    environ.update(GREPSEEK_ROOT=str(ROOT), GREPSEEK_MODEL_PATH=a.model,
                   GREPSEEK_TRAIN_FILES=str(Path(a.rl_train_file).resolve() if a.rl_train_file else data / 'train.jsonl'), GREPSEEK_VAL_FILES=str(Path(a.rl_dev_file).resolve() if a.rl_dev_file else data / 'dev.jsonl'),
                   GREPSEEK_OUTPUT_DIR=str(out), VERL_TOOL_CORPUS_ROOT=str(Path(a.corpus).resolve()),
                   MINI_OBSERVE_DIR=str(out / 'observations'))
    environ['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'mini' / 'observe'), str(ROOT), str(ROOT / 'rl'), str(ROOT / 'verl'), environ.get('PYTHONPATH', '')])
    # RL vLLM memory pool cannot use expandable_segments.
    if a.stage == 'rl' and 'expandable_segments' in environ.get('PYTORCH_CUDA_ALLOC_CONF', ''):
        raise ValueError('unset PYTORCH_CUDA_ALLOC_CONF expandable_segments for RL')
    if a.stage == 'sft':
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={a.gpus}', '-m', 'verl.trainer.sft_trainer']
        kv = {'data.train_files': str(Path(a.sft_train_file).resolve() if a.sft_train_file else data / 'sft.parquet'), 'data.val_files': str(Path(a.sft_val_file).resolve()) if a.sft_val_file else 'null', 'data.train_batch_size': a.prompt_batch,
              'data.micro_batch_size_per_gpu': a.micro, 'data.max_length': a.sft_length, 'data.max_token_len_per_gpu': a.sft_length,
              'data.messages_key': 'messages', 'data.truncation': 'error', 'data.ignore_input_ids_mismatch': 'false',
              'data.num_workers': 0, 'model.path': a.model, 'model.use_remove_padding': 'true',
              'model.enable_gradient_checkpointing': 'true', 'model.enable_activation_offload': 'true',
              'engine.strategy': 'fsdp', 'engine.dtype': 'bfloat16', 'engine.param_offload': 'true', 'engine.optimizer_offload': 'true',
              'engine.ulysses_sequence_parallel_size': a.sp,
              '+engine.wrap_policy.transformer_layer_cls_to_wrap': '[Qwen3_5DecoderLayer]',
              'data.custom_cls.path': str(ROOT / 'mini/sft_dataset.py'), 'data.custom_cls.name': 'Qwen35MiniSFTDataset',
              '+data.apply_chat_template_kwargs.enable_thinking': 'true',
              'optim.lr': a.lr, 'trainer.test_freq': a.eval_freq if a.sft_val_file else -1}
    else:
        cmd = [sys.executable, '-m', 'mini.rl_entry', '--config-path', str(ROOT / 'rl/grepseek/trainer/config'), '--config-name', 'grpo_trainer']
        kv = {'data.train_batch_size': a.prompt_batch, 'data.max_prompt_length': a.prompt_length, 'data.max_response_length': a.response_length,
              '+data.apply_chat_template_kwargs.enable_thinking': 'true',
              'actor_rollout_ref.actor.ppo_mini_batch_size': a.ppo_mini_batch,
              'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': a.micro,
              'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': total,
              'actor_rollout_ref.actor.fsdp_config.fsdp_size': a.gpus,
              'actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size': a.sp,
              'actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size': a.sp,
              'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': 1,
              'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu': total,
              'actor_rollout_ref.actor.optim.lr': a.lr,
              'actor_rollout_ref.rollout.n': a.group, 'actor_rollout_ref.rollout.tensor_model_parallel_size': a.tp,
              'actor_rollout_ref.rollout.prompt_length': a.prompt_length, 'actor_rollout_ref.rollout.response_length': a.response_length,
              'actor_rollout_ref.rollout.max_model_len': total, 'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': total,
              'actor_rollout_ref.rollout.max_num_seqs': 4, 'actor_rollout_ref.rollout.max_num_batched_tokens': total,
              'actor_rollout_ref.rollout.gpu_memory_utilization': a.memory_utilization,
              'trainer.test_freq': a.eval_freq, 'ray_kwargs.ray_init.num_gpus': a.gpus}
    kv.update({'trainer.n_gpus_per_node': a.gpus, 'trainer.nnodes': 1, 'trainer.total_training_steps': a.steps,
               'trainer.total_epochs': 10, 'trainer.save_freq': a.save_freq, 'trainer.resume_mode': 'disable',
               'trainer.logger': '[console]', 'trainer.default_local_dir': str(out)})
    cmd += [f'{k}={v}' for k, v in kv.items()] + extra
    if a.compose_only:
        cmd += ['--cfg', 'job', '--resolve']
    return cmd, environ, out


def main():
    a, extra = parser().parse_known_args()
    cmd, environ, out = build(a, extra)
    if a.dry_run:
        print(json.dumps({'command': cmd, 'environment': {k: v for k, v in environ.items() if k.startswith(('GREPSEEK_', 'VERL_TOOL_', 'MINI_'))}}, indent=2))
        return
    if not a.compose_only:
        if a.stage == 'rl' and a.model == 'Qwen/Qwen3.5-2B':
            raise SystemExit('RL requires --model pointing to the exported cold-start SFT checkpoint')
        if os.name == 'nt':
            raise SystemExit('GPU training requires the official Linux CUDA container; use --dry-run here')
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f'use a fresh --output directory: {out}')
        needed = Path(a.sft_train_file or (Path(a.data_dir) / 'sft.parquet')) if a.stage == 'sft' else Path(environ['GREPSEEK_TRAIN_FILES'])
        if not needed.is_file():
            raise SystemExit(f'missing {needed}; run mini.prepare_data')
        if a.stage == 'rl':
            from mini.common import read_jsonl
            qa_rows=read_jsonl(needed)
            if not qa_rows:
                raise SystemExit('RL QA input is empty; difficulty selection may still be pending frozen-SFT profiling')
            for row in qa_rows:
                if any(key in row for key in ('rollouts','raw_record','messages','output')) or not isinstance(row.get('question'),str) or not isinstance(row.get('golden_answers'),list):
                    raise SystemExit('RL input must contain QA records only; cached profiling trajectories are forbidden')
        if a.stage == 'rl' and not (Path(a.corpus) / 'wiki_corpus.jsonl').is_file():
            raise SystemExit('missing official wiki_corpus.jsonl')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'launch.json').write_text(json.dumps({'args': vars(a), 'command': cmd}, indent=2))
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=environ, check=True)


if __name__ == '__main__':
    main()
