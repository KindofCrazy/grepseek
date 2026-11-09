from dataclasses import dataclass, asdict
import math


@dataclass
class Config:
    stage: str = 'sft'
    model: str = 'Qwen/Qwen3.5-4B'
    tokenizer: str = 'mini/.cache/qwen35-4b-tokenizer'
    data: str = 'mini/curated-qwen35-4b'
    corpus: str = 'data/wiki_18_corpus'
    steps: int = 2
    stop_after_step: int = 0
    seed: int = 42
    rank: int = 32
    lr: float = 5e-5
    beta1: float = .9
    beta2: float = .95
    eps: float = 1e-8
    grad_clip: float = 1.
    sft_batch: int = 8
    warmup_ratio: float = .05
    lr_schedule: str = 'constant'
    min_lr_ratio: float = .1
    curriculum: str = ''
    validation_batch: int = 8
    reload_count: int = 16
    eval_file: str = ''
    pace_audit_every: int = 0
    context: int = 12288
    prompt: int = 1024
    response: int = 11264
    max_turn_tokens: int = 2048
    max_turns: int = 6
    max_tools: int = 5
    tool_tokens: int = 2048
    tool_timeout: float = 180.
    group: int = 4
    effective_groups: int = 4
    sampling_rounds: int = 5
    max_candidate_groups: int = 20
    dynamic_sampling_mode: str = 'bounded'
    reward_version: str = 'outcome_v1'
    format_reward_weight: float = .2
    f1_reward_weight: float = 1.
    advantage_std_floor: float = 0.
    min_reward_std: float = 1e-6
    advantage_eps: float = 1e-6
    temperature: float = 1.
    top_p: float = 1.
    clip: float = .2
    concurrency: int = 2
    question_concurrency: int = 4
    sample_concurrency: int = 4
    search_concurrency: int = 1
    pace_head_candidate_lines: int = 100000
    pace_head_materialize_bytes: int = 16777216
    pace_head_term_postings: int = 250000
    pace_head_sa_matches: int = 100000
    search_backend: str = 'pace'
    shards: str = ''
    pace_dir: str = 'data/pace-full'
    search_cache_entries: int = 256
    search_cache_bytes: int = 33554432
    save_every: int = 20
    val_every: int = 0
    val_count: int = 8
    swanlab: str = 'offline'
    max_prefill_tokens: int = 1500000
    max_sample_tokens: int = 100000
    max_train_tokens: int = 500000
    max_estimated_cny: float = 20.
    max_wall_seconds: int = 1800
    request_timeout: int = 600
    initial_checkpoint: str = ''
    resume: str = ''
    resume_migration: str = ''
    resume_load_mode: str = 'factory'
    budget_parent: str = ''

    def validate(self):
        if self.stage not in ('sft', 'rl', 'eval'):
            raise ValueError('stage must be sft, rl or eval')
        for name in ('steps', 'sft_batch', 'context', 'prompt', 'response', 'max_turn_tokens', 'max_turns',
                     'max_tools', 'tool_tokens', 'group', 'effective_groups', 'sampling_rounds',
                     'max_candidate_groups', 'concurrency', 'question_concurrency',
                     'sample_concurrency', 'search_concurrency', 'pace_head_candidate_lines',
                     'pace_head_materialize_bytes', 'pace_head_term_postings', 'pace_head_sa_matches',
                     'save_every', 'val_count',
                     'max_prefill_tokens', 'max_sample_tokens', 'max_train_tokens',
                     'max_wall_seconds', 'request_timeout'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.group < 2 or self.prompt + self.response > self.context:
            raise ValueError('group >= 2 and context >= prompt + response required')
        if self.stage == 'rl' and self.steps > 100:
            raise ValueError('RL hard ceiling is 100 successful optimizer updates')
        if self.stop_after_step < 0 or self.stop_after_step > self.steps or (self.stage != 'rl' and self.stop_after_step):
            raise ValueError('stop_after_step must be zero or an RL checkpoint boundary within steps')
        if self.max_candidate_groups < self.effective_groups or not 4 <= self.rank <= 64:
            raise ValueError('Invalid group budget or LoRA rank')
        if not 0 <= self.warmup_ratio <= 1 or not 0 < self.clip < 1 or self.lr <= 0:
            raise ValueError('Invalid optimizer/clip settings')
        if self.lr_schedule not in ('constant','cosine') or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError('Invalid learning rate schedule')
        if self.validation_batch < 1 or self.reload_count < 1:
            raise ValueError('Invalid validation batch/reload count')
        if self.pace_audit_every < 0:
            raise ValueError('Invalid PACE audit frequency')
        if self.val_every < 0 or self.max_estimated_cny <= 0 or self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError('Invalid evaluation/budget/sampling settings')
        if self.resume and self.initial_checkpoint:
            raise ValueError('Use resume OR initial_checkpoint')
        if self.resume_migration and (self.stage != 'rl' or not self.resume):
            raise ValueError('resume_migration requires an RL resume checkpoint')
        if self.resume_load_mode not in ('factory','explicit'):
            raise ValueError('Invalid resume load mode')
        if self.resume_load_mode=='explicit' and (self.stage!='rl' or not self.resume):
            raise ValueError('Explicit state loading requires an RL resume checkpoint')
        if self.dynamic_sampling_mode not in ('bounded','continuous'):
            raise ValueError('Invalid dynamic sampling mode')
        if self.reward_version not in ('outcome_v1','format_f1_v2'):
            raise ValueError('Invalid reward version')
        if (not math.isfinite(self.format_reward_weight) or not 0 <= self.format_reward_weight <= 1
            or not math.isfinite(self.f1_reward_weight) or not 0 < self.f1_reward_weight <= 1
            or not math.isfinite(self.advantage_std_floor) or not 0 <= self.advantage_std_floor <= 1):
            raise ValueError('Invalid auxiliary reward or advantage floor')
        if self.stage != 'rl' and (self.reward_version!='outcome_v1' or self.dynamic_sampling_mode!='bounded'):
            raise ValueError('Auxiliary reward and continuous sampling are RL-only')
        if self.budget_parent and (self.stage!='rl' or self.resume or not self.initial_checkpoint):
            raise ValueError('budget_parent requires a fresh RL run from an initial checkpoint')
        if self.stage == 'rl' and not (self.resume or self.initial_checkpoint):
            raise ValueError('RL requires a saved SFT checkpoint (optimizer reset) or exact resume bundle')
        if self.swanlab not in ('offline', 'online', 'cloud', 'disabled'):
            raise ValueError('Invalid SwanLab mode')
        if not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1 or self.eps <= 0 or self.grad_clip <= 0:
            raise ValueError('Invalid Adam parameters')
        if self.min_reward_std < 0 or self.advantage_eps <= 0 or self.tool_timeout <= 0:
            raise ValueError('Invalid reward tolerance or tool timeout')
        if self.search_backend not in ('sequential','sharded','pace'):
            raise ValueError('Invalid search backend')
        if self.search_backend=='sharded' and not self.shards:
            raise ValueError('Sharded search requires a shard directory')
        if self.search_backend=='pace' and not self.pace_dir:
            raise ValueError('PACE requires a verified local index directory')
        if self.search_cache_entries < 0 or self.search_cache_bytes < 0:
            raise ValueError('Cache bounds must be nonnegative')
        return self

    def to_dict(self):
        return asdict(self)
