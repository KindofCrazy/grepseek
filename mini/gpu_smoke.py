"""Runs bounded upstream training and rejects missing real CUDA update evidence."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from mini.common import read_jsonl


def verify(directory, stage, steps):
    events = [r for p in (Path(directory) / stage / 'observations').glob('*.jsonl') for r in read_jsonl(p)]
    forwards = {e['pid'] for e in events if e['event'] == 'cuda_forward'}
    updates = [e for e in events if e['event'] == 'optimizer_step' and e['cuda_available'] and e['gradient_tensors'] > 0 and e['pid'] in forwards]
    logs = [e for e in events if e['event'] == 'training_metrics']
    if len(updates) < steps or not logs or max(e['step'] for e in logs) < steps:
        raise RuntimeError('missing CUDA forward/backward/optimizer-step or requested trainer-step evidence')
    if stage == 'rl':
        groups = [e for e in events if e['event'] == 'grpo_groups']
        actor = [e for e in logs if any(k.startswith('actor/') for k in e['metrics'])]
        if len(groups) < steps or not actor:
            raise RuntimeError('missing actual agentic GRPO advantage/update metrics')
        executed = 0
        for path in (Path(directory) / stage / 'train_rollouts').glob('*.jsonl'):
            for row in read_jsonl(path):
                for text in re.findall(r'<tool_response>\s*(.*?)\s*</tool_response>', row.get('output', ''), re.S):
                    try:
                        tool = json.loads(text)
                        executed += int(tool.get('exit_code') == 0 and 'corpus.jsonl' in tool.get('command', ''))
                    except (ValueError, TypeError, AttributeError):
                        continue
        if not executed:
            raise RuntimeError('GRPO updates completed, but no successful corpus tool execution was observed; agentic smoke is incomplete')
    return {'stage': stage, 'verified_optimizer_steps': len(updates), 'verified_trainer_step': max(e['step'] for e in logs),
            'peak_allocated_bytes': max(e['max_memory_allocated_bytes'] for e in updates)}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=['sft', 'rl'])
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--verify-only', action='store_true')
    a, extra = p.parse_known_args()
    if not 2 <= a.steps <= 5:
        raise ValueError('smoke test must contain 2–5 steps')
    if not a.verify_only:
        # Device-wide sampling also sees vLLM/driver allocations outside torch's allocator.
        output = Path(a.output)
        output.mkdir(parents=True, exist_ok=True)
        memory_log = output / f'{a.stage}-gpu-memory.csv'
        if memory_log.exists():
            raise ValueError('refusing to overwrite prior GPU samples')
        with memory_log.open('w') as stream:
            sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu', '--format=csv,nounits', '-l', '1'], stdout=stream, stderr=subprocess.STDOUT)
            try:
                subprocess.run([sys.executable, '-m', 'mini.launch', a.stage, '--steps', str(a.steps), '--output', a.output, *extra], check=True)
            finally:
                sampler.terminate()
                sampler.wait(timeout=10)
            if sampler.returncode not in (0, -15, 1):
                print('Inspect nvidia-smi sampling log for platform-specific termination status', file=sys.stderr)
    result = verify(a.output, a.stage, a.steps)
    import csv
    peaks = {}
    memory_log = Path(a.output) / f'{a.stage}-gpu-memory.csv'
    if memory_log.exists():
        for row in csv.reader(memory_log.read_text().splitlines()[1:]):
            if len(row) == 4:
                try:
                    peaks[row[1].strip()] = max(peaks.get(row[1].strip(), 0), float(row[2]))
                except ValueError:
                    pass
    if not peaks:
        raise RuntimeError('missing valid device-wide nvidia-smi memory samples')
    result['device_peak_memory_used_mib'] = peaks
    (Path(a.output) / a.stage / 'smoke_verified.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
