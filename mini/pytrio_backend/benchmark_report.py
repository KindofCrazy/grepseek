"""Read-only benchmark reports from persisted PyTRIO trajectories; no model calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from mini.benchmark_metrics import markdown, report
from mini.common import read_jsonl
from mini.curation_utils import file_hash, json_file
from mini.pytrio_backend.eval_history import history


def as_record(trajectory):
    messages = trajectory['messages']
    answers = [answer for message in messages if message['role'] == 'assistant'
               for answer in re.findall(r'<answer>\s*(.*?)\s*</answer>', message['content'], re.S)]
    finish = trajectory['finish_reason']
    turns = []
    assistant_index = 0
    for message in messages:
        entry = dict(message)
        if message['role'] == 'assistant':
            entry['finish_reason'] = trajectory['turns'][assistant_index]['stop_reason']
            assistant_index += 1
        turns.append(entry)
    return {
        'id': trajectory['question_id'], 'source': trajectory['source'],
        'question': trajectory['question'], 'gold_answers': trajectory['gold'],
        'prediction': answers[-1].strip() if answers else '',
        'turns': turns, 'n_tool_calls': trajectory['tool_count'],
        'finish_reason': 'max_turns' if finish == 'turn_budget' else finish,
        'response_budget_exhausted': finish == 'context_budget',
        'observation_budget_overflow': finish == 'observation_budget_overflow',
        'generated_completion_tokens': sum(len(t['tokens']) for t in trajectory['turns']),
        'context_growth_tokens': len(trajectory['tokens']) - trajectory['initial_length'],
        'assistant_content_tokens': None,
    }


def build(trajectories, scheduled):
    expected = {row['id']: row for row in scheduled}
    if not scheduled or len(expected) != len(scheduled):
        raise ValueError('A nonempty schedule with unique IDs is required')
    observed = set()
    records = []
    for trajectory in trajectories:
        qid = trajectory['question_id']
        if qid in observed:
            raise ValueError('Duplicate evaluation trajectory; cannot silently reweight a question')
        if qid not in expected:
            raise ValueError('Unscheduled question in evaluation')
        row = expected[qid]
        if (trajectory['question'] != row['question'] or trajectory['source'] != row['source'] or
                trajectory['gold'] != row['golden_answers']):
            raise ValueError('Recorded question/source/golds differ from frozen schedule')
        observed.add(qid)
        records.append(as_record(trajectory))
    if observed != set(expected):
        raise ValueError(f'Incomplete benchmark: {len(observed)}/{len(expected)}; not a final report')
    return report(records), records


def audit_summary(events,kinds):
    audits=[e for e in events if e['kind'] in kinds]
    mismatches=sum(not e['equal'] for e in audits)
    return {'compared_queries':len(audits),'mismatched_queries':mismatches,
            'mismatch_rate':mismatches/len(audits) if audits else None}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run / 'result.json').read_text())
    config = json.loads((args.run / 'config.json').read_text())
    if result['status'] != 'completed' or config['stage'] != 'eval':
        raise ValueError('A completed eval run is required')
    chain = history(args.run)
    if args.prior and (len(chain)<2 or chain[-2][0] != args.prior.resolve()):
        raise ValueError('--prior does not match the saved continuation chain')
    for path,_ in chain:
        link = path/'continuation.json'
        if link.exists() and json.loads(link.read_text())['schedule_sha256'] != file_hash(args.schedule):
            raise ValueError('Continuation schedule changed')
    events = [event for _,segment in chain for event in segment]
    prior_duration = sum(next((e['budget']['wall_s'] for e in reversed(segment)
                              if e['kind']=='request_finished'),0.) for _,segment in chain[:-1])
    prior_hashes = [{name:file_hash(path/name) for name in
                    ('events.jsonl','config.json','provenance.json')} for path,_ in chain[:-1]]
    trajectories = [event['trajectory'] for event in events if event['kind'] == 'trajectory']
    measurements, records = build(trajectories, read_jsonl(args.schedule))
    from mini.pytrio_backend.rollout import Trajectory
    from mini.pytrio_backend.agent_metrics import trajectory as agent_measure, aggregate as agent_aggregate
    reconstructed = []
    for raw in trajectories:
        t = Trajectory(**raw)
        t.metrics.update(agent_measure(t,t.metrics.get('trajectory_latency_s')))
        reconstructed.append(t)
    agent_summary = agent_aggregate(reconstructed)
    fee_estimate = result['budget']['estimated_uncached_cny']
    if (args.run/'reuse.json').exists():
        fee_estimate += json.loads((args.run/'reuse.json').read_text()).get('prior_fee_estimate_cny',0.)
    successes = agent_summary['strict_success_count']
    agent_summary['estimated_cny_per_strict_success_including_failures'] = fee_estimate/successes if successes else None
    agent_summary['cost_definition'] = 'all eval attempts, including reused prior baseline consumption; uncached estimate, not invoice'
    execution = [event for event in events if event['kind'] == 'pace_execution']
    summary = {
        'status': 'completed', 'scheduled_questions': len(records),
        'model': config['model'], 'sampler_weights': result.get('sampler_weights'),
        'temperature': config['temperature'], 'seed': config['seed'],
        'config': config, 'budget': result['budget'],
        'observed_active_wall_s': prior_duration + result['budget']['wall_s'],
        'prior_source_sha256': prior_hashes,
        'finish_reasons': dict(Counter(t['finish_reason'] for t in trajectories)),
        'metrics': measurements, 'pace_events': len(execution),
        'agent_metrics': agent_summary,
        'estimated_complete_evaluation_cny':fee_estimate,
        'pace_native_audit':audit_summary(events,{'pace_native_audit'}),
        'sharded_native_audit':audit_summary(events,{'sharded_native_audit'}),
        'accelerated_native_audit':audit_summary(events,{'pace_native_audit','sharded_native_audit'}),
        'pace_outcomes': dict(Counter(str(e.get('served', 'unknown'))
                                     for e in execution)),
        'source_sha256': {name: file_hash(args.run / name) for name in
                          ('events.jsonl', 'config.json', 'provenance.json', 'result.json')},
        'schedule_sha256': file_hash(args.schedule),
        'quality_improvement_claimed': False,
        'limitations': [
            f'Frozen subset: {len(records)} questions; not the complete datasets or paper benchmark.',
            'Sampling temperature zero is not a cross-server bitwise determinism guarantee.',
            'Fees are uncached estimates, not an invoice; CPU/cloud/network fees excluded.',
            'Source question isolation does not certify absence from base-model pretraining.',
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json_file(args.out / 'summary.json', summary)
    # Compact per-question outcomes; full text, commands and raw token IDs stay in the run.
    from mini.benchmark_metrics import trajectory as measure
    from mini.common import write_jsonl
    write_jsonl(args.out / 'questions.jsonl', [
        {'id': r['id'], 'source': r['source'], 'finish_reason': t['finish_reason'], **measure(r)}
        for r, t in zip(records, trajectories)])
    content = markdown({'base' if not config['initial_checkpoint'] else 'checkpoint': measurements})
    content += (f"\n\nCompleted: {len(records)} questions. Model: {config['model']}. "
                f"Temperature: {config['temperature']}. Context: {config['context']}; "
                f"response including observations: {config['response']}.\n\n"
                f"Uncached fee estimate: CNY {result['budget']['estimated_uncached_cny']:.4f}; "
                f"observed active wall time across segments: {summary['observed_active_wall_s']:.1f} seconds. "
                'Interruption downtime excluded; unrecorded tail time unavailable. Actual invoice unavailable.\n\n'
                f'This is a frozen {len(records)}-question subset. Use independent dev data to select checkpoints.\n')
    (args.out / 'README.md').write_text(content, encoding='utf-8')
    print(json.dumps({'questions': len(records), 'metrics': measurements['overall'],
                      'budget': result['budget']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
