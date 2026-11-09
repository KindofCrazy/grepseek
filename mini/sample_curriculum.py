"""Generate reproducible SFT candidate queues; no model calls or semantic approval."""
import argparse
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path

from mini.common import ROOT, digest, question_key, read_jsonl, write_jsonl
from mini.curation_utils import NearIndex, file_hash, json_file, records

SOURCES = ('nq', 'hotpotqa')
LENGTH_WEIGHTS = (.60, .25, .12, .03)


def length_bin(row):
    total = row['tokens']['total_sequence_tokens']
    return next((i for i, limit in enumerate((4096, 6144, 8192, 12288)) if total <= limit), 4)


def key(seed, namespace, row):
    return digest([seed, namespace, row['id']])


def coverage_order(rows, seed, namespace):
    """Soft length/command coverage; answer labels never participate in ranking."""
    groups = defaultdict(list)
    for row in rows:
        group = (length_bin(row), tuple(sorted(set(row.get('command_categories', [])))))
        groups[group].append(row)
    denominators = Counter()
    for group, values in groups.items():
        denominators[group[0]] += math.sqrt(len(values))
    weights = {group: LENGTH_WEIGHTS[group[0]] * math.sqrt(len(values)) / denominators[group[0]]
               for group, values in groups.items()}
    buckets = {group: deque(sorted(values, key=lambda r: key(seed, namespace, r)))
               for group, values in groups.items()}
    positions = Counter()
    result = []
    while buckets:
        group = min(buckets, key=lambda g: ((positions[g] + 1) / weights[g], digest([seed, namespace, g])))
        result.append(buckets[group].popleft())
        positions[group] += 1
        if not buckets[group]:
            del buckets[group]
    return result


def allocate(rows, calls, per_source):
    """Exact two-source and call-count margins, bounded by observed cell capacity."""
    if 2 * per_source != sum(calls.values()):
        raise ValueError('Source and call quotas disagree')
    capacity = Counter((r['source'], r['tool_calls']) for r in rows)
    lower, upper, ideal, nq = {}, {}, {}, {}
    for count, target in calls.items():
        n, h = capacity['nq', count], capacity['hotpotqa', count]
        lower[count], upper[count] = max(0, target - h), min(target, n)
        if lower[count] > upper[count]:
            raise ValueError(f'Insufficient unique candidates for {count} calls: need {target}, have {n+h}')
        ideal[count] = target * n / (n + h) if n + h else 0.
        nq[count] = min(upper[count], max(lower[count], math.floor(ideal[count])))
    if not sum(lower.values()) <= per_source <= sum(upper.values()):
        raise ValueError('Insufficient source/call joint capacity; no duplicate filling')
    while sum(nq.values()) != per_source:
        direction = 1 if sum(nq.values()) < per_source else -1
        candidates = [c for c in calls if lower[c] <= nq[c] + direction <= upper[c]]
        count = min(candidates, key=lambda c: (
            ((nq[c] + direction - ideal[c]) ** 2 - (nq[c] - ideal[c]) ** 2) / max(1, calls[c]), c))
        nq[count] += direction
    return {(source, count): nq[count] if source == 'nq' else calls[count] - nq[count]
            for source in SOURCES for count in calls}


def select_stage(rows, calls, per_source, seed, namespace):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Candidate IDs must be unique')
    targets = allocate(rows, calls, per_source)
    chosen = []
    for (source, count), target in sorted(targets.items()):
        cell = [r for r in rows if r['source'] == source and r['tool_calls'] == count]
        chosen.extend(coverage_order(cell, seed, namespace)[:target])
    return sorted(chosen, key=lambda r: key(seed, namespace + ':mix', r))


def summarize(rows):
    return {'count': len(rows), 'sources': dict(Counter(r['source'] for r in rows)),
            'tool_calls': dict(sorted(Counter(r['tool_calls'] for r in rows).items())),
            'length_bins': dict(Counter(str(length_bin(r)) for r in rows)),
            'answer_match_labels': dict(Counter(r.get('gold_match_status', 'unavailable') for r in rows))}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--answers', type=Path, default=Path('mini/runs/sft-answer-audit-20261107/answers.jsonl'))
    parser.add_argument('--tokenizer', type=Path, default=Path('mini/.cache/qwen35-4b-tokenizer'))
    args = parser.parse_args()
    if args.out.exists() or args.report.exists():
        raise ValueError('Use fresh output and report directories')
    audit_path = Path('mini/curated/sft_audit.jsonl')
    old_val_path = Path('mini/curated/sft_validation.jsonl')
    raw_path = Path('mini/.cache/sft-train.parquet')
    source_manifest = json.loads(Path('mini/curated/sft_manifest.json').read_text())
    expected_raw = next(value for name, value in source_manifest['input_files_sha256'].items()
                        if name.replace('\\', '/').endswith('/sft-train.parquet'))
    if file_hash(raw_path) != expected_raw:
        raise ValueError('Original parquet changed')
    audits = read_jsonl(audit_path)
    all_by_id = {r['id']: r for r in audits}
    if len(all_by_id) != len(audits):
        raise ValueError('Duplicate original IDs')
    pool = [r for r in audits if r['eligible']]
    if len(pool) != 9973:
        raise ValueError('This extraction requires the discussed unchanged 9973-row audit')
    # Labels are attached for reporting; they are never eligibility or ranking inputs.
    answers = {r['id']: r for r in read_jsonl(args.answers)}
    for row in pool:
        row['gold_match_status'] = answers.get(row['id'], {}).get('gold_match_status', 'unavailable')
    heldout_paths = [Path('mini/curated/rl_dev.jsonl'), Path('mini/curated/rl_test.jsonl'),
                     Path('mini/benchmark30/test.jsonl')]
    heldout = [r for path in heldout_paths for r in read_jsonl(path)]
    heldout_keys = {question_key(r['question']) for r in heldout}
    rl_path = Path('mini/curated/rl_candidates.jsonl')
    rl_keys = {question_key(r['question']) for r in read_jsonl(rl_path)}
    old_val = [all_by_id[r['id']] for r in read_jsonl(old_val_path)]
    old_ids = {r['id'] for r in old_val}
    if len(old_ids) != 100 or not old_ids <= {r['id'] for r in pool}:
        raise ValueError('Original frozen 100 validation members changed')
    if any(question_key(r['question']) in heldout_keys | rl_keys for r in old_val):
        raise ValueError('Original SFT validation overlaps other heldout or RL train')
    initial = [r for r in pool if r['source'] in SOURCES and length_bin(r) < 4
               and r['id'] not in old_ids and question_key(r['question']) not in heldout_keys]
    validation = list(old_val)
    for source in SOURCES:
        available = [r for r in initial if r['source'] == source and question_key(r['question']) not in rl_keys]
        needed = 100 - sum(r['source'] == source for r in validation)
        if needed < 0 or len(available) < needed:
            raise ValueError('Cannot extend validation to 100/source')
        validation.extend(sorted(available, key=lambda r: key(args.seed, 'validation-extension', r))[:needed])
    validation_ids = {r['id'] for r in validation}
    fixed = heldout + validation
    fixed_keys = {question_key(r['question']) for r in fixed}
    near = NearIndex(.90)
    for row in fixed:
        near.add(row['question'], {'id': row['id'], 'role': 'heldout'})
    available, blocked = [], []
    for row in pool:
        reason = None
        if row['id'] in validation_ids:
            continue
        if row['source'] not in SOURCES:
            reason = 'source_unresolved'
        elif length_bin(row) == 4:
            reason = 'over_context_budget'
        elif question_key(row['question']) in fixed_keys:
            reason = 'exact_heldout_overlap'
        matches = near.query(row['question']) if reason is None else []
        if matches:
            reason = 'lexical_heldout_near_match_pending'
        if reason:
            blocked.append({'id': row['id'], 'reason': reason, 'near_matches': matches})
        else:
            available.append(row)
    if len({question_key(r['question']) for r in available}) != len(available):
        raise ValueError('Duplicate normalized questions remain in pool')
    stage_a = select_stage(available, {1: 700, 2: 1200, 3: 100}, 1000, args.seed, 'stage-a')
    a_ids = {r['id'] for r in stage_a}
    remainder = [r for r in available if r['id'] not in a_ids]
    stage_b = select_stage(remainder, {1: 500, 2: 2000, 3: 500}, 1500, args.seed, 'stage-b')
    b_ids = {r['id'] for r in stage_b}
    reserve = sorted((r for r in remainder if r['id'] not in b_ids),
                     key=lambda r: key(args.seed, 'reserve', r))
    groups = {'stage_a_2000': stage_a, 'stage_b_3000': stage_b, 'validation_200': validation}
    wanted = {r['id'] for rows in groups.values() for r in rows}
    messages = {}
    for index, raw in enumerate(records(raw_path)):
        if raw['id'] != audits[index]['id']:
            raise ValueError('Parquet and audit order/IDs changed')
        if raw['id'] in wanted:
            messages[raw['id']] = raw['messages']
    if set(messages) != wanted:
        raise ValueError('Missing original messages for selected candidates')
    from transformers import AutoTokenizer
    from mini.retokenize import measure_sft, verify_tokenizer
    provenance_path = Path('mini/qwen35_4b_source.json')
    provenance = json.loads(provenance_path.read_text())
    verify_tokenizer(args.tokenizer, provenance)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    token_audit = []
    for number, qid in enumerate(sorted(wanted), 1):
        measured, tokens, mask = measure_sft(tokenizer, {'id': qid, 'messages': messages[qid]}, 12288)
        row = all_by_id[qid]
        token_audit.append({'id': qid, 'tokens': measured, 'matches_prior_token_counts': measured == row['tokens'],
                            'shifted_trainable_tokens': sum(mask[1:])})
        if number % 500 == 0:
            print(f'Validated actual 4B token/mask: {number}/{len(wanted)}', flush=True)
    args.out.mkdir(parents=True)
    args.report.mkdir(parents=True)
    for name, rows in groups.items():
        write_jsonl(args.out / f'{name}_candidates.jsonl', [
            {'id': row['id'], 'source': row['source'], 'question': row['question'],
             'tool_calls': row['tool_calls'], 'candidate_rank': rank,
             'gold_match_status': row.get('gold_match_status', 'unavailable'),
             'quality_status': 'candidate_only_not_semantically_approved',
             'messages': messages[row['id']]} for rank, row in enumerate(rows, 1)])
    write_jsonl(args.out / 'combined_5000_candidate_ids.jsonl', [
        {'id': r['id'], 'stage': stage} for stage, rows in [('a', stage_a), ('b', stage_b)] for r in rows])
    write_jsonl(args.out / 'reserve_queue.jsonl', [
        {'id': r['id'], 'source': r['source'], 'tool_calls': r['tool_calls'],
         'gold_match_status': r.get('gold_match_status', 'unavailable'),
         'length_bin': length_bin(r), 'candidate_rank': i} for i, r in enumerate(reserve, 1)])
    write_jsonl(args.out / 'blocked_candidates.jsonl', blocked)
    write_jsonl(args.out / 'token_audit.jsonl', token_audit)
    report = {'status': 'candidate_queues_generated', 'training_ready': False,
              'seed': args.seed, 'original_pool': summarize(pool), 'available_after_reservation': summarize(available),
              'groups': {name: summarize(rows) for name, rows in groups.items()}, 'reserve': summarize(reserve),
              'old_validation_preserved': 100, 'new_validation_added': 100,
              'old_27_quarantines_unchanged': True, 'em_filter_applied': False,
              'blocked_reasons': dict(Counter(r['reason'] for r in blocked)),
              'stages_disjoint': not bool(a_ids & b_ids), 'nested_2k_in_5k': a_ids <= a_ids | b_ids,
              'sft_rl_train_question_overlap': {
                  stage: sum(question_key(r['question']) in rl_keys for r in rows)
                  for stage, rows in [('a', stage_a), ('b', stage_b)]},
              'train_heldout_exact_overlap': 0, 'actual_4b_token_mask_checks': len(token_audit),
              'token_counts_changed_from_old_audit': sum(not r['matches_prior_token_counts'] for r in token_audit),
              'pending': ['full tool replay', 'stratified semantic review and substantive answer/evidence issues'],
              'model_api_requests': 0, 'tool_commands_executed': 0}
    json_file(args.report / 'summary.json', report)
    inputs = [audit_path, old_val_path, raw_path, args.answers, provenance_path, rl_path, *heldout_paths]
    manifest = {'status': 'candidates_only', 'training_ready': False, 'model': provenance['model'],
                'model_provenance': provenance, 'selection': report,
                'input_sha256': {str(p): file_hash(p) for p in inputs},
                'code_sha256': {name: file_hash(ROOT / name) for name in
                                ('mini/sample_curriculum.py', 'mini/curation_utils.py', 'mini/retokenize.py', 'mini/template.py')},
                'files_sha256': {str(p.relative_to(args.out)): file_hash(p) for p in args.out.iterdir() if p.is_file()}}
    json_file(args.out / 'candidate_manifest.json', manifest)
    json_file(args.report / 'candidate_manifest.json', manifest)
    lines = ['# SFT课程候选队列：2026-11-07', '',
             '从现有9973条池抽样；原27条隔离记录保持不变，没有复核或恢复。',
             'EM不作过滤或排序条件。候选保留原始messages，未改写答案或observation。', '',
             '| 阶段 | 数量 | 1次调用 | 2次调用 | 3次调用 | NQ | HotpotQA |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, rows in groups.items():
        stats = summarize(rows)
        lines.append('| ' + ' | '.join(map(str, [name, len(rows), *[stats['tool_calls'].get(c, 0) for c in (1, 2, 3)],
                                                  *[stats['sources'].get(s, 0) for s in SOURCES]])) + ' |')
    lines += ['', f'备用队列：{len(reserve)}条。新增自动阻断原因：{report["blocked_reasons"]}。',
              '已有100条SFT验证成员原样保留，再新增100条；验证与训练分离，SFT/RL训练问题允许重叠。',
              '5200条入选候选均用实际4B tokenizer检查模板、长度、shift和mask；未执行工具或模型。',
              '这些是候选队列，尚未做全量工具重放和语义审查；不是已经验收的训练快照。',
              '完整轨迹和token审计位于独立curated-curriculum-candidates目录，原冻结800条快照未变。']
    (args.report / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
