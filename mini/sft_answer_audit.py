"""Read-only QA answer matching for planning; not evidence or trajectory certification."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from inference.scoring import normalize_answer, score
from mini.common import question_key, read_jsonl
from mini.curation_utils import file_hash, json_file, records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('mini/.cache'))
    parser.add_argument('--audit', type=Path, default=Path('mini/curated/sft_audit.jsonl'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite a previous audit')
    audits = read_jsonl(args.audit)
    keys = {question_key(r['question']) for r in audits}
    lookup = defaultdict(list)
    qa_paths = []
    for source in ('nq', 'hotpotqa'):
        for split in ('train', 'dev'):
            path = args.input_dir / f'{source}_{split}.jsonl'
            qa_paths.append(path)
            for row in records(path):
                key = question_key(row['question'])
                if key in keys:
                    lookup[key].append({'id': f'{source}:{split}:{row["id"]}', 'source': source,
                                        'split': split, 'golden_answers': row['golden_answers']})
    source_path = args.input_dir / 'sft-train.parquet'
    counts = Counter()
    eligible = Counter()
    by_source = defaultdict(Counter)
    total = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        for index, raw in enumerate(records(source_path)):
            audit = audits[index]
            if raw['id'] != audit['id'] or index != audit['row_index']:
                raise ValueError('Original rows no longer align with frozen structure audit')
            matches = lookup[question_key(audit['question'])]
            final = raw['messages'][-1]
            answers = re.findall(r'<answer>\s*(.*?)\s*</answer>', final['content'], re.S)
            answer = answers[-1].strip() if answers else ''
            results = [score(answer, q['golden_answers']) for q in matches]
            gold_sets = {tuple(sorted({normalize_answer(g) for g in q['golden_answers']})) for q in matches}
            status = ('gold_unavailable' if not results else
                      'exact_match' if any(r['em'] for r in results) else
                      'partial_match_review' if max(r['f1'] for r in results) > 0 else 'zero_overlap_review')
            counts[status] += 1
            if audit['eligible']:
                eligible[status] += 1
                by_source[audit['source']][status] += 1
            if len(gold_sets) > 1:
                counts['multiple_distinct_gold_sets_review'] += 1
            row = {'id': raw['id'], 'source': audit['source'], 'structure_eligible': audit['eligible'],
                   'answer': answer, 'answer_tags_in_final_message': len(answers),
                   'gold_match_status': status, 'qa_matches': matches,
                   'gold_sets_differ_review': len(gold_sets) > 1,
                   'max_f1': max((r['f1'] for r in results), default=None),
                   'evidence_correctness': 'unknown', 'causal_order_correctness': 'unknown'}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            total += 1
    if total != len(audits):
        raise ValueError('Source/audit row counts differ')
    summary = {'scope': 'preliminary_answer_string_matching_only', 'raw_count': total,
               'all_rows': dict(counts), 'structure_eligible_rows': dict(eligible),
               'structure_eligible_by_source': {s: dict(c) for s, c in by_source.items()},
               'input_sha256': {str(p): file_hash(p) for p in [source_path, args.audit, *qa_paths]},
               'output_sha256': file_hash(args.out), 'paid_calls': 0, 'tools_replayed': 0,
               'training_selection_created': False,
               'limits': ['Exact answer match does not establish supporting evidence or causal reasoning.',
                          'Non-exact answers need review; valid alternate aliases may be missing.',
                          'Earlier structure eligibility is not final quality acceptance or heldout isolation.']}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    json_file(args.summary, summary)
    print(json.dumps({k: summary[k] for k in ('raw_count', 'structure_eligible_rows',
                                            'structure_eligible_by_source', 'paid_calls')}))


if __name__ == '__main__':
    main()
