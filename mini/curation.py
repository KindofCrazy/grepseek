"""Real-data SFT/RL curation. CPU only; never executes tools or calls an LLM."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import re
import shlex

from mini.common import ROOT, digest, question_key, reward, shell, validate_sft, write_jsonl
from mini.curation_utils import (NearIndex, artifact_hashes, file_hash, json_file, qa_records,
                                 quantiles, records, safe_output, stratified_order)

HEADERS = re.compile(r'<\|im_start\|>(system|user|assistant)\n')


def token_accounting(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, tools=[], enable_thinking=True, add_generation_prompt=False)
    spans = list(HEADERS.finditer(rendered))
    if not spans or spans[0].start() != 0:
        raise ValueError('unexpected chat template role layout')
    values = Counter()
    assembled = []
    for i, match in enumerate(spans):
        segment = rendered[match.start():spans[i + 1].start() if i + 1 < len(spans) else len(rendered)]
        ids = tokenizer.encode(segment, add_special_tokens=False)
        assembled.extend(ids)
        if match.group(1) == 'assistant':
            values['assistant_tokens'] += len(ids)
            header = tokenizer.encode('<|im_start|>assistant\n<think>\n', add_special_tokens=False)
            common = 0
            for a, b in zip(ids, header):
                if a != b:
                    break
                common += 1
            values['trainable_assistant_tokens'] += len(ids) - common
        elif segment.startswith('<|im_start|>user\n<tool_response>'):
            values['tool_observation_tokens'] += len(ids)
        else:
            values['prompt_tokens'] += len(ids)
    full = tokenizer.encode(rendered, add_special_tokens=False)
    if assembled != full:
        raise ValueError('segment tokenization does not equal the full chat template')
    values['total_sequence_tokens'] = len(full)
    values['masked_tokens'] = len(full) - values['trainable_assistant_tokens']
    return {name: values[name] for name in ('prompt_tokens', 'assistant_tokens', 'tool_observation_tokens', 'total_sequence_tokens', 'trainable_assistant_tokens', 'masked_tokens')}


def command_category(command):
    try:
        programs = [shlex.split(s)[0].split('/')[-1] for s in command.split('|')]
    except (ValueError, IndexError, AttributeError):
        return 'unparseable'
    search = sum(p in ('rg', 'grep') for p in programs)
    if search >= 2:
        return 'search_narrow_pipeline'
    if search and 'wc' in programs:
        return 'search_count'
    if search:
        return 'search_capped' if any(p in ('head', 'tail') for p in programs) else 'search_uncapped'
    return 'read_transform' if any(p in ('sed', 'awk', 'cat', 'cut') for p in programs) else 'other_allowed'


def audit_sft(raw, index, tokenizer, qa_lookup, max_tokens, reject_observation_truncation=False):
    record = raw if isinstance(raw, dict) else {}
    result = {'id': str(record.get('id', f'missing-id:{index}')), 'row_index': index,
              'original_id': record.get('id'), 'original_question': record.get('question'),
              'original_source': record.get('source'), 'original_fields': list(record),
              'question': None, 'question_provenance': 'missing', 'source': 'unknown', 'source_provenance': 'unknown',
              'answer_correctness': 'unknown', 'task_success': 'unknown',
              'messages_count': None, 'assistant_turns': 0, 'tool_calls': 0, 'tool_observations': 0,
              'tool_names': [], 'command_categories': [], 'commands': [], 'command_validation_errors': [],
              'tool_error_count': 0, 'observation_truncation_count': 0, 'repair_observed': False,
              'has_final_answer': False, 'normal_termination': False, 'format_valid': False,
              'format_reason': None, 'obvious_trajectory_truncation': False,
              'multi_hop_proxy': False, 'filter_reasons': [], 'leakage_matches': [], 'near_duplicate_matches': [],
              'tokens': None, 'eligible': False}
    reasons = result['filter_reasons']
    if '_parse_error' in record or not isinstance(raw, dict):
        reasons.append('unparseable_record')
    if record.get('id') is None:
        reasons.append('missing_original_id')
    messages = record.get('messages')
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        reasons.append('invalid_messages')
        return result
    result['messages_count'] = len(messages)
    if any(isinstance(m.get('content'), str) and any(tag in m['content'] for tag in ('<|im_start|>', '<|im_end|>')) for m in messages):
        reasons.append('embedded_chat_control_token')
    users = [m.get('content') for m in messages if m.get('role') == 'user' and isinstance(m.get('content'), str)]
    question = record.get('question')
    if isinstance(question, str) and question.strip():
        result['question_provenance'] = 'original_question_field'
    elif users:
        question = users[0].removeprefix('Query: ').strip()
        result['question_provenance'] = 'parsed_initial_user_message'
    else:
        question = None
    result['question'] = question
    if not isinstance(question, str) or not question_key(question):
        reasons.append('missing_question')
    else:
        matches = qa_lookup.get(question_key(question), [])
        sources = sorted({r['source'] for r in matches})
        result['qa_matches'] = [{'id': r['id'], 'source': r['source'], 'split': r['original_split']} for r in matches]
        if len(sources) == 1:
            result['source'] = sources[0]
            result['source_provenance'] = 'exact_normalized_question_join_to_flashrag'
        elif result['id'].startswith('nq_') and not sources:
            result['source'] = 'nq'
            result['source_provenance'] = 'explicit_nq_id_prefix_only'
        elif len(sources) > 1:
            result['source_provenance'] = 'ambiguous_question_join'
        result['leakage_matches'] = [r['id'] for r in matches if r['original_split'] == 'dev']
        if result['leakage_matches']:
            reasons.append('heldout_dev_question_overlap')
    assistants = [m for m in messages if m.get('role') == 'assistant']
    result['assistant_turns'] = len(assistants)
    if not any(isinstance(m.get('content'), str) and m['content'].strip() for m in assistants):
        reasons.append('missing_assistant_output')
    previous_error = False
    saw_new_call = False
    for m in messages:
        text = m.get('content')
        if not isinstance(text, str):
            continue
        if m.get('role') == 'assistant':
            result['tool_calls'] += text.count('<tool_call')
            for block in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S):
                try:
                    call = json.loads(block)
                    result['tool_names'].append(call.get('name') if isinstance(call, dict) else None)
                    args = call.get('arguments') if isinstance(call, dict) else None
                    cmd = args.get('command') if isinstance(args, dict) else None
                    if isinstance(cmd, str):
                        result['commands'].append(cmd)
                        result['command_categories'].append(command_category(cmd))
                        try:
                            shell._validate_pipeline(cmd)
                            saw_new_call = True
                        except shell.ToolError as exc:
                            result['command_validation_errors'].append(str(exc))
                except (ValueError, TypeError):
                    pass
        elif m.get('role') == 'tool':
            result['tool_observations'] += 1
            try:
                observation = json.loads(text)
                if not isinstance(observation, dict):
                    continue
                error = observation.get('exit_code') != 0 or bool(observation.get('timed_out'))
                result['tool_error_count'] += int(error)
                result['observation_truncation_count'] += int(bool(observation.get('truncated')))
                if previous_error and saw_new_call and not error:
                    result['repair_observed'] = True
                previous_error = bool(error)
                saw_new_call = False
            except (ValueError, TypeError):
                pass
    final = messages[-1] if messages else {}
    final_text = final.get('content', '') if isinstance(final.get('content', ''), str) else ''
    result['has_final_answer'] = final.get('role') == 'assistant' and bool(re.search(r'<answer>\s*\S.*?</answer>', final_text, re.S))
    result['normal_termination'] = result['has_final_answer'] and final_text.rstrip().endswith('</answer>')
    result['obvious_trajectory_truncation'] = not result['normal_termination'] or record.get('finish_reason') == 'length'
    if not result['has_final_answer']:
        reasons.append('no_final_answer')
    if result['obvious_trajectory_truncation']:
        reasons.append('unfinished_or_obviously_truncated')
    if reject_observation_truncation and result['observation_truncation_count']:
        reasons.append('observation_truncated_by_policy')
    try:
        validate_sft(record)
        result['format_valid'] = True
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        result['format_reason'] = str(exc)
        reasons.append('protocol_or_format_error')
    from mini.diagnostics import protocol_metrics
    static_protocol = protocol_metrics(messages)
    result['tool_schema_valid'] = static_protocol['invalid_tool_calls'] == 0
    if not result['tool_schema_valid']:
        reasons.append('tool_schema_or_static_command_error')
    try:
        result['tokens'] = token_accounting(tokenizer, messages)
        if result['tokens']['total_sequence_tokens'] > max_tokens:
            reasons.append('context_budget_exceeded')
    except Exception as exc:
        reasons.append('chat_template_or_token_accounting_error')
        result['token_error'] = f'{type(exc).__name__}: {exc}'
    result['multi_hop_proxy'] = result['tool_calls'] >= 2 and len(set(result['commands'])) >= 2
    # This proxy describes search behavior; it is NOT proof of reasoning correctness/multi-hop necessity.
    result['eligible'] = not reasons
    return result


def stratum(row):
    length = row['tokens']['total_sequence_tokens']
    return (row['source'], min(row['tool_calls'], 4), next((b for b in (4096, 6144, 8192, 12288, 16384) if length <= b), 'long'), row['multi_hop_proxy'], row['repair_observed'])


def summarize(audits, selected, validation, budget):
    counts = Counter(r for a in audits for r in a['filter_reasons'])
    available_tokens = [a['tokens'] for a in audits if a['tokens'] is not None]
    return {'raw_rows': len(audits), 'eligible_rows': sum(a['eligible'] for a in audits),
            'train_rows': len(selected), 'validation_rows': len(validation), 'context_budget': budget,
            'filter_reason_counts_nonexclusive': dict(counts),
            'answer_correctness': {'unknown': len(audits)},
            'token_statistics': {k: quantiles([t[k] for t in available_tokens]) for k in (available_tokens[0] if available_tokens else [])},
            'length_retention_all_tokenizable': {str(b): sum(t['total_sequence_tokens'] <= b for t in available_tokens) / len(available_tokens) if available_tokens else None for b in (4096, 6144, 8192, 12288, 16384)},
            'sources': dict(Counter(a['source'] for a in audits)),
            'selected_sources': dict(Counter(a['source'] for a in selected)),
            'selected_tool_calls': dict(Counter(a['tool_calls'] for a in selected)),
            'selected_length_bins':dict(Counter(str(stratum(a)[2]) for a in selected)),
            'selected_max_tokens':max((a['tokens']['total_sequence_tokens'] for a in selected),default=None),
            'selected_command_categories': dict(Counter(c for a in selected for c in set(a['command_categories']))),
            'selected_multihop_proxy': sum(a['multi_hop_proxy'] for a in selected), 'selected_repairs': sum(a['repair_observed'] for a in selected),
            'unknown_fields_policy': 'No exported gold/evidence correctness labels: answer_correctness/task_success stay unknown. Semantic equivalence remains unreviewed.',
            'sampling': 'Deterministic source/turn/length/behavior strata; commands annotate diversity, not equal-count quotas.'}


def charts(out, audits, selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'svg.hashsalt':'grepseek-mini-curation'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    lengths = [a['tokens']['total_sequence_tokens'] for a in audits if a['tokens']]
    axes[0, 0].hist(lengths, bins=40, color='#457b9d')
    axes[0, 0].set(xlabel='Total tokens (real template)', ylabel='Raw trajectories', title='Sequence length')
    for ax, values, title in [(axes[0,1], Counter(a['tool_calls'] for a in selected), 'Selected tool calls'),
                              (axes[1,0], Counter(a['source'] for a in selected), 'Selected task sources'),
                              (axes[1,1], Counter(r for a in audits for r in a['filter_reasons']), 'Filter reasons (nonexclusive)')]:
        labels, nums = zip(*sorted(values.items(), key=lambda kv: str(kv[0]))) if values else ([], [])
        ax.barh([str(k) for k in labels], nums, color='#2a9d8f')
        ax.set(title=title, xlabel='Count')
    fig.savefig(out / 'sft_curation_charts.png')
    fig.savefig(out / 'sft_curation_charts.svg', metadata={'Date':None})
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input-dir', default='mini/.cache')
    p.add_argument('--sft-file', help='Override official SFT parquet/JSONL path')
    p.add_argument('--model', default='Qwen/Qwen3.5-2B')
    p.add_argument('--model-revision', default='main')
    p.add_argument('--out', default='mini/curated')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-count', type=int, default=800)
    p.add_argument('--val-count', type=int, default=100)
    p.add_argument('--context-budget', type=int, default=12288)
    p.add_argument('--rl-pool-count', type=int, default=600)
    p.add_argument('--near-threshold', type=float, default=.9)
    p.add_argument('--reject-observation-truncation', action='store_true')
    p.add_argument('--allow-sft-rl-overlap', action='store_true')
    a = p.parse_args()
    if min(a.train_count, a.val_count, a.context_budget) < 1 or not 300 <= a.rl_pool_count <= 600 or not 0 < a.near_threshold <= 1:
        p.error('positive counts/context, RL pool 300–600, near threshold in (0,1] required')
    source = Path(a.input_dir)
    sft_path = Path(a.sft_file) if a.sft_file else source / 'sft-train.parquet'
    input_files = [sft_path, *[source / f'{s}_{k}.jsonl' for s in ('nq', 'hotpotqa') for k in ('train', 'dev')]]
    provenance = {str(f.resolve()): file_hash(f) for f in input_files}
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(a.model, revision=a.model_revision)
    except Exception as exc:
        raise SystemExit(f'Actual Qwen3.5 tokenizer unavailable: {exc}. Use --model PATH containing tokenizer.json/tokenizer_config.json/chat_template.jinja; no fallback tokenizer.')
    model_path = Path(a.model)
    tokenizer_provenance = {f.name: file_hash(f) for f in sorted(model_path.iterdir()) if f.is_file()} if model_path.is_dir() else {'model': a.model, 'revision_requested': a.model_revision, 'commit_resolved': tok.init_kwargs.get('_commit_hash')}
    out = safe_output(a.out, {'arguments': vars(a), 'input_hashes': provenance, 'tokenizer': tokenizer_provenance})
    print('Loading real QA fields and building held-out question index...', flush=True)
    qa = list(qa_records(source))
    lookup = defaultdict(list)
    heldout = NearIndex(a.near_threshold)
    for row in qa:
        if row['basic_valid']:
            lookup[question_key(row['question'])].append(row)
            if row['original_split'] == 'dev':
                heldout.add(row['question'], {'id': row['id'], 'split': 'official_dev_reserved'})
    audits, raw_by_id, reviews = [], {}, []
    seen_keys, seen_ids, seen_trajectory_fingerprints = {}, set(), {}
    near_train = NearIndex(a.near_threshold)
    trajectory_index = NearIndex(.97)
    print('Auditing actual SFT rows with Qwen tokenizer (no command execution)...', flush=True)
    for index, raw in enumerate(records(sft_path)):
        audit = audit_sft(raw, index, tok, lookup, a.context_budget, a.reject_observation_truncation)
        rid, question = audit['id'], audit['question']
        if rid in seen_ids:
            audit['filter_reasons'].append('duplicate_original_id')
        seen_ids.add(rid)
        if isinstance(raw, dict) and isinstance(raw.get('messages'), list):
            assistant_text = ' '.join(m.get('content', '') for m in raw['messages'] if isinstance(m, dict) and m.get('role') == 'assistant' and isinstance(m.get('content'), str))
            fingerprint = digest(question_key(assistant_text))
            audit['parsed_trajectory_fingerprint'] = fingerprint
            if assistant_text and fingerprint in seen_trajectory_fingerprints:
                audit['filter_reasons'].append('duplicate_assistant_trajectory')
            elif assistant_text:
                seen_trajectory_fingerprints[fingerprint] = rid
        if isinstance(question, str) and question_key(question):
            key = question_key(question)
            if key in seen_keys:
                audit['filter_reasons'].append('duplicate_question')
                audit['duplicate_of'] = seen_keys[key]
            else:
                seen_keys[key] = rid
            matches = heldout.query(question) if not audit['leakage_matches'] else []
            if matches:
                audit['filter_reasons'].append('near_heldout_question_quarantine')
                reviews += [{'id': rid, 'question': question, 'scope': 'sft_vs_heldout', **m} for m in matches]
            nearby = near_train.query(question)
            if nearby and not audit['filter_reasons']:
                audit['filter_reasons'].append('near_duplicate_question_quarantine')
                audit['near_duplicate_matches'] = nearby
                reviews += [{'id': rid, 'question': question, 'scope': 'sft_near_duplicate', **m} for m in nearby]
            if not audit['filter_reasons']:
                action_text = ' '.join([question, *audit['commands'], *re.findall(r'<answer>(.*?)</answer>', assistant_text, re.S)])
                trajectory_matches = trajectory_index.query(action_text)
                if trajectory_matches:
                    audit['filter_reasons'].append('near_duplicate_action_trajectory_quarantine')
                    reviews += [{'id':rid, 'question':question, 'scope':'sft_action_trajectory', **m} for m in trajectory_matches]
                else:
                    trajectory_index.add(action_text, {'id':rid, 'split':'sft_eligible_actions'})
            if not audit['filter_reasons']:
                near_train.add(question, {'id': rid, 'split': 'sft_eligible'})
        audit['eligible'] = not audit['filter_reasons']
        audits.append(audit)
        if audit['eligible']:
            raw_by_id[rid] = raw
        if (index + 1) % 1000 == 0:
            print(f'  audited {index + 1}', flush=True)
    eligible = [r for r in audits if r['eligible']]
    # Reserve independent SFT validation first. Nested scale-ablation subsets share it.
    # Source-proportional hash selection for validation avoids consuming all
    # singleton long/rare-behavior strata before training coverage sampling.
    order = stratified_order(eligible, a.seed, lambda r:(r['source'],))
    validation = order[:a.val_count]
    train_order = stratified_order(order[a.val_count:], a.seed + 1, stratum)
    selected = train_order[:a.train_count]
    val_ids, train_ids = {r['id'] for r in validation}, {r['id'] for r in selected}
    for r in audits:
        r['selected_split'] = 'sft_val' if r['id'] in val_ids else 'sft_train' if r['id'] in train_ids else None
    write_jsonl(out / 'sft_audit.jsonl', audits)
    def output_row(r):
        return dict(raw_by_id[r['id']], question=r['question'], source=r['source'], answer_correctness='unknown')
    write_jsonl(out / 'sft_selected.jsonl', [output_row(r) for r in selected])
    write_jsonl(out / 'sft_validation.jsonl', [output_row(r) for r in validation])
    import pandas as pd
    for name, rows in [('sft_train', selected), ('sft_val', validation)]:
        pd.DataFrame([{'id': r['id'], 'messages': raw_by_id[r['id']]['messages']} for r in rows], columns=['id', 'messages']).to_parquet(out / f'{name}.parquet', index=False)
    for size in (300, 800, 1200):
        write_jsonl(out / f'sft_candidate_{size}.jsonl', [output_row(r) for r in train_order[:size]])
        json_file(out / f'sft_candidate_{size}_config.json', {'target': size, 'actual': min(size, len(train_order)), 'seed': a.seed, 'validation_ids': sorted(val_ids), 'train_ids': [r['id'] for r in train_order[:size]], 'context_budget': a.context_budget, 'shortfall': max(0, size-len(train_order))})
    report = summarize(audits, selected, validation, a.context_budget)
    report['shortfalls'] = {'train': max(0, a.train_count-len(selected)), 'validation': max(0, a.val_count-len(validation))}
    report['near_duplicate_limit'] = 'Lexical MinHash/Jaccard candidates only, conservatively quarantined. Semantic paraphrases are NOT certified absent; manual review required.'
    json_file(out / 'sft_curation_report.json', report)
    charts(out, audits, selected)
    lines = ['# SFT Curation：真实数据 CPU 审查', '', f'原始 {len(audits)} 条；可用 {len(eligible)} 条；训练 {len(selected)} / 目标 {a.train_count}，独立验证 {len(validation)}。',
             '', 'SFT 原始字段仅 id/messages。Question 从初始 user 消息解析；Source 主要按真实 FlashRAG 问题连接。答案正确性全部 unknown。格式正常结束不等于任务成功。',
             '', 'Token 统计采用实际 Qwen3.5 模板；prompt、assistant、tool 是互斥的消息段；assistant 中另记录 trainable token，其他 token 均 mask。',
             '', '```json', json.dumps(report, ensure_ascii=False, indent=2), '```', '', '![真实轨迹统计](sft_curation_charts.png)',
             '', '输入文件 SHA256、配置、样本 ID 与导出 hash 见 sft_manifest.json。未执行 Shell、GPU Rollout、训练或付费 API。']
    (out / 'sft_curation_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Building 600-or-fewer audited RL train candidates...', flush=True)
    curated_sft = {question_key(r['question']): r for r in selected + validation}
    excluded_index = NearIndex(a.near_threshold)
    for row in validation + ([] if a.allow_sft_rl_overlap else selected):
        excluded_index.add(row['question'], {'id': row['id'], 'split': 'sft_val' if row['id'] in val_ids else 'sft_train_excluded'})
    seen_rl = set()
    candidates = []
    qa_audit = []
    for r in qa:
        row = dict(r)
        row['question_tokens'] = len(tok.encode(row['question'], add_special_tokens=False)) if row['basic_valid'] else None
        row['filter_reasons'] = [] if row['basic_valid'] else ['invalid_basic_fields']
        row['supervised_overlap_ids'] = []
        row['near_heldout_status'] = 'not_assessed_unselected'
        if row['basic_valid'] and row['original_split'] == 'train':
            key = question_key(row['question'])
            matches = lookup[key]
            if any(m['original_split'] == 'dev' for m in matches):
                row['filter_reasons'].append('heldout_dev_question_overlap')
            if key in seen_rl:
                row['filter_reasons'].append('duplicate_rl_question')
            seen_rl.add(key)
            overlap = curated_sft.get(key)
            if overlap:
                row['supervised_overlap_ids'] = [overlap['id']]
                if overlap['id'] in val_ids or not a.allow_sft_rl_overlap:
                    row['filter_reasons'].append('sft_validation_overlap' if overlap['id'] in val_ids else 'sft_train_overlap_excluded_by_policy')
            if not row['filter_reasons']:
                candidates.append(row)
        qa_audit.append(row)
    candidates.sort(key=lambda r: digest([a.seed, r['id']]))
    # Balance source coverage without duplicates, borrowing unused quota if one source is short.
    pools = {s: [r for r in candidates if r['source'] == s] for s in ('nq','hotpotqa')}
    selected_rl = []
    candidate_index = NearIndex(a.near_threshold)
    while len(selected_rl) < a.rl_pool_count and any(pools.values()):
        source_name = min((s for s in pools if pools[s]), key=lambda s: (sum(r['source'] == s for r in selected_rl), s))
        r = pools[source_name].pop(0)
        near = heldout.query(r['question']) + excluded_index.query(r['question'])
        r['near_heldout_status'] = 'lexical_candidates_checked_semantics_unreviewed'
        if near:
            r['filter_reasons'].append('near_heldout_question_quarantine')
            reviews += [{'id': r['id'], 'question': r['question'], 'scope': 'rl_vs_heldout', **m} for m in near]
            continue
        matches = candidate_index.query(r['question'])
        if matches:
            r['filter_reasons'].append('near_duplicate_rl_candidate_quarantine')
            reviews += [{'id': r['id'], 'question': r['question'], 'scope': 'rl_candidate_near_duplicate', **m} for m in matches]
            continue
        selected_rl.append(r)
        candidate_index.add(r['question'], {'id': r['id'], 'split': 'rl_candidates'})
    selected_rl_ids={r['id'] for r in selected_rl}
    for r in qa_audit:
        r['selected_candidate']=r['id'] in selected_rl_ids
        r['original_split_role']='heldout_reserved' if r['original_split']=='dev' else 'training_candidate'
    write_jsonl(out / 'qa_audit.jsonl', qa_audit)
    write_jsonl(out / 'rl_candidates.jsonl', selected_rl)
    write_jsonl(out / 'manual_review.jsonl', reviews)
    # Preserve the prior frozen dev/final test partition if supplied in the repo.
    for split in ('dev', 'test'):
        frozen = ROOT / 'mini/data' / f'{split}.jsonl'
        if frozen.exists():
            (out / f'rl_{split}.jsonl').write_bytes(frozen.read_bytes())
    (out / 'rl_profile.jsonl').write_text('', encoding='utf-8')
    (out / 'rl_selected.jsonl').write_text('', encoding='utf-8')
    uniform = sorted(selected_rl, key=lambda r: digest([a.seed, 'uniform', r['id']]))[:300]
    write_jsonl(out / 'rl_uniform.jsonl', uniform)
    json_file(out / 'rl_profile_report.json', {'status': 'pending_frozen_sft_rollouts', 'profiled_questions': 0, 'planned_initial_questions': min(100, len(selected_rl)), 'rollouts_per_question': 4, 'no_rollouts_executed': True})
    json_file(out / 'rl_sampling_manifest.json', {'status':'difficulty_selection_pending_profiles', 'seed':a.seed, 'pool_size':len(selected_rl), 'uniform_ids':[r['id'] for r in uniform], 'requested_ratio':{'easy':.15,'medium':.70,'hard':.15}, 'training_contract':'Only QA records are training inputs. Profiling trajectories must never be fed to GRPO; fresh on-policy rollout is mandatory.'})
    json_file(out / 'sft_manifest.json', {'sources': {'sft':'alireza7/GrepSeek-ColdStart-SFT-10k', 'qa':'RUC-NLPIR/FlashRAG_datasets', 'corpus':'PeterJinGo/wiki-18-corpus', 'model':'Qwen/Qwen3.5-2B'},
              'hub_revision':json.loads((source / 'download_manifest.json').read_text()) if (source / 'download_manifest.json').exists() else 'unknown_for_offline_cache; content hashes authoritative', 'input_files_sha256':provenance, 'tokenizer':tokenizer_provenance, 'config':vars(a),
              'train_ids':[r['id'] for r in selected], 'validation_ids':[r['id'] for r in validation], 'rl_candidate_ids':[r['id'] for r in selected_rl],
              'curation_code_sha256':{name:file_hash(ROOT/'mini'/name) for name in ('curation.py','curation_utils.py','common.py','template.py')},
              'heldout_policy':'All original FlashRAG dev excluded from training, SFT validation also excluded from RL. Prior mini/data dev/test partitions preserved.',
              'files_sha256':artifact_hashes(out)})
    print(json.dumps({'sft':len(selected), 'sft_val':len(validation), 'rl_candidates':len(selected_rl), 'manual_review_pairs':len(reviews), 'gpu_rollouts':0}), flush=True)


if __name__ == '__main__':
    main()
