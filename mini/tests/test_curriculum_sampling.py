import copy

import pytest

from mini.sample_curriculum import allocate, select_stage


def candidates():
    return [{'id': f'{source}:{calls}:{i}', 'source': source, 'tool_calls': calls,
             'tokens': {'total_sequence_tokens': 2000 + (i % 4) * 2000},
             'command_categories': ['search_capped'], 'gold_match_status': 'partial_match_review'}
            for source in ('nq', 'hotpotqa') for calls in (1, 2, 3) for i in range(20)]


def test_joint_quotas_are_exact_and_answer_labels_do_not_rank():
    rows = candidates()
    selected = select_stage(rows, {1: 6, 2: 12, 3: 2}, 10, 42, 'a')
    assert sum(r['source'] == 'nq' for r in selected) == 10
    assert [sum(r['tool_calls'] == c for r in selected) for c in (1, 2, 3)] == [6, 12, 2]
    changed = copy.deepcopy(list(reversed(rows)))
    for row in changed:
        row['gold_match_status'] = 'exact_match'
    assert [r['id'] for r in selected] == [r['id'] for r in select_stage(changed, {1: 6, 2: 12, 3: 2}, 10, 42, 'a')]


def test_later_stage_preserves_first_and_never_duplicates():
    rows = candidates()
    a = select_stage(rows, {1: 6, 2: 12, 3: 2}, 10, 42, 'a')
    a_ids = {r['id'] for r in a}
    b = select_stage([r for r in rows if r['id'] not in a_ids], {1: 4, 2: 18, 3: 8}, 15, 42, 'b')
    assert not a_ids & {r['id'] for r in b}
    assert len({r['id'] for r in a + b}) == 50


def test_joint_capacity_shortage_fails_instead_of_duplicate_filling():
    rows = [r for r in candidates() if r['source'] == 'nq' or r['tool_calls'] == 1]
    with pytest.raises(ValueError, match='joint capacity'):
        allocate(rows, {1: 0, 2: 8, 3: 2}, 5)
