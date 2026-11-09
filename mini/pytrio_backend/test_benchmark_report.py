import copy

import pytest

from mini.pytrio_backend.benchmark_report import build


def fixtures():
    schedule = [{'id': 'a', 'source': 'nq', 'question': 'Fixture A?', 'golden_answers': ['Paris']},
                {'id': 'b', 'source': 'hotpotqa', 'question': 'Fixture B?', 'golden_answers': ['Paris']}]
    trajectories = []
    for row in schedule:
        success = row['id'] == 'a'
        text = '<think>Known.</think><answer>Paris</answer>' if success else '<think>No action.</think>'
        trajectories.append({'question_id': row['id'], 'source': row['source'],
                             'question': row['question'], 'gold': row['golden_answers'],
                             'finish_reason': 'answer' if success else 'no_action',
                             'messages': [{'role': 'assistant', 'content': text}],
                             'turns': [{'tokens': [1, 2], 'stop_reason': 'stop'}],
                             'tool_count': 0, 'tokens': [1, 2, 3], 'initial_length': 1})
    return trajectories, schedule


def test_no_answer_remains_in_benchmark_denominator():
    trajectories, schedule = fixtures()
    summary, _ = build(trajectories, schedule)
    overall = summary['overall']
    assert overall['n'] == 2
    assert overall['exact_match'] == .5 and overall['no_final_answer_rate'] == .5
    assert overall['success_count'] == 1
    assert summary['by_source']['hotpotqa']['exact_match'] == 0


def test_incomplete_duplicate_and_changed_golds_are_rejected():
    trajectories, schedule = fixtures()
    with pytest.raises(ValueError, match='Incomplete'):
        build(trajectories[:1], schedule)
    with pytest.raises(ValueError, match='Duplicate'):
        build(trajectories + trajectories[:1], schedule)
    changed = copy.deepcopy(trajectories)
    changed[0]['gold'] = ['London']
    with pytest.raises(ValueError, match='golds'):
        build(changed, schedule)


def test_interrupted_request_retains_full_reservation():
    from mini.pytrio_backend.continue_eval import recovery
    events = [
        {'kind': 'request_submitted', 'request_id': 1, 'reserved_work': {'sample': 100}},
        {'kind': 'request_finished', 'request_id': 1, 'budget': {'tokens': {'sample': 10}}},
        {'kind': 'request_submitted', 'request_id': 2, 'reserved_work': {'sample': 200, 'prefill': 50}},
    ]
    used, uncertain = recovery(events)
    assert used == {'sample': 210, 'prefill': 50}
    assert len(uncertain) == 1 and uncertain[0]['request_id'] == 2
