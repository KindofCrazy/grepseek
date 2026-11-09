"""Read and verify a multi-segment evaluation, including crash tails."""
import json
from pathlib import Path

from mini.common import ROOT
from mini.curation_utils import file_hash


INFERENCE_FIELDS = ('stage','model','rank','seed','context','prompt','response',
    'max_turn_tokens','max_turns','max_tools','tool_tokens','tool_timeout',
    'temperature','top_p','initial_checkpoint','eval_file','data','corpus','tokenizer')


def read_events(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    events = []
    for index, line in enumerate(lines):
        try: events.append(json.loads(line))
        except ValueError:
            if index != len(lines)-1:
                raise ValueError('Corrupted event log before final crash tail')
    return events


def history(run, seen=None):
    run = Path(run).resolve()
    seen = set() if seen is None else seen
    if run in seen:
        raise ValueError('Evaluation continuation cycle')
    seen.add(run)
    chain = []
    continuation = run/'continuation.json'
    if continuation.exists():
        link = json.loads(continuation.read_text())
        prior = (ROOT/link['prior']).resolve()
        if file_hash(prior/'events.jsonl') != link['prior_events_sha256']:
            raise ValueError('Prior evaluation events changed since continuation')
        chain = history(prior, seen)
        previous = json.loads((prior/'config.json').read_text())
        current = json.loads((run/'config.json').read_text())
        if any(previous.get(k) != current.get(k) for k in INFERENCE_FIELDS):
            raise ValueError('Evaluation inference config changed across continuation')
    return chain + [(run, read_events(run/'events.jsonl'))]


def recovery(events, initial=None):
    from collections import Counter
    submitted = {e['request_id']: e for e in events if e['kind']=='request_submitted'}
    used = Counter(initial or {})
    done = set()
    for event in events:
        if event['kind']=='request_finished':
            done.add(event['request_id'])
            for key, value in event['budget']['tokens'].items():
                used[key] = max(used[key], value)
    uncertain = [row for key,row in submitted.items() if key not in done]
    for row in uncertain:
        used.update(row['reserved_work'])
    return dict(used), uncertain
