"""CPU-only provenance, loading, lexical review and deterministic sampling."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import zlib

from mini.common import digest, question_key, write_jsonl


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_file(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def records(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing real input: {path.resolve()}. Supply --input-dir with downloaded official files; no synthetic fallback.')
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    else:
        with path.open(encoding='utf-8') as f:
            for line, text in enumerate(f, 1):
                if text.strip():
                    try:
                        yield json.loads(text)
                    except ValueError as exc:
                        yield {'_parse_error': str(exc), '_line': line, '_raw_sha256': hashlib.sha256(text.encode()).hexdigest()}


def qa_records(directory):
    for source in ('nq', 'hotpotqa'):
        for split in ('train', 'dev'):
            for i, r in enumerate(records(Path(directory) / f'{source}_{split}.jsonl')):
                valid = isinstance(r, dict) and isinstance(r.get('question'), str) and bool(question_key(r['question']))
                gold = r.get('golden_answers') if isinstance(r, dict) else None
                valid = valid and isinstance(gold, list) and bool(gold) and all(isinstance(g, str) and g.strip() for g in gold)
                raw_id = r.get('id') if isinstance(r, dict) else None
                yield {'id': f'{source}:{split}:{raw_id if raw_id is not None else "row-"+str(i)}',
                       'original_id': raw_id, 'source': source, 'original_split': split,
                       'question': r.get('question') if isinstance(r, dict) else None,
                       'golden_answers': gold, 'basic_valid': bool(valid),
                       'original_fields': list(r) if isinstance(r, dict) else [],
                       'retrievable': 'unknown'}


def quantiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {k: None for k in ('p50', 'p75', 'p90', 'p95', 'p99', 'max')}
    def q(p):
        n = (len(values) - 1) * p
        lo, hi = math.floor(n), math.ceil(n)
        return values[lo] * (hi - n) + values[hi] * (n - lo) if lo != hi else values[lo]
    return {**{f'p{p}': q(p / 100) for p in (50, 75, 90, 95, 99)}, 'max': values[-1]}


class NearIndex:
    """Deterministic MinHash candidate retrieval + exact word-set Jaccard.

    Lexical screening only. LSH can miss pairs; it never certifies semantic isolation.
    No model, shell execution, network or embedding API is used.
    """
    def __init__(self, threshold=.85):
        self.threshold = threshold
        self.bands = defaultdict(list)
        self.items = []

    def signature(self, text):
        words = frozenset(question_key(text).split())
        hashes = [zlib.crc32(w.encode()) for w in words]
        if not hashes:
            return words, ()
        sig = [min(((h * (2 * i + 1) + 2654435761 * (i + 1)) % 4294967311) for h in hashes) for i in range(12)]
        return words, tuple((i // 2, sig[i], sig[i + 1]) for i in range(0, 12, 2))

    def query(self, text):
        words, bands = self.signature(text)
        candidates = set(i for b in bands for i in self.bands.get(b, ()))
        result = []
        for i in sorted(candidates):
            other, label = self.items[i]
            similarity = len(words & other) / len(words | other) if words | other else 0.
            if similarity >= self.threshold:
                result.append({'other': label, 'word_jaccard': similarity, 'method': 'minhash_candidates_exact_word_jaccard', 'semantic_equivalence': 'unreviewed'})
        return result

    def add(self, text, label):
        words, bands = self.signature(text)
        index = len(self.items)
        self.items.append((words, label))
        for b in bands:
            self.bands[b].append(index)


def stratified_order(rows, seed, stratum):
    """Proportional source/behavior coverage; rare strata get early representation.

    Within each source allocate sqrt(N) to strata (capped by availability).
    Across sources preserve eligible source prevalence. No duplicate filling.
    A single deterministic order yields nested 300/800/1200 train subsets.
    """
    buckets = defaultdict(list)
    for row in rows:
        buckets[stratum(row)].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda r: digest([seed, r['id']]))
    positions = Counter()
    source_count = Counter(r.get('source', 'unknown') for r in rows)
    source_positions = Counter()
    result = []
    while buckets:
        # Weighted fair queue, rather than mechanically equalizing command counts.
        sources = {k[0] for k in buckets}
        source = min(sources, key=lambda s: ((source_positions[s]+1) / source_count[s], str(s)))
        key = min((k for k in buckets if k[0] == source), key=lambda k: ((positions[k] + 1) / math.sqrt(len(buckets[k]) + positions[k]), digest([seed, k])))
        result.append(buckets[key].pop(0))
        positions[key] += 1
        source_positions[source] += 1
        if not buckets[key]:
            del buckets[key]
    return result


def safe_output(directory, signature):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'input_config_signature.json'
    if lock.exists() and json.loads(lock.read_text()) != signature:
        raise ValueError(f'{directory} contains a different experiment; use a new --out')
    json_file(lock, signature)
    return directory


def artifact_hashes(directory):
    return {str(p.relative_to(directory)): file_hash(p) for p in sorted(Path(directory).rglob('*')) if p.is_file() and 'manifest' not in p.name and p.name != 'input_config_signature.json'}
