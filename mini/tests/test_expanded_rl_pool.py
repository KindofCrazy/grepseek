import random

from mini.common import question_key
from mini.expand_rl_pool import ExactNearIndex


def test_exact_prefix_filter_matches_exhaustive_search_during_growth():
    rng=random.Random(42)
    index=ExactNearIndex()
    stored=[]
    queries=[]
    vocabulary=[f'w{i}' for i in range(80)]
    for i in range(350):
        words=set(rng.sample(vocabulary,rng.randint(1,40)))
        index.add(' '.join(words),i)
        stored.append(words)
        altered=set(words)
        altered.discard(next(iter(sorted(words))))
        altered.add(f'new{i}')
        queries.extend([words,altered,set(rng.sample(vocabulary,rng.randint(1,40)))])
        for query in queries[-3:]:
            expected=any(10*len(query&other)>=9*len(query|other) for other in stored)
            found=index.query(' '.join(query))
            assert bool(found)==expected
            if found:
                other=stored[found['other']]
                assert 10*len(query&other)>=9*len(query|other)


def test_exact_threshold_boundary_and_normalized_duplicates():
    index=ExactNearIndex()
    index.add(' '.join(f'w{i}' for i in range(10)),'ten')
    assert index.query(' '.join(f'w{i}' for i in range(9)))['word_jaccard']==.9
    assert index.query(' '.join(f'w{i}' for i in range(8))) is None
    index.add('The capital of France?','punctuation')
    assert index.query('THE CAPITAL OF FRANCE')['other']=='punctuation'
    assert index.query('') is None
