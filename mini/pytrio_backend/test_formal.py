import math

import pytest

from mini.pytrio_backend.agent_metrics import tool_outcome
from mini.pytrio_backend.config import Config
from mini.pytrio_backend.data import CurriculumCursor
from mini.pytrio_backend.runtime import adam


def test_curriculum_resume_across_stage_boundary():
    rows=[{'id':str(i)} for i in range(12)]
    stages=[{'name':'a','ids':[str(i) for i in range(4)]},{'name':'b','ids':[str(i) for i in range(4,12)]}]
    c=CurriculumCursor(rows,42,stages)
    a=c.take(4); assert c.phase_end
    restored=CurriculumCursor(rows,42,stages,c.state())
    assert restored.take(4)==c.take(4) and c.phase=='b'
    b=c.take(4)
    assert len(set(r['id'] for r in a+b))==8
    with pytest.raises(ValueError,match='exhausted'):c.take(4)


def test_cosine_is_continuous_across_curriculum_and_reaches_floor():
    c=Config(steps=625,lr_schedule='cosine')
    assert adam(c,32).learning_rate==pytest.approx(5e-5)
    assert adam(c,625).learning_rate==pytest.approx(5e-6)
    assert 0 < adam(c,250).learning_rate-adam(c,251).learning_rate < 1e-6
    assert adam(Config(),2).learning_rate==5e-5


def test_search_no_match_count_zero_and_real_errors_differ():
    base={'command':'rg -F x corpus.jsonl','stdout':'','stderr':'','exit_code':1,'timed_out':False}
    assert tool_outcome(base)['execution_success'] and tool_outcome(base)['empty_search']
    assert tool_outcome({**base,'exit_code':2,'stderr':'missing file'})['error']
    counted={**base,'command':'rg x corpus.jsonl | wc -l','exit_code':0,'stdout':'0\n'}
    assert tool_outcome(counted)['empty_search']
    assert not tool_outcome(counted)['nonempty_search']
    assert tool_outcome({**base,'command':'rg x missing | head -n 1','exit_code':0,'stderr':'No such file'})['error']
