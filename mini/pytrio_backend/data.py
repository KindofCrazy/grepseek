import json
import math
from pathlib import Path
import random

import numpy as np
import pytrio as trio

from mini.common import read_jsonl
from mini.curation_utils import file_hash
from mini.retokenize import measure_sft, verify_files, verify_tokenizer


def preflight(config):
    from transformers import AutoTokenizer
    manifest = json.loads((Path(config.data) / 'sft_manifest.json').read_text())
    verify_files(config.data, manifest['files_sha256'])
    if manifest['sources']['model'] != config.model:
        raise ValueError('Model differs from the frozen data snapshot')
    verify_tokenizer(config.tokenizer, manifest['model_provenance'])
    tok = AutoTokenizer.from_pretrained(config.tokenizer, local_files_only=True)
    return tok, manifest


def supervised(tokenizer, row, context):
    _, tokens, weights = measure_sft(tokenizer, row, context)
    return trio.Datum(model_input=trio.ModelInput.from_ints(tokens[:-1]),
                      loss_fn_inputs={'target_tokens': np.asarray(tokens[1:], dtype=np.int64),
                                      'weights': np.asarray(weights[1:], dtype=np.float32)})


class Cursor:
    def __init__(self, rows, seed, state=None):
        if not rows or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Nonempty data with unique IDs required')
        self.rows, self.seed = rows, seed
        self.epoch, self.offset = (state['epoch'], state['offset']) if state else (0, 0)
        self.draws = state.get('draws', {}) if state else {}

    def take(self, count):
        result = []
        while len(result) < count:
            order = list(range(len(self.rows)))
            random.Random(self.seed + self.epoch).shuffle(order)
            if self.offset == len(order):
                self.epoch += 1
                self.offset = 0
                continue
            row = self.rows[order[self.offset]]
            self.offset += 1
            self.draws[row['id']] = self.draws.get(row['id'], 0) + 1
            result.append(row)
        return result

    def state(self):
        return {'epoch': self.epoch, 'offset': self.offset, 'draws': self.draws}


class CurriculumCursor:
    """One pass per immutable stage, with a separately shuffled order and exact resume."""
    def __init__(self, rows, seed, stages, state=None):
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate curriculum IDs')
        index = {r['id']:r for r in rows}
        planned = [qid for stage in stages for qid in stage['ids']]
        if len(set(planned)) != len(planned) or set(planned) != set(index):
            raise ValueError('Curriculum stages must partition the training snapshot')
        self.stages, self.seed, self.index = stages, seed, index
        if state and state.get('kind') != 'curriculum':
            raise ValueError('Cannot restore a random cursor as a curriculum')
        self.stage_index = state['stage_index'] if state else 0
        self.draws = state.get('draws',{}) if state else {}
        self.cursor = Cursor([index[qid] for qid in stages[self.stage_index]['ids']],
                             seed+self.stage_index, state['stage_cursor'] if state else None)

    @property
    def epoch(self): return self.stage_index

    @property
    def phase(self): return self.stages[self.stage_index]['name']

    @property
    def phase_end(self): return self.cursor.offset == len(self.cursor.rows)

    def take(self,count):
        if self.phase_end:
            if self.stage_index+1 == len(self.stages):
                raise ValueError('Curriculum exhausted; refusing repeated exposure')
            self.stage_index += 1
            self.cursor = Cursor([self.index[qid] for qid in self.stages[self.stage_index]['ids']],
                                 self.seed+self.stage_index)
        if count > len(self.cursor.rows)-self.cursor.offset:
            raise ValueError('Batch crosses curriculum boundary')
        batch = self.cursor.take(count)
        for row in batch:
            self.draws[row['id']] = self.draws.get(row['id'],0)+1
        return batch

    def state(self):
        return {'kind':'curriculum','stage_index':self.stage_index,'stage_cursor':self.cursor.state(),
                'draws':self.draws}


def datum_tokens(datums):
    return sum(len(d.model_input) for d in datums)


def supervised_metrics(result, datums):
    loss_sum = 0.
    count = 0
    if len(result.loss_fn_outputs) != len(datums):
        raise ValueError('Forward output count mismatch')
    for output, datum in zip(result.loss_fn_outputs, datums):
        values = output['logprobs'].to_numpy().reshape(-1)
        weights = datum.loss_fn_inputs['weights'].to_numpy().reshape(-1)
        if len(values) != len(weights) or not np.isfinite(values).all():
            raise ValueError('Invalid forward logprobs')
        loss_sum += float(-np.dot(values, weights))
        count += int(weights.sum())
    if count <= 0 or not math.isfinite(loss_sum):
        raise ValueError('No finite supervised loss')
    return {'assistant_nll': loss_sum/count, 'supervised_tokens': count,
            'computed_input_tokens': datum_tokens(datums), 'remote_metrics': result.metrics}
