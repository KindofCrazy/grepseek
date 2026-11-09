"""Persistent, read-only PACE planner and bounded native literal verification.

No model-provided shell command executes in this process. Unsupported shapes
and native guard trips return to the existing credential-free search sandbox.
"""
from dataclasses import asdict,replace
import json
from pathlib import Path
import threading
import time

from mini.pytrio_backend.acceleration import eligible
from mini.pytrio_backend.pace_artifacts import CORPUS_SHA256, digest


class PaceSearch:
    def __init__(self,index,corpus,log=None,head_budget=None):
        from inference.pace.core import PaceStructure
        import inference.pace.core as core
        started=time.monotonic()
        self.index,self.corpus,self.log = Path(index).resolve(),Path(corpus).resolve(),log
        binding = json.loads((self.index/'binding.json').read_text())
        if (binding['corpus_sha256']!=CORPUS_SHA256 or binding['corpus_path']!=str(self.corpus) or
                binding['artifact_manifest_sha256']!=digest(self.index/'artifact_manifest.json')):
            raise ValueError('PACE needs a verified local corpus binding')
        manifest = json.loads((self.index/'artifact_manifest.json').read_text())
        # Hash the compact data once at load; the full corpus was hashed during bind.
        for name,expected in manifest['data_sha256'].items():
            path = (self.index/name).resolve()
            if path.parent!=self.index or digest(path)!=expected:
                raise ValueError('PACE data changed after verified import')
        self.pace = PaceStructure.open(self.index,expected_corpus_path=self.corpus)
        self.file_states = {p:(p.stat().st_size,p.stat().st_mtime_ns) for p in
                            [self.index/name for name in manifest['data_sha256']]}
        self.pace.budget = replace(self.pace.budget,max_sa_matches=100000,max_term_postings=250000,
                                   max_candidate_lines=100000,max_materialize_bytes=16*1024**2)
        self.base_budget = self.pace.budget
        self.head_budget = replace(self.base_budget, **(head_budget or {}))
        self.lock = threading.Lock()
        if self.log:
            self.log.event('pace_loaded',index_directory=str(self.index),corpus_sha256=CORPUS_SHA256,
                           artifact_manifest_sha256=digest(self.index/'artifact_manifest.json'),
                           core_sha256=digest(Path(core.__file__)),
                           native_library_sha256=digest(Path(core.__file__).parent/'_native/materialize.so'),
                           budget=asdict(self.base_budget),head_budget=asdict(self.head_budget),
                           load_and_verify_s=time.monotonic()-started)

    def execute(self,command,max_bytes):
        from inference.pace.core import decide_for_pace,verify_natively_streaming,classify_for_pace
        from inference.parallel_search.pipeline import parse_pipeline
        started = time.monotonic()
        stats = {'served':False,'reason':'unsupported_bounded_shape','n_candidates':None,'total_bytes':None}
        result = None
        try:
            if any((path.stat().st_size,path.stat().st_mtime_ns)!=state for path,state in self.file_states.items()):
                raise ValueError('PACE data changed during this run')
            if not eligible(command):
                return None
            # The upstream expansion helper can invoke bash. Never expand model
            # text in the controller process; such commands stay in the sandbox.
            if any(ch in command for ch in '$`~*?[{\\'):
                stats['reason']='shell_expansion_fallback'
                return None
            actual = command.replace('corpus.jsonl','wiki_corpus.jsonl')
            # Bound simultaneous candidate arrays under the current 2GB WSL limit.
            with self.lock:
                stages=[list(stage.tokens) for stage in parse_pipeline(actual)]
                shape = classify_for_pace(stages,cwd=self.corpus.parent,corpus_path=self.corpus)
                is_head = bool(shape and shape.native and shape.native.tail and shape.native.tail[0]=='head')
                # A head tail streams until the requested matches are found;
                # count/full-output commands retain the smaller scan budget.
                self.pace.budget = self.head_budget if is_head else self.base_budget
                decision = decide_for_pace(stages,cwd=self.corpus.parent,pace=self.pace)
                stats = decision.summary()
                stats['budget_profile'] = 'head' if is_head else 'base'
                stats['candidate_pruned'] = decision.served
                stats['served'] = False
                if not decision.served or decision.native is None:
                    return None
                # On a small-RAM SSD host, a 4MiB chunk can fault in thousands
                # of scattered passages before head needs its first three.
                chunk_bytes=16*1024 if decision.native.tail and decision.native.tail[0]=='head' else 1024**2
                stats['chunk_bytes']=chunk_bytes
                result = verify_natively_streaming(self.pace.iter_materialized_chunks(decision.lines,chunk_bytes=chunk_bytes),decision.native)
                if result is None:
                    stats['reason']='native_guard_fallback'
                    return None
                if len(result[0])>=max_bytes:
                    stats['reason']='capture_limit_fallback'
                    return None
                stats.update(served=True,reason='native_verified')
                return result
        finally:
            if self.log:
                self.log.event('pace_execution',command=command,**stats,wall_s=time.monotonic()-started)
