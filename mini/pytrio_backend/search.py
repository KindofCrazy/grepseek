"""Original GrepSeek observation contract in a credential-free, read-only sandbox."""
import os
import json
from collections import Counter, OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import time
import threading

from mini.common import ROOT, shell


class SearchEnvironmentError(RuntimeError):
    pass


def pace_options(config):
    return {'max_candidate_lines':config.pace_head_candidate_lines,
            'max_materialize_bytes':config.pace_head_materialize_bytes,
            'max_term_postings':config.pace_head_term_postings,
            'max_sa_matches':config.pace_head_sa_matches}


class Search:
    def __init__(self, corpus, tokenizer, timeout=180., max_tokens=2048, max_bytes=1048576,
                 cache_entries=64, cache_bytes=8388608, log=None, backend='sequential', shards='',pace_dir='',audit_every=0,
                 pace_head_budget=None):
        if os.name != 'posix' or not shutil.which('bwrap') or not shutil.which('prlimit'):
            raise SearchEnvironmentError('Linux bubblewrap and prlimit are required')
        self.corpus = str(Path(corpus).resolve())
        if not (Path(self.corpus) / 'wiki_corpus.jsonl').is_file():
            raise SearchEnvironmentError('Missing complete Wikipedia corpus')
        manifest_path = Path(self.corpus)/'corpus_manifest.json'
        if not manifest_path.is_file():
            raise SearchEnvironmentError('Corpus has no verification manifest; run environment preparation')
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('synthetic') is not False or manifest['output_lines'] != 21015324 or
                manifest['output_sha256'] != '43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d' or
                (Path(self.corpus)/'wiki_corpus.jsonl').stat().st_size != manifest['output_bytes']):
            raise SearchEnvironmentError('Corpus differs from the prepared complete Wikipedia snapshot')
        self.tokenizer, self.timeout, self.max_tokens, self.max_bytes = tokenizer, timeout, max_tokens, max_bytes
        self.signature = self.corpus_signature()
        self.cache_entries,self.cache_limit,self.log = cache_entries,cache_bytes,log
        self.backend,self.shards = backend,str(Path(shards).resolve()) if shards else ''
        if backend not in ('sequential','sharded','pace') or (backend=='sharded' and not shards):
            raise SearchEnvironmentError('Invalid search backend or missing shard directory')
        self.shard_hash = manifest['output_sha256']
        self.shard_signature = None
        if backend=='sharded' or (backend=='pace' and shards):
            from mini.pytrio_backend.acceleration import shard_signature
            self.shard_signature = shard_signature(self.shards,self.shard_hash)
        self.cache,self.pending,self.counts = OrderedDict(),{},Counter()
        self.cache_size = 0
        self.lock = threading.RLock()
        self.execution_local = threading.local()
        self.pace = None
        self.audit_every,self.audit_count,self.audit_lock = audit_every,0,threading.Lock()
        self.audit_gate = threading.BoundedSemaphore(1)
        if backend=='pace':
            if not pace_dir:
                raise SearchEnvironmentError('PACE requires a locally bound index')
            from mini.pytrio_backend.pace_search import PaceSearch
            self.pace = PaceSearch(pace_dir,Path(self.corpus)/'wiki_corpus.jsonl',log,head_budget=pace_head_budget)

    def corpus_signature(self):
        stat = (Path(self.corpus)/'wiki_corpus.jsonl').stat()
        return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)

    def stats(self):
        with self.lock:
            return {**self.counts,'cache_entries':len(self.cache),'cache_bytes':self.cache_size}

    def execute(self,command):
        started = time.monotonic()
        if self.corpus_signature() != self.signature:
            raise SearchEnvironmentError('Corpus changed during execution; refusing stale cached results')
        if self.shard_signature is not None:
            from mini.pytrio_backend.acceleration import shard_signature
            if shard_signature(self.shards,self.shard_hash) != self.shard_signature:
                raise SearchEnvironmentError('Shard files changed during execution')
        hit = coalesced = False
        with self.lock:
            self.counts['requests'] += 1
            if command in self.cache:
                value,size = self.cache.pop(command)
                self.cache[command] = (value,size)
                self.counts['cache_hits'] += 1
                hit = True
                result = deepcopy(value)
                future = None
            elif command in self.pending:
                future = self.pending[command]
                self.counts['coalesced_requests'] += 1
                coalesced = True
            else:
                future = Future()
                self.pending[command] = future
                self.counts['backend_attempts'] += 1
        try:
            if coalesced:
                result = deepcopy(future.result(timeout=self.timeout+15))
            elif not hit:
                result = self._execute(command)
                size = len(json.dumps(result).encode())
                reusable = (result['exit_code'] in (0,1) and not result['timed_out'] and
                            not result['truncated'] and not result['stderr'])
                with self.lock:
                    if reusable and self.cache_entries and size <= self.cache_limit:
                        self.cache[command] = (deepcopy(result),size)
                        self.cache_size += size
                        while len(self.cache)>self.cache_entries or self.cache_size>self.cache_limit:
                            _,(_,old_size) = self.cache.popitem(last=False)
                            self.cache_size -= old_size
                    future.set_result(deepcopy(result))
            return result
        except BaseException as error:
            if not hit and not coalesced:
                future.set_exception(error)
            raise
        finally:
            with self.lock:
                if not hit and not coalesced:
                    self.pending.pop(command,None)
            if self.log:
                route = 'cache' if hit else 'coalesced' if coalesced else getattr(self.execution_local,'route','unknown')
                self.log.event('search_execution',command=command,backend=self.backend,cache_hit=hit,
                               execution_route=route,coalesced=coalesced,wall_s=time.monotonic()-started,totals=self.stats())

    def argv(self, command):
        base = ['prlimit', '--cpu=120', '--nproc=128', '--fsize=16777216', '--',
                'bwrap', '--die-with-parent', '--unshare-all', '--ro-bind', '/usr', '/usr',
                '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/lib', '/lib',
                '--symlink', 'usr/lib64', '/lib64', '--proc', '/proc', '--dev', '/dev',
                '--tmpfs', '/tmp', '--ro-bind', self.corpus, '/corpus', '--chdir', '/corpus',
                '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin', '--setenv', 'LANG', 'C.UTF-8']
        actual = command.replace('corpus.jsonl', 'wiki_corpus.jsonl')
        if self.shards and self.backend in ('sharded','pace'):
            from mini.pytrio_backend.acceleration import eligible
            if eligible(command):
                return base+['--ro-bind',self.shards,'/shards',
                             '--ro-bind',str(ROOT/'inference'),'/app/inference',
                             '--ro-bind',str(Path(__file__).with_name('sharded_worker.py')),'/worker.py',
                             '/usr/bin/python3','/worker.py','--command',actual,'--shards','/shards',
                             '--corpus','/corpus/wiki_corpus.jsonl','--timeout',str(self.timeout)]
        return base+['/bin/bash','-c',actual]

    def _execute(self, command):
        self.execution_local.route = 'validation'
        payload = {'command':command, 'stdout':'', 'stderr':'', 'exit_code':0,
                   'timed_out':False, 'truncated':False, 'information_lines':[]}
        try:
            shell._validate_pipeline(command)
        except shell.ToolError as error:
            payload.update(stderr=f'validation error: {error}', exit_code=-2)
            return payload
        if self.pace is not None:
            accelerated = self.pace.execute(command,self.max_bytes)
            if accelerated is not None:
                data,code = accelerated
                stdout,truncated = shell._truncate_tokens(data.decode('utf-8',errors='replace'),self.max_tokens,self.tokenizer)
                if truncated:
                    stdout += f'\n[... output truncated at {self.max_tokens} tokens]'
                payload.update(stdout=stdout,exit_code=code,truncated=truncated,
                               information_lines=[line for line in stdout.split('\n') if line.strip()])
                with self.lock:
                    self.counts['pace_executions'] += 1
                self.execution_local.route = 'pace'
                self._audit(command,payload,'pace')
                return payload
            with self.lock:
                self.counts['pace_declines'] += 1
        from mini.pytrio_backend.acceleration import eligible
        sharded = bool(self.shards and self.backend in ('sharded','pace') and eligible(command))
        self.execution_local.route = 'sharded' if sharded else 'sequential'
        with self.lock:
            self.counts['physical_executions'] += 1
            self.counts['sharded_executions' if sharded else 'sequential_executions'] += 1
        proc = subprocess.Popen(self.argv(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env={'PATH':'/usr/bin:/bin', 'LANG':'C.UTF-8'}, start_new_session=True)
        chunks = {'stdout':bytearray(), 'stderr':bytearray()}
        deadline = time.monotonic() + self.timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ, 'stdout')
                selector.register(proc.stderr, selectors.EVENT_READ, 'stderr')
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        payload.update(timed_out=True, exit_code=-1)
                        break
                    for key, _ in selector.select(min(.2, remaining)):
                        data = os.read(key.fd, 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        target = chunks[key.data]
                        target.extend(data[:max(0, self.max_bytes-len(target))])
                        if len(target) >= self.max_bytes:
                            payload['truncated'] = True
                            break
                    if payload['truncated']:
                        break
            if payload['timed_out'] or payload['truncated']:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            code = proc.wait(timeout=5)
            if not payload['timed_out']:
                payload['exit_code'] = code
            stdout = chunks['stdout'].decode('utf-8', errors='replace')
            stderr = chunks['stderr'].decode('utf-8', errors='replace')
            if stderr.startswith('bwrap:') or stderr.startswith('prlimit:') or 'sharded_executor_failed' in stderr or stderr.startswith('Traceback (most recent call last)'):
                raise SearchEnvironmentError('Sandbox infrastructure failed: ' + stderr[:300])
            if payload['timed_out']:
                stdout, stderr = f'[command timed out after {self.timeout}s]', ''
            stdout, token_truncated = shell._truncate_tokens(stdout, self.max_tokens, self.tokenizer)
            payload['truncated'] |= token_truncated
            if token_truncated:
                stdout += f'\n[... output truncated at {self.max_tokens} tokens]'
            elif payload['truncated']:
                stdout += '\n[... output capture byte limit reached]'
            payload.update(stdout=stdout, stderr=stderr,
                           information_lines=[line for line in stdout.split('\n') if line.strip()])
            if sharded and not payload['timed_out']:
                self._audit(command,payload,'sharded')
            return payload
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            proc.stdout.close()
            proc.stderr.close()

    def _audit(self,command,payload,route):
        if not self.audit_every:
            return
        with self.audit_lock:
            self.audit_count += 1
            if self.audit_count % self.audit_every:
                return
        # An audited request waits for its own comparison. Unrelated accelerated
        # results need only increment the counter, not wait for a full-corpus scan.
        audit_started = time.monotonic()
        with self.audit_gate:
            native = Search(self.corpus,self.tokenizer,self.timeout,self.max_tokens,self.max_bytes,
                            cache_entries=0,backend='sequential')._execute(command)
            equal = native == payload
            with self.lock:
                self.counts['native_audits'] += 1
                self.counts['audit_mismatches'] += int(not equal)
            if self.log:
                self.log.event(route+'_native_audit',command=command,equal=equal,
                               mismatched_fields=[k for k in native if native[k]!=payload[k]],
                               compared_queries=1,mismatched_queries=int(not equal),
                               wall_s=time.monotonic()-audit_started)
            if not equal:
                raise SearchEnvironmentError(route.upper()+'/native result mismatch')
