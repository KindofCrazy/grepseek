"""One bounded sampling/search scheduler shared by all questions in a run."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import threading
import time


class SchedulingStopped(RuntimeError):
    pass


class Scheduler:
    def __init__(self, config, log):
        self.width = config.question_concurrency
        self.log = log
        self.gates = {'sample': threading.BoundedSemaphore(config.sample_concurrency),
                      'search': threading.BoundedSemaphore(config.search_concurrency)}
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.first_error = None
        self.active = dict.fromkeys(self.gates, 0)
        self.peak = dict.fromkeys(self.gates, 0)
        self.wait_s = dict.fromkeys(self.gates, 0.)
        self.calls = dict.fromkeys(self.gates, 0)

    @contextmanager
    def slot(self, kind):
        start = time.monotonic()
        gate = self.gates[kind]
        while True:
            if self.cancelled.is_set():
                raise SchedulingStopped('Another rollout failed; new requests cancelled')
            if gate.acquire(timeout=.1):
                break
        try:
            if self.cancelled.is_set():
                raise SchedulingStopped('Another rollout failed; new requests cancelled')
            with self.lock:
                self.wait_s[kind] += time.monotonic()-start
                self.calls[kind] += 1
                self.active[kind] += 1
                self.peak[kind] = max(self.peak[kind], self.active[kind])
            try:
                yield
            finally:
                with self.lock:
                    self.active[kind] -= 1
        finally:
            gate.release()

    def snapshot(self):
        with self.lock:
            return {f'{kind}_{key}': values[kind]
                    for key, values in (('active', self.active), ('peak', self.peak),
                                        ('wait_s', self.wait_s), ('calls', self.calls))
                    for kind in self.gates}

    def ordered(self, function, items):
        """Bound submissions, persist within workers, accept in input order.

        Never select only fast-finishing RL groups. Join in-flight workers before
        the caller writes a final budget or closes its event log.
        """
        iterator = iter(items)
        pending = deque()
        pool = ThreadPoolExecutor(max_workers=self.width, thread_name_prefix='question')

        def execute(item):
            try:
                return function(item)
            except BaseException as error:
                with self.lock:
                    if self.first_error is None:
                        self.first_error = error
                self.cancelled.set()
                raise

        try:
            for _ in range(self.width):
                try: item = next(iterator)
                except StopIteration: break
                pending.append(pool.submit(execute, item))
            while pending:
                result = pending.popleft().result()
                if self.cancelled.is_set():
                    raise self.first_error
                yield result
                try: item = next(iterator)
                except StopIteration: continue
                pending.append(pool.submit(execute, item))
        except BaseException as error:
            self.cancelled.set()
            for future in pending:
                future.cancel()
            if isinstance(error, SchedulingStopped) and self.first_error is not None:
                raise self.first_error
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            self.log.event('scheduler_state', **self.snapshot())

    def completed(self, function, items):
        """Evaluation rolling window: replace finished questions immediately."""
        iterator = iter(items)
        pending = set()
        pool = ThreadPoolExecutor(max_workers=self.width, thread_name_prefix='evaluation')
        def execute(item):
            try:
                return function(item)
            except BaseException as error:
                with self.lock:
                    if self.first_error is None:
                        self.first_error = error
                self.cancelled.set()
                raise
        def submit():
            try: item = next(iterator)
            except StopIteration: return
            pending.add(pool.submit(execute,item))
        try:
            for _ in range(self.width): submit()
            while pending:
                done,_ = wait(pending,return_when=FIRST_COMPLETED)
                if self.cancelled.is_set():
                    raise self.first_error
                for future in done:
                    pending.remove(future)
                    yield future.result()
                    submit()
        except BaseException:
            self.cancelled.set()
            for future in pending: future.cancel()
            raise
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
            self.log.event('scheduler_state',**self.snapshot())

    def ordered_refill(self, function, items, refill):
        """Append known-required replacements while older jobs are still running.

        Refill decisions run in submission order on the caller thread. New jobs
        stay behind already allocated jobs, preserving question/seed order and
        logical round boundaries. At most ``width`` groups are in flight.
        """
        queued, pending = deque(items), deque()
        pool = ThreadPoolExecutor(max_workers=self.width, thread_name_prefix='refill')

        def execute(item):
            try:
                return function(item)
            except BaseException as error:
                with self.lock:
                    if self.first_error is None:
                        self.first_error = error
                self.cancelled.set()
                raise

        def submit():
            while queued and len(pending) < self.width:
                if self.cancelled.is_set():
                    raise self.first_error or SchedulingStopped('Rollout cancelled')
                pending.append(pool.submit(execute, queued.popleft()))

        try:
            submit()
            while pending:
                result = pending.popleft().result()
                if self.cancelled.is_set():
                    raise self.first_error
                queued.extend(refill(result))
                submit()
                yield result
        except BaseException as error:
            self.cancelled.set()
            for future in pending:
                future.cancel()
            if isinstance(error, SchedulingStopped) and self.first_error is not None:
                raise self.first_error
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            self.log.event('scheduler_state', **self.snapshot())
