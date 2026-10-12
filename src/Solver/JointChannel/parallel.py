"""Bounded, ordered process execution for independent gravity arcs."""
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context


def map_arcs(func, arcs, arguments=(), workers=1, ordered=True):
    workers = int(workers)
    if workers < 1:
        raise ValueError('arc worker count must be positive')
    # Workers do small per-arc linear algebra; BLAS thread fan-out would
    # multiply the reserved core count and can exhaust node memory.
    names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = '1'
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=get_context('spawn')) as pool:
            # Completion-order consumers identify their results by arc number.
            # Refill before yielding so the parent single-writer cannot idle workers.
            if not ordered:
                source = iter(arcs)
                pending = set()
                for _ in range(workers):
                    try:
                        pending.add(pool.submit(func, next(source), *arguments))
                    except StopIteration:
                        break
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        try:
                            pending.add(pool.submit(func, next(source), *arguments))
                        except StopIteration:
                            pass
                        yield result
                return
            pending = deque()
            source = iter(arcs)
            for arc in source:
                pending.append(pool.submit(func, arc, *arguments))
                if len(pending) == workers:
                    break
            while pending:
                yield pending.popleft().result()
                try:
                    arc = next(source)
                except StopIteration:
                    continue
                pending.append(pool.submit(func, arc, *arguments))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
