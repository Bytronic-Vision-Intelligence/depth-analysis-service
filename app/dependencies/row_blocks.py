"""Run whole-image numpy/OpenCV work as row blocks on a small thread pool.

numpy and OpenCV release the GIL on large array operations but each call runs on one
core. Work whose result for a row depends only on that row (a horizontal kernel, a
per-pixel formula) can be split into row blocks that fill disjoint rows of a
preallocated output, giving the same result in a fraction of the time.
"""

from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from typing import Callable

_POOL = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=min(8, cpu_count() or 1), thread_name_prefix="row_block")
    return _POOL


def run_in_row_blocks(rows: int, work: Callable[[int, int], None], blocks: int = 4, min_rows: int = 256) -> None:
    """Call ``work(top, bottom)`` for ``blocks`` row ranges in parallel and wait for all.
    Small images, or ``blocks`` of 1, run in one piece on the calling thread."""
    blocks = min(blocks, cpu_count() or 1)
    if blocks <= 1 or rows < min_rows:
        work(0, rows)
        return
    cuts = [rows * i // blocks for i in range(blocks + 1)]
    for future in [_pool().submit(work, cuts[i], cuts[i + 1]) for i in range(blocks)]:
        future.result()
