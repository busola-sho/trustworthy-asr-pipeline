"""
src/concurrent_ollama.py

Reusable helper for running many independent Ollama calls concurrently,
instead of one at a time. This is the actual bottleneck in your current
ensemble scripts - they call client.chat(...) sequentially in a for loop,
so even if the Ollama SERVER is configured to allow parallelism
(OLLAMA_NUM_PARALLEL), nothing takes advantage of it, since your client
never sends more than one request at a time.

REQUIRES: the Ollama server must be started with parallelism enabled,
e.g. in your sbatch script or interactive session, before `ollama serve`:

    export OLLAMA_NUM_PARALLEL=8      # match roughly to --max-workers below
    export OLLAMA_MAX_LOADED_MODELS=1 # you're only using 1 model at a time per script
    ollama serve &
    sleep 5   # give the server a moment to start before hitting it

Without OLLAMA_NUM_PARALLEL set on the SERVER side, raising concurrency
on the CLIENT side just means requests queue up server-side instead of
client-side - no actual speedup, just moves where the waiting happens.

Usage (see the naive.py rewrite for the full pattern):
    from src.concurrent_ollama import run_concurrent

    results = run_concurrent(
        work_items=[(idx, prompt_args) for idx, prompt_args in enumerate(...)],
        worker_fn=lambda item: ollama_select(client, ..., *item[1]),
        max_workers=8,
        progress_every=10,
    )
    # results is a list, SAME ORDER as work_items, even though execution
    # happened concurrently - safe to zip back against your original
    # sample list without re-sorting anything.
"""

import concurrent.futures
from typing import Callable, List, Any


def run_concurrent(
    work_items: List[Any],
    worker_fn: Callable[[Any], Any],
    max_workers: int = 8,
    progress_every: int = 10,
) -> List[Any]:
    """
    Runs worker_fn(item) concurrently across work_items using a thread
    pool (threads, not processes - correct choice here since the actual
    work is waiting on network I/O to the Ollama server, not CPU-bound
    computation; threads share memory cheaply and avoid the overhead of
    spinning up separate processes for what's fundamentally an I/O-bound
    workload).

    Returns results in the SAME ORDER as work_items, regardless of which
    order they actually complete in - uses executor.map(), which
    guarantees this, so you can safely zip results back against your
    original ordered sample list without extra bookkeeping.

    A single failed worker_fn call (returns None, or raises) does NOT
    crash the whole batch - exceptions are caught per-item and the
    corresponding result slot is set to None, matching how your existing
    scripts already treat a failed Ollama call (results.append({"error":
    True, ...})) - the caller's existing None-checking logic after
    calling this still works unchanged.
    """
    results = [None] * len(work_items)
    completed = 0

    def _safe_worker(index_and_item):
        index, item = index_and_item
        try:
            return index, worker_fn(item)
        except Exception as e:
            print(f"  ERROR (concurrent worker, item {index}): {e}")
            return index, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_safe_worker, (i, item)) for i, item in enumerate(work_items)]
        for future in concurrent.futures.as_completed(futures):
            index, result = future.result()
            results[index] = result
            completed += 1
            if progress_every and completed % progress_every == 0:
                print(f"  {completed}/{len(work_items)} done")

    return results