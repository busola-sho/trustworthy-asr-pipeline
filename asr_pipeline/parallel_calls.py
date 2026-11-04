"""Call models in parallel and return results in input order."""

import concurrent.futures
from typing import Callable, List, Any


def call_models_in_parallel(
    items: List[Any],
    process_item: Callable[[Any], Any],
    max_workers: int = 8,
    progress_every: int = 10,
) -> List[Any]:
    """Call process_item for each item. Keep the input order.

    Threads let calls wait for Ollama at the same time. A failed call returns None
    so the caller can record the failure without losing the other results."""
    # Store each result at its input index, even if another call finishes first.
    results = [None] * len(items)
    completed = 0

    def _safe_worker(index_and_item):
        index, item = index_and_item
        try:
            return index, process_item(item)
        except Exception as error:
            print(f"  ERROR (concurrent worker, item {index}): {error}")
            return index, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_safe_worker, (i, item)) for i, item in enumerate(items)
        ]
        for future in concurrent.futures.as_completed(futures):
            index, result = future.result()
            results[index] = result
            completed += 1
            if progress_every and completed % progress_every == 0:
                print(f"  {completed}/{len(items)} done")

    return results
