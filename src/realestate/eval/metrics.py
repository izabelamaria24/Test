import math
from typing import Callable

from realestate.eval.generate_queries import EvalPair


def recall_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    top_k = retrieved_ids[:k]
    if relevant_id not in top_k:
        return 0.0
    rank = top_k.index(relevant_id) + 1  # 1-indexed
    return 1.0 / math.log2(rank + 1)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_id: str) -> float:
    if relevant_id not in retrieved_ids:
        return 0.0
    return 1.0 / (retrieved_ids.index(relevant_id) + 1)


def evaluate(
    pairs: list[EvalPair], search_fn: Callable[[str], list[str]], k: int = 10
) -> dict[str, float]:
    recalls, ndcgs, rrs = [], [], []
    for pair in pairs:
        retrieved = search_fn(pair.query)
        recalls.append(recall_at_k(retrieved, pair.relevant_listing_id, k))
        ndcgs.append(ndcg_at_k(retrieved, pair.relevant_listing_id, k))
        rrs.append(mean_reciprocal_rank(retrieved, pair.relevant_listing_id))

    n = len(pairs)
    return {
        "recall_at_k": sum(recalls) / n,
        "ndcg_at_k": sum(ndcgs) / n,
        "mrr": sum(rrs) / n,
    }
