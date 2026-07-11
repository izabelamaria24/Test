import math

from realestate.eval.generate_queries import EvalPair
from realestate.eval.metrics import evaluate, mean_reciprocal_rank, ndcg_at_k, recall_at_k


def test_recall_at_k_hit():
    assert recall_at_k(["a", "b", "c"], relevant_id="b", k=3) == 1.0


def test_recall_at_k_miss_outside_k():
    assert recall_at_k(["a", "b", "c"], relevant_id="d", k=3) == 0.0


def test_ndcg_at_k_rewards_higher_rank():
    top_rank = ndcg_at_k(["target", "b", "c"], relevant_id="target", k=3)
    bottom_rank = ndcg_at_k(["a", "b", "target"], relevant_id="target", k=3)
    assert top_rank == 1.0
    assert 0 < bottom_rank < top_rank


def test_mean_reciprocal_rank_first_position():
    assert mean_reciprocal_rank(["target", "b"], relevant_id="target") == 1.0


def test_mean_reciprocal_rank_second_position():
    assert mean_reciprocal_rank(["a", "target"], relevant_id="target") == 0.5


def test_mean_reciprocal_rank_not_found():
    assert mean_reciprocal_rank(["a", "b"], relevant_id="target") == 0.0


def test_evaluate_averages_metrics_across_pairs():
    pairs = [
        EvalPair(query="q1", relevant_listing_id="1"),
        EvalPair(query="q2", relevant_listing_id="2"),
    ]

    def fake_search(query: str) -> list[str]:
        return {"q1": ["1", "x", "y"], "q2": ["x", "2", "y"]}[query]

    scores = evaluate(pairs, search_fn=fake_search, k=3)

    assert scores["recall_at_k"] == 1.0
    assert 0.0 < scores["mrr"] < 1.0
