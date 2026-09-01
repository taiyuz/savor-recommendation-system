from __future__ import annotations

from savor.metrics import coverage_at_k, mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_at_k_hand_computed() -> None:
    ranked = ["a", "b", "c"]
    relevant = {"c", "d"}
    assert recall_at_k(ranked, relevant, k=2) == 0.0
    assert recall_at_k(ranked, relevant, k=3) == 0.5
    assert recall_at_k(ranked, set(), k=3) == 0.0


def test_ndcg_perfect_and_partial() -> None:
    relevant = {"a", "b"}
    assert ndcg_at_k(["a", "b", "c"], relevant, k=2) == 1.0
    swapped = ndcg_at_k(["c", "a"], relevant, k=2)
    perfect = ndcg_at_k(["a", "b"], relevant, k=2)
    assert 0.0 < swapped < perfect


def test_ndcg_idcg_truncates_when_more_relevant_than_k() -> None:
    # Three relevant items, k=1: a perfect prefix must still be 1.0.
    # Dividing by an untruncated IDCG (all three relevant) would score < 1.
    relevant = {"a", "b", "c"}
    assert ndcg_at_k(["a"], relevant, k=1) == 1.0


def test_mrr_at_k_is_first_hit_reciprocal_rank() -> None:
    relevant = {"x"}
    assert mrr_at_k(["x", "a", "b"], relevant, k=3) == 1.0
    # A recall-shaped implementation would score 1.0 here too (hit anywhere in k).
    assert mrr_at_k(["a", "x", "b"], relevant, k=3) == 0.5
    assert mrr_at_k(["a", "b", "x"], relevant, k=3) == 1.0 / 3.0
    assert mrr_at_k(["a", "b", "c"], relevant, k=3) == 0.0


def test_mrr_at_k_ignores_hits_beyond_k() -> None:
    relevant = {"x"}
    assert mrr_at_k(["a", "b", "x"], relevant, k=2) == 0.0
    assert mrr_at_k(["a", "x"], relevant, k=2) == 0.5


def test_mrr_at_k_uses_only_the_first_relevant_item() -> None:
    # Averaging 1/rank over every hit (MAP-shaped) would score (1 + 1/3) / 2 here.
    relevant = {"a", "c"}
    assert mrr_at_k(["a", "b", "c"], relevant, k=3) == 1.0
    # First hit at rank 2, later hit at rank 3 → still 1/2, not (1/2 + 1/3) / 2.
    assert mrr_at_k(["b", "a", "c"], relevant, k=3) == 0.5


def test_mrr_at_k_empty_and_nonpositive_k() -> None:
    assert mrr_at_k(["a"], set(), k=1) == 0.0
    assert mrr_at_k([], {"a"}, k=5) == 0.0
    assert mrr_at_k(["a"], {"a"}, k=0) == 0.0
    assert mrr_at_k(["a"], {"a"}, k=-1) == 0.0


def test_coverage_at_k() -> None:
    recs = {"u1": ["a", "b"], "u2": ["b", "c"]}
    assert coverage_at_k(recs, catalog_size=4, k=2) == 0.75
