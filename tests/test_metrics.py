from __future__ import annotations

from savor.metrics import coverage_at_k, ndcg_at_k, recall_at_k


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


def test_coverage_at_k() -> None:
    recs = {"u1": ["a", "b"], "u2": ["b", "c"]}
    assert coverage_at_k(recs, catalog_size=4, k=2) == 0.75
