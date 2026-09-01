from __future__ import annotations

from savor.data.loader import Catalog
from savor.evaluate import evaluate
from savor.pipeline import Recommender


def test_ranker_fits_and_scores_in_unit_interval(train_catalog: Catalog) -> None:
    rec = Recommender().fit(train_catalog)
    assert rec.ranker.fitted
    warm = next(uid for uid, items in rec.generator.history.items() if items)
    ranked = rec.recommend(warm, k=10)
    assert len(ranked) == 10
    for row in ranked:
        assert 0.0 <= row.score <= 1.0


def test_ranked_scores_are_monotone_nonincreasing(train_catalog: Catalog) -> None:
    rec = Recommender().fit(train_catalog)
    warm = next(uid for uid, items in rec.generator.history.items() if items)
    scores = [row.score for row in rec.recommend(warm, k=10)]
    assert scores
    # argsort(scores) instead of argsort(-scores) would invert the list.
    assert scores == sorted(scores, reverse=True)


def test_offline_eval_has_a_real_ranking_signal(
    train_catalog: Catalog, valid_catalog: Catalog
) -> None:
    rec = Recommender().fit(train_catalog)
    report = evaluate(rec, train_catalog, valid_catalog, k=10)
    assert report.n_eval_users >= 20
    # Synthetic data, not a claim that two-stage NDCG beats popularity every run.
    assert report.recall > 0.0
    assert 0.0 <= report.ndcg <= 1.0
