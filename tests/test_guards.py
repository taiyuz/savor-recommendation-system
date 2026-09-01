from __future__ import annotations

from fastapi.testclient import TestClient

from savor.api.app import _catalog, _recommender, create_app
from savor.data.loader import Catalog
from savor.evaluate import evaluate
from savor.pipeline import Recommender
from savor.ranking.features import FeatureBuilder


def test_api_cold_start_user_uses_popularity(catalog: Catalog) -> None:
    seen_users = set(catalog.interactions["user_id"].to_list())
    cold = next(uid for uid in catalog.user_ids() if uid not in seen_users)
    _catalog.cache_clear()
    _recommender.cache_clear()
    client = TestClient(create_app())
    response = client.get("/recommend", params={"user_id": cold, "k": 10})
    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "cold_start_popularity"
    assert len(payload["items"]) == 10
    for item in payload["items"]:
        assert "popularity" in item["sources"]


def test_ranked_list_does_not_leak_seen_items(train_catalog: Catalog) -> None:
    rec = Recommender().fit(train_catalog)
    warm = next(uid for uid, items in rec.generator.history.items() if len(items) >= 4)
    ranked = {row.item_id for row in rec.recommend(warm, k=10, exclude_seen=True)}
    assert ranked
    assert ranked.isdisjoint(rec.generator.seen(warm))


def test_user_activity_feature_ignores_future_events(
    catalog: Catalog, train_catalog: Catalog
) -> None:
    train_only = FeatureBuilder().fit(
        train_catalog.users, train_catalog.items, train_catalog.interactions
    )
    full = FeatureBuilder().fit(catalog.users, catalog.items, catalog.interactions)
    leaked = [
        uid for uid, feat in train_only.users.items() if feat.n_events < full.users[uid].n_events
    ]
    assert leaked, "valid-period events should exist so train activity is a strict subset"
    train_n: dict[str, int] = {}
    if train_catalog.interactions.height:
        counts = train_catalog.interactions.group_by("user_id").len()
        train_n = {str(user_id): int(n) for user_id, n in counts.iter_rows()}
    for uid, feat in train_only.users.items():
        assert feat.n_events <= full.users[uid].n_events
        assert feat.n_events == train_n.get(uid, 0)


def test_sample_eval_table_coverage_gap(train_catalog: Catalog, valid_catalog: Catalog) -> None:
    rec = Recommender().fit(train_catalog)
    report = evaluate(rec, train_catalog, valid_catalog, k=10)
    assert "synthetic" in report.dataset
    assert report.n_eval_users >= 100
    # Sample catalog only: popularity hugs the head, two-stage should not.
    # Do not pin two-stage NDCG above popularity: HGB jitter on 98 items can lose.
    assert report.coverage > 0.8
    assert report.popularity_coverage < 0.4
    assert report.recall > 0.0
    assert 0.0 <= report.ndcg <= 1.0
    assert 0.0 < report.mrr <= 1.0
    assert 0.0 <= report.popularity_mrr <= 1.0
    rows = report.as_table_rows()
    assert rows[0][0].startswith("two-stage")
    assert rows[1][0].startswith("popularity")
