from __future__ import annotations

from savor.config import CANDIDATE_K
from savor.data.loader import Catalog
from savor.pipeline import Recommender


def test_candidate_cap_and_no_seen_items(train_catalog: Catalog) -> None:
    rec = Recommender().fit(train_catalog)
    warm = next(uid for uid, items in rec.generator.history.items() if len(items) >= 4)
    candidates = rec.generator.retrieve(warm, exclude_seen=True)
    assert 1 <= len(candidates) <= 2 * CANDIDATE_K + max(10, CANDIDATE_K // 4)
    seen = rec.generator.seen(warm)
    assert seen.isdisjoint({c.item_id for c in candidates})


def test_cold_start_falls_back_to_popularity(train_catalog: Catalog) -> None:
    rec = Recommender().fit(train_catalog)
    cold_users = [u for u in train_catalog.user_ids() if rec.generator.is_cold(u)]
    assert cold_users, "synthetic catalog should include cold-start users"
    items = rec.recommend(cold_users[0], k=5)
    assert len(items) == 5
    assert all("popularity" in row.sources for row in items)


def test_recommendations_are_unique(train_catalog: Catalog) -> None:
    rec = Recommender().fit(train_catalog)
    warm = next(uid for uid, items in rec.generator.history.items() if items)
    ids = [row.item_id for row in rec.recommend(warm, k=10)]
    assert ids == list(dict.fromkeys(ids))
