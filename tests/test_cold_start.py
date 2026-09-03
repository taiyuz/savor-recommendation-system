from __future__ import annotations

import polars as pl

from savor.data.cold_start import cold_item_ids, future_only_item_ids
from savor.data.loader import Catalog
from savor.evaluate import held_out_positives
from savor.metrics import recall_at_k
from savor.retrieval.candidates import CandidateGenerator


def _train_valid() -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    # Two users share i_warm in train. Both visit i_new only after cutoff.
    # Concatenating the frames would make i_new a co-occurring neighbor of i_warm.
    train = pl.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "item_id": ["i_warm", "i_warm", "i_other"],
            "event": ["visit", "save", "visit"],
        }
    )
    valid = pl.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["i_new", "i_new"],
            "event": ["visit", "visit"],
        }
    )
    item_ids = ["i_warm", "i_other", "i_new", "i_ghost"]
    return train, valid, item_ids


def test_cold_item_ids_are_catalog_minus_train_support() -> None:
    train, valid, item_ids = _train_valid()
    assert cold_item_ids(item_ids, train) == {"i_new", "i_ghost"}
    assert future_only_item_ids(train, valid) == {"i_new"}
    assert "i_ghost" not in future_only_item_ids(train, valid)


def test_held_out_positives_keep_train_cold_items() -> None:
    """A first-time valid visit is still a label. Dropping it fakes recall.

    Filtering labels to items that already have train support would make this
    user look like they have no held-out positives, so they would be skipped
    and mean recall would not pay for the miss.
    """
    train, valid, _item_ids = _train_valid()
    labels = held_out_positives(train, valid)
    assert labels == {"u1": {"i_new"}, "u2": {"i_new"}}
    assert recall_at_k(["i_warm", "i_other"], labels["u1"], k=2) == 0.0


def test_future_only_item_is_not_a_cf_neighbor_on_train_fit() -> None:
    """Item-item co-occurrence must not see the valid window.

    If fit() were given train concatenated with valid, i_new would co-occur with
    i_warm (both users visit it after cutoff) and would surface as a neighbor.
    """
    train, _valid, item_ids = _train_valid()
    gen = CandidateGenerator().fit(train, item_ids)
    assert gen.popularity.score_map()["i_new"] == 0.0
    assert gen.popularity.score_map()["i_ghost"] == 0.0
    assert "i_new" not in gen.seen("u1")
    neighbors = {item_id for item_id, _score in gen.item_item.top("u1", k=10, exclude=set())}
    assert "i_new" not in neighbors
    assert "i_ghost" not in neighbors


def test_train_cold_items_have_zero_popularity_on_sample(
    train_catalog: Catalog, valid_catalog: Catalog
) -> None:
    cold = cold_item_ids(train_catalog.item_ids(), train_catalog.interactions)
    # Two restaurants have no events in the whole sample; more may be train-only-cold.
    assert len(cold) >= 2
    future_only = future_only_item_ids(train_catalog.interactions, valid_catalog.interactions)
    assert future_only <= cold
    gen = CandidateGenerator().fit(train_catalog.interactions, train_catalog.item_ids())
    scores = gen.popularity.score_map()
    for item_id in cold:
        assert scores.get(item_id, 0.0) == 0.0
    warm_user = next(uid for uid, items in gen.history.items() if items)
    n_items = len(gen.item_item.item_ids)
    neighbors = {
        item_id for item_id, _score in gen.item_item.top(warm_user, k=n_items, exclude=set())
    }
    assert neighbors.isdisjoint(cold)
