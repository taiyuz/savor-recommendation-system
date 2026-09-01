from __future__ import annotations

import polars as pl

from savor.config import SPLIT_CUTOFF
from savor.data.loader import Catalog
from savor.evaluate import held_out_positives
from savor.ranking.features import FeatureBuilder


def test_split_is_strictly_temporal(split_frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    train, valid = split_frames
    assert train.height > 0
    assert valid.height > 0
    assert train["timestamp"].max() < SPLIT_CUTOFF
    assert valid["timestamp"].min() >= SPLIT_CUTOFF


def test_no_shared_interaction_rows(split_frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    train, valid = split_frames
    train_keys = set(
        zip(
            train["user_id"].to_list(),
            train["item_id"].to_list(),
            train["event"].to_list(),
            train["timestamp"].to_list(),
            strict=True,
        )
    )
    valid_keys = set(
        zip(
            valid["user_id"].to_list(),
            valid["item_id"].to_list(),
            valid["event"].to_list(),
            valid["timestamp"].to_list(),
            strict=True,
        )
    )
    assert train_keys.isdisjoint(valid_keys)


def test_held_out_labels_are_unseen_in_train(
    train_catalog: Catalog, valid_catalog: Catalog
) -> None:
    labels = held_out_positives(train_catalog.interactions, valid_catalog.interactions)
    train_pairs = {
        (str(u), str(i))
        for u, i in train_catalog.interactions.select(["user_id", "item_id"]).iter_rows()
    }
    for user_id, items in labels.items():
        for item_id in items:
            assert (user_id, item_id) not in train_pairs


def test_held_out_positives_ignore_view_events() -> None:
    """Views are weak implicit feedback. Treating them as eval labels inflates recall."""
    train = pl.DataFrame({"user_id": ["u1"], "item_id": ["i1"], "event": ["save"]})
    valid = pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i2", "i3", "i1"],
            "event": ["view", "visit", "view"],
        }
    )
    labels = held_out_positives(train, valid)
    # i2 is view-only, i1 already appeared in train, i3 is the only held-out visit.
    assert labels == {"u1": {"i3"}}


def test_popularity_feature_ignores_future_events(
    train_catalog: Catalog, valid_catalog: Catalog
) -> None:
    features = FeatureBuilder().fit(
        train_catalog.users, train_catalog.items, train_catalog.interactions
    )
    future_only = (
        valid_catalog.interactions.join(
            train_catalog.interactions.select("item_id").unique(),
            on="item_id",
            how="anti",
        )
        if valid_catalog.interactions.height
        else valid_catalog.interactions
    )
    # Any item that only appears after cutoff must have zero train popularity.
    if future_only.height:
        for item_id in future_only["item_id"].unique().to_list():
            assert features.popularity.get(str(item_id), 0.0) == 0.0
