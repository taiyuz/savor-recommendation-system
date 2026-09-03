"""Train-cold items: listed in the catalog, zero events in this interaction slice.

Collaborative retrieval (item-item, SVD) has nothing to condition on for these
ids. Treating a post-cutoff first visit as train support is item-index leakage.
Dropping those ids from held-out labels is the complementary eval cheat: recall
rises because the unretrievable positives quietly disappear.
"""

from __future__ import annotations

import polars as pl


def interacted_item_ids(interactions: pl.DataFrame) -> set[str]:
    if interactions.height == 0:
        return set()
    return {str(item_id) for item_id in interactions["item_id"].to_list()}


def cold_item_ids(
    catalog_item_ids: list[str] | set[str],
    interactions: pl.DataFrame,
) -> set[str]:
    """Catalog items that never appear in this slice (train-cold / never-interacted)."""
    warm = interacted_item_ids(interactions)
    return {str(item_id) for item_id in catalog_item_ids} - warm


def future_only_item_ids(train: pl.DataFrame, valid: pl.DataFrame) -> set[str]:
    """Items whose first logged event is on or after the cutoff."""
    return interacted_item_ids(valid) - interacted_item_ids(train)
