"""Catalog popularity: a cold-start fallback, not a personalization model."""

from __future__ import annotations

import numpy as np
import polars as pl

from savor.config import EVENT_WEIGHTS


class PopularityModel:
    """Weighted event counts. Power-law catalogs will dominate this list."""

    def __init__(self) -> None:
        self.item_index: dict[str, int] = {}
        self.scores: np.ndarray = np.zeros(0, dtype=np.float64)
        self.item_ids: list[str] = []

    def fit(self, interactions: pl.DataFrame, item_ids: list[str]) -> PopularityModel:
        self.item_ids = list(item_ids)
        self.item_index = {item_id: i for i, item_id in enumerate(self.item_ids)}
        scores = np.zeros(len(self.item_ids), dtype=np.float64)
        if interactions.height == 0:
            self.scores = scores
            return self
        weights = interactions.with_columns(
            pl.col("event").replace_strict(EVENT_WEIGHTS, default=0.0).alias("w")
        )
        grouped = weights.group_by("item_id").agg(pl.col("w").sum())
        for item_id, weight in grouped.iter_rows():
            idx = self.item_index.get(str(item_id))
            if idx is not None:
                scores[idx] = float(weight)
        self.scores = scores
        return self

    def top(self, k: int, exclude: set[str]) -> list[tuple[str, float]]:
        order = np.argsort(-self.scores)
        out: list[tuple[str, float]] = []
        for idx in order:
            item_id = self.item_ids[int(idx)]
            if item_id in exclude:
                continue
            out.append((item_id, float(self.scores[int(idx)])))
            if len(out) >= k:
                break
        return out

    def score_map(self) -> dict[str, float]:
        return {item_id: float(self.scores[i]) for i, item_id in enumerate(self.item_ids)}
