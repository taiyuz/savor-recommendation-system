"""Item-item CF: cosine on the item-user implicit matrix (train slice only)."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity

from savor.config import EVENT_WEIGHTS


class ItemItemModel:
    def __init__(self) -> None:
        self.item_ids: list[str] = []
        self.item_index: dict[str, int] = {}
        self.user_index: dict[str, int] = {}
        self.sim: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.user_items: dict[str, list[tuple[int, float]]] = {}

    def fit(self, interactions: pl.DataFrame, item_ids: list[str]) -> ItemItemModel:
        self.item_ids = list(item_ids)
        self.item_index = {item_id: i for i, item_id in enumerate(self.item_ids)}
        users = sorted(set(interactions["user_id"].to_list())) if interactions.height else []
        self.user_index = {u: i for i, u in enumerate(users)}
        n_u, n_i = len(users), len(self.item_ids)
        matrix = np.zeros((n_u, n_i), dtype=np.float64)
        self.user_items = {u: [] for u in users}

        if interactions.height == 0 or n_u == 0 or n_i == 0:
            self.sim = np.zeros((n_i, n_i), dtype=np.float64)
            return self

        weighted = interactions.with_columns(
            pl.col("event").replace_strict(EVENT_WEIGHTS, default=0.0).alias("w")
        )
        grouped = weighted.group_by(["user_id", "item_id"]).agg(pl.col("w").sum())
        for user_id, item_id, weight in grouped.iter_rows():
            u = self.user_index.get(str(user_id))
            i = self.item_index.get(str(item_id))
            if u is None or i is None:
                continue
            matrix[u, i] += float(weight)
            self.user_items[str(user_id)].append((i, float(weight)))

        if n_i == 1:
            self.sim = np.array([[1.0]], dtype=np.float64)
        else:
            self.sim = cosine_similarity(matrix.T)
            np.fill_diagonal(self.sim, 0.0)
        return self

    def scores_for_user(self, user_id: str) -> np.ndarray:
        n_i = len(self.item_ids)
        history = self.user_items.get(user_id, [])
        if not history:
            return np.zeros(n_i, dtype=np.float64)
        acc = np.zeros(n_i, dtype=np.float64)
        for item_idx, weight in history:
            acc += self.sim[item_idx] * weight
        return acc

    def top(self, user_id: str, k: int, exclude: set[str]) -> list[tuple[str, float]]:
        scores = self.scores_for_user(user_id)
        order = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        for idx in order:
            item_id = self.item_ids[int(idx)]
            if item_id in exclude or scores[int(idx)] <= 0:
                continue
            out.append((item_id, float(scores[int(idx)])))
            if len(out) >= k:
                break
        return out
