"""Linear two-tower retrieval via truncated SVD of the implicit matrix.

User tower = user factor row. Item tower = item factor row. Score = dot
product. This is matrix factorization, not a deep two-tower; it is the
CPU-honest version of that architecture and fits in milliseconds.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD

from savor.config import EVENT_WEIGHTS, LATENT_DIM, RANDOM_SEED


class LatentTwoTower:
    def __init__(self, dim: int = LATENT_DIM) -> None:
        self.dim = dim
        self.item_ids: list[str] = []
        self.item_index: dict[str, int] = {}
        self.user_index: dict[str, int] = {}
        self.user_factors: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.item_factors: np.ndarray = np.zeros((0, 0), dtype=np.float64)

    def fit(self, interactions: pl.DataFrame, item_ids: list[str]) -> LatentTwoTower:
        self.item_ids = list(item_ids)
        self.item_index = {item_id: i for i, item_id in enumerate(self.item_ids)}
        users = sorted(set(interactions["user_id"].to_list())) if interactions.height else []
        self.user_index = {u: i for i, u in enumerate(users)}
        n_u, n_i = len(users), len(self.item_ids)
        matrix = np.zeros((max(n_u, 1), max(n_i, 1)), dtype=np.float64)

        if interactions.height and n_u and n_i:
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

        n_comp = min(self.dim, max(1, min(matrix.shape) - 1))
        if n_comp < 1 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            self.user_factors = np.zeros((n_u, 1), dtype=np.float64)
            self.item_factors = np.zeros((n_i, 1), dtype=np.float64)
            return self

        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_SEED)
        self.user_factors = svd.fit_transform(matrix[:n_u, :n_i])
        self.item_factors = svd.components_.T
        return self

    def scores_for_user(self, user_id: str) -> np.ndarray:
        u = self.user_index.get(user_id)
        if u is None or self.item_factors.size == 0:
            return np.zeros(len(self.item_ids), dtype=np.float64)
        return self.user_factors[u] @ self.item_factors.T

    def top(self, user_id: str, k: int, exclude: set[str]) -> list[tuple[str, float]]:
        scores = self.scores_for_user(user_id)
        order = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        for idx in order:
            item_id = self.item_ids[int(idx)]
            if item_id in exclude:
                continue
            out.append((item_id, float(scores[int(idx)])))
            if len(out) >= k:
                break
        return out
