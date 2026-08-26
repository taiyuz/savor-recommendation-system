"""Fit retrieval + ranker and produce a ranked list for one user."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from savor.config import DEFAULT_K, RANDOM_SEED
from savor.data.loader import Catalog
from savor.ranking.features import FeatureBuilder
from savor.ranking.ranker import GradientBoostRanker, stack_features
from savor.retrieval.candidates import CandidateGenerator


@dataclass(frozen=True)
class ScoredItem:
    item_id: str
    name: str
    cuisine: str
    neighborhood: str
    price_tier: int
    score: float
    sources: tuple[str, ...]


class Recommender:
    def __init__(self) -> None:
        self.generator = CandidateGenerator()
        self.features = FeatureBuilder()
        self.ranker = GradientBoostRanker()
        self.items: pl.DataFrame | None = None
        self.item_lookup: dict[str, dict[str, object]] = {}

    def fit(self, catalog: Catalog) -> Recommender:
        item_ids = catalog.item_ids()
        self.generator.fit(catalog.interactions, item_ids)
        self.features.fit(catalog.users, catalog.items, catalog.interactions)
        rng = np.random.default_rng(RANDOM_SEED)
        self.ranker.fit(
            interactions=catalog.interactions,
            features=self.features,
            generator=self.generator,
            rng=rng,
        )
        self.items = catalog.items
        self.item_lookup = {
            str(row["item_id"]): dict(row) for row in catalog.items.iter_rows(named=True)
        }
        return self

    def recommend(
        self,
        user_id: str,
        k: int = DEFAULT_K,
        *,
        exclude_seen: bool = True,
    ) -> list[ScoredItem]:
        candidates = self.generator.retrieve(user_id, exclude_seen=exclude_seen)
        if not candidates:
            return []
        matrix = stack_features(user_id, candidates, self.features)
        if self.ranker.fitted:
            scores = self.ranker.predict_proba(matrix)
        else:
            scores = np.array([c.popularity_score for c in candidates], dtype=np.float64)
        order = np.argsort(-scores)
        out: list[ScoredItem] = []
        for idx in order[:k]:
            cand = candidates[int(idx)]
            meta = self.item_lookup.get(cand.item_id, {})
            out.append(
                ScoredItem(
                    item_id=cand.item_id,
                    name=str(meta.get("name", cand.item_id)),
                    cuisine=str(meta.get("cuisine", "")),
                    neighborhood=str(meta.get("neighborhood", "")),
                    price_tier=int(meta.get("price_tier", 0) or 0),
                    score=float(scores[int(idx)]),
                    sources=cand.sources,
                )
            )
        return out

    def popularity_recommend(
        self,
        user_id: str,
        k: int = DEFAULT_K,
        *,
        exclude_seen: bool = True,
    ) -> list[ScoredItem]:
        exclude = self.generator.seen(user_id) if exclude_seen else set()
        out: list[ScoredItem] = []
        for item_id, score in self.generator.popularity.top(k, exclude):
            meta = self.item_lookup.get(item_id, {})
            out.append(
                ScoredItem(
                    item_id=item_id,
                    name=str(meta.get("name", item_id)),
                    cuisine=str(meta.get("cuisine", "")),
                    neighborhood=str(meta.get("neighborhood", "")),
                    price_tier=int(meta.get("price_tier", 0) or 0),
                    score=float(score),
                    sources=("popularity",),
                )
            )
        return out
