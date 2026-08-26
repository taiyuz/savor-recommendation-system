"""Merge two-tower, item-item, and popularity into a candidate set."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from savor.config import CANDIDATE_K
from savor.retrieval.item_item import ItemItemModel
from savor.retrieval.popularity import PopularityModel
from savor.retrieval.two_tower import LatentTwoTower


@dataclass
class Candidate:
    item_id: str
    two_tower_score: float = 0.0
    item_item_score: float = 0.0
    popularity_score: float = 0.0
    sources: tuple[str, ...] = field(default_factory=tuple)


class CandidateGenerator:
    def __init__(self, candidate_k: int = CANDIDATE_K) -> None:
        self.candidate_k = candidate_k
        self.two_tower = LatentTwoTower()
        self.item_item = ItemItemModel()
        self.popularity = PopularityModel()
        self.history: dict[str, set[str]] = {}

    def fit(self, interactions: pl.DataFrame, item_ids: list[str]) -> CandidateGenerator:
        self.two_tower.fit(interactions, item_ids)
        self.item_item.fit(interactions, item_ids)
        self.popularity.fit(interactions, item_ids)
        self.history = {}
        if interactions.height:
            grouped = interactions.group_by("user_id").agg(pl.col("item_id").unique())
            for user_id, items in grouped.iter_rows():
                self.history[str(user_id)] = {str(i) for i in items}
        return self

    def seen(self, user_id: str) -> set[str]:
        return set(self.history.get(user_id, set()))

    def is_cold(self, user_id: str) -> bool:
        return user_id not in self.history or not self.history[user_id]

    def retrieve(
        self,
        user_id: str,
        *,
        k: int | None = None,
        exclude_seen: bool = True,
    ) -> list[Candidate]:
        cap = k or self.candidate_k
        exclude = self.seen(user_id) if exclude_seen else set()
        merged: dict[str, Candidate] = {}

        def add(item_id: str, source: str, **scores: float) -> None:
            cand = merged.get(item_id)
            if cand is None:
                cand = Candidate(item_id=item_id)
                merged[item_id] = cand
            if "two_tower_score" in scores:
                cand.two_tower_score = scores["two_tower_score"]
            if "item_item_score" in scores:
                cand.item_item_score = scores["item_item_score"]
            if "popularity_score" in scores:
                cand.popularity_score = scores["popularity_score"]
            if source not in cand.sources:
                cand.sources = (*cand.sources, source)

        if not self.is_cold(user_id):
            for item_id, score in self.two_tower.top(user_id, cap, exclude):
                add(item_id, "two_tower", two_tower_score=score)
            for item_id, score in self.item_item.top(user_id, cap, exclude):
                add(item_id, "item_item", item_item_score=score)

        pop_k = cap if self.is_cold(user_id) else max(10, cap // 4)
        for item_id, score in self.popularity.top(pop_k, exclude):
            add(item_id, "popularity", popularity_score=score)

        pop_map = self.popularity.score_map()
        if not self.is_cold(user_id):
            tt = self.two_tower.scores_for_user(user_id)
            ii = self.item_item.scores_for_user(user_id)
            for cand in merged.values():
                idx = self.two_tower.item_index.get(cand.item_id)
                if idx is not None:
                    cand.two_tower_score = float(tt[idx])
                    cand.item_item_score = float(ii[idx])
                cand.popularity_score = pop_map.get(cand.item_id, cand.popularity_score)
        else:
            for cand in merged.values():
                cand.popularity_score = pop_map.get(cand.item_id, cand.popularity_score)

        return list(merged.values())
