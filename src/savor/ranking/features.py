"""Pointwise ranking features. All aggregates are fit on the train slice only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from savor.config import EVENT_WEIGHTS, POSITIVE_EVENTS
from savor.retrieval.candidates import Candidate

FEATURE_NAMES: tuple[str, ...] = (
    "two_tower_score",
    "item_item_score",
    "log_popularity",
    "cuisine_match",
    "price_diff",
    "same_neighborhood",
    "dietary_ok",
    "item_rating",
    "log_user_activity",
    "cuisine_affinity",
    "distance_km",
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


@dataclass
class UserFeat:
    neighborhood: str
    cuisine_prefs: set[str]
    price_pref: int
    dietary: str
    n_events: int
    cuisine_counts: dict[str, int]


@dataclass
class ItemFeat:
    neighborhood: str
    cuisine: str
    price_tier: int
    avg_rating: float
    n_reviews: int
    lat: float
    lon: float
    vegetarian_ok: bool
    vegan_ok: bool
    gluten_free_ok: bool


def _dietary_ok(user: UserFeat, item: ItemFeat) -> float:
    if user.dietary == "none":
        return 1.0
    if user.dietary == "vegetarian":
        return 1.0 if item.vegetarian_ok else 0.0
    if user.dietary == "vegan":
        return 1.0 if item.vegan_ok else 0.0
    if user.dietary == "gluten_free":
        return 1.0 if item.gluten_free_ok else 0.0
    return 1.0


class FeatureBuilder:
    def __init__(self) -> None:
        self.users: dict[str, UserFeat] = {}
        self.items: dict[str, ItemFeat] = {}
        self.popularity: dict[str, float] = {}
        self.hood_centroids: dict[str, tuple[float, float]] = {}

    def fit(
        self,
        users: pl.DataFrame,
        items: pl.DataFrame,
        interactions: pl.DataFrame,
    ) -> FeatureBuilder:
        self.items = {}
        for row in items.iter_rows(named=True):
            self.items[str(row["item_id"])] = ItemFeat(
                neighborhood=str(row["neighborhood"]),
                cuisine=str(row["cuisine"]),
                price_tier=int(row["price_tier"]),
                avg_rating=float(row["avg_rating"]),
                n_reviews=int(row["n_reviews"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                vegetarian_ok=bool(row["vegetarian_ok"]),
                vegan_ok=bool(row["vegan_ok"]),
                gluten_free_ok=bool(row["gluten_free_ok"]),
            )

        pos = (
            interactions.filter(pl.col("event").is_in(sorted(POSITIVE_EVENTS)))
            if interactions.height
            else interactions
        )
        cuisine_by_user: dict[str, dict[str, int]] = {}
        if pos.height:
            joined = pos.join(items.select(["item_id", "cuisine"]), on="item_id", how="left")
            for user_id, cuisine in joined.select(["user_id", "cuisine"]).iter_rows():
                cuisine_by_user.setdefault(str(user_id), {})
                cuisine_by_user[str(user_id)][str(cuisine)] = (
                    cuisine_by_user[str(user_id)].get(str(cuisine), 0) + 1
                )

        n_events: dict[str, int] = {}
        if interactions.height:
            counts = interactions.group_by("user_id").len()
            n_events = {str(u): int(n) for u, n in counts.iter_rows()}

        self.users = {}
        for row in users.iter_rows(named=True):
            prefs = tuple(p for p in str(row["cuisine_prefs"]).split("|") if p)
            user_id = str(row["user_id"])
            self.users[user_id] = UserFeat(
                neighborhood=str(row["neighborhood"]),
                cuisine_prefs=set(prefs),
                price_pref=int(row["price_pref"]),
                dietary=str(row["dietary"]),
                n_events=n_events.get(user_id, 0),
                cuisine_counts=cuisine_by_user.get(user_id, {}),
            )

        pop: dict[str, float] = dict.fromkeys(self.items, 0.0)
        if interactions.height:
            weighted = interactions.with_columns(
                pl.col("event").replace_strict(EVENT_WEIGHTS, default=0.0).alias("w")
            )
            grouped = weighted.group_by("item_id").agg(pl.col("w").sum())
            for item_id, weight in grouped.iter_rows():
                pop[str(item_id)] = float(weight)
        self.popularity = pop

        self.hood_centroids = {}
        centroids = items.group_by("neighborhood").agg(
            pl.col("lat").mean().alias("lat"),
            pl.col("lon").mean().alias("lon"),
        )
        for row in centroids.iter_rows(named=True):
            self.hood_centroids[str(row["neighborhood"])] = (
                float(row["lat"]),
                float(row["lon"]),
            )
        return self

    def vector(self, user_id: str, candidate: Candidate) -> np.ndarray:
        user = self.users.get(user_id)
        item = self.items[candidate.item_id]
        cuisine_match = 0.0
        price_diff = 0.0
        same_hood = 0.0
        dietary = 1.0
        activity = 0.0
        affinity = 0.0
        if user is not None:
            cuisine_match = 1.0 if item.cuisine in user.cuisine_prefs else 0.0
            price_diff = abs(user.price_pref - item.price_tier) / 3.0
            same_hood = 1.0 if user.neighborhood == item.neighborhood else 0.0
            dietary = _dietary_ok(user, item)
            activity = float(np.log1p(user.n_events))
            total = sum(user.cuisine_counts.values()) or 1
            affinity = user.cuisine_counts.get(item.cuisine, 0) / total
        pop = self.popularity.get(candidate.item_id, 0.0)
        dist = self.distance_km(user_id, candidate.item_id)
        distance = 0.0 if dist is None else min(float(dist) / 15.0, 1.0)
        return np.array(
            [
                candidate.two_tower_score,
                candidate.item_item_score,
                float(np.log1p(pop)),
                cuisine_match,
                price_diff,
                same_hood,
                dietary,
                item.avg_rating / 5.0,
                activity,
                affinity,
                distance,
            ],
            dtype=np.float64,
        )

    def distance_km(self, user_id: str, item_id: str) -> float | None:
        user = self.users.get(user_id)
        item = self.items.get(item_id)
        if user is None or item is None:
            return None
        centroid = self.hood_centroids.get(user.neighborhood)
        if centroid is None:
            return None
        return _haversine_km(centroid[0], centroid[1], item.lat, item.lon)
