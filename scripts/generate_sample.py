#!/usr/bin/env python3
"""Generate the checked-in synthetic Pittsburgh restaurant catalog.

This is not a scrape and not production traffic. Users, restaurants, and
interactions are sampled from a preference model so collaborative filtering
and content features have a real (but synthetic) signal.
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

SEED = 42
N_USERS = 220
TRAIN_START = datetime(2024, 1, 15, tzinfo=UTC)
VALID_END = datetime(2024, 4, 15, tzinfo=UTC)

NEIGHBORHOODS: dict[str, tuple[float, float]] = {
    "Downtown": (40.4410, -80.0028),
    "Strip District": (40.4504, -79.9959),
    "Lawrenceville": (40.4672, -79.9645),
    "Bloomfield": (40.4614, -79.9493),
    "Shadyside": (40.4552, -79.9348),
    "Squirrel Hill": (40.4384, -79.9228),
    "Oakland": (40.4415, -79.9569),
    "East Liberty": (40.4611, -79.9214),
    "South Side": (40.4286, -79.9747),
    "North Shore": (40.4458, -80.0123),
    "Polish Hill": (40.4568, -79.9654),
    "Highland Park": (40.4793, -79.9162),
    "Point Breeze": (40.4449, -79.9081),
    "Mount Washington": (40.4284, -80.0102),
}

# (name, neighborhood, cuisine, price_tier, veg, vegan, gf)
RESTAURANTS: list[tuple[str, str, str, int, bool, bool, bool]] = [
    ("Primanti Bros Strip", "Strip District", "American", 1, True, False, False),
    ("Gaucho Parrilla", "Strip District", "Argentinian", 3, False, False, True),
    ("Pamela's Diner Strip", "Strip District", "Diner", 1, True, False, False),
    ("Smallman Galley", "Strip District", "American", 3, True, True, True),
    ("Kelly O's Diner", "Strip District", "Diner", 1, True, False, False),
    ("DiAnoia's Eatery", "Strip District", "Italian", 3, True, False, True),
    ("Church Brew Works", "Lawrenceville", "American", 2, True, False, True),
    ("The Vandal", "Lawrenceville", "American", 2, True, True, True),
    ("Apteka", "Lawrenceville", "Vegan", 2, True, True, True),
    ("Dijleh", "Lawrenceville", "Mediterranean", 2, True, False, True),
    ("Morcilla", "Lawrenceville", "Spanish", 3, True, False, True),
    ("Round Corner Cantina", "Lawrenceville", "Mexican", 2, True, True, False),
    ("Noodlehead", "Bloomfield", "Thai", 1, True, True, True),
    ("Tessaro's", "Bloomfield", "American", 2, False, False, True),
    ("Coca Cafe", "Polish Hill", "Cafe", 1, True, True, True),
    ("Piccolo Forno", "Lawrenceville", "Italian", 2, True, False, False),
    ("Casbah", "Shadyside", "Mediterranean", 3, True, True, True),
    ("Grit & Grace", "Downtown", "American", 3, True, False, True),
    ("Meat & Potatoes", "Downtown", "American", 3, False, False, True),
    ("Noodlehead Downtown", "Downtown", "Thai", 1, True, True, True),
    ("Condado Tacos Downtown", "Downtown", "Mexican", 1, True, True, True),
    ("Sienna Mercato", "Downtown", "Italian", 3, True, False, True),
    ("The Commoner", "Downtown", "American", 3, True, False, True),
    ("Everyday Noodles", "Squirrel Hill", "Chinese", 1, True, True, False),
    ("Chengdu Gourmet", "Squirrel Hill", "Chinese", 2, True, False, True),
    ("Sushi Kim", "Squirrel Hill", "Japanese", 2, True, False, True),
    ("Aladdin's Eatery", "Squirrel Hill", "Mediterranean", 1, True, True, True),
    ("Mineo's Pizza", "Squirrel Hill", "Pizza", 1, True, False, False),
    ("Silk Elephant", "Squirrel Hill", "Thai", 2, True, True, True),
    ("Cafe 33", "Squirrel Hill", "Chinese", 1, True, True, False),
    ("India Garden", "Oakland", "Indian", 2, True, True, True),
    ("The O", "Oakland", "American", 1, True, False, False),
    ("Conflict Kitchen Ghost", "Oakland", "Cafe", 1, True, True, True),
    ("Fuel & Fuddle", "Oakland", "American", 2, True, False, False),
    ("Stacks Pancake House", "Oakland", "Diner", 1, True, False, False),
    ("Bangkok Balcony", "Oakland", "Thai", 2, True, True, True),
    ("Sushi Fuku Oakland", "Oakland", "Japanese", 1, True, False, True),
    ("Mad Mex Oakland", "Oakland", "Mexican", 2, True, True, True),
    ("Legume", "Point Breeze", "American", 4, True, True, True),
    ("Point Brugge", "Point Breeze", "Belgian", 3, True, False, True),
    ("Kelly's Bar Point Breeze", "Point Breeze", "American", 2, True, False, False),
    ("Whitfield", "East Liberty", "American", 3, True, True, True),
    ("Condado Tacos East Lib", "East Liberty", "Mexican", 1, True, True, True),
    ("Smoke BBQ East Liberty", "East Liberty", "BBQ", 2, False, False, True),
    ("Tako", "Downtown", "Mexican", 2, True, True, True),
    ("Bakersfield", "East Liberty", "Mexican", 2, True, False, True),
    ("Dish Osteria", "South Side", "Italian", 3, True, False, True),
    ("Mallorca", "South Side", "Spanish", 3, True, False, True),
    ("Double Wide Grill", "South Side", "BBQ", 2, True, False, True),
    ("Fat Head's Saloon", "South Side", "American", 2, True, False, False),
    ("Nakama", "South Side", "Japanese", 3, True, False, True),
    ("Piper's Pub", "South Side", "American", 2, True, False, True),
    ("Grand Concourse", "South Side", "Seafood", 4, True, False, True),
    ("Hofbrauhaus Pittsburgh", "South Side", "American", 2, True, False, False),
    ("Burgatory North Shore", "North Shore", "American", 2, True, True, True),
    ("Jerome Bettis Grille", "North Shore", "American", 2, False, False, True),
    ("Yard House North Shore", "North Shore", "American", 2, True, True, True),
    ("Monterey Bay Fish Grotto", "Mount Washington", "Seafood", 4, True, False, True),
    ("Altius", "Mount Washington", "American", 4, True, True, True),
    ("Shiloh Grill", "Mount Washington", "American", 2, True, False, False),
    ("The Grandview Saloon", "Mount Washington", "American", 3, True, False, True),
    ("Park Bruges", "Highland Park", "Belgian", 3, True, False, True),
    ("E2", "Highland Park", "American", 3, True, True, True),
    ("Park House Highland", "Highland Park", "Cafe", 1, True, True, True),
    ("Kiin Kaow", "Bloomfield", "Thai", 2, True, True, True),
    ("HowLee", "Bloomfield", "Chinese", 2, True, False, True),
    ("Quiet Storm", "Garfield", "Vegan", 1, True, True, True),
    ("Coca Cafe Annex", "Polish Hill", "Cafe", 1, True, True, False),
    ("Gooski's Kitchen", "Polish Hill", "American", 1, True, False, False),
    ("Allegheny Wine Mixer", "Lawrenceville", "American", 2, True, True, True),
    ("Smiling Banana Leaf", "Bloomfield", "Thai", 1, True, True, True),
    ("Kiin Lao", "East Liberty", "Thai", 2, True, True, True),
    ("Pusadee's Garden", "Bloomfield", "Thai", 3, True, True, True),
    ("Udipi Cafe", "Squirrel Hill", "Indian", 1, True, True, True),
    ("Tamarind", "East Liberty", "Indian", 2, True, True, True),
    ("Seoul Kitchen", "Oakland", "Korean", 2, True, False, True),
    ("Goejei", "Squirrel Hill", "Korean", 2, True, True, True),
    ("Kaya", "Strip District", "Caribbean", 3, True, True, True),
    ("Eleven", "Strip District", "American", 4, True, False, True),
    ("Bitter Ends Garden", "Lawrenceville", "Vegan", 2, True, True, True),
    ("The Milk Shake Factory", "South Side", "Cafe", 1, True, True, False),
    ("Pamela's Shadyside", "Shadyside", "Diner", 1, True, False, False),
    ("Harris Grill", "Shadyside", "American", 2, True, False, False),
    ("Noodlehead Shadyside", "Shadyside", "Thai", 1, True, True, True),
    ("Sushi Fuku Shadyside", "Shadyside", "Japanese", 1, True, False, True),
    ("Mad Mex Shadyside", "Shadyside", "Mexican", 2, True, True, True),
    ("Girasole", "Shadyside", "Italian", 3, True, False, True),
    ("Lucca", "Shadyside", "Italian", 3, True, False, True),
    ("Industry Public House", "Lawrenceville", "American", 2, True, False, True),
    ("Smoke BBQ South Side", "South Side", "BBQ", 2, False, False, True),
    ("Pizza Perfect Oakland", "Oakland", "Pizza", 1, True, False, False),
    ("Aiello's Pizza", "Squirrel Hill", "Pizza", 1, True, False, False),
    ("Beto's Pizza", "South Side", "Pizza", 1, True, False, False),
    ("Ephesus Pizza", "Bloomfield", "Pizza", 1, True, False, True),
    ("The Porch at Schenley", "Oakland", "American", 2, True, True, True),
    ("Social House Seven", "East Liberty", "Korean", 3, True, False, True),
    ("Chengdu Gourmet Aspinwall", "Highland Park", "Chinese", 2, True, False, True),
    ("Del's Bar", "Squirrel Hill", "American", 1, True, False, False),
]

# Quiet Storm is listed as Garfield which is not in NEIGHBORHOODS; remap below.
NEIGHBORHOOD_ALIAS = {"Garfield": "Bloomfield"}

CUISINES = sorted({row[2] for row in RESTAURANTS})
AGE_BANDS = ("18-24", "25-34", "35-44", "45-54", "55+")
DIETARY = ("none", "none", "none", "none", "vegetarian", "vegan", "gluten_free")


def _jitter(rng: np.random.Generator, lat: float, lon: float) -> tuple[float, float]:
    return float(lat + rng.normal(0, 0.003)), float(lon + rng.normal(0, 0.003))


def build_items(rng: np.random.Generator) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for i, spec in enumerate(RESTAURANTS, start=1):
        name, hood, cuisine, price, veg, vegan, gf = spec
        hood = NEIGHBORHOOD_ALIAS.get(hood, hood)
        lat0, lon0 = NEIGHBORHOODS[hood]
        lat, lon = _jitter(rng, lat0, lon0)
        popularity_prior = float(rng.lognormal(mean=3.2, sigma=0.7))
        n_reviews = int(max(8, min(4200, popularity_prior * 40)))
        rating = float(np.clip(rng.normal(4.15, 0.35), 3.2, 4.9))
        rows.append(
            {
                "item_id": f"r{i:04d}",
                "name": name,
                "neighborhood": hood,
                "cuisine": cuisine,
                "price_tier": price,
                "avg_rating": round(rating, 2),
                "n_reviews": n_reviews,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "vegetarian_ok": veg,
                "vegan_ok": vegan,
                "gluten_free_ok": gf,
            }
        )
    return pl.DataFrame(rows)


def build_users(rng: np.random.Generator) -> pl.DataFrame:
    hoods = list(NEIGHBORHOODS)
    rows: list[dict[str, object]] = []
    for i in range(1, N_USERS + 1):
        n_pref = int(rng.integers(1, 3))
        prefs = rng.choice(CUISINES, size=n_pref, replace=False)
        rows.append(
            {
                "user_id": f"u{i:04d}",
                "neighborhood": str(rng.choice(hoods)),
                "age_band": str(rng.choice(AGE_BANDS, p=[0.18, 0.34, 0.24, 0.14, 0.10])),
                "cuisine_prefs": "|".join(sorted(str(p) for p in prefs)),
                "price_pref": int(rng.choice([1, 2, 3, 4], p=[0.18, 0.42, 0.28, 0.12])),
                "dietary": str(rng.choice(DIETARY)),
            }
        )
    return pl.DataFrame(rows)


def _dietary_ok(dietary: str, item: dict[str, object]) -> bool:
    if dietary == "none":
        return True
    if dietary == "vegetarian":
        return bool(item["vegetarian_ok"])
    if dietary == "vegan":
        return bool(item["vegan_ok"])
    if dietary == "gluten_free":
        return bool(item["gluten_free_ok"])
    return True


def _affinity(
    rng: np.random.Generator,
    user: dict[str, object],
    item: dict[str, object],
    pop_weight: float,
) -> float:
    prefs = str(user["cuisine_prefs"]).split("|")
    cuisine_hit = 1.0 if item["cuisine"] in prefs else 0.18
    price_pen = math.exp(-0.55 * abs(int(user["price_pref"]) - int(item["price_tier"])))
    same_hood = 2.4 if user["neighborhood"] == item["neighborhood"] else 1.0
    diet = 1.0 if _dietary_ok(str(user["dietary"]), item) else 0.05
    rating = (float(item["avg_rating"]) / 5.0) ** 1.4
    noise = float(rng.lognormal(0.0, 0.25))
    return cuisine_hit * price_pen * same_hood * diet * rating * (pop_weight**0.35) * noise


def build_interactions(
    rng: np.random.Generator,
    users: pl.DataFrame,
    items: pl.DataFrame,
) -> pl.DataFrame:
    user_rows = users.to_dicts()
    item_rows = items.to_dicts()
    pop = np.array([max(1, int(r["n_reviews"])) for r in item_rows], dtype=np.float64)
    pop_w = pop / pop.max()

    rows: list[dict[str, object]] = []

    # ~8% true cold-start users (no events). ~6% almost-cold (1-2 events).
    n_cold = 18
    n_almost = 14
    order = rng.permutation(len(user_rows))
    cold = set(order[:n_cold].tolist())
    almost = set(order[n_cold : n_cold + n_almost].tolist())

    # Two catalog items stay unseen (item cold start).
    held_out_items = {item_rows[-1]["item_id"], item_rows[-2]["item_id"]}

    for u_idx, user in enumerate(user_rows):
        if u_idx in cold:
            continue
        scores = np.array(
            [_affinity(rng, user, item, pop_w[j]) for j, item in enumerate(item_rows)],
            dtype=np.float64,
        )
        for j, item in enumerate(item_rows):
            if item["item_id"] in held_out_items:
                scores[j] = 0.0
        if scores.sum() <= 0:
            continue
        probs = scores / scores.sum()
        cutoff = datetime(2024, 4, 1, tzinfo=UTC)
        train_span = int((cutoff - TRAIN_START).total_seconds())
        valid_span = int((VALID_END - cutoff).total_seconds())

        if u_idx in almost:
            n_train = int(rng.integers(1, 3))
            n_holdout = 0
        else:
            n_train = int(np.clip(rng.lognormal(2.35, 0.35), 6, 28))
            n_holdout = int(rng.integers(1, 4))

        n_pick = min(len(item_rows) - len(held_out_items), n_train + n_holdout)
        chosen = rng.choice(len(item_rows), size=n_pick, replace=False, p=probs)

        uid = str(user["user_id"])

        def emit(
            item_idx: int, ts: datetime, force_positive: bool = False, user_id: str = uid
        ) -> None:
            item = item_rows[int(item_idx)]
            if force_positive:
                event = "visit" if rng.random() < 0.45 else "save"
            else:
                draw = float(rng.random())
                if draw < 0.55:
                    event = "view"
                elif draw < 0.82:
                    event = "save"
                else:
                    event = "visit"
            rating = None
            if event == "visit":
                rating = float(np.clip(rng.normal(4.3, 0.5), 3.0, 5.0))
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item["item_id"],
                    "event": event,
                    "rating": None if rating is None else round(rating, 1),
                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

        train_idx = chosen[:n_train]
        hold_idx = chosen[n_train : n_train + n_holdout]
        for item_idx in train_idx:
            ts = TRAIN_START + timedelta(seconds=int(rng.integers(0, max(train_span, 1))))
            emit(int(item_idx), ts)
        for item_idx in hold_idx:
            ts = cutoff + timedelta(seconds=int(rng.integers(0, max(valid_span, 1))))
            emit(int(item_idx), ts, force_positive=True)

    frame = pl.DataFrame(rows).unique(subset=["user_id", "item_id", "event", "timestamp"])
    return frame.sort("timestamp")


def main() -> int:
    out = Path(__file__).resolve().parents[1] / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    items = build_items(rng)
    users = build_users(rng)
    interactions = build_interactions(rng, users, items)
    users.write_csv(out / "users.csv")
    items.write_csv(out / "items.csv")
    interactions.write_csv(out / "interactions.csv")
    print(
        f"wrote {users.height} users, {items.height} items, "
        f"{interactions.height} interactions -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
