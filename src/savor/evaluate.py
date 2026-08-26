"""Offline eval on a temporal split. Metrics are labeled synthetic in reports."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from savor.config import DEFAULT_K, POSITIVE_EVENTS, SPLIT_CUTOFF
from savor.data.loader import Catalog, load_catalog
from savor.data.splits import temporal_split
from savor.metrics import coverage_at_k, ndcg_at_k, recall_at_k
from savor.pipeline import Recommender, ScoredItem


@dataclass(frozen=True)
class EvalReport:
    dataset: str
    k: int
    n_eval_users: int
    recall: float
    ndcg: float
    coverage: float
    popularity_recall: float
    popularity_ndcg: float
    popularity_coverage: float

    def as_table_rows(self) -> list[tuple[str, str, str, str]]:
        return [
            (
                "two-stage (item-item + SVD + HGB)",
                f"{self.recall:.3f}",
                f"{self.ndcg:.3f}",
                f"{self.coverage:.3f}",
            ),
            (
                "popularity baseline",
                f"{self.popularity_recall:.3f}",
                f"{self.popularity_ndcg:.3f}",
                f"{self.popularity_coverage:.3f}",
            ),
        ]


def held_out_positives(
    train: pl.DataFrame,
    valid: pl.DataFrame,
) -> dict[str, set[str]]:
    """Future save/visit pairs that never appeared in train for that user."""
    train_pairs: set[tuple[str, str]] = set()
    if train.height:
        train_pairs = {
            (str(u), str(i)) for u, i in train.select(["user_id", "item_id"]).iter_rows()
        }
    positives = valid.filter(pl.col("event").is_in(sorted(POSITIVE_EVENTS)))
    out: dict[str, set[str]] = {}
    if positives.height == 0:
        return out
    for user_id, item_id in positives.select(["user_id", "item_id"]).iter_rows():
        pair = (str(user_id), str(item_id))
        if pair in train_pairs:
            continue
        out.setdefault(str(user_id), set()).add(str(item_id))
    return out


def _ids(items: list[ScoredItem]) -> list[str]:
    return [row.item_id for row in items]


def evaluate(
    recommender: Recommender,
    train: Catalog,
    valid: Catalog,
    *,
    k: int = DEFAULT_K,
) -> EvalReport:
    labels = held_out_positives(train.interactions, valid.interactions)
    train_users = set(recommender.generator.history)
    eval_users = sorted(uid for uid in labels if uid in train_users and labels[uid])

    recs: dict[str, list[str]] = {}
    pop_recs: dict[str, list[str]] = {}
    recalls: list[float] = []
    ndcgs: list[float] = []
    pop_recalls: list[float] = []
    pop_ndcgs: list[float] = []

    for user_id in eval_users:
        relevant = labels[user_id]
        ranked = _ids(recommender.recommend(user_id, k=k, exclude_seen=True))
        popular = _ids(recommender.popularity_recommend(user_id, k=k, exclude_seen=True))
        recs[user_id] = ranked
        pop_recs[user_id] = popular
        recalls.append(recall_at_k(ranked, relevant, k))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        pop_recalls.append(recall_at_k(popular, relevant, k))
        pop_ndcgs.append(ndcg_at_k(popular, relevant, k))

    n_items = train.items.height
    n = len(eval_users) or 1
    return EvalReport(
        dataset="synthetic Pittsburgh restaurant sample (not production traffic)",
        k=k,
        n_eval_users=len(eval_users),
        recall=sum(recalls) / n if eval_users else 0.0,
        ndcg=sum(ndcgs) / n if eval_users else 0.0,
        coverage=coverage_at_k(recs, n_items, k),
        popularity_recall=sum(pop_recalls) / n if eval_users else 0.0,
        popularity_ndcg=sum(pop_ndcgs) / n if eval_users else 0.0,
        popularity_coverage=coverage_at_k(pop_recs, n_items, k),
    )


def run_offline_eval(k: int = DEFAULT_K) -> EvalReport:
    catalog = load_catalog()
    train_events, valid_events = temporal_split(catalog.interactions, SPLIT_CUTOFF)
    train = Catalog(
        users=catalog.users,
        items=catalog.items,
        interactions=train_events,
        source_dir=catalog.source_dir,
    )
    valid = Catalog(
        users=catalog.users,
        items=catalog.items,
        interactions=valid_events,
        source_dir=catalog.source_dir,
    )
    recommender = Recommender().fit(train)
    return evaluate(recommender, train, valid, k=k)
