"""CPU gradient-boosting ranker. Pointwise logistic loss, tiny trees."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from savor.config import MAX_RANK_ITER, NEGATIVE_RATIO, POSITIVE_EVENTS, RANDOM_SEED
from savor.ranking.features import FEATURE_NAMES, FeatureBuilder
from savor.retrieval.candidates import Candidate, CandidateGenerator


class GradientBoostRanker:
    def __init__(self) -> None:
        self.model = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=MAX_RANK_ITER,
            learning_rate=0.08,
            min_samples_leaf=20,
            l2_regularization=0.4,
            random_state=RANDOM_SEED,
        )
        self.fitted = False

    def fit(
        self,
        *,
        interactions: pl.DataFrame,
        features: FeatureBuilder,
        generator: CandidateGenerator,
        rng: np.random.Generator,
    ) -> GradientBoostRanker:
        item_ids = list(features.items)
        positives: dict[str, set[str]] = {}
        if interactions.height:
            pos = interactions.filter(pl.col("event").is_in(sorted(POSITIVE_EVENTS)))
            if pos.height:
                grouped = pos.group_by("user_id").agg(pl.col("item_id").unique())
                positives = {str(u): {str(i) for i in items} for u, items in grouped.iter_rows()}

        xs: list[np.ndarray] = []
        ys: list[int] = []
        for user_id, pos_items in positives.items():
            if not pos_items:
                continue
            seen = generator.seen(user_id)
            negatives_pool = [i for i in item_ids if i not in seen]
            n_neg = min(len(negatives_pool), max(1, NEGATIVE_RATIO * len(pos_items)))
            neg_items = (
                list(rng.choice(negatives_pool, size=n_neg, replace=False))
                if negatives_pool
                else []
            )
            tt = generator.two_tower.scores_for_user(user_id)
            ii = generator.item_item.scores_for_user(user_id)
            pop = generator.popularity.scores
            for item_id, label in [(i, 1) for i in pos_items] + [(i, 0) for i in neg_items]:
                idx = generator.two_tower.item_index.get(item_id)
                cand = Candidate(
                    item_id=item_id,
                    two_tower_score=float(tt[idx]) if idx is not None else 0.0,
                    item_item_score=float(ii[idx]) if idx is not None else 0.0,
                    popularity_score=float(pop[idx]) if idx is not None else 0.0,
                )
                xs.append(features.vector(user_id, cand))
                ys.append(label)

        if not xs or len(set(ys)) < 2:
            self.fitted = False
            return self

        matrix = np.vstack(xs)
        self.model.fit(matrix, np.array(ys, dtype=np.int32))
        self.fitted = True
        return self

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if not self.fitted or matrix.size == 0:
            return np.zeros(matrix.shape[0], dtype=np.float64)
        return self.model.predict_proba(matrix)[:, 1]


def stack_features(
    user_id: str,
    candidates: list[Candidate],
    features: FeatureBuilder,
) -> np.ndarray:
    if not candidates:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([features.vector(user_id, c) for c in candidates])
