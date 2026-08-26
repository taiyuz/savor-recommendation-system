"""Standard ranking metrics. Binary relevance, no invented names."""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    hits = sum(1 for item_id in ranked[:k] if item_id in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    score = 0.0
    for rank, item_id in enumerate(ranked[:k], start=1):
        if item_id in relevant:
            score += 1.0 / math.log2(rank + 1)
        return score


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    ideal = dcg_at_k(list(relevant), relevant, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / ideal


def coverage_at_k(recommendations: dict[str, list[str]], catalog_size: int, k: int) -> float:
    if catalog_size <= 0:
        return 0.0
    recommended = {item_id for ranked in recommendations.values() for item_id in ranked[:k]}
    return len(recommended) / catalog_size
