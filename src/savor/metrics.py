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


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant item, truncated at k.

    Unlike recall@k this is position-sensitive inside the window: a hit at rank 1
    scores 1.0, a hit at rank k scores 1/k, and a hit at rank k+1 scores 0.
    Unlike NDCG it ignores every hit after the first.
    """
    if not relevant or k <= 0:
        return 0.0
    for rank, item_id in enumerate(ranked[:k], start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def coverage_at_k(recommendations: dict[str, list[str]], catalog_size: int, k: int) -> float:
    if catalog_size <= 0:
        return 0.0
    recommended = {item_id for ranked in recommendations.values() for item_id in ranked[:k]}
    return len(recommended) / catalog_size


def cold_item_coverage_at_k(
    recommendations: dict[str, list[str]],
    cold_items: set[str],
    k: int,
) -> float:
    """Fraction of train-cold catalog items that appear in any top-k list.

    This is a diagnostic, not a quality score: collaborative retrieval has no
    train support for these ids. 0.0 means none were shown. A popularity pad
    of zero-score rows can still surface them; a leaked valid slice would give
    them real CF/popularity support and they would show up for the wrong reason.
    """
    if not cold_items or k <= 0:
        return 0.0
    shown = {item_id for ranked in recommendations.values() for item_id in ranked[:k]}
    return len(shown & cold_items) / len(cold_items)


def cold_label_fraction(labels: dict[str, set[str]], cold_items: set[str]) -> float:
    """Fraction of held-out positive items that have zero train support.

    Keeping these labels is what makes recall honest. Filtering them out (because
    CF cannot retrieve them) quietly inflates the metric.
    """
    positives = {item_id for items in labels.values() for item_id in items}
    if not positives:
        return 0.0
    return len(positives & cold_items) / len(positives)
