"""FastAPI serving surface for /recommend?user_id=."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from savor.api.schemas import HealthResponse, RecommendedItem, RecommendResponse
from savor.config import DEFAULT_K
from savor.data.loader import Catalog, load_catalog
from savor.pipeline import Recommender, ScoredItem


@lru_cache(maxsize=1)
def _catalog() -> Catalog:
    return load_catalog()


@lru_cache(maxsize=1)
def _recommender() -> Recommender:
    return Recommender().fit(_catalog())


def _serialize(user_id: str, k: int, strategy: str, items: list[ScoredItem]) -> RecommendResponse:
    return RecommendResponse(
        user_id=user_id,
        k=k,
        strategy=strategy,
        items=[
            RecommendedItem(
                item_id=row.item_id,
                name=row.name,
                cuisine=row.cuisine,
                neighborhood=row.neighborhood,
                price_tier=row.price_tier,
                score=round(row.score, 6),
                sources=list(row.sources),
            )
            for row in items
        ],
    )


def create_app() -> FastAPI:
    api = FastAPI(
        title="Savor recommendation API",
        version="0.1.0",
        description="Two-stage restaurant recommendations for the Savor product.",
    )

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        catalog = _catalog()
        return HealthResponse(
            status="ok",
            n_users=catalog.users.height,
            n_items=catalog.items.height,
            n_interactions=catalog.interactions.height,
        )

    @api.get("/recommend", response_model=RecommendResponse)
    def recommend(
        user_id: str = Query(..., min_length=1),
        k: int = Query(DEFAULT_K, ge=1, le=50),
    ) -> RecommendResponse:
        catalog = _catalog()
        if user_id not in set(catalog.user_ids()):
            raise HTTPException(status_code=404, detail=f"unknown user_id: {user_id}")
        rec = _recommender()
        strategy = "cold_start_popularity" if rec.generator.is_cold(user_id) else "two_stage"
        items = rec.recommend(user_id, k=k, exclude_seen=True)
        return _serialize(user_id, k, strategy, items)

    return api


app = create_app()
