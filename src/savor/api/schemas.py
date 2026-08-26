"""HTTP request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendedItem(BaseModel):
    item_id: str
    name: str
    cuisine: str
    neighborhood: str
    price_tier: int
    score: float
    sources: list[str]


class RecommendResponse(BaseModel):
    user_id: str
    k: int
    strategy: str
    items: list[RecommendedItem]


class HealthResponse(BaseModel):
    status: str = "ok"
    n_users: int
    n_items: int
    n_interactions: int


class ErrorResponse(BaseModel):
    detail: str = Field(min_length=1)
