"""Pydantic contracts for the three rec-sys tables."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Dietary = Literal["none", "vegetarian", "vegan", "gluten_free"]
EventType = Literal["view", "save", "visit"]


class UserRecord(BaseModel):
    user_id: str
    neighborhood: str
    age_band: str
    cuisine_prefs: tuple[str, ...]
    price_pref: int = Field(ge=1, le=4)
    dietary: Dietary

    @field_validator("cuisine_prefs", mode="before")
    @classmethod
    def _split_prefs(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(p for p in value.split("|") if p)
        return value


class ItemRecord(BaseModel):
    item_id: str
    name: str
    neighborhood: str
    cuisine: str
    price_tier: int = Field(ge=1, le=4)
    avg_rating: float = Field(ge=0.0, le=5.0)
    n_reviews: int = Field(ge=0)
    lat: float
    lon: float
    vegetarian_ok: bool
    vegan_ok: bool
    gluten_free_ok: bool


class InteractionRecord(BaseModel):
    user_id: str
    item_id: str
    event: EventType
    rating: float | None = Field(default=None, ge=1.0, le=5.0)
    timestamp: datetime
