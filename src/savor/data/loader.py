"""Load and validate the checked-in synthetic catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from savor.config import resolve_data_dir
from savor.data.schema import InteractionRecord, ItemRecord, UserRecord


@dataclass(frozen=True)
class Catalog:
    users: pl.DataFrame
    items: pl.DataFrame
    interactions: pl.DataFrame
    source_dir: Path

    def user_ids(self) -> list[str]:
        return self.users["user_id"].to_list()

    def item_ids(self) -> list[str]:
        return self.items["item_id"].to_list()


def _read_csv(path: Path) -> pl.DataFrame:
    if not path.is_file():
        msg = f"missing catalog file: {path}"
        raise FileNotFoundError(msg)
    return pl.read_csv(path)


def _read_interactions(directory: Path) -> pl.DataFrame:
    parts = sorted(directory.glob("interactions_part*.csv"))
    if parts:
        return pl.concat([_read_csv(path) for path in parts], how="vertical")
    return _read_csv(directory / "interactions.csv")


def _validate_users(frame: pl.DataFrame) -> None:
    for row in frame.iter_rows(named=True):
        UserRecord.model_validate(row)


def _validate_items(frame: pl.DataFrame) -> None:
    bool_cols = ("vegetarian_ok", "vegan_ok", "gluten_free_ok")
    coerced = frame.with_columns([pl.col(c).cast(pl.Boolean) for c in bool_cols])
    for row in coerced.iter_rows(named=True):
        ItemRecord.model_validate(row)


def _validate_interactions(frame: pl.DataFrame) -> None:
    for row in frame.iter_rows(named=True):
        InteractionRecord.model_validate(row)


def load_catalog(data_dir: Path | None = None, *, validate: bool = True) -> Catalog:
    directory = data_dir or resolve_data_dir()
    users = _read_csv(directory / "users.csv")
    items = _read_csv(directory / "items.csv")
    interactions = _read_interactions(directory)
    interactions = interactions.with_columns(
        pl.col("timestamp").str.to_datetime(time_zone="UTC"),
        pl.col("rating").cast(pl.Float64),
    )
    items = items.with_columns(
        pl.col("vegetarian_ok").cast(pl.Boolean),
        pl.col("vegan_ok").cast(pl.Boolean),
        pl.col("gluten_free_ok").cast(pl.Boolean),
    )
    if validate:
        _validate_users(users)
        _validate_items(items)
        _validate_interactions(interactions)
    return Catalog(users=users, items=items, interactions=interactions, source_dir=directory)
