from __future__ import annotations

import polars as pl
import pytest

from savor.config import SPLIT_CUTOFF
from savor.data.loader import Catalog, load_catalog
from savor.data.splits import temporal_split


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return load_catalog()


@pytest.fixture(scope="session")
def split_frames(catalog: Catalog) -> tuple[pl.DataFrame, pl.DataFrame]:
    return temporal_split(catalog.interactions, SPLIT_CUTOFF)


@pytest.fixture(scope="session")
def train_catalog(catalog: Catalog, split_frames: tuple[pl.DataFrame, pl.DataFrame]) -> Catalog:
    train, _valid = split_frames
    return Catalog(
        users=catalog.users,
        items=catalog.items,
        interactions=train,
        source_dir=catalog.source_dir,
    )


@pytest.fixture(scope="session")
def valid_catalog(catalog: Catalog, split_frames: tuple[pl.DataFrame, pl.DataFrame]) -> Catalog:
    _train, valid = split_frames
    return Catalog(
        users=catalog.users,
        items=catalog.items,
        interactions=valid,
        source_dir=catalog.source_dir,
    )
