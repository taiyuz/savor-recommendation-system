"""Time-based split. Never random-split interactions for rec-sys eval."""

from __future__ import annotations

from datetime import datetime

import polars as pl


def temporal_split(
    interactions: pl.DataFrame,
    cutoff: datetime,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (train, valid) where train timestamps are strictly before cutoff.

    A random row split would leak a user's future visit into training features
    (popularity, item-item neighbors, SVD factors). Time is the unit of leakage.
    """
    if "timestamp" not in interactions.columns:
        msg = "interactions must include a timestamp column"
        raise ValueError(msg)

    ts = interactions["timestamp"]
    if ts.dtype != pl.Datetime:
        interactions = interactions.with_columns(
            pl.col("timestamp").str.to_datetime(time_zone="UTC")
        )

    train = interactions.filter(pl.col("timestamp") < cutoff)
    valid = interactions.filter(pl.col("timestamp") >= cutoff)
    return train, valid
