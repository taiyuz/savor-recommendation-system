"""Paths and modeling constants. Data lives on disk so the stack runs offline."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

EVENT_WEIGHTS: dict[str, float] = {"view": 1.0, "save": 3.0, "visit": 5.0}
POSITIVE_EVENTS: frozenset[str] = frozenset({"save", "visit"})

# Train = interactions strictly before this instant. Valid = on or after.
SPLIT_CUTOFF = datetime(2024, 4, 1, 0, 0, 0, tzinfo=UTC)

DEFAULT_K = 10
CANDIDATE_K = 40
LATENT_DIM = 16
RANDOM_SEED = 42
NEGATIVE_RATIO = 4
MAX_RANK_ITER = 80


def resolve_data_dir() -> Path:
    """Prefer SAVOR_DATA_DIR, then cwd, then the repo checkout next to src/."""
    env = os.environ.get("SAVOR_DATA_DIR")
    if env:
        return Path(env)

    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "data" / "sample",
        here.parents[2] / "data" / "sample",
    ]
    for path in candidates:
        if (path / "users.csv").is_file():
            return path
    searched = ", ".join(str(c) for c in candidates)
    msg = f"Could not find data/sample/users.csv. Looked in: {searched}"
    raise FileNotFoundError(msg)
