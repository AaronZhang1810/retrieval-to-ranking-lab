"""Deterministic MovieLens-compatible fixtures used by the offline test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from recsys_lab.data import MovieLensData, TemporalSplit, chronological_split, load_movielens
from recsys_lab.ranking import FeatureStore, estimate_feature_store
from recsys_lab.retrieval import RetrievalModel, fit_retrieval


def write_movielens_fixture(
    directory: Path,
    *,
    n_users: int = 30,
    n_items: int = 48,
) -> Path:
    """Write small ``u.data`` and ``u.item`` files with MovieLens 100K schemas.

    Each user has 20--25 timestamped ratings. The suffix always contains at least two rating levels and a 4/5-star observation, while history lengths vary enough to exercise activity slices in the end-to-end experiment. Rows are intentionally written in reverse chronological blocks; production code must sort them rather than rely on file order.
    """

    if n_users < 15:
        raise ValueError("the fixture needs at least 15 users for user-disjoint splits")
    if n_items < 32:
        raise ValueError("the fixture needs at least 32 movies")
    directory.mkdir(parents=True, exist_ok=True)

    item_lines: list[str] = []
    for item in range(1, n_items + 1):
        title = f"Fixture Movie {item:03d} ({1980 + item % 30})"
        if item == 2:
            title = "Am\u00e9lie Fixture (2001)"
        release = f"01-Jan-{1980 + item % 30}"
        flags = ["0"] * 19
        flags[1 + item % 18] = "1"
        if item % 7 == 0:
            flags[1 + (item + 4) % 18] = "1"
        fields = [
            str(item),
            title,
            release,
            "",
            f"https://example.invalid/title/{item}",
            *flags,
        ]
        item_lines.append("|".join(fields))
    (directory / "u.item").write_text("\n".join(item_lines) + "\n", encoding="latin-1")

    events: list[tuple[int, int, int, int]] = []
    base_timestamp = 880_000_000
    for user_offset in range(n_users):
        user_id = 700_001 + 13 * user_offset  # distinctive, non-consecutive IDs test remapping
        n_ratings = 20 + user_offset % 6
        for step in range(n_ratings):
            item_id = 1 + (user_offset * 7 + step) % n_items
            rating = 1 + (step - (n_ratings - 5)) % 5
            timestamp = base_timestamp + user_offset * 10_000 + step * 10
            events.append((user_id, item_id, rating, timestamp))

    # The official file is not guaranteed to be time ordered. Reverse blocks make a parser/split that accidentally trusts input order fail loudly.
    events.reverse()
    lines = [f"{user}\t{item}\t{rating}\t{timestamp}" for user, item, rating, timestamp in events]
    (directory / "u.data").write_text("\n".join(lines) + "\n", encoding="ascii")
    return directory


def load_fixture(directory: Path) -> tuple[MovieLensData, TemporalSplit]:
    """Write, load, and split the standard fixture."""

    data = load_movielens(write_movielens_fixture(directory))
    temporal = chronological_split(data)
    return data, temporal


def fit_fixture_pipeline(
    directory: Path,
    *,
    rank: int = 8,
    seed: int = 41,
) -> tuple[MovieLensData, TemporalSplit, FeatureStore, RetrievalModel]:
    """Return the deterministic data, split, feature store, and retriever."""

    data, temporal = load_fixture(directory)
    retrieval = fit_retrieval(
        temporal.history_values,
        seen=temporal.history_seen,
        rank=rank,
        seed=seed,
    )
    store = estimate_feature_store(data, temporal)
    return data, temporal, store, retrieval


def clone_temporal(temporal: TemporalSplit, **updates: np.ndarray | int | float) -> TemporalSplit:
    """Copy a frozen temporal split while replacing selected arrays."""

    values = {
        "history_values": temporal.history_values,
        "history_ratings": temporal.history_ratings,
        "history_seen": temporal.history_seen,
        "history_timestamps": temporal.history_timestamps,
        "holdout_ratings": temporal.holdout_ratings,
        "holdout_mask": temporal.holdout_mask,
        "holdout_timestamps": temporal.holdout_timestamps,
        "cutoff_timestamps": temporal.cutoff_timestamps,
        "eligible_users": temporal.eligible_users,
        "excluded_users": temporal.excluded_users,
        "holdout_fraction": temporal.holdout_fraction,
    }
    values.update(updates)
    return TemporalSplit(**values)
