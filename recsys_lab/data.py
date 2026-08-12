"""MovieLens 100K parsing and leakage-aware temporal splits.

The public repository does not redistribute MovieLens data. Run ``python download_data.py --accept-license`` before loading this module's default dataset path. Dataset terms and citation details live in DATA.md.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GENRES = (
    "Unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-100k"


@dataclass(frozen=True)
class MovieLensData:
    """Parsed numbered user records, movie metadata, and explicit ratings."""

    user_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    item_names: tuple[str, ...]
    item_year: np.ndarray
    item_genre: np.ndarray
    rating_users: np.ndarray
    rating_items: np.ndarray
    rating_values: np.ndarray
    rating_timestamps: np.ndarray
    source_fingerprint: str

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_ratings(self) -> int:
        return len(self.rating_values)


@dataclass(frozen=True)
class TemporalSplit:
    """Frozen per-user history and later observed-rating outcomes."""

    history_values: np.ndarray
    history_ratings: np.ndarray
    history_seen: np.ndarray
    history_timestamps: np.ndarray
    holdout_ratings: np.ndarray
    holdout_mask: np.ndarray
    holdout_timestamps: np.ndarray
    cutoff_timestamps: np.ndarray
    eligible_users: np.ndarray
    excluded_users: int
    holdout_fraction: float


def _combined_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _release_year(title: str, release_date: str) -> float:
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d{4})\s*$", release_date)
    return float(match.group(1)) if match else np.nan


def load_movielens(data_dir: str | Path = DEFAULT_DATA_DIR) -> MovieLensData:
    """Load MovieLens 100K's ``u.data`` and ``u.item`` files.

    The parser intentionally ignores demographic fields. User and item IDs are represented by numbered IDs; internal arrays are remapped to dense zero-based indices while the source identifiers are retained for local diagnostics. The absence of names is not treated as proof of anonymity.
    """

    data_dir = Path(data_dir)
    ratings_path = data_dir / "u.data"
    items_path = data_dir / "u.item"
    missing = [path.name for path in (ratings_path, items_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"MovieLens files missing from {data_dir}: {', '.join(missing)}. "
            "Run `python download_data.py --accept-license`."
        )

    item_rows: dict[int, tuple[str, float, np.ndarray]] = {}
    with items_path.open(encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 24:
                raise ValueError(f"u.item row {row_number} has {len(row)} fields; expected at least 24")
            item_id = int(row[0])
            genres = np.asarray([float(value) for value in row[5:24]], dtype=float)
            if genres.shape != (len(GENRES),):
                raise ValueError(f"u.item row {row_number} has an invalid genre vector")
            item_rows[item_id] = (row[1], _release_year(row[1], row[2]), genres)

    raw_ratings: list[tuple[int, int, float, int]] = []
    user_ids: set[int] = set()
    rated_item_ids: set[int] = set()
    with ratings_path.open(encoding="ascii", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 4:
                raise ValueError(f"u.data row {row_number} has {len(row)} fields; expected 4")
            user_id, item_id, rating, timestamp = int(row[0]), int(row[1]), float(row[2]), int(row[3])
            if item_id not in item_rows:
                raise ValueError(f"u.data row {row_number} references unknown movie {item_id}")
            if not 1.0 <= rating <= 5.0:
                raise ValueError(f"u.data row {row_number} has rating outside [1, 5]")
            raw_ratings.append((user_id, item_id, rating, timestamp))
            user_ids.add(user_id)
            rated_item_ids.add(item_id)

    ordered_users = sorted(user_ids)
    ordered_items = sorted(rated_item_ids)
    user_index = {identifier: index for index, identifier in enumerate(ordered_users)}
    item_index = {identifier: index for index, identifier in enumerate(ordered_items)}
    rating_users = np.asarray([user_index[row[0]] for row in raw_ratings], dtype=np.int32)
    rating_items = np.asarray([item_index[row[1]] for row in raw_ratings], dtype=np.int32)
    rating_values = np.asarray([row[2] for row in raw_ratings], dtype=float)
    rating_timestamps = np.asarray([row[3] for row in raw_ratings], dtype=np.int64)

    names = tuple(item_rows[item_id][0] for item_id in ordered_items)
    years = np.asarray([item_rows[item_id][1] for item_id in ordered_items], dtype=float)
    genre = np.vstack([item_rows[item_id][2] for item_id in ordered_items])
    genre /= np.maximum(genre.sum(axis=1, keepdims=True), 1.0)
    return MovieLensData(
        user_ids=tuple(str(identifier) for identifier in ordered_users),
        item_ids=tuple(str(identifier) for identifier in ordered_items),
        item_names=names,
        item_year=years,
        item_genre=genre,
        rating_users=rating_users,
        rating_items=rating_items,
        rating_values=rating_values,
        rating_timestamps=rating_timestamps,
        source_fingerprint=_combined_fingerprint((ratings_path, items_path)),
    )


def chronological_split(
    data: MovieLensData,
    holdout_fraction: float = 0.20,
    min_history: int = 15,
    min_holdout: int = 4,
) -> TemporalSplit:
    """Create a deterministic per-user history/outcome split.

    Ratings are ordered by timestamp, with movie index as a deterministic tie breaker because MovieLens records may share a one-second timestamp. A user enters ranker/evaluation splits only when the held-out suffix contains at least one relevant rating (>=4) and at least two distinct rating levels.
    """

    if not 0.05 <= holdout_fraction <= 0.50:
        raise ValueError("holdout_fraction must be between 0.05 and 0.50")
    shape = (data.n_users, data.n_items)
    history_ratings = np.zeros(shape, dtype=np.float32)
    history_seen = np.zeros(shape, dtype=bool)
    history_timestamps = np.zeros(shape, dtype=np.int64)
    holdout_ratings = np.zeros(shape, dtype=np.float32)
    holdout_mask = np.zeros(shape, dtype=bool)
    holdout_timestamps = np.zeros(shape, dtype=np.int64)
    cutoff_timestamps = np.zeros(data.n_users, dtype=np.int64)
    eligible: list[int] = []

    for user in range(data.n_users):
        rows = np.flatnonzero(data.rating_users == user)
        order = np.lexsort((data.rating_items[rows], data.rating_timestamps[rows]))
        rows = rows[order]
        n_ratings = len(rows)
        n_holdout = max(min_holdout, int(np.ceil(n_ratings * holdout_fraction)))
        n_holdout = min(n_holdout, n_ratings - min_history)
        if n_holdout < min_holdout:
            raise ValueError(f"user {data.user_ids[user]} has insufficient ratings for the split")
        history_rows, holdout_rows = rows[:-n_holdout], rows[-n_holdout:]
        history_items = data.rating_items[history_rows]
        later_items = data.rating_items[holdout_rows]
        history_seen[user, history_items] = True
        history_ratings[user, history_items] = data.rating_values[history_rows]
        history_timestamps[user, history_items] = data.rating_timestamps[history_rows]
        holdout_mask[user, later_items] = True
        holdout_ratings[user, later_items] = data.rating_values[holdout_rows]
        holdout_timestamps[user, later_items] = data.rating_timestamps[holdout_rows]
        cutoff_timestamps[user] = int(data.rating_timestamps[history_rows].max())

        later_values = data.rating_values[holdout_rows]
        if np.any(later_values >= 4.0) and len(np.unique(later_values)) >= 2:
            eligible.append(user)

    # Positive preference strength for weighted SVD; low ratings remain in the seen mask but do not become positive implicit-feedback mass.
    history_values = np.maximum(history_ratings.astype(float) - 2.5, 0.0)
    eligible_users = np.asarray(eligible, dtype=int)
    return TemporalSplit(
        history_values=history_values,
        history_ratings=history_ratings,
        history_seen=history_seen,
        history_timestamps=history_timestamps,
        holdout_ratings=holdout_ratings,
        holdout_mask=holdout_mask,
        holdout_timestamps=holdout_timestamps,
        cutoff_timestamps=cutoff_timestamps,
        eligible_users=eligible_users,
        excluded_users=data.n_users - len(eligible_users),
        holdout_fraction=holdout_fraction,
    )


def make_user_split(users: np.ndarray, seed: int = 20260810) -> dict[str, np.ndarray]:
    """Create disjoint user clusters for ranker train/validation/development test."""

    users = np.asarray(users, dtype=int)
    if len(users) < 15:
        raise ValueError("at least 15 eligible users are required")
    rng = np.random.default_rng(seed)
    order = users[rng.permutation(len(users))]
    n_train = int(round(0.62 * len(order)))
    n_validation = int(round(0.18 * len(order)))
    split = {
        "train": np.sort(order[:n_train]),
        "validation": np.sort(order[n_train : n_train + n_validation]),
        "test": np.sort(order[n_train + n_validation :]),
    }
    assert not (set(split["train"]) & set(split["validation"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["validation"]) & set(split["test"]))
    return split


def observed_relevance(ratings: np.ndarray) -> np.ndarray:
    """Map held-out 4/5-star ratings to graded gains 1/2; others to zero."""

    return np.maximum(np.asarray(ratings, dtype=float) - 3.0, 0.0)
