"""Pairwise learning-to-rank from observed later MovieLens ratings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import MovieLensData, TemporalSplit
from .retrieval import RetrievalModel


FEATURE_NAMES = (
    "retrieval affinity",
    "genre affinity",
    "Bayesian item mean",
    "log rating count",
    "release-year affinity",
    "popularity affinity",
)


@dataclass(frozen=True)
class FeatureStore:
    """User profiles and item statistics estimated from history only."""

    user_genre: np.ndarray
    user_year_z: np.ndarray
    user_popularity_z: np.ndarray
    item_bayesian_mean: np.ndarray
    item_log_count: np.ndarray
    item_year_z: np.ndarray
    item_popularity_z: np.ndarray
    global_mean_rating: float


@dataclass(frozen=True)
class RankDataset:
    differences: np.ndarray
    labels: np.ndarray
    query_users: np.ndarray
    rating_gaps: np.ndarray


@dataclass(frozen=True)
class PairwiseRanker:
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    l2: float
    blend_alpha: float
    train_loss: tuple[float, ...]

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.feature_mean) / self.feature_std

    def pure_score(self, features: np.ndarray) -> np.ndarray:
        """Unblended pairwise model score used for diagnostics."""

        return self.transform(features) @ self.weights

    def score(self, features: np.ndarray) -> np.ndarray:
        """Conservative challenger score blended with Stage-1 affinity."""

        transformed = self.transform(features)
        return (1.0 - self.blend_alpha) * transformed[:, 0] + self.blend_alpha * (
            transformed @ self.weights
        )

    def contributions(self, features: np.ndarray) -> np.ndarray:
        transformed = self.transform(features)
        parts = self.blend_alpha * transformed * self.weights[None, :]
        parts[:, 0] += (1.0 - self.blend_alpha) * transformed[:, 0]
        return parts


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() + 1e-8)


def estimate_feature_store(
    data: MovieLensData,
    temporal: TemporalSplit,
    prior_count: float = 20.0,
    profile_prior: float = 5.0,
) -> FeatureStore:
    """Estimate point-in-history features without held-out ratings."""

    seen = temporal.history_seen
    ratings = temporal.history_ratings.astype(float)
    item_count = seen.sum(axis=0).astype(float)
    item_sum = ratings.sum(axis=0)
    global_mean = float(item_sum.sum() / max(item_count.sum(), 1.0))
    item_bayesian_mean = (item_sum + prior_count * global_mean) / (item_count + prior_count)
    item_log_count = np.log1p(item_count)
    item_popularity_z = _zscore(item_log_count)

    year = data.item_year.astype(float).copy()
    finite_year = np.isfinite(year)
    fill = float(np.median(year[finite_year])) if finite_year.any() else 1990.0
    year[~finite_year] = fill
    item_year_z = _zscore(year)

    positive = temporal.history_values.astype(float)
    mass = positive.sum(axis=1)
    fallback = seen.astype(float)
    use_fallback = mass <= 1e-12
    positive[use_fallback] = fallback[use_fallback]
    mass = positive.sum(axis=1)

    global_genre = (positive.sum(axis=0)[:, None] * data.item_genre).sum(axis=0)
    global_genre /= global_genre.sum() + 1e-12
    user_genre = (
        positive @ data.item_genre + profile_prior * global_genre[None, :]
    ) / (mass[:, None] + profile_prior)
    user_genre /= user_genre.sum(axis=1, keepdims=True) + 1e-12
    global_year = float(np.average(item_year_z, weights=item_count + 1e-6))
    global_popularity = float(np.average(item_popularity_z, weights=item_count + 1e-6))
    user_year_z = (positive @ item_year_z + profile_prior * global_year) / (mass + profile_prior)
    user_popularity_z = (
        positive @ item_popularity_z + profile_prior * global_popularity
    ) / (mass + profile_prior)
    return FeatureStore(
        user_genre=user_genre,
        user_year_z=user_year_z,
        user_popularity_z=user_popularity_z,
        item_bayesian_mean=item_bayesian_mean,
        item_log_count=item_log_count,
        item_year_z=item_year_z,
        item_popularity_z=item_popularity_z,
        global_mean_rating=global_mean,
    )


def item_features(
    data: MovieLensData,
    store: FeatureStore,
    retrieval: RetrievalModel,
    user: int,
    items: np.ndarray,
    retrieval_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Build transparent, history-derived reranking features."""

    items = np.asarray(items, dtype=int)
    if retrieval_scores is None:
        retrieval_scores = retrieval.all_scores(user)[items]
    genre_affinity = data.item_genre[items] @ store.user_genre[user]
    year_affinity = -np.abs(store.item_year_z[items] - store.user_year_z[user])
    popularity_affinity = -np.abs(
        store.item_popularity_z[items] - store.user_popularity_z[user]
    )
    return np.column_stack(
        [
            retrieval_scores,
            genre_affinity,
            store.item_bayesian_mean[items],
            store.item_log_count[items],
            year_affinity,
            popularity_affinity,
        ]
    )


def _observed_features(
    data: MovieLensData,
    temporal: TemporalSplit,
    store: FeatureStore,
    retrieval: RetrievalModel,
    users: np.ndarray,
    candidate_k: int | None = None,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for user in np.asarray(users, dtype=int):
        items = np.flatnonzero(temporal.holdout_mask[user])
        if candidate_k is not None:
            candidates, _ = retrieval.candidates(int(user), candidate_k)
            items = items[np.isin(items, candidates, assume_unique=False)]
        if len(items) < 2:
            continue
        ratings = temporal.holdout_ratings[user, items].astype(float)
        raw = item_features(data, store, retrieval, int(user), items)
        rows.append((int(user), items, ratings, raw))
    return rows


def feature_scaler(
    data: MovieLensData,
    temporal: TemporalSplit,
    store: FeatureStore,
    retrieval: RetrievalModel,
    train_users: np.ndarray,
    candidate_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaling on observed outcomes that passed the Stage-1 gate."""

    rows = _observed_features(data, temporal, store, retrieval, train_users, candidate_k)
    stacked = np.vstack([row[3] for row in rows])
    return stacked.mean(axis=0), np.maximum(stacked.std(axis=0), 1e-6)


def make_pairwise_dataset(
    data: MovieLensData,
    temporal: TemporalSplit,
    store: FeatureStore,
    retrieval: RetrievalModel,
    users: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    candidate_k: int,
    max_pairs_per_user: int = 80,
    minimum_rating_gap: float = 1.0,
    seed: int = 20260812,
) -> RankDataset:
    """Construct observed higher-rating versus lower-rating comparisons."""

    rng = np.random.default_rng(seed)
    differences: list[np.ndarray] = []
    labels: list[int] = []
    query_users: list[int] = []
    gaps: list[float] = []
    for user, _, ratings, raw in _observed_features(
        data, temporal, store, retrieval, users, candidate_k
    ):
        standardized = (raw - feature_mean) / feature_std
        left, right = np.triu_indices(len(ratings), k=1)
        valid = np.flatnonzero(np.abs(ratings[left] - ratings[right]) >= minimum_rating_gap)
        if len(valid) > max_pairs_per_user:
            valid = np.sort(rng.choice(valid, size=max_pairs_per_user, replace=False))
        for pair_index in valid:
            a, b = int(left[pair_index]), int(right[pair_index])
            label = 1 if ratings[a] > ratings[b] else -1
            if rng.random() < 0.5:
                a, b, label = b, a, -label
            differences.append(standardized[a] - standardized[b])
            labels.append(label)
            query_users.append(user)
            gaps.append(abs(float(ratings[a] - ratings[b])))
    if not differences:
        raise ValueError("no observed rating pairs satisfy the configured minimum gap")
    return RankDataset(
        differences=np.asarray(differences, dtype=float),
        labels=np.asarray(labels, dtype=float),
        query_users=np.asarray(query_users, dtype=int),
        rating_gaps=np.asarray(gaps, dtype=float),
    )


def pairwise_loss_gradient(
    weights: np.ndarray,
    dataset: RankDataset,
    l2: float,
) -> tuple[float, np.ndarray]:
    margin = dataset.labels * (dataset.differences @ weights)
    loss = float(np.mean(np.logaddexp(0.0, -margin)) + 0.5 * l2 * np.dot(weights, weights))
    factor = -dataset.labels / (1.0 + np.exp(np.clip(margin, -40.0, 40.0)))
    gradient = dataset.differences.T @ factor / len(dataset.labels) + l2 * weights
    return loss, gradient


def fit_pairwise_ranker(
    train: RankDataset,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    l2: float = 0.01,
    learning_rate: float = 0.04,
    steps: int = 700,
) -> PairwiseRanker:
    """Fit a linear RankNet/Bradley–Terry model with full-batch Adam."""

    weights = np.zeros(train.differences.shape[1], dtype=float)
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    history: list[float] = []
    for step in range(1, steps + 1):
        loss, gradient = pairwise_loss_gradient(weights, train, l2)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.999**step)
        weights -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        if step == 1 or step % 25 == 0 or step == steps:
            history.append(loss)
    return PairwiseRanker(
        weights=weights,
        feature_mean=feature_mean,
        feature_std=feature_std,
        l2=l2,
        blend_alpha=1.0,
        train_loss=tuple(history),
    )


def pairwise_accuracy(model: PairwiseRanker, dataset: RankDataset) -> float:
    margin = dataset.labels * (dataset.differences @ model.weights)
    return float(np.mean(margin > 0.0))


def select_regularization(
    train: RankDataset,
    validation: RankDataset,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    grid: tuple[float, ...] = (0.001, 0.01, 0.05, 0.2),
) -> tuple[PairwiseRanker, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    models: list[PairwiseRanker] = []
    for l2 in grid:
        model = fit_pairwise_ranker(train, feature_mean, feature_std, l2=l2)
        accuracy = pairwise_accuracy(model, validation)
        rows.append({"l2": l2, "validation_pair_accuracy": accuracy})
        models.append(model)
    best = max(range(len(models)), key=lambda index: (rows[index]["validation_pair_accuracy"], -grid[index]))
    return models[best], rows
