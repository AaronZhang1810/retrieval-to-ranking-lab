"""Ranking metrics and paired user-cluster uncertainty."""

from __future__ import annotations

import numpy as np


def dcg(relevance: np.ndarray) -> float:
    relevance = np.asarray(relevance, dtype=float)
    if relevance.size == 0:
        return 0.0
    gains = np.exp2(relevance) - 1.0
    discount = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(gains / discount))


def ndcg_at_k(ranked_items: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    ranked_items = np.asarray(ranked_items, dtype=int)[:k]
    observed = dcg(relevance[ranked_items])
    ideal = dcg(np.sort(relevance)[::-1][:k])
    return observed / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_items: np.ndarray, relevant_items: np.ndarray, k: int) -> float:
    target = set(np.asarray(relevant_items, dtype=int).tolist())
    if not target:
        return 0.0
    found = len(target.intersection(np.asarray(ranked_items, dtype=int)[:k].tolist()))
    return found / len(target)


def reciprocal_rank(ranked_items: np.ndarray, relevant_items: np.ndarray, k: int = 10) -> float:
    target = set(np.asarray(relevant_items, dtype=int).tolist())
    for position, item in enumerate(np.asarray(ranked_items, dtype=int)[:k], start=1):
        if int(item) in target:
            return 1.0 / position
    return 0.0


def intralist_diversity(ranked_items: np.ndarray, item_genre: np.ndarray, k: int = 10) -> float:
    items = np.asarray(ranked_items, dtype=int)[:k]
    if len(items) < 2:
        return 0.0
    genres = item_genre[items]
    norms = np.linalg.norm(genres, axis=1)
    similarities = (genres @ genres.T) / np.maximum(norms[:, None] * norms[None, :], 1e-12)
    upper = similarities[np.triu_indices(len(items), k=1)]
    return float(np.mean(1.0 - upper))


def paired_cluster_bootstrap(
    user_ids: np.ndarray,
    treatment: np.ndarray,
    control: np.ndarray,
    n_bootstrap: int = 3000,
    seed: int = 20260814,
) -> dict[str, float]:
    """Percentile CI after aggregating repeated queries within each user."""
    user_ids = np.asarray(user_ids, dtype=int)
    difference = np.asarray(treatment, dtype=float) - np.asarray(control, dtype=float)
    unique = np.unique(user_ids)
    cluster_mean = np.array([difference[user_ids == user].mean() for user in unique])
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(unique), size=(n_bootstrap, len(unique)), replace=True)
    estimates = cluster_mean[draws].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(cluster_mean.mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "bootstrap_fraction_positive": float(np.mean(estimates > 0.0)),
        "clusters": int(len(unique)),
        "replicates": int(n_bootstrap),
    }
