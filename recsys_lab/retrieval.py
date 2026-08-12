"""Context-free weighted-SVD candidate retrieval from frozen rating history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalModel:
    user_factors: np.ndarray
    item_factors: np.ndarray
    popularity_z: np.ndarray
    seen: np.ndarray
    rank: int

    def all_scores(self, user: int) -> np.ndarray:
        collaborative = self.item_factors @ self.user_factors[user]
        scale = np.std(collaborative) + 1e-8
        return collaborative / scale + 0.08 * self.popularity_z

    def candidates(self, user: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        score = self.all_scores(user)
        score = score.copy()
        score[self.seen[user]] = -np.inf
        k = min(k, int(np.isfinite(score).sum()))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        # This lab already performs an exact scan, so a full lexicographic sort makes score ties deterministic by ascending item ID, including at K.
        idx = np.lexsort((np.arange(len(score)), -score))
        idx = idx[np.isfinite(score[idx])][:k]
        return idx, score[idx]


def randomized_svd(
    matrix: np.ndarray,
    rank: int,
    seed: int = 20260811,
    oversample: int = 8,
    power_iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halko-style randomized SVD implemented only with NumPy."""
    rng = np.random.default_rng(seed)
    width = min(rank + oversample, min(matrix.shape))
    omega = rng.normal(size=(matrix.shape[1], width))
    q, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(power_iterations):
        q, _ = np.linalg.qr(matrix @ (matrix.T @ q), mode="reduced")
    small = q.T @ matrix
    u_small, singular, vt = np.linalg.svd(small, full_matrices=False)
    u = q @ u_small
    return u[:, :rank], singular[:rank], vt[:rank]


def fit_retrieval(
    interactions: np.ndarray,
    seen: np.ndarray | None = None,
    rank: int = 24,
    seed: int = 20260811,
) -> RetrievalModel:
    """Fit a positive-preference latent-factor retriever.

    MovieLens ratings above 2.5 provide positive preference mass. All historical ratings—including low ones—remain excluded at serving through ``seen``. Each row is log-scaled, item-frequency weighted, and L2 normalized before a low-rank approximation. The production analogue would index item factors in an ANN service; this compact benchmark scores the full catalog exactly.
    """
    interactions = np.asarray(interactions, dtype=float)
    positive = interactions > 0
    if seen is None:
        seen = positive
    seen = np.asarray(seen, dtype=bool)
    if seen.shape != interactions.shape:
        raise ValueError("seen mask must have the same shape as interactions")
    document_frequency = positive.sum(axis=0)
    idf = np.log((interactions.shape[0] + 1.0) / (document_frequency + 1.0)) + 1.0
    weighted = np.log1p(interactions) * idf[None, :]
    row_norm = np.linalg.norm(weighted, axis=1, keepdims=True)
    weighted = weighted / np.maximum(row_norm, 1e-12)
    u, singular, vt = randomized_svd(weighted, rank=rank, seed=seed)
    root = np.sqrt(np.maximum(singular, 1e-12))
    user_factors = u * root[None, :]
    item_factors = vt.T * root[None, :]
    pop = np.log1p(document_frequency.astype(float))
    popularity_z = (pop - pop.mean()) / (pop.std() + 1e-8)
    return RetrievalModel(
        user_factors=user_factors,
        item_factors=item_factors,
        popularity_z=popularity_z,
        seen=seen,
        rank=rank,
    )
