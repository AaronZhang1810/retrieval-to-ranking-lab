from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from recsys_lab.data import make_user_split
from recsys_lab.experiment import _select_guarded_blend
from recsys_lab.ranking import (
    FEATURE_NAMES,
    PairwiseRanker,
    RankDataset,
    estimate_feature_store,
    feature_scaler,
    fit_pairwise_ranker,
    item_features,
    make_pairwise_dataset,
    pairwise_loss_gradient,
)
from tests.fixture_factory import clone_temporal, fit_fixture_pipeline


def rank_dataset(differences: np.ndarray, labels: np.ndarray) -> RankDataset:
    """Build a compact complete RankDataset for optimizer unit tests."""

    n_rows = len(labels)
    return RankDataset(
        differences=np.asarray(differences, dtype=float),
        labels=np.asarray(labels, dtype=float),
        query_users=np.arange(n_rows, dtype=int),
        rating_gaps=np.ones(n_rows, dtype=float),
    )


class RankingMathTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(3)
        dataset = rank_dataset(rng.normal(size=(25, 5)), rng.choice([-1.0, 1.0], 25))
        weights = rng.normal(size=5)
        _, analytic = pairwise_loss_gradient(weights, dataset, l2=0.03)
        numeric = np.zeros_like(weights)
        epsilon = 1e-6
        for feature in range(len(weights)):
            plus, minus = weights.copy(), weights.copy()
            plus[feature] += epsilon
            minus[feature] -= epsilon
            loss_plus, _ = pairwise_loss_gradient(plus, dataset, l2=0.03)
            loss_minus, _ = pairwise_loss_gradient(minus, dataset, l2=0.03)
            numeric[feature] = (loss_plus - loss_minus) / (2 * epsilon)
        np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=2e-6)

    def test_pair_orientation_symmetry(self) -> None:
        dataset = rank_dataset(np.array([[1.0, -2.0]]), np.array([1.0]))
        swapped = RankDataset(
            differences=-dataset.differences,
            labels=-dataset.labels,
            query_users=dataset.query_users,
            rating_gaps=dataset.rating_gaps,
        )
        weights = np.array([0.7, -0.1])
        first, first_gradient = pairwise_loss_gradient(weights, dataset, l2=0.0)
        second, second_gradient = pairwise_loss_gradient(weights, swapped, l2=0.0)
        self.assertAlmostEqual(first, second)
        np.testing.assert_allclose(first_gradient, second_gradient)

    def test_training_reduces_loss_and_handles_extreme_margins(self) -> None:
        rng = np.random.default_rng(5)
        differences = rng.normal(size=(300, 3))
        labels = np.where(differences @ np.array([1.2, -0.8, 0.3]) > 0, 1.0, -1.0)
        dataset = rank_dataset(differences, labels)
        model = fit_pairwise_ranker(dataset, np.zeros(3), np.ones(3), steps=250)
        self.assertLess(model.train_loss[-1], model.train_loss[0])

        extreme = rank_dataset(np.array([[1000.0], [-1000.0]]), np.array([1.0, 1.0]))
        loss, gradient = pairwise_loss_gradient(np.array([1000.0]), extreme, l2=0.0)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(gradient).all())


class RealFeatureAndPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.data, cls.temporal, cls.store, cls.retrieval = fit_fixture_pipeline(
            Path(cls.temporary.name)
        )
        cls.user_split = make_user_split(cls.temporal.eligible_users, seed=43)
        cls.feature_mean, cls.feature_std = feature_scaler(
            cls.data,
            cls.temporal,
            cls.store,
            cls.retrieval,
            cls.user_split["train"],
            candidate_k=48,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_feature_store_is_invariant_to_heldout_rating_changes(self) -> None:
        changed_holdout = self.temporal.holdout_ratings.copy()
        changed_holdout[self.temporal.holdout_mask] = 99.0
        mutated = clone_temporal(self.temporal, holdout_ratings=changed_holdout)

        changed_store = estimate_feature_store(self.data, mutated)

        for field in (
            "user_genre",
            "user_year_z",
            "user_popularity_z",
            "item_bayesian_mean",
            "item_log_count",
            "item_year_z",
            "item_popularity_z",
        ):
            np.testing.assert_allclose(getattr(self.store, field), getattr(changed_store, field))
        self.assertEqual(self.store.global_mean_rating, changed_store.global_mean_rating)

    def test_features_and_scaler_are_finite_and_history_derived(self) -> None:
        self.assertEqual(self.feature_mean.shape, (len(FEATURE_NAMES),))
        self.assertEqual(self.feature_std.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(self.feature_mean).all())
        self.assertTrue(np.isfinite(self.feature_std).all())
        self.assertTrue((self.feature_std > 0).all())

        user = int(self.user_split["train"][0])
        items, scores = self.retrieval.candidates(user, 20)
        features = item_features(self.data, self.store, self.retrieval, user, items, scores)
        self.assertEqual(features.shape, (len(items), len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[:, 0], scores)

    def test_observed_pair_construction_is_seeded_gated_and_user_disjoint(self) -> None:
        kwargs = {
            "data": self.data,
            "temporal": self.temporal,
            "store": self.store,
            "retrieval": self.retrieval,
            "users": self.user_split["train"],
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "candidate_k": 48,
            "max_pairs_per_user": 4,
            "minimum_rating_gap": 1.0,
            "seed": 47,
        }
        first = make_pairwise_dataset(**kwargs)
        second = make_pairwise_dataset(**kwargs)

        np.testing.assert_allclose(first.differences, second.differences)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.query_users, second.query_users)
        np.testing.assert_allclose(first.rating_gaps, second.rating_gaps)
        self.assertTrue(set(first.query_users.tolist()) <= set(self.user_split["train"].tolist()))
        self.assertFalse(set(first.query_users.tolist()) & set(self.user_split["test"].tolist()))
        self.assertTrue(np.all(first.rating_gaps >= 1.0))
        for user in np.unique(first.query_users):
            self.assertLessEqual(int(np.sum(first.query_users == user)), 4)

        with self.assertRaisesRegex(ValueError, "no observed rating pairs"):
            make_pairwise_dataset(**{**kwargs, "candidate_k": 1})

    def test_guarded_blend_and_contributions_obey_the_score_contract(self) -> None:
        pairs = make_pairwise_dataset(
            self.data,
            self.temporal,
            self.store,
            self.retrieval,
            self.user_split["train"],
            self.feature_mean,
            self.feature_std,
            48,
            max_pairs_per_user=5,
            seed=51,
        )
        fitted = fit_pairwise_ranker(
            pairs,
            self.feature_mean,
            self.feature_std,
            steps=100,
        )
        guarded = PairwiseRanker(
            weights=fitted.weights,
            feature_mean=fitted.feature_mean,
            feature_std=fitted.feature_std,
            l2=fitted.l2,
            blend_alpha=0.25,
            train_loss=fitted.train_loss,
        )
        user = int(self.user_split["validation"][0])
        items, scores = self.retrieval.candidates(user, 20)
        features = item_features(self.data, self.store, self.retrieval, user, items, scores)
        np.testing.assert_allclose(
            guarded.contributions(features).sum(axis=1),
            guarded.score(features),
            atol=1e-12,
        )

        retrieval_only = PairwiseRanker(
            weights=fitted.weights,
            feature_mean=fitted.feature_mean,
            feature_std=fitted.feature_std,
            l2=fitted.l2,
            blend_alpha=0.0,
            train_loss=fitted.train_loss,
        )
        np.testing.assert_allclose(
            retrieval_only.score(features),
            retrieval_only.transform(features)[:, 0],
        )

        selected, rows = _select_guarded_blend(
            self.data,
            self.temporal,
            self.store,
            self.retrieval,
            fitted,
            self.user_split["validation"],
            48,
            10,
            (0.0, 0.1, 0.2),
        )
        self.assertIn(selected.blend_alpha, (0.0, 0.1, 0.2))
        self.assertEqual([row["blend_alpha"] for row in rows], [0.0, 0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "blend_grid"):
            _select_guarded_blend(
                self.data,
                self.temporal,
                self.store,
                self.retrieval,
                fitted,
                self.user_split["validation"],
                48,
                10,
                (-0.1, 0.1),
            )


if __name__ == "__main__":
    unittest.main()
