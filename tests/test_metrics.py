from __future__ import annotations

import unittest

import numpy as np

from recsys_lab.metrics import (
    dcg,
    intralist_diversity,
    ndcg_at_k,
    paired_cluster_bootstrap,
    recall_at_k,
    reciprocal_rank,
)


class RankingMetricTests(unittest.TestCase):
    def test_dcg_and_ndcg_match_hand_orderings(self) -> None:
        relevance = np.array([2.0, 1.0, 0.0, 0.0])
        expected_dcg = 3.0 + 1.0 / np.log2(3.0)
        self.assertAlmostEqual(dcg(np.array([2.0, 1.0])), expected_dcg)
        self.assertAlmostEqual(ndcg_at_k(np.array([0, 1, 2, 3]), relevance, 4), 1.0)
        self.assertLess(ndcg_at_k(np.array([3, 2, 1, 0]), relevance, 4), 1.0)
        self.assertEqual(ndcg_at_k(np.array([0, 1]), np.zeros(2), 2), 0.0)

    def test_recall_and_reciprocal_rank_handle_hits_and_empty_targets(self) -> None:
        ranked = np.array([4, 2, 7])
        relevant = np.array([2, 7, 9, 11])
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 3), 0.5)
        self.assertAlmostEqual(reciprocal_rank(ranked, relevant, 3), 0.5)
        self.assertEqual(recall_at_k(ranked, np.array([], dtype=int), 3), 0.0)
        self.assertEqual(reciprocal_rank(ranked, np.array([99]), 3), 0.0)

    def test_intralist_diversity_uses_pairwise_cosine_distance(self) -> None:
        genres = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        # Pair distances are 0, 1, and 1.
        self.assertAlmostEqual(intralist_diversity(np.array([0, 1, 2]), genres, 3), 2.0 / 3.0)
        self.assertEqual(intralist_diversity(np.array([0]), genres, 1), 0.0)


class BootstrapTests(unittest.TestCase):
    def test_zero_delta_is_exact_and_seeded(self) -> None:
        users = np.repeat(np.arange(8), 2)
        values = np.linspace(0.1, 0.8, len(users))
        first = paired_cluster_bootstrap(users, values, values, n_bootstrap=300, seed=2)
        second = paired_cluster_bootstrap(users, values, values, n_bootstrap=300, seed=2)
        self.assertEqual(first, second)
        self.assertIn("bootstrap_fraction_positive", first)
        self.assertNotIn("bootstrap_probability_positive", first)
        self.assertEqual(first["estimate"], 0.0)
        self.assertEqual(first["ci_low"], 0.0)
        self.assertEqual(first["ci_high"], 0.0)
        self.assertEqual(first["clusters"], 8)

    def test_positive_interval_resamples_users_not_rows(self) -> None:
        users = np.repeat(np.arange(15), 3)
        control = np.zeros(len(users))
        treatment = np.repeat(np.linspace(0.1, 0.3, 15), 3)
        result = paired_cluster_bootstrap(users, treatment, control, n_bootstrap=500, seed=3)
        self.assertGreater(result["ci_low"], 0.0)
        self.assertLessEqual(result["ci_low"], result["estimate"])
        self.assertLessEqual(result["estimate"], result["ci_high"])
        self.assertEqual(result["clusters"], 15)
        self.assertEqual(result["replicates"], 500)


if __name__ == "__main__":
    unittest.main()
