from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from recsys_lab.data import (
    MovieLensData,
    chronological_split,
    load_movielens,
    make_user_split,
    observed_relevance,
)
from recsys_lab.metrics import recall_at_k
from recsys_lab.retrieval import RetrievalModel, fit_retrieval, randomized_svd
from tests.fixture_factory import fit_fixture_pipeline, write_movielens_fixture


class MovieLensParserTests(unittest.TestCase):
    def test_parser_remaps_ids_decodes_titles_and_fingerprints_sources(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path = write_movielens_fixture(Path(first))
            second_path = write_movielens_fixture(Path(second))
            data = load_movielens(first_path)
            copy = load_movielens(second_path)

            self.assertEqual((data.n_users, data.n_items), (30, 48))
            self.assertEqual(data.n_ratings, sum(20 + user % 6 for user in range(30)))
            self.assertEqual(data.user_ids[:3], ("700001", "700014", "700027"))
            self.assertEqual(data.item_ids[:3], ("1", "2", "3"))
            self.assertEqual(data.item_names[1], "Am\u00e9lie Fixture (2001)")
            self.assertEqual(data.source_fingerprint, copy.source_fingerprint)
            np.testing.assert_allclose(data.item_genre.sum(axis=1), 1.0)

            item_text = (second_path / "u.item").read_text(encoding="latin-1")
            (second_path / "u.item").write_text(
                item_text.replace("Fixture Movie 001", "Changed Movie 001", 1),
                encoding="latin-1",
            )
            changed = load_movielens(second_path)
            self.assertNotEqual(data.source_fingerprint, changed.source_fingerprint)

    def test_parser_rejects_missing_and_malformed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "u.data, u.item"):
                load_movielens(root)

            write_movielens_fixture(root)
            (root / "u.data").write_text("700001\t1\t9\t880000000\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, r"outside \[1, 5\]"):
                load_movielens(root)


class TemporalSplitTests(unittest.TestCase):
    def test_split_is_chronological_disjoint_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = load_movielens(write_movielens_fixture(Path(temporary)))
            first = chronological_split(data)
            second = chronological_split(data)

            np.testing.assert_array_equal(first.history_seen, second.history_seen)
            np.testing.assert_array_equal(first.holdout_mask, second.holdout_mask)
            self.assertFalse(np.any(first.history_seen & first.holdout_mask))
            self.assertEqual(first.eligible_users.tolist(), list(range(data.n_users)))
            for user in range(data.n_users):
                history_times = first.history_timestamps[user, first.history_seen[user]]
                holdout_times = first.holdout_timestamps[user, first.holdout_mask[user]]
                self.assertLessEqual(int(history_times.max()), int(holdout_times.min()))
                expected = int(np.sum(data.rating_users == user))
                self.assertEqual(len(history_times) + len(holdout_times), expected)

    def test_movie_index_breaks_timestamp_ties(self) -> None:
        n_items = 20
        ratings = np.full(n_items, 3.0)
        ratings[-4:] = np.array([2.0, 3.0, 4.0, 5.0])
        data = MovieLensData(
            user_ids=("fixture-user",),
            item_ids=tuple(str(item + 1) for item in range(n_items)),
            item_names=tuple(f"Movie {item + 1}" for item in range(n_items)),
            item_year=np.arange(1980, 1980 + n_items, dtype=float),
            item_genre=np.eye(n_items, 1, dtype=float),
            rating_users=np.zeros(n_items, dtype=np.int32),
            rating_items=np.arange(n_items, dtype=np.int32),
            rating_values=ratings,
            rating_timestamps=np.full(n_items, 1234, dtype=np.int64),
            source_fingerprint="fixture",
        )

        split = chronological_split(data, min_history=15, min_holdout=4)

        self.assertEqual(np.flatnonzero(split.holdout_mask[0]).tolist(), [16, 17, 18, 19])
        self.assertEqual(int(split.cutoff_timestamps[0]), 1234)

    def test_user_split_is_seeded_disjoint_and_exhaustive(self) -> None:
        users = np.arange(30)
        first = make_user_split(users, seed=17)
        second = make_user_split(users, seed=17)
        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(first[name], second[name])
        joined = np.concatenate(list(first.values()))
        self.assertEqual(sorted(joined.tolist()), users.tolist())
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["validation"]) & set(first["test"]))
        with self.assertRaisesRegex(ValueError, "at least 15"):
            make_user_split(np.arange(14))

    def test_observed_relevance_uses_only_four_and_five_star_gain(self) -> None:
        np.testing.assert_array_equal(
            observed_relevance(np.array([0.0, 1.0, 3.0, 4.0, 5.0])),
            np.array([0.0, 0.0, 0.0, 1.0, 2.0]),
        )


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.data, cls.temporal, cls.store, cls.model = fit_fixture_pipeline(root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_randomized_svd_is_deterministic_finite_and_has_expected_shapes(self) -> None:
        matrix = np.arange(60, dtype=float).reshape(10, 6)
        first = randomized_svd(matrix, rank=3, seed=4)
        second = randomized_svd(matrix, rank=3, seed=4)
        self.assertEqual(first[0].shape, (10, 3))
        self.assertEqual(first[1].shape, (3,))
        self.assertEqual(first[2].shape, (3, 6))
        for left, right in zip(first, second):
            self.assertTrue(np.isfinite(left).all())
            np.testing.assert_allclose(left, right)

    def test_candidates_are_nested_unique_finite_and_exclude_every_seen_rating(self) -> None:
        user = 0
        top_ten, ten_scores = self.model.candidates(user, 10)
        top_twenty, twenty_scores = self.model.candidates(user, 20)
        self.assertEqual(top_ten.tolist(), top_twenty[:10].tolist())
        self.assertEqual(len(top_twenty), len(set(top_twenty.tolist())))
        self.assertTrue(np.isfinite(ten_scores).all())
        self.assertTrue(np.isfinite(twenty_scores).all())
        self.assertFalse(self.temporal.history_seen[user, top_twenty].any())

        low_rated_history = np.flatnonzero(
            self.temporal.history_seen[user] & (self.temporal.history_values[user] == 0)
        )
        self.assertGreater(len(low_rated_history), 0)
        self.assertFalse(set(low_rated_history.tolist()) & set(top_twenty.tolist()))

    def test_candidate_recall_is_monotone_for_observed_heldout_relevance(self) -> None:
        user = 3
        relevant = np.flatnonzero(self.temporal.holdout_ratings[user] >= 4.0)
        top_ten, _ = self.model.candidates(user, 10)
        top_thirty, _ = self.model.candidates(user, 30)
        self.assertLessEqual(
            recall_at_k(top_ten, relevant, len(top_ten)),
            recall_at_k(top_thirty, relevant, len(top_thirty)),
        )
        self.assertEqual(top_ten.tolist(), top_thirty[:10].tolist())

    def test_candidates_return_only_the_remaining_catalog(self) -> None:
        seen = self.model.seen.copy()
        seen[0, :] = True
        seen[0, :3] = False
        model = RetrievalModel(
            user_factors=self.model.user_factors,
            item_factors=self.model.item_factors,
            popularity_z=self.model.popularity_z,
            seen=seen,
            rank=self.model.rank,
        )
        items, _ = model.candidates(0, 20)
        self.assertEqual(sorted(items.tolist()), [0, 1, 2])

    def test_fit_retrieval_rejects_a_misaligned_seen_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            fit_retrieval(np.ones((4, 5)), seen=np.ones((4, 4), dtype=bool), rank=2)


if __name__ == "__main__":
    unittest.main()
