from __future__ import annotations

import csv
import errno
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

import run_demo
from app import config_payload, make_handler, recommendation_payload
from recsys_lab.evaluation import rank_user
from recsys_lab.experiment import ExperimentConfig, _write_csv, load_state, run_experiment
from recsys_lab.report import write_reports
from tests.fixture_factory import write_movielens_fixture


class RealDataIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data_dir = write_movielens_fixture(cls.root / "ml-100k")
        cls.output = cls.root / "artifacts"
        cls.config = ExperimentConfig(
            seed=61,
            retrieval_rank=8,
            candidate_k=40,
            slate_k=10,
            max_pairs_per_user=8,
            minimum_rating_gap=1.0,
            blend_grid=(0.0, 0.1),
            bootstrap_replicates=120,
        )
        cls.state = run_experiment(cls.output, cls.config, cls.data_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_artifacts_are_complete_strict_json_and_aggregate_manifest(self) -> None:
        expected = {
            "experiment.json",
            "recommendations.csv",
            "query_metrics.csv",
            "demo_state.npz",
            "demo_metadata.json",
        }
        self.assertTrue(expected.issubset({path.name for path in self.output.iterdir()}))
        raw = (self.output / "experiment.json").read_text(encoding="utf-8")
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        manifest = json.loads(raw)
        self.assertEqual(manifest["result_schema_version"], 1)
        self.assertEqual(len(manifest["config_sha256"]), 64)
        self.assertEqual(len(manifest["reproducibility"]["source_tree_sha256"]), 64)
        self.assertEqual(len(manifest["reproducibility"]["runtime_lock_sha256"]), 64)
        self.assertEqual(manifest["config"]["seed"], 61)
        self.assertEqual(manifest["data"]["users"], 30)
        self.assertEqual(manifest["data"]["movies"], 48)
        self.assertEqual(manifest["data"]["ratings"], sum(20 + user % 6 for user in range(30)))
        self.assertEqual(
            manifest["data"]["history_ratings"] + manifest["data"]["heldout_ratings"],
            manifest["data"]["ratings"],
        )
        self.assertEqual(
            set(manifest["local_artifact_manifest"]["files"]),
            {"demo_metadata.json", "demo_state.npz", "query_metrics.csv", "recommendations.csv"},
        )

    def test_serialization_preserves_recommendation_predictions(self) -> None:
        loaded = load_state(self.output)
        user = int(self.state.split["test"][0])
        before = recommendation_payload(self.state, user, 50)
        after = recommendation_payload(loaded, user, 50)

        def assert_nested_close(left: object, right: object) -> None:
            if isinstance(left, dict) and isinstance(right, dict):
                self.assertEqual(set(left), set(right))
                for key in left:
                    assert_nested_close(left[key], right[key])
            elif isinstance(left, list) and isinstance(right, list):
                self.assertEqual(len(left), len(right))
                for left_item, right_item in zip(left, right):
                    assert_nested_close(left_item, right_item)
            elif isinstance(left, float) or isinstance(right, float):
                self.assertAlmostEqual(float(left), float(right), places=12)
            else:
                self.assertEqual(left, right)

        assert_nested_close(before, after)
        np.testing.assert_allclose(self.state.ranker.weights, loaded.ranker.weights)
        np.testing.assert_array_equal(self.state.retrieval.seen, loaded.retrieval.seen)

    def test_artifact_loader_rejects_tampering_bombs_shapes_and_nonfinite_values(self) -> None:
        experiment_path = self.output / "experiment.json"
        metadata_path = self.output / "demo_metadata.json"
        state_path = self.output / "demo_state.npz"
        originals = {
            path: path.read_bytes()
            for path in (experiment_path, metadata_path, state_path)
        }

        def contract(path: Path) -> dict[str, int | str]:
            return {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        def synchronize_contracts() -> None:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["state_archive"] = contract(state_path)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
            files = experiment["local_artifact_manifest"]["files"]
            files["demo_state.npz"] = contract(state_path)
            files["demo_metadata.json"] = contract(metadata_path)
            experiment_path.write_text(json.dumps(experiment), encoding="utf-8")

        try:
            state_path.write_bytes(originals[state_path] + b"tamper")
            with self.assertRaisesRegex(ValueError, "recorded size and SHA-256"):
                load_state(self.output)

            state_path.write_bytes(originals[state_path])
            with np.load(state_path, allow_pickle=False) as archive:
                values = {name: np.asarray(archive[name]).copy() for name in archive.files}

            oversized = dict(values)
            oversized["ranker_weights"] = np.zeros(5_000_000, dtype=np.float64)
            np.savez_compressed(state_path, **oversized)
            synchronize_contracts()
            with self.assertRaisesRegex(ValueError, "uncompressed member boundary"):
                load_state(self.output)

            wrong_shape = dict(values)
            wrong_shape["ranker_weights"] = wrong_shape["ranker_weights"][:-1]
            np.savez_compressed(state_path, **wrong_shape)
            synchronize_contracts()
            with self.assertRaisesRegex(ValueError, "invalid array header"):
                load_state(self.output)

            nonfinite = dict(values)
            nonfinite["ranker_weights"][0] = np.nan
            np.savez_compressed(state_path, **nonfinite)
            synchronize_contracts()
            with self.assertRaisesRegex(ValueError, "finite values"):
                load_state(self.output)

            for path, raw in originals.items():
                path.write_bytes(raw)
            experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
            experiment["result_schema_version"] = 999
            experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported or missing"):
                load_state(self.output)
        finally:
            for path, raw in originals.items():
                path.write_bytes(raw)

    def test_quick_run_cannot_overwrite_checked_public_reports(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Refusing to publish"):
            run_demo.main(["--quick", "--publish-reports"])

    def test_csv_export_neutralizes_spreadsheet_formulas(self) -> None:
        path = self.root / "formula-boundary.csv"
        _write_csv(path, [{"name": "=HYPERLINK(\"https://example.com\")", "score": 1.0}])
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        self.assertTrue(row["name"].startswith("'="))

    def test_nonreference_seed_cannot_overwrite_checked_public_reports(self) -> None:
        with self.assertRaisesRegex(SystemExit, "non-reference seed"):
            run_demo.main(["--seed", "7", "--publish-reports"])

    def test_fixture_data_cannot_be_published_as_official_movielens(self) -> None:
        with self.assertRaisesRegex(SystemExit, "unverified data"):
            run_demo.main(
                [
                    "--data",
                    str(self.data_dir),
                    "--output",
                    str(self.root / "publish-attempt"),
                    "--publish-reports",
                ]
            )

    def test_stage_two_is_a_candidate_permutation_and_scores_decompose(self) -> None:
        user = int(self.state.split["test"][0])
        payload = recommendation_payload(self.state, user, 50)
        result = rank_user(
            self.state.data,
            self.state.temporal,
            self.state.store,
            self.state.retrieval,
            self.state.ranker,
            user,
            50,
        )
        self.assertTrue(payload["stage_2_is_candidate_permutation"])
        self.assertEqual(
            set(np.asarray(result["two_stage"]).tolist()),
            set(np.asarray(result["candidates"]).tolist()),
        )
        self.assertEqual(len({row["item"] for row in payload["final_top"]}), 10)
        for row in payload["final_top"]:
            self.assertAlmostEqual(row["contribution_sum"], row["rank_score"], places=10)
            self.assertEqual(row["rank_change"], row["retrieval_rank"] - row["final_rank"])

    def test_user_partitions_and_manifest_document_the_leakage_contract(self) -> None:
        split = self.state.split
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertFalse(set(split["validation"]) & set(split["test"]))
        self.assertEqual(self.state.metrics["evaluation_contract"]["relevant_item"], "held-out rating >= 4")
        self.assertIn(self.state.ranker.blend_alpha, self.config.blend_grid)
        self.assertGreater(self.state.metrics["ranker"]["train_pairs"], 0)
        self.assertGreater(self.state.metrics["ranker"]["validation_pairs"], 0)
        self.assertGreater(self.state.metrics["ranker"]["test_pairs"], 0)

    def test_reports_are_github_readable_aggregate_and_contain_no_raw_identifiers(self) -> None:
        markdown, html = write_reports(
            self.root / "REPORT.md",
            self.root / "docs" / "index.html",
            self.state.metrics,
        )
        markdown_text = markdown.read_text(encoding="utf-8")
        html_text = html.read_text(encoding="utf-8")
        for content in (markdown_text, html_text):
            self.assertIn("MovieLens 100K", content)
            self.assertIn("NDCG@10", content)
            self.assertIn("held-out", content.lower())
            self.assertNotIn("700001", content)
            self.assertNotIn("Am\u00e9lie Fixture", content)
            self.assertNotIn("Fixture Movie", content)
            self.assertNotIn(str(self.root), content)
        self.assertIn("DATA.md", markdown_text)
        embedded = html_text.split(
            '<script type="application/json" id="experiment-manifest">', 1
        )[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(embedded)["config_hash"], self.state.metrics["config_hash"])

    def test_config_exposes_only_locked_test_users(self) -> None:
        payload = config_payload(self.state)
        exposed = {int(row["index"]) for row in payload["users"]}
        self.assertEqual(exposed, set(self.state.split["test"].tolist()))
        self.assertEqual(payload["candidate_widths"], (50, 100, 200, 400))

    def test_http_contract_and_validation_when_loopback_sockets_are_available(self) -> None:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state, b"offline app"))
        except OSError as exc:
            if isinstance(exc, PermissionError) or exc.errno in {errno.EPERM, errno.EACCES}:
                self.skipTest(f"loopback sockets unavailable in this sandbox: {exc}")
            raise

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        valid_user = int(self.state.split["test"][0])
        non_test_user = int(self.state.split["train"][0])
        try:
            with urlopen(base + "/", timeout=3) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), b"offline app")
                self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            with self.assertRaises(HTTPError) as bad_host:
                urlopen(Request(base + "/api/config", headers={"Host": "attacker.example"}), timeout=3)
            self.assertEqual(bad_host.exception.code, 400)
            with urlopen(
                f"{base}/api/recommend?user={valid_user}&k=50", timeout=3
            ) as response:
                payload = json.load(response)
                self.assertEqual(payload["user"], valid_user)
                self.assertTrue(payload["stage_2_is_candidate_permutation"])
            for path in (
                f"/api/recommend?user={non_test_user}&k=50",
                f"/api/recommend?user={valid_user}&k=51",
            ):
                with self.subTest(path=path), self.assertRaises(HTTPError) as bad_request:
                    urlopen(base + path, timeout=3)
                self.assertEqual(bad_request.exception.code, 400)
            with self.assertRaises(HTTPError) as missing:
                urlopen(base + "/missing", timeout=3)
            self.assertEqual(missing.exception.code, 404)
            with self.assertRaises(HTTPError) as method:
                urlopen(Request(base + "/api/config", method="POST"), timeout=3)
            self.assertEqual(method.exception.code, 405)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
