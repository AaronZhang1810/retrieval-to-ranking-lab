import json
from pathlib import Path
import unittest

import public_release
from download_data import OFFICIAL_PARSED_FINGERPRINT
from recsys_lab.experiment import reproducibility_record
from recsys_lab.report import html_report, markdown_report


ROOT = Path(__file__).resolve().parents[1]


class CheckedPublicOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads((ROOT / "RESULTS.json").read_text(encoding="utf-8"))

    def test_reports_match_the_canonical_aggregate_result(self) -> None:
        self.assertEqual((ROOT / "REPORT.md").read_text(encoding="utf-8"), markdown_report(self.result))
        self.assertEqual((ROOT / "docs" / "index.html").read_text(encoding="utf-8"), html_report(self.result))

    def test_result_is_bound_to_current_source_and_runtime_lock(self) -> None:
        self.assertEqual(self.result["result_schema_version"], 1)
        self.assertEqual(self.result["reproducibility"], reproducibility_record(ROOT))
        self.assertEqual(set(self.result), public_release.PUBLIC_RESULT_TOP_LEVEL)

    def test_result_contains_aggregate_evidence_not_rows(self) -> None:
        keys = public_release._walk_json_keys(self.result)
        self.assertFalse(keys & public_release.FORBIDDEN_PUBLIC_RESULT_KEYS)
        self.assertEqual(self.result["data"]["ratings"], 100_000)
        self.assertEqual(self.result["data"]["users"], 943)
        self.assertEqual(self.result["data"]["movies"], 1_682)
        self.assertEqual(
            self.result["data"]["source_fingerprint"],
            OFFICIAL_PARSED_FINGERPRINT,
        )
        files = self.result["local_artifact_manifest"]["files"]
        self.assertEqual(
            set(files),
            {"demo_metadata.json", "demo_state.npz", "query_metrics.csv", "recommendations.csv"},
        )
        for contract in files.values():
            self.assertEqual(set(contract), {"bytes", "sha256"})
            self.assertGreater(contract["bytes"], 0)
            self.assertRegex(contract["sha256"], r"^[0-9a-f]{64}$")

    def test_interviewer_readme_matches_the_canonical_headline_metrics(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        metrics = self.result["metrics"]
        delta = self.result["paired_user_bootstrap"][
            "two_stage_minus_retrieval_ndcg_at_10"
        ]
        expected = {
            self.result["config_hash"],
            f"{metrics['retrieval']['ndcg_at_10']:.4f}",
            f"{metrics['two_stage']['ndcg_at_10']:.4f}",
            f"{metrics['unblended_ranker']['ndcg_at_10']:.4f}",
            f"{delta['estimate']:+.4f}",
            f"{delta['ci_low']:+.4f}",
            f"{delta['ci_high']:+.4f}",
            f"{100 * self.result['ranker']['test_pair_accuracy']:.1f}%",
        }
        for value in expected:
            self.assertIn(value, readme)


if __name__ == "__main__":
    unittest.main()
