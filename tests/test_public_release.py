import contextlib
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import public_release


class PublicReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.seed_required_files(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def write_text(self, relative: str, content: str = "public\n", root: Path | None = None) -> Path:
        path = (root or self.root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def seed_required_files(self, root: Path) -> None:
        for relative in public_release.REQUIRED_PUBLIC_FILES:
            content = "public\n"
            if relative == "RESULTS.json":
                content = json.dumps(self.aggregate_result()) + "\n"
            elif relative == "docs/index.html":
                content = "<!doctype html><title>Aggregate report</title>\n"
            self.write_text(relative, content, root)

    def codes(self) -> set[str]:
        return {finding.code for finding in public_release.scan_tree(self.root).findings}

    @staticmethod
    def initialize_repository(repository: Path) -> None:
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "config", "user.name", "Release Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "config",
                "user.email",
                "release-test@example.invalid",
            ],
            check=True,
        )

    @staticmethod
    def aggregate_result() -> dict[str, object]:
        result = {name: {} for name in public_release.PUBLIC_RESULT_TOP_LEVEL}
        result["result_schema_version"] = 1
        return result

    def test_clean_allowlisted_source_and_docs_pass(self):
        self.write_text("README.md", "# Public project\n")
        self.write_text("RESULTS.json", json.dumps(self.aggregate_result()) + "\n")
        self.write_text("public_release.py", "# scanner\n")
        self.write_text("recsys_lab/model.py", "VALUE = 1\n")
        self.write_text("tests/test_example.py", "# test\n")
        self.write_text("docs/index.html", "<!doctype html><title>Report</title>\n")
        self.write_text("docs/.nojekyll", "")
        self.write_text(".github/workflows/ci.yml", "name: CI\n")
        result = public_release.scan_tree(self.root)
        self.assertTrue(result.clean, result.findings)
        self.assertEqual(result.candidate_files, len(public_release.REQUIRED_PUBLIC_FILES) + 2)
        self.assertEqual(len(result.source_digest), 64)

    def test_missing_required_file_blocks_release(self):
        (self.root / "RESULTS.json").unlink()
        findings = public_release.scan_tree(self.root).findings
        self.assertIn(
            ("missing-required-file", "RESULTS.json"),
            {(finding.code, finding.path) for finding in findings},
        )

    def test_data_artifacts_caches_and_archives_block_release(self):
        self.write_text("data/raw/ratings.csv", "rating\n")
        self.write_text("artifacts/experiment.json", "{}\n")
        self.write_text("__pycache__/module.pyc", "not bytecode\n")
        self.write_text("bundle.zip", "not a zip\n")
        result = public_release.scan_tree(self.root)
        codes = {finding.code for finding in result.findings}
        paths = {finding.path for finding in result.findings}
        self.assertIn("forbidden-path", codes)
        self.assertIn("forbidden-archive", codes)
        self.assertTrue({"data", "artifacts", "__pycache__"}.issubset(paths))

    def test_personal_path_and_assigned_secret_are_detected(self):
        home_path = "/" + "Users" + "/portfolio-owner/private/file.csv"
        credential_name = "api" + "_key"
        self.write_text(
            "README.md",
            f"input = {home_path}\n{credential_name} = \"this-is-not-a-real-key\"\n",
        )
        self.assertTrue({"personal-path", "secret"}.issubset(self.codes()))

    def test_fine_grained_registry_and_temporary_cloud_tokens_are_detected(self):
        samples = (
            "github" + "_pat_" + "A" * 30,
            "glpat" + "-" + "B" * 30,
            "npm" + "_" + "C" * 30,
            "pypi" + "-" + "D" * 30,
            "ASIA" + "E" * 16,
        )
        self.write_text("README.md", "\n".join(samples) + "\n")
        findings = public_release.scan_tree(self.root).findings
        self.assertEqual(sum(finding.code == "secret" for finding in findings), len(samples))

    def test_non_utf8_text_and_unapproved_binary_are_detected(self):
        (self.root / "README.md").write_bytes(b"\xff\xfe")
        image = self.root / "docs" / "assets" / "chart.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"portfolio")
        result = public_release.scan_tree(self.root)
        self.assertTrue({"non-utf8", "unexpected-path"}.issubset(self.codes()))
        self.assertFalse(result.clean)

    def test_reviewed_binary_hash_can_be_approved_explicitly(self):
        image = self.root / "docs" / "chart.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        raw = b"\x89PNG\r\n\x1a\n" + b"reviewed"
        image.write_bytes(raw)
        approved = {"docs/chart.png": hashlib.sha256(raw).hexdigest()}
        with patch.dict(public_release.APPROVED_BINARY_SHA256, approved, clear=True):
            self.assertTrue(public_release.scan_tree(self.root).clean)

    def test_unreviewed_docs_assets_and_scripts_are_rejected(self):
        self.write_text("docs/examples.json", "{}\n")
        self.write_text("assets/notes.md", "private notes\n")
        self.write_text("docs/app.js", "document.body.textContent = 'unexpected';\n")
        findings = {
            (finding.code, finding.path)
            for finding in public_release.scan_tree(self.root).findings
        }
        self.assertEqual(
            findings,
            {
                ("unexpected-path", "assets/notes.md"),
                ("unexpected-path", "docs/app.js"),
                ("unexpected-path", "docs/examples.json"),
            },
        )

    def test_pages_rejects_active_content_but_allows_one_inert_manifest(self):
        self.write_text(
            "docs/index.html",
            '<!doctype html><script type="application/json" id="experiment-manifest">{}</script>\n',
        )
        self.assertTrue(public_release.scan_tree(self.root).clean)
        self.write_text("docs/index.html", "<!doctype html><script>alert(1)</script>\n")
        findings = public_release.scan_tree(self.root).findings
        self.assertIn(
            ("active-pages-html", "docs/index.html"),
            {(finding.code, finding.path) for finding in findings},
        )

    def test_active_svg_is_rejected(self):
        self.write_text(
            "assets/architecture.svg",
            '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>\n',
        )
        findings = public_release.scan_tree(self.root).findings
        self.assertIn(
            ("active-svg", "assets/architecture.svg"),
            {(finding.code, finding.path) for finding in findings},
        )

    def test_public_result_rejects_row_level_or_unknown_fields(self):
        result = self.aggregate_result()
        result["metrics"] = {"query_id": 7}
        self.write_text("RESULTS.json", json.dumps(result) + "\n")
        self.assertIn("row-level-public-result", self.codes())
        result = self.aggregate_result()
        result["unexpected"] = "field"
        self.write_text("RESULTS.json", json.dumps(result) + "\n")
        self.assertIn("invalid-public-result", self.codes())

    def test_public_result_rejects_duplicate_keys_and_nonfinite_numbers(self):
        self.write_text("RESULTS.json", '{"metrics": {}, "metrics": {}}\n')
        self.assertIn("invalid-public-result", self.codes())
        result = self.aggregate_result()
        result["runtime_seconds"] = float("nan")
        self.write_text("RESULTS.json", json.dumps(result) + "\n")
        self.assertIn("invalid-public-result", self.codes())

    def test_unknown_file_and_symlink_are_rejected(self):
        self.write_text("private_notes.md", "do not publish\n")
        if hasattr(os, "symlink"):
            self.write_text("README.md", "public\n")
            os.symlink(self.root / "README.md", self.root / "linked.md")
        codes = self.codes()
        self.assertIn("unexpected-path", codes)
        if hasattr(os, "symlink"):
            self.assertIn("symlink", codes)

    def test_cli_returns_nonzero_for_blocked_tree(self):
        self.write_text("README.md", "/" + "home" + "/owner/private.txt\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = public_release.main(["--check", "--root", str(self.root)])
        self.assertEqual(status, 1)
        self.assertIn("PUBLIC RELEASE CHECK: BLOCKED", output.getvalue())

    def test_staged_scan_reads_index_instead_of_cleaned_worktree(self):
        repository = self.root / "repository"
        repository.mkdir()
        self.initialize_repository(repository)
        self.seed_required_files(repository)
        readme = repository / "README.md"
        readme.write_text("/" + "Users" + "/portfolio-owner/private.txt\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
        readme.write_text("# Public project\n", encoding="utf-8")

        self.assertTrue(public_release.scan_tree(repository).clean)
        staged = public_release.scan_staged(repository)
        self.assertFalse(staged.clean)
        self.assertIn("personal-path", {finding.code for finding in staged.findings})

    def test_history_scan_detects_removed_secret(self):
        repository = self.root / "history-secret"
        repository.mkdir()
        self.initialize_repository(repository)
        tracked = repository / "temporary.txt"
        tracked.write_text("github" + "_pat_" + "A" * 30 + "\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "temporary.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-qm", "temporary credential"],
            check=True,
        )
        tracked.write_text("removed\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-qam", "remove credential"],
            check=True,
        )
        findings = public_release.git_history_findings(repository)
        self.assertIn("historical-secret", {finding.code for finding in findings})

    def test_history_scan_detects_removed_movielens_data(self):
        repository = self.root / "history-data"
        repository.mkdir()
        self.initialize_repository(repository)
        raw = repository / "data" / "ml-100k" / "u.data"
        raw.parent.mkdir(parents=True)
        raw.write_text("1\t1\t5\t1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-qm", "temporary data"],
            check=True,
        )
        raw.unlink()
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-qam", "remove data"],
            check=True,
        )
        findings = public_release.git_history_findings(repository)
        self.assertIn("historical-third-party-data", {finding.code for finding in findings})


if __name__ == "__main__":
    unittest.main()
