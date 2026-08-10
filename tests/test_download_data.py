"""Offline tests for MovieLens acquisition integrity and safe extraction."""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from download_data import (
    ARCHIVE_SHA256,
    ARCHIVE_SIZE_BYTES,
    ARCHIVE_URL,
    DataAcquisitionError,
    EXTRACTED_FILE_SHA256,
    MANIFEST_NAME,
    REQUIRED_FILES,
    IntegrityError,
    UnsafeArchiveError,
    _dataset_is_current,
    file_digest,
    safe_extract_zip,
    validate_download_url,
    verify_archive,
)


class DownloadIntegrityTests(unittest.TestCase):
    def test_verify_archive_accepts_exact_fixture_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "fixture.zip"
            archive.write_bytes(b"local deterministic fixture")
            sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
            md5 = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()

            verify_archive(
                archive,
                expected_sha256=sha256,
                expected_size=archive.stat().st_size,
                expected_md5=md5,
            )

    def test_verify_archive_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "fixture.zip"
            archive.write_bytes(b"tampered")
            with self.assertRaisesRegex(IntegrityError, "SHA-256 mismatch"):
                verify_archive(archive, expected_sha256="0" * 64)

    def test_provenance_marker_is_idempotent_and_detects_local_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "ml-100k"
            dataset.mkdir()
            for name in REQUIRED_FILES:
                (dataset / name).write_text(f"fixture for {name}\n", encoding="utf-8")
            manifest = {
                "source_url": ARCHIVE_URL,
                "archive_sha256": ARCHIVE_SHA256,
                "archive_size_bytes": ARCHIVE_SIZE_BYTES,
            }
            expected = {name: file_digest(dataset / name) for name in REQUIRED_FILES}
            manifest["extracted_file_sha256"] = expected
            (dataset / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
            with patch.dict(EXTRACTED_FILE_SHA256, expected, clear=True):
                self.assertTrue(_dataset_is_current(dataset))
                (dataset / "u.data").write_text("locally changed\n", encoding="utf-8")
                self.assertFalse(_dataset_is_current(dataset))

    def test_download_origin_is_exactly_allowlisted(self) -> None:
        validate_download_url(ARCHIVE_URL)
        for url in (
            "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
            "https://example.com/ml-100k.zip",
            "https://" + "user" + ":" + "password" + "@files.grouplens.org/ml-100k.zip",
            "https://files.grouplens.org:444/ml-100k.zip",
        ):
            with self.subTest(url=url), self.assertRaises(DataAcquisitionError):
                validate_download_url(url)


class SafeExtractionTests(unittest.TestCase):
    @staticmethod
    def _write_zip(path: Path, members: dict[str, bytes]) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)

    def test_extracts_local_fixture_under_expected_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "fixture.zip"
            output = root / "output"
            self._write_zip(
                archive,
                {
                    "ml-100k/README": b"fixture terms",
                    "ml-100k/u.data": b"1\t1\t5\t1\n",
                    "ml-100k/u.item": b"1|Fixture Movie\n",
                },
            )

            extracted = safe_extract_zip(archive, output)

            self.assertEqual(extracted, output / "ml-100k")
            self.assertEqual((extracted / "u.data").read_bytes(), b"1\t1\t5\t1\n")
            self.assertFalse((root / "u.data").exists())

    def test_rejects_parent_directory_traversal_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "traversal.zip"
            output = root / "output"
            self._write_zip(
                archive,
                {
                    "ml-100k/good.txt": b"must not be written",
                    "ml-100k/../../escape.txt": b"escape",
                },
            )

            with self.assertRaisesRegex(UnsafeArchiveError, "escapes"):
                safe_extract_zip(archive, output)

            self.assertFalse((output / "ml-100k" / "good.txt").exists())
            self.assertFalse((root / "escape.txt").exists())

    def test_rejects_absolute_and_windows_drive_paths(self) -> None:
        for name in ("/ml-100k/escape.txt", "C:/ml-100k/escape.txt", "C:\\ml-100k\\escape.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "absolute.zip"
                self._write_zip(archive, {name: b"escape"})
                with self.assertRaises(UnsafeArchiveError):
                    safe_extract_zip(archive, root / "output")

    def test_rejects_symlink_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "symlink.zip"
            member = zipfile.ZipInfo("ml-100k/link")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(member, "../../escape")

            with self.assertRaisesRegex(UnsafeArchiveError, "Symbolic|symbolic"):
                safe_extract_zip(archive_path, root / "output")

    def test_rejects_case_colliding_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "duplicate.zip"
            self._write_zip(
                archive,
                {
                    "ml-100k/u.data": b"one",
                    "ml-100k/U.DATA": b"two",
                },
            )
            with self.assertRaises(UnsafeArchiveError):
                safe_extract_zip(archive, root / "output")


if __name__ == "__main__":
    unittest.main()
