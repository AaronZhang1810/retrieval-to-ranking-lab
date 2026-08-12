#!/usr/bin/env python3
"""Download MovieLens 100K after explicit acceptance of its usage terms.

The dataset is not licensed by this repository. This script downloads the archive directly from GroupLens, verifies a pinned digest, and extracts it with strict path and size checks. See DATA.md before running it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO


ROOT = Path(__file__).resolve().parent
DATASET_NAME = "MovieLens 100K"
DATASET_DIRNAME = "ml-100k"
ARCHIVE_NAME = "ml-100k.zip"
ARCHIVE_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
TERMS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k-README.txt"
ALLOWED_DOWNLOAD_HOSTS = {"files.grouplens.org"}

# GroupLens publishes this MD5 next to the archive. SHA-256 is independently pinned here after downloading that exact official 4,924,029-byte object.
ARCHIVE_MD5 = "0e33842e24a9c977be4e0107933c0723"
ARCHIVE_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
ARCHIVE_SIZE_BYTES = 4_924_029

MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_EXTRACTED_BYTES = 64 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
COPY_CHUNK_BYTES = 1024 * 1024

REQUIRED_FILES = (
    "README",
    "u.data",
    "u.genre",
    "u.info",
    "u.item",
)
EXTRACTED_FILE_SHA256 = {
    "README": "4883b8cf340ed0059971b33f56ad3adc88a40792108ebca3d24aafc4c82e8b52",
    "u.data": "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490",
    "u.genre": "9757e0d3e6b298159a80dd48eaf938114690e911e39af97a76949ef1663610bb",
    "u.info": "5b0a40b2e47a729ee6203063f6dcf8790c2e63f5a18663e6ee318a72c5f3c27b",
    "u.item": "553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701",
}
# Fingerprint produced by ``recsys_lab.data.load_movielens`` from the exact pinned ``u.data`` and ``u.item`` files above. Public reports must match it; this catches a data change between pre-run validation and parsing.
OFFICIAL_PARSED_FINGERPRINT = (
    "ec3f1ed75ed74ee8ed7a7aa17efbcd206de908383c04d8424ee5de6a585dbb05"
)
MANIFEST_NAME = ".download-manifest.json"


class DataAcquisitionError(RuntimeError):
    """Base class for safe, user-facing acquisition failures."""


class IntegrityError(DataAcquisitionError):
    """Raised when downloaded bytes do not match the pinned artifact."""


class UnsafeArchiveError(DataAcquisitionError):
    """Raised when a ZIP member violates the extraction policy."""


class HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject any redirect that would downgrade the official HTTPS request."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        resolved = urllib.parse.urljoin(req.full_url, newurl)
        try:
            validate_download_url(resolved)
        except DataAcquisitionError as exc:
            raise urllib.error.URLError(str(exc)) from exc
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def validate_download_url(url: str) -> None:
    """Allow only the pinned GroupLens HTTPS origin without credentials."""

    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise DataAcquisitionError(f"refusing malformed dataset URL: {url!r}") from exc
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise DataAcquisitionError(f"refusing unapproved dataset URL: {url!r}")


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Return a streaming hexadecimal digest for *path*."""
    # MD5 is checked only to match GroupLens's published legacy checksum; the SHA-256 pin is the security-relevant digest.
    digest = hashlib.md5(usedforsecurity=False) if algorithm == "md5" else hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
    expected_md5: str | None = None,
) -> None:
    """Fail closed unless an archive matches every supplied pin."""
    if not path.is_file():
        raise IntegrityError(f"archive does not exist: {path}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise IntegrityError(
            f"archive size mismatch: expected {expected_size:,} bytes, "
            f"found {path.stat().st_size:,}"
        )
    actual_sha256 = file_digest(path, "sha256")
    if actual_sha256.lower() != expected_sha256.lower():
        raise IntegrityError(
            "archive SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, found {actual_sha256.lower()}"
        )
    if expected_md5 is not None:
        actual_md5 = file_digest(path, "md5")
        if actual_md5.lower() != expected_md5.lower():
            raise IntegrityError(
                f"archive MD5 mismatch: expected {expected_md5.lower()}, found {actual_md5.lower()}"
            )


def _safe_member_parts(name: str, expected_root: str) -> tuple[str, ...]:
    if not name or "\x00" in name or "\\" in name:
        raise UnsafeArchiveError(f"unsafe ZIP member name: {name!r}")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise UnsafeArchiveError(f"absolute ZIP member path: {name!r}")
    parts = tuple(part for part in posix.parts if part not in ("", "."))
    if not parts or ".." in parts or parts[0] != expected_root:
        raise UnsafeArchiveError(f"ZIP member escapes {expected_root!r}: {name!r}")
    for part in parts:
        windows_part = PureWindowsPath(part)
        if (
            ":" in part
            or part.endswith((" ", "."))
            or windows_part.is_reserved()
            or any(ord(character) < 32 for character in part)
        ):
            raise UnsafeArchiveError(f"non-portable ZIP member name: {name!r}")
    return parts


def _is_symlink(member: zipfile.ZipInfo) -> bool:
    mode = member.external_attr >> 16
    return stat.S_IFMT(mode) == stat.S_IFLNK


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    expected_root: str = DATASET_DIRNAME,
    max_members: int = MAX_ARCHIVE_MEMBERS,
    max_member_bytes: int = MAX_MEMBER_BYTES,
    max_total_bytes: int = MAX_EXTRACTED_BYTES,
    max_compression_ratio: int = MAX_COMPRESSION_RATIO,
) -> Path:
    """Extract a ZIP without ``extractall`` and return its expected root.

    Paths, duplicate/case-colliding names, symlinks, encryption, compression methods, member sizes, aggregate size, and suspicious ratios are validated before the first output file is written.
    """
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    prepared: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
    seen: set[str] = set()
    total_size = 0

    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsafeArchiveError(f"invalid ZIP archive: {exc}") from exc

    with archive:
        members = archive.infolist()
        if not members or len(members) > max_members:
            raise UnsafeArchiveError(
                f"unexpected ZIP member count: {len(members)} (maximum {max_members})"
            )

        for member in members:
            parts = _safe_member_parts(member.filename, expected_root)
            collision_key = "/".join(parts).casefold().rstrip("/")
            if collision_key in seen:
                raise UnsafeArchiveError(f"duplicate or case-colliding ZIP member: {member.filename!r}")
            seen.add(collision_key)
            if member.flag_bits & 0x1:
                raise UnsafeArchiveError(f"encrypted ZIP member is not supported: {member.filename!r}")
            if _is_symlink(member):
                raise UnsafeArchiveError(f"symbolic link is not allowed: {member.filename!r}")
            mode_type = stat.S_IFMT(member.external_attr >> 16)
            if mode_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise UnsafeArchiveError(f"special file is not allowed: {member.filename!r}")
            if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise UnsafeArchiveError(f"unsupported compression method: {member.filename!r}")
            if member.file_size < 0 or member.file_size > max_member_bytes:
                raise UnsafeArchiveError(f"oversized ZIP member: {member.filename!r}")
            total_size += member.file_size
            if total_size > max_total_bytes:
                raise UnsafeArchiveError("ZIP expands beyond the aggregate size limit")
            if member.file_size >= 1024 * 1024:
                ratio = member.file_size / max(member.compress_size, 1)
                if ratio > max_compression_ratio:
                    raise UnsafeArchiveError(f"suspicious compression ratio: {member.filename!r}")
            target = destination.joinpath(*parts).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise UnsafeArchiveError(f"ZIP member escapes extraction directory: {member.filename!r}")
            prepared.append((member, parts))

        for member, parts in prepared:
            target = destination.joinpath(*parts)
            if member.is_dir() or member.filename.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with archive.open(member, "r") as source, target.open("xb") as output:
                while chunk := source.read(COPY_CHUNK_BYTES):
                    written += len(chunk)
                    if written > member.file_size or written > max_member_bytes:
                        raise UnsafeArchiveError(f"member exceeded declared size: {member.filename!r}")
                    output.write(chunk)
            if written != member.file_size:
                raise UnsafeArchiveError(f"member size changed while reading: {member.filename!r}")

    return destination / expected_root


def _download_to_verified_archive(archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "two-stage-recommender-lab/1.0 (dataset acquisition)"},
    )
    opener = urllib.request.build_opener(HTTPSOnlyRedirectHandler())
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{ARCHIVE_NAME}.",
            suffix=".part",
            dir=archive_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with opener.open(request, timeout=30) as response:
                final_url = response.geturl()
                validate_download_url(final_url)
                header = response.headers.get("Content-Length")
                if header is not None and int(header) > MAX_DOWNLOAD_BYTES:
                    raise DataAcquisitionError("server advertised an unexpectedly large archive")
                received = 0
                while chunk := response.read(COPY_CHUNK_BYTES):
                    received += len(chunk)
                    if received > MAX_DOWNLOAD_BYTES:
                        raise DataAcquisitionError("download exceeded the maximum allowed size")
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())

        verify_archive(
            temporary_path,
            expected_sha256=ARCHIVE_SHA256,
            expected_size=ARCHIVE_SIZE_BYTES,
            expected_md5=ARCHIVE_MD5,
        )
        temporary_path.replace(archive_path)
        temporary_path = None
    except (OSError, ValueError, urllib.error.URLError) as exc:
        if isinstance(exc, DataAcquisitionError):
            raise
        raise DataAcquisitionError(f"could not download {DATASET_NAME}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _dataset_is_current(dataset_path: Path) -> bool:
    manifest_path = dataset_path / MANIFEST_NAME
    if not manifest_path.is_file() or any(not (dataset_path / name).is_file() for name in REQUIRED_FILES):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    pins_match = (
        manifest.get("source_url") == ARCHIVE_URL
        and manifest.get("archive_sha256") == ARCHIVE_SHA256
        and manifest.get("archive_size_bytes") == ARCHIVE_SIZE_BYTES
    )
    recorded_digests = manifest.get("extracted_file_sha256")
    if (
        not pins_match
        or not isinstance(recorded_digests, dict)
        or recorded_digests != EXTRACTED_FILE_SHA256
    ):
        return False
    try:
        return all(
            EXTRACTED_FILE_SHA256[name] == file_digest(dataset_path / name)
            for name in REQUIRED_FILES
        )
    except OSError:
        return False


def _validate_official_dataset(dataset_path: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (dataset_path / name).is_file()]
    if missing:
        raise DataAcquisitionError(f"archive is missing required files: {', '.join(missing)}")

    def line_count(path: Path) -> int:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)

    ratings = line_count(dataset_path / "u.data")
    movies = line_count(dataset_path / "u.item")
    if ratings != 100_000 or movies != 1_682:
        raise DataAcquisitionError(
            "unexpected MovieLens dimensions: "
            f"u.data has {ratings:,} rows and u.item has {movies:,} rows"
        )
    mismatches = [
        name
        for name in REQUIRED_FILES
        if file_digest(dataset_path / name) != EXTRACTED_FILE_SHA256[name]
    ]
    if mismatches:
        raise DataAcquisitionError(
            "extracted MovieLens files do not match the pinned archive: " + ", ".join(mismatches)
        )


def validate_official_dataset(dataset_path: str | Path) -> Path:
    """Require the exact downloader-installed dataset before publishing reports."""

    path = Path(dataset_path).resolve()
    if not _dataset_is_current(path):
        raise DataAcquisitionError(
            "dataset provenance is missing, stale, or does not match the pinned MovieLens files"
        )
    _validate_official_dataset(path)
    return path


def acquire_dataset(data_dir: Path) -> Path:
    """Acquire, verify, and safely extract MovieLens 100K idempotently."""
    data_dir = data_dir.resolve()
    archive_path = data_dir / "raw" / ARCHIVE_NAME
    dataset_path = data_dir / DATASET_DIRNAME
    if _dataset_is_current(dataset_path):
        print(f"Dataset already verified: {dataset_path}")
        return dataset_path
    if dataset_path.exists():
        raise DataAcquisitionError(
            f"refusing to replace unverified existing path: {dataset_path}. "
            "Move or remove it after checking its contents, then rerun."
        )

    archive_is_valid = False
    if archive_path.is_file():
        try:
            verify_archive(
                archive_path,
                expected_sha256=ARCHIVE_SHA256,
                expected_size=ARCHIVE_SIZE_BYTES,
                expected_md5=ARCHIVE_MD5,
            )
            archive_is_valid = True
        except IntegrityError:
            print("Existing archive failed integrity checks; downloading a verified replacement.")
    if not archive_is_valid:
        print(f"Downloading {DATASET_NAME} directly from GroupLens...")
        _download_to_verified_archive(archive_path)

    data_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ml-100k-extract-", dir=data_dir))
    try:
        staged_dataset = safe_extract_zip(archive_path, staging)
        _validate_official_dataset(staged_dataset)
        manifest = {
            "dataset": DATASET_NAME,
            "source_url": ARCHIVE_URL,
            "terms_url": TERMS_URL,
            "archive_sha256": ARCHIVE_SHA256,
            "archive_md5": ARCHIVE_MD5,
            "archive_size_bytes": ARCHIVE_SIZE_BYTES,
            "extracted_file_sha256": EXTRACTED_FILE_SHA256,
        }
        (staged_dataset / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_dataset.replace(dataset_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"Verified dataset ready: {dataset_path}")
    return dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm that you read and accept the separate GroupLens usage terms in DATA.md.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Local ignored data directory (default: ./data).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.accept_license:
        raise SystemExit(
            "Refusing to download without explicit acceptance. Read DATA.md and the GroupLens "
            "terms, then rerun with --accept-license."
        )
    try:
        acquire_dataset(args.data_dir)
    except DataAcquisitionError as exc:
        raise SystemExit(f"Data acquisition failed: {exc}") from exc


if __name__ == "__main__":
    main()
