#!/usr/bin/env python3
"""Audit a public source candidate and its reachable Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
MAX_HISTORY_BYTES = 64 * 1024 * 1024

PUBLIC_ROOT_FILES = {
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "DATA.md",
    "LICENSE",
    "LIMITATIONS.md",
    "MATH.md",
    "Makefile",
    "NOTICE.md",
    "README.md",
    "REPORT.md",
    "RESULTS.json",
    "SECURITY.md",
    "app.py",
    "download_data.py",
    "public_release.py",
    "pyproject.toml",
    "requirements-audit-lock.txt",
    "requirements-dev-lock.txt",
    "requirements-audit-lock.txt",
    "requirements-lock.txt",
    "requirements.txt",
    "run_demo.py",
    "run_demo.sh",
}

PUBLIC_DOCUMENT_FILES = {
    "assets/architecture.svg",
    "docs/.nojekyll",
    "docs/index.html",
    "static/index.html",
}

PUBLIC_GITHUB_FILES = {
    ".github/REPOSITORY_METADATA.md",
    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/pages.yml",
}

# A release scan must prove completeness as well as absence of forbidden files.
# Optional future files may remain allowlisted above, but these files define the
# minimum reviewable, reproducible portfolio repository.
REQUIRED_PUBLIC_FILES = {
    ".gitattributes",
    ".github/REPOSITORY_METADATA.md",
    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/pages.yml",
    ".gitignore",
    "CITATION.cff",
    "DATA.md",
    "LICENSE",
    "LIMITATIONS.md",
    "MATH.md",
    "Makefile",
    "NOTICE.md",
    "README.md",
    "REPORT.md",
    "RESULTS.json",
    "SECURITY.md",
    "app.py",
    "assets/architecture.svg",
    "docs/.nojekyll",
    "docs/index.html",
    "download_data.py",
    "public_release.py",
    "pyproject.toml",
    "recsys_lab/__init__.py",
    "recsys_lab/data.py",
    "recsys_lab/evaluation.py",
    "recsys_lab/experiment.py",
    "recsys_lab/metrics.py",
    "recsys_lab/ranking.py",
    "recsys_lab/report.py",
    "recsys_lab/retrieval.py",
    "requirements-dev-lock.txt",
    "requirements-lock.txt",
    "requirements.txt",
    "run_demo.py",
    "run_demo.sh",
    "static/index.html",
    "tests/__init__.py",
    "tests/fixture_factory.py",
    "tests/test_data_retrieval.py",
    "tests/test_download_data.py",
    "tests/test_integration.py",
    "tests/test_metrics.py",
    "tests/test_public_outputs.py",
    "tests/test_public_release.py",
    "tests/test_ranking.py",
}

PUBLIC_TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".svg",
    ".txt",
    ".xml",
    ".yml",
    ".yaml",
}

PUBLIC_BINARY_SUFFIXES = {".gif", ".ico", ".jpeg", ".jpg", ".png", ".webp"}

# Binary report assets require a deliberate, reviewed source change. Add the
# relative path and SHA-256 only after checking the image itself and its metadata.
APPROVED_BINARY_SHA256: dict[str, str] = {}

FORBIDDEN_DIRECTORY_NAMES = {
    ".eggs",
    ".idea",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".rproj.user",
    ".venv",
    ".vscode",
    "__pycache__",
    "artifacts",
    "build",
    "data",
    "dist",
    "htmlcov",
    "node_modules",
    "venv",
}

FORBIDDEN_FILE_NAMES = {".coverage", ".ds_store", ".rdata", ".rhistory"}

FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bz2",
    ".db",
    ".dmg",
    ".egg",
    ".feather",
    ".gz",
    ".joblib",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pyc",
    ".rar",
    ".rds",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".whl",
    ".xz",
    ".zip",
}

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_CANDIDATE_BYTES = 20 * 1024 * 1024

# MovieLens is acquired locally under GroupLens's separate terms. These names
# must not appear as reachable Git paths, even if a later commit deleted them.
MOVIELENS_DATA_NAMES = {
    "allbut.pl",
    "mku.sh",
    "u.data",
    "u.genre",
    "u.info",
    "u.item",
    "u.occupation",
    "u.user",
    "ua.base",
    "ua.test",
    "ub.base",
    "ub.test",
}
MOVIELENS_SPLIT_NAME = re.compile(r"u[1-5]\.(?:base|test)\Z", re.IGNORECASE)

PUBLIC_RESULT_TOP_LEVEL = {
    "activity_slices",
    "candidate_curve",
    "config",
    "config_hash",
    "config_sha256",
    "data",
    "environment",
    "evaluation_contract",
    "judged_candidate_diagnostic",
    "latency",
    "limitations",
    "local_artifact_manifest",
    "metrics",
    "paired_user_bootstrap",
    "project",
    "ranker",
    "reproducibility",
    "result_schema_version",
    "runtime_seconds",
    "score_zeroing_sensitivity",
}
FORBIDDEN_PUBLIC_RESULT_KEYS = {
    "heldout_rating",
    "item_id",
    "item_ids",
    "item_name",
    "item_names",
    "movie_id",
    "prediction",
    "predictions",
    "prompt",
    "query",
    "query_id",
    "rating_items",
    "rating_timestamps",
    "rating_users",
    "rating_values",
    "recommendations",
    "title",
    "user_id",
}


def _walk_json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_walk_json_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_walk_json_keys(item) for item in value), set())
    return set()


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _user_home_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    # Split sensitive path prefixes so this scanner does not flag its own source.
    mac_home = "/" + r"Users/[^/\s]+/"
    linux_home = "/" + r"home/[^/\s]+/"
    windows_home = r"[A-Za-z]:\\" + r"Users\\[^\\\s]+\\"
    file_home = "file:///" + r"(?:Users|home)/[^/\s]+/"
    return (
        ("macOS user-home path", re.compile(mac_home)),
        ("Linux user-home path", re.compile(linux_home)),
        ("Windows user-home path", re.compile(windows_home, re.IGNORECASE)),
        ("user-home file URL", re.compile(file_home, re.IGNORECASE)),
        ("tilde-relative home path", re.compile(r"(?<![\w.])" + "~" + r"/")),
    )


def _secret_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    private_key = "-----BEGIN " + r"(?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
    openai_style = "s" + r"k-[A-Za-z0-9_-]{20,}"
    return (
        ("private key", re.compile(private_key)),
        ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
        ("AWS temporary access key", re.compile(r"ASIA[0-9A-Z]{16}")),
        ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
        ("GitHub fine-grained token", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
        ("GitLab token", re.compile(r"glpat-[A-Za-z0-9_-]{20,}")),
        ("npm token", re.compile(r"npm_[A-Za-z0-9]{20,}")),
        ("PyPI token", re.compile(r"pypi-[A-Za-z0-9_-]{20,}")),
        ("API token", re.compile(openai_style)),
        ("Google API key", re.compile(r"AIza[A-Za-z0-9_-]{20,}")),
        ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
        ("bearer credential", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~-]{20,}")),
        (
            "assigned credential",
            re.compile(
                r"(?ix)\b(?:api[_-]?key|client[_-]?secret|password|passwd|"
                r"access[_-]?token|auth[_-]?token)\b\s*[\"']?\s*[:=]\s*"
                r"[\"'][^\"'\n]{8,}[\"']"
            ),
        ),
        (
            "credential in URL",
            re.compile(r"(?i)https?://[^\s/:@]+:[^\s/@]+@[^\s/]+"),
        ),
    )


PERSONAL_PATH_PATTERNS = _user_home_patterns()
SECRET_PATTERNS = _secret_patterns()


BINARY_MAGIC = {
    ".gif": (b"GIF87a", b"GIF89a"),
    ".ico": (b"\x00\x00\x01\x00",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".jpg": (b"\xff\xd8\xff",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".webp": (b"RIFF",),
}


@dataclass(frozen=True, order=True)
class Finding:
    code: str
    path: str
    detail: str


@dataclass(frozen=True)
class ScanResult:
    findings: tuple[Finding, ...]
    candidate_files: int
    candidate_bytes: int
    source_digest: str

    @property
    def clean(self) -> bool:
        return not self.findings


def _public_candidate(relative: Path) -> bool:
    portable = relative.as_posix()
    parts = relative.parts
    if len(parts) == 1:
        return portable in PUBLIC_ROOT_FILES
    if portable in PUBLIC_GITHUB_FILES:
        return True
    if parts[0] == "recsys_lab":
        return len(parts) == 2 and relative.suffix == ".py"
    if parts[0] == "tests":
        return len(parts) == 2 and relative.suffix == ".py"
    if parts[0] in {"assets", "docs", "static"}:
        return portable in PUBLIC_DOCUMENT_FILES or portable in APPROVED_BINARY_SHA256
    return False


def _line_number(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def _content_findings(relative: Path, raw: bytes) -> list[Finding]:
    portable = relative.as_posix()
    suffix = relative.suffix.lower()
    findings: list[Finding] = []

    if suffix in PUBLIC_BINARY_SUFFIXES:
        digest = hashlib.sha256(raw).hexdigest()
        if APPROVED_BINARY_SHA256.get(portable) != digest:
            findings.append(
                Finding(
                    "unexpected-binary",
                    portable,
                    "binary asset is not present in the reviewed SHA-256 allowlist",
                )
            )
        signatures = BINARY_MAGIC[suffix]
        if not any(raw.startswith(signature) for signature in signatures):
            findings.append(
                Finding("invalid-binary-format", portable, f"content does not match {suffix} magic")
            )
        if suffix == ".webp" and (len(raw) < 12 or raw[8:12] != b"WEBP"):
            findings.append(
                Finding("invalid-binary-format", portable, "RIFF file is not a WebP image")
            )
        # This catches uncompressed metadata. The digest approval remains the
        # authoritative review boundary for content hidden by compression.
        text = raw.decode("latin-1", errors="ignore")
    else:
        if b"\x00" in raw:
            findings.append(
                Finding("unexpected-binary", portable, "NUL byte found in a declared text file")
            )
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            findings.append(
                Finding(
                    "non-utf8",
                    portable,
                    f"strict UTF-8 decoding failed at byte {exc.start}",
                )
            )
            return findings

    if suffix == ".svg":
        active_svg_patterns = (
            r"(?i)<\s*script\b",
            r"(?i)<\s*foreignObject\b",
            r"(?i)\bon[a-z]+\s*=",
            r"(?i)javascript\s*:",
            r"(?i)(?:href|src)\s*=\s*[\"']\s*https?://",
        )
        if any(re.search(pattern, text) for pattern in active_svg_patterns):
            findings.append(
                Finding("active-svg", portable, "SVG contains scriptable or remote active content")
            )

    if portable == "docs/index.html":
        inert_json = re.compile(
            r'(?is)<script\s+type=["\']application/json["\']\s+id=["\']experiment-manifest["\']>'
            r".*?</script>"
        )
        without_inert_manifest, replacements = inert_json.subn("", text)
        active_html_patterns = (
            r"(?i)<\s*script\b",
            r"(?i)<\s*(?:iframe|object|embed)\b",
            r"(?i)\bon[a-z]+\s*=",
            r"(?i)javascript\s*:",
            r"(?i)\b(?:src|action)\s*=\s*[\"']\s*https?://",
        )
        if replacements > 1 or any(
            re.search(pattern, without_inert_manifest) for pattern in active_html_patterns
        ):
            findings.append(
                Finding(
                    "active-pages-html",
                    portable,
                    "Pages HTML contains active or remote content outside its inert JSON manifest",
                )
            )

    if portable == "RESULTS.json":
        try:
            result = json.loads(
                text,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            findings.append(
                Finding("invalid-public-result", portable, f"invalid strict JSON: {exc}")
            )
        else:
            if not isinstance(result, dict) or set(result) != PUBLIC_RESULT_TOP_LEVEL:
                findings.append(
                    Finding(
                        "invalid-public-result",
                        portable,
                        "top-level schema differs from the aggregate-only allowlist",
                    )
                )
            else:
                forbidden = sorted(_walk_json_keys(result) & FORBIDDEN_PUBLIC_RESULT_KEYS)
                if forbidden:
                    findings.append(
                        Finding(
                            "row-level-public-result",
                            portable,
                            "forbidden row-level keys: " + ", ".join(forbidden),
                        )
                    )
                if result.get("result_schema_version") != 1:
                    findings.append(
                        Finding(
                            "invalid-public-result",
                            portable,
                            "unsupported or missing aggregate result schema",
                        )
                    )

    for label, pattern in PERSONAL_PATH_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                Finding(
                    "personal-path",
                    portable,
                    f"{label} at line {_line_number(text, match.start())}",
                )
            )
    for label, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                Finding(
                    "secret",
                    portable,
                    f"possible {label} at line {_line_number(text, match.start())}",
                )
            )
    return findings


def _walk_files(root: Path) -> tuple[list[Path], list[Finding]]:
    files: list[Path] = []
    findings: list[Finding] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_names.sort()
        file_names.sort()
        kept_directories: list[str] = []
        for name in directory_names:
            path = directory_path / name
            relative = path.relative_to(root)
            if name == ".git":
                continue
            if path.is_symlink():
                findings.append(
                    Finding("symlink", relative.as_posix(), "symbolic-link directories are not public candidates")
                )
                continue
            if name.lower() in FORBIDDEN_DIRECTORY_NAMES:
                findings.append(
                    Finding("forbidden-path", relative.as_posix(), "local/generated directory is present")
                )
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in file_names:
            path = directory_path / name
            relative = path.relative_to(root)
            if path.is_symlink():
                findings.append(
                    Finding("symlink", relative.as_posix(), "symbolic-link files are not public candidates")
                )
                continue
            files.append(path)
    return files, findings


def scan_tree(root: str | Path = ROOT) -> ScanResult:
    root = Path(root).resolve()
    files, findings = _walk_files(root)
    source_hash = hashlib.sha256()
    candidate_files = 0
    candidate_bytes = 0
    present_candidates: set[str] = set()

    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        portable = relative.as_posix()
        lower_name = relative.name.lower()
        suffix = relative.suffix.lower()

        if lower_name in FORBIDDEN_FILE_NAMES:
            findings.append(Finding("forbidden-path", portable, "local/generated file is present"))
            continue
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(
                Finding("forbidden-archive", portable, f"forbidden generated/archive suffix {suffix}")
            )
            continue
        if not _public_candidate(relative):
            findings.append(
                Finding("unexpected-path", portable, "path is outside the public source/docs allowlist")
            )
            continue

        try:
            raw = path.read_bytes()
        except OSError as exc:
            findings.append(Finding("read-error", portable, str(exc)))
            continue

        candidate_files += 1
        candidate_bytes += len(raw)
        present_candidates.add(portable)
        source_hash.update(portable.encode("utf-8"))
        source_hash.update(b"\x00")
        source_hash.update(str(len(raw)).encode("ascii"))
        source_hash.update(b"\x00")
        source_hash.update(raw)

        if len(raw) > MAX_FILE_BYTES:
            findings.append(
                Finding(
                    "oversized-file",
                    portable,
                    f"{len(raw):,} bytes exceeds the {MAX_FILE_BYTES:,}-byte limit",
                )
            )
        findings.extend(_content_findings(relative, raw))

    if candidate_bytes > MAX_CANDIDATE_BYTES:
        findings.append(
            Finding(
                "oversized-release",
                ".",
                f"{candidate_bytes:,} candidate bytes exceeds the {MAX_CANDIDATE_BYTES:,}-byte limit",
            )
        )

    for missing in sorted(REQUIRED_PUBLIC_FILES - present_candidates):
        findings.append(
            Finding("missing-required-file", missing, "required public release file is absent")
        )

    return ScanResult(
        findings=tuple(sorted(set(findings))),
        candidate_files=candidate_files,
        candidate_bytes=candidate_bytes,
        source_digest=source_hash.hexdigest(),
    )


def scan_staged(root: str | Path = ROOT) -> ScanResult:
    """Audit exactly the Git index proposed for the next commit."""

    root = Path(root).resolve()
    try:
        boundary = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return ScanResult(
            findings=(Finding("git-index", ".", f"could not resolve a standalone Git index: {exc}"),),
            candidate_files=0,
            candidate_bytes=0,
            source_digest=hashlib.sha256().hexdigest(),
        )
    if Path(boundary).resolve() != root:
        return ScanResult(
            findings=(Finding("git-boundary", ".", "repository root resolves outside this project"),),
            candidate_files=0,
            candidate_bytes=0,
            source_digest=hashlib.sha256().hexdigest(),
        )

    with tempfile.TemporaryDirectory(prefix="recsys-public-index-") as temporary:
        checkout = Path(temporary) / "candidate"
        checkout.mkdir()
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "checkout-index",
                    "--all",
                    f"--prefix={checkout}{os.sep}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or str(exc)
            return ScanResult(
                findings=(Finding("git-index", ".", f"could not materialize staged files: {detail}"),),
                candidate_files=0,
                candidate_bytes=0,
                source_digest=hashlib.sha256().hexdigest(),
            )
        return scan_tree(checkout)


def _git_root(root: str | Path) -> Path:
    root = Path(root).resolve()
    try:
        boundary = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("history audit requires a readable Git repository") from exc
    git_root = Path(boundary).resolve()
    if git_root != root:
        raise RuntimeError("repository root resolves outside this project")
    return git_root


def _historical_path_finding(path_text: str) -> Finding | None:
    path = Path(path_text)
    lower_parts = {part.lower() for part in path.parts}
    lower_name = path.name.lower()
    if (
        lower_name in MOVIELENS_DATA_NAMES
        or MOVIELENS_SPLIT_NAME.fullmatch(lower_name)
        or "ml-100k" in lower_parts
    ):
        return Finding(
            "historical-third-party-data",
            path_text,
            "MovieLens source data exists in reachable Git history",
        )
    if lower_parts & FORBIDDEN_DIRECTORY_NAMES:
        return Finding(
            "historical-forbidden-path",
            path_text,
            "local/generated directory exists in reachable Git history",
        )
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return Finding(
            "historical-forbidden-archive",
            path_text,
            "generated/archive file exists in reachable Git history",
        )
    return None


def git_history_findings(root: str | Path = ROOT) -> list[Finding]:
    """Audit all reachable patches and paths, including deleted content."""

    root = _git_root(root)
    try:
        commit_count = subprocess.run(
            ["git", "-C", str(root), "rev-list", "--all", "--count"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if int(commit_count or "0") == 0:
            return []
        object_listing = subprocess.run(
            ["git", "-C", str(root), "rev-list", "--objects", "--all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        patches = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "log",
                "--all",
                "--format=commit %H%n%B",
                "--patch",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
            ],
            check=True,
            capture_output=True,
            timeout=60,
        ).stdout
    except (
        FileNotFoundError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise RuntimeError("could not scan complete Git history") from exc

    if len(patches) > MAX_HISTORY_BYTES:
        return [
            Finding(
                "history-size-limit",
                ".git-history",
                "history exceeds the built-in limit; use a dedicated full-history scanner",
            )
        ]

    findings: list[Finding] = []
    for line in object_listing.splitlines():
        _object_id, _separator, path_text = line.partition(" ")
        if path_text:
            finding = _historical_path_finding(path_text)
            if finding is not None:
                findings.append(finding)

    patch_text = patches.decode("utf-8", errors="replace")
    for label, pattern in PERSONAL_PATH_PATTERNS:
        for match in pattern.finditer(patch_text):
            findings.append(
                Finding(
                    "historical-personal-path",
                    ".git-history",
                    f"{label} at patch line {_line_number(patch_text, match.start())}",
                )
            )
    for label, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(patch_text):
            findings.append(
                Finding(
                    "historical-secret",
                    ".git-history",
                    f"possible {label} at patch line {_line_number(patch_text, match.start())}",
                )
            )
    return sorted(set(findings))


def format_result(result: ScanResult) -> str:
    status = "CLEAN" if result.clean else "BLOCKED"
    lines = [
        f"PUBLIC RELEASE CHECK: {status}",
        f"candidate files: {result.candidate_files}",
        f"candidate bytes: {result.candidate_bytes:,}",
        f"source digest: {result.source_digest}",
    ]
    if result.findings:
        lines.append("findings:")
        lines.extend(
            f"- [{finding.code}] {finding.path}: {finding.detail}"
            for finding in result.findings
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="audit the release candidate and exit")
    action.add_argument(
        "--source-digest",
        action="store_true",
        help="print the digest after a clean candidate audit",
    )
    parser.add_argument("--staged", action="store_true", help="audit the Git index, not the live tree")
    parser.add_argument(
        "--history",
        action="store_true",
        help="also audit every reachable Git patch and historical path",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = scan_staged(args.root) if args.staged else scan_tree(args.root)
    if args.history:
        try:
            history_findings = git_history_findings(args.root)
        except RuntimeError as exc:
            print(f"PUBLIC RELEASE CHECK: BLOCKED\n- [git-history] .: {exc}")
            return 2
        result = ScanResult(
            findings=tuple(sorted(set(result.findings) | set(history_findings))),
            candidate_files=result.candidate_files,
            candidate_bytes=result.candidate_bytes,
            source_digest=result.source_digest,
        )

    if args.check or not result.clean:
        print(format_result(result))
        return 0 if result.clean else 1
    print(result.source_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
