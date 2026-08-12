"""End-to-end MovieLens experiment orchestration and local serialization."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
import zipfile

import numpy as np

from .data import (
    DEFAULT_DATA_DIR,
    MovieLensData,
    TemporalSplit,
    chronological_split,
    load_movielens,
    make_user_split,
)
from .evaluation import EvaluationResult, evaluate, rank_user
from .metrics import ndcg_at_k, paired_cluster_bootstrap
from .ranking import (
    FEATURE_NAMES,
    FeatureStore,
    PairwiseRanker,
    estimate_feature_store,
    feature_scaler,
    make_pairwise_dataset,
    pairwise_accuracy,
    select_regularization,
)
from .retrieval import RetrievalModel, fit_retrieval


ROOT = Path(__file__).resolve().parents[1]
RESULT_SCHEMA_VERSION = 1
REPRODUCIBILITY_FILES = (
    "download_data.py",
    "pyproject.toml",
    "requirements-lock.txt",
    "run_demo.py",
    "recsys_lab/__init__.py",
    "recsys_lab/data.py",
    "recsys_lab/evaluation.py",
    "recsys_lab/experiment.py",
    "recsys_lab/metrics.py",
    "recsys_lab/ranking.py",
    "recsys_lab/report.py",
    "recsys_lab/retrieval.py",
)
STATE_ARRAY_NAMES = {
    "item_year",
    "item_genre",
    "history_seen",
    "holdout_ratings",
    "holdout_mask",
    "store_user_genre",
    "store_user_year_z",
    "store_user_popularity_z",
    "store_item_bayesian_mean",
    "store_item_log_count",
    "store_item_year_z",
    "store_item_popularity_z",
    "retrieval_user_factors",
    "retrieval_item_factors",
    "retrieval_popularity_z",
    "retrieval_seen",
    "ranker_weights",
    "ranker_feature_mean",
    "ranker_feature_std",
    "split_train",
    "split_validation",
    "split_test",
}
MAX_EXPERIMENT_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_STATE_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_STATE_MEMBER_BYTES = 32 * 1024 * 1024
MAX_STATE_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_STATE_USERS = 10_000
MAX_STATE_ITEMS = 100_000
MAX_STATE_CELLS = 50_000_000


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260808
    holdout_fraction: float = 0.20
    retrieval_rank: int = 32
    candidate_k: int = 200
    slate_k: int = 10
    max_pairs_per_user: int = 80
    minimum_rating_gap: float = 1.0
    blend_grid: tuple[float, ...] = (0.0, 0.025, 0.05, 0.075, 0.10)
    bootstrap_replicates: int = 3000


@dataclass(frozen=True)
class ExperimentState:
    data: MovieLensData
    temporal: TemporalSplit
    split: dict[str, np.ndarray]
    store: FeatureStore
    retrieval: RetrievalModel
    ranker: PairwiseRanker
    metrics: dict[str, Any]
    demo_cases: list[dict[str, object]]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _config_sha256(config: ExperimentConfig, source_fingerprint: str) -> str:
    payload = {"config": asdict(config), "source_fingerprint": source_fingerprint}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_contract(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def reproducibility_record(source_root: str | Path = ROOT) -> dict[str, Any]:
    """Bind aggregate results to the executable files and exact runtime lock."""

    root = Path(source_root)
    digest = hashlib.sha256()
    for relative in REPRODUCIBILITY_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"reproducibility source is missing: {relative}")
        raw = path.read_bytes()
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return {
        "source_tree_sha256": digest.hexdigest(),
        "runtime_lock_sha256": _sha256_file(root / "requirements-lock.txt"),
        "source_files": list(REPRODUCIBILITY_FILES),
    }


def _delta_from_rows(
    rows: list[dict[str, float | int | str]],
    metric: str,
    treatment: str,
    control: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    treatment_rows = sorted(
        (row for row in rows if row["method"] == treatment), key=lambda row: int(row["query_id"])
    )
    control_rows = sorted(
        (row for row in rows if row["method"] == control), key=lambda row: int(row["query_id"])
    )
    assert [row["query_id"] for row in treatment_rows] == [row["query_id"] for row in control_rows]
    return paired_cluster_bootstrap(
        np.asarray([row["user"] for row in treatment_rows], dtype=int),
        np.asarray([row[metric] for row in treatment_rows], dtype=float),
        np.asarray([row[metric] for row in control_rows], dtype=float),
        n_bootstrap=replicates,
        seed=seed,
    )


def _select_guarded_blend(
    data: MovieLensData,
    temporal: TemporalSplit,
    store: FeatureStore,
    retrieval: RetrievalModel,
    ranker: PairwiseRanker,
    validation_users: np.ndarray,
    candidate_k: int,
    slate_k: int,
    grid: tuple[float, ...],
) -> tuple[PairwiseRanker, list[dict[str, float]]]:
    """Choose a deliberately small residual blend using validation NDCG only."""

    if not grid or min(grid) < 0.0 or max(grid) > 1.0:
        raise ValueError("blend_grid must contain values in [0, 1]")
    rows: list[dict[str, float]] = []
    models: list[PairwiseRanker] = []
    for alpha in grid:
        model = PairwiseRanker(
            weights=ranker.weights,
            feature_mean=ranker.feature_mean,
            feature_std=ranker.feature_std,
            l2=ranker.l2,
            blend_alpha=float(alpha),
            train_loss=ranker.train_loss,
        )
        values: list[float] = []
        for user in np.asarray(validation_users, dtype=int):
            result = rank_user(data, temporal, store, retrieval, model, int(user), candidate_k)
            values.append(
                ndcg_at_k(np.asarray(result["two_stage"]), np.asarray(result["relevance"]), slate_k)
            )
        rows.append({"blend_alpha": float(alpha), "validation_ndcg_at_10": float(np.mean(values))})
        models.append(model)
    best = max(range(len(models)), key=lambda index: (rows[index]["validation_ndcg_at_10"], -grid[index]))
    return models[best], rows


def _save_state(output_dir: Path, state: ExperimentState) -> None:
    data, temporal, store = state.data, state.temporal, state.store
    retrieval, ranker = state.retrieval, state.ranker
    state_path = output_dir / "demo_state.npz"
    np.savez_compressed(
        state_path,
        item_year=data.item_year,
        item_genre=data.item_genre,
        history_seen=temporal.history_seen,
        holdout_ratings=temporal.holdout_ratings,
        holdout_mask=temporal.holdout_mask,
        store_user_genre=store.user_genre,
        store_user_year_z=store.user_year_z,
        store_user_popularity_z=store.user_popularity_z,
        store_item_bayesian_mean=store.item_bayesian_mean,
        store_item_log_count=store.item_log_count,
        store_item_year_z=store.item_year_z,
        store_item_popularity_z=store.item_popularity_z,
        retrieval_user_factors=retrieval.user_factors,
        retrieval_item_factors=retrieval.item_factors,
        retrieval_popularity_z=retrieval.popularity_z,
        retrieval_seen=retrieval.seen,
        ranker_weights=ranker.weights,
        ranker_feature_mean=ranker.feature_mean,
        ranker_feature_std=ranker.feature_std,
        split_train=state.split["train"],
        split_validation=state.split["validation"],
        split_test=state.split["test"],
    )
    metadata = {
        "artifact_schema_version": 1,
        "user_ids": data.user_ids,
        "item_ids": data.item_ids,
        "item_names": data.item_names,
        "feature_names": FEATURE_NAMES,
        "source_fingerprint": data.source_fingerprint,
        "retrieval_rank": retrieval.rank,
        "ranker_l2": ranker.l2,
        "ranker_blend_alpha": ranker.blend_alpha,
        "global_mean_rating": store.global_mean_rating,
        "state_archive": _file_contract(state_path),
    }
    (output_dir / "demo_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def _read_bounded_json(path: Path, maximum: int, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required {label} is missing or is a symbolic link")
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise ValueError(f"{label} size is outside the accepted boundary")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return parsed


def _verify_file_contract(path: Path, contract: object, label: str, maximum: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required {label} is missing or is a symbolic link")
    if not isinstance(contract, dict):
        raise ValueError(f"{label} integrity contract is missing")
    expected_bytes = contract.get("bytes")
    expected_sha256 = contract.get("sha256")
    actual_bytes = path.stat().st_size
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes != actual_bytes
        or actual_bytes <= 0
        or actual_bytes > maximum
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or _sha256_file(path) != expected_sha256
    ):
        raise ValueError(f"{label} does not match its recorded size and SHA-256")


def _artifact_dimensions(metadata: dict[str, Any]) -> tuple[int, int, int]:
    """Validate bounded metadata fields needed before array materialization."""

    user_ids = metadata.get("user_ids")
    item_ids = metadata.get("item_ids")
    item_names = metadata.get("item_names")
    if not all(isinstance(values, list) for values in (user_ids, item_ids, item_names)):
        raise ValueError("artifact metadata has invalid identifier arrays")
    if not all(
        isinstance(value, str)
        for values in (user_ids, item_ids, item_names)
        for value in values
    ):
        raise ValueError("artifact metadata identifiers must be strings")
    n_users, n_items = len(user_ids), len(item_ids)
    if len(item_names) != n_items:
        raise ValueError("artifact metadata item names do not match item IDs")
    if (
        not 1 <= n_users <= MAX_STATE_USERS
        or not 1 <= n_items <= MAX_STATE_ITEMS
        or n_users * n_items > MAX_STATE_CELLS
    ):
        raise ValueError("artifact dimensions are outside the accepted boundary")
    rank = metadata.get("retrieval_rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= min(n_users, n_items):
        raise ValueError("artifact retrieval rank is invalid")
    if metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("artifact feature manifest is invalid")
    fingerprint = metadata.get("source_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("artifact source fingerprint is invalid")
    for name in ("ranker_l2", "ranker_blend_alpha", "global_mean_rating"):
        value = metadata.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"artifact scalar is invalid: {name}")
    return n_users, n_items, rank


def _state_array_specs(
    metadata: dict[str, Any], data_manifest: dict[str, Any]
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Derive the exact safe NPY layout from bounded JSON metadata."""

    n_users, n_items, rank = _artifact_dimensions(metadata)
    if data_manifest.get("users") != n_users or data_manifest.get("movies") != n_items:
        raise ValueError("artifact dimensions do not match the experiment manifest")
    split_sizes: dict[str, int] = {}
    for array_name, manifest_name in (
        ("split_train", "rank_train_users"),
        ("split_validation", "rank_validation_users"),
        ("split_test", "test_users"),
    ):
        size = data_manifest.get(manifest_name)
        if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= n_users:
            raise ValueError("artifact split count is invalid")
        split_sizes[array_name] = size
    if sum(split_sizes.values()) > n_users:
        raise ValueError("artifact split counts exceed the user count")

    specs: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "item_year": ((n_items,), np.dtype("float64")),
        "item_genre": ((n_items, 19), np.dtype("float64")),
        "history_seen": ((n_users, n_items), np.dtype("bool")),
        "holdout_ratings": ((n_users, n_items), np.dtype("float32")),
        "holdout_mask": ((n_users, n_items), np.dtype("bool")),
        "store_user_genre": ((n_users, 19), np.dtype("float64")),
        "store_user_year_z": ((n_users,), np.dtype("float64")),
        "store_user_popularity_z": ((n_users,), np.dtype("float64")),
        "store_item_bayesian_mean": ((n_items,), np.dtype("float64")),
        "store_item_log_count": ((n_items,), np.dtype("float64")),
        "store_item_year_z": ((n_items,), np.dtype("float64")),
        "store_item_popularity_z": ((n_items,), np.dtype("float64")),
        "retrieval_user_factors": ((n_users, rank), np.dtype("float64")),
        "retrieval_item_factors": ((n_items, rank), np.dtype("float64")),
        "retrieval_popularity_z": ((n_items,), np.dtype("float64")),
        "retrieval_seen": ((n_users, n_items), np.dtype("bool")),
        "ranker_weights": ((len(FEATURE_NAMES),), np.dtype("float64")),
        "ranker_feature_mean": ((len(FEATURE_NAMES),), np.dtype("float64")),
        "ranker_feature_std": ((len(FEATURE_NAMES),), np.dtype("float64")),
    }
    specs.update(
        {
            name: ((size,), np.dtype("int64"))
            for name, size in split_sizes.items()
        }
    )
    return specs


def _validate_state_zip(
    path: Path,
    specs: dict[str, tuple[tuple[int, ...], np.dtype[Any]]],
) -> None:
    expected = {f"{name}.npy" for name in STATE_ARRAY_NAMES}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("state archive has an unexpected ZIP member manifest")
            total = 0
            for member in members:
                if member.flag_bits & 0x1:
                    raise ValueError("state archive must not contain encrypted members")
                if member.file_size <= 0 or member.file_size > MAX_STATE_MEMBER_BYTES:
                    raise ValueError("state archive exceeds an uncompressed member boundary")
                total += member.file_size
                array_name = member.filename.removesuffix(".npy")
                try:
                    with archive.open(member) as handle:
                        version = np.lib.format.read_magic(handle)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(
                                handle, max_header_size=10_000
                            )
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(
                                handle, max_header_size=10_000
                            )
                        else:
                            raise ValueError("state archive uses an unsupported NPY format")
                        header_bytes = handle.tell()
                except (EOFError, OSError, ValueError) as exc:
                    if isinstance(exc, ValueError) and str(exc).startswith("state archive"):
                        raise
                    raise ValueError("state archive contains an invalid NPY header") from exc
                expected_shape, expected_dtype = specs[array_name]
                if tuple(shape) != expected_shape or np.dtype(dtype) != expected_dtype:
                    raise ValueError(f"state archive has an invalid array header: {array_name}")
                expected_payload = int(np.prod(expected_shape, dtype=np.int64)) * expected_dtype.itemsize
                if member.file_size - header_bytes != expected_payload:
                    raise ValueError(f"state archive has an invalid array payload: {array_name}")
            if total > MAX_STATE_UNCOMPRESSED_BYTES:
                raise ValueError("state archive exceeds the total uncompressed size boundary")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("state archive is not a valid NPZ container") from exc


def _validate_loaded_arrays(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> tuple[int, int, int]:
    n_users, n_items, rank = _artifact_dimensions(metadata)

    shapes = {
        "item_year": (n_items,),
        "item_genre": (n_items, 19),
        "history_seen": (n_users, n_items),
        "holdout_ratings": (n_users, n_items),
        "holdout_mask": (n_users, n_items),
        "store_user_genre": (n_users, 19),
        "store_user_year_z": (n_users,),
        "store_user_popularity_z": (n_users,),
        "store_item_bayesian_mean": (n_items,),
        "store_item_log_count": (n_items,),
        "store_item_year_z": (n_items,),
        "store_item_popularity_z": (n_items,),
        "retrieval_user_factors": (n_users, rank),
        "retrieval_item_factors": (n_items, rank),
        "retrieval_popularity_z": (n_items,),
        "retrieval_seen": (n_users, n_items),
        "ranker_weights": (len(FEATURE_NAMES),),
        "ranker_feature_mean": (len(FEATURE_NAMES),),
        "ranker_feature_std": (len(FEATURE_NAMES),),
    }
    boolean_arrays = {"history_seen", "holdout_mask", "retrieval_seen"}
    for name, shape in shapes.items():
        value = arrays[name]
        if value.shape != shape:
            raise ValueError(f"state array has an invalid shape: {name}")
        if name in boolean_arrays:
            if value.dtype != np.dtype("bool"):
                raise ValueError(f"state array has an invalid dtype: {name}")
        elif name == "holdout_ratings":
            if value.dtype != np.dtype("float32") or not np.all(np.isfinite(value)):
                raise ValueError("holdout ratings must contain finite float32 values")
        elif value.dtype != np.dtype("float64"):
            raise ValueError(f"state array has an invalid dtype: {name}")
        elif name == "item_year":
            if np.any(np.isinf(value)):
                raise ValueError("item years must not contain infinite values")
        elif not np.all(np.isfinite(value)):
            raise ValueError(f"state array must contain finite values: {name}")

    split_values: list[np.ndarray] = []
    for name in ("split_train", "split_validation", "split_test"):
        value = arrays[name]
        if value.ndim != 1 or value.dtype.kind not in "iu" or len(value) == 0:
            raise ValueError(f"state split array is invalid: {name}")
        if np.any(value < 0) or np.any(value >= n_users) or len(np.unique(value)) != len(value):
            raise ValueError(f"state split contains invalid user indices: {name}")
        split_values.append(value.astype(int, copy=False))
    if any(set(left.tolist()) & set(right.tolist()) for index, left in enumerate(split_values) for right in split_values[index + 1 :]):
        raise ValueError("state user splits overlap")
    if np.any(arrays["ranker_feature_std"] <= 0):
        raise ValueError("ranker feature scales must be positive")
    if not np.array_equal(arrays["retrieval_seen"], arrays["history_seen"]):
        raise ValueError("retrieval seen mask does not match frozen history")
    if np.any(arrays["history_seen"] & arrays["holdout_mask"]):
        raise ValueError("frozen history and held-out outcomes overlap")
    return n_users, n_items, rank


def load_state(artifact_dir: str | Path) -> ExperimentState:
    """Load local derived state used by the browser trace."""

    artifact_dir = Path(artifact_dir)
    experiment_path = artifact_dir / "experiment.json"
    metadata_path = artifact_dir / "demo_metadata.json"
    state_path = artifact_dir / "demo_state.npz"
    metrics = _read_bounded_json(experiment_path, MAX_EXPERIMENT_BYTES, "experiment manifest")
    if metrics.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported or missing experiment result schema")
    local_manifest = metrics.get("local_artifact_manifest")
    if not isinstance(local_manifest, dict) or not isinstance(local_manifest.get("files"), dict):
        raise ValueError("experiment manifest is missing local artifact contracts")
    contracts = local_manifest["files"]
    _verify_file_contract(
        metadata_path,
        contracts.get("demo_metadata.json"),
        "artifact metadata",
        MAX_METADATA_BYTES,
    )
    _verify_file_contract(
        state_path,
        contracts.get("demo_state.npz"),
        "state archive",
        MAX_STATE_ARCHIVE_BYTES,
    )
    metadata = _read_bounded_json(metadata_path, MAX_METADATA_BYTES, "artifact metadata")
    if metadata.get("artifact_schema_version") != 1:
        raise ValueError("unsupported or missing local artifact schema")
    _verify_file_contract(
        state_path,
        metadata.get("state_archive"),
        "state archive",
        MAX_STATE_ARCHIVE_BYTES,
    )
    data_manifest = metrics.get("data")
    config_manifest = metrics.get("config")
    if not isinstance(data_manifest, dict) or not isinstance(config_manifest, dict):
        raise ValueError("experiment manifest has an invalid data or configuration section")
    specs = _state_array_specs(metadata, data_manifest)
    _validate_state_zip(state_path, specs)
    try:
        with np.load(state_path, allow_pickle=False) as archive:
            if set(archive.files) != STATE_ARRAY_NAMES:
                raise ValueError("state archive has an unexpected NumPy array manifest")
            arrays = {name: np.asarray(archive[name]).copy() for name in STATE_ARRAY_NAMES}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("state archive"):
            raise
        raise ValueError("state archive is invalid or unsupported") from exc
    n_users, n_items, _ = _validate_loaded_arrays(arrays, metadata)
    if data_manifest.get("users") != n_users or data_manifest.get("movies") != n_items:
        raise ValueError("artifact dimensions do not match the experiment manifest")
    if metadata.get("source_fingerprint") != data_manifest.get("source_fingerprint"):
        raise ValueError("artifact source fingerprint does not match the experiment manifest")
    for name, key in (
        ("split_train", "rank_train_users"),
        ("split_validation", "rank_validation_users"),
        ("split_test", "test_users"),
    ):
        if len(arrays[name]) != data_manifest.get(key):
            raise ValueError("artifact split counts do not match the experiment manifest")
    data = MovieLensData(
        user_ids=tuple(metadata["user_ids"]),
        item_ids=tuple(metadata["item_ids"]),
        item_names=tuple(metadata["item_names"]),
        item_year=arrays["item_year"],
        item_genre=arrays["item_genre"],
        rating_users=np.array([], dtype=np.int32),
        rating_items=np.array([], dtype=np.int32),
        rating_values=np.array([], dtype=float),
        rating_timestamps=np.array([], dtype=np.int64),
        source_fingerprint=str(metadata["source_fingerprint"]),
    )
    temporal = TemporalSplit(
        history_values=np.zeros((n_users, n_items), dtype=float),
        history_ratings=np.zeros((n_users, n_items), dtype=np.float32),
        history_seen=arrays["history_seen"],
        history_timestamps=np.zeros((n_users, n_items), dtype=np.int64),
        holdout_ratings=arrays["holdout_ratings"],
        holdout_mask=arrays["holdout_mask"],
        holdout_timestamps=np.zeros((n_users, n_items), dtype=np.int64),
        cutoff_timestamps=np.zeros(n_users, dtype=np.int64),
        eligible_users=np.array([], dtype=int),
        excluded_users=int(metrics["data"]["excluded_users"]),
        holdout_fraction=float(metrics["config"]["holdout_fraction"]),
    )
    store = FeatureStore(
        user_genre=arrays["store_user_genre"],
        user_year_z=arrays["store_user_year_z"],
        user_popularity_z=arrays["store_user_popularity_z"],
        item_bayesian_mean=arrays["store_item_bayesian_mean"],
        item_log_count=arrays["store_item_log_count"],
        item_year_z=arrays["store_item_year_z"],
        item_popularity_z=arrays["store_item_popularity_z"],
        global_mean_rating=float(metadata["global_mean_rating"]),
    )
    retrieval = RetrievalModel(
        user_factors=arrays["retrieval_user_factors"],
        item_factors=arrays["retrieval_item_factors"],
        popularity_z=arrays["retrieval_popularity_z"],
        seen=arrays["retrieval_seen"],
        rank=int(metadata["retrieval_rank"]),
    )
    ranker = PairwiseRanker(
        weights=arrays["ranker_weights"],
        feature_mean=arrays["ranker_feature_mean"],
        feature_std=arrays["ranker_feature_std"],
        l2=float(metadata["ranker_l2"]),
        blend_alpha=float(metadata["ranker_blend_alpha"]),
        train_loss=(),
    )
    split = {
        "train": arrays["split_train"],
        "validation": arrays["split_validation"],
        "test": arrays["split_test"],
    }
    return ExperimentState(
        data=data,
        temporal=temporal,
        split=split,
        store=store,
        retrieval=retrieval,
        ranker=ranker,
        metrics=metrics,
        demo_cases=[],
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(
            {
                key: "'" + value
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r"))
                else value
                for key, value in row.items()
            }
            for row in rows
        )


def _score_zeroing_sensitivity(
    state_parts: tuple[MovieLensData, TemporalSplit, FeatureStore, RetrievalModel],
    ranker: PairwiseRanker,
    users: np.ndarray,
    candidate_k: int,
    slate_k: int,
) -> list[dict[str, float | str]]:
    data, temporal, store, retrieval = state_parts
    rows: list[dict[str, float | str]] = []
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        weights = ranker.weights.copy()
        weights[feature_index] = 0.0
        ablated = PairwiseRanker(
            weights=weights,
            feature_mean=ranker.feature_mean,
            feature_std=ranker.feature_std,
            l2=ranker.l2,
            blend_alpha=ranker.blend_alpha,
            train_loss=(),
        )
        ndcg_values: list[float] = []
        for user in np.asarray(users, dtype=int):
            result = rank_user(data, temporal, store, retrieval, ablated, int(user), candidate_k)
            ndcg_values.append(
                ndcg_at_k(np.asarray(result["two_stage"]), np.asarray(result["relevance"]), slate_k)
            )
        rows.append({"zeroed_feature": feature_name, "ndcg_at_10": float(np.mean(ndcg_values))})
    return rows


def _activity_slices(
    query_rows: list[dict[str, float | int | str]],
    temporal: TemporalSplit,
    users: np.ndarray,
) -> list[dict[str, float | int | str]]:
    activity = temporal.history_seen.sum(axis=1)
    values = activity[np.asarray(users, dtype=int)]
    lower, upper = np.quantile(values, [1 / 3, 2 / 3])
    labels = {
        int(user): "low" if activity[user] <= lower else "high" if activity[user] > upper else "medium"
        for user in users
    }
    result: list[dict[str, float | int | str]] = []
    for slice_name in ("low", "medium", "high"):
        slice_users = {user for user, label in labels.items() if label == slice_name}
        for method in ("retrieval", "two_stage"):
            rows = [
                row for row in query_rows if row["method"] == method and int(row["user"]) in slice_users
            ]
            result.append(
                {
                    "activity_slice": slice_name,
                    "method": method,
                    "users": len(rows),
                    "ndcg_at_10": float(np.mean([float(row["ndcg_at_10"]) for row in rows])),
                    "recall_at_10": float(np.mean([float(row["recall_at_10"]) for row in rows])),
                }
            )
    return result


def run_experiment(
    output_dir: str | Path,
    config: ExperimentConfig | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> ExperimentState:
    config = config or ExperimentConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    data = load_movielens(data_dir)
    temporal = chronological_split(data, holdout_fraction=config.holdout_fraction)
    split = make_user_split(temporal.eligible_users, seed=config.seed + 2)
    retrieval = fit_retrieval(
        temporal.history_values,
        seen=temporal.history_seen,
        rank=config.retrieval_rank,
        seed=config.seed + 3,
    )
    store = estimate_feature_store(data, temporal)
    feature_mean, feature_std = feature_scaler(
        data, temporal, store, retrieval, split["train"], config.candidate_k
    )
    train_pairs = make_pairwise_dataset(
        data,
        temporal,
        store,
        retrieval,
        split["train"],
        feature_mean,
        feature_std,
        config.candidate_k,
        max_pairs_per_user=config.max_pairs_per_user,
        minimum_rating_gap=config.minimum_rating_gap,
        seed=config.seed + 7,
    )
    validation_pairs = make_pairwise_dataset(
        data,
        temporal,
        store,
        retrieval,
        split["validation"],
        feature_mean,
        feature_std,
        config.candidate_k,
        max_pairs_per_user=config.max_pairs_per_user,
        minimum_rating_gap=config.minimum_rating_gap,
        seed=config.seed + 8,
    )
    test_pairs = make_pairwise_dataset(
        data,
        temporal,
        store,
        retrieval,
        split["test"],
        feature_mean,
        feature_std,
        config.candidate_k,
        max_pairs_per_user=config.max_pairs_per_user,
        minimum_rating_gap=config.minimum_rating_gap,
        seed=config.seed + 9,
    )
    ranker, regularization_grid = select_regularization(
        train_pairs, validation_pairs, feature_mean, feature_std
    )
    ranker, blend_grid = _select_guarded_blend(
        data,
        temporal,
        store,
        retrieval,
        ranker,
        split["validation"],
        config.candidate_k,
        config.slate_k,
        config.blend_grid,
    )
    evaluation: EvaluationResult = evaluate(
        data,
        temporal,
        store,
        retrieval,
        ranker,
        split["test"],
        candidate_k=config.candidate_k,
        slate_k=config.slate_k,
    )

    ndcg_delta = _delta_from_rows(
        evaluation.query_rows,
        "ndcg_at_10",
        "two_stage",
        "retrieval",
        config.bootstrap_replicates,
        config.seed + 10,
    )
    recall_delta = _delta_from_rows(
        evaluation.query_rows,
        "recall_at_10",
        "two_stage",
        "retrieval",
        config.bootstrap_replicates,
        config.seed + 11,
    )
    all_timestamps = data.rating_timestamps
    config_sha256 = _config_sha256(config, data.source_fingerprint)
    metrics: dict[str, Any] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "project": "Two-stage recommendation on MovieLens 100K",
        "config": asdict(config),
        "config_hash": config_sha256[:12],
        "config_sha256": config_sha256,
        "reproducibility": reproducibility_record(ROOT),
        "data": {
            "dataset": "MovieLens 100K",
            "source": "GroupLens Research, University of Minnesota",
            "source_url": "https://grouplens.org/datasets/movielens/100k/",
            "source_fingerprint": data.source_fingerprint,
            "users": data.n_users,
            "movies": data.n_items,
            "ratings": data.n_ratings,
            "density": data.n_ratings / (data.n_users * data.n_items),
            "date_start": datetime.fromtimestamp(
                int(all_timestamps.min()), tz=timezone.utc
            ).date().isoformat(),
            "date_end": datetime.fromtimestamp(
                int(all_timestamps.max()), tz=timezone.utc
            ).date().isoformat(),
            "history_ratings": int(temporal.history_seen.sum()),
            "heldout_ratings": int(temporal.holdout_mask.sum()),
            "eligible_users": len(temporal.eligible_users),
            "excluded_users": temporal.excluded_users,
            "rank_train_users": len(split["train"]),
            "rank_validation_users": len(split["validation"]),
            "test_users": len(split["test"]),
        },
        "metrics": evaluation.summaries,
        "paired_user_bootstrap": {
            "two_stage_minus_retrieval_ndcg_at_10": ndcg_delta,
            "two_stage_minus_retrieval_recall_at_10": recall_delta,
        },
        "candidate_curve": evaluation.candidate_curve,
        "judged_candidate_diagnostic": evaluation.judged_candidate_diagnostic,
        "activity_slices": _activity_slices(evaluation.query_rows, temporal, split["test"]),
        "score_zeroing_sensitivity": _score_zeroing_sensitivity(
            (data, temporal, store, retrieval),
            ranker,
            split["test"],
            config.candidate_k,
            config.slate_k,
        ),
        "ranker": {
            "feature_names": FEATURE_NAMES,
            "weights": ranker.weights,
            "selected_l2": ranker.l2,
            "selected_blend_alpha": ranker.blend_alpha,
            "regularization_selection": regularization_grid,
            "blend_selection": blend_grid,
            "train_pairs": len(train_pairs.labels),
            "validation_pairs": len(validation_pairs.labels),
            "test_pairs": len(test_pairs.labels),
            "validation_pair_accuracy": pairwise_accuracy(ranker, validation_pairs),
            "test_pair_accuracy": pairwise_accuracy(ranker, test_pairs),
            "training_loss_start": ranker.train_loss[0],
            "training_loss_end": ranker.train_loss[-1],
        },
        "latency": evaluation.latency,
        "runtime_seconds": perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "evaluation_contract": {
            "split": "per-user timestamp order; movie ID breaks timestamp ties",
            "relevant_item": "held-out rating >= 4",
            "graded_relevance": "max(rating - 3, 0)",
            "unobserved_items": "assigned zero evaluation gain but not interpreted as dislikes",
            "ranker_users": "train, validation, and test users are disjoint",
            "retrieval_history": "historical prefix for every warm-start user; no held-out ratings",
            "judged_candidate_diagnostic": (
                "post-hoc ordering among later-rated Stage-1 candidates, with IDCG restricted "
                "to those judged candidates; not the primary full-catalog estimand"
            ),
        },
        "limitations": [
            "Offline explicit ratings are selectively observed; unrated movies are unknown, not negative feedback.",
            "The per-user chronological split is not a single global point-in-time cutoff.",
            "The bootstrap interval conditions on this fitted model, dataset, split, and seed; it is not causal uncertainty.",
            "Exact NumPy catalog scoring demonstrates architecture, not an ANN latency or production-scale claim.",
            "This benchmark does not establish online CTR, retention, revenue, fairness, or business lift.",
        ],
    }
    metrics = _jsonable(metrics)
    state = ExperimentState(
        data=data,
        temporal=temporal,
        split=split,
        store=store,
        retrieval=retrieval,
        ranker=ranker,
        metrics=metrics,
        demo_cases=evaluation.demo_cases,
    )
    _write_csv(output_dir / "recommendations.csv", evaluation.recommendation_rows)
    _write_csv(output_dir / "query_metrics.csv", evaluation.query_rows)
    _save_state(output_dir, state)
    local_files = (
        "demo_metadata.json",
        "demo_state.npz",
        "query_metrics.csv",
        "recommendations.csv",
    )
    metrics["local_artifact_manifest"] = {
        "privacy": "local-only; public hashes do not make row-level contents distributable",
        "files": {name: _file_contract(output_dir / name) for name in local_files},
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return state


def write_public_result(path: str | Path, metrics: dict[str, Any]) -> Path:
    """Write the canonical aggregate-only result consumed by both reports."""

    path = Path(path)
    path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
