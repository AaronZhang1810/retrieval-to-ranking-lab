#!/usr/bin/env python3
"""Run the reproducible MovieLens two-stage recommendation benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import webbrowser
from pathlib import Path

from download_data import (
    OFFICIAL_PARSED_FINGERPRINT,
    DataAcquisitionError,
    validate_official_dataset,
)
from recsys_lab.data import DEFAULT_DATA_DIR
from recsys_lab.experiment import ExperimentConfig, run_experiment, write_public_result
from recsys_lab.report import write_reports


ROOT = Path(__file__).parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR, help="Extracted ml-100k directory.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--quick", action="store_true", help="Use fewer factors/pairs/bootstrap draws.")
    parser.add_argument(
        "--publish-reports",
        action="store_true",
        help="replace checked aggregate outputs after a full reference run",
    )
    parser.add_argument("--open", action="store_true", help="Open the generated HTML report.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick and args.publish_reports:
        raise SystemExit("Refusing to publish aggregate reports from a quick smoke run.")
    if args.publish_reports and args.seed != ExperimentConfig().seed:
        raise SystemExit("Refusing to publish reports from a non-reference seed.")
    if args.publish_reports:
        try:
            validate_official_dataset(args.data)
        except DataAcquisitionError as exc:
            raise SystemExit(f"Refusing to publish from unverified data: {exc}") from exc
    config = ExperimentConfig(seed=args.seed)
    if args.quick:
        config = ExperimentConfig(
            seed=args.seed,
            retrieval_rank=16,
            candidate_k=100,
            max_pairs_per_user=30,
            bootstrap_replicates=300,
        )
    state = run_experiment(args.output, config, args.data)
    if args.publish_reports:
        try:
            validate_official_dataset(args.data)
        except DataAcquisitionError as exc:
            raise SystemExit(f"Refusing to publish after a data integrity change: {exc}") from exc
        if state.metrics["data"]["source_fingerprint"] != OFFICIAL_PARSED_FINGERPRINT:
            raise SystemExit("Refusing to publish: parsed data fingerprint is not the pinned reference.")
    local_markdown, local_html = write_reports(
        args.output / "REPORT.md",
        args.output / "report.html",
        state.metrics,
    )
    metrics = state.metrics["metrics"]
    delta = state.metrics["paired_user_bootstrap"]["two_stage_minus_retrieval_ndcg_at_10"]
    print(f"Local artifacts: {args.output.resolve()}")
    print(
        "NDCG@10: weighted SVD "
        f"{metrics['retrieval']['ndcg_at_10']:.4f} -> guarded Stage 2 {metrics['two_stage']['ndcg_at_10']:.4f}"
    )
    print(
        f"Paired user-bootstrap delta {delta['estimate']:+.4f} "
        f"(95% interval {delta['ci_low']:+.4f}, {delta['ci_high']:+.4f})"
    )
    print(f"Local Markdown report: {local_markdown.resolve()}")
    print(f"Local HTML report: {local_html.resolve()}")
    if args.publish_reports:
        result_path = write_public_result(ROOT / "RESULTS.json", state.metrics)
        markdown_path, html_path = write_reports(
            ROOT / "REPORT.md",
            ROOT / "docs" / "index.html",
            state.metrics,
        )
        print(f"Canonical aggregate result: {result_path.resolve()}")
        print(f"GitHub report: {markdown_path.resolve()}")
        print(f"Pages report: {html_path.resolve()}")
    if args.open:
        webbrowser.open(local_html.resolve().as_uri())


if __name__ == "__main__":
    main()
