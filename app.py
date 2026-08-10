#!/usr/bin/env python3
"""Local, read-only browser trace for the MovieLens two-stage pipeline."""

from __future__ import annotations

import argparse
import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from recsys_lab.data import GENRES
from recsys_lab.evaluation import rank_user
from recsys_lab.experiment import ExperimentState, load_state
from recsys_lab.ranking import FEATURE_NAMES


ROOT = Path(__file__).parent
ALLOWED_K = (50, 100, 200, 400)


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite model output")
    return value


def recommendation_payload(
    state: ExperimentState,
    user: int,
    candidate_k: int,
) -> dict[str, object]:
    """Produce an auditable local trace; Stage 2 is a candidate permutation."""

    result = rank_user(
        state.data,
        state.temporal,
        state.store,
        state.retrieval,
        state.ranker,
        user,
        candidate_k,
    )
    candidates = np.asarray(result["candidates"], dtype=int)
    two_stage = np.asarray(result["two_stage"], dtype=int)
    relevant = np.asarray(result["relevant_items"], dtype=int)
    candidate_set = set(candidates.tolist())
    retrieval_rank = {int(item): position + 1 for position, item in enumerate(candidates)}
    scores = np.asarray(result["retrieval_scores"])
    rank_scores = np.asarray(result["rank_scores"])
    contributions = np.asarray(result["contributions"])
    observed_mask = np.asarray(result["observed_mask"], dtype=bool)
    observed_rating = np.asarray(result["observed_rating"], dtype=float)

    def row(item: int, final_rank: int | None = None) -> dict[str, object]:
        index = int(np.flatnonzero(candidates == item)[0])
        parts = [
            {"feature": name, "value": _finite(value)}
            for name, value in zip(FEATURE_NAMES, contributions[index])
        ]
        observed = bool(observed_mask[item])
        payload: dict[str, object] = {
            "item": int(item),
            "item_id": state.data.item_ids[item],
            "name": state.data.item_names[item],
            "year": None if not np.isfinite(state.data.item_year[item]) else int(state.data.item_year[item]),
            "genres": [genre for genre, value in zip(GENRES, state.data.item_genre[item]) if value > 0],
            "retrieval_rank": retrieval_rank[item],
            "retrieval_score": _finite(scores[index]),
            "rank_score": _finite(rank_scores[index]),
            "heldout_rating": _finite(observed_rating[item]) if observed else None,
            "contributions": parts,
            "contribution_sum": _finite(sum(float(part["value"]) for part in parts)),
        }
        if final_rank is not None:
            payload["final_rank"] = final_rank
            payload["rank_change"] = retrieval_rank[item] - final_rank
        return payload

    missed = [int(item) for item in relevant if int(item) not in candidate_set]
    return {
        "user": user,
        "user_id": state.data.user_ids[user],
        "eligible_catalog": int((~state.retrieval.seen[user]).sum()),
        "candidate_k": len(candidates),
        "candidate_recall": len(candidate_set.intersection(relevant.tolist())) / max(len(relevant), 1),
        "missed_relevant": [
            {"item_id": state.data.item_ids[item], "name": state.data.item_names[item]} for item in missed
        ],
        "retrieval_top": [row(int(item)) for item in candidates[:15]],
        "final_top": [row(int(item), rank + 1) for rank, item in enumerate(two_stage[:10])],
        "stage_2_is_candidate_permutation": set(two_stage.tolist()) == candidate_set,
        "blend_alpha": state.ranker.blend_alpha,
    }


def config_payload(state: ExperimentState) -> dict[str, object]:
    users = []
    best_user = int(state.split["test"][0])
    best_recall = -1.0
    for user in state.split["test"]:
        user = int(user)
        top_genres = np.argsort(state.store.user_genre[user])[::-1][:2]
        history_count = int(state.temporal.history_seen[user].sum())
        users.append(
            {
                "index": user,
                "id": state.data.user_ids[user],
                "taste": " + ".join(GENRES[index] for index in top_genres),
                "history_count": history_count,
            }
        )
        result = rank_user(
            state.data,
            state.temporal,
            state.store,
            state.retrieval,
            state.ranker,
            user,
            200,
        )
        candidates = set(np.asarray(result["candidates"]).tolist())
        relevant = np.asarray(result["relevant_items"]).tolist()
        recall = len(candidates.intersection(relevant)) / max(len(relevant), 1)
        if recall > best_recall:
            best_recall = recall
            best_user = user
    return {
        "users": users,
        "candidate_widths": ALLOWED_K,
        "feature_names": FEATURE_NAMES,
        "default_user": best_user,
        "default_k": 200,
        "dataset": state.metrics["data"]["dataset"],
    }


def make_handler(
    state: ExperimentState,
    index_html: bytes,
) -> type[BaseHTTPRequestHandler]:
    normalized_hosts = {"127.0.0.1", "localhost", "::1"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "RecsysLab/0.2"

        def _headers(self, content_type: str, length: int, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
            self.end_headers()

        def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._headers("application/json; charset=utf-8", len(body), status)
            self.wfile.write(body)

        def _host_is_allowed(self) -> bool:
            raw = self.headers.get("Host", "")
            parsed = urlparse(f"//{raw}")
            return (
                parsed.username is None
                and parsed.password is None
                and parsed.hostname is not None
                and parsed.hostname.lower().rstrip(".") in normalized_hosts
            )

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_is_allowed():
                self._send_json({"error": "invalid host header"}, HTTPStatus.BAD_REQUEST)
                return
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._headers("text/html; charset=utf-8", len(index_html), HTTPStatus.OK)
                self.wfile.write(index_html)
                return
            if parsed.path == "/api/config":
                self._send_json(config_payload(state))
                return
            if parsed.path == "/api/recommend":
                try:
                    query = parse_qs(parsed.query)
                    user = int(query.get("user", [str(int(state.split["test"][0]))])[0])
                    candidate_k = int(query.get("k", ["200"])[0])
                    if user not in set(np.asarray(state.split["test"], dtype=int).tolist()):
                        raise ValueError("user must belong to the development-test set")
                    if candidate_k not in ALLOWED_K:
                        raise ValueError(f"k must be one of {', '.join(map(str, ALLOWED_K))}")
                    self._send_json(recommendation_payload(state, user, candidate_k))
                except (TypeError, ValueError, IndexError) as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            self._send_json({"error": "method not allowed"}, HTTPStatus.METHOD_NOT_ALLOWED)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[recsys] {self.address_string()} {format % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Refusing a non-loopback bind; this unauthenticated trace is local-only.")
    required = (
        args.artifacts / "experiment.json",
        args.artifacts / "demo_state.npz",
        args.artifacts / "demo_metadata.json",
    )
    if not all(path.exists() for path in required):
        raise SystemExit("Artifacts are missing. Run `python run_demo.py` first.")
    try:
        state = load_state(args.artifacts)
    except ValueError as exc:
        raise SystemExit(f"Artifacts failed integrity validation: {exc}") from exc
    index_html = (ROOT / "static" / "index.html").read_bytes()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(state, index_html),
    )
    print(f"Two-stage recommender: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
