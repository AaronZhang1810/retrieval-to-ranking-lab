"""Stage-specific evaluation on chronologically held-out MovieLens ratings."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .data import GENRES, MovieLensData, TemporalSplit, observed_relevance
from .metrics import intralist_diversity, ndcg_at_k, recall_at_k, reciprocal_rank
from .ranking import FeatureStore, PairwiseRanker, item_features
from .retrieval import RetrievalModel


@dataclass(frozen=True)
class EvaluationResult:
    query_rows: list[dict[str, float | int | str]]
    recommendation_rows: list[dict[str, float | int | str]]
    summaries: dict[str, dict[str, float]]
    candidate_curve: list[dict[str, float | int]]
    judged_candidate_diagnostic: dict[str, dict[str, float]]
    latency: dict[str, float]
    demo_cases: list[dict[str, object]]


METHODS = (
    "popularity",
    "retrieval",
    "two_stage",
    "unblended_ranker",
    "observed_ceiling_in_candidates",
)


def rank_user(
    data: MovieLensData,
    temporal: TemporalSplit,
    store: FeatureStore,
    retrieval: RetrievalModel,
    ranker: PairwiseRanker,
    user: int,
    candidate_k: int,
) -> dict[str, object]:
    """Rank one warm-start user and expose the exact Stage-1 gate."""

    candidates, retrieval_scores = retrieval.candidates(user, candidate_k)
    raw_features = item_features(data, store, retrieval, user, candidates, retrieval_scores)
    rank_scores = ranker.score(raw_features)
    pure_rank_scores = ranker.pure_score(raw_features)
    rank_order = np.lexsort((candidates, -rank_scores))
    pure_rank_order = np.lexsort((candidates, -pure_rank_scores))
    relevance = observed_relevance(temporal.holdout_ratings[user])
    observed_rating = temporal.holdout_ratings[user].astype(float)
    relevant_items = np.flatnonzero(relevance > 0)
    candidate_relevance = relevance[candidates]
    ceiling_order = np.lexsort((candidates, -retrieval_scores, -candidate_relevance))

    eligible = np.flatnonzero(~retrieval.seen[user])
    popularity_order = np.lexsort((eligible, -store.item_log_count[eligible]))
    return {
        "candidates": candidates,
        "retrieval_scores": retrieval_scores,
        "raw_features": raw_features,
        "rank_scores": rank_scores,
        "pure_rank_scores": pure_rank_scores,
        "contributions": ranker.contributions(raw_features),
        "relevance": relevance,
        "observed_rating": observed_rating,
        "observed_mask": temporal.holdout_mask[user],
        "relevant_items": relevant_items,
        "popularity": eligible[popularity_order],
        "retrieval": candidates,
        "two_stage": candidates[rank_order],
        "unblended_ranker": candidates[pure_rank_order],
        "observed_ceiling_in_candidates": candidates[ceiling_order],
    }


def _method_metrics(
    items: np.ndarray,
    relevance: np.ndarray,
    relevant_items: np.ndarray,
    observed_mask: np.ndarray,
    item_genre: np.ndarray,
    k: int,
) -> dict[str, float]:
    top = np.asarray(items, dtype=int)[:k]
    return {
        "ndcg_at_10": ndcg_at_k(top, relevance, k),
        "recall_at_10": recall_at_k(top, relevant_items, k),
        "hit_rate_at_10": float(bool(set(top.tolist()) & set(relevant_items.tolist()))),
        "mrr_at_10": reciprocal_rank(top, relevant_items, k),
        "mean_graded_gain": float(np.mean(relevance[top])) if len(top) else 0.0,
        "judged_fraction_at_10": float(np.mean(observed_mask[top])) if len(top) else 0.0,
        "intralist_diversity": intralist_diversity(top, item_genre, k),
    }


def evaluate(
    data: MovieLensData,
    temporal: TemporalSplit,
    store: FeatureStore,
    retrieval: RetrievalModel,
    ranker: PairwiseRanker,
    users: np.ndarray,
    candidate_k: int = 200,
    slate_k: int = 10,
    candidate_widths: tuple[int, ...] = (50, 100, 200, 400),
) -> EvaluationResult:
    query_rows: list[dict[str, float | int | str]] = []
    recommendation_rows: list[dict[str, float | int | str]] = []
    demo_cases: list[dict[str, object]] = []
    judged_rows: list[dict[str, float | int | str]] = []
    unique_recommended: dict[str, set[int]] = {method: set() for method in METHODS}

    for query_id, user in enumerate(np.asarray(users, dtype=int)):
        result = rank_user(data, temporal, store, retrieval, ranker, int(user), candidate_k)
        relevant_items = np.asarray(result["relevant_items"], dtype=int)
        candidates = np.asarray(result["candidates"], dtype=int)
        candidate_recall = recall_at_k(candidates, relevant_items, len(candidates))
        for method in METHODS:
            ranked = np.asarray(result[method], dtype=int)
            metrics = _method_metrics(
                ranked,
                np.asarray(result["relevance"]),
                relevant_items,
                np.asarray(result["observed_mask"], dtype=bool),
                data.item_genre,
                slate_k,
            )
            query_rows.append(
                {
                    "query_id": query_id,
                    "user": int(user),
                    "method": method,
                    "candidate_recall_at_k": candidate_recall,
                    "heldout_relevant_items": len(relevant_items),
                    **metrics,
                }
            )
            unique_recommended[method].update(ranked[:slate_k].tolist())

        observed_mask = np.asarray(result["observed_mask"], dtype=bool)
        observed_candidates = candidates[observed_mask[candidates]]
        if len(observed_candidates) >= 2:
            candidate_relevance = np.zeros_like(np.asarray(result["relevance"], dtype=float))
            candidate_relevance[observed_candidates] = np.asarray(result["relevance"], dtype=float)[
                observed_candidates
            ]
            for method in ("retrieval", "two_stage", "unblended_ranker"):
                ranked = np.asarray(result[method], dtype=int)
                ranked_observed = ranked[observed_mask[ranked]]
                judged_rows.append(
                    {
                        "user": int(user),
                        "method": method,
                        "ndcg_at_10": ndcg_at_k(ranked_observed, candidate_relevance, slate_k),
                    }
                )

        retrieval_position = {int(item): position + 1 for position, item in enumerate(candidates)}
        two_stage = np.asarray(result["two_stage"], dtype=int)
        for position, item in enumerate(two_stage[:slate_k], start=1):
            candidate_index = int(np.flatnonzero(candidates == item)[0])
            observed = bool(np.asarray(result["observed_mask"])[item])
            recommendation_rows.append(
                {
                    "query_id": query_id,
                    "user_id": data.user_ids[user],
                    "item_id": data.item_ids[item],
                    "item_name": data.item_names[item],
                    "retrieval_rank": retrieval_position[int(item)],
                    "final_rank": position,
                    "rank_change": retrieval_position[int(item)] - position,
                    "retrieval_score": float(np.asarray(result["retrieval_scores"])[candidate_index]),
                    "rerank_score": float(np.asarray(result["rank_scores"])[candidate_index]),
                    "heldout_rating": (
                        float(np.asarray(result["observed_rating"])[item]) if observed else "unobserved"
                    ),
                }
            )

        if len(demo_cases) < 12:
            candidate_set = set(candidates.tolist())
            missed = [int(item) for item in relevant_items if int(item) not in candidate_set]
            top_rows: list[dict[str, object]] = []
            contributions = np.asarray(result["contributions"])
            for position, item in enumerate(two_stage[:slate_k], start=1):
                candidate_index = int(np.flatnonzero(candidates == item)[0])
                observed = bool(np.asarray(result["observed_mask"])[item])
                top_rows.append(
                    {
                        "item": int(item),
                        "item_id": data.item_ids[item],
                        "name": data.item_names[item],
                        "year": None if not np.isfinite(data.item_year[item]) else int(data.item_year[item]),
                        "final_rank": position,
                        "retrieval_rank": retrieval_position[int(item)],
                        "rank_change": retrieval_position[int(item)] - position,
                        "retrieval_score": float(np.asarray(result["retrieval_scores"])[candidate_index]),
                        "rank_score": float(np.asarray(result["rank_scores"])[candidate_index]),
                        "heldout_rating": (
                            float(np.asarray(result["observed_rating"])[item]) if observed else None
                        ),
                        "genres": [
                            genre for genre, value in zip(GENRES, data.item_genre[item]) if value > 0
                        ],
                        "contributions": contributions[candidate_index].tolist(),
                    }
                )
            demo_cases.append(
                {
                    "query_id": query_id,
                    "user": int(user),
                    "user_id": data.user_ids[user],
                    "candidate_recall": candidate_recall,
                    "missed_relevant_items": missed,
                    "top_items": top_rows,
                }
            )

    summaries: dict[str, dict[str, float]] = {}
    metric_names = (
        "ndcg_at_10",
        "recall_at_10",
        "hit_rate_at_10",
        "mrr_at_10",
        "mean_graded_gain",
        "judged_fraction_at_10",
        "intralist_diversity",
        "candidate_recall_at_k",
    )
    for method in METHODS:
        rows = [row for row in query_rows if row["method"] == method]
        summaries[method] = {
            metric: float(np.mean([float(row[metric]) for row in rows])) for metric in metric_names
        }
        summaries[method]["catalog_coverage"] = len(unique_recommended[method]) / data.n_items

    candidate_curve: list[dict[str, float | int]] = []
    curve_users = np.asarray(users, dtype=int)
    for width in candidate_widths:
        recalls: list[float] = []
        retrieval_ndcg: list[float] = []
        rerank_ndcg: list[float] = []
        unblended_ndcg: list[float] = []
        for user in curve_users:
            result = rank_user(data, temporal, store, retrieval, ranker, int(user), width)
            candidates = np.asarray(result["candidates"], dtype=int)
            relevant = np.asarray(result["relevant_items"], dtype=int)
            relevance = np.asarray(result["relevance"])
            recalls.append(recall_at_k(candidates, relevant, len(candidates)))
            retrieval_ndcg.append(ndcg_at_k(np.asarray(result["retrieval"]), relevance, slate_k))
            rerank_ndcg.append(ndcg_at_k(np.asarray(result["two_stage"]), relevance, slate_k))
            unblended_ndcg.append(
                ndcg_at_k(np.asarray(result["unblended_ranker"]), relevance, slate_k)
            )
        candidate_curve.append(
            {
                "candidate_k": width,
                "candidate_recall": float(np.mean(recalls)),
                "retrieval_ndcg_at_10": float(np.mean(retrieval_ndcg)),
                "two_stage_ndcg_at_10": float(np.mean(rerank_ndcg)),
                "unblended_ranker_ndcg_at_10": float(np.mean(unblended_ndcg)),
                "stage_2_feature_scores": width * len(ranker.weights),
                "users": len(curve_users),
            }
        )

    benchmark_users = np.asarray(users, dtype=int)[: min(100, len(users))]
    start = perf_counter()
    for user in benchmark_users:
        retrieval.candidates(int(user), candidate_k)
    retrieval_ms = (perf_counter() - start) * 1000 / len(benchmark_users)
    start = perf_counter()
    for user in benchmark_users:
        candidates, scores = retrieval.candidates(int(user), candidate_k)
        features = item_features(data, store, retrieval, int(user), candidates, scores)
        ranker.score(features)
    total_ms = (perf_counter() - start) * 1000 / len(benchmark_users)
    latency = {
        "exact_retrieval_ms_per_user": retrieval_ms,
        "retrieval_plus_ranking_ms_per_user": total_ms,
        "incremental_ranking_ms_per_user": max(total_ms - retrieval_ms, 0.0),
        "benchmark_users": len(benchmark_users),
    }
    judged_candidate_diagnostic: dict[str, dict[str, float]] = {}
    for method in ("retrieval", "two_stage", "unblended_ranker"):
        rows = [row for row in judged_rows if row["method"] == method]
        judged_candidate_diagnostic[method] = {
            "ndcg_at_10": float(np.mean([float(row["ndcg_at_10"]) for row in rows])),
            "users_with_two_or_more_judged_candidates": float(len(rows)),
        }
    return EvaluationResult(
        query_rows=query_rows,
        recommendation_rows=recommendation_rows,
        summaries=summaries,
        candidate_curve=candidate_curve,
        judged_candidate_diagnostic=judged_candidate_diagnostic,
        latency=latency,
        demo_cases=demo_cases,
    )
