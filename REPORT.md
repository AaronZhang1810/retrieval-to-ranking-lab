# Benchmark Report: Two-Stage Recommendation on MovieLens 100K

> Configuration/data ID `30a640be674a` · real explicit ratings · CPU-only · no raw MovieLens data committed

## Executive result

The benchmark uses **100,000 ratings from 943 numbered user records across 1,682 movies**. A per-user timestamp-ordered split freezes 79,619 history ratings and holds out 20,381 later ratings. The development-test evaluation contains 177 users whose later outcomes were never used to fit or tune the ranker.

| Method | NDCG@10 | Recall@10 | HitRate@10 | Catalog coverage@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.0744 | 0.0649 | 38.4% | 3.2% |
| Stage 1 · weighted SVD | 0.1537 | 0.1365 | 53.1% | 21.5% |
| Stage 2 · guarded blend | 0.1548 | 0.1374 | 53.1% | 21.0% |
| Unblended RankNet diagnostic | 0.1172 | 0.0982 | 44.6% | 11.0% |
| Observed-label ceiling in candidates* | 0.7681 | 0.6095 | 94.9% | 29.0% |

\*The ceiling sorts candidates with held-out labels and is not deployable. It measures headroom inside the candidate set.

NDCG, recall, and hit rate are macro-averages over 177 development-test users. Catalog coverage is the fraction of all 1,682 movies appearing in their top-10 slates.

The guarded Stage-2 point estimate changed NDCG@10 by **+0.0010** (paired user percentile-bootstrap 95% interval **[-0.0006, +0.0030]**; 177 users, 3,000 resamples). The conditional interval crosses zero; this run does not establish the direction of the Stage-2 difference.

## What the experiment found

1. **The candidate gate is a material hard limit.** At `K=200`, mean per-user candidate recall was 65.9% for held-out movies rated 4–5 stars. A reranker cannot recover relevant movies outside that gate.
2. **Pairwise accuracy did not guarantee end-to-end lift.** The unblended RankNet reached 64.7% accuracy on sampled development-test pairs, while its end-to-end NDCG@10 point estimate was 0.1172 versus 0.1537 for weighted SVD.
3. **The post-hoc judged-only diagnostic answers a narrower question.** Among users with at least two later-rated Stage-1 candidates, candidate-restricted judged NDCG@10 changed from 0.7350 to 0.7622. That point estimate has no interval and is intentionally secondary: it conditions on both later rating and Stage-1 selection.
4. **A validation-selected coefficient constrained the challenger.** The final score uses `α=0.100` from the configuration-declared 0–0.10 grid. This bounds the learned score's coefficient but does not guarantee preservation of any fixed fraction of Stage-1 positions.

This is a useful inconclusive result: it shows why a plausible reranker must be evaluated through the candidate gate with the primary end-to-end offline endpoint, rather than promoted from pairwise accuracy alone.

## Candidate-width trade-off

| K | Candidate recall | SVD NDCG@10 | Guarded NDCG@10 | Feature scores/query |
|---:|---:|---:|---:|---:|
| 50 | 37.2% | 0.1537 | 0.1548 | 300 |
| 100 | 50.5% | 0.1537 | 0.1548 | 600 |
| 200 | 65.9% | 0.1537 | 0.1548 | 1,200 |
| 400 | 80.1% | 0.1537 | 0.1548 | 2,400 |

Increasing `K` improves candidate recall but linearly increases rich feature evaluation. The SVD top-10 does not change with `K`; the curve exposes the coverage/cost boundary the reranker inherits. This is an exploratory development-test curve and was not used to select `K`.

## Model and validation

Stage 1 applies a low-rank decomposition to positive preference mass from the frozen history. Stage 2 fits a linear RankNet/Bradley–Terry objective on later, observed higher-vs-lower rating pairs from rank-training users. All features are computed from the frozen history:

| Feature | Standardized weight |
|---|---:|
| retrieval affinity | +0.2175 |
| genre affinity | +0.0747 |
| Bayesian item mean | +0.3451 |
| log rating count | +0.0485 |
| release-year affinity | -0.0330 |
| popularity affinity | -0.0202 |

Regularization was selected on 159 validation users; the selected `λ=0.2` produced 66.6% validation-pair accuracy. The blend was then selected on end-to-end validation NDCG@10 using the same validation group. Those validation scores are selection criteria, not unbiased performance estimates. Development-test later outcomes were used only for reporting.

## Leakage and evaluation contract

```text
each user's timestamp-ordered ratings
        │
        ├── earlier prefix → SVD + item statistics + user profiles (frozen)
        └── later suffix  → observed pair labels / final offline relevance
                              │
eligible users ──→ rank-train / validation / development-test (disjoint)
```

- A held-out movie is relevant at rating ≥4; graded relevance is `max(rating − 3, 0)`.
- Unrated movies receive zero *evaluation gain* but are described as unjudged, not disliked.
- The split is chronological within user; movie ID deterministically breaks same-second timestamp ties.
- Retrieval is transductive warm-start: every evaluated user contributes only their earlier history.
- NDCG and recall compare methods on the same held-out users; Stage 2 only permutes Stage-1 candidates.
- The paired bootstrap resamples users and is conditional on this dataset, split, fitted model, and seed.
- The bootstrap fraction above zero in RESULTS.json is neither a posterior probability nor a p-value.

## Reproducibility record

- Dataset: MovieLens 100K, 1997-09-20 through 1998-04-22
- Parsed `u.data` + `u.item` fingerprint: `ec3f1ed75ed74ee8ed7a7aa17efbcd206de908383c04d8424ee5de6a585dbb05`
- Configuration/data SHA-256: `30a640be674a8557a8ab7077da561dd2640c9713ea2b3af8d617a73725040f9b`
- Result schema: `1`
- Python `3.12.5`; NumPy `2.2.6`
- Local descriptive runtime: `4.82` seconds
- Executable source-tree SHA-256: `f4369f7333593201046f6b4d69ce52ebf15171b8afcd9d3f534c3cded74e0df1`
- Runtime-lock SHA-256: `573749f545b2b39ae2f35dbcb5d42416f36f75eaff2af8ca941b528757390ec3`
- RESULTS.json is the canonical aggregate result; row-level outputs and derived model state stay local

Run `python download_data.py --accept-license && python run_demo.py --publish-reports` to regenerate the checked reports. Dataset terms and citation are in [DATA.md](DATA.md); model derivations are in [MATH.md](MATH.md).

## Limitations and next experiment

- Offline explicit ratings are selectively observed; unrated movies are unknown, not negative feedback.
- The per-user chronological split is not a single global point-in-time cutoff.
- The bootstrap interval conditions on this fitted model, dataset, split, and seed; it is not causal uncertainty.
- Exact NumPy catalog scoring demonstrates architecture, not an ANN latency or production-scale claim.
- This benchmark does not establish online CTR, retention, revenue, fairness, or business lift.
- The candidate-restricted judged diagnostic is post hoc, selection-biased, and not the primary endpoint.
- This is a development benchmark, not a preregistered confirmatory study.

The next useful experiment is impression-aware training with randomized exposure or propensity information. That could help distinguish unexposed movies from exposed non-engagement and align reranker training more closely with the evaluated policy distribution.

## Data acknowledgment

This analysis uses the MovieLens 100K dataset from GroupLens Research. The University of Minnesota and GroupLens do not endorse this project. Code is MIT-licensed; MovieLens data is governed separately and is not redistributed here.
