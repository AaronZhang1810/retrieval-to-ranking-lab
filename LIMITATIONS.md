# Claims, Limitations, and Promotion Boundary

## Defensible project description

> Built a repeatable CPU-only two-stage recommendation benchmark
> on MovieLens 100K, combining weighted-SVD candidate retrieval with a fixed
> popularity prior, observed pairwise RankNet, a validation-selected residual
> guardrail, stage-specific metrics, and paired user-bootstrap uncertainty.

Numeric claims should identify the dataset, configuration, and offline setting.
For example:

> On MovieLens 100K reference configuration/data ID `30a640be674a`, weighted SVD retrieval
> achieved 0.1537 NDCG@10 and the guarded challenger achieved 0.1548; the paired change was
> +0.0010 with a descriptive 95% percentile user-bootstrap interval of
> [-0.0006, +0.0030].

That statement does **not** establish a reliable improvement. The interval crosses
zero, and the benchmark is developmental rather than a preregistered confirmatory
study.

## What the repository supports

- The code parses real historical MovieLens ratings and creates a deterministic,
  within-user timestamp-ordered history/holdout split.
- Stage 1 learns a positive-preference weighted-SVD representation, adds a fixed
  standardized popularity prior, and excludes every earlier-rated movie from
  recommendation.
- Stage 2 only permutes Stage-1 candidates; the implementation verifies rather
  than obscures that gate.
- Ranker pairs use observed higher-vs-lower later ratings from rank-training users,
  not invented negative outcomes for unobserved movies.
- Supervised rank-training, validation, and development-test roles are disjoint. Development-
  test users' later outcomes are excluded from fitting and tuning in code, while
  their earlier histories remain available in the transductive warm-start design.
- The configuration-declared `0`–`0.10` blend grid bounds the learned score's
  coefficient, and validation NDCG selects the blend. It does not guarantee a
  fixed amount of rank preservation.
- The aggregate report exposes an inconclusive guarded result and an auxiliary/
  primary mismatch: 64.7% development-test pair accuracy coexists with lower unblended
  end-to-end NDCG.
- `RESULTS.json` records configuration, seed, parsed-input and executable-source
  fingerprints, the runtime-lock digest, model choices, paired metrics, local-
  artifact integrity contracts, and descriptive local timing.
- Dataset acquisition is explicit, checksum-pinned, safely extracted, and kept
  outside Git under the separate GroupLens terms described in [DATA.md](DATA.md).

## Claims the repository does not support

- a reliable Stage-2 improvement from the reference run;
- live engagement, retention, satisfaction, revenue, or other business impact;
- a causal effect or an online randomized-experiment conclusion;
- unbiased evaluation under the historical exposure policy;
- calibrated response probabilities from the RankNet item scores;
- new-user or new-item cold-start performance;
- fairness across demographic groups, privacy guarantees, or safety outcomes;
- large-catalog, distributed, or approximate-nearest-neighbor performance;
- a service-level latency, throughput, availability, or cost claim;
- robustness across datasets, time periods, repeated splits, or random seeds;
- causal feature importance from additive `weight x standardized value` terms.

## 1. Selective explicit ratings

MovieLens records ratings that users chose to provide. It does not provide the full
set of impressions, positions, skipped recommendations, or exposure propensities.
An unrated movie may be unseen, seen but ignored, or deliberately not rated.

The evaluation assigns zero gain to unobserved movies so that NDCG and recall are
computable. This is a metric convention, not a negative label. It can favor methods
that surface frequently rated movies and cannot distinguish lack of exposure from
lack of preference.

## 2. Chronology is local to each user

The split freezes an earlier prefix and later suffix separately for every user.
This prevents a user's own later rating from entering that user's history, but it
is not a global calendar cutoff. Global item statistics and the collaborative
matrix may include another user's event that occurred after the first user's
holdout event.

A deployment study would require event-time features and point-in-time joins under
one or more global cutoffs. The current design should be described as
within-user chronological, not globally time-consistent.

## 3. The estimand is warm-start and transductive

All evaluated users have at least 15 earlier ratings. Their earlier prefixes enter
the same weighted-SVD fit and their history-derived profiles are available at
scoring time. Validation and development-test **later outcomes** are excluded from
fitting and tuning, but the users themselves are not absent from the collaborative
matrix.

This is appropriate for a warm-start ranking question. It does not measure
inductive generalization to a new account. Every movie also belongs to the
historical catalog, so new-item cold start is outside scope.

## 4. Eligibility changes the evaluated population

Of 943 users, 885 are retained because their holdout suffix contains at least one
4- or 5-star movie and at least two distinct rating levels. The rule ensures that
recall and observed pair comparisons are defined, but excludes 58 users with less
informative suffixes. Results therefore apply to this eligible warm-start subset,
not automatically to all MovieLens users.

## 5. Pairwise supervision is doubly selected

A training pair exists only when both movies:

1. were explicitly rated in the later suffix; and
2. passed the Stage-1 top-200 candidate gate.

Pairs must also differ by at least one star, and each user contributes at most 80.
The learned comparison is therefore conditional on historical rating behavior and
the retriever's policy. No inverse-propensity weighting, randomized exploration,
doubly robust evaluation, or correction for candidate-selection bias is included.

Feature scaling is likewise fit on observed, candidate-gated later rows from
rank-training users. This remains isolated from validation/test users, but it is
not representative of all catalog items.

## 6. Pairwise accuracy is not the primary end-to-end metric

The unblended RankNet reached 64.7% accuracy on sampled development-test pairs. A
post-hoc candidate-restricted judged NDCG separately compares its ordering with
retrieval inside later-rated Stage-1 candidates. Its end-to-end NDCG@10 point
estimate was 0.1172, below weighted SVD retrieval's 0.1537.

This is not contradictory. Pairwise accuracy weights observed candidate
comparisons, while delivered-slate NDCG evaluates a top-10 ordering against sparse
later judgments and penalizes promoted unjudged items through the zero-gain
convention. The candidate-restricted judged diagnostic removes unjudged items and
recomputes IDCG within later-rated Stage-1 candidates. It conditions on having at
least two such candidates, is selection-biased and post hoc, and has no uncertainty
interval; it should not replace the primary metric.

The guarded blend reduces this model risk, but the selected `alpha=0.10` is a
design constraint, not proof that 10% is generally optimal.

## 7. Retrieval remains a hard boundary

Mean per-user Candidate Recall@200 is 65.9% in the reference evaluation. No
reranker can place a relevant movie that Stage 1 omitted. The label-assisted within-candidate ceiling
reports headroom after this gate, but it uses later ratings and is not deployable.
It is also not a full-relevance oracle because most catalog movies are unjudged.

The weighted-SVD retriever itself is a compact educational baseline. The positive
threshold, IDF weighting, rank 32, and small popularity term are modeling choices,
not a comparison against a comprehensive retrieval search.

## 8. Validation and uncertainty are developmental

Regularization is selected from four values using validation pairwise accuracy;
the residual blend is then selected from a configuration-declared five-value grid
using the same validation users' NDCG@10. These are sequential selection criteria,
not unbiased validation performance estimates. Development-test later outcomes are
excluded from both choices in code.

Nevertheless, this public portfolio benchmark is inspectable and iterated during
development; its uncertainty is descriptive rather than confirmatory. One split
and one seed do not measure variation from alternative user allocations, model
fits, hyperparameter searches, or datasets.

The percentile bootstrap resamples the 177 development-test users and preserves the
paired method comparison. Its interval is conditional on the fixed dataset,
split, fitted model, feature construction, configuration, and seed. It is
descriptive uncertainty over users in this evaluation, not causal, training, or
policy uncertainty.

The stored fraction of bootstrap replicates above zero is a descriptive resample
fraction, not a posterior probability or p-value. Candidate-width curves, activity
slices, score-zeroing sensitivity, the judged diagnostic, and the observed-label
ceiling are exploratory development-test summaries without multiplicity adjustment.

## 9. Feature and model capacity are intentionally narrow

The ranker is linear and uses six features derived from historical ratings, genre,
release year, and popularity. Missing release years are median-filled. Movie text,
creators, sequences, session context, recency at request time, and uncertainty in
the user profile are absent. Demographic attributes are intentionally unused.

The displayed contributions sum exactly to the guarded score and make the model
auditable. Correlation among features, selection into the training sample, and
regularization mean those terms are associative score decompositions, not causal
explanations of preference.
The score-zeroing sensitivity does not refit the model and is likewise not a
causal or model-reliance estimate.

## 10. Systems measurements are local demonstrations

The catalog is only 1,682 movies. Retrieval scans it exactly in one local NumPy
process, and Stage 2 computes six feature values for each candidate. Recorded
runtime is useful for local comparison and regression detection on that machine,
not for extrapolating to a larger index or concurrent traffic. Fixed seeds and
tie-breaking do not guarantee bit-for-bit identity across NumPy/BLAS platforms.

The browser trace is a read-only loopback tool over locally serialized model
state. The project does not implement authentication, multi-tenant isolation,
online feature freshness, an index lifecycle, distributed caches, observability,
canarying, drift response, or rollback.

## Stronger next experiments

The highest-value next step is an impression-aware dataset with event timestamps,
display position, and randomized exploration or logged propensities. That would
support better separation of non-exposure from non-engagement and align training
with the evaluated policy.

Further work could then add:

- global time cutoffs and point-in-time feature joins;
- repeated temporal splits or nested validation to measure development variance;
- retrieval baselines such as implicit ALS, BPR, or a content-aware two-tower
  model, with candidate-recall slices;
- listwise or calibrated downstream objectives evaluated against the same
  end-to-end metric;
- new-user and new-item protocols with content fallbacks;
- an ANN index with recall/latency/load measurements at realistic catalog scale;
- demographic and item-group audits where lawful, meaningful attributes exist;
- an online shadow/canary experiment with monitoring and a rollback criterion.

Until those extensions exist, the strongest conclusion is methodological: the
project shows why candidate recall, label selection, auxiliary-target mismatch,
and validation guardrails must be examined together before promoting a reranker.
