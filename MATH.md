# Mathematics, Data Contract, and Evaluation Design

This note defines the implemented benchmark rather than an idealized recommender.
It states which data enter each stage, what each metric estimates, and where the
interpretation stops. The aggregate output for the reference configuration is in
[REPORT.md](REPORT.md).

## 1. Decision problem

For warm-start user \(u\), the system must return an ordered slate of ten unseen
movies from \(I=1{,}682\) catalog items. The implementation separates the decision
into:

1. **retrieval:** score the catalog with a rank-32 latent model and retain
   \(K=200\) candidates, \(C_K(u)\);
2. **ranking:** evaluate \(p=6\) history-derived features for those candidates and
   permute them into the delivered top ten.

Stage 2 cannot recover a relevant movie outside \(C_K(u)\). Candidate recall is
therefore part of the estimand, not merely a systems detail. Because this catalog
is small, Stage 1 performs an exact NumPy scan; no approximate-nearest-neighbor
index or large-scale latency result is implied.

## 2. Dataset and temporal split

MovieLens 100K contains 100,000 explicit 1--5 ratings from 943 numbered user
records on 1,682 movies. Only `u.data` and movie title/year/genre fields from
`u.item` enter the model. The demographic file is deliberately ignored; no
anonymity or privacy guarantee is inferred from that choice. Source, checksum,
terms, and citation are documented in [DATA.md](DATA.md).

For each user independently, ratings are sorted by

\[
(\text{Unix timestamp},\ \text{dense movie index}).
\]

The movie index deterministically breaks same-second ties. If user \(u\) has
\(n_u\) ratings, the later suffix size is

\[
h_u=\min\!\left\{\max\!\left(4,\lceil 0.20n_u\rceil\right),\ n_u-15\right\}.
\]

The first \(n_u-h_u\) ratings form frozen history and the last \(h_u\) form later
outcomes. This yields 79,619 history ratings and 20,381 holdout ratings. It is a
**within-user chronological split**, not a single global cutoff: one user's
history may occur later in calendar time than another user's holdout.

A user is eligible for supervised ranking/evaluation only if the later suffix
contains at least one rating \(\ge 4\) and at least two distinct rating levels.
There are 885 eligible and 58 excluded users. A seeded 62%/18%/20% allocation gives
disjoint user groups:

| Role | Users | Use of later ratings |
|---|---:|---|
| Rank training | 549 | feature scaling and observed pair labels |
| Validation | 159 | regularization and blend selection |
| Development test | 177 | reported metrics and diagnostics only |

Development-test users' later outcomes are excluded from fitting and tuning in code. The
experiment is nevertheless **transductive warm-start**: the collaborative model
and history-derived feature store use every user's earlier history, including the
earlier prefixes of validation and test users. It does not estimate performance
for a brand-new user.

## 3. History signal and later relevance

Let \(R^{H}_{ui}\in\{1,\ldots,5\}\) be an earlier rating when one exists and zero
otherwise. The seen indicator is

\[
S_{ui}=\mathbf 1\{R^{H}_{ui}>0\}.
\]

All seen movies are excluded at recommendation time. Positive preference mass for
retrieval and user profiles is

\[
C_{ui}=\max(R^{H}_{ui}-2.5,0).
\]

Thus ratings 3, 4, and 5 contribute increasing positive mass; ratings 1 and 2 do
not become positive feedback, but they still remain seen and cannot be
recommended.

For a later rating \(R^{L}_{ui}\), graded evaluation relevance is

\[
r_{ui}=\max(R^{L}_{ui}-3,0),
\]

so a 4-star rating has relevance 1 and a 5-star rating has relevance 2. The binary
relevant set is \(G_u=\{i:R^{L}_{ui}\ge4\}\). Lower later ratings and unobserved
movies receive zero **evaluation gain**. An unobserved movie is unknown, not an
observed dislike; this bookkeeping convention is a major limitation of the
offline metric.

## 4. Stage 1: weighted SVD retrieval

Let

\[
\operatorname{df}_i=\sum_u \mathbf 1\{C_{ui}>0\},\qquad
\operatorname{idf}_i=\log\frac{U+1}{\operatorname{df}_i+1}+1,
\]

where \(U=943\). The weighted positive-preference matrix is

\[
A_{ui}=\log(1+C_{ui})\operatorname{idf}_i,
\qquad
X_{u:}=\frac{A_{u:}}{\max(\lVert A_{u:}\rVert_2,10^{-12})}.
\]

A randomized rank-\(d\) decomposition with \(d=32\), eight oversampling
dimensions, and two stabilized power iterations approximates

\[
X\approx U_d\Sigma_dV_d^\top.
\]

The symmetric factors are

\[
p_u=U_d[u,:]\Sigma_d^{1/2},\qquad
q_i=V_d[i,:]\Sigma_d^{1/2}.
\]

For each user, the exact catalog score is

\[
s^{(1)}_{ui}
=\frac{p_u^\top q_i}
{\operatorname{sd}_{j}(p_u^\top q_j)+10^{-8}}
+0.08\,z_i\!\left(\log(1+\operatorname{df}_i)\right).
\]

Seen movies receive score \(-\infty\). A full lexicographic sort by descending
score and ascending movie ID gives the first \(K=200\) unseen movies. This is an
inspectable weighted-SVD baseline with a fixed popularity prior, not implicit ALS,
BPR, a two-tower network, or an ANN service.

## 5. Frozen feature store and Stage-2 features

Every Stage-2 variable is estimated from earlier history. Let

\[
n_i=\sum_u S_{ui},\qquad
\bar R=\frac{\sum_{ui}R^H_{ui}}{\sum_i n_i}.
\]

The item-level statistics are

\[
\operatorname{BayesMean}_i
=\frac{\sum_uR^H_{ui}+20\bar R}{n_i+20},
\qquad
\operatorname{LogCount}_i=\log(1+n_i).
\]

Movie release year is median-filled when missing and standardized across items.
Log count is also standardized to form the popularity coordinate.

Let \(g_i\) be the normalized 19-dimensional genre vector. A user's positive
history weights are \(W_{ui}=C_{ui}\); if a user has zero positive mass, the seen
mask is used as a fallback. With total mass \(m_u=\sum_iW_{ui}\), history-weighted
global genre prior \(\bar g\), and prior mass \(a=5\),

\[
\widehat g_u
=\frac{\sum_i W_{ui}g_i+a\bar g}{m_u+a},
\]

followed by sum normalization. For release year and popularity, the global prior
coordinates are item-count-weighted means,

\[
\bar y=\frac{\sum_i(n_i+10^{-6})z(\operatorname{year}_i)}
{\sum_i(n_i+10^{-6})},\qquad
\bar p=\frac{\sum_i(n_i+10^{-6})z(\operatorname{popularity}_i)}
{\sum_i(n_i+10^{-6})}.
\]

The corresponding user coordinates use positive-preference weights (or the seen-
mask fallback) and the same prior mass:

\[
\widehat y_u=\frac{\sum_iW_{ui}z(\operatorname{year}_i)+a\bar y}{m_u+a},
\qquad
\widehat p_u=\frac{\sum_iW_{ui}z(\operatorname{popularity}_i)+a\bar p}{m_u+a}.
\]

For candidate \(i\in C_K(u)\), the six raw ranking features are

\[
x_{ui}=\begin{bmatrix}
s^{(1)}_{ui}\\
g_i^\top\widehat g_u\\
\operatorname{BayesMean}_i\\
\operatorname{LogCount}_i\\
-|z(\operatorname{year}_i)-\widehat{\operatorname{year}}_u|\\
-|z(\operatorname{popularity}_i)-\widehat{\operatorname{popularity}}_u|
\end{bmatrix}.
\]

Feature means and standard deviations are fit on candidate-gated, later-observed
movie rows from rank-training users only. With a standard-deviation floor of
\(10^{-6}\),

\[
z_{ui,j}=\frac{x_{ui,j}-\mu_j}{\max(\sigma_j,10^{-6})}.
\]

Validation and development-test rows do not determine \(\mu_j\) or \(\sigma_j\).

## 6. Observed pair construction and RankNet

Within each supervised user's top-200 candidate set, the code considers pairs of
movies that both have later ratings and satisfy

\[
|R^L_{ui}-R^L_{uj}|\ge1.
\]

At most 80 pairs are sampled per user, so a user with more qualifying pairs can
still receive more weight than a user with fewer pairs. Pair orientation is
randomized; if
\(y_{uij}=+1\), the left movie has the higher later rating. The standardized
difference is \(\Delta z_{uij}=z_{ui}-z_{uj}\). The reference run contains 19,585
training pairs, 6,129 validation pairs, and 5,775 development-test diagnostic pairs.

The linear Bradley--Terry/RankNet model is

\[
P_w(i\succ j\mid u)=\sigma(w^\top\Delta z_{uij}),
\qquad
\sigma(t)=\frac{1}{1+e^{-t}}.
\]

For \(M\) training pairs, the regularized objective is

\[
\mathcal L(w)=\frac1M\sum_{m=1}^M
\log\!\left(1+\exp[-y_m w^\top\Delta z_m]\right)
+\frac\lambda2\lVert w\rVert_2^2.
\]

The implementation evaluates the logistic term stably with `logaddexp` and
optimizes all six weights with 700 full-batch Adam steps. No intercept is needed
because it cancels in a pairwise score difference. The grid

\[
\lambda\in\{0.001,0.01,0.05,0.2\}
\]

is selected by pairwise accuracy on the 159 validation users, with smaller
\(\lambda\) breaking exact ties. The selected value is \(\lambda=0.2\), producing
66.6% validation-pair accuracy and 64.7% development-test pairwise accuracy. These are
pair-level accuracy proportions on sampled, selectively observed candidate pairs,
not calibrated response probabilities or user-level success rates. Likewise, the
item score \(w^\top z_{ui}\) is not a calibrated response probability.

## 7. Guarded residual blend

Pairwise accuracy is not the primary end-to-end objective in this benchmark, so
the unblended score \(f_{ui}=w^\top z_{ui}\) is treated as a diagnostic. The
evaluated Stage-2 candidate score is

\[
s^{(2)}_{ui}(\alpha)
=(1-\alpha)z_{ui,1}+\alpha w^\top z_{ui}.
\]

The configuration declares the conservative grid

\[
\alpha\in\{0,0.025,0.05,0.075,0.10\}.
\]

After \(\lambda\) is selected, the code evaluates this grid on validation
end-to-end NDCG@10 and favors the smaller \(\alpha\) on exact ties. The same
validation user group therefore supports two sequential hyperparameter choices;
its scores are selection criteria, not unbiased performance estimates. The
selected value is \(\alpha=0.10\). Development-test outcomes do not enter either
selection step. Bounding \(\alpha\) constrains the coefficient on the learned
score; it does not guarantee that any fixed fraction of Stage-1 positions remains
unchanged.

The displayed additive contribution for retrieval affinity is

\[
(1-\alpha)z_{ui,1}+\alpha w_1z_{ui,1},
\]

and for feature \(j>1\) it is \(\alpha w_jz_{ui,j}\). The six contributions sum
exactly to \(s^{(2)}_{ui}\). They explain the fitted linear score algebraically;
they are not causal attributions.

## 8. End-to-end estimands

### Candidate recall

\[
\operatorname{CandidateRecall@K}(u)
=\frac{|C_K(u)\cap G_u|}{|G_u|}.
\]

At \(K=200\), the reference development-test mean is 65.9%. This is the maximum fraction
of observed relevant movies available to any Stage-2 permutation, although it is
not itself a top-10 metric.

### End-to-end NDCG@10 with catalog-level holdout normalization

For delivered order \(\pi_u\),

\[
\operatorname{DCG@10}(u)
=\sum_{\ell=1}^{10}
\frac{2^{r_{u,\pi_{u\ell}}}-1}{\log_2(\ell+1)}.
\]

`IDCG@10` sorts the entire later-relevance vector, not only retrieved candidates.
Then

\[
\operatorname{NDCG@10}(u)=
\frac{\operatorname{DCG@10}(u)}{\operatorname{IDCG@10}(u)}.
\]

The benchmark also reports top-10 recall over \(G_u\), whether the delivered top
ten contains at least one member of \(G_u\), reciprocal rank of the first such
movie, mean graded gain, judged fraction, genre-vector dissimilarity, and catalog
coverage.

### Label-assisted diagnostics

The **observed-label ceiling within candidates** sorts the retrieved set using
later relevance, with retrieval score and movie ID as tie-breakers. Its IDCG is
still computed from the complete later-relevance vector, so candidate misses
remain visible. It is nondeployable and does not reveal the relevance of unjudged
movies; it only exposes headroom under the benchmark's observed-label convention.

The **candidate-restricted judged diagnostic** removes unjudged candidates and
computes both DCG and IDCG from later-rated Stage-1 candidates. It asks whether the
ranker orders known judgments inside the fixed gate more sensibly, conditional on
there being at least two such candidates. It is computed post hoc without an
uncertainty interval. Because later rating and candidate inclusion are both
selective, this is a secondary, selection-biased ordering diagnostic rather than
the primary end-to-end endpoint.

## 9. Paired user bootstrap

For development-test user \(u\), let

\[
\Delta_u=
\operatorname{NDCG@10}_u(\text{guarded})
-\operatorname{NDCG@10}_u(\text{weighted SVD}).
\]

The point estimate is the mean of \(\Delta_u\) over 177 users. Each of 3,000
replicates samples 177 user IDs with replacement and recomputes that mean; the
reported interval is the 2.5th/97.5th percentile of the bootstrap distribution.
Resampling users preserves the paired comparison and avoids treating method rows
as independent observations.

For reference configuration/data ID `30a640be674a`,

\[
\widehat\Delta=+0.0010,
\qquad
95\%\ \text{percentile interval}=[-0.0006,+0.0030].
\]

This is a development benchmark with descriptive uncertainty, not a
preregistered confirmatory experiment. The interval is conditional on this
dataset, split, fitted model, configuration, and seed. It does not include
training, dataset, or policy uncertainty and has no causal interpretation.
The fraction of bootstrap replicates above zero, available in the machine-readable
result, is a descriptive resample fraction; it is neither a posterior probability
nor a p-value.

## 10. Leakage and tuning contract

| Component | Data allowed | Data excluded |
|---|---|---|
| Weighted SVD retrieval, item statistics, user profiles | earlier history from all 943 users | all later ratings as values/labels |
| Feature scaler and RankNet weights | candidate-gated later rows/pairs from 549 rank-training users | validation and development-test later rows |
| Regularization selection | pair accuracy for 159 validation users | development-test users |
| Blend selection | end-to-end NDCG@10 for 159 validation users | development-test users |
| Development-test reporting | later outcomes for 177 development-test users | use in fitting or tuning |

Important qualifications:

- the per-user split is not a global event-time feature join;
- earlier histories of development-test users are deliberately available in this
  warm-start, transductive design;
- supervised pairs exist only for observed later ratings that passed Stage 1;
- Stage 1 and Stage 2 are compared on the same users, outcomes, and candidate
  boundary;
- fixed seeds and deterministic tie-breaking support repeatability in the pinned
  environment, not universal or bit-for-bit cross-platform reproducibility.

The candidate-width curve, activity slices, score-zeroing sensitivity, candidate-
restricted judged diagnostic, and observed-label ceiling are exploratory
development-test descriptions. They are not used to choose \(K\), features, or model
parameters; they also receive no multiplicity adjustment. Score zeroing does not
refit the model and must not be interpreted as causal feature importance.

## 11. Interpretation of the reference result

The candidate-restricted judged diagnostic reports a post-hoc comparison of
retrieval and the unblended ranker inside the selected, later-rated candidate set.
Separately, the unblended ranker's end-to-end NDCG@10 point estimate was 0.1172
versus 0.1537 for retrieval. The diagnostic can motivate a hypothesis about
ordering selected, observed comparisons, but it cannot prove that a learned signal
transfers to the mixture of judged and unjudged items in the delivered slate. The
validation-selected 10% residual yielded
0.1548 NDCG@10 versus 0.1537 for retrieval, but its paired interval includes zero.

The defensible conclusion is therefore not “Stage 2 improved recommendation.” It
is that the benchmark exposes candidate-gate constraints, auxiliary/target metric
mismatch, and the value of testing a constrained challenger before model
promotion.

Floating-point SVD and linear-algebra kernels can differ across platforms, so
bit-for-bit metric identity across operating systems or BLAS implementations is
not claimed.
