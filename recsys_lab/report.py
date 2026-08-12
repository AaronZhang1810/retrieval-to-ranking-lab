"""Generate GitHub-native and self-contained aggregate benchmark reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "popularity": "Popularity",
    "retrieval": "Stage 1 · weighted SVD",
    "two_stage": "Stage 2 · guarded blend",
    "unblended_ranker": "Unblended RankNet diagnostic",
    "observed_ceiling_in_candidates": "Observed-label ceiling in candidates*",
}


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def _decision(delta: dict[str, float]) -> str:
    if delta["ci_low"] > 0:
        return (
            "The conditional interval is entirely above zero for this fixed run. "
            "That supports a positive offline difference here, not online or causal lift."
        )
    if delta["ci_high"] < 0:
        return (
            "The conditional interval is entirely below zero for this fixed run; "
            "the challenger does not support offline promotion."
        )
    return (
        "The conditional interval crosses zero; this run does not establish the direction "
        "of the Stage-2 difference."
    )


def markdown_report(manifest: dict[str, Any]) -> str:
    data = manifest["data"]
    metrics = manifest["metrics"]
    delta = manifest["paired_user_bootstrap"]["two_stage_minus_retrieval_ndcg_at_10"]
    ranker = manifest["ranker"]
    judged = manifest["judged_candidate_diagnostic"]
    config = manifest["config"]
    reproducibility = manifest.get("reproducibility", {})
    lines = [
        "# Benchmark Report: Two-Stage Recommendation on MovieLens 100K",
        "",
        f"> Configuration/data ID `{manifest['config_hash']}` · real explicit ratings · CPU-only · no raw MovieLens data committed",
        "",
        "## Executive result",
        "",
        f"The benchmark uses **{data['ratings']:,} ratings from {data['users']:,} numbered user records across {data['movies']:,} movies**. A per-user timestamp-ordered split freezes {data['history_ratings']:,} history ratings and holds out {data['heldout_ratings']:,} later ratings. The development-test evaluation contains {data['test_users']:,} users whose later outcomes were never used to fit or tune the ranker.",
        "",
        "| Method | NDCG@10 | Recall@10 | HitRate@10 | Catalog coverage@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("popularity", "retrieval", "two_stage", "unblended_ranker", "observed_ceiling_in_candidates"):
        row = metrics[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['ndcg_at_10']:.4f} | {row['recall_at_10']:.4f} | "
            f"{_pct(row['hit_rate_at_10'])} | {_pct(row['catalog_coverage'])} |"
        )
    lines += [
        "",
        "\\*The ceiling sorts candidates with held-out labels and is not deployable. It measures headroom inside the candidate set.",
        "",
        f"NDCG, recall, and hit rate are macro-averages over {data['test_users']:,} development-test users. Catalog coverage is the fraction of all {data['movies']:,} movies appearing in their top-10 slates.",
        "",
        f"The guarded Stage-2 point estimate changed NDCG@10 by **{delta['estimate']:+.4f}** (paired user percentile-bootstrap 95% interval **[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]**; {int(delta['clusters'])} users, {int(delta['replicates']):,} resamples). {_decision(delta)}",
        "",
        "## What the experiment found",
        "",
        f"1. **The candidate gate is a material hard limit.** At `K={config['candidate_k']}`, mean per-user candidate recall was {_pct(metrics['retrieval']['candidate_recall_at_k'])} for held-out movies rated 4–5 stars. A reranker cannot recover relevant movies outside that gate.",
        f"2. **Pairwise accuracy did not guarantee end-to-end lift.** The unblended RankNet reached {_pct(ranker['test_pair_accuracy'])} accuracy on sampled development-test pairs, while its end-to-end NDCG@10 point estimate was {metrics['unblended_ranker']['ndcg_at_10']:.4f} versus {metrics['retrieval']['ndcg_at_10']:.4f} for weighted SVD.",
        f"3. **The post-hoc judged-only diagnostic answers a narrower question.** Among users with at least two later-rated Stage-1 candidates, candidate-restricted judged NDCG@10 changed from {judged['retrieval']['ndcg_at_10']:.4f} to {judged['unblended_ranker']['ndcg_at_10']:.4f}. That point estimate has no interval and is intentionally secondary: it conditions on both later rating and Stage-1 selection.",
        f"4. **A validation-selected coefficient constrained the challenger.** The final score uses `α={ranker['selected_blend_alpha']:.3f}` from the configuration-declared 0–0.10 grid. This bounds the learned score's coefficient but does not guarantee preservation of any fixed fraction of Stage-1 positions.",
        "",
        "This is a useful inconclusive result: it shows why a plausible reranker must be evaluated through the candidate gate with the primary end-to-end offline endpoint, rather than promoted from pairwise accuracy alone.",
        "",
        "## Candidate-width trade-off",
        "",
        "| K | Candidate recall | SVD NDCG@10 | Guarded NDCG@10 | Feature scores/query |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in manifest["candidate_curve"]:
        lines.append(
            f"| {row['candidate_k']} | {_pct(row['candidate_recall'])} | "
            f"{row['retrieval_ndcg_at_10']:.4f} | {row['two_stage_ndcg_at_10']:.4f} | "
            f"{int(row['stage_2_feature_scores']):,} |"
        )
    lines += [
        "",
        "Increasing `K` improves candidate recall but linearly increases rich feature evaluation. The SVD top-10 does not change with `K`; the curve exposes the coverage/cost boundary the reranker inherits. This is an exploratory development-test curve and was not used to select `K`.",
        "",
        "## Model and validation",
        "",
        "Stage 1 applies a low-rank decomposition to positive preference mass from the frozen history. Stage 2 fits a linear RankNet/Bradley–Terry objective on later, observed higher-vs-lower rating pairs from rank-training users. All features are computed from the frozen history:",
        "",
        "| Feature | Standardized weight |",
        "|---|---:|",
    ]
    for name, weight in zip(ranker["feature_names"], ranker["weights"]):
        lines.append(f"| {name} | {float(weight):+.4f} |")
    lines += [
        "",
        f"Regularization was selected on {data['rank_validation_users']:,} validation users; the selected `λ={ranker['selected_l2']}` produced {_pct(ranker['validation_pair_accuracy'])} validation-pair accuracy. The blend was then selected on end-to-end validation NDCG@10 using the same validation group. Those validation scores are selection criteria, not unbiased performance estimates. Development-test later outcomes were used only for reporting.",
        "",
        "## Leakage and evaluation contract",
        "",
        "```text",
        "each user's timestamp-ordered ratings",
        "        │",
        "        ├── earlier prefix → SVD + item statistics + user profiles (frozen)",
        "        └── later suffix  → observed pair labels / final offline relevance",
        "                              │",
        "eligible users ──→ rank-train / validation / development-test (disjoint)",
        "```",
        "",
        "- A held-out movie is relevant at rating ≥4; graded relevance is `max(rating − 3, 0)`.",
        "- Unrated movies receive zero *evaluation gain* but are described as unjudged, not disliked.",
        "- The split is chronological within user; movie ID deterministically breaks same-second timestamp ties.",
        "- Retrieval is transductive warm-start: every evaluated user contributes only their earlier history.",
        "- NDCG and recall compare methods on the same held-out users; Stage 2 only permutes Stage-1 candidates.",
        "- The paired bootstrap resamples users and is conditional on this dataset, split, fitted model, and seed.",
        "- The bootstrap fraction above zero in RESULTS.json is neither a posterior probability nor a p-value.",
        "",
        "## Reproducibility record",
        "",
        f"- Dataset: MovieLens 100K, {data['date_start']} through {data['date_end']}",
        f"- Parsed `u.data` + `u.item` fingerprint: `{data['source_fingerprint']}`",
        f"- Configuration/data SHA-256: `{manifest.get('config_sha256', manifest['config_hash'])}`",
        f"- Result schema: `{manifest.get('result_schema_version', 'legacy')}`",
        f"- Python `{manifest['environment']['python']}`; NumPy `{manifest['environment']['numpy']}`",
        f"- Local descriptive runtime: `{manifest['runtime_seconds']:.2f}` seconds",
    ]
    if reproducibility.get("source_tree_sha256"):
        lines.append(f"- Executable source-tree SHA-256: `{reproducibility['source_tree_sha256']}`")
    if reproducibility.get("runtime_lock_sha256"):
        lines.append(f"- Runtime-lock SHA-256: `{reproducibility['runtime_lock_sha256']}`")
    lines += [
        "- RESULTS.json is the canonical aggregate result; row-level outputs and derived model state stay local",
        "",
        "Run `python download_data.py --accept-license && python run_demo.py --publish-reports` to regenerate the checked reports. Dataset terms and citation are in [DATA.md](DATA.md); model derivations are in [MATH.md](MATH.md).",
        "",
        "## Limitations and next experiment",
        "",
    ]
    lines.extend(f"- {item}" for item in manifest["limitations"])
    lines += [
        "- The candidate-restricted judged diagnostic is post hoc, selection-biased, and not the primary endpoint.",
        "- This is a development benchmark, not a preregistered confirmatory study.",
        "",
        "The next useful experiment is impression-aware training with randomized exposure or propensity information. That could help distinguish unexposed movies from exposed non-engagement and align reranker training more closely with the evaluated policy distribution.",
        "",
        "## Data acknowledgment",
        "",
        "This analysis uses the MovieLens 100K dataset from GroupLens Research. The University of Minnesota and GroupLens do not endorse this project. Code is MIT-licensed; MovieLens data is governed separately and is not redistributed here.",
    ]
    return "\n".join(lines) + "\n"


def _metric_cards(manifest: dict[str, Any]) -> str:
    metrics = manifest["metrics"]
    data = manifest["data"]
    cards = [
        ("real ratings", f"{data['ratings']:,}"),
        ("mean candidate recall @ 200", _pct(metrics["retrieval"]["candidate_recall_at_k"])),
        ("SVD NDCG@10", f"{metrics['retrieval']['ndcg_at_10']:.4f}"),
        ("guarded NDCG@10", f"{metrics['two_stage']['ndcg_at_10']:.4f}"),
    ]
    return "".join(
        f'<article class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></article>'
        for label, value in cards
    )


def html_report(manifest: dict[str, Any]) -> str:
    metrics = manifest["metrics"]
    delta = manifest["paired_user_bootstrap"]["two_stage_minus_retrieval_ndcg_at_10"]
    ranker = manifest["ranker"]
    reproducibility = manifest.get("reproducibility", {})
    source_prefix = str(reproducibility.get("source_tree_sha256", "unrecorded"))[:12]
    lock_prefix = str(reproducibility.get("runtime_lock_sha256", "unrecorded"))[:12]
    visible_provenance = (
        f"schema {html.escape(str(manifest.get('result_schema_version', 'legacy')))} · "
        f"source {html.escape(source_prefix)} · lock {html.escape(lock_prefix)} · "
        f"Python {html.escape(str(manifest['environment']['python']))} · "
        f"NumPy {html.escape(str(manifest['environment']['numpy']))}"
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(METHOD_LABELS[method])}</td>"
        f"<td>{metrics[method]['ndcg_at_10']:.4f}</td>"
        f"<td>{metrics[method]['recall_at_10']:.4f}</td>"
        f"<td>{_pct(metrics[method]['hit_rate_at_10'])}</td>"
        "</tr>"
        for method in ("popularity", "retrieval", "two_stage", "unblended_ranker", "observed_ceiling_in_candidates")
    )
    curve = "".join(
        f"<tr><td>{row['candidate_k']}</td><td>{_pct(row['candidate_recall'])}</td>"
        f"<td>{row['retrieval_ndcg_at_10']:.4f}</td><td>{row['two_stage_ndcg_at_10']:.4f}</td>"
        f"<td>{int(row['stage_2_feature_scores']):,}</td></tr>"
        for row in manifest["candidate_curve"]
    )
    weights = "".join(
        f"<div class='weight'><span>{html.escape(name)}</span><i style='--w:{min(abs(float(weight))/0.7,1):.3f}' class='{'pos' if weight >= 0 else 'neg'}'></i><code>{float(weight):+.3f}</code></div>"
        for name, weight in zip(ranker["feature_names"], ranker["weights"])
    )
    embedded = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ).replace("</", "<\\/")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MovieLens two-stage benchmark</title><style>
:root{{--ink:#15221f;--muted:#60736e;--paper:#f4f3ec;--card:#fff;--green:#176b57;--mint:#75e3c2;--orange:#e86f4a;--line:#d6ddd9}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:42px 26px 70px}}h1,h2{{font-family:Georgia,serif;letter-spacing:-.025em}}h1{{font-size:clamp(40px,7vw,74px);line-height:.95;margin:12px 0 22px;max-width:920px}}h2{{font-size:31px;margin:0 0 12px}}.eyebrow{{font:700 11px ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--green)}}.lead{{font-size:19px;color:var(--muted);max-width:820px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:30px 0}}.metric,.panel{{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:19px}}.metric span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase}}.metric strong{{font:700 31px Georgia,serif}}section{{margin-top:34px}}table{{width:100%;border-collapse:collapse;background:white;border:1px solid var(--line)}}th,td{{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}.decision{{border-left:5px solid var(--green);background:#e5f3ee;padding:18px 20px;font-size:17px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.weight{{display:grid;grid-template-columns:155px 1fr 60px;gap:9px;align-items:center;margin:10px 0}}.weight i{{height:9px;width:calc(var(--w)*100%);background:var(--mint);border-radius:2px}}.weight i.neg{{background:var(--orange)}}code{{font-family:ui-monospace,monospace}}.fine{{color:var(--muted);font-size:12px}}footer{{margin-top:40px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted)}}@media(max-width:760px){{.metrics,.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:520px){{.metrics,.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="eyebrow">Real-data recommender systems benchmark · configuration/data ID {manifest['config_hash']}</div><h1>From 1,682 movies to 10 recommendations.</h1><p class="lead">Weighted-SVD retrieval, observed pairwise learning-to-rank, and an explicit audit of the candidate bottleneck on 100,000 MovieLens ratings.</p><div class="metrics">{_metric_cards(manifest)}</div>
<section><div class="eyebrow">01 · Held-out result</div><h2>Did Stage 2 earn promotion?</h2><div class="decision">NDCG@10 delta <strong>{delta['estimate']:+.4f}</strong> · paired user percentile-bootstrap 95% interval [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]. {_decision(delta)}</div><table><thead><tr><th>Method</th><th>NDCG@10</th><th>Recall@10</th><th>HitRate@10</th></tr></thead><tbody>{rows}</tbody></table><p class="fine">NDCG, recall, and hit rate are macro-averages. The observed-label ceiling uses held-out ratings and is not deployable.</p></section>
<section><div class="eyebrow">02 · Candidate gate</div><h2>Coverage rises; scoring cost rises with it.</h2><table><thead><tr><th>K</th><th>Candidate recall</th><th>SVD NDCG</th><th>Guarded NDCG</th><th>Feature scores</th></tr></thead><tbody>{curve}</tbody></table></section>
<section class="grid"><article class="panel"><div class="eyebrow">03 · Learned ranker</div><h2>Inspectable residual features</h2>{weights}<p class="fine">Linear standardized weights are associative model parameters, not causal explanations.</p></article><article class="panel"><div class="eyebrow">04 · Decision quality</div><h2>Why the constraint matters</h2><p>The unblended ranker achieved <strong>{_pct(ranker['test_pair_accuracy'])}</strong> accuracy on sampled development-test pairs, while its end-to-end NDCG@10 point estimate was <strong>{metrics['unblended_ranker']['ndcg_at_10']:.4f}</strong>. Validation selected α={ranker['selected_blend_alpha']:.3f}; this bounds the learned score's coefficient but does not guarantee preservation of a fixed fraction of Stage-1 positions.</p><p>An encouraging auxiliary diagnostic can coexist with a worse primary offline endpoint.</p></article></section>
<section><div class="eyebrow">05 · Evaluation contract</div><h2>What is—and is not—known</h2><p>Earlier per-user ratings fit model-derived features; static metadata supplies genre and release year. Rank-training, validation, and development-test later outcomes have disjoint roles. A 4–5 star held-out movie is relevant. Unrated movies receive zero evaluation gain but remain <em>unjudged</em>, not disliked. Bootstrap uncertainty is conditional on this fitted model and split; it is neither causal uncertainty nor a posterior probability.</p></section>
<footer>MovieLens data are governed separately by GroupLens terms and are not redistributed. Code is MIT-licensed. Aggregate evidence is in RESULTS.json; see README.md, REPORT.md, DATA.md, MATH.md, and LIMITATIONS.md.<p class="fine">Visible provenance: {visible_provenance}</p><script type="application/json" id="experiment-manifest">{embedded}</script></footer></main></body></html>"""


def write_reports(
    markdown_path: str | Path,
    html_path: str | Path,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    markdown_path, html_path = Path(markdown_path), Path(html_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report(manifest), encoding="utf-8")
    html_path.write_text(html_report(manifest), encoding="utf-8")
    return markdown_path, html_path


def write_report(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Backward-compatible helper for a single HTML report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_report(manifest), encoding="utf-8")
    return path
