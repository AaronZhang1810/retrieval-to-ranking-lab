# Suggested repository metadata

- **Repository name:** `two-stage-recommender-lab`
- **Description:** Leakage-aware weighted-SVD retrieval and guarded pairwise ranking on MovieLens 100K, with chronological evaluation and an auditable report.
- **Topics:** `recommender-system`, `learning-to-rank`, `movielens`, `ranking`, `information-retrieval`, `machine-learning`, `statistics`, `offline-evaluation`, `numpy`, `python`

When the repository is ready for a public report, enable GitHub Pages with **GitHub Actions** as the source and create the repository variable `ENABLE_GITHUB_PAGES=true`. Until then, the Pages workflow is intentionally skipped so a private-first push remains green. The workflow publishes only the reviewed `docs/` directory; raw data and local model artifacts are neither uploaded nor included in the deployment artifact.
