# Security policy

## Supported version

The current `main` branch is the only supported version of this portfolio project. Historical snapshots and generated artifacts do not receive security updates.

## Reporting a vulnerability

Please use GitHub's private vulnerability-reporting feature when it is enabled for this repository. If it is unavailable, contact the repository owner through the private contact method shown on their GitHub profile. Do not disclose an unpatched vulnerability, credential, or personal data in a public issue.

Include the affected revision, reproduction steps, impact, and any suggested mitigation. Please avoid including real user data in a report.

## Deployment boundary

The browser application is a local demonstration and accepts loopback binds only. It also rejects non-loopback `Host` headers. It has no authentication, authorization, TLS termination, rate limits, or multi-tenant isolation; do not modify that boundary to expose it to an untrusted network or use it to process confidential interaction data.

Downloaded datasets and generated model artifacts are local inputs, not trusted application code. Keep them outside version control. Public report generation requires the exact downloader provenance marker and pinned MovieLens file hashes. The artifact loader checks JSON/NPZ size limits, exact archive members, bounded uncompressed sizes, array schemas, finite values, cross-field consistency, and recorded SHA-256 values; regenerate artifacts from verified data instead of accepting files from an untrusted party.

GitHub Pages publishes only the static aggregate report under `docs/`; it does not deploy the local browser application, raw data, row-level outputs, or model state. Run `python public_release.py --check` against the working tree and `python public_release.py --check --staged --history` against the staged tree and reachable history before publishing a revision.

Runtime, development, and audit-tool lock files pin exact package versions, and CI installs wheels only. They do not currently pin every platform-specific wheel hash; the version locks prevent resolver drift but are not byte-for-byte package-artifact locks. GitHub Actions are pinned to reviewed full commit SHAs, and CI audits all three dependency locks against OSV.
