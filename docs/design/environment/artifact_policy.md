# Artifact Policy

Keep git for durable source and small human-readable records. Keep raw run data
out of git.

## Commit

- Source, tests, scenario JSON, schemas, runbooks, and design docs.
- Dated experiment logs under `docs/experiments/`.
- Small golden fixtures only after they are curated into `scenarios/`,
  `tests/fixtures/`, or a clearly named docs example.

Do not commit raw files from `artifacts/`, `runs/`, `tmp/`, `logs/`,
`checkpoints/`, `replay/`, or `videos/`.

## Local Artifacts

Use local ignored paths for scratch output:

- `artifacts/local/` for local trace and diff probes.
- `tmp/` for fetched Modal files and one-off command output.
- `logs/` for local stdout/stderr captures.
- Tool caches such as `.cache/`, `.pytest_cache/`, `.ruff_cache/`, and
  `.mypy_cache/`.

These files should be reproducible or disposable. If a local output becomes a
real fixture, copy only the minimal curated version into a committed fixture
path.

## Modal Volume

Put complete remote run payloads in the Modal Volume, not git. The first shared
Volume is `curvyzero-runs` mounted at `/runs`.

- Fidelity runs: `/runs/fidelity/<run_id>/`
- One-off experiment attempts: `/runs/experiments/<run_id>/attempts/<attempt_id>/`
- Training payloads: checkpoints, replay chunks, sampled videos, manifests, and
  large metrics files under the run directory.

Write manifests after payload files exist. Treat the manifest path as the stable
reference for later fetches.

## Experiment Logs

For each useful run, add a short dated log in `docs/experiments/` with:

- question and setup
- exact command
- code ref or package version, if known
- run id and Modal artifact path
- key metrics, fingerprints, failures, and interpretation
- follow-ups

Link to raw artifacts by path. Do not paste full traces, long logs, replay data,
or checkpoint metadata into the markdown log.
