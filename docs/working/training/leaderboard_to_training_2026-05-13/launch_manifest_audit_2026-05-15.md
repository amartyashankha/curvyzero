# Launch Manifest Audit: 2026-05-15

Purpose: make the next real CurvyTron training launch hard to run against stale
storage, stale tournaments, or a static no-refresh opponent setup.

## Current Code Changes

- `src/curvyzero/contracts/curvytron.py`
  - current storage/app getters reject non-`-v2` overrides;
  - current dashboard ids now point at the all-v2 canary proof until fresh real
    tournament ids are chosen;
  - shared launch refresh interval constant is `2000` learner train iterations.
- `scripts/build_curvytron_tonight18_manifest.py`
  - requires `--ratings-snapshot`;
  - defaults to `--opponent-source assignment`;
  - defaults assignment artifacts and refresh pointers to `control:`;
  - defaults row names to `curvy-r18v2-*` / `try-r18v2-*`;
  - defaults `assignment_refresh_interval_train_iter=2000`.
- `scripts/submit_curvytron_survivaldiag_manifest.py`
  - rejects app-name mismatches between the selected app, each row's
    `deployed_app_submission.app_name`, and the current shared-contract trainer
    app.
- `src/curvyzero/infra/modal/curvytron_gif_browser.py`
  - "Current batch" prefix is now `curvy-r18v2-*`, not invalidated
    `curvy-v2real18-*`.

## What This Fixes

- No silent `/private/tmp/...20260514...` snapshot default.
- No silent inline-mixture launch when the goal is live assignment refresh.
- No silent no-refresh launch.
- No default assignment write to the runs volume when the active control path is
  `control:`.
- No accidental v2-shaped manifest submission into an old trainer app.
- No dashboard default hard-coded to old loop18/v2real18 lanes in current code.

## Still Needed Before Launch

- Choose or materialize the exact source leaderboard snapshot in all-v2 storage.
  Current v2 storage only has the all-v2 canary leaderboard, which is a wiring
  proof and not a production-quality source.
- Read-only v2 inventory found no other production-quality source currently in
  recreated v2 storage. Historical `v2real18`, `v2refresh18p`, and looplive
  snapshots are absent from the current v2 tournament volume and remain
  diagnostic unless explicitly rematerialized/rerated.
- Add and run a manifest checkpoint-ref existence audit before launch. It must
  check that the initial checkpoint ref and all frozen assignment/mixture refs
  exist in the selected all-v2 runs volume. This is separate from syntactic
  `iteration_N.pth.tar` checks.
- Choose fresh real tournament/rating ids and update the shared contract in the
  same patch as the dashboard redeploy.
- Generate the candidate restart18 manifest and save the resolved JSON.
- Audit the manifest for:
  - all checkpoint refs are exact `iteration_N.pth.tar`;
  - no `curvyzero-runs` / non-v2 assignment or control refs;
  - all rows have `random_per_episode`;
  - all rows have `initial_policy_checkpoint_ref` from the rank-1 source row;
  - all rows have `opponent_assignment_refresh_ref` and refresh interval `2000`;
  - assignment bank contains exactly three immutable recipe assignments and
    three refresh pointers;
  - policy surface is `browser_lines + simple_symbols`;
  - checkpoint cadence is `10000` and `commit_on_checkpoint=true`.
- Run the manifest submitter in dry-run mode, then launch only after the JSON
  and submission summary match this audit.

## Diagnostic Dry Run

Built a local diagnostic restart18 manifest from the all-v2 canary leaderboard
only to verify builder/submitter wiring. Do not launch this as the real batch:
the source leaderboard has only `25` rows, `4` active rows, relaxed maturity
gates, and rank 1 is `iteration_0.pth.tar`.

- Manifest:
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18v2-dryrun-canarysource-20260515a/curvy-r18v2-dryrun-canarysource-20260515a.json`
- Submission dry-run:
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18v2-dryrun-canarysource-20260515a/submission-dryrun.json`
- Source snapshot:
  `artifacts/local/curvytron_e2e_canary/promotions/e2e-allv2-canary-live-r3-20260515a-e2e-allv2-canary-live-r3-20260515a-e2e-allv2-canary-live-r3-a/fetched/leaderboard_snapshot.json`
- Source snapshot canonical sha:
  `c7ac3d894b3780b29d1e99572c5e5c91b5cd3008ff0f68308101510f15319ed1`

Dry-run audit results:

- `18` rows, `3 x 3 x 2` axes.
- `opponent_source=assignment`.
- assignment target volume: `control`.
- refresh pointer volume: `control`.
- refresh interval: `2000`.
- app: `curvyzero-lightzero-curvytron-visual-survival-train-v2`.
- learner seat: `random_per_episode`.
- policy surface: `browser_lines + simple_symbols`.
- checkpoint cadence: `10000`, `commit_on_checkpoint=true`.
- assignment refs: `3` immutable recipe assignments.
- refresh refs: `3` control-volume refresh pointers.
- stale non-v2 volume names in row JSON: none found.
- initial checkpoint source: canary rank 1, `iteration_0.pth.tar`; this is the
  reason this manifest is diagnostic only.

## Validation So Far

- `uv run pytest tests/test_curvytron_tonight18_manifest.py tests/test_curvytron_shared_contracts.py -q`
  passed with `16 passed`.
- Focused launch bundle
  `tests/test_curvytron_tonight18_manifest.py`,
  `tests/test_curvytron_survivaldiag_submitter.py`,
  `tests/test_curvytron_shared_contracts.py`, and
  `tests/test_promote_curvytron_rating_round.py` passed with `26 passed`.
- Broader E2E-adjacent slice covering tournament, GIF browser, trainer refresh
  plumbing, env, and opponent modules passed with `343 passed, 24 skipped`.
- Ruff passed for touched launch/contract/GIF-browser/test files.
