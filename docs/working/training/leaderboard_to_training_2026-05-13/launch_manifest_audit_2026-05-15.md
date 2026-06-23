# Launch Manifest Audit: 2026-05-15

Purpose: make the next real CurvyTron training launch hard to run against stale
storage, stale tournaments, or a static no-refresh opponent setup.

Plain correction from the latest reorientation:

- A perfect starting ranking is not required for bootstrap training.
- Modal Dict leaderboard pointers are only live pointers/cache. Durable truth is
  exact Volume JSON and exact checkpoint refs.
- The stricter rerate/stability work is only for the optional
  leaderboard-derived top-slot source. It is not a reason to block a bootstrap
  launch that uses curated exact checkpoint refs plus immortal hard-coded
  pressure.
- Blank and hard-coded sentinel opponents should be immortal. Frozen checkpoint
  slots should mostly be ordinary/mortal, with only small explicit immortal
  slices. Keep total immortal opponent exposure around `20-30%`, and generally
  not above about `30%`.

## Current Code Changes

- `src/curvyzero/contracts/curvytron.py`
  - current storage/app getters reject non-`-v2` overrides;
  - current dashboard ids now point at the all-v2 canary proof until fresh real
    tournament ids are chosen;
  - shared launch refresh interval constant is `2000` learner train iterations.
- `scripts/build_curvytron_tonight18_manifest.py`
  - requires exactly one of `--ratings-snapshot` or `--checkpoint-refs-file`;
  - `--checkpoint-refs-file` is the bootstrap path when we want curated exact
    refs without pretending there is a trusted ranking;
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

- Choose the exact bootstrap checkpoint refs or, if using a ranked source, the
  exact leaderboard-derived candidate snapshot in all-v2 storage. Current v2
  storage only has the all-v2 canary leaderboard, which is a wiring proof and
  not a production-quality ranked source. That does not block bootstrap launch
  if curated exact refs are used.
- Read-only v2 inventory found no other production-quality source currently in
  recreated v2 storage. Historical `v2real18`, `v2refresh18p`, and looplive
  snapshots are absent from the current v2 tournament volume and remain
  diagnostic unless explicitly rematerialized/rerated.
- Add and run checkpoint-ref existence audits before launch. The launch manifest
  audit must check that the initial checkpoint ref and all frozen
  assignment/mixture refs exist in the selected all-v2 runs volume. The
  source-rematerialization audit can run earlier on `refs.txt` before a launch
  manifest exists.
- Implemented guardrail:
  `scripts/audit_curvytron_launch_manifest_refs.py`. Local tests cover
  collection, missing refs, good refs, and bad mutable/control-prefixed refs.
  A real Modal audit of the canary dry-run manifest against
  `curvyzero-runs-v2` passed with `4/4` unique checkpoint refs present.
  The same tool now accepts `--refs-file`.
- Ranked-source recommendation, if we want stronger starting frozen refs:
  select top active candidate refs from historical
  `loop18-main-adaptive417`, copy/rematerialize those checkpoint files into
  `curvyzero-runs-v2`, and rerate under fresh all-v2 ids. The old leaderboard
  is candidate selection only, not launch truth.
- Local plan artifact:
  `artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top100-20260515a/`.
  It includes `refs.txt`, `selection.json`, and `commands.txt` for copying the
  old top-active checkpoint files into `curvyzero-runs-v2` and then launching
  `curvy-restart18-source-rerate-20260515a` /
  `elo-restart18-source-rerate-20260515a`. Selection summary: `100` active
  candidates, `4` `iteration_0` candidates, max iteration `306755`.
- Real audits on that source-ref plan:
  - syntax audit: `100/100` exact checkpoint refs, no mutable/control-prefixed
    refs;
  - old source volume audit: `100/100` refs present in `curvyzero-runs`;
  - v2 target before-copy audit: `0/100` refs present in `curvyzero-runs-v2`,
    which is expected before the copy and confirms the target lane is clean.
- Copy/audit result: `scripts/rematerialize_curvytron_checkpoint_refs.py`
  copied `100/100` refs from old `curvyzero-runs` into `curvyzero-runs-v2`;
  `source-refs-v2-target-after-copy-audit.json` passed with `100/100` present.
- `commands.txt` embeds the source audit first, the copy block second, the v2
  target audit third, and the fresh rerate command last. The generated command
  was corrected after the current CLI rejected stale `--policy-batch-size`.
- Fresh source rerate launched with `modal run --detach`:
  `curvy-restart18-source-rerate-20260515a` /
  `elo-restart18-source-rerate-20260515a`, call
  `fc-01KRPJE1C28EJZQK6VRYQ75JT7`; first progress query showed
  `games_running`, `0/6300` complete.
- Fresh ranked-source rerate status update: the 100-ref rerate completed through
  round 6 with `stable=false` and is diagnostic only because `iteration_0`
  rows rose to top ranks. The current ranked-source candidate is the 96-ref
  nonzero rerate `curvy-restart18-source-rerate-nonzero-20260515a` /
  `elo-restart18-source-rerate-nonzero-20260515a`; round 6 completed with all
  `96` rows active, `0` failures, and `stable=false`
  (`max_abs_delta=25.199213332028748`). This is worse than round 5, so diagnose
  the max mover and scheduler exposure before another blind continuation.
- Launch audit gate for leaderboard-derived top slots only: use no ranked
  source snapshot from this rerate until latest is `stable=true`,
  coverage-mature, and published with expected round/context/roster/snapshot
  hashes. Bootstrap/static launch can still use curated exact refs.
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
the canary ranked snapshot has only `25` rows, `4` active rows, relaxed
maturity gates, and rank 1 is `iteration_0.pth.tar`.

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
- Added ref-audit validation:
  `tests/test_curvytron_launch_manifest_ref_audit.py` -> `4 passed`;
  focused launch/promotion/ref-audit/source-plan bundle -> `34 passed`;
  broad E2E-adjacent slice with the new tests included -> `352 passed,
  24 skipped`;
  ruff passed after exporting
  `DEFAULT_TOURNAMENT_POLICY_OBSERVATION_BACKEND` from tournament contracts.
