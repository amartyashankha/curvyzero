# Granular Action Cadence Launch / Manifest Side-Effect Audit

Date: 2026-05-13

## Plain Verdict

The active trusted stock LightZero launch lane is mostly protected now.

Trusted `source_state_fixed_opponent` `--mode train` and `--mode dry` reject
stale bundled `decision_ms` values, and the active survivaldiag and
opponent-mixture manifest builders no longer put `200.0` into grouped submit
kwargs.

The remaining risk is around edges:

- generated review command text does not include `--decision-ms`, so it relies
  on the current launcher default;
- grouped manifest submission only checks that `decision_ms` is present, not
  that it is one source frame;
- `--mode profile` can carry stale `decision_ms` in command metadata even though
  the env config also forces `decision_source_frames=1`;
- checkpoint tournaments still default to the old vector-env 12-frame cadence;
- historical two-seat scripts are guarded, but if rerun they now inherit the new
  one-frame top-level default unless `--decision-ms` is passed explicitly;
- direct old smoke/eval helpers still default to `300.0` ms and should be
  treated as legacy or test-only.

## What Is Protected

### Trusted train/dry launcher path

The trainer default now comes from the source-state survival wrapper:

- `DEFAULT_DECISION_SOURCE_FRAMES = SOURCE_STATE_DEFAULT_DECISION_SOURCE_FRAMES`
- `DEFAULT_DECISION_MS = SOURCE_STATE_DEFAULT_DECISION_MS`
- `DEFAULT_SOURCE_PHYSICS_STEP_MS = DEFAULT_DECISION_MS / DEFAULT_DECISION_SOURCE_FRAMES`

See `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:356`.

The trusted-lane guard rejects any `decision_ms` that is not exactly the one
source physics step default for `source_state_fixed_opponent`
(`lightzero_curvyzero_stacked_debug_visual_survival_train.py:682`). It is called
for `mode in {"train", "dry"}` before config build
(`lightzero_curvyzero_stacked_debug_visual_survival_train.py:3198`).

`_build_visual_survival_configs` also calls the guard when profile timing is not
enabled (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:4526`) and
writes explicit cadence fields into `main_config.env`:

- `decision_ms`
- `decision_source_frames`
- `source_physics_step_ms`
- `source_max_steps_semantics = "source_physics_steps"`

See `lightzero_curvyzero_stacked_debug_visual_survival_train.py:4600`.

Recommended test to add:

- Add a local-entrypoint spawn test with fake Modal functions:
  `main(mode="train", env_variant="source_state_fixed_opponent",
  decision_ms=300.0, wait_for_train=False)` should raise before any train or
  poller spawn.

### Active manifest builders

`scripts/build_curvytron_survivaldiag_manifest.py` now imports
`SOURCE_PHYSICS_STEP_MS` and sets `DECISION_MS = SOURCE_PHYSICS_STEP_MS`
(`scripts/build_curvytron_survivaldiag_manifest.py:22`,
`scripts/build_curvytron_survivaldiag_manifest.py:46`).

`scripts/build_curvytron_opponent_mixture_manifest.py` does the same
(`scripts/build_curvytron_opponent_mixture_manifest.py:21`,
`scripts/build_curvytron_opponent_mixture_manifest.py:40`).

Both builders put that value in `train_kwargs` for grouped submission:

- survivaldiag: `scripts/build_curvytron_survivaldiag_manifest.py:1072`
- opponent mixture: `scripts/build_curvytron_opponent_mixture_manifest.py:557`

The tests now assert `train_kwargs["decision_ms"] < 20.0` for both active
builders.

One caveat: the human review command text from both builders does not pass
`--decision-ms`; it relies on the trainer default. That is fine today, but it is
a weaker artifact than grouped submit kwargs.

Recommended fixes/tests:

- In both command builders, add `--decision-ms str(DECISION_MS)` so copied
  review commands are self-contained.
- Add manifest validation that every row has
  `train_kwargs["decision_ms"] == DECISION_MS` and `< 20.0`.
- Add a text-level test that `row["command_text"]` either contains
  `--decision-ms <one-frame-value>` or the manifest has a clear
  `decision_ms_source = "launcher_default"` guard.

## Grouped Submit Risk

`scripts/submit_curvytron_survivaldiag_manifest.py` requires a `decision_ms`
field in `train_kwargs` (`scripts/submit_curvytron_survivaldiag_manifest.py:13`),
but it does not validate the value before spawning the deployed Modal functions.

This means an old manifest file with `train_kwargs["decision_ms"] = 200.0` or
`300.0` would still be submitted. The train function should reject it, but the
submitter would spawn the poller first, then spawn a train call that fails early.
That can leave confusing poller/status noise.

Recommended fix:

- Add submitter-side validation before spawning either function:
  `env_variant == "source_state_fixed_opponent"` and `mode == "train"` require
  `0 < decision_ms < 20.0` or exactly `SOURCE_PHYSICS_STEP_MS`.
- Also validate `poller_kwargs` cadence if cadence fields are added there.

Recommended tests:

- `test_submitter_rejects_stale_decision_ms_before_spawn`: feed a manifest row
  with `decision_ms=200.0`, monkeypatch fake Modal functions, and assert no
  spawn occurs.
- `test_submitter_accepts_one_frame_decision_ms`: same shape with the active
  one-frame value.

## Profile Mode Behavior

Profile mode is intentionally not guarded the same way as train/dry:

- `_run_visual_survival_train` only calls the trusted cadence guard for
  `mode in {"train", "dry"}` (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:3198`).
- `_build_visual_survival_configs` skips the guard when
  `profile_env_timing_enabled=True`
  (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:4526`).

The current runtime is less scary than the metadata: `_build_visual_survival_configs`
still writes `decision_source_frames=1`, and the env wrapper prefers
`decision_source_frames` when present. But the command and env config can still
say `decision_ms=300.0` while the effective env cadence is one frame. That is a
metadata foot-gun for profile readouts.

The optimizer profile manifest currently omits `--decision-ms` and therefore
uses the new default (`scripts/build_curvytron_optimizer_profile_manifest.py:474`).
That is okay.

Recommended fix:

- For `mode="profile"` with `source_state_fixed_opponent`, either reject stale
  `decision_ms` too, or normalize the recorded `command["decision_ms"]` to the
  derived one-frame value whenever `decision_source_frames=1` is forced.
- Add a clear `profile_cadence_guard = "one_frame_normalized"` or
  `profile_cadence_guard = "legacy_decision_ms_allowed"` field.

Recommended tests:

- `test_profile_with_stale_decision_ms_records_effective_one_frame_cadence` if
  profile keeps allowing stale input.
- Or `test_profile_rejects_stale_decision_ms_for_source_state_fixed_opponent` if
  profile should share the train/dry guard.

## Background Eval / GIF

Background eval and GIF configs now carry cadence fields from the train command.
The direct eval harness builds envs with `decision_ms=DEFAULT_DECISION_MS`
(`src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py:720`),
and eval rows now report `decision_ms`, `decision_source_frames`, and
`source_physics_step_ms`
(`lightzero_curvytron_visual_survival_eval.py:952`,
`lightzero_curvytron_visual_survival_eval.py:1065`).

This is good. The remaining issue is that the poller spawn path in the local
entrypoint passes `source_max_steps` and other eval settings, but not explicit
cadence fields (`lightzero_curvyzero_stacked_debug_visual_survival_train.py:10067`).
That is okay only because eval imports the active default.

Recommended fix/test:

- Add cadence args to the checkpoint eval poller function and pass them through
  local-entrypoint spawn, so eval does not depend on imported defaults.
- Add a fake-poller spawn test that asserts poller kwargs include the one-frame
  cadence fields.

## Historical Two-Seat Paths

The historical zsh launchers are guarded by
`ALLOW_HISTORICAL_CUSTOM_TWO_SEAT_RERUN=1`:

- `scripts/launch_curvytron_mixpast_20260512.zsh:5`
- `scripts/launch_curvytron_overnight40_20260512.zsh:5`

They still launch `--mode two-seat-selfplay` if the guard is bypassed:

- `scripts/launch_curvytron_mixpast_20260512.zsh:48`
- `scripts/launch_curvytron_overnight40_20260512.zsh:66`

They do not pass `--decision-ms`. Because the top-level `main` default changed
to the one-frame source-state default, rerunning these historical scripts now
changes their cadence unless the user passes `--decision-ms` explicitly. That is
not a trusted train problem, but it is a reproducibility trap.

The direct two-seat trainer still has `decision_ms: float = 300.0`, and its CLI
still defaults `--decision-ms 300.0`
(`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:166`,
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:4476`).

Recommended fix:

- If these zsh scripts are only for postmortem reproduction, add explicit
  `--decision-ms 300` and label that as legacy.
- If they should use the new cadence, add `--decision-ms SOURCE_PHYSICS_STEP_MS`
  is not shell-friendly, so write the literal current one-frame value or call a
  tiny helper to print it. Also update the warning text to say the rerun is not
  May 12 cadence-equivalent.

Recommended test:

- Add a unit test for the two-seat branch of `main` with a fake two-seat Modal
  function. Assert the payload includes the intended legacy or one-frame
  `decision_ms`.

## Checkpoint Tournament Defaults

Checkpoint tournaments still use the vector-env default cadence:

- `curvyzero.tournament.curvytron.contracts` imports
  `DEFAULT_DECISION_SOURCE_FRAMES` from `vector_multiplayer_env`
  (`src/curvyzero/tournament/curvytron/contracts.py:10`).
- `vector_multiplayer_env.DEFAULT_DECISION_SOURCE_FRAMES` is still `12`
  (`src/curvyzero/env/vector_multiplayer_env.py:52`).
- Tournament `DEFAULT_DECISION_MS` is therefore about `200.0` ms
  (`src/curvyzero/tournament/curvytron/contracts.py:53`).

`_source_frame_runtime_settings` will use checkpoint runtime metadata when it is
available, but if no metadata overrides are present it falls back to the
tournament default and multiplies `source_max_ticks` by
`decision_source_frames`
(`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:2735`,
`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:2771`).

This is probably intentionally legacy for old checkpoint comparisons, but it
must not be mistaken for the new trusted train cadence.

Recommended fix:

- Add an explicit tournament cadence mode:
  `legacy_vector_default_12_frame` versus `checkpoint_metadata_or_one_frame`.
- For new stock source-state checkpoints, default to checkpoint metadata and
  warn or fail if metadata is missing.
- Add the cadence mode to tournament summaries and website tables.

Recommended tests:

- Keep existing legacy tests if old tournaments need stability.
- Add a new test with checkpoint metadata containing
  `decision_source_frames=1`; assert tournament games use one-frame cadence.
- Add a missing-metadata test for new checkpoint refs that fails or warns
  instead of silently using 12 frames.

## Lower-Level Legacy Defaults

Several lower-level helpers still default to old long decisions:

- `VectorMultiplayerEnv` has `DEFAULT_DECISION_SOURCE_FRAMES = 12` and falls
  back to `decision_ms=300.0` if neither `decision_source_frames` nor
  `decision_ms` is passed (`src/curvyzero/env/vector_multiplayer_env.py:52`,
  `src/curvyzero/env/vector_multiplayer_env.py:285`).
- `curvytron_current_policy_selfplay_smoke` defaults `decision_ms=300.0`
  (`src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py:782`).
- `curvytron_baseline_eval` defaults `decision_ms=300.0`
  (`src/curvyzero/training/curvytron_baseline_eval.py:69`).
- Some source-state survival env tests explicitly use `decision_ms=300.0` for
  profile/no-death natural-bonus stress coverage.

These should not block the trusted train change, but they need labels. The
important rule is: old values are fine only when the path says it is legacy,
profile-only, direct env testing, or tournament-legacy.

Recommended fix:

- Add comments or metadata labels to old helpers:
  `cadence_contract = "legacy_multi_frame_not_trusted_stock_train"`.
- Do not change these defaults in the same cut unless the owning tests are
  updated deliberately.

## Priority Fix List

1. Add submitter-side stale `decision_ms` validation before any poller/train
   spawn.
2. Make active manifest command text explicit about cadence, or explicitly mark
   it as using launcher default.
3. Add a fake local-entrypoint test proving stale `decision_ms=300.0` in
   `--mode train` fails before spawn.
4. Decide profile policy: reject stale `decision_ms`, or normalize and report
   effective cadence.
5. Label tournament and two-seat cadence as legacy, or add explicit one-frame
   modes for new checkpoint work.
6. Run a tiny post-patch `--mode train` smoke before serious launches.

## Suggested Focused Test Command

```text
uv run pytest \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_curvytron_survivaldiag_manifest.py \
  tests/test_curvytron_opponent_mixture_manifest.py \
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py \
  tests/test_curvytron_survivaldiag_submitter.py \
  tests/test_curvytron_checkpoint_tournament.py -q
```

Do not treat this as enough for a learning claim. It only protects launch,
manifest, and cadence plumbing.
