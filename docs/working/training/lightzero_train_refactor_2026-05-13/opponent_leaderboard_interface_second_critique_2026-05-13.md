# Opponent Assignment Launcher Test Cut

Date: 2026-05-13

## Recommendation

Follow Dalton's ordering discipline here too: keep live hook/control-plane work
behind pure extraction. For this lane, that means pure snapshot/assignment
helpers and trainer tests before any Modal Dict wiring.

Conservative cut order:

1. pure assignment schema helpers in `src/curvyzero/training/opponent_registry.py`;
2. pure launcher adapter tests around assignment-to-mixture conversion;
3. trainer plumbing tests around `_run_visual_survival_train`;
4. only later, selector/controller code that reads a live Dict pointer.

The trainer should consume an explicitly supplied immutable assignment, not
discover one. Modal Dict is only a live pointer/cache for the future
selector/controller. Volume JSON is the durable source if assignment files are
immutable, hash-checked, and recorded in attempt metadata.

## Trainer Contract

Before wiring assignment snapshots into the Modal launcher, tests should prove:

1. assignment resolution happens once before `_build_visual_survival_configs`;
2. the result is the existing resolved `opponent_mixture` object;
3. stock `lzero.entry.train_muzero` receives only static config;
4. resume reuses the prior assignment unless an explicit refresh assignment is
   supplied;
5. no Modal Dict, leaderboard, selector, or `latest` pointer is reachable from
   inside `train_muzero`, LightZero hooks, or env reset/step.

## Current Seams To Preserve

- `parse_opponent_assignment_snapshot` is pure and does not read Dicts,
  Volumes, rankings, or checkpoints:
  `src/curvyzero/training/opponent_registry.py:21`.
- It rejects mutable/non-iteration frozen refs:
  `src/curvyzero/training/opponent_registry.py:79`.
- `_run_visual_survival_train` currently resolves `opponent_mixture_spec`
  before config build:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3224`.
- `_resolve_opponent_mixture_for_env` resolves checkpoint files and rejects bad
  frozen refs again:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4093`.
- `_build_visual_survival_configs` injects static `env.opponent_mixture`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4630`.
- The trusted call remains stock `train_muzero`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3620`.
- The env parses static mixture config once and selects per reset, not from a
  live source:
  `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:497`
  and `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:760`.

## Schema Recommendations

Keep the trainer-consumed schema small. Do not put leaderboard rows or ranking
logic in the file that `_run_visual_survival_train` reads.

Minimum assignment payload accepted by
`parse_opponent_assignment_snapshot` in
`src/curvyzero/training/opponent_registry.py:21`:

```json
{
  "schema_id": "curvyzero_opponent_assignment/v0",
  "assignment_id": "run123-attempt001-refresh000",
  "source_epoch": 42,
  "source_ref": "tournaments/curvytron/leaderboards/main/snapshots/42.json",
  "created_at": "2026-05-13T00:00:00Z",
  "seed": 7,
  "entries": [
    {
      "name": "recent_active_001",
      "weight": 10,
      "age_label": "recent",
      "tags": ["leaderboard", "active"],
      "opponent_policy_kind": "frozen_lightzero_checkpoint",
      "opponent_checkpoint_ref": "training/run/attempts/a/train/lightzero_exp/ckpt/iteration_120000.pth.tar"
    }
  ]
}
```

Launcher metadata should add, but not require the parser to own:

- `opponent_assignment_ref`
- `opponent_assignment_sha256`
- `opponent_assignment_id`
- `opponent_assignment_source_ref`
- `opponent_assignment_source_epoch`
- `opponent_assignment_refresh_index`

Recommended pure helper additions before launcher wiring:

- `canonical_assignment_json_sha256(value)` in
  `src/curvyzero/training/opponent_registry.py`
- `validate_opponent_assignment_snapshot(value)` or keep
  `parse_opponent_assignment_snapshot` but make `schema_id` strict in
  `src/curvyzero/training/opponent_registry.py:42`
- no `modal.Dict`, `modal.Volume`, `runs.resolve_mounted_ref_or_path`, or
  tournament imports in `src/curvyzero/training/opponent_registry.py`

## Required Tests Before Launcher Wiring

1. `test_assignment_ref_resolves_to_env_opponent_mixture_before_train_muzero`

   Fake an assignment JSON ref with one frozen exact checkpoint and one scripted
   entry. Monkeypatch file/ref resolution. Run `_run_visual_survival_train` with
   fake `lzero.entry.train_muzero`. Assert the captured main config has
   `env.opponent_mixture` with resolved checkpoint path/file metadata, and
   assert `train_muzero` was called once.

2. `test_assignment_metadata_is_written_to_command_summary_and_attempt_state`

   Assert command/summary include `opponent_assignment_ref`,
   `opponent_assignment_sha256`, `opponent_assignment_id`,
   `opponent_assignment_source_ref`, `opponent_assignment_source_epoch`,
   optional `opponent_assignment_refresh_index`, and the resolved
   `opponent_mixture`. This prevents "training used X, metadata says latest"
   drift.

3. `test_assignment_ref_and_opponent_mixture_spec_are_mutually_exclusive`

   If both are supplied, fail before config build. Do not invent precedence.
   This blocks accidental hand-built mixture overrides of a frozen assignment.

4. `test_bad_assignment_checkpoint_blocks_before_train_muzero`

   Parametrize assignment entries with `latest.pth.tar`, `ckpt_best.pth.tar`,
   `custom_ref.pth.tar`, and a missing `iteration_N.pth.tar`. Assert the
   launcher fails before fake `train_muzero` is called. This should exercise
   both `opponent_registry.py:79` and launcher-side file resolution.

5. `test_assignment_hash_mismatch_blocks_before_config_build`

   Supply `opponent_assignment_sha256`, corrupt the assignment JSON, and assert
   config build and `train_muzero` do not run.

6. `test_assignment_schema_id_is_required_and_exact`

   Add this to `tests/test_opponent_registry.py`. Reject missing or wrong
   `schema_id` before launcher wiring depends on the contract.

7. `test_assignment_hash_is_canonical_and_stable`

   Add this to `tests/test_opponent_registry.py`. Same assignment with
   different input key order should hash the same; changed entries should hash
   differently.

8. `test_resume_reuses_prior_assignment_by_default`

   Simulate auto-resume found at
   `lightzero_curvyzero_stacked_debug_visual_survival_train.py:3473`. Assert the
   launcher reuses the recorded prior assignment ref/hash and does not call any
   "current leaderboard" or "latest assignment" resolver.

9. `test_explicit_refresh_assignment_uses_new_assignment_id_without_overwriting_prior`

   Model refresh as a new launch-time assignment ref. Assert the command records
   the new assignment id/ref/hash, `refresh_index > 0` if present, and the prior
   assignment file is not modified.

10. `test_background_eval_and_gif_receive_same_assignment_metadata`

   Extend the existing mixture-threading test at
   `tests/test_opponent_mixture.py:223`. Assert background eval and GIF config
   receive the same resolved mixture plus assignment id/ref/hash as training.

11. `test_no_live_pointer_access_after_config_build`

   Monkeypatch the future Dict/leaderboard/latest resolver to raise after the
   assignment has been converted into config. Assert fake `train_muzero`,
   installed LightZero hooks, and env construction do not touch it. This is the
   regression tripwire for accidental polling inside the stock loop.

12. `test_assignment_wiring_preserves_stock_train_muzero_entrypoint`

   Extend `tests/test_curvytron_live_checkpoint_eval_plumbing.py:336` so the
   assignment path still reports `trainer_entrypoint == "lzero.entry.train_muzero"`
   and `called_train_muzero is True`.

## Exact Test Files

Put pure tests in `tests/test_opponent_registry.py`:

- strict schema id;
- canonical assignment hash;
- unknown top-level keys still rejected;
- frozen refs must be exact `iteration_N.pth.tar`.

Put launcher/trainer plumbing tests in
`tests/test_curvytron_live_checkpoint_eval_plumbing.py`:

- assignment ref resolves before `_build_visual_survival_configs`;
- metadata appears in command, summary, and attempt state;
- bad assignment blocks before fake `train_muzero`;
- resume reuses prior assignment;
- stock `train_muzero` entrypoint remains called.

Put background artifact propagation in `tests/test_opponent_mixture.py` near
`test_background_eval_config_threads_opponent_mixture_to_eval_and_gif`.

## Foot Guns To Guard

- **Dict becomes truth:** trainer reads `current:<leaderboard_id>` directly.
  Block with `test_no_live_pointer_access_after_config_build`.
- **Refresh becomes polling:** assignment changes at env reset or hook time.
  Block with resume/default and explicit-refresh tests.
- **Mutable refs sneak in:** assignment parser accepts `latest`, or launcher
  accepts a hand-built mixture bypass. Block with mutual-exclusion and bad-ref
  tests.
- **Metadata lies:** config uses one assignment while attempt metadata records
  another. Block with metadata and background propagation tests.
- **Resume silently upgrades opponents:** auto-resume picks a newer assignment
  because some resolver reads "latest." Block with prior-assignment reuse test.
- **Selector leaks into trainer:** launcher imports tournament ranking or bucket
  selection code. The trainer-side helper should only read/verify one provided
  assignment and resolve it into an `opponent_mixture`.
- **Hook extraction gets entangled:** do not combine assignment wiring with
  LightZero hook extraction. Keep this behind the same conservative boundary as
  resume/checkpoint extraction.

## Minimal Wiring Shape

After the pure tests above are green, add one launcher helper near
`_resolve_opponent_mixture_for_env` in
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4093`:

```text
_resolve_opponent_assignment_for_env(
    opponent_assignment_ref,
    opponent_assignment_sha256,
) -> resolved_assignment_metadata_and_mixture
```

It should read one assignment JSON, verify hash, call
`parse_opponent_assignment_snapshot`, then reuse `_resolve_opponent_mixture_for_env`
on the parsed `opponent_mixture`. It should not read Dicts, scan leaderboards,
choose "latest", or refresh anything.

Do not add live Modal Dict wiring in this cut. The first launcher patch should
only accept an explicit assignment ref/hash from the caller and prove that the
trusted `--mode train` lane remains a stock `train_muzero` call with static
config.
