# Next Matrix Manifest Design

Date: 2026-05-12/13

Purpose: working design notes for the stock LightZero CurvyTron run tensor.
The first survivaldiag tensor has now launched as `survivaldiag-v1b-20260513h`.
The older 48-row sketch below is stale and is kept only as historical raw
material while next-wave tensors are designed around repeat groups, separated
seed meanings, and staged blocks.

STOP: do not launch rows directly from this document. The active truth is stock
`train_muzero`, reward `survival_plus_bonus_no_outcome`,
`opponent_runtime_mode=blank_canvas_noop` as the anchor lane, matched render
twins for serious cells, repeated seed/copy groups, and no scripted/random
opponent rows until those lanes are wired and canaried. Rich status/export
fields are cleared for the launched first-wave exact lanes.

## Current Status

Do not launch a new matrix from this document yet.

The current dry-run generator emitted the 50-row
`survivaldiag-v1b-20260513h` review manifest plus 10 gated specs and
`current_launch_approved=false`. That batch is running. Anything below that
looks like another runnable matrix is either a design target or a stale sketch
until the gates in this doc and the current source of truth are satisfied.

Aggressive scale is allowed after gates pass. See
[aggressive_matrix_scale_plan.md](aggressive_matrix_scale_plan.md). The target
is organized scale: about 50, 100, or 200+ rows depending on which opponent
families are actually implemented and canaried.

## 2026-05-13 Matrix Refinement

After the first launch, the next matrix should still be staged, not Cartesian.
The main diagnostic claim starts with
`opponent_runtime_mode=blank_canvas_noop` plus
`survival_plus_bonus_no_outcome`, because this is the cleanest test of visual
wall-avoidance learning. Passive `opponent_death_mode=immortal` remains a
separate canary/control, not the main lane, unless it is deliberately labeled as
an out-of-bounds trail-generator diagnostic.

Recommended batch order:

1. Blank-canvas deterministic sanity, with matched fast/browser renders.
2. Blank-canvas stochasticity ladder, repeating only the stochastic levels that
   drive the claim.
3. Fixed-straight passive-immortal control, labeled as a dirty trail-maker, with
   minimal repeats.
4. Frozen-checkpoint ancestor controls, one or two copies only after exact
   checkpoint identity and lane canaries clear.
5. Scripted or random/checkpoint opponent blocks only after the exact opponent
   kind is wired, manifest-addressable, and canaried; use repeated copies only
   after that.
6. Tiny reward ablation and sim16/C64/B64 sentinels only on the strongest cell.

If scale budget is high, add rows by increasing repeated copies and confirmation
blocks first. Do not multiply every axis equally.

Avoid repeating render twins as independent evidence. Render rows should share
the same logical seed/copy so differences can be read as renderer effects, not
as lucky training runs.

The current phase is running-batch monitoring and next-wave design cleanup:

- keep old v1d evidence as context, not as launch authority;
- keep the dry-run row schema explicit and review-only;
- keep passive immortal as a dirty control unless deliberately promoted;
- for any next-wave opponent family, clear exact-lane canaries and real
  rich-status/eval snapshots before another large launch.

## Revised Design Principles

| Principle | Reason |
| --- | --- |
| Use staged blocks, not one giant crossed product. | If everything changes at once, we cannot tell what helped. |
| Keep main reward fixed. | Main lane should be survival plus bonus, with outcome reward off/zero. |
| Separate opponent interventions. | Immortal opponent and no-trail opponent answer different questions. Do not bundle them first. |
| Pair renders deliberately. | Serious rows should have fast/browser twins with the same seed and knobs. |
| Make stochasticity a real axis. | v1d only tested tiny `0.05`; this is underswept. |
| Keep search/collector/batch as sentinels. | v1d did not show these as the main blocker. |
| Name rows for analysis. | The row id must expose block, reward, opponent, render, stochasticity, seed, and sentinel status. |
| Repeat important stochastic rows. | Random opponent policies and stochastic environments need several copies before we trust a result. |

## Future Tensor Shape

The future tensor should be built as staged blocks, not as one large crossed
product. Each block should have a single question, a small row set, and an
explicit rule for how many repeated copies it receives.

Copy-count rule:

| Row family | Default copies | Reason |
| --- | ---: | --- |
| Random/checkpoint opponent rows | about `5` after wiring/canary | Opponent identity or initialization can dominate the trail distribution. |
| Stochastic opponent behavior rows | about `5` | One behavior seed can look unusually easy or hard. |
| Stochastic reset/environment rows that drive the claim | about `5` | Reset luck can otherwise masquerade as learning. |
| Ancestor checkpoint controls | `1` to `2` after gate | The policy is already trained/frozen, so these are controls unless they become the main signal. They still need exact identity/canary review before launch. |
| Render twins | matched, not extra evidence | `body_circles_fast` and `browser_lines` should share the same logical seed/copy when comparing renders. |

Seed fields should be separate in the manifest. Do not hide all randomness
behind one `seed` column.

| Manifest field | Meaning |
| --- | --- |
| `training_seed` | learner initialization and LightZero-level randomness |
| `reset_seed` | starts, headings, bonus placement, and environment resets |
| `opponent_policy_seed` | random opponent initialization or sampled policy choice |
| `opponent_behavior_seed` | stochastic actions/noise inside the opponent policy |
| `eval_seed` | held-out eval randomness |
| `copy_id` | stable label for repeated copies of the same logical setting |
| `reward_survival_weight` | survival reward weight actually used by the trainer |
| `reward_bonus_weight` | bonus pickup reward weight actually used by the trainer |
| `reward_outcome_weight` | terminal outcome weight, expected zero in diagnostic lane |
| `opponent_runtime_mode` | normal, blank-canvas/no-op, scripted, or other exact runtime contract |
| `opponent_trail_mode` | whether opponent leaves collision/visual trail |
| `opponent_collision_effect` | whether learner can die on opponent trail/body |
| `opponent_visibility_mode` | whether opponent appears in observations/GIFs |
| `logical_pair_id` | groups matched fast/browser rows or compute sentinels |
| `render_pair_role` | fast/browser member of a matched pair |
| `block_id` | staged matrix block the row belongs to |
| `hypothesis_id` | question the row answers |
| `primary_metric` | survival, reward, or other main readout for the block |

Repeated copies should vary only the seed fields needed by the row family. For
example, a random-policy opponent copy group should vary
`opponent_policy_seed`; a behavior-noise copy group should vary
`opponent_behavior_seed`; a reset-stability group should vary `reset_seed`.
When a row mixes these sources, the manifest should show which seeds moved.

## Better Shape

Start with blocks, staged in this order:

1. Blank-canvas sanity and stochasticity ladder:
   `blank_canvas_noop`, `survival_plus_bonus_no_outcome`, high cap, matched
   fast/browser renders, repeated copy groups.
2. Normal/fixed controls: minimal rows under the same reward and render-pair
   rules, interpreted as controls rather than the main claim.
3. Passive immortal dirty canary: fixed-straight passive trail-maker only if
   deliberately labeled dirty, with small copy count and GIF/manual checks.
4. Tiny reward ablation: survival-only only after the main reward lane is moving
   or if a canary requires it; do not re-open a broad reward sweep.
5. Projection sentinels: sim16, C64, and B64 only on the strongest diagnostic
   cell.
6. Scripted or random/checkpoint opponent families: second wave only after
   first-class trainer wiring, immutable identity/seed fields, and e2e canaries.

The question for each block should be explicit before rows are generated.

## Stale Historical Draft Below

The remaining row tables are the old 48-row draft. Keep them only as raw
material for the next generator design. Do not implement them as-is, do not
launch them, and do not treat their row count as a target.

Known problems in the old draft:

- it uses stale reward names such as `survival_plus_bonus`;
- it bundles `immortal` and `no-trail`;
- it treats seed slots as one opaque seed instead of separated seed/copy
  fields;
- it can look runnable even though no launch has been approved.

Current truth: use `survival_plus_bonus_no_outcome` for the main stock trainer
lane. `blank_canvas_noop` is the anchor. Scripted/random opponent rows stay out
until wired and canaried. Status/export fields for reward components, bonus,
terminal causes, action histograms/entropy, and eval/GIF health remain launch
blockers.

## Archived Matrix Sketch - Not Runnable

Name: `stock-survivaldiag-v1e`

Default prefixes:

- run prefix: `curvytron-stock-stock-survivaldiag-v1e`
- attempt prefix: `stock-survivaldiag-v1e-attempt`

Old sketch row count: `48`. This is not currently approved for launch.

Optional reward-ablation addendum: `+4` rows, only if the reward wiring needs a
minimal sanity comparison before launch.

Primary question:

```text
Can stock train_muzero learn visual survival when reward is survival-first and
weak opponent deaths cannot saturate outcome?
```

## Archived Row Shape - Not Runnable

Core diagnostic grid: `32` rows.

| Axis | Values | Notes |
| --- | --- | --- |
| Reward | stale: `survival_plus_bonus` | Replace with `survival_plus_bonus_no_outcome` before any real manifest. Outcome reward must be off/zero. |
| Opponent setting | old draft: `fixed-normaltrail-normaldeath`, `fixed-notrail-immortal` | Stale: bundles no-trail with immortal. Split before launch. |
| Render | `body_circles_fast`, `browser_lines` | Real matched axis for every serious row. Do not reduce browser rows to sentinels. |
| Stochasticity | `none`, `straight005`, `straight015`, `straight030` | More aggressive than v1d; implemented initially via straight-action override probability. |
| Seed slot | `s701`, `s702` | Keep paired across all core axes. |

Control rows: `8` rows.

| Axis | Values |
| --- | --- |
| Opponent setting | `old-normaltrail-normaldeath`, `recent-normaltrail-normaldeath` |
| Reward | stale: `survival_plus_bonus`; use `survival_plus_bonus_no_outcome` in any future real design |
| Stochasticity | `none`, `straight015` |
| Render | `body_circles_fast`, `browser_lines` |
| Seed slot | `s703` |

Sentinel rows: `8` rows, all on
stale `survival_plus_bonus/fixed-notrail-immortal/straight015`.

| Sentinel | Rows | Override |
| --- | ---: | --- |
| Search | 2 | `num_simulations=16`, both renders, seed `704` |
| Collector | 2 | `collector_env_num=64`, `n_episode=64`, both renders, seed `705` |
| Learner batch | 2 | `batch_size=64`, both renders, seed `706` |
| Death/trail isolation | 2 | `fixed-normaltrail-immortal`, both renders, seed `707` |

Optional reward ablation: `4` rows, not part of the broad matrix.

| Axis | Values |
| --- | --- |
| Reward | `survival_only` |
| Opponent setting | stale `fixed-notrail-immortal`; do not use without redesign |
| Stochasticity | `straight015` |
| Render | `body_circles_fast`, `browser_lines` |
| Seed slot | `s708`, `s709` |

Do not sweep episode cap. Set `source_max_steps=65536` everywhere.

## Labeling

Use compact labels so generated ids stay under the existing safety limit:

```text
<reward>-<opp>-<render>-<stoch>-<seed>-b<batch>-sim<sims>
```

Tokens:

- reward: `survonly`, `survbonus`
- opponent: `fixed-norm`, `fixed-immnt`, `fixed-immnorm`, `old-norm`, `recent-norm`
- render: `fast`, `browser`
- stochasticity: `stoch0`, `stoch005`, `stoch015`, `stoch030`

Examples:

- `survbonus-fixed-immnt-fast-stoch015-s701-b32-sim8`
- `survbonus-fixed-norm-browser-stoch030-s702-b32-sim8`
- `survbonus-recent-norm-fast-stoch015-s703-b32-sim8`
- `survbonus-fixed-immnt-browser-stoch015-s704-b32-sim16`
- optional ablation: `survonly-fixed-immnt-fast-stoch015-s708-b32-sim8`

## Archived Common Flags - Not Runnable

This is an old command fragment kept for context. Do not copy it into a
launcher or manifest generator.

```text
--mode train
--env-variant source_state_fixed_opponent
--compute gpu-l4-t4-cpu40
--max-train-iter 100000
--max-env-step 10000000
--save-ckpt-after-iter 2000
--collector-env-num 32
--evaluator-env-num 1
--n-evaluator-episode 1
--n-episode 32
--source-max-steps 65536
--batch-size 32
--num-simulations 8
--lightzero-eval-freq 0
--env-manager-type subprocess
--background-eval-launch-kind poller
--background-eval-seed-count 8
--background-eval-max-steps 65536
--background-eval-num-simulations 8
--background-eval-batch-size 64
--background-gif-max-steps 4096
--background-gif-frame-stride 4
--output-detail compact
```

Per-row flags:

- main rows: stale `--reward-variant survival_plus_bonus`; any real manifest
  must use `--reward-variant survival_plus_bonus_no_outcome`
- optional ablation rows only: `--reward-variant survival_only`
- `--source-state-trail-render-mode body_circles_fast|browser_lines`
- `--opponent-death-mode normal|immortal`
- `--opponent-trail-mode normal|none` once implemented
- `--ego-action-straight-override-probability 0.0|0.05|0.15|0.30`
- `--control-noise-profile-id none|straight_override_0.05|straight_override_0.15|straight_override_0.30`
- frozen controls use existing `--opponent-policy-kind frozen_lightzero_checkpoint`,
  immutable checkpoint refs, and snapshot refs for `old` and `recent`.

## Generator Extension Notes

If/when a real manifest generator is updated, extend
`scripts/build_curvytron_stock_train_manifest.py` rather than adding a separate
launcher. Treat the old matrix name below as archival until a fresh row list is
approved.

Needed changes:

1. Add matrix name `stock-survivaldiag-v1e` to `--matrix-name`, `_rows_for_matrix`,
   and the unknown-matrix error.
2. Add `Row` fields and manifest/command output for:
   `opponent_death_mode`, `opponent_trail_mode`, and any stock bonus reward knob
   needed by `survival_plus_bonus_no_outcome`.
3. Pass `--opponent-death-mode` through generated commands. The trainer has the
   stock CLI surface; the generator does not currently emit it.
4. Do not resurrect stale reward constants from this sketch. The current stock
   diagnostic reward is `survival_plus_bonus_no_outcome`; optional
   `survival_only` rows, if used, must be behind an explicit
   `--include-reward-ablation` flag and marked as ablation rows.
5. Add `opponent_trail_mode` only after the stock env/trainer has that flag.
   Current source-state render mode is available; opponent trail suppression is
   a separate missing control.
6. Keep render as an explicit per-row axis. Do not use one global
   `--source-state-trail-render-mode` value for this matrix.
7. Encode stochasticity levels from a small table that sets both
   `ego_action_straight_override_probability` and `control_noise_profile_id`.
8. Preserve existing guards: `--mode train`, stock env variants only, no
   `two-seat-selfplay`, immutable frozen checkpoint refs, background eval/GIF
   refs in metadata.

Do not add broad search, collector, learner-batch, or cap sweeps. The v1d
projection says those are secondary; keep them as the eight sentinel rows above.
