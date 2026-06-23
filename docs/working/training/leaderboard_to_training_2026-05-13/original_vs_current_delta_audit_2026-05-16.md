# Original Vs Current Delta Audit - 2026-05-16

Status: docs-only audit. I did not edit training code, manifests, or launch
artifacts.

## Bottom Line

I did not find `mu0light0pat` as an exact local identifier. The closest empirical
CurvyTron baseline that actually showed survival signal is the 2026-05-10 native
LightZero frozen-opponent lane, especially `s92` matched against frozen `s47
iteration_200`. The closest code template baseline is LightZero v0.2.0
`zoo.atari.config.atari_muzero_config`, which our trainer still imports and
patches.

The strongest survival-hurting deltas in current r18fresh/all-v2 are:

1. Current training is a live refreshed weighted-opponent curriculum, while the
   best old survival signal was against one static matched frozen opponent.
2. Current collection is huge and chunky: `collector_env_num=256`,
   `n_episode=256`, `batch_size=32`, compared with LightZero Atari's `8/8/256`
   shape and several early CurvyTron canaries using tiny env counts.
3. Current search is shallow: `num_simulations=8` versus stock Atari `50`.
4. Dense survival/value targets can imply million-step returns but are capped to
   model support scale `300`.
5. Current `sparse_outcome` keeps stock `td_steps=5`, which is weak for long
   delayed survival outcome; dense variants avoid that somewhat but run into the
   support-cap issue.
6. Current stochastic rows can train on requested actions while transitions may
   have executed straight overrides/repeats.

These are not proof of a single bug. They are the most plausible config/algorithm
changes that could explain "finds a mid-run policy, then regresses" rather than
steady survival improvement.

## Closest Original Baseline

The useful old CurvyTron signal was not broad self-play. It was narrow
learner-vs-frozen-checkpoint training:

- The archived experiment explicitly labels the refresh runs as staged
  learner-vs-frozen-checkpoint, not live current-policy self-play
  (`docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md:1281`).
- `s92` used run id
  `curvytron-visual-survival-debug-lz-refresh-s47iter200-s92-65536`, attempt
  `train-gpu-l4t4-refresh-s47iter200-65536x256-s92-20260510`, opponent
  `s47 iteration_200`, and produced checkpoints through `iteration_434`
  (`docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md:1301`,
  `docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md:1342`).
- Its fixed-straight eval stayed flat, but matched frozen-opponent eval rose from
  `503.125` at `iteration_0` to `589.000` at `iteration_384` and `541.750` at
  `iteration_434`
  (`docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md:1426`,
  `docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md:1437`).
- A 32-seed confirmation strengthened the same read: `151.781` at
  `iteration_0`, `417.031` at `iteration_384`, and `500.438` at `iteration_434`
  against the matched frozen opponent
  (`docs/working/training/archive_2026-05-12_two_seat_purge/curvytron_next_native_experiment_decision_2026-05-10.md:94`).
- Follow-on long frozen-opponent run `s204` also showed a real but unstable
  matched-opponent lift, while `s206` got worse from a strong initialization
  (`docs/working/training/archive_2026-05-12_two_seat_purge/curvytron_background_training_ledger_2026-05-10.md:259`,
  `docs/working/training/archive_2026-05-12_two_seat_purge/curvytron_background_training_ledger_2026-05-10.md:282`).

So the baseline to preserve is not "general survival solved"; it is "stock
LightZero can improve against a static matched frozen opponent under small,
explicit runs."

## Current r18fresh/all-v2 Setup

Current source of truth:

- all-v2 storage/app/tournament names live in
  `src/curvyzero/contracts/curvytron.py:18`;
  current arena defaults are `curvy-r18fresh-live-bounded-dsf1-20260516b` /
  `elo-r18fresh-live-bounded-dsf1-20260516b`
  (`src/curvyzero/contracts/curvytron.py:33`).
- The active manifest is
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18fresh-allv2-20260516a/curvy-r18fresh-allv2-20260516a.json`
  (`docs/working/training/leaderboard_to_training_2026-05-13/survival_stagnation_investigation_2026-05-16.md:13`).
- Matrix shape is `18` rows = `3` reward variants x `3` opponent recipes x `2`
  noise modes (`scripts/build_curvytron_tonight18_manifest.py:973`,
  `scripts/build_curvytron_tonight18_manifest.py:1048`).
- Fixed training knobs are `collector_env_num=256`, `n_episode=256`,
  `num_simulations=8`, `batch_size=32`, `lightzero_eval_freq=0`,
  `source_max_steps=1048576`, and background eval/GIF enabled
  (`scripts/build_curvytron_tonight18_manifest.py:642`,
  `scripts/build_curvytron_tonight18_manifest.py:694`).
- Reward axes are `sparse_outcome`, `survival_plus_bonus_no_outcome`, and
  `survival_plus_bonus_plus_outcome`
  (`src/curvyzero/contracts/curvytron.py:101`,
  `src/curvyzero/contracts/curvytron.py:106`).
- Opponent recipes are weighted mixtures with 20-30% immortal
  blank/wall-avoidant pressure and the rest ranked/checkpoint-style slots
  (`scripts/build_curvytron_tonight18_manifest.py:95`,
  `scripts/build_curvytron_tonight18_manifest.py:1248`).
- Current read: `18/18` runs have a best checkpoint above iteration 0, but only
  `10/18` latest checkpoints are above iteration 0; mean first/best/latest is
  `159.9 / 246.0 / 175.4`
  (`docs/working/training/leaderboard_to_training_2026-05-13/survival_stagnation_investigation_2026-05-16.md:31`).

## Delta Audit

### 1. Static matched opponent -> live refreshed opponent curriculum

Old useful signal: one frozen opponent, evaluated against that same opponent.
Current r18fresh: the trainer consumes assignment refs and mutable refresh
pointers; the deployed loop has fed generation 9/10/12 assignments back into
still-running trainers with provider-load success
(`docs/working/training/leaderboard_to_training_2026-05-13/NOW.md:18`).

Code path: assignment refresh buckets are train-iteration based
(`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5408`),
and refreshed assignments reset env opponent mixtures
(`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5435`).

Why it could hurt survival: replay now spans changing opponent distributions.
That can turn the problem into a nonstationary curriculum before the policy has
learned a stable survival primitive. It also makes latest-vs-best regression more
likely than the old static-opponent runs.

### 2. LightZero stock cadence -> 256-env collection waves with small batches

LightZero Atari MuZero baseline: `collector_env_num=8`, `n_episode=8`,
`num_simulations=50`, `update_per_collect=None`, `replay_ratio=0.25`,
`batch_size=256` (`/tmp/lightzero-src-audit-20260516/zoo/atari/config/atari_muzero_config.py:10`).
LightZero computes updates after each collect and samples `batch_size` rows per
learner update (`/tmp/lightzero-src-audit-20260516/lzero/entry/train_muzero.py:186`,
`/tmp/lightzero-src-audit-20260516/lzero/entry/train_muzero.py:197`).

Current r18fresh sets `collector_env_num=256`, `n_episode=256`,
`batch_size=32` (`scripts/build_curvytron_tonight18_manifest.py:651`).

Why it could hurt survival: each collect phase is a large policy-staleness chunk,
then learner updates use an 8x smaller minibatch than stock Atari. That is a
very different gradient/noise regime even if LightZero's replay-ratio rule is
technically unchanged.

### 3. Search budget dropped hard

Stock Atari MuZero uses `num_simulations=50`
(`/tmp/lightzero-src-audit-20260516/zoo/atari/config/atari_muzero_config.py:13`).
Current r18fresh uses `num_simulations=8`
(`scripts/build_curvytron_tonight18_manifest.py:655`), and the trainer patches
that directly into the LightZero policy config
(`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6195`).

Why it could hurt survival: CurvyTron has only 3 ego actions, so `8` simulations
is not automatically invalid, but collision survival is sensitive to short
horizon consequences. With stock root noise and shallow search, policy targets
can be noisy, especially in the stochastic action-override lanes.

### 4. Reward/value scale changed more than the support can represent

For current fixed-opponent runs, `_lightzero_target_config_for_reward` sets
`discount_factor=1.0`, computes support scale from `source_max_steps`, then caps
support scale at `300`
(`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:749`,
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:778`).
The environment cap is `1_048_576` source steps
(`src/curvyzero/contracts/curvytron.py:74`).

Current reward semantics:

- `survival_plus_bonus_no_outcome`: alive helper plus same-step bonus, no terminal
  outcome (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1746`).
- `survival_plus_bonus_plus_outcome`: alive helper plus bonus plus terminal
  sparse outcome scaled by physical step index
  (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1760`).

Why it could hurt survival: dense variants can produce value/reward magnitudes
far beyond `300`, so LightZero's categorical targets may saturate. The existing
mechanics audit calls this the largest actionable risk
(`docs/working/training/leaderboard_to_training_2026-05-13/survival_mechanics_reward_audit_2026-05-16.md:7`).

### 5. Long-horizon outcome with stock TD horizon

The trainer intentionally keeps stock `td_steps` for
`source_state_fixed_opponent` unless a separate target-horizon knob is added
(`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:789`).
That fixed a prior crash where `td_steps=source_max_steps=1048576`
(`docs/working/training/leaderboard_to_training_2026-05-13/NOW.md:1370`).

Why it could hurt survival: for `sparse_outcome`, the signal is still a long
delayed terminal outcome backed up over a stock 5-step horizon. For dense
variants, the immediate alive helper helps, but the support cap above becomes the
main risk.

### 6. Requested action -> possibly different executed action in stochastic rows

The manifest includes a stochastic lane with straight override probability `0.10`
and one-extra-step repeat probability `0.10`
(`scripts/build_curvytron_tonight18_manifest.py:152`). The env reward/action info
does track requested and executed actions, but LightZero's normal GameSegment
action is the policy-selected action, not a separate executed-action field.

Why it could hurt survival: in these rows, model learning can be asked to explain
a transition caused partly by an action override/repeat. That may be valid
environment stochasticity, but it is another departure from the clean old static
matched-opponent read.

### 7. Single-agent fixed-opponent formulation remains different from true self-play

The current trainer still imports LightZero's Atari config template and patches
`env_type="not_board_games"` plus one learner action per step
(`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6140`,
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6191`).
The env reports the fixed-opponent lane is not two-seat self-play
(`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:477`).

This was also true of the old successful signal, so it is not a new regression by
itself. The new risk is combining that single-agent formulation with a changing
opponent population and tournament feedback, then reading latest checkpoint
survival as if it were stable self-play improvement.

## Practical Control To Isolate The Regression

Run one deliberately boring static control before changing algorithms:

- one reward variant first, preferably `survival_plus_bonus_plus_outcome` only if
  support scaling is also reduced/controlled; otherwise use sparse as a clean
  baseline;
- one immutable opponent assignment, no refresh pointer;
- no stochastic override/repeat;
- smaller collection wave or explicit replay/batch ladder: e.g. `collector_env_num`
  `32` or `64`, `batch_size` `128` or `256`;
- compare eval against the training opponent and one held-out opponent, like the
  old s92 discipline.

If that holds latest near best, the live curriculum/replay nonstationarity is the
likely culprit. If it still finds a mid-run best then regresses, look first at
support scale, search budget, and TD/replay cadence.
