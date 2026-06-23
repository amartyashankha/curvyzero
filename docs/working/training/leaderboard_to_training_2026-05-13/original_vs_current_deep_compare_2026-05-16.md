# Original vs Current Deep Compare - 2026-05-16

Status: docs-only comparison. I did not edit training code, manifests, launch
artifacts, or existing docs.

## Scope and Baseline Anchors

I did not find `mu0`, `light0`, or `pat` as exact local training identifiers.
The closest usable "original" anchors are:

- Code/config baseline: LightZero v0.2.0 Atari MuZero config, which the CurvyTron
  trainer still imports and patches. Local audit checkout:
  `/tmp/lightzero-src-audit-20260516`, commit
  `de74055298068f53b70e07bc38c41101fce51766`.
- CurvyTron empirical baseline: 2026-05-10 staged learner-vs-static-frozen
  checkpoint runs, especially `s92` against frozen `s47 iteration_200`.
- Git-history original CurvyTron source-state wrapper: commit `23c721b`, where
  the source-state survival env was player-0-only and simpler.

Bottom line: the current system is more sophisticated and less obviously wrong
at the plumbing layer, but it is no longer the same learning problem. The old
survival lift came from a static matched frozen opponent. Current `r18fresh`
trains from scratch against a refreshed weighted opponent curriculum, with huge
collection waves, small learner batches, shallow search, and dense returns that
can exceed model support by orders of magnitude. That combination can explain
"best checkpoint improves mid-run, latest regresses."

## Closest Original Behavior

The strongest historical signal was narrow:

- `s90/s91/s92` are explicitly described as staged
  learner-vs-frozen-checkpoint runs, not live current-policy self-play
  (`docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md:1281`).
- `s92` trained against frozen `s47 iteration_200`
  (`docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md:1301`).
- The 8-seed matched eval improved from `503.125` at `iteration_0` to `589.000`
  at `iteration_384`, then `541.750` at `iteration_434`
  (`docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md:1426`).
- The 32-seed confirmation improved from `151.781` at `iteration_0` to
  `417.031` at `iteration_384` and `500.438` at `iteration_434`
  (`docs/working/training/archive_2026-05-12_two_seat_purge/curvytron_next_native_experiment_decision_2026-05-10.md:94`).

That is evidence that native LightZero can learn a matched frozen-opponent
survival behavior. It is not evidence that live tournament feedback or broad
opponent mixtures are stable.

## Current Behavior Under Investigation

Current aggregate read: `18/18` `r18fresh` rows found a best checkpoint above
iteration 0, but only `10/18` latest checkpoints were above iteration 0 and only
`4/18` latest checkpoints were within 10% of their own best. Mean first/best/latest
survival was `159.9 / 246.0 / 175.4`
(`docs/working/training/leaderboard_to_training_2026-05-13/survival_stagnation_investigation_2026-05-16.md:31`).

This is not "nothing learns." It is "many rows find something and then lose it."

## Deep Delta Matrix

| Area | Original / closer baseline | Current implementation | Regression relevance |
| --- | --- | --- | --- |
| Opponent setup | Static fixed-straight or one matched frozen checkpoint. `s92` used frozen `s47 iteration_200`. | 18-row matrix uses weighted recipes with blank, wall-avoidant immortal, rank slots, and assignment refresh pointers. Recipes are in `scripts/build_curvytron_tonight18_manifest.py:100`; refresh pointers are built at `scripts/build_curvytron_tonight18_manifest.py:1021`. | Very high. Nonstationary opponents plus old replay can make latest drift below best. |
| Player perspective / role | Git baseline `23c721b` source-state env effectively forced `ego_player_index=0`; see historical error path from `git show 23c721b:...curvyzero_source_state_visual_survival_lightzero_env.py`, lines `180-183`. | Current default is `random_per_episode` (`src/curvyzero/contracts/curvytron.py:91`, `src/curvyzero/contracts/curvytron.py:99`), selected deterministically per reset (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:914`). | Probably fixed/improved, not top regression suspect. It changes the distribution, though, so old fixed-seat curves are not perfectly comparable. |
| Training relation | Old useful signal was learner vs one frozen opponent. | Current fixed-opponent lane still declares `two_seat_self_play=False` (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:479`) and records `current_policy_two_seat_action_collection=False` in trainer metadata (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4315`). | Medium-high. This is acceptable as a lane, but tournament feedback can be over-read as self-play. |
| Reward definition | Git baseline source-state wrapper gave dense alive reward only: reward += survival after each step (`git show 23c721b:...curvyzero_source_state_visual_survival_lightzero_env.py`, lines `359-365`). Early `s92` used survival-step eval against matched opponent. | Current matrix has `sparse_outcome`, `survival_plus_bonus_no_outcome`, and `survival_plus_bonus_plus_outcome` (`src/curvyzero/contracts/curvytron.py:101`). Current env implements bonus and terminal outcome scaled by physical step index (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1730`). | Very high, especially with support saturation. Sparse has weak delayed credit; dense can saturate support. |
| Action / no-op behavior | Three actions existed, but original source-state fixed opponent was straight-only (`git show 23c721b:...curvyzero_source_state_visual_survival_lightzero_env.py`, lines `348-351`). | Canonical action names are `left`, `straight`, `right` (`src/curvyzero/env/trainer_contract.py:59`). Current noise lane can override learner action to straight and repeat actions (`scripts/build_curvytron_tonight18_manifest.py:146`, `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1659`). | High in stochastic rows. Replay stores selected action while transition may reflect straight override/repeat. |
| Observation/render mode | Early source-state env rendered a gray64 source-state stack and normalized player perspective (`git show 23c721b:...curvyzero_source_state_visual_survival_lightzero_env.py`, lines `455-478`). | Current production policy surface is controlled-player view with `browser_lines + simple_symbols` (`src/curvyzero/env/observation_surface_contract.py:27`, `src/curvyzero/env/observation_surface_contract.py:28`), and env metadata records player-perspective observation slices (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:2122`). | Medium. Current is more faithful and better tagged, not a primary suspect unless render parity is wrong. |
| MCTS params | LightZero Atari MuZero baseline uses `num_simulations=50` (`/tmp/lightzero-src-audit-20260516/zoo/atari/config/atari_muzero_config.py:13`). Early CurvyTron refresh also used low sims, but with simpler opponents. | Current manifest defaults `num_simulations=8` (`scripts/build_curvytron_tonight18_manifest.py:68`) and passes it to trainer kwargs (`scripts/build_curvytron_tonight18_manifest.py:660`). | High. With 3 actions, 8 may be serviceable, but root noise and collision horizon can make targets noisy. |
| Model support / value scaling | LightZero Atari defaults are support-sized for Atari-ish rewards (`/tmp/lightzero-src-audit-20260516/lzero/policy/muzero.py:166`, `/tmp/lightzero-src-audit-20260516/lzero/policy/muzero.py:168`). | Current target config uses `discount_factor=1.0`, computes dense requested scales from `source_max_steps`, then caps to `SOURCE_STATE_FIXED_OPPONENT_MAX_MODEL_SUPPORT_SCALE=300` (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:753`, `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:784`). | Very high. Dense variants can imply million-step returns but share a 601-bin `[-300,300]` support. |
| Collector/env counts | LightZero Atari baseline uses `collector_env_num=8`, `n_episode=8` (`/tmp/lightzero-src-audit-20260516/zoo/atari/config/atari_muzero_config.py:10`). Trainer defaults were tiny in original CurvyTron (`git show 23c721b:...lightzero_curvyzero_stacked_debug_visual_survival_train.py`, lines `156-162`). | Current 18-run builder defaults to `collector_env_num=256`; if `n_episode<=0`, it sets `n_episode=collector_env_num` (`scripts/build_curvytron_tonight18_manifest.py:68`, `scripts/build_curvytron_tonight18_manifest.py:1542`). | High. Large stale collect waves can produce mid-run peaks then noisy or stale updates. |
| Replay/batch settings | LightZero Atari baseline uses `update_per_collect=None`, `replay_ratio=0.25`, `batch_size=256` (`/tmp/lightzero-src-audit-20260516/zoo/atari/config/atari_muzero_config.py:14`). LightZero computes updates after collect and samples batches in `train_muzero` (`/tmp/lightzero-src-audit-20260516/lzero/entry/train_muzero.py:186`). | Current batch size is `32` (`scripts/build_curvytron_tonight18_manifest.py:70`) while stock replay ratio remains inherited unless patched. Replay buffer is still stock-style, but now under changing opponents. | High. Same replay ratio with 8x smaller batches and 32x larger collection chunks is a different optimizer regime. |
| Checkpoint/eval cadence | Stock Atari eval freq is `2000` (`/tmp/lightzero-src-audit-20260516/zoo/atari/config/atari_muzero_config.py:80`). Early CurvyTron used very frequent checkpointing for inspection (`docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md:614`). | Current save cadence is `10_000` (`src/curvyzero/contracts/curvytron.py:77`) and stock LightZero eval is set to `0` in manifest kwargs (`scripts/build_curvytron_tonight18_manifest.py:664`), relying on external background eval. | Medium diagnostic risk. It does not directly cause regression, but it makes best-vs-latest selection depend on external eval/tournament timing. |
| Tournament feedback | Original survival signal was offline evaluation against the training opponent, not a live leaderboard loop. | Current loop intakes immutable checkpoints, rates direct checkpoint-vs-checkpoint games, materializes leaderboard assignments, and refreshes opponents. Tournament seating is balanced per game spec (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:2238`), rating uses seat-aware checkpoint wins (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:2354`), and assignments select top/stable slots (`src/curvyzero/training/opponent_leaderboard.py:404`). | Very high as a system-level suspect. The feedback loop may be working as designed but still creates a moving curriculum before survival skill is stable. |

## Important Non-Suspects / Exonerations

- The current tournament player-perspective path appears correct: each seat gets
  `observation[0, seat]`, controls the same physical seat, and the metadata says
  so (`src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3619`,
  `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3769`).
- The current learner-seat fix is real: `random_per_episode` is a first-class
  default (`src/curvyzero/contracts/curvytron.py:99`) and reset selection sets
  physical player 0 or 1 (`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:914`).
- Straight/no-op is not an illegal action bug. It is the canonical middle action
  (`src/curvyzero/env/trainer_contract.py:59`). The risk is learning dynamics:
  straight may be over-attractive or hidden by stochastic override, not invalid.
- Checkpoint refs are more strictly immutable now. Frozen opponent mixture entries
  reject mutable refs and require `iteration_N.pth.tar`
  (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5330`).

## Why Mid-Run Improvement Then Regression Fits These Deltas

1. A scratch policy can find a local survival behavior against the current
   opponent slice.
2. The next 256-env collect wave can be generated under stale policy and a
   refreshed or different opponent assignment.
3. Replay keeps older opponent distributions and old MCTS targets without
   reanalysis.
4. Dense value targets past a few hundred steps may collapse into the same support
   edge, weakening the value head's ability to separate mediocre and excellent
   survival.
5. Low-simulation targets plus stock exploration noise can push the policy away
   from the discovered behavior.
6. Tournament feedback rewards relative checkpoint wins, not necessarily stable
   survival against the fixed matched opponent that exposed the original signal.

That pattern naturally produces "best improves, latest falls" without requiring
one catastrophic code bug.

## Prioritized Suspects

1. Dense reward/value support saturation.
   Exact refs: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:753`,
   `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:784`,
   `src/curvyzero/contracts/curvytron.py:76`,
   `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1760`.

2. Live refreshed opponent curriculum before stability.
   Exact refs: `scripts/build_curvytron_tonight18_manifest.py:100`,
   `scripts/build_curvytron_tonight18_manifest.py:1021`,
   `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4354`,
   `src/curvyzero/training/opponent_leaderboard.py:404`.

3. Collection/replay cadence mismatch: 256-env collection waves with batch 32.
   Exact refs: `scripts/build_curvytron_tonight18_manifest.py:68`,
   `scripts/build_curvytron_tonight18_manifest.py:70`,
   `/tmp/lightzero-src-audit-20260516/zoo/atari/config/atari_muzero_config.py:10`,
   `/tmp/lightzero-src-audit-20260516/zoo/atari/config/atari_muzero_config.py:16`,
   `/tmp/lightzero-src-audit-20260516/lzero/entry/train_muzero.py:186`.

4. Shallow search targets.
   Exact refs: `scripts/build_curvytron_tonight18_manifest.py:69`,
   `scripts/build_curvytron_tonight18_manifest.py:660`,
   `/tmp/lightzero-src-audit-20260516/zoo/atari/config/atari_muzero_config.py:13`,
   `/tmp/lightzero-src-audit-20260516/lzero/policy/muzero.py:216`.

5. Stochastic selected-action vs executed-action mismatch.
   Exact refs: `scripts/build_curvytron_tonight18_manifest.py:146`,
   `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:979`,
   `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1659`,
   `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1940`.

6. Sparse-outcome temporal credit remains too weak.
   Exact refs: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:800`,
   `/tmp/lightzero-src-audit-20260516/lzero/policy/muzero.py:170`,
   `/tmp/lightzero-src-audit-20260516/lzero/policy/muzero.py:172`.

7. External eval/tournament feedback is diagnostically different from the
   original matched-opponent eval.
   Exact refs: `scripts/build_curvytron_tonight18_manifest.py:664`,
   `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:2354`,
   `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:2538`,
   `src/curvyzero/tournament/checkpoint_intake_service.py:8`.

## Cleanest Next Control

The highest-signal next control is not another full 18-row loop. Run a boring
static lane:

- no tournament refresh pointer;
- one immutable opponent assignment or one frozen checkpoint;
- clean actions only, no straight override/repeat;
- `learner_seat_mode=random_per_episode`;
- raise `batch_size` toward `128` or `256` or shrink `collector_env_num`;
- test `num_simulations=25` before `50`;
- either scale dense rewards down or raise/remove the support cap deliberately.

If that keeps latest near best, the main culprit is live curriculum plus cadence.
If it still peaks then regresses, prioritize support scaling and MCTS/cadence
before touching tournament code.
