# Survival Baseline Delta Audit - 2026-05-16

Status: findings only. No training code, manifest, or launch file changes were made.

## Scope

This audit compares the current CurvyTron 18-run setup against the closest stock
LightZero/MuZero pattern I could find locally or in source. The user-mentioned
`mu0light0pat` string was not found as an exact local identifier.

Sources checked:

- Current active 18-run manifest:
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18fresh-allv2-20260516a/curvy-r18fresh-allv2-20260516a.json`
- Current manifest/code paths:
  - `scripts/build_curvytron_tonight18_manifest.py`
  - `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  - `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
  - `src/curvyzero/contracts/curvytron.py`
- Local working docs under
  `docs/working/training/leaderboard_to_training_2026-05-13/`
- LightZero source:
  - `/tmp/lightzero-src` was present but did not contain a usable source checkout.
  - Fresh audit checkout:
    `/tmp/lightzero-src-audit-20260516`, commit
    `de74055298068f53b70e07bc38c41101fce51766`, version `0.2.0`.

## Closest Baseline

The closest original LightZero baseline is `zoo.atari.config.atari_muzero_config`.
That is the actual template imported and deep-copied by our trainer before patching.
It is also a visual, single-policy, fixed-action-space MuZero setup with
`env_type='not_board_games'`.

The board-game configs, especially TicTacToe, are a useful conceptual contrast for
true two-player MuZero semantics, but they are not the current CurvyTron template.
Those configs use `env_type='board_games'`, varied action spaces, larger terminal
TD horizons for small games, and explicit self-play/bot battle modes.

## Current 18-Run Setup

The active manifest is the scratch all-v2 18-run set:

- Manifest id: `curvy-r18fresh-allv2-20260516a`
- Rows: 18
- Initialization: scratch random policy, no `initial_policy_checkpoint_ref`
- Rewards:
  - `sparse_outcome`
  - `survival_plus_bonus_no_outcome`
  - `survival_plus_bonus_plus_outcome`
- Opponent recipes:
  - `blank10-wall10-rank2_25-rank1_55`
  - `blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5`
  - `blank20-wall5-rank1_70-rank1imm5`
- Noise lanes:
  - 9 clean rows
  - 9 stochastic rows with straight override probability `0.1`, action repeat
    `1..2`, and extra repeat probability `0.1`
- Fixed knobs:
  - `collector_env_num=256`
  - `n_episode=256`
  - `num_simulations=8`
  - `batch_size=32`
  - `source_max_steps=1048576`
  - `max_train_iter=300000`
  - `max_env_step=30000000`
  - `save_ckpt_after_iter=10000`
  - `learner_seat_mode=random_per_episode`
  - `lightzero_eval_freq=0`, which the trainer converts to effectively no stock
    periodic eval before the last iteration
  - background eval enabled outside the stock LightZero evaluator

## High-Signal Deltas

### 1. Replay Ratio Is Stock, But Batch Size And Collection Chunks Are Not

Stock Atari MuZero uses `update_per_collect=None`, `replay_ratio=0.25`,
`collector_env_num=8`, `n_episode=8`, and `batch_size=256`.

Current CurvyTron inherits the same implicit update rule:
`updates = int(collected_transitions * replay_ratio)`, minimum `1`, because
`update_per_collect` remains unset and `replay_ratio` remains stock.

The major delta is that CurvyTron collects 256 episodes per collect phase and
trains with `batch_size=32`. Relative to the Atari template, equal episode lengths
would produce 32x more updates per collect, but each update uses an 8x smaller
batch. The result is a very different collect/train cadence and 8x fewer sampled
slots per newly collected transition under the same `replay_ratio`. With the
inherited Atari SGD learning rate and small batch, this is a plausible source of
noisy or unstable value/policy learning.

Severity for weak survival: high.

### 2. Collector Scale Is 32x The Atari Baseline

Stock Atari collects `8` episodes from `8` collector envs per collection.
Current CurvyTron collects `256` episodes from `256` collector envs per collection.

This creates very large policy-staleness chunks: a whole 256-env wave is generated
before the next learner phase. It also means opponent assignments and scratch
placeholder behavior can dominate large replay slices at once. This is a large
conceptual departure from the closest LightZero baseline even though it remains
legal under the collector assertion that `n_episode >= env_num`.

Severity for weak survival: medium-high.

### 3. MCTS Simulations Are Much Lower Than Stock

Stock Atari MuZero uses `num_simulations=50`. The board-game TicTacToe example
uses `25`. Current CurvyTron uses `8`.

CurvyTron has only three ego actions, so a lower count is not automatically wrong,
but survival depends on short-horizon collision avoidance and opponent interaction.
With only `8` simulations and stock Dirichlet noise at the root, the visit-count
targets can be very noisy. This is especially concerning for the stochastic lanes,
where the executed action can differ from the selected action.

Severity for weak survival: high.

### 4. Dense Survival Supports Saturate Far Below The Episode Cap

Stock Atari uses the default scalar support range `(-300, 301, 1)`, i.e. support
size `601`, for Atari-style reward scales.

Current CurvyTron patches supports by reward variant:

- `sparse_outcome`: requested/effective support scale `1`, support size `3`,
  range `(-1, 2, 1)`.
- `survival_plus_bonus_no_outcome`: requested reward scale `2`, requested value
  scale `2097152`. The reward scale is not individually over the cap, but the
  value scale is capped and the shared model support scale becomes `300`, so both
  reward and value heads use support size `601`.
- `survival_plus_bonus_plus_outcome`: requested reward scale `1048578`,
  requested value scale `3145728`. Both requested scales exceed the cap, so the
  shared model support scale is `300` and support size is `601`.

For dense survival rewards, the environment cap is `1048576` source steps and the
reward includes alive-per-step terms, bonus terms, and for
`survival_plus_bonus_plus_outcome`, a terminal outcome scaled by the physical step
index. The model support cap of `300` means many materially different survival
lengths collapse to saturated value targets. Once a policy can survive past a few
hundred rewarded steps, the value head may have little representational room to
distinguish "survives a bit" from "survives a long time".

Severity for weak survival: very high.

### 5. Horizon Settings Remain Atari-Like

Stock Atari config:

- `game_segment_length=400`
- `td_steps=5`
- `num_unroll_steps=5`
- `discount_factor=0.997`

Current CurvyTron fixed-opponent configs intentionally leave `td_steps` and
`num_unroll_steps` at the template defaults. The reward patch sets
`discount_factor=1.0`, but does not lengthen TD backup for sparse terminal
survival.

For `sparse_outcome`, a five-step TD target is a weak carrier for long-delayed
win/loss survival signal. For dense variants, short TD is less problematic, but
the support cap above can erase long-survival distinctions. The board-game
TicTacToe config shows LightZero's own small-game pattern for terminal outcomes:
it sets `td_steps=9` with a comment that the larger horizon ensures the value
target reaches final outcome.

Severity for weak survival: high for sparse outcome, medium for dense variants.

### 6. CurvyTron Is Two-Player, But The Training Lane Is Atari-Style Single-Agent

Current CurvyTron uses:

- `env_type='not_board_games'`
- `action_type='fixed_action_space'`
- `to_play=-1`
- one learner action per step
- opponent action chosen internally by the environment

This matches the Atari template mechanically, but it is conceptually different
from true two-player MuZero. The current fixed-opponent lane trains an ego value
function against an internally sampled opponent population. It does not alternate
players in the tree, does not use board-game `to_play` semantics, and does not
condition replay targets on a two-seat self-play identity except through the
observation.

The recent `learner_seat_mode=random_per_episode` work removes the earlier
seat-0-only failure mode, because observations are now controlled-player
perspective. That is good. It still remains a single-agent fixed-opponent
formulation, not the closest LightZero two-player baseline pattern.

Severity for weak survival: medium-high.

### 7. Exploration Is Stock, But The Action Space Is Much Smaller

The LightZero collect path uses visit-count temperature, root Dirichlet noise, and
optional epsilon-greedy. In stock config:

- manual temperature decay is off
- fixed temperature value is `0.25`
- root Dirichlet alpha is `0.3`
- root noise weight is `0.25`
- epsilon-greedy is disabled, so epsilon is `0` in the train loop

Current CurvyTron appears to inherit these values. With only three actions and
`num_simulations=8`, the stock root noise can be large relative to the amount of
search. This can make policy targets noisy even in clean lanes.

Severity for weak survival: medium-high.

### 8. Stochastic Lanes Introduce Hidden Action Mismatch

In stochastic rows, CurvyTron can override the requested action to straight and
can repeat an action for multiple source steps. The LightZero collector stores the
policy-selected action returned by MCTS. It does not store
`info["executed_ego_action"]`.

Therefore, in those rows, replay can associate an observation with the requested
action while the transition was produced by an overridden or repeated executed
action. This is valid as environment stochasticity only if the model is expected
to learn that noise process, but it is a direct model-target mismatch compared
with stock deterministic Atari action application.

Severity for weak survival: high in stochastic rows, not applicable to clean rows.

### 9. Stock Eval Is Mostly Disabled

Stock Atari has:

- `eval_freq=2000`
- `evaluator_env_num=3`
- `n_evaluator_episode=3`
- an initial eval before the training loop

Current CurvyTron manifest sets `lightzero_eval_freq=0`, and the trainer converts
that to `max_train_iter + 1`. With `max_train_iter=300000`, stock periodic eval is
effectively disabled during training. Current runs rely on the external background
eval/poller/tournament path instead.

This should not directly weaken the learner unless eval artifacts are used for
selection or refresh elsewhere, but it does reduce the stock loop's immediate
feedback and makes weak-survival diagnosis more dependent on our external eval
plumbing.

Severity for weak survival: low direct, medium diagnostic.

### 10. Replay Buffer Settings Are Mostly Stock Despite Nonstationary Opponents

Stock Atari uses:

- `replay_buffer_size=1000000`
- `use_priority=False`
- `reanalyze_ratio=0`
- `random_collect_episode_num=0`

Current CurvyTron appears to inherit those values. That means a large FIFO replay
buffer, no prioritization, and no reanalysis of older MCTS targets. Unlike Atari,
the effective environment includes a changing opponent assignment system and
scratch placeholder refreshes. Replay can therefore mix old targets from weaker
opponent populations with newer targets from refreshed assignments, without
reanalyzing them under the current network.

Severity for weak survival: medium.

### 11. Action Space Includes Straight As A Real Learner Action

Current fixed-opponent CurvyTron exposes three ego actions:

- `0`: left
- `1`: straight
- `2`: right

Straight is effectively the no-turn/no-op action. The opponent recipes also
include blank-canvas no-op opponents and wall-avoidant placeholder opponents.
That can make "go straight" locally attractive in early scratch learning, while
long-term survival often requires timely turning. In stochastic rows, straight is
also the override action, further increasing its behavioral footprint.

This is not necessarily wrong for CurvyTron physics, but it is a meaningful delta
from a carefully tuned survival-control action design. It should be audited
against the intended game contract: if historical CurvyTron agents only chose
left/right while straight meant absence of input, exposing straight as an equally
sampled MuZero action may encourage no-op collapse.

Severity for weak survival: medium.

### 12. Player Perspective Is Improved, But Still Not True Self-Play

Current docs and code indicate the policy observation is controlled-player
perspective, and `learner_seat_mode=random_per_episode` chooses player 0 or player
1 per episode. That addresses the most obvious player-perspective bug: the learner
is no longer always trained as physical player 0.

However, LightZero still receives `to_play=-1`, a single ego reward, and a
single-agent observation/action/reward stream. This is a fixed-opponent training
contract, not the board-game self-play contract where value signs and legal action
spaces are tied to alternating players.

Severity for weak survival: medium.

## Delta Matrix

| Area | Stock LightZero Atari Pattern | Current CurvyTron 18-Run Pattern | Survival Risk |
| --- | --- | --- | --- |
| Template | `zoo.atari.config.atari_muzero_config` | Same template, patched | Low |
| `update_per_collect` | `None` | Inherited `None` | Low by itself |
| `replay_ratio` | `0.25` | Inherited `0.25` | Medium with small batch |
| Batch size | `256` | `32` | High |
| Collector envs | `8` | `256` | Medium-high |
| Episodes per collect | `8` | `256` | Medium-high |
| MCTS simulations | `50` | `8` | High |
| Eval | Every `2000` iters, 3 envs/episodes | Stock periodic eval effectively off; external eval used | Diagnostic risk |
| Value support | Atari default `[-300, 300]` | Sparse `[-1, 1]`; dense capped to `[-300, 300]` despite huge horizon | Very high |
| Reward support | Atari default `[-300, 300]` | Sparse `[-1, 1]`; dense plus-outcome requested huge but capped to `[-300, 300]` | High |
| `td_steps` | `5` | Inherited `5` | High for sparse |
| `num_unroll_steps` | `5` | Inherited `5` | Medium |
| `game_segment_length` | `400` | Inherited unless patched elsewhere | Medium |
| Discount | `0.997` | Patched to `1.0` | Probably appropriate |
| Env type | `not_board_games` | `not_board_games` | Conceptual two-player mismatch |
| Action type | `fixed_action_space` | `fixed_action_space` | Low by itself |
| `to_play` | `-1` | `-1` | Medium-high for two-player semantics |
| Exploration | Temp `0.25`, Dirichlet alpha `0.3`, weight `0.25`, epsilon `0` | Inherited, but with only 8 sims and 3 actions | Medium-high |
| Replay buffer | 1M, no priority, no reanalyze | Inherited under nonstationary opponents | Medium |
| Action space | Atari game actions | Curvy left/straight/right | Medium |
| Player perspective | Single-player Atari | Controlled-player perspective, random physical seat | Improved but still fixed-opponent |

## Most Plausible Explanations For Weak Survival

1. Dense value support saturation is the strongest candidate. The dense survival
   variants can generate returns far above `300`, while the model support caps all
   such outcomes into the stock Atari range.

2. The search target is weak: `8` simulations with stock root noise is a very
   small search budget for learning collision avoidance from scratch.

3. Sparse outcome has too little temporal credit assignment. With `td_steps=5`,
   scratch sparse-outcome runs may get almost no useful learning signal until
   value bootstrapping is already competent.

4. The update regime is not the Atari baseline despite inheriting the same replay
   ratio. The combination of `collector_env_num=256`, `n_episode=256`,
   `batch_size=32`, and stock optimizer settings is a major training-dynamics
   change.

5. Stochastic rows likely corrupt the learned dynamics/action model unless the
   hidden override/repeat process is intentionally treated as environment
   stochasticity. The replay action is the requested action, not necessarily the
   executed action.

6. The training contract is single-agent fixed-opponent, while the domain is
   two-player. Randomized controlled-player perspective helps, but the run is not
   using LightZero's board-game/self-play semantics.

## Follow-Up Checks Worth Doing Before Another Large Run

- Run a tiny ablation with dense support uncapped or scaled/compressed explicitly
  for the CurvyTron reward contract.
- Compare `num_simulations=8` against `25` or `50` on a short fixed-opponent
  survival smoke run.
- Restore a closer Atari update shape for one lane: `batch_size=256` and smaller
  `collector_env_num/n_episode`, or explicitly set `update_per_collect` to control
  learner work per collect.
- For sparse outcome, test a longer `td_steps` or a reward curriculum that avoids
  pure terminal credit assignment from scratch.
- Keep one clean lane with no straight override/repeat until baseline survival is
  established.
- Decide whether straight should remain a first-class MuZero action or whether
  the intended control contract is left/right with straight as absence of input.
