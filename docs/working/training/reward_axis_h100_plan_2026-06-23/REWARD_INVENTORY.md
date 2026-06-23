# Reward Inventory

Status: active inventory. The focus is the current source-state visual
LightZero path plus compact-owned and eval/tournament distinctions.

## Current Source-State Visual LightZero Rewards

These variants are declared in `src/curvyzero/contracts/curvytron.py` and
contracted in `src/curvyzero/training/reward_contracts.py`.

| Variant | Where used | Formula shape | Current read |
| --- | --- | --- | --- |
| `sparse_outcome` | Fixed-opponent source-state visual training, auto default for fixed-opponent if unspecified | Nonterminal `0`; terminal survivor `+1`, loser `-1`, draw/truncation `0` | Clean game-outcome baseline. Strong mid-run lift in r18fresh, weak latest retention under current TD/search/cadence. |
| `dense_survival_plus_outcome` | Supported fixed-opponent variant, not part of the main tonight18 matrix | Post-transition alive helper `+1` plus sparse terminal outcome | Useful conceptual midpoint, but not a current main launch arm. |
| `survival_plus_bonus_no_outcome` | Fixed-opponent source-state visual training, compact default | Alive helper `+1` plus same-step bonus pickup reward; sparse outcome is telemetry only | Clean survival objective. Good control arm, but r18fresh/postmortem reads do not make it the current strongest main arm. |
| `survival_plus_bonus_plus_outcome` | Fixed-opponent source-state visual training, current next-batch default direction | Alive helper `+1` plus bonus pickup plus terminal sparse outcome scaled by accumulated non-outcome return and `reward_outcome_alpha` | Current most promising main arm, but support saturation and terminal volatility must be controlled. |
| `all_players_alive_diagnostic` | Source-state joint-action diagnostic only | Centralized `+1` if all players are alive after the step, else `0` | Diagnostic/blocker-shaped. Not a production self-play reward. |

Implementation references:

- Reward policy/support bounds:
  `src/curvyzero/training/reward_contracts.py`
- Reward components in the source-state wrapper:
  `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- LightZero config patching:
  `src/curvyzero/training/lightzero_config_builder.py`
- Stock trainer launch facade:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`

## Compact-Owned Reward

Compact-owned training currently has an explicit no-RND, extrinsic-only
contract:

- Contract mode: `extrinsic_reward_no_rnd_v1`
- Default reward variant: `survival_plus_bonus_no_outcome`
- Exploration bonus: `none`
- RND: rejected for this path

Primary reference:

- `src/curvyzero/training/compact_reward_rnd_contract.py`

Interpretation: compact speed work should not be mixed into the RND/reward-axis
claims. If compact quality is tested, its reward claim is extrinsic-only
`survival_plus_bonus_no_outcome` unless the compact contract changes.

## RND Exploration Bonus

RND is not one of the extrinsic reward variants above. It is an exploration
bonus path that can train/log a novelty model and, in the positive mode, mutate
the replay target reward.

| Mode | Reward target effect | Current status |
| --- | --- | --- |
| `none` | No intrinsic reward | Stock baseline. |
| `rnd_meter_v0` | No target mutation; metrics only | Believable diagnostic/control. |
| `rnd_replay_target_v0` | Adds normalized intrinsic reward to replay targets | Implemented and worth an aggressive sweep, but not yet proven successful. |

Primary references:

- `src/curvyzero/training/exploration_bonus.py`
- `scripts/build_curvytron_rnd_blank_sweep_manifest.py`
- `docs/working/training/exploration_bonus_rnd_2026-05-19/`

Interpretation: RND should be evaluated as a separate intrinsic-reward lane
against stock and meter controls. A positive blank-canvas result promotes RND to
fixed-opponent testing; it does not automatically change the default extrinsic
reward contract.

## Eval And Tournament Scoring Are Separate

CurvyZero checkpoint eval primarily reports survival duration:

- `steps_survived`
- action histograms and action-collapse flags
- sparse outcome and reward components as telemetry

Tournament/rating uses head-to-head game outcomes:

- wins
- losses
- draws
- rating score from win/draw result
- exposure counts and distinct opponents

This means:

1. Raw trainer reward is comparable only within the same reward variant.
2. Cross-variant comparison should use survival curves, retention, action
   collapse, and tournament exposure.
3. `model_reward_variant` in eval/tournament is mostly model reconstruction
   metadata, not the scoring objective.

## Historical Or Non-Current Rewards

Older two-seat/custom lanes used shaped rewards such as tiny alive helpers,
bonus pickup helpers, and terminal outcome scaled by episode length. Those lanes
are historical unless their native replay/target contract is re-proven. Do not
use them as launch guidance for the stock LightZero reward-axis sweep.

## Current Reward Ranking

Working extrinsic reward ranking for the next experiments:

1. `survival_plus_bonus_plus_outcome`: main candidate, especially with alpha and
   support/cadence controls.
2. `survival_plus_bonus_no_outcome`: clean survival control and compact default.
3. `sparse_outcome`: diagnostic baseline for game outcome and long-horizon
   credit assignment.
4. `dense_survival_plus_outcome`: supported midpoint, lower priority unless we
   need to isolate bonus pickup.
5. `all_players_alive_diagnostic`: joint-action diagnostic only.

Separate intrinsic lane: `rnd_replay_target_v0` is now high priority for
testing, but it is not ranked against extrinsic reward variants until it has
stock and meter-controlled survival evidence.
