# 2026-05-09 LightZero dummy Pong contact-pressure state/action oracle

## Question

In scoreable custom dummy Pong contact-pressure states where the true sparse
oracle needs `down`, do LightZero policy logits and MCTS root visit targets put
mass on `down`, or is `down` lost through action mapping/eval selection?

Important target note: LightZero MuZero trains policy logits against MCTS root
visit distributions. The selected/evaluated/collected action can differ due to
tie-breaking, temperature, or epsilon, and is not itself the policy target.

## Commands

No training and no pytest.

```sh
uv run --extra modal python -m py_compile \
  src/curvyzero/infra/modal/lightzero_dummy_pong_contact_pressure_oracle.py
```

```sh
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_dummy_pong_contact_pressure_oracle \
  --state-seeds 20260510,20260515,20260523 \
  --num-simulations 2,8,16,25 \
  --eval-id state-action-oracle-contact-pressure-down-needed
```

```sh
uv run --extra modal modal volume get curvyzero-runs \
  training/lightzero-dummy-pong/lz-dpong-20260509T175407Z-77159cc3a6b4/attempts/attempt-20260509T175407Z-8105d62c1e00/eval/state-action-oracle-contact-pressure-down-needed/contact_pressure_state_action_oracle.json \
  artifacts/local/lightzero-dummy-pong-contact-pressure-oracle-2026-05-09/contact_pressure_state_action_oracle_down_needed.json
```

## Files

- Probe: `src/curvyzero/infra/modal/lightzero_dummy_pong_contact_pressure_oracle.py`
- Result JSON:
  `artifacts/local/lightzero-dummy-pong-contact-pressure-oracle-2026-05-09/contact_pressure_state_action_oracle_down_needed.json`
- Source scoreability rows:
  `artifacts/local/dummy-pong-contact-pressure-scoreability-probe-2026-05-09/rows.jsonl`

## States

All three sampled states are `player_0` pressure states versus
`lagged_track_ball_1`. In each, the true sparse rollout oracle gives
`up=-1`, `stay=0`, `down=+1`, so `down` is the unique score-winning action.

Feature order:
`ego_paddle_y, opponent_paddle_y, ego_paddle_x, opponent_paddle_x, ball_dx_forward, ball_dy_from_ego_center, ball_vx_forward, ball_vy, ball_y, step`.

| State | Tabular features | Direct env `down` first step | Sparse oracle |
| --- | --- | --- | --- |
| `player_0-seed-20260510` | `[0.6667, 0.3333, 0.0714, 0.9286, 0.2143, 0.1250, -1.0, 0.0, 0.75, 0.0]` | `player_0_y 4 -> 5`, joint `down/down` | `up=-1, stay=0, down=+1` |
| `player_0-seed-20260515` | `[0.6667, 0.3333, 0.0714, 0.9286, 0.1429, -0.1250, -1.0, 1.0, 0.50, 0.0]` | `player_0_y 4 -> 5`, joint `down/down` | `up=-1, stay=0, down=+1` |
| `player_0-seed-20260523` | `[0.8333, 0.6667, 0.0714, 0.9286, 0.2143, 0.0, -1.0, 0.0, 0.75, 0.0]` | `player_0_y 5 -> 6`, joint `down/down` | `up=-1, stay=0, down=+1` |

This rules out an env action-label mismatch in these examples: action id `2`
does move the ego paddle down.

## Policy Logits

Policy logits are `up/stay/down` from `model.initial_inference`.

| Checkpoint | Per-state logits read | Argmax | Read |
| --- | --- | --- | --- |
| `iteration_0` | about `[0.00366, 0.00649, -0.00649]` | `stay` | `down` already lowest, but only by about `0.013` |
| `iteration_3` | about `[-0.0143, 0.0524, -0.0301]` | `stay` | training made `stay` strongly top and pushed `down` lower, by about `0.081-0.083` vs best non-down |
| `ckpt_best` | `[0.0, 0.0, 0.0]` | tie, adapter argmax `up` | flat prior; any down preference comes from search noise/dynamics/value, not policy head |

## MCTS Root Visits

Counts below are LightZero `visit_count_distributions` in `up/stay/down` order.
The selected action is shown after `->`.
The MCTS adapter call used `action_mask = [[1.0, 1.0, 1.0]]`,
`to_play=[-1]`, and `ready_env_id=[0]`.

| Checkpoint | Sims | `20260510` | `20260515` | `20260523` |
| --- | ---: | --- | --- | --- |
| `iteration_0` | 2 | `[1,1,0] -> up` | `[1,1,0] -> up` | `[1,1,0] -> up` |
| `iteration_0` | 8 | `[3,3,2] -> up` | `[3,3,2] -> up` | `[3,3,2] -> up` |
| `iteration_0` | 16 | `[5,6,5] -> stay` | `[5,6,5] -> stay` | `[6,5,5] -> up` |
| `iteration_0` | 25 | `[8,9,8] -> stay` | `[8,9,8] -> stay` | `[9,8,8] -> up` |
| `iteration_3` | 2 | `[0,1,1] -> stay` | `[0,1,1] -> stay` | `[0,1,1] -> stay` |
| `iteration_3` | 8 | `[3,3,2] -> up` | `[3,3,2] -> up` | `[3,3,2] -> up` |
| `iteration_3` | 16 | `[5,6,5] -> stay` | `[5,6,5] -> stay` | `[5,6,5] -> stay` |
| `iteration_3` | 25 | `[8,9,8] -> stay` | `[8,9,8] -> stay` | `[8,9,8] -> stay` |
| `ckpt_best` | 2 | `[1,0,1] -> up` | `[0,1,1] -> stay` | `[0,1,1] -> stay` |
| `ckpt_best` | 8 | `[3,3,2] -> up` | `[3,2,3] -> up` | `[3,2,3] -> up` |
| `ckpt_best` | 16 | `[6,5,5] -> up` | `[5,5,6] -> down` | `[5,6,5] -> stay` |
| `ckpt_best` | 25 | `[8,8,9] -> down` | `[8,8,9] -> down` | `[8,9,8] -> stay` |

## Read

This is not an adapter/action-label bug. The direct env step confirms `down`
is action id `2` and moves the paddle down; the true sparse scoreability oracle
then wins with `down` in all sampled states.

The sharper issue is the MCTS target itself. At the actual modest-rung setting
of `num_simulations=2`, `iteration_0` assigns zero visits to `down` in all
three down-needed states. After training, `iteration_3` gives `down` one visit
at sims 2 but only as a tie with `stay`, while the selected action remains
`stay`; at larger simulation counts it consistently makes `stay` top. That is
exactly the wrong policy target for these states.

`ckpt_best` is different: its policy head is flat on these states, and higher
simulation counts can make `down` top in 1/3 states at sims 16 and 2/3 states
at sims 25. That means `down` is not impossible for the MCTS implementation,
but the learned policy/value/search combination is weak and unstable.

The earlier trainer-side executed-action histogram included `down`, but that is
not the policy target. Executed action can be affected by collection
temperature/epsilon/tie-breaking. This probe only treats root visits as the
training target and selected action as secondary telemetry.

## Likely Next Fix

Instrument collection artifacts to persist, per decision, both root
`visit_count_distributions` and selected/executed action separately. Then gate
contact-pressure training on root-target support for down-needed oracle states,
not on selected-action histograms.

For this particular rung, `num_simulations=2` is too weak as a target generator
for scoreable contact-pressure states. The next diagnostic should either raise
collection simulations for this curriculum slice or inject an oracle/replay
target for these down-needed states, then verify that the root visit target
puts `down` top before launching any longer training.
