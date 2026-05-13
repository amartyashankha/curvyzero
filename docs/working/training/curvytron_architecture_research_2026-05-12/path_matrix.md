# CurvyTron LightZero Path Matrix

> Historical warning: this path comparison is still useful, but its
> "recent frozen opponent" optimism predates the v1d readout. Current guidance:
> keep stock `train_muzero`, avoid the old custom two-seat learning lane, and
> use [current_source_of_truth.md](current_source_of_truth.md) for the active
> survival-first opponent plan.

Purpose: prevent different paths from being collapsed into one vague
"self-play" label.

## Summary

| Path | Calls stock `train_muzero` | Uses stock collector/GameBuffer | What one env step means | What it proves |
| --- | --- | --- | --- | --- |
| `source_state_fixed_opponent` | yes | yes | one ego action, env supplies opponent action | stock visual CurvyTron can learn against an env-owned opponent |
| frozen/checkpoint opponent control | yes | yes in `source_state_fixed_opponent` canaries | one ego action, opponent comes from frozen policy | practical stock-loop training against named recent opponents, not exact live self-play |
| `source_state_turn_commit` | profile yes, train blocked | profile yes | fake pending action then later physical commit | stock plumbing only; current train form has bad replay semantics |
| `source_state_joint_action` | yes | yes | one scalar action decodes to both players' actions, then one real tick | stock loop can learn real joint physics under centralized control |
| `--mode two-seat-selfplay` | no | no | custom collector picks both seat actions, then one real tick | simultaneous current-policy collection prototype, not a trusted trainer |

## Key Distinction

The hard part is not visual input. It is replay semantics.

CurvyTron physical reality:

```text
one tick -> player 0 action + player 1 action -> per-player outcome
```

Stock LightZero row:

```text
one env step -> one scalar action -> one reward -> one next observation
```

The project moved to custom two-seat code to preserve the first shape. That was
reasonable as a collector experiment. It became risky when we also replaced the
stock replay and target path.

## Notes From Current Inspection

- `source_state_fixed_opponent` is not live self-play, but it is the cleanest
  stock loop proof.
- A recent frozen checkpoint opponent may be a valid practical training route
  if it stays on stock `train_muzero` and is labeled as frozen/recent-opponent
  training, not live same-policy self-play.
- Tiny CPU and L4 GPU canaries now prove the stock frozen-checkpoint opponent
  route can call `train_muzero` and strictly load a real checkpoint. They do
  not prove learning.
- `source_state_joint_action` keeps one physical tick per replay row. It is not
  competitive self-play because one centralized policy controls both players.
- `source_state_turn_commit` is blocked for train because fake pending rows
  enter replay and can receive value credit from later physical commit rows.
- `two-seat-selfplay` should be labeled experimental until it feeds native
  `GameSegment` / `MuZeroGameBuffer` targets or has parity-tested target logic.
