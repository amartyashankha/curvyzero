# 2026-05-09 Vector Debug Obs/Reward Packing

## Question

Can the current fixture-seeded vector rows pack fixed observation, reward, mask,
and ego-id arrays quickly enough to be useful for the next policy/search bridge?

This is a debug packing benchmark. It is not the final CurvyTron training
observation or reward schema.

## Script

New script:

```sh
python3 scripts/benchmark_vector_obs_reward_packing.py \
  --batch-sizes 1 32 128 \
  --repeat 10000 \
  --warmup 500 \
  --body-capacity 4 \
  --format plain
```

What it does:

- Seeds the current supported fixture rows.
- Runs the vector comparator once per fixture before timing. State and compact
  event rows must pass for the supported slice.
- Groups rows by fixed array profile, currently `P=2,K=4` and `P=3,K=4`.
- Runs one B>1 array step per group and batch size during setup so rewards can
  compare previous and next state.
- Times only the packing loop after that setup step.
- Packs `obs[B,P,9]` as `float32` with these debug features:
  `x_over_map_size`, `y_over_map_size`, `heading_sin`, `heading_cos`, `alive`,
  `printing`, `score`, `round_score`, and `map_size_over_1000`.
- Packs `reward[B,P]` as `float32`.
- Packs `done[B]`, `truncated[B]`, `terminated_agent[B,P]`,
  `truncated_agent[B,P]`, `legal_action_mask[B,P,3]`, `ego_row_id[B,P]`,
  `ego_env_id[B,P]`, `ego_player_id[B,P]`, and `ego_mask[B,P]`.

Reward status:

- The debug formula is `score_delta + round_score_delta - died_this_step`.
- `died_this_step` uses existing compact `die` event rows when they are present.
- If a caller provides state without event rows, the script falls back to
  `previous_alive & ~alive`.
- The score and round-score terms still come from state deltas.
- This is placeholder/narrow reward plumbing, not a training reward contract.

Mask status:

- `legal_action_mask` marks left, straight, and right as legal for active live
  ego rows.
- `done` is `state.done OR alive_count <= 1`.
- `truncated` is derived from overflow flags, including event overflow when
  present.
- There is no autoreset or horizon truncation policy here.

## Result

Local runtime labels from the script environment: macOS arm64, Python 3.11.14,
NumPy 2.4.0.

Preflight passed: 8 fixtures passed, 0 failed, 0 unsupported. Batch-state
preflight passed for both fixed-shape groups.

| Group | B | Obs shape | Pack bucket | Env rows/sec | Ego rows/sec | Live ego | Done rows | Reward die source |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `P=2,K=4` | 1 | `[1, 2, 9]` | 0.186438s | 53,637.2 | 107,274.3 | 2 | 0 | event rows |
| `P=2,K=4` | 32 | `[32, 2, 9]` | 0.238957s | 1,339,151.9 | 2,678,303.8 | 32 | 16 | event rows |
| `P=2,K=4` | 128 | `[128, 2, 9]` | 0.324502s | 3,944,502.2 | 7,889,004.4 | 128 | 64 | event rows |
| `P=3,K=4` | 1 | `[1, 3, 9]` | 0.181446s | 55,112.7 | 165,338.1 | 3 | 0 | event rows |
| `P=3,K=4` | 32 | `[32, 3, 9]` | 0.247807s | 1,291,327.7 | 3,873,983.2 | 80 | 0 | event rows |
| `P=3,K=4` | 128 | `[128, 3, 9]` | 0.339334s | 3,772,091.0 | 11,316,273.1 | 320 | 0 | event rows |

The useful fact: fixed obs/reward/mask/ego-id arrays now exist on the current
vector shapes, and packing is cheap relative to the current narrow env step.
This keeps the speed lane moving toward the policy/search boundary.

## What This Does Not Prove

- It does not define the final training observation. The current observation is
  privileged debug state, not an ego-relative ray, raster, or learned schema.
- It does not define the final reward. The death penalty can read event rows,
  but score and round-score terms are still simple state deltas.
- It does not prove reset/autoreset, horizon truncation, replay chunks, policy
  batching, MCTS/search, or wrapper dict conversion.
- It does not broaden source fidelity beyond the currently supported vector
  fixture slice.
- It does not remove the need for PrintManager/trail-gap vector semantics and
  broader event-row coverage.
- It does not claim GPU or production self-play throughput.
