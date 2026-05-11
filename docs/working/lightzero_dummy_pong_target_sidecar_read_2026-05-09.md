# LightZero Dummy Pong Target Sidecar Read

## Scope

This is a read of the completed target replay telemetry smoke, not a training
claim. The useful artifacts are:

- `target_replay_steps.jsonl`: per collected transition target rows mirrored
  from LightZero `GameSegment`.
- `target_replay_summary.json`: compact row/episode/action summary over the
  same sidecar.

Completed smoke:

- App run: `ap-rdvkRpLGRYedx39SggsVvm`
- Run id: `lz-dpong-20260509T184118Z-bf893dffebfd`
- Attempt id: `attempt-20260509T184118Z-933639b09f7c`
- Status: `completed`

## What We Can Read Now

The sidecar is already enough to inspect the actual policy target rows that
LightZero collected during the tiny custom Pong smoke.

From `target_replay_summary.json`:

```json
{
  "rows": 16,
  "episodes": 1,
  "collect_calls": 1,
  "terminal_steps": 1,
  "action_counts": {
    "up": 3,
    "stay": 4,
    "down": 9
  },
  "has_child_visit_segment": true,
  "has_reward": true
}
```

From `target_replay_steps.jsonl`, the child-visit target distribution counts
were:

```json
{
  "[0.5, 0.5, 0.0]": 4,
  "[0.5, 0.0, 0.5]": 9,
  "[0.0, 0.5, 0.5]": 3
}
```

Reward counts were:

```json
{
  "0.0": 15,
  "1.0": 1
}
```

The first rows show that the sidecar separates executed action from target
policy mass:

```jsonl
{"global_step_index": 0, "action_segment": 0, "action_label": "up", "child_visit_segment": [0.5, 0.5, 0.0], "reward": 0.0, "done": false}
{"global_step_index": 1, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 0.0, "done": false}
{"global_step_index": 2, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 0.0, "done": false}
{"global_step_index": 3, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.0, 0.5, 0.5], "reward": 0.0, "done": false}
```

The terminal row was:

```jsonl
{"global_step_index": 15, "action_segment": 2, "action_label": "down", "child_visit_segment": [0.5, 0.0, 0.5], "reward": 1.0, "done": true}
```

## Config Read

The completed smoke used:

```sh
--mode train
--pong-reset-profile contact_pressure
--max-env-step 16
--pong-episode-max-steps 16
--max-train-iter 1
--collector-env-num 1
--evaluator-env-num 1
--n-evaluator-episode 1
--num-simulations 2
--batch-size 2
--update-per-collect 1
--n-episode 1
--game-segment-length 16
--random-collect-episode-num 1
```

The row-level `target_config` schema also records:

- Algorithm: `LightZero MuZero`
- Env: default `dummy_pong_lag1`
- Feature mode: default `tabular_ego`
- Reset profile: `contact_pressure`
- Episode cap: `16`
- MCTS sims: `2`
- Opponent policy: default `random_uniform`
- Ego agent: default `player_0`
- Support ranges: requested reward/value support fields are `null` unless
  explicitly supplied, with patched support metadata copied into the snapshot.

## Interpretation

The important unlock is that we can now read policy target mass directly from
`child_visit_segment` instead of inferring it from executed actions.

For this tiny smoke:

- Executed action was `down` on 9/16 rows.
- Target mass for `down` was nonzero on 12/16 rows:
  9 rows with `[0.5, 0.0, 0.5]` and 3 rows with `[0.0, 0.5, 0.5]`.
- Target mass for `down` was zero on 4/16 rows:
  4 rows with `[0.5, 0.5, 0.0]`.
- The only positive reward row executed `down` and assigned `down` target mass
  `0.5`.

That is a support-lane sanity check, not evidence that the learned policy is
good. The run is one short episode with low simulation count.

## Current Gap

The current sidecar does not by itself identify the oracle-winning action for
each row. It has:

- Executed action: `action_segment` / `action_label`
- MCTS policy target: `child_visit_segment`
- Reward/root value/done
- Segment and collect indices
- Config snapshot

It does not have enough state/oracle metadata to answer:

> Did the oracle-winning action get target mass on this specific transition?

So the next meaningful custom Pong run needs either row-level oracle fields or
a deterministic join from row to an oracle replay.

## Exact Next Query

For each target replay row in the next meaningful custom Pong run, add or join:

- `oracle_winning_action_id`
- `oracle_winning_action_label`
- `oracle_margin` or a simple `oracle_outcomes_by_action`
- enough state identity to audit the join, such as reset seed/profile,
  episode index, step index, and compact observation/state hash

Then compute exactly:

```text
oracle_target_mass = child_visit_segment[oracle_winning_action_id]
oracle_got_any_mass = oracle_target_mass > 0
oracle_is_top_target = oracle_target_mass == max(child_visit_segment)
executed_is_oracle = action_segment == oracle_winning_action_id
executed_target_mass = child_visit_segment[action_segment]
```

Report these tables:

1. Overall oracle target support
   - rows
   - mean/median `oracle_target_mass`
   - count and percent with `oracle_got_any_mass`
   - count and percent with `oracle_is_top_target`

2. Oracle support by executed action
   - rows grouped by `action_label`
   - `executed_is_oracle` rate
   - mean `oracle_target_mass`
   - zero-oracle-mass rate

3. Oracle support by oracle action
   - rows grouped by `oracle_winning_action_label`
   - mean target mass for that oracle action
   - zero-mass rate
   - top-target rate

4. Disagreement matrix
   - rows grouped by `(oracle_winning_action_label, action_label)`
   - mean `oracle_target_mass`
   - count with `oracle_target_mass == 0`

5. Reward overlay
   - same metrics split by `reward > 0`, `reward == 0`, and terminal rows

The key decision read is:

```text
When the oracle-winning action differs from the executed action, does the
target still place mass on the oracle action, or does collect/MCTS erase it?
```

If the oracle action often gets target mass even when execution differs, the
training target is less collapsed than action histograms imply. If the oracle
action often gets zero mass, the target policy itself is starving the useful
move and the next fix should focus on MCTS/config/curriculum rather than only
collector exploration.
