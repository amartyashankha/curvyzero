# 2026-05-08 Source Kinematics Scenarios

## Question

What tiny scenario files should come after the first forced two-player source
kinematics match?

## Added Fixtures

- `source_kinematics_straight_step.json`: proves source move `0` keeps heading
  fixed and advances position by source speed for one `1000 / 60` ms step.
- `source_kinematics_left_turn_step.json`: proves source move `-1` applies the
  source angular velocity before movement. Player `p1` stays straight as a
  control.
- `source_kinematics_right_turn_step.json`: proves source move `1` applies the
  source angular velocity before movement. Player `p1` stays straight as a
  control.

All three fixtures are forced-state only. They do not assert collisions, trails,
bonuses, scores, deaths, winners, or round state beyond the small setup needed
for the reference runner.

## Multi-Step Note

The JS reference scenario runner already loops over all `steps`. The Python
`source-kinematics` CLI path now accepts the current `source_kinematics_*`
fixtures plus `forced_two_player_turn_step`, but it is still movement-only. A
two-step constant-turn fixture can be added next, still without collision,
trail, bonus, score, or round expectations.

## Validation

JSON parse validation passed with:

```sh
node -e "const fs=require('fs'); for (const f of process.argv.slice(1)) JSON.parse(fs.readFileSync(f,'utf8'))" scenarios/environment/source_kinematics_straight_step.json scenarios/environment/source_kinematics_left_turn_step.json scenarios/environment/source_kinematics_right_turn_step.json
```

The Python `source-kinematics` runner now accepts each fixture:

- `scenarios/environment/source_kinematics_straight_step.json`
- `scenarios/environment/source_kinematics_left_turn_step.json`
- `scenarios/environment/source_kinematics_right_turn_step.json`

The local batch manifest is:

- `scenarios/environment/source_kinematics_batch.json`
