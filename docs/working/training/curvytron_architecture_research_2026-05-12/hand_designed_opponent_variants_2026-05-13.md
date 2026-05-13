# Hand-Designed CurvyTron Opponent Variants

Purpose: define a small menu of legal-action opponent policies for a future
training matrix without touching trainer/source-state plumbing.

These variants use only left/straight/right action ids. They do not teleport,
clamp, flip heading, or implement real bounce/reflection. Runtime integration
should be a simple opponent policy kind or wrapper-owned runtime mode that can
read source-state geometry and return standard action sidecars.

## Existing Probe Baseline Inspected

The current baseline note
[`scripted_wall_avoidant_opponent_baseline_2026-05-13.md`](scripted_wall_avoidant_opponent_baseline_2026-05-13.md)
already establishes the best first wall policy:

- `proactive_force_field`, `M=20`: go straight outside the danger band; near a
  wall, turn left/right toward the arena-interior danger-field vector.
- Earlier local probes showed this survived 384 1024-step starts with normal
  trail writing and no out-of-bounds rows.
- Contact-only and pure reflected-heading target policies failed. Inward-biased
  reflection and short rollout can be made survivable, but they are less clean
  and more turn-heavy.

I extended the bounded local probe script rather than changing env/trainer
code:

- `scripts/probe_curvytron_wall_avoidant_opponent.py`
- new option: `--suite --suite-kind hand_designed`

## Candidate Policy Kinds

| Policy kind | Expected behavior | Why it is useful | Main risk |
| --- | --- | --- | --- |
| `proactive_force_field` | Mostly straight, legal turns only inside wall danger band. | Stable baseline; easy to interpret; already strongest evidence. | Deterministic and only wall-aware, so trail layouts may be too simple. |
| `lazy_weave` | Periodic short left/right bursts while safe, force-field override near walls. | Gentle S-curves; still mostly straight; predictable and likely easy to beat. | Fixed cadence can create repeated layouts or self-intersections. |
| `jitter_force_field` | Sparse deterministic pseudo-random one-step turns while safe, force-field override near walls. | Cheap variety axis via `behavior_seed`; produces less regular trails. | Needs repeated behavior seeds in any matrix; one seed is not representative. |
| `wall_follower` | Drifts into a loose perimeter lane and follows wall tangents with inward correction. | Creates boundary pressure and long side trails without actual bounce. | More turn-heavy; may overteach center camping or create dense edge clutter. |
| `waypoint_patrol` | Chases four interior waypoints in row-specific direction, with wall override. | Produces crossing diagonal/box-like trails in the learner's workspace. | Turn-heavy in the probe; may be less "easy opponent" and more trail stressor. |

Rejected or lower-priority existing probe kinds:

- `reactive_reflection_proxy`: waits too late.
- `margin_reflection` / `predictive_reflection`: legal-action reflection targets
  are either unreliable at low margin or too turn-heavy at high margin.
- `rollout_clearance`: useful diagnostic/stress policy, but it is more like a
  tiny planner than a simple hand-designed opponent.

## Bounded Probe Results

Command shape:

```bash
PYTHONPATH=src python3 scripts/probe_curvytron_wall_avoidant_opponent.py \
  --suite --suite-kind hand_designed --batch-size 32 --steps 256 \
  --safe-margin 20 --seed <0-or-1> --behavior-seed <0-or-11>
```

Setup:

- real `VectorMultiplayerEnv`;
- opponent player id `1`;
- ego fixed straight;
- `death_immunity_player_ids=(0, 1)` for geometry measurement only;
- natural bonuses disabled;
- normal body/trail insertion left enabled.

Combined readout across two bounded runs: 64 starts, 256 steps each.

| Policy kind | OOB rows | Death rows | Min clearance | Action mix | Read |
| --- | ---: | ---: | ---: | --- | --- |
| `proactive_force_field` | 0/64 | 0 opp / 0 ego | 8.583 | L 0.136 / S 0.724 / R 0.140 | Keep as baseline row. |
| `lazy_weave` | 0/64 | 0 opp / 0 ego | 8.653 | L 0.201 / S 0.603 / R 0.197 | Strong first variant row. |
| `wall_follower` | 0/64 | 0 opp / 0 ego | 8.891 | L 0.362 / S 0.283 / R 0.356 | Useful stress/trail-density row. |
| `waypoint_patrol` | 0/64 | 0 opp / 0 ego | 8.625 | L 0.441 / S 0.110 / R 0.449 | Probe-only or late stress row until tuned. |
| `jitter_force_field` | 0/64 | 0 opp / 0 ego | 8.653 | L 0.170 / S 0.654 / R 0.176 | Strong stochastic/seeded variant row. |

The per-policy 32x256 runs scanned about 67M body slots and kept
`body_write_cursor` around 535, so this was still a real trail-writing probe.
Do not scale this script into large sweeps; it is only a local geometry/design
check.

## Matrix Recommendation

First matrix rows once scripted opponent plumbing exists:

1. `proactive_force_field`, `safe_margin=20`, one or two reset seeds.
2. `lazy_weave`, `safe_margin=20`, one or two behavior seeds.
3. `jitter_force_field`, `safe_margin=20`, at least five behavior seeds if it
   becomes an important stochastic row.

Add after canaries:

4. `wall_follower`, `safe_margin=20`, as a trail-density/boundary-pressure row.

Hold back initially:

5. `waypoint_patrol`, `safe_margin=20`, unless we want an intentionally more
   turn-heavy stressor. It survived the bounded probe, but its low straight
   fraction makes it less clean as an "easy enough" opponent.
