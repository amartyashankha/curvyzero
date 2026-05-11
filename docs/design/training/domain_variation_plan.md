# Domain Variation Plan

Status: Proposed
Date: 2026-05-08

## Scope

This plan is for later training robustness. It is not part of the current
source-fidelity reconstruction lane.

The source-fidelity lane should continue to answer whether CurvyZero matches the
chosen CurvyTron reference behavior. Domain variation should only start after
the fixed ruleset is deterministic, learnable, and easy to replay.

## Rule

Keep three modes separate:

| Mode | Purpose | Randomize rules? |
| --- | --- | --- |
| `curvytron-v1-reference` | Source comparison and reconstruction. | No. |
| `curvyzero-v0` | Fixed first training and baseline learning. | No. |
| `curvyzero-robust-*` | Later robustness and curriculum. | Yes, by named profile. |

Changing a training profile must not change the canonical ruleset. It should
create or update a named variation profile.

## Start Gates

Do not enable domain variation until these are true:

- Source-fidelity work has its own trace/golden-test lane.
- `curvyzero-v0` has deterministic reset/step/replay under fixed seeds.
- Random-vs-random stress is stable and seat-balanced.
- A heuristic beats random on fixed and held-out seeds.
- A small learned baseline beats random on the fixed ruleset.
- Replays store rules, observation, reward, seed, and opponent metadata.

## First Implementation Shape

Add one `variation_profile` field to training config:

```text
variation_profile:
  name: curvyzero-robust-observe-v0
  sample_at: episode_reset
  physics: fixed
  observation:
    color_palette: random
    ray_noise_std: small
    raster_zoom: small
    raster_translation: small
```

Then add a second profile for mild physics variation:

```text
variation_profile:
  name: curvyzero-robust-physics-v0
  sample_at: episode_reset
  physics:
    arena_size_mult: [0.8, 1.25]
    speed_mult: [0.9, 1.1]
    turn_rate_mult: [0.9, 1.1]
    trail_width_mult: [0.8, 1.2]
  observation:
    color_palette: random
    ray_noise_std: small
    raster_zoom: small
```

Keep sampled values constant for the whole episode. Per-tick rule randomization
would make failures hard to replay.

## Phases

| Phase | Add | Keep fixed | Pass signal |
| --- | --- | --- | --- |
| 0 | No variation. | Everything. | Fixed v0 baselines pass. |
| 1 | Observation augmentation: colors, ray noise, raster crop/scale. | Physics and rewards. | No loss on canonical eval; better held-out visual eval. |
| 2 | Mild physics variation: arena size, speed, turn rate, trail width. | Trail gaps, bonuses, player count. | Policy still passes canonical eval and improves held-out ranges. |
| 3 | Trail gaps and bonus frequency profiles. | Source-fidelity tests. | Robust eval improves without hiding rule-specific failures. |
| 4 | Player-count curriculum and opponent pools. | Evaluation protocol. | Better play against held-out counts and held-out opponents. |

## Evaluation Matrix

Every robust checkpoint should run:

| Eval set | Meaning |
| --- | --- |
| Canonical fixed | Did robustness hurt the target game? |
| In-range fixed | Did training learn the sampled profile? |
| Held-out mild | Does it generalize just outside training ranges? |
| Stress edge | Where does it break? |
| Opponent holdout | Did self-play overfit to its own pool? |

Report win/loss/tie/timeout, episode length, collision cause, seat delta, sampled
config, opponent id, and confidence intervals.

## CurvyZero Defaults

Recommended first profiles:

- `curvyzero-robust-observe-v0`: color/palette changes, small ray noise, small
  raster crop or scale jitter. No physics changes.
- `curvyzero-robust-physics-v0`: arena size, speed, turn rate, and trail width
  multipliers only.
- `curvyzero-robust-events-v0`: trail gaps and bonus frequency, after bonuses
  and gaps are implemented and tested.
- `curvyzero-robust-multiplayer-v0`: player count and opponent-pool variation,
  after 1v1 is strong.

## Logging Contract

Each episode record should include:

- `ruleset_id` and `rules_hash`
- `variation_profile_name` and profile version
- sampled arena size, speed, turn rate, trail width, gap settings, bonus rate,
  player count, observation noise, and raster scale
- RNG seed and sampled spawn data
- observation schema hash and reward schema hash
- opponent policy/checkpoint ids
- train/eval split name

## Recommendations

- Start with observation augmentation because it is cheap and does not disturb
  source reconstruction.
- Keep physics variation small until fixed v0 learning is reliable.
- Use named profiles instead of ad hoc random ranges in scripts.
- Always keep canonical fixed evaluation. A robust agent that forgets the target
  game is not useful yet.
- Use opponent pools for self-play robustness, but keep held-out opponents that
  training never samples.
