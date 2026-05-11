# Environment Fidelity, Variation, And Curriculum

Status: Proposed research note

Source read:

- `curvytron_muzero_modal_handoff.md`, Version 2, May 8, 2026.
- `third_party/curvytron-reference/src/shared/model/BaseGame.js`
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js`
- `third_party/curvytron-reference/src/server/model/Game.js`
- `third_party/curvytron-reference/src/server/core/World.js`
- `third_party/curvytron-reference/src/server/core/Island.js`
- `third_party/curvytron-reference/src/server/core/AvatarBody.js`
- `third_party/curvytron-reference/src/server/manager/PrintManager.js`
- `third_party/curvytron-reference/src/shared/model/BaseRoomConfig.js`
- `third_party/curvytron-reference/src/server/model/RoomConfig.js`

## Short Answer

Do not make the training environment a byte-for-byte browser clone. Do make the target ruleset explicit, source-calibrated, deterministic, and hard to drift accidentally.

The game-defining pieces need source fidelity: continuous forward motion, turn dynamics, head/trail/wall collisions, trail thickness and spacing, round scoring, same-tick death behavior, spawn constraints, player count scaling, and bonus semantics when bonuses are enabled. The implementation can differ from the JavaScript internals if golden cases and rollout fingerprints preserve the intended behavior.

The first v0 environment can deliberately invent simplifications: fixed physics timestep, action repeat, one round per episode, no warmup/warmdown, no match-to-max-score, 1v1 only, no bonuses, reward normalization, observation design, max tick cap, curriculum spawn fixtures, and a faster collision backend. These must be labeled as project choices, not described as CurvyTron reference behavior.

Parameterize anything that could later become curriculum or domain randomization. Store the sampled episode parameters, ruleset version, observation schema, reward schema, and environment hash in every replay chunk and checkpoint.

## Fidelity Classes

Use these labels in code comments, config docs, golden fixtures, and decision records:

| Label | Meaning | Examples |
| --- | --- | --- |
| `source-derived` | Matched to CurvyTron reference source or gameplay evidence. | Base speed, turn rate, radius, wall boundary semantics. |
| `source-inspired` | Preserves the spirit but not exact implementation. | Swept collision backend matching source golden cases. |
| `v0-choice` | Project invention for the first learnable environment. | Fixed `dt`, no bonuses, one round per episode. |
| `curriculum-only` | Training simplification or staged difficulty. | Larger arena, slower speed, fixed spawns, solid trails. |
| `domain-randomized` | Randomized across an explicit distribution for robustness. | Speed range, arena size range, spawn distributions. |
| `unresolved` | Known ambiguity that must not silently become canonical. | Head-head tie behavior before a golden rule is chosen. |

## Rule Details That Need Source Fidelity

These are the details most likely to change policy behavior if they drift.

| Rule surface | Reference evidence | Recommendation |
| --- | --- | --- |
| Continuous forward motion | `BaseAvatar.updatePosition` advances every update from velocity components. | Source-derived. Agents should always move while alive. |
| Action semantics | `PlayerInput.resolve` emits left, right, or false/no-turn; server `GameController.onMove` calls `updateAngularVelocity`. | Prefer 3 actions for CurvyTron v1 fidelity: left, straight, right. A forced left/right-only variant is allowed, but name it separately. |
| Time integration | `BaseGame.framerate` is nominally 60 FPS, but source uses elapsed milliseconds. | Use a fixed deterministic `physics_dt_ms` in training. Calibrate default to the source 60 Hz rhythm and record that this is a v0-choice. |
| Speed and turn dynamics | `BaseAvatar.velocity = 16`, `angularVelocityBase = 2.8/1000`, and velocity affects angular velocity via `updateBaseAngularVelocity`. | Source-derived defaults. Parameterize speed, turn rate, and the speed-turn coupling. |
| Head and trail radius | `BaseAvatar.radius = 0.6`; `AvatarBody` uses avatar radius. | Source-derived default. Parameterize head radius and trail radius, even if they start equal. |
| Self-collision grace | `BaseAvatar.trailLatency = 3`; `AvatarBody.match` ignores recent own trail points. | Source-derived. Test exact latency behavior because it strongly affects turning tactics. |
| Trail body spacing | Server `Avatar.isTimeToDraw` adds a trail body only after distance exceeds radius. | Source-derived default. If replaced by occupancy rasterization, golden cases must preserve effective thickness and spacing. |
| Trail holes/gaps | `PrintManager` alternates printing and non-printing distances with `printDistance = 60` and `holeDistance = 5`. | This is a real source rule, not a bonus. A solid-trail v0 is a valid curriculum invention, but it should not masquerade as reference CurvyTron. |
| Collision predicate | `Island.bodiesTouch` uses distance `<` sum of radii; `World.getBody` checks body corners against spatial islands. | Source-derived semantics, implementation can be source-inspired. Test strict boundary, near miss, grazing, and tunneling cases. |
| Wall boundary | `World.getBoundIntersect` kills when body plus radius crosses the square unless `borderless` is active. | Source-derived. Keep wall inclusivity explicit. |
| Arena size scaling | `BaseGame.perPlayerSize = 80`; `getSize(players)` grows as players increase. | Source-derived for reference rulesets. v0 may use fixed arena size, but record it as a v0-choice. |
| Spawn position and heading | `World.getRandomPosition` uses radius plus `spawnMargin = 0.05`; `getRandomDirection` uses `spawnAngleMargin = 0.3`. | Source-derived defaults. Curriculum may use fixed or easier spawns, but canonical evaluation should include random valid spawns. |
| Round scoring | On death, `Game.kill` adds current `deaths.count()` captured at frame start; `resolveScores` gives the sole survivor `players - 1`. | Source-derived behavior should be preserved in a reference ruleset. v0 reward may normalize this, but the underlying rank/tie policy must be explicit. |
| Same-tick deaths | Source code gives deaths in one update the same pre-frame death count, but sequential collision checks can still create ambiguity. | Decide canonical same-tick behavior, write tests, and cite whether it is source-derived or v0-choice. |
| Bonuses | Room config enables many bonus classes; default `bonusRate` is 0, but the source has spawn timing, cap, effects, durations, and probabilities. | Disable in v0 as a v0-choice. When added, each bonus needs source-derived effect tests. |
| Match scoring | `BaseRoomConfig.getDefaultMaxScore` uses `(players - 1) * 10`; games can run multiple rounds. | Omit from v0. Add later as a separate episode mode. |

## v0 Choices That Can Be Explicit Inventions

These choices are safe to invent if the ruleset name, config, docs, replay metadata, and tests make the invention visible:

- One round equals one episode.
- `num_players = 2` only.
- No bonuses or powerups.
- Fixed deterministic timestep instead of browser elapsed-time frames.
- `action_repeat` greater than one to reduce horizon and search cost.
- Terminal reward normalized from rank or win/loss instead of raw match score.
- Max tick cap and truncation handling.
- Observation type: egocentric raster, rays, compact features, history frames, or hidden-state debug views.
- Faster collision backend: occupancy grid, swept circles, or hybrid.
- Fixed spawn fixtures for tests and early curriculum.
- Larger arena, slower speed, smaller turn rate, or delayed self-collision for curriculum.
- Solid trails as an early training variant, if it is named as such.
- Policy-only opponents, ego-seat rotation, and opponent checkpoint sampling.

The key rule is provenance: a v0 invention is fine; a silent invention is technical debt.

## Parameterization For Variation

The environment config should separate canonical rules from sampled variation. A suggested shape:

```text
ruleset:
  id
  version
  source_family              # curvytron-v1, curvytron2, curvyzero
  source_commit              # when source-derived
  rules_hash
  observation_schema_hash
  reward_schema_hash

core:
  num_players
  arena_size
  boundary_mode              # walls, wrap/borderless
  episode_mode               # round, match
  max_ticks

physics:
  physics_dt_ms
  action_repeat
  speed
  turn_rate
  speed_turn_coupling
  action_set                 # left_straight_right, left_right
  action_latency_ticks
  numeric_precision

collision:
  backend                    # source_points, occupancy, swept
  head_radius
  trail_radius
  trail_point_spacing
  self_collision_latency
  strict_overlap             # source uses distance < radius sum
  collision_epsilon

trails:
  gap_mode                   # source_random, solid, fixed_pattern
  print_distance_distribution
  hole_distance_distribution
  initial_printing

spawn:
  sampler                    # source_random, fixed_fixture, curriculum
  margin
  heading_margin
  min_player_distance
  seed_stream

scoring:
  score_mode                 # source_round, normalized_rank, win_loss
  same_tick_tie_policy
  survivor_bonus
  match_max_score
  truncation_reward

bonuses:
  enabled
  enabled_types
  spawn_rate
  cap
  durations
  effect_magnitudes
  probabilities

observation:
  family
  egocentric
  resolution
  crop_radius
  ray_count
  channels
  history_frames
  observation_noise

randomization:
  distribution_id
  sampled_fields
  per_episode_sample_recording
```

Randomized fields must be sampled from named distributions, not ad hoc code. Every trajectory should record the actual sampled values, not only the distribution name.

## Curriculum Recommendation

Keep one canonical evaluation target and add curriculum variants around it.

| Stage | Purpose | Example variant |
| --- | --- | --- |
| 0. Fixtures | Debug deterministic physics and scoring. | Fixed spawns, scripted actions, no random gaps. |
| 1. Simple learnability | Let random/heuristic/baseline agents learn the core loop. | 1v1, no bonuses, large arena, slower speed, possibly solid trails. |
| 2. Canonical no-bonus | Match the intended base target. | Source-calibrated speed, turn, radius, walls, scoring, spawn, and trail gap setting. |
| 3. Robust base game | Prevent overfitting to one geometry. | Randomize arena size, spawn, speed, turn rate, action repeat within narrow ranges. |
| 4. Multiplayer | Expose rank scoring and crowding. | 3-4 players with policy-only or checkpoint opponents. |
| 5. Bonuses | Add one stochastic mechanic at a time. | Start with clear/simple effects, then speed, size, inverse, borderless, and clear-trails. |
| 6. Held-out randomization | Measure transfer rather than train-test leakage. | Wider randomization ranges and unseen combinations. |

Curriculum variants should never replace the canonical target. Report evaluation on at least:

- Fixed canonical seeds.
- Held-out canonical seeds.
- In-distribution randomized variants.
- Held-out randomized variants.
- Reference/golden scenarios for rule drift.

## Ruleset Versioning

Rulesets should be versioned semantically and hashed mechanically.

Recommended identifiers:

```text
curvytron-v1-reference@1.0.0
curvyzero-v0-solid@0.1.0
curvyzero-v0-source-gaps@0.2.0
curvyzero-v0-randomized-narrow@0.1.0
```

Version rules:

- MAJOR: changes that can invalidate trajectory meaning, rewards, observations, legal actions, or terminal outcomes.
- MINOR: adds optional parameters or variants without changing defaults.
- PATCH: documentation, provenance, tests, or implementation fixes that preserve golden behavior.

Hash rules:

- `rules_hash`: canonical JSON of dynamics, collision, spawn, scoring, bonus, and randomization defaults.
- `observation_schema_hash`: observation family, shape, channels, normalization, perspective, and history.
- `reward_schema_hash`: reward transform, tie handling, truncation, and discount assumptions if stored with data.
- `implementation_hash`: code/package version used for reproducibility, separate from behavioral rules.

Replay chunks, model checkpoints, evaluation summaries, videos, and benchmark outputs should store all four identifiers. Training should refuse to mix replay with incompatible hashes unless an explicit migration script exists.

## Drift Tests

These tests prevent silent behavior changes:

| Test | What it catches |
| --- | --- |
| Config hash snapshot | Any behavior-affecting config change without an intentional ruleset update. |
| Deterministic reset/step | RNG, spawn, and update-order nondeterminism. |
| Fixed action rollout fingerprint | Physics, collision, reward, and termination drift across commits. |
| Wall collision goldens | Boundary inclusivity and radius mistakes. |
| Self-trail latency goldens | Off-by-one errors in recent-own-trail immunity. |
| Opponent trail goldens | Cross-player collision regressions. |
| Head-head and same-tick death goldens | Tie and update-order regressions. |
| Grazing/near-miss/tunneling goldens | Collision backend drift. |
| Trail gap distribution tests | Accidental removal or reshaping of source gap behavior. |
| Spawn validity tests | Invalid starts, wall-facing starts, and overlapping players. |
| Scoring table tests for N players | Rank reward and survivor bonus regressions. |
| Vectorized-vs-single equivalence | Batch stepping bugs. |
| Save/load replay equivalence | Serialization and resume drift. |
| Observation schema snapshots | Channel order, shape, scaling, and perspective drift. |
| Cross-backend parity tests | Occupancy/swept/source-inspired backend differences beyond tolerance. |

Golden fixtures should say whether expected behavior is source-derived, source-inspired, or v0-choice. That prevents tests from accidentally canonizing a temporary training simplification.

## Drift Metrics

Track these metrics for fixed canary seeds and random-agent/heuristic-agent suites:

| Metric | Why it matters |
| --- | --- |
| Episode length distribution | Sensitive to speed, collision, spawn, and gap changes. |
| Death cause distribution | Catches wall/trail/head collision drift. |
| Terminal rank/score distribution | Catches scoring and tie-policy drift. |
| Collision counts by type | Shows whether a backend changed effective geometry. |
| Spawn distance and heading histograms | Catches sampler drift. |
| Trail gap length histograms | Catches gap RNG or spacing changes. |
| Random agent win rate by seat | Catches spawn asymmetry and update-order bias. |
| Heuristic-vs-random win rate | Catches behavior changes large enough to affect policy quality. |
| Vectorized mismatch rate | Catches batch-only bugs. |
| Replay incompatibility count | Catches accidental mixing of rules/schema hashes. |
| Physics ticks/sec and episodes/sec | Catches performance regressions that may force unwanted rule shortcuts. |

Use statistical tolerances for distribution metrics, but require exact matches for deterministic fingerprints under fixed config, seed, and action trace.

## Recommended Decision

Define `curvyzero-v0` as a small, explicit training ruleset, not a vague clone. Use source-derived defaults for movement, collision scale, wall rules, spawning, and rank scoring. Disable bonuses and match scoring at first. Treat fixed timestep, action repeat, observation, reward normalization, and any solid-trail simplification as named v0 inventions.

Create a separate `curvytron-v1-reference` ruleset once enough golden cases are extracted from the JavaScript reference. Use it as a fidelity anchor and regression oracle, not necessarily as the fastest training mode.

Before MuZero consumes data, require ruleset hashes, deterministic rollout fingerprints, collision goldens, scoring/tie tests, vectorized equivalence, and canary-agent drift metrics.

## Open Questions

- Is the project target closer to original CurvyTron v1 source behavior or CurvyTron 2 public rules?
- Should the first canonical v0 include source-style trail gaps, or should solid trails be a named curriculum variant only?
- Should the default action set be three actions from the v1 source input model, or two actions to match the stricter CurvyTron 2 wording?
- Which collision backend becomes the default after parity and throughput benchmarks?
- What tolerance is acceptable between source point-body collision and a training occupancy/swept backend?
