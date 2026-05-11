# Environment Fidelity Plan Critique

Status: Draft critique
Date: 2026-05-08

## Read

- `docs/design/environment/` and `docs/research/environment/` are present but empty.
- `docs/design/deterministic_environment.md`
- `docs/design/rulesets.md`
- `docs/design/training_architecture.md`
- `docs/research/curvytron_reference_notes.md`
- `docs/research/env_fidelity_curriculum.md`
- `docs/research/baseline_learnability.md`
- `docs/research/performance_vectorization.md`
- `docs/research/observation_reward_design.md`
- `docs/research/training_architecture_notes.md`
- `docs/research/multiplayer_selfplay_muzero.md`
- `curvytron_muzero_modal_handoff.md`
- Spot checks in `third_party/curvytron-reference` for client/server messages, room lifecycle, disconnect handling, and client rendering.

## Short Critique

The current fidelity plan is directionally right: it separates source-derived facts from v0 choices, keeps the simulator deterministic, and asks for golden tests before serious MuZero. The weak spot is that it treats source fidelity mostly as local mechanics: movement, collision, trail gaps, and scoring. That is not enough.

The plan needs multiplayer as a main axis from the start. A 1v1-only test suite can pass while hiding wrong map scaling, bad spawn sampling, seat-order bias, incorrect rank scoring, wrong same-frame death behavior, broken bonus targeting, missing server message fan-out, observation channel leaks, and training data that only works from one ego perspective.

The other missing axis is browser/server reality. Training v0 does not need exact sockets or rendering, but the project still needs to know which parts are authoritative server state, which are client prediction/interpolation, and which are only UI-visible. Otherwise a future demo can disagree with the trained simulator in ways that are hard to debug.

## Sharpest Critiques

1. Multiplayer is underweighted.
   The docs say "start 1v1," which is fine for training v0, but many source rules only become meaningful with 3+ players. The plan should add multiplayer canary tests before claiming source fidelity, even if training stays 1v1.

2. Golden tests are too local.
   Wall/self/opponent/head-head tests are necessary, but not enough. Add whole-round fingerprints and distribution checks for 2, 3, 4, and 8 players: episode length, death ranks, death causes, same-frame death groups, spawn distances, and score deltas.

3. "Source-compatible simultaneous actions" is not the same as source update order.
   The reference server updates avatars in an order while also capturing some per-frame values. A clean two-phase simulator is better for fairness, but it is a v0 choice unless verified against reference behavior. This matters most in multiplayer.

4. Solid trails are a dangerous simplification if they become the default target.
   Source trail holes are not a bonus. A solid-trail v0 is a good learnability step, but canonical no-bonus evaluation should decide whether source-style holes are included.

5. The collision comparison can be too weak.
   Matching isolated circle-overlap cases does not prove an occupancy grid matches the source. Need sampled rollout parity, grazing/tunneling fixtures, effective trail-width metrics, and backend mismatch reports.

6. Current tests can overfit to the implemented toy env.
   The current simulator uses fixed two-player spawns, a 64x64 grid, rounded cells, solid trails, no self-collision grace, no source timing, and flat global observations. Tests around that env prove determinism, not CurvyTron fidelity.

7. Network and client behavior should be classified, not ignored.
   Per-message timing, compression, spectator attach state, disconnect removal, ready/warmup/warmdown, position/angle/point events, and render prediction do not belong in training v0. But they do belong in a browser/demo fidelity checklist.

8. Some proposed work is overkill for v0.
   Full bonus semantics, exact JS elapsed-time jitter, exact wire protocol, pixel-perfect canvas rendering, JAX-native envs, native extensions, and full multi-agent MCTS should wait until the simulator and baselines are stable.

## Multiplayer: What Breaks If We Only Test 1v1

| Area | What 1v1 hides | Needed check |
| --- | --- | --- |
| Map size | The source arena grows with player count. A fixed 64x64 map can look fine in 1v1 and be wrong for 4-8 players. | Assert source map size for 1, 2, 4, and 8 players. Run stress at each size. |
| Spawn | Fixed mirrored 1v1 spawns do not test random valid positions, heading margins, overlap avoidance, wall-facing starts, or crowded starts. | Sample many 3+ player spawns. Track wall margin, pair distances, heading distribution, and immediate death rate. |
| Scoring | 1v1 win/loss hides rank scoring, tied rank averaging, survivor bonus, and match max-score behavior. | Table tests for 3 and 4 players with known death orders and ties. |
| Same-frame deaths | In 1v1, same-frame death is just a draw. In 3+ players, tie groups affect ranks and scores while other players may survive. | Script frames where 2 of 4 die together, 3 of 4 die together, and all die together. |
| Player order | Seat-order bias is easier to miss in 1v1, especially if spawns are symmetric. In 3+ players, update order can change who hits a trail, who catches a bonus, or who receives a score first. | Run mirrored/permuted player-order tests. Same physical setup should give order-independent v0 results. |
| Bonuses | Many bonuses target self, enemies, all alive players, or game state. Those semantics are trivial or misleading in 1v1. | Bonus effect tests with 3+ players before enabling bonuses in any canonical ruleset. |
| Server messages | 1v1 does not stress spectator snapshots, player leave, score fan-out, bonus events, round events, and per-avatar property streams. | Browser/demo message replay with 3+ players and one spectator. |
| Observation perspective | 1v1 can hide seat/color leaks. Multiplayer needs stable ego-centered observations with multiple opponents and no fixed player-slot shortcut. | Permute player ids/colors/seats and check ego observations transform correctly. |
| Training ego perspective | A 1v1 ego wrapper can accidentally train only player_0 logic. Multiplayer needs every live player to become an ego sample with correct opponent metadata. | Replay records should include ego id, perspective transform, joint action, opponent policy ids, and per-player payoff. |

Multiplayer does not need to block v0 training. It should block claims like "source faithful," "canonical no-bonus," or "ready for multiplayer self-play."

## Fidelity Inventory

| Surface | Needs fidelity decision | Training v0 | Browser/demo fidelity |
| --- | --- | --- | --- |
| Timing | Fixed dt vs elapsed milliseconds, tick rate, action repeat, timeout, warmup/warmdown, trail-start delay. | Must pin fixed deterministic dt, action repeat, max ticks, timeout handling. | Need source warmup/warmdown, 60 Hz target, elapsed-step behavior, and trail-start delay. |
| Controls | Left/right/straight mapping, both-keys/no-key behavior, inverse controls, one-shot right-angle bonus. | Must pin static 3-action schema and masks. | Need client key/touch/gamepad behavior and inverse/straight-angle effects. |
| Spawn | Position sampler, heading sampler, margins, overlap checks, seed stream. | Must have deterministic seed contract and valid non-overlapping starts. | Need source random position/direction behavior and room/player count effects. |
| Coordinate system | Units, origin, axes, wall inclusivity, float precision, grid mapping. | Must pin continuous-to-grid mapping and wall semantics. | Need source units and transport compression behavior if replaying messages. |
| Map size | Fixed training grid vs source formula by player count. | Must pin v0 map size and hash it. | Need source formula and resizing when present player count changes. |
| Collision | Wall, old trail, own trail, opponent trail, head-head, crossed paths, strict overlap, update order. | Must cover deterministic collision classes and order-independent ties. | Need source endpoint-circle/island behavior or a documented approximation. |
| Self-collision delay | Source ignores recent own trail points by point-number latency. | Must choose and test either no grace or source-style grace. | Need exact source threshold. |
| Trail gaps | Solid trails vs distance-based source print/hole lengths and delayed printing. | Solid is okay only as named v0/curriculum. | Need source print manager distances and randomness. |
| Scoring | Terminal reward, rank score, same-frame score capture, survivor bonus, match max score. | Must pin 1v1 payoff, ties, truncation, and future rank schema. | Need source round score and match winner behavior. |
| Bonuses | Spawn rate, cap, clear position, type weights, targets, durations, stacking, borderless, clear trails. | Disable and hash as disabled. | Need one bonus at a time, tested mostly in 3+ players. |
| Network timing | Input arrival, event batching interval, forced latency events, ready timeout, spectator attach. | Not in hot-loop v0. | Need for live browser policy demo and server replay. |
| Client interpolation/rendering | Client local prediction, server correction, trail drawing, position/angle/point events, canvas scale. | Not needed except debug rendering. | Need if demo should feel like CurvyTron. |
| Randomness | Spawn, trail gaps, bonuses, opponent sampling, search noise, seed splits. | Must record seeds and sampled values. | Need source RNG behavior only for reference replay, not training. |
| Player count | 1v1, 3-4 player, 8 player, present vs dead players. | Train 1v1 first, but add multiplayer canaries. | Need source player-count scaling and room behavior. |
| Disconnects | Player leave, avatar removal/death, game end when one present player remains, spectators. | Not needed for v0 learning. | Needed for server/demo fidelity and robust evaluation harnesses. |
| Round lifecycle | Ready, launch, warmup, game:start, round:new, round:end, game:end, warmdown. | One round per episode is fine if explicit. | Need source lifecycle for browser/demo and match mode. |
| UI-visible state | Scoreboard, round score, kill log, bonus stack, borderless state, spectators, ready/activity. | Only store debug info needed for replay. | Need event/message fidelity if a policy drives the real UI. |

## Must-Have For Training v0

- One explicit ruleset id and rules hash for the active v0 behavior.
- Deterministic reset and step for seed, config, and joint action trace.
- Fixed action schema: left, straight, right, plus legal masks.
- Fixed coordinate convention, grid mapping, wall inclusivity, and action repeat.
- 1v1 no-bonus one-round episode with clear timeout truncation.
- Valid spawn contract, even if v0 starts with simple fixed or symmetric spawns.
- Collision tests for wall, self, opponent trail, head-head, crossed paths, and same-tick death.
- Tie handling that is deterministic and independent of player iteration order.
- Terminal reward alignment: reward after joint action, not before it.
- Replay fields for seed, config hash, joint actions, death cause, death tick, and terminal outcome.
- Observation schema hash and basic perspective tests, even if the first learned observation is rays.
- Random-vs-random, heuristic-vs-random, and simple learned baseline gates before MuZero.
- Multiplayer canary tests for map size, spawn validity, rank scoring, same-frame deaths, and player-order permutation, even before training multiplayer agents.

## Nice-To-Have For Browser Or Demo Fidelity

- Source elapsed-millisecond stepping and wall-clock jitter.
- Warmup, warmdown, launch flow, ready timeout, and spectator attach.
- Source trail print/hole randomness and delayed print start.
- Exact endpoint-circle island lookup, unless a measured approximation is accepted.
- Full match-to-max-score ladder.
- Bonuses, durations, stacking, targeting, spawn cap, and visual state.
- Borderless wrapping and clear-trail game bonus.
- WebSocket event batching, message order, compression rounding, latency pings, and disconnect behavior.
- Client local prediction, server correction, canvas scaling, and trail rendering.
- UI elements such as scoreboard, round score, kill log, bonus stack, spectator count, and activity state.

## Weak Comparisons To Strengthen

- Constants-only comparison is weak. Add executable fixtures.
- Isolated collision goldens are weak. Add rollout fingerprints.
- 1v1 parity is weak. Add 3+ player canaries.
- Average random win rate is weak. Add swapped-seat and player-order permutation tests.
- "No crashes" is weak. Add death-cause and timeout distributions.
- Backend equality on hand-picked cases is weak. Add sampled mismatch reports between occupancy and geometry/source-inspired backends.
- Observation shape tests are weak. Add ego-perspective invariance tests under player id, color, and seat permutations.

## Overkill To Avoid For Now

- Reproducing exact JavaScript timing jitter in the training environment.
- Pixel-perfect browser rendering before the simulator is learnable.
- Implementing every bonus before the no-bonus game is solid.
- Full wire-protocol replay in the training hot path.
- Full joint-action MCTS for multiplayer.
- JAX-native, PyTorch-native, C++/Rust, SIMD, or bitpacking before profiling proves the need.
- A full league or population system before 1v1 baselines pass.

## Fastest Safe Path

1. Freeze `curvyzero-v0-solid-1v1` as a deliberately simple training ruleset.
   Keep one round per episode, no bonuses, deterministic fixed dt, fixed action schema, and terminal win/loss/tie reward.

2. Add source-derived metadata without consuming it silently.
   Constants from the reference are useful, but changing metadata must not change v0 behavior unless the ruleset version changes.

3. Build the smallest serious golden suite.
   Cover deterministic rollout, wall, self, opponent trail, head-head, crossed paths, same-tick deaths, timeout truncation, reward alignment, and observation perspective.

4. Add multiplayer canaries now.
   Do not train multiplayer yet. Just test source map-size formula, random spawn validity, 3-4 player scoring tables, same-frame tie groups, player-order permutation, and ego-perspective observation transforms.

5. Run baseline gates before MuZero.
   Random stress, heuristic beats random, then PPO or imitation-plus-PPO beats random on held-out seeds.

6. Create one source-fidelity comparison harness.
   It can start as fixtures and rollout summaries, not a full browser automation stack. The goal is to compare source-derived behavior against the Python simulator and label every mismatch.

7. Promote only after evidence.
   Once v0 is learnable, decide whether canonical no-bonus should use solid trails or source-style gaps. Add bonuses and browser/demo fidelity one piece at a time.

The fastest path is not "clone everything." It is to make v0 small, tested, and honest, while adding enough multiplayer canaries that the project does not build a 1v1 toy and mistake it for CurvyTron.
