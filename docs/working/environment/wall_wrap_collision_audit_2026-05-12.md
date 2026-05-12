# Wall, Wrap, And Collision Audit

Date: 2026-05-12

Scope: original CurvyTron source, current Python source-shaped env, current fast
vector runtime, and existing environment docs/tests. This doc answers the narrow
question: normal wall death or borderless wrap, and what body/trail collision
order actually means for 2-player fidelity.

## Plain Answer

- Original CurvyTron defaults to normal wall death. It is not borderless by
  default and it is not a torus by default.
- Borderless exists in the original game as a game bonus (`BonusGameBorderless`)
  and as the `game.borderless` state; in CurvyZero it can also be forced by
  explicit source/vector fixture config. While active, an avatar whose center
  moves strictly outside the map is teleported to the opposite edge.
- Borderless is not full toroidal collision. The source does not use periodic
  distance checks, does not duplicate bodies across borders, and skips body
  lookup on the wrap frame. A body at the destination can kill the avatar on the
  next frame, not the same wrap frame.
- CurvyZero's source-shaped scalar env implements the source branch: normal
  wall death by default, optional borderless wrap, wall-check priority over body
  collisions, and source-shaped body/trail collision latency.
- CurvyZero's fast vector runtime implements the covered normal wall, borderless
  wrap, and body/trail collision rules, including source reverse player order.
  The main open implementation gap is ambiguous multi-body killer ordering:
  source searches island bodies newest-first, while the vector owner resolver
  scans flat body slots oldest-first.
- Default training should stay normal wall death for the no-bonus ruleset
  (`curvytron_no_bonus/v0`). Borderless should appear only when the source
  bonus/ruleset says it is active. `profile_no_death` forcing borderless is a
  profiling/debug mode, not source fidelity.

## Original Source References

Normal wall/default:

- `third_party/curvytron-reference/src/shared/model/BaseGame.js:81` sets
  `BaseGame.prototype.borderless = false`.
- `third_party/curvytron-reference/src/shared/model/BaseGame.js:170-172`
  resets each new round back to the base `borderless` default.
- `third_party/curvytron-reference/src/server/model/Game.js:51` calls
  `world.getBoundIntersect(avatar.body, this.borderless ? 0 : avatar.radius)`.
- `third_party/curvytron-reference/src/server/model/Game.js:58` kills with
  `killer=null` when a normal border is hit.
- `third_party/curvytron-reference/src/client/controller/game/RoundController.js:115-117`
  only toggles the client `borderless` CSS class from `game.borderless`.

Borderless source path:

- `third_party/curvytron-reference/src/server/model/Bonus/BonusGameBorderless.js:20`
  gives the bonus a `10000` ms duration.
- `third_party/curvytron-reference/src/server/model/Bonus/BonusGameBorderless.js:36-39`
  applies `['borderless', true]`.
- `third_party/curvytron-reference/src/server/model/Game.js:55` wraps by calling
  `world.getOposite(...)` when `this.borderless` is true.
- `third_party/curvytron-reference/src/server/core/World.js:276-294`
  checks bounds using strict outside tests: `< 0` and `> this.size`.
- `third_party/curvytron-reference/src/server/core/World.js:305-323`
  maps one boundary coordinate to the opposite edge. Because the bound check
  returns early, corner exits resolve the first matching axis.

Body/trail collision source path:

- `third_party/curvytron-reference/src/server/model/Game.js:37-78`
  updates avatars in reverse order. One avatar is fully processed before the
  next: movement, optional point insertion, border branch, body branch,
  print-manager/bonus checks if still alive.
- `third_party/curvytron-reference/src/server/model/Game.js:62` performs body
  lookup only in the non-border branch. Wall death therefore has priority over
  body death; borderless wrap also skips body lookup in that frame.
- `third_party/curvytron-reference/src/server/model/Game.js:89-94` kills an
  avatar, records the death, and marks a death in frame.
- `third_party/curvytron-reference/src/server/model/Game.js:113-118` inserts an
  `AvatarBody` into `World` synchronously on every point event while the world
  is active.
- `third_party/curvytron-reference/src/server/model/Avatar.js:40-47` emits a
  normal point only when distance from the last trail point is strictly greater
  than avatar radius.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:285-289`
  makes death emit one more point at the death position.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:297-307`
  makes printing toggles emit an important point; toggling off clears the
  visible trail object, not already stored world bodies.
- `third_party/curvytron-reference/src/server/core/Island.js:62-69` checks
  candidate bodies newest-first within an island.
- `third_party/curvytron-reference/src/server/core/Island.js:83-89` uses strict
  circle overlap (`distance < radius sum`), so exact tangent is safe.
- `third_party/curvytron-reference/src/server/core/AvatarBody.js:33-37` gates
  same-avatar hits with `current.num - stored.num > trailLatency`.
- `third_party/curvytron-reference/src/shared/model/BaseAvatar.js:61` sets
  `trailLatency = 3`.

## Collision Semantics

"A trail hits another trail" means the moving avatar's current circular head
overlaps a stored point body owned by that other avatar. It is not a swept line
intersection and it is not a rendered-polyline crossing test.

Consequences:

- Opponent stored bodies are lethal immediately on strict overlap.
- Own stored bodies are ignored until the live body's point number is more than
  `trailLatency` ahead of the stored body's number. With source default latency
  `3`, delta `3` is safe and delta `4` kills.
- Exact tangent is safe because the predicate is strict `<`.
- Visual trail gaps and `Trail.clear()` do not prove the collision world is
  empty. Stored world bodies remain until world/game clear or round reset.
- A higher-index avatar can emit a point first in the same source frame and that
  point can kill a lower-index avatar later in the same frame because the source
  loop runs players in reverse order.
- A death point can also kill a later-updated avatar in the same frame. Round
  end is checked after the reverse-order loop, not immediately after the first
  kill.
- Source collision is endpoint-only per update. If a large single step leaps
  over a small stored body without the endpoint overlapping, the source semantics
  miss the collision. Training wrappers need source-sized substeps to avoid
  introducing extra tunneling.

## Current Implemented Status

`CurvyTronSourceEnv`:

- `src/curvyzero/env/source_env.py:98-116` models source bodies and same-owner
  latency.
- `src/curvyzero/env/source_env.py:170-205` stores island bodies and scans
  them in reverse insertion order.
- `src/curvyzero/env/source_env.py:221-283` adds and looks up bodies through a
  source-shaped world.
- `src/curvyzero/env/source_env.py:605-637` allows explicit reset with
  `borderless`, defaulting false.
- `src/curvyzero/env/source_env.py:1327-1382` mirrors source update order:
  reverse player loop, normal point before border/body checks, borderless wrap
  branch, normal wall death branch, then body lookup.
- `src/curvyzero/env/source_env.py:1400-1427` emits the death point and death
  event.
- `src/curvyzero/env/source_env.py:1777-1782` applies game borderless changes.
- `src/curvyzero/env/source_env.py:1961-1989` mirrors strict bound checks,
  opposite-edge mapping, and strict circle overlap.

Fast vector runtime:

- `src/curvyzero/env/vector_multiplayer_env.py:38-40` separates the no-bonus,
  seeded-bonus, and natural-bonus rulesets. The no-bonus ruleset is
  `curvytron_no_bonus/v0`.
- `src/curvyzero/env/vector_multiplayer_env.py:342-355` derives map size,
  radius, and trail latency from `CurvyTronReferenceDefaults`.
- `src/curvyzero/env/vector_multiplayer_env.py:685-686` forces borderless only
  in `profile_no_death`; this is explicitly not source fidelity.
- `src/curvyzero/env/vector_multiplayer_env.py:2985-3094` exposes public
  `ruleset_id`, `borderless`, death cause, hit owner, and death order policy
  metadata.
- `src/curvyzero/env/vector_multiplayer_env.py:3603-3617` initializes
  `borderless=false` and source default `trail_latency`.
- `src/curvyzero/env/vector_runtime.py:518-710` processes players in reverse
  order and checks wrap, wall, then body collision in source order.
- `src/curvyzero/env/vector_runtime.py:625-645` calls borderless wrap before
  normal wall detection.
- `src/curvyzero/env/vector_runtime.py:661` records wall deaths as
  `DEATH_CAUSE_WALL`.
- `src/curvyzero/env/vector_runtime.py:706-710` blocks body lookup after a wall
  death or borderless wrap.
- `src/curvyzero/env/vector_runtime.py:1486-1533` implements borderless wrap
  with strict center-outside checks and first-axis behavior.
- `src/curvyzero/env/vector_runtime.py:1582-1625` implements normal wall hits
  as radius-margin bound overlap only when `borderless` is false.
- `src/curvyzero/env/vector_runtime.py:5179-5197` implements strict body overlap
  and same-owner latency.
- `src/curvyzero/env/vector_runtime.py:5203-5226` chooses the public hit owner
  for body deaths.
- `src/curvyzero/env/vector_runtime.py:5231-5238` maps same-owner vs opponent
  hit owners to death-cause codes.
- `src/curvyzero/env/vector_runtime.py:2897-2906`,
  `src/curvyzero/env/vector_runtime.py:3983-4061`, and
  `src/curvyzero/env/vector_runtime.py:4143-4162` implement
  `BonusGameBorderless` catch and expiry state.

## Existing Tests And Docs

Docs already record source-pinned behavior:

- `docs/working/environment/borderless_trail_probe_plan.md` records
  borderless wrap, destination-body skip until next frame, exact-edge safety,
  and first-axis corner behavior.
- `docs/working/environment/body_trail_investigation.md` records strict
  overlap, tangent safety, opponent immediate hits, own-body latency, same-frame
  point insertion, and print-manager order.
- `docs/working/environment/trail_gap_collision_probe_plan.md` records forced
  hole safe crossing, stored bodies still killing through visual gaps, and
  print-to-hole boundary bodies killing later players.
- `docs/working/environment/collision_order_probe_plan.md` records reverse
  player order, death-point same-frame effects, and head-head/order asymmetry.
- `docs/working/environment/browser_rendering_spec_2026-05-11.md` separates
  rendered trail facts from collision truth and calls out source-frame stepping
  to avoid tunneling.

Focused workspace tests cover the 2P slices directly:

- `tests/test_2p_collision_fidelity.py:208-234` covers source/vector opponent
  stored-body death.
- `tests/test_2p_collision_fidelity.py:237-257` covers strict tangent safety.
- `tests/test_2p_collision_fidelity.py:405-462` covers source/vector same-owner
  trail latency.
- `tests/test_2p_collision_fidelity.py:465-489` covers rendered-line crossing
  without stored-point overlap as safe.
- `tests/test_2p_collision_fidelity.py:492-523` covers wall death priority over
  body overlap.
- `tests/test_2p_collision_fidelity.py:526-569` covers borderless wrap skipping
  destination body until the next frame.
- `tests/test_vector_multiplayer_env.py:2398-2504` covers public collision-order
  canary fixtures for death-point and head-head reverse-order behavior.

## Open Gaps

- Ambiguous multi-body killer order is not fully source-faithful in the fast
  vector path. Source uses island corner lookup and newest-first body order
  (`Island.getBody` reverse iteration). The vector runtime's
  `_first_hit_body_owner` scans flat slots from `0..capacity-1`, which is
  oldest-first for normal append order. Add a focused fixture where two
  different owners' stored bodies overlap the same victim endpoint and verify
  source killer identity.
- The fast vector public metadata records hit owner and cause, but does not
  preserve source `old` body age for kill-log style semantics. SourceEnv has
  `SourceBodyState.is_old`; vector debug events currently pass `old=False` for
  body hits.
- Source body lookup uses island/corner broad phase. The vector runtime scans
  all active body slots. This is usually stricter and simpler for covered cases,
  but ambiguous island-boundary candidate order remains unpinned.
- Training pathways must keep using source-sized internal frames. A single large
  trainer decision step can tunnel through endpoint-only stored bodies; this is
  source-compatible for one large update but bad as a default training wrapper.
- Browser-like rendering should keep collision truth separate from visual line
  topology. A rendered line crossing is not a source collision unless there is a
  stored body overlap at the checked endpoint.

## Recommendation

Use normal wall death as the default 2-player training rule. Enable borderless
only as source-default bonus state or explicit fixture state. Do not describe the
game as toroidal: the source behavior is a temporary opposite-edge teleport with
no same-frame destination-body lookup and no periodic collision geometry.
