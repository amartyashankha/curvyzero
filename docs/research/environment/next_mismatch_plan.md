# Next Mismatch Plan After Kinematics

Status: updated after PrintManager exact-zero verification,
2026-05-09

Scope: after the source-kinematics first-step match, close one game mechanic at
a time with forced-state JS traces and matching Python source-fidelity tests.
Keep `curvyzero-v0` honest as a separate toy ruleset.

Current target: preserve the verified wall/border event batch, verified 3P/4P
normal-wall scoring/death-order/terminal-draw canaries, verified
source-body-canary batch with six body/trail cases, verified
source-print-manager batch with eight deterministic toggle/start/death-stop cases, and verified
source-trail-cadence batch with normal point/no-point behavior, and verified
source-trail-gap batch with forced hole safe, stored body kill, and
print-to-hole boundary kill behavior. The wall/border batch now includes the
borderless exact-edge/corner-axis control. The shared
scenario schema, toy-v0 runner, and source-fidelity runners are now split; keep
`curvyzero.env.scenarios` as a compatibility facade/CLI. Do not broaden into
browser, Modal, benchmark, or public training-interface work while implementing
the next local mechanic slice.

Post-refactor guards are in place: source kinematics and borderless runners
reject forced-state position/heading/alive length mismatches against
`player_count`, and nested `players[].initial.alive` is accepted.

## Source Facts To Preserve

- `Game.update(step)` runs avatars in reverse order.
- Per avatar order is move, wall check, body collision, trail print test, bonus
  catch.
- Wall collision uses the endpoint avatar body and radius as the border margin.
- Borderless collision uses margin `0`; exact edge equality is safe, and a
  corner overshoot resolves the first x-axis hit only.
- Body collision is strict overlap: equal distance is safe.
- Own trail collision is delayed by point number: `current - stored > 3`.
- Point events insert `AvatarBody` objects synchronously through
  `Game.onPoint`, so a higher-index avatar can print a body before a lower-index
  avatar checks body collision in the same `Game.update(step)`.
- Trail print gaps are distance-based: print `60`, hole `5`, with random
  multipliers.
- Scoring uses `deaths.count()` captured once at frame start. Same-frame deaths
  share that value. A survivor gets `max(players - 1, 1)`.

## Ranked Next Mechanics

| Rank | Mechanic | Feedback | Importance | First check |
| ---: | --- | --- | --- | --- |
| 1 | Normal-wall death | Verified for two forced state/event cases | High | `source_normal_wall_death_step` and `source_normal_wall_same_frame_draw_step` pass in the mixed source-border batch. |
| 2 | Source borderless wrap | Verified for plain wrap, PrintManager wrap, destination-body skip, and exact-edge/corner-axis behavior | High | `source_borderless_wrap_step`, `source_borderless_print_manager_wrap_toggle_step`, `source_borderless_wrap_skips_destination_body_then_next_frame_kills`, and `source_borderless_exact_edge_corner_axis_step` pass in the mixed source-border batch. |
| 3 | 3P/4P normal-wall canaries | Verified, narrow only | High | `source_normal_wall_3p_two_die_one_survivor_step`, `source_normal_wall_4p_ordered_deaths_survivor_score`, and `source_normal_wall_4p_two_prior_then_same_frame_terminal_draw` pass in the multiplayer batch. Checks map size, reverse update order, death order, per-death scores, survivor score, no-survivor terminal draw, and round end. |
| 4 | Fidelity runner cleanup | Done | Medium | Shared schema/parsing lives in `curvyzero.env.scenario_schema`, the toy-v0 runner lives in `curvyzero.env.toy_runner`, and source-fidelity implementation lives in `curvyzero.fidelity.source_runners`. `curvyzero.env.scenarios` remains a compatibility facade/CLI. |
| 5 | Seeded body canaries | Verified, narrow only | High | `source_body_opponent_tangent_safe_step` proves the strict-overlap safe side, `source_body_opponent_overlap_kills_step` proves an opponent stored body kills immediately, and `source_body_own_delta3_safe_step` / `source_body_own_delta4_kills_step` prove the own-body `current - stored > 3` latency gate through `source-body-canary`. The tangent fixture uses `21.200000000000003` rather than literal `21.2` because Node represents the literal difference from `20` slightly below the `1.2` radius-sum threshold. |
| 6 | Same-frame point materialization | Verified, narrow only | High | `source_body_same_frame_point_kills_step` proves a p1 point inserted during `Avatar.update` kills p0 later in the same frame. `source_body_same_frame_point_control_safe_step` proves the live overlapping p1 head alone does not kill. |
| 7 | Deterministic print-manager toggles, delayed start, active death stop, and random call order | Verified, narrow only | High | The stable eight-case `source_print_manager_batch.json` plus the separate `source_print_manager_random_batch.json` pass through `source-print-manager-canary`. Checks forced active state, delayed start timing, distance subtraction, exact `<= 0` toggle, property event payload, important point side effect, new distances `5.25` and `39`, random tape order (`p1` consumes `0.1` before `p0` consumes `0.9`), active printing stop-on-death order, active already-hole stop without an important stop point, seeded body-collision death-stop order, cleared manager state, and final trail/body counters. |
| 8 | Normal trail cadence | Verified, narrow only | High | `source_trail_normal_point_step` and `source_trail_no_point_below_radius_step` pass through `source-trail-cadence-canary`. This pins strict `> radius`, ordered point events, visible trail point count, body count, and `worldBodyCount`; it also preserves the distinction between hidden draw cursor and visible `lastTrailPoint`. |
| 9 | Trail gap body absence/crossing | Verified, narrow only | High | `source_trail_gap_hole_space_safe_step`, `source_trail_gap_stored_body_still_kills_step`, and `source_trail_gap_print_to_hole_boundary_kills_step` pass through `source-trail-gap-canary`. This pins forced hole-space safety, stored-body-in-visual-hole kill, print-to-hole boundary collision, reverse update order, p1 own-latency safety, p0 death point insertion, and p1 print-manager distance update. |

## Concrete Next Steps

1. Keep the verified wall/border event batch in regression:
   `uv run python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --artifact-root /private/tmp/curvy-source-border-events-batch-final-docs`.
   Verified result: `6` pass, `0` fail, `0` blocked.
2. Keep the verified 3P/4P normal-wall multiplayer batch in regression:
   `uv run python tools/run_fidelity_batch.py scenarios/environment/source_normal_wall_multiplayer_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-normal-wall-multiplayer-batch-final-docs`.
   Verified result: `3` pass, `0` fail, `0` blocked.
3. Preserve source forced-state guards when extending runners: validate forced
   positions/headings/alive lengths against `player_count` and keep supporting
   nested `players[].initial.alive`.
4. Keep the verified source-body canaries in regression:
   `uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_body_canary_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-body-canary-same-frame`.
   Verified result: `6` pass, `0` fail, `0` blocked; tangent-safe leaves `p0`
   alive with no common `die` event; overlap-kill marks `p0` dead with
   `killer_id: p1` and `old: false`; own delta `3` is safe and own delta `4`
   kills p0 with `killer_id: p0`; same-frame positive kills p0 with
   `killer_id: p1`, while the same-frame control remains all alive with
   `worldBodyCount: 0`.
5. Keep the verified source print-manager canaries in regression:
   `uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_print_manager_batch.json --python-runner source-print-manager-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-print-manager-active-hole-stop`.
   Verified result: `8` pass, `0` fail, `0` blocked; print-to-hole ends with
   `printing: false`, distance `5.25`, cleared trail, and one world body;
   hole-to-print and exact-zero both end with `printing: true`, distance `39`,
   one trail point, and one world body; no-toggle ends with distance `8.4` and no point/body;
   active stop-on-death emits the non-important death point before the important
   stop point/property and clears manager state to zero; active already-hole
   stop-on-death emits no important stop point, keeps the death trail point, and
   clears manager state to zero; seeded body-collision stop-on-death preserves
   collision-before-PrintManager-test order.
   Keep the random call-order probe in its separate batch:
   `uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_print_manager_random_batch.json --python-runner source-print-manager-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-print-manager-random-batch`.
   Verified result: `1` pass, `0` fail, `0` blocked; p1 consumes random value
   `0.1` and gets distance `22.2`, then p0 consumes `0.9` and gets distance
   `55.8`.
6. Keep the verified source trail-cadence canaries in regression:
   `uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_trail_batch.json --python-runner source-trail-cadence-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-trail-cadence-regression`.
   Verified result: `2` pass, `0` fail, `0` blocked; normal point insertion
   ends with one non-important point/body, while the below-radius case emits
   only position and keeps visible `lastTrailPoint: null`.
7. Keep the verified source trail-gap canaries in regression:
   `uv run --extra dev python tools/run_fidelity_batch.py scenarios/environment/source_trail_gap_batch.json --python-runner source-trail-gap-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-trail-gap-regression`.
   Verified result: `3` pass, `0` fail, `0` blocked; forced hole space is safe,
   a stored body in the same visual hole kills p0, and the print-to-hole
   boundary body kills p0 later in the same source update.
8. Keep trainer-facing code importing only `curvyzero.env`. Do not import source
   runners, scenario schema helpers, toy runners, trace projectors, or diff
   tools from training code.
9. Choose the next narrow source-fidelity slice after preserving the promoted
   PrintManager death-stop batch; do not bundle same-frame order, head-head
   cases, bonuses, replay messages, and observation checks.
10. Keep optimization and vector/backend work deferred until source-fidelity
   fixtures and the single-env interface are stable enough to compare backends.

## Interface Boundary

The public training interface is a separate contract from this fidelity lane.
Use source-fidelity traces to decide what behavior exists; design coach-facing
observations, rewards, wrappers, and batch APIs after those mechanics are named
and verified. Do not let interface convenience hide a source mismatch.
Trainer-facing code should import only `curvyzero.env`; the split source-fidelity
implementation under `curvyzero.fidelity.source_runners` is offline evidence
machinery.
