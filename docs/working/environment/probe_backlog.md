# Probe Backlog

Status: working

This is the next practical probe list for JS/Python environment fidelity. Older
movement, wall, borderless, 3P/4P normal-wall, six-case body-canary, and
eight-case print-manager probes are already covered in the current regression
batches. Keep this list pointed at the next holes.

## Ranked Probes

1. `normal_print_trail_cadence_threshold`
   - Force a prior trail cursor and step just below and just above the avatar
     radius.
   - Status: verified through `source-trail-cadence-canary` in the promoted
     `source_trail_batch.json` trail-cadence claim.
   - Goal: keep this as a regression while gap fixtures are added. The important
     lesson is that the hidden draw cursor and visible `lastTrailPoint` are
     separate source states.

2. `trail_gap_body_absence`
   - Use a real print-manager hole or a forced hole state after cadence is
     pinned.
   - Status: verified through `source-trail-gap-canary` in the promoted
     `source_trail_gap_batch.json` forced trail-gap claim.
   - Goal: keep forced hole-space and empty visual trail state as regressions.

3. `trail_gap_collision_safe`
   - Cross the hole only after body absence is proven.
   - Status: old-body positive control, print-to-hole boundary collision, and
     hole-to-print emitted-body collision are verified through
     `source-trail-gap-canary`.
   - Goal: add multi-step emitted-trail variants only after the current gap
     batch stays green.

4. `print_manager_exact_zero_toggle`
   - Force remaining distance to exactly zero.
   - Status: verified through `source-print-manager-canary` as
     `source_print_manager_exact_zero_toggle_step` in the promoted
     `source_print_manager_batch.json` deterministic PrintManager claim.
   - Goal: keep this as the exact `distance <= 0` threshold regression.

5. `print_manager_delayed_start`
   - Exercise the 3000 ms scheduled `start()` path separately from forced active
     manager state.
   - Status: verified through `source-print-manager-canary` as
     `source_print_manager_delayed_start_timer_step` in the promoted
     `source_print_manager_batch.json` deterministic PrintManager claim.
   - Goal: keep start-time side effects separate from trail gaps.

6. `print_manager_body_collision_death`
   - Status: verified through `source-print-manager-canary` as
     `source_print_manager_body_collision_stop_on_death_step` in the promoted
     `source_print_manager_batch.json` deterministic PrintManager claim.
   - Goal: keep as the seeded-body death-stop regression. Add a natural
     emitted-body version only if it isolates a new rule.

7. `multi_step_straight_motion`
   - Add a two-player scenario with several straight steps.
   - Status: verified through `source_kinematics_straight_multistep.json` in
     the promoted `source_kinematics_batch.json` movement claim.
   - Goal: keep this as the fixed-60Hz straight movement regression.

8. `multi_step_turn_motion`
   - Add a two-player scenario with left, right, and straight moves across
     several ticks.
   - Status: verified through `source_kinematics_turn_multistep.json` in
     the promoted `source_kinematics_batch.json` movement claim.
   - Goal: keep this as the fixed-60Hz turn movement regression. The remaining
     Python movement gap is bonus-modified movement.

9. `multi_step_varied_elapsed_same_total`
   - Add a two-player varied elapsed-ms trace with the same total elapsed time
     as the fixed four-step controls.
   - Status: promoted through the `source_kinematics_batch.json` movement
     claim.
   - Goal: keep this as the measured elapsed-ms regression. Keep `0.000001`
     movement tolerance explicit.

10. `head_to_head_same_tick_death`
   - Force two players into a same-tick collision.
   - Status: death-point side-effect and head-head/order fixtures are
     Python/common-trace promoted in `source_collision_order_batch.json`.
     The head-head fixture is
     `source_collision_head_head_reverse_order_single_death_step.json` and is
     documented in
     [collision_order_probe_plan.md](collision_order_probe_plan.md).
   - Goal: keep the promoted collision-order batch green. Add 3P order stress
     only when it isolates a new source rule.

11. `borderless_body_wrap_corner`
   - Force a borderless wrap near an existing body.
   - Status: destination-body skip is Python/common-trace verified as
     `source_borderless_wrap_skips_destination_body_then_next_frame_kills`; the
     first PrintManager/trail wrap fixture is also verified in
     `source_border_batch.json`. Exact-edge/corner-axis behavior is now verified
     as `source_borderless_exact_edge_corner_axis_step`.
   - Goal: keep the promoted border batch green. Add next-frame second-axis
     wrap only if it isolates a new source rule.

12. `round_lifecycle_messages`
    - Start from a forced near-terminal setup.
    - Goal: check `score`, `score:round`, `round:end`, winner fields, and later
      server-message payloads.

## Follow Later

- Bonus-modified movement, then bonus spawn and pickup: source-read plan in [randomness_bonus_probe_plan.md](randomness_bonus_probe_plan.md); first fixture is `source_bonus_forced_catch_self_small_step`.
- Three-player and four-player random spawn checks.
- Observation-perspective probes.
- Browser replay and screenshot probes.
- Modal scenario batches.
- Benchmark manifest and split timers for the future vectorization lane.

## Notes

- Prefer shared scenario fixtures over new one-off JS scripts.
- Keep every probe small and deterministic.
- Prove cadence before gap crossing. A visual gap is not the same as deleting
  old world bodies.
- Store the first mismatch in JSON before adding richer reports.
- Common-trace mode should be the normal comparison path for these probes.
- Mirror promoted probe status in
  [coverage_tracker.md](coverage_tracker.md) before moving to the next slice.
