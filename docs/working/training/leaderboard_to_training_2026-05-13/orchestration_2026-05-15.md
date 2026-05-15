# Orchestration: 2026-05-15

Live-control note: this file is now background detail. For the current control
panel, start with `NOW.md`, `TODO.md`, `TOURNAMENT_DEBUG.md`,
`TRAINING_CONTROL.md`, `FULL_LOOP_PROOF.md`, and `OPERATING_PATTERN.md`.

This doc is the live task board for the main thread plus subagents. Keep it
short, factual, and updated as lanes move.

## Operating Pattern

- Keep the main thread on planning, integration, direct verification, and user
  status.
- Delegate bounded searches and critiques in parallel.
- If a lane blocks, start a smaller honest version of the same proof while the
  fix continues.
- Record useful subagent findings here or in `active_plan_2026-05-15.md`; do
  not leave important facts only in a chat reply.
- Be explicit about what is proven, what is inferred, and what is still
  unproven.

## Current Main-Thread Priorities

| Priority | Main-thread work | Status |
| --- | --- | --- |
| P0 | Make fresh training defaults unambiguous. | Active: `random_per_episode` is the default learner seat mode; old `ego_player_index` config is removed; `fixed_player_0/1` are diagnostic-only. |
| P0 | Separate opponent policy from immortality. | Locally fixed for fresh manifest + stable-slot assignment: public entries use `opponent_immortal`; `opponent_death_mode` is derived only at episode selection/runtime. |
| P0 | Purge invalid v2 namespace before restart. | Done: exact v2 volumes/dicts/queue deleted and verified absent; non-v2 storage remains. |
| P0 | Integrate tournament balanced-seat implementation. | Active: Zeno implemented `seat_order_mode=balanced_random` and focused tests; main must keep docs/tests aligned before any deploy/restart. |
| P0 | Resolve player-perspective validity before a clean real relaunch. | Active: delegated training and tournament seat audits at 09:40 EDT |
| P0 | Quantify current live run and leaderboard state before purge/relaunch. | Active: delegated live metrics inventory at 09:40 EDT |
| P0 | Keep cleanup organized without deleting current evidence. | Active: delegated workspace inventory at 09:40 EDT |
| P0 | Monitor v2 real18 replacement rows and assignment refresh uptake. | Active: 4 original rows applied new tournament-derived hashes; 3 failed rows were relaunched from refreshed manifest after `td_steps` fix |
| P0 | Keep tournament-to-trainer loop honest for real18. | Active: v2 tournament round completed `231` pairs / `4,851` games / `0` failures with `22` active rows; pointers updated; full survival improvement not proven |
| P0 | Isolate tournament worker stall. | Resolved as stale progress read; direct and intake smokes passed |
| P0 | Decide whether tournament evaluation is valid enough to trust ratings. | Active |
| P0 | Patch and redeploy corrected real18 tournament evaluator. | Active: current 67-ref rerate is liveness-only because it used 20ms source ticks instead of trainer/checkpoint 16.6667ms ticks |
| P0 | Keep full-loop proof honest: small loop proven, long behavior proof stopped/blocked. | Superseded by controlrun2 deployed proof |
| P0 | Close the behavior proof while the trainer is still alive. | Passed: controlrun2 applied promoted sha at train iter `1798` |
| P0 | Prove the same storage pattern in v2. | Passed: v2 intake-spawned rating completed, direct rating also completed, and v2 proof3 applied promoted sha `adb04ed3905fb9c8984e5e213a9261079f0e4be188315912d12ae5290d55b770` at train iter `1904` |
| P0 | Prove the recreated all-v2 lane after deleting/recreating v2 objects. | Passed: `curvy-e2e-allv2-canary-20260515a` wrote checkpoints, v2 intake/tournament completed `round-000003` with `18/18` games and `0` failures, promotion wrote sha `0597bceb176580d19d658fd513f752a47a7d4e0f5c9094d5c0f58f60f422c2e0`, and the same trainer applied it at train iter `5061`; latest env telemetry fetch had `1836` provider-ok rows with that sha. |
| P0 | Find a production-quality all-v2 source leaderboard for restart18. | Recommended path: use historical `loop18-main-adaptive417` only to select top active checkpoint refs, copy them into `curvyzero-runs-v2`, then rerate fresh in v2. |
| P0 | Prevent missing-ref launches. | Done locally: `scripts/audit_curvytron_launch_manifest_refs.py` checks manifest checkpoint syntax and local/Modal existence for initial and frozen opponent refs. |
| P0 | Track survival metrics with numbers, not vibes. | Eval summary done; fresh checkpoint/survival audit delegated |
| P0 | Keep docs current while work moves in parallel. | Active |
| P1 | Redeploy/verify Tournament Arena current-marker UI. | Patched locally; not the current blocker |

## Parallel Lanes

| Lane | Owner | Question | Next action |
| --- | --- | --- | --- |
| Tournament worker stall | Confucius + main | Why do `controllong` ratings reach `game_map_started` with zero started pairs? | Resolved: progress snapshot was stale while games were running. Long rating and tiny intake smoke both completed. |
| Training/checkpoint metrics | Boole + main | What checkpoints exist now, is the long trainer stopped, and what survival signal exists? | Boole quantifies current state; main integrates. |
| Controlrun2 proof | Main | Can a fresh checkpoint be rated and promoted back into the same running trainer? | Passed. `iteration_400` was rated, promoted to sha `3ff1af447117e4e90cd1e82277530063d20ba14086d180df5474e7d5309dfa9d`, and applied by the same trainer at train iter `1798`. |
| Eval validity | Meitner, Aristotle, Erdos, main | Do tournament games use the right observations, rules, timing, and action selection? | Merge returned findings into a short validity doc; add parity test if needed. |
| Observation-surface parity | Meitner + main | Did invalidated diagnostic policies trained on `body_circles_fast + simple_symbols` get evaluated on the same surface, and do fresh policies use CPU `cpu_oracle` `browser_lines + simple_symbols`? | P0 follow-up sent. Need verify backend, bonus render mode, and trail render mode. |
| Live artifacts | Carver | What does the current tournament actually run? | Returned: live is eval mode, one-frame, sim8, body-circles policy observations, max_steps currently 8000. |
| Full-loop proof | Wegener | Did checkpoint -> tournament -> leaderboard -> assignment -> trainer actually happen? | Wait for concrete refs/timestamps. |
| V2 intake smoke | Main | Does the v2 subscriber/intake path complete cleanly, not just direct rating? | Passed on recheck: `elo-v2-looplive-proof3-r0-20260515a` is complete, `1` pair / `3` games / `0` failures, `stable=true`. |
| Resume/refresh safety | Ptolemy | Can trainer relaunch safely and can refresh interval move from 50? | Wait for proof before changing interval. |
| Old champion anchors | Wegener | Which old full-sweep tournament had the best checkpoint standings, and what are the top five exact refs? | Reused existing agent because new spawn limit is hit; inject only after refs and target tournament are clear. |
| Render path sanity | Aristotle + main | Is the fresh production surface CPU `cpu_oracle` `browser_lines + simple_symbols`, with GPU rendering lab-only? | Main-thread read: H100 compute is not GPU rendering; body-circles is historical/control only. Aristotle follow-up will verify. |
| Cleanup | Ptolemy | Hide/purge old arenas/apps/artifacts without deleting current live lane or old full-sweep anchor source. | Reused existing agent because new spawn limit is hit. |
| Weak-run immortal bump | Erdos | Which five runs are weak and how do we raise blank/immortal exposure to about 50%? | Follow-up sent. Do not mutate live assignment until exact current mix and update mechanism are known. |
| Contract cleanup | Main later | Remove hidden fallback soup in tournament observation fields. | Track as follow-up after urgent parity fix, redeploy, and re-rating. |
| Corrected v2real18 rerate | Main | Produce final 67-ref tournament evidence. | Patch timing guard + bonus render contract, redeploy, launch `elo-v2real18-rerate67-allpairs-16ms-20260515a`, then publish/materialize recipe assignments only from corrected output. |
| Training seat perspective | Averroes | Does Coach ever train the learner as seat 1, and what minimal fix/test plan is needed if not? | New at 09:40 EDT. Write `player_perspective_audit_2026-05-15.md`. |
| Tournament seat perspective | McClintock | Does tournament eval compare policies under the same POV/action semantics used in training? | New at 09:40 EDT. Write `tournament_eval_seat_perspective_audit_2026-05-15.md`. |
| Live v2real18 inventory | Rawls | What is actually running, how many checkpoints exist, are metrics improving, and what does the leaderboard contain? | New at 09:40 EDT. Write `v2real18_live_metrics_inventory_2026-05-15.md`. |
| Workspace cleanup inventory | Volta | Which apps/arenas/artifacts are necessary and which are cleanup candidates? | New at 09:40 EDT. Write `workspace_cleanup_inventory_2026-05-15.md`. |
| Weak-run immortal intervention | Godel | Was the requested 50% blank/immortal intervention applied, and how can it be applied safely to only weak rows? | New at 09:40 EDT. Write `weak_run_immortal_intervention_2026-05-15.md`. |
| Production source strategy | Rawls | Which source should feed restart18 now that v2 contains only the canary? | Returned: rematerialize top active refs from `loop18-main-adaptive417` into v2, then rerate fresh. Do not copy old leaderboard as truth. |

## Current V2 Real18 Lane

What is true now:

- Tournament `curvy-v2real18-live-20260515a` /
  `elo-v2real18-live-20260515a` completed corrected `round-000001`:
  `231` pairs, `4,851` games, `0` failures.
- Latest snapshot has `22` active rows and `0` provisional rows.
- Three recipe-specific control pointers now point at assignments built from
  that active v2 snapshot:
  `9717c8...`, `e34871...`, and `4db8fe...`.
- First real-batch refresh read showed `4` rows had already applied one of
  those hashes. This proves the real lane is not just static launch wiring.
- Three original rows failed under old trainer code because `source_max_steps`
  was incorrectly copied into LightZero `td_steps`.
- Latest live check at 06:57 EDT:
  `round-000003` has `40` active rows and `0` provisional rows. The tracked
  trainers have already produced `67` discoverable checkpoint refs across
  `20/21` run ids, so the current tournament is behind the training volume.
- `15/21` tracked rows have applied a refreshed assignment at least once. The
  rows without applied refresh are old failed originals or new replacements.
- Replacement `r008` has reached `iteration_10000`, so the replay crash fix is
  plausibly working there; the other replacements still need more runtime.

Immediate order:

1. Submit all `67` discovered exact checkpoint refs to current v2 intake in
   batches of ten.
2. Run `intake-drain` with rating spawn as a detached Modal run so child rating
   work survives the local command.
3. Recheck tournament latest and intake status until the ranked row count grows
   beyond `40` or a concrete failure appears.
4. Recheck the original live rows until more of them apply the refreshed hashes
   or fail clearly.
5. Let replacements reach more checkpoints and confirm intake/tournament admits
   those checkpoints.
6. Do not claim survival improvement until eval summaries or a trustworthy
   telemetry aggregation exists.

## Current P0 Bug: Tournament Observation Mismatch

What was wrong in the invalidated diagnostic lane:

- Trainer fast rows used `body_circles_fast + simple_symbols`.
- Tournament games only carried `policy_trail_render_mode`.
- Tournament stack treated `body_circles_fast` as a canvas/downsample surface and
  used the default browser-sprite bonus rendering.
- So those tournament policy observations may not match what the policies were
  trained on. Ratings from that surface are historical/forensic only.

Fresh restart rule:

- Production policy observations are CPU `cpu_oracle`
  `browser_lines + simple_symbols`.
- GPU `browser_lines + simple_symbols` is lab/profiling-only until
  trainer-visible contract parity passes.
- `body_circles_fast` is historical/control only and should be rejected by the
  trainer-facing source-state env for fresh production launches.

Fix shape:

1. Carry `policy_bonus_render_mode` through checkpoint, pair, game, roster, and
   summary artifacts.
2. Make `SourceStateGray64Stack4` accept `bonus_render_mode`.
3. Use direct fast gray64 for `body_circles_fast` whenever bonus mode is not
   browser sprites, matching the trainer.
4. Add tests that fail if tournament eval sees only trail mode or loses
   `simple_symbols`.
5. Redeploy tournament, start a clean re-rating, and do not call the current
   leaderboard final until this has run.

Cleanup rule after the fire is out:

- New tournament artifacts should have one explicit observation-surface contract.
- Old-name fallbacks are allowed only at the boundary where old artifacts are
  repaired.
- No hidden fallback should silently change the surface a policy sees.

## Weak-Run Immortal Intervention

User request:

- Pick the five current runs whose survival is improving least.
- Measure their current opponent slot probabilities first.
- Change only those five so blank-canvas/no-op plus immortal/invincible
  opponent exposure is roughly 50% overall.
- Keep some leaderboard checkpoint exposure. This is a live experiment, not the
  default for all runs.
- If those five runs get worse, that is acceptable; the point is to learn
  whether high immortal/blank pressure recovers weak survival.

## Old Champion Anchor Plan

Goal: add a few strong prior checkpoint-sweep winners to the current tournament
as anchors.

Steps:

1. Find the most comprehensive prior tournament from the overnight/override
   runs.
2. Read its latest usable leaderboard or standings artifact.
3. Extract the top five exact checkpoint refs, including run/checkpoint ids and
   ranks.
4. Check whether those refs still exist on the relevant Volume.
5. Inject them into the current tournament only after we know the target arena
   and continuation behavior.
6. Monitor acceptance, games scheduled, games completed, and rating placement.

This is a parallel proof lane. It should not distract from fixing the main
`51`-ranked-checkpoints and eval-validity questions.

## Render Path Sanity

Current understanding:

- `gpu-h100-cpu40` is the Modal compute target.
- `body_circles_fast` is a CPU-side source-state observation renderer.
- H100 does not imply GPU rendering.
- The bonus-symbol work is wired as `simple_symbols` in the CPU fast gray64
  path.
- The true GPU renderer exists as standalone probe/benchmark code, not as the
  stock trainer observation backend.

Question to verify: whether any remaining written optimizer handoff still
implies body-circles or GPU rendering as fresh launch guidance. If so, treat the
doc as stale unless a newer parity gate explicitly promoted GPU observations.

## Follow-Up Rhythm

- Poll subagents when their result is needed for the next decision.
- Send follow-ups with concrete missing evidence, not broad “keep looking.”
- Promote a finding to `active_plan_2026-05-15.md` when it changes a launch
  decision.

## Current Separator Test

Direct rating with two exact checkpoint refs, tiny game count, no GIF pressure,
and `--wait` passed. The same tiny shape through intake also passed. The
controlrun2 proof then closed the deployed running-trainer loop.

Interpretation:

- Game workers can run.
- Intake can spawn and finish a tiny rating.
- The long-proof issue was mostly stale progress interpretation plus the trainer
  finishing before we closed the loop again.
- The basic live wiring is now proven in the recreated all-v2 lane. Next:
  quantify survival/progress after refresh, audit the larger manifest for stale
  non-v2 refs, and keep the next real batch on all-v2 storage.
