# Active Plan Snapshot: 2026-05-15

Live-control note: this file is historical background detail, not the current
plan. It contains v2 and `body_circles_fast` statements from the invalidated
run. For the current control panel, start with `NOW.md`, `TODO.md`,
`TOURNAMENT_DEBUG.md`, `TRAINING_CONTROL.md`, `FULL_LOOP_PROOF.md`, and
`OPERATING_PATTERN.md`.

Do not use the artifact table below as current launch guidance.

## North Star

Prove the full feedback loop with real training jobs:

```text
trainer writes checkpoint
-> subscriber sees checkpoint
-> tournament rates checkpoint
-> public leaderboard updates
-> Coach materializes assignment
-> trainer refresh consumes assignment as frozen opponents
-> survival metrics improve or clearly fail
```

Do not treat a partial loop as proof of the whole loop.

## Superseded Artifact Table

These rows describe the invalidated `v2refresh18p` lane. They are not current
restart guidance.

| Item | Historical value |
| --- | --- |
| Training app | `curvyzero-lightzero-curvytron-visual-survival-train-v2` |
| Tournament app | `curvyzero-checkpoint-tournament-v2` |
| Historical tournament | `curvy-v2refresh18p-live-20260514b` |
| Historical rating run | `elo-v2refresh18p-live-20260514b` |
| Historical batch prefix | `curvy-v2refresh18p-` |
| Historical local manifest | `artifacts/local/curvytron_tonight18_manifests/curvy-v2refresh18p-20260514b/curvy-v2refresh18p-20260514b.json` |
| Manifest builder | `scripts/build_curvytron_tonight18_manifest.py` |
| Main status doc | `current_state.md` |
| Orchestration doc | `orchestration_2026-05-15.md` |

Current ranked-source status lives in `TOURNAMENT_DEBUG.md`: the 100-ref
rerate `curvy-restart18-source-rerate-20260515a` is diagnostic only, and the
96-ref nonzero rerate `curvy-restart18-source-rerate-nonzero-20260515a` is only
a candidate if we want leaderboard-derived top slots. It is not a bootstrap
launch blocker.

Plain correction: do not wait for a perfect starting leaderboard before
launching bootstrap training. Use exact old checkpoint refs, immortal
blank/hard-coded pressure, and live tournament intake. The tournament rankings
can improve while training runs.

## Facts Right Now

- Tournament and trainer apps were redeployed after raising explicit source-step
  caps to `1_048_576`.
- The already-running jobs do not automatically inherit that new cap. A clean
  relaunch/resume must come from a regenerated manifest.
- Historical `curvy-v2refresh18p-20260514b` manifest was stale and recorded
  `source_max_steps=65_536`. It has now been regenerated with
  `source_max_steps=1_048_576`, `background_eval_max_steps=1_048_576`, and
  the old 50-iteration refresh cadence for all 18 rows.
- Restart18 builder/shared-contract default is now refresh interval `2000`.
  The `50` interval belongs to invalidated v2real18/v2refresh18p history.
- Tournament GIFs sampled from the current live arena are not frame-truncated:
  sampled games had `frame_count = physical_steps + 1`, `frame_stride=1`, and
  `duration_ms_per_frame=12`.
- The short-looking GIF problem is probably playback speed: 100-180 frames at
  80 fps plays in roughly 1-2 seconds.
- Raising tournament `max_steps` while saving every frame can explode memory for
  very long sampled games. This needs an explicit GIF sampling/stride/cap policy,
  not hidden "infinite" behavior.
- The Tournament Arena API marks `curvy-v2refresh18p-live-20260514b` as current.
  The page default selects it. The UI now has a dedicated current-arena banner
  patch locally; redeploy and verify the page HTML before calling this fixed.
- Current tournament rankings show only the latest reduced rating snapshot. The
  user sees `51` ranked checkpoints right now, which is probably not the full
  intake universe. Need determine whether this is because newer checkpoints are
  queued/running/unreduced, because active-pool scheduling excludes them, or
  because intake is stuck.
- Eval survival summary was pulled directly for all 18 `v2refresh18p` rows:
  all rows are running; latest checkpoints are around `iteration_70000` to
  `iteration_130000`; every row has a best eval mean above its first eval mean,
  but latest eval means are noisy and often below best. Latest means in the pull
  range roughly `95` to `209.5` source steps.
- Quantified eval survival summary:

  | Group | Rows | First mean | Best mean | Latest mean | Latest - first | Best - first | Latest up |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | All rows | 18 | 131.2 | 217.7 | 164.0 | +32.7 | +86.4 | 13/18 |
  | Outcome only | 6 | 111.7 | 213.0 | 167.1 | +55.5 | +101.3 | 6/6 |
  | Survival + bonus | 6 | 154.3 | 200.8 | 141.5 | -12.8 | +46.4 | 2/6 |
  | Survival + bonus + outcome | 6 | 127.8 | 239.3 | 183.3 | +55.5 | +111.5 | 5/6 |
  | Clean | 9 | 128.8 | 236.3 | 169.5 | +40.7 | +107.5 | 7/9 |
  | 10% noise/skip | 9 | 133.8 | 199.1 | 158.5 | +24.7 | +65.3 | 6/9 |

  Meaning: best-seen survival is clearly up; latest survival is up overall but
  noisy. `survival_plus_bonus_no_outcome` is the weak group on latest eval,
  despite having improved best checkpoints. Outcome-only and
  survival+bonus+outcome look stronger by latest eval right now.
- The Tournament Arena progress API explains the `51` visible ranking count:
  `latest.json` is still `round-000004`, while `round-000005` is running with
  `pair_count=18528` and `game_count=389088`. That pair count is exactly
  unordered all-pairs over 193 checkpoints (`193*192/2`). So more checkpoints
  exist, but they are in a giant unreduced round. This is not a good operator
  experience and may be a scheduling bug or active-pool policy miss.
- Render-path correction: `compute=gpu-h100-cpu40` does not mean GPU
  observation rendering. The invalidated `v2refresh18p` manifest used H100
  compute plus CPU-side `body_circles_fast + simple_symbols` observations.
  Fresh production guidance is CPU `cpu_oracle`
  `browser_lines + simple_symbols`; faithful GPU rendering is still
  lab/profiling-only until trainer-visible contract parity passes.
- P0 tournament parity bug found locally: tournament eval carried
  `policy_trail_render_mode` but did not carry `policy_bonus_render_mode`.
  Worse, `SourceStateGray64Stack4(trail_render_mode="body_circles_fast")`
  used the canvas/downsample path with default browser-sprite bonuses, while
  the trainer's historical fast lane used direct gray64 with `simple_symbols`.
  Those tournament ratings are therefore suspect for the historical H100 fast
  training rows until patched, redeployed, and re-rated.
- Local patch in progress: add `policy_bonus_render_mode` to checkpoint specs,
  pair/game specs, rating roster/context, policy loader metadata, and tournament
  game summaries; update `SourceStateGray64Stack4` so
  `body_circles_fast + simple_symbols` uses the same direct fast gray64 path as
  training; add focused regression tests before redeploy.
- Weak-run live intervention is dropped for the current invalidated rows. The
  lesson goes into the next manifest instead: blank and hard-coded sentinel
  opponents are immortal; frozen checkpoint slots are mostly mortal, with small
  explicit immortal slices; keep total immortal exposure around `20-30%` and
  generally not above about `30%`.
- Code-quality note from the tournament patch: the current code has too many
  implicit fallback names (`policy_*`, `observation_*`, `source_state_*`,
  generic `trail_render_mode` / `bonus_render_mode`). Use fallbacks only to
  repair old artifacts; after the urgent patch, refactor this into one clean
  explicit observation-surface contract so new artifacts cannot silently omit a
  field.
- New parallel lane: find the strongest old checkpoint-sweep winners from the
  most comprehensive overnight/override tournament, select roughly the top five
  exact checkpoint refs, inject them into the current tournament, and monitor
  whether they are accepted, rated, and placed correctly. This is a small
  champion-anchor test, not a replacement for the full loop.

## Open Questions

| Question | Owner | Needed evidence |
| --- | --- | --- |
| Is survival actually improving since new checkpoints came in? | Main + Ptolemy | Eval says best-seen survival improves in every row; latest is noisy. Still need collector summary. |
| Has the full loop happened at least once? | Wegener | Concrete refs/timestamps for every handoff in the loop. |
| Is tournament evaluation valid? | Main + subagents | Compare tournament observation stack, one-frame timing, game rules, checkpoint loading, MCTS/eval action selection, and training collect action selection. Decide whether ranking should be greedy/eval or noisy/collect. |
| Do tournament eval observations match training observations? | Main + Meitner | P0. Fresh production must verify CPU `cpu_oracle` `browser_lines + simple_symbols`; historical rerates may still need `body_circles_fast + simple_symbols` parity for forensic reads. A matching trail mode alone is not enough. |
| Is relaunch/resume safe if we change cap and refresh interval? | Ptolemy | What state resumes, what resets, and what artifacts prove it. |
| Should refresh interval move away from the current `2000` default? | Main after Ptolemy | Resume safety plus rough overhead evidence. |
| How should tournament GIFs stay safe with `1_048_576` max steps? | Main + Carver | Explicit frame sampling policy that still shows useful games. |
| Does the current Tournament Arena dropdown visibly say current? | Main | Fresh page HTML after deploy, not just API JSON. |
| Why are only 51 checkpoints ranked? | Main + delegated audit | Current answer: latest reduced round has 51; running round has 193 checkpoints / 18,528 pairs. Need decide whether to kill/replace that giant round. |
| Should old sweep winners be injected as anchors? | New old-champion lane | Find the best prior full-sweep tournament, extract top five exact checkpoint refs, inject them, and monitor acceptance/rating. |
| Did we accidentally expect GPU rendering from H100? | Main + render audit | Current read from optimizer handoff: no. H100 accelerates model/search/training; `body_circles_fast + simple_symbols` was CPU fast render in the invalidated diagnostic lane. Fresh production should use CPU `cpu_oracle` `browser_lines + simple_symbols`; GPU rendering is lab/profiling-only until trainer-visible parity passes. |

## Immediate To-Do

| Priority | Task | Status |
| --- | --- | --- |
| P0 | Regenerate `curvy-v2refresh18p-20260514b` manifest with `1_048_576` cap. | Done |
| P0 | Confirm regenerated manifest changes only intended fields. | Partly done: cap/refresh/reward checks passed; still need optional diff review. |
| P0 | Pull survival metrics directly; do not wait only on subagents. | Done for eval summary; collector summary pending. |
| P0 | Get full-loop proof or identify exact missing link. | Delegated |
| P0 | Explain and fix/label why leaderboard only ranks 51 checkpoints. | Explained; fix/label pending. |
| P0 | Validate tournament evaluation semantics before trusting rankings. | In progress |
| P0 | Verify training/tournament observation-surface parity for current checkpoints. | Follow-up sent; do not assume until checked |
| P0 | Patch tournament render-surface parity bug and add tests. | Active |
| P0 | Redeploy tournament and force a clean re-rating after parity patch. | Pending patch/tests |
| P0 | Start old champion-anchor lane: find prior strongest checkpoint refs and plan injection. | Added now; delegated next |
| P0 | Decide whether to relaunch all 18 rows or a smaller honest fallback first. | Pending |
| P1 | Fix visible Tournament Arena current-label if it is still absent. | Patched locally; redeploy/verify pending. |
| P1 | Confirm body-circles/H100/GPU-render distinction and decide if any manifest change is needed. | Historical manifest was CPU fast render on H100. Fresh manifests should target CPU `cpu_oracle` `browser_lines + simple_symbols`; true GPU render is not a production backend yet. |
| P1 | Decide explicit GIF safety policy for huge tournament max steps. | Pending |
| P1 | Refresh interval decision: keep `50` until resume-safety audit returns. | Pending |
| P2 | Cleanup old arenas/apps after the current lane is stable. | Pending |
| P2 | Keep next-manifest immortal exposure simple: blank/hard-coded immortal, small frozen immortal slices, total around `20-30%`. | Locally implemented in builder; keep testing |
| P2 | Refactor tournament observation-surface field fallbacks into a clean explicit contract. | After urgent patch/redeploy |

## Delegation State

| Agent | Lane | Status |
| --- | --- | --- |
| Ptolemy | Survival metrics and trainer resume/refresh safety. | Running |
| Carver | Tournament GIF/frame audit and huge-cap risk. | Returned: GIFs not truncated; memory risk if very long GIFs save every frame. |
| Meitner | Tournament-vs-training observation/rules/policy parity. | Running |
| Wegener | Full-loop validation evidence. | Running |
| Aristotle | Red-team current recovery plan plus eval-validity risks. | Running |
| Erdos | Manifest regeneration correctness. | Running |
| Wegener | Find strongest prior full-sweep tournament and top checkpoint refs. | Follow-up sent because new-agent limit is hit |
| Aristotle | Confirm whether current runs use CPU body-circles, browser, or any GPU render path. | Follow-up sent; main-thread read says CPU fast render, not GPU render. |
| Ptolemy | Cleanup old arenas/apps/artifacts without deleting current lane or anchor source. | Follow-up sent because new-agent limit is hit |

## Launch Gate

Before any clean relaunch:

1. Manifest says `source_max_steps=1_048_576` for every row.
2. Poller/background eval says `background_eval_max_steps=1_048_576`.
3. Assignment source and initial champion checkpoint are unchanged unless
   deliberately changed.
4. Refresh interval is either deliberately kept at `50` or changed with resume
   evidence in this doc.
5. Tournament/training apps are deployed.
6. If using huge tournament max steps with GIFs on, the GIF frame policy is
   explicit.
7. Tournament policy observations include both the checkpoint's trail render
   mode and bonus render mode. For fresh production runs that means CPU
   `cpu_oracle` `browser_lines + simple_symbols`.

Parallel rule: if one gate is slow, start a smaller honest loop that tests the
same contract rather than waiting passively.
