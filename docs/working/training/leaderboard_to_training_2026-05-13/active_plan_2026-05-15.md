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

## Current Artifacts

| Item | Current value |
| --- | --- |
| Training app | `curvyzero-lightzero-curvytron-visual-survival-train-v2` |
| Tournament app | `curvyzero-checkpoint-tournament-v2` |
| Current tournament | `curvy-v2refresh18p-live-20260514b` |
| Current rating run | `elo-v2refresh18p-live-20260514b` |
| Current batch prefix | `curvy-v2refresh18p-` |
| Current local manifest | `artifacts/local/curvytron_tonight18_manifests/curvy-v2refresh18p-20260514b/curvy-v2refresh18p-20260514b.json` |
| Manifest builder | `scripts/build_curvytron_tonight18_manifest.py` |
| Main status doc | `current_state.md` |
| Orchestration doc | `orchestration_2026-05-15.md` |

## Facts Right Now

- Tournament and trainer apps were redeployed after raising explicit source-step
  caps to `1_048_576`.
- The already-running jobs do not automatically inherit that new cap. A clean
  relaunch/resume must come from a regenerated manifest.
- The local `curvy-v2refresh18p-20260514b` manifest was stale and recorded
  `source_max_steps=65_536`. It has now been regenerated with
  `source_max_steps=1_048_576`, `background_eval_max_steps=1_048_576`, and
  refresh interval `50` for all 18 rows.
- The trainer refresh interval has not been changed to `1000` or `2000`.
  Changing it is blocked on resume-safety evidence.
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
  observation rendering. The current `v2refresh18p` manifest uses H100 compute
  plus CPU-side `body_circles_fast + simple_symbols` observations. This matches
  the latest optimizer Coach handoff, which explicitly says the faithful GPU
  renderer is still a prototype and not wired into stock training. If the user
  wants a true GPU-render training lane, that is an implementation gap and a
  launch-critical decision.
- P0 tournament parity bug found locally: tournament eval carried
  `policy_trail_render_mode` but did not carry `policy_bonus_render_mode`.
  Worse, `SourceStateGray64Stack4(trail_render_mode="body_circles_fast")`
  used the canvas/downsample path with default browser-sprite bonuses, while
  the trainer's current fast lane uses direct gray64 with `simple_symbols`.
  Current tournament ratings are therefore suspect for the current H100 fast
  training rows until patched, redeployed, and re-rated.
- Local patch in progress: add `policy_bonus_render_mode` to checkpoint specs,
  pair/game specs, rating roster/context, policy loader metadata, and tournament
  game summaries; update `SourceStateGray64Stack4` so
  `body_circles_fast + simple_symbols` uses the same direct fast gray64 path as
  training; add focused regression tests before redeploy.
- New side experiment requested: identify the five weakest current training
  runs by survival progress and, if assignment refresh supports it safely, bump
  their total blank/immortal opponent exposure to about 50%. This is useful
  exploration but must not block the tournament parity fix.
- Specific weak-run slot request: for the five runs that are not improving much,
  first measure their current slot probabilities, then aggressively raise total
  exposure to invincible/immortal opponents to roughly half of samples. The
  half can be split between blank-canvas/no-op and actual frozen opponents with
  immortal death mode. Do not make every opponent immortal; leaderboard
  checkpoints should still be the majority signal for the broader batch. This
  is intentionally a wonky live intervention: if those five runs get worse, that
  is acceptable signal.
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
| Do tournament eval observations match training observations? | Main + Meitner | P0. Specifically verify `body_circles_fast + simple_symbols` parity, including bonus render mode. A matching trail mode alone is not enough. |
| Is relaunch/resume safe if we change cap and refresh interval? | Ptolemy | What state resumes, what resets, and what artifacts prove it. |
| Should refresh interval move from `50` to `1000`/`2000`? | Main after Ptolemy | Resume safety plus rough overhead evidence. |
| How should tournament GIFs stay safe with `1_048_576` max steps? | Main + Carver | Explicit frame sampling policy that still shows useful games. |
| Does the current Tournament Arena dropdown visibly say current? | Main | Fresh page HTML after deploy, not just API JSON. |
| Why are only 51 checkpoints ranked? | Main + delegated audit | Current answer: latest reduced round has 51; running round has 193 checkpoints / 18,528 pairs. Need decide whether to kill/replace that giant round. |
| Should old sweep winners be injected as anchors? | New old-champion lane | Find the best prior full-sweep tournament, extract top five exact checkpoint refs, inject them, and monitor acceptance/rating. |
| Did we accidentally expect GPU rendering from H100? | Main + render audit | Current read from optimizer handoff: no. H100 accelerates model/search/training; `body_circles_fast + simple_symbols` is CPU fast render. Need decide whether that is acceptable or whether true GPU render is now required. |

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
| P1 | Confirm body-circles/H100/GPU-render distinction and decide if any manifest change is needed. | Current manifest is CPU fast render on H100. True GPU render is not wired; treat as a decision, not an assumption. |
| P1 | Decide explicit GIF safety policy for huge tournament max steps. | Pending |
| P1 | Refresh interval decision: keep `50` until resume-safety audit returns. | Pending |
| P2 | Cleanup old arenas/apps after the current lane is stable. | Pending |
| P2 | Bump immortal/blank exposure for five weak runs to about 50% if live assignment update is safe. | Delegated |
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
   mode and bonus render mode. For current runs that means
   `body_circles_fast + simple_symbols`, not just `body_circles_fast`.

Parallel rule: if one gate is slow, start a smaller honest loop that tests the
same contract rather than waiting passively.
