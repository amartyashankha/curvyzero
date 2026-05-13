# Opponent Mixture Batch Plan - 2026-05-13

Purpose: track the current CurvyTron opponent-mixture plan after the 300-row
survival diagnostic became healthy background training.

This lane trains one stock LightZero ego policy against an episode-level
mixture of opponent sources, with a large fraction of recent frozen checkpoints.

## Current State

The trusted stock path now supports weighted episode-level opponent mixtures.
The selected opponent source is sampled once per reset, passed through training,
and recorded in eval/GIF summaries.

Current live surfaces:

| Surface | Status | Current use |
| --- | --- | --- |
| `curvy-survive-bonus-large-20260513b` | healthy 300-row background batch | source of current frozen checkpoints |
| `curvy-mix2-clean-20260513a` | healthy-ish 156-row mixture batch | cadence/GIF/readout lane; wait for more eval coverage before ranking recipes |
| `curvy-mix3-currentckpt-20260513a` | newly launched 300-row mixture batch | next wave using recent/mid/old checkpoints from 300b |

Historical canary failures:

- `curvy-mix-recent-canary-20260513a` failed because the deployed train function
  did not accept `opponent_mixture_spec`.
- `curvy-mix2-canary-20260513a` failed because command metadata said
  `learner_vs_fixed_straight` while env config said
  `learner_vs_weighted_episode_opponent_mixture`.
- `curvy-mix2-canary-20260513b` proved the corrected relation path and reached
  `k10` checkpoints with selected mixture component fields in GIF summaries.

Do not relaunch the stale canaries. Use `curvy-mix2-clean-20260513a` and
`curvy-mix3-currentckpt-20260513a` for current monitoring.

## Plain Goal

Keep the trainer close to stock LightZero and make the environment choose a
different opponent source at reset time.

The opponent choice must be:

- per episode/reset, not per step;
- logged in telemetry and eval/GIF summaries;
- strict, with no silent fallback if a checkpoint or opponent kind is missing;
- reproducible from run seed plus reset index;
- static for a given training run, meaning frozen checkpoint refs are immutable
  `iteration_N.pth.tar` files, not `latest` pointers.

This is close to self-play in the practical curriculum sense because about half
of the opponent episodes use a very recent frozen checkpoint, but it is not live
same-policy self-play.

## Base Settings

Use a compact base grid so the mixture recipe remains the main variable while
we still learn whether the faster visual path is good enough.

| Setting | Value |
| --- | --- |
| Trainer | stock LightZero `train_muzero` |
| Mode | `train` |
| Env | `source_state_fixed_opponent` with opponent mixture enabled |
| Reward | `survival_plus_bonus_no_outcome` |
| Outcome reward | off; telemetry only |
| Episode cap | `source_max_steps=65536` |
| Compute | `gpu-l4-t4-cpu40` |
| Render | paired `body_circles_fast` and `browser_lines` for core rows |
| Search | `num_simulations=8` |
| Collectors | `collector_env_num=32`, `n_episode=32` |
| Learner batch | `batch_size=32` |
| Stochasticity | medium held-action repeat: min/max `1/3`, extra probability `0.20` |
| Eval/GIF | CurvyZero background eval/GIF on |
| Stock LightZero eval | off, `lightzero_eval_freq=0` |
| Checkpoint cadence | `save_ckpt_after_iter=10000` working default |
| Train cap | `max_train_iter=300000`, `max_env_step=30000000` |

Do not sweep reward or episode cap in this batch. Render fidelity is now a real
axis because the current readout still has uncertainty about fast-vs-browser
behavior. Keep search, collector count, learner batch, and stochasticity small
and named; do not let them dominate the mixture question.

Why this base grid is small:

- the active question is "which opponent mixture creates useful survival
  pressure?";
- changing search, batch, render, reward, and stochasticity freely would make
  the result hard to read, so only use a compact set of baseline settings;
- the 300-row rescue batch is already covering the broad blank-canvas
  diagnostic axis, so the mixture batch should not repeat that whole tensor;
- recent-checkpoint pressure is the point of this batch, so about half of the
  episodes should use `recent` in the main recipes.

If the base canary fails, fix the base before scaling the mixture batch. Do not
compensate by launching more recipes.

## Current Critique

The 228-row `curvy-mix2` artifact is useful as a review manifest, but it should
not automatically become the launch set. The canary has not yet proved that the
deployed trainer reaches `train_muzero` with mixture metadata intact.

The 156-row `curvy-mix2-clean-20260513a` shape is the recommended first full
launch after the canary clears. It drops the sim16/C64/B64 sentinels and also
drops passive rows. The older 180-row `curvy-mix2-core-20260513a` shape remains
a review ceiling, not the first launch set.

Baseline knob decision:

| Area | Decision for first full launch | Reason |
| --- | --- | --- |
| Trainer path | Keep stock LightZero `--mode train` only. | This is the trusted path. |
| Reward | Keep `survival_plus_bonus_no_outcome`. | Outcome reward would confuse this diagnostic. |
| Episode cap | Keep `source_max_steps=65536`; do not sweep. | Long survival should be allowed to show up. |
| Checkpoint cadence | Keep `k10`, meaning `save_ckpt_after_iter=10000`. | It should give visible checkpoints without flooding eval/GIF jobs. |
| Core compute | Keep L4/T4, sim8, C32, B32. | It is the simplest readable baseline. |
| Render | Keep paired `body_circles_fast` and `browser_lines` for core rows. | We still need a fidelity check. |
| Held-action repeat | Keep `rep0`, `repM`, and `repH` as the only small robustness axis. | This gives no/medium/high stochasticity without adding more knobs. |
| Heavy sentinels | Hold sim16, C64, and B64 out of the first full launch unless canaries are clean. | They answer speed/scale questions, not the main mixture question. |
| Background artifacts | Keep CurvyZero checkpoint eval and GIF on. | The website and curve tooling depend on these artifacts. |
| Stock LightZero eval | Keep off with `lightzero_eval_freq=0`. | Our eval/GIF path is the readout. |
| Submission style | Keep grouped submitter into one deployed app. | Avoid app sprawl and preserve poller-then-train spawning. |

Recipe decision:

| Recipe | Decision | Reason |
| --- | --- | --- |
| `r50-blank50` | Keep. | Clean survival baseline with recent pressure. |
| `r50-mid50` | Keep. | Simple curriculum comparison against recent. |
| `r50-old50` | Keep. | Checks whether weaker frozen opponents help early learning. |
| `r50-scr50` | Keep only after scripted canary records the scripted component. | Useful trail-maker, but must be proven in remote artifacts. |
| `r50-pass50` | Drop from first launch. | Passive immortal behavior is artificial; keep it only as canary/dirty-control evidence. |
| `r50-blank25-scr25` | Keep. | Good mixed clean/scripted pressure if scripted canary passes. |
| `r50-mid25-old25` | Keep. | Compact frozen-checkpoint curriculum mix. |
| `r50-blank20-mid15-scr15` | Keep after scripted canary. | Broad but still readable. |
| `recent100` | Keep as control. | Shows whether recent-only is too hard or too narrow. |
| `blank100` | Keep as control. | Pure wall-survival baseline. |
| `mid100`, `old100` | Keep small. | Helps interpret frozen checkpoint strength. |
| `scr100` | Keep only after scripted canary. | Isolates scripted opponent effect. |
| `pass100` | Drop from first launch. | Dirty control can pollute interpretation if mixed into the first ranking. |
| Extra passive-heavy mixtures | Remove for now. | They add dirty behavior before the clean question is answered. |
| Extra 4-5 component recipes | Remove for now. | They make attribution harder. |
| New recipes | Add none before canary. | The current core is already enough; do not expand while relation gate is unproven. |

Recommended first full launch after clean canaries:
`curvy-mix2-clean-20260513a`. That means 7 non-passive main recipes across the
6 core base profiles with 3 seeds each, plus 5 non-passive controls across the
same core base profiles with 1 seed each. Hold passive rows and sim16/C64/B64
sentinel rows for later.

This keeps the main learning question intact: about half the episodes in the
main recipes use a recent frozen checkpoint, while the other half tests one
clear pressure source at a time. Passive rows stay out of the first full launch.

## Final Launch Shape After Canary Clears

This is the already-launched mix2 clean shape. Keep it here as the baseline for
reading mix2 and comparing it with mix3.

| Piece | Exact choice | Decision |
| --- | --- | --- |
| Manifest scope | `--batch-scope core` | Use this, not the builder default. |
| Row count | 156 rows | 126 main recipe rows plus 30 control rows. |
| Render modes | paired `body_circles_fast` and `browser_lines` | Keep both for every recipe/base row. Fast is the speed lane; browser is the higher-fidelity anchor. |
| Baseline compute | L4/T4, sim8, C32, B32 | Keep this fixed for the first large mixture batch. |
| Repeat settings | `rep0`, `repM`, `repH` | This is the only robustness axis in the first large launch. |
| Checkpoint cadence | `save_ckpt_after_iter=10000` | Use one cadence for fast and browser rows. |
| Canary cadence | `save_ckpt_after_iter=5000` | Only for tiny canaries. |
| Passive rows | exclude `r50-pass50` and `pass100` | Keep passive only as canary/dirty-control evidence. |
| Heavy sentinels | hold sim16/C64/B64 | Do not launch them in the first full mixture batch. |

Concrete generator shape:

```bash
uv run python scripts/build_curvytron_opponent_mixture_manifest.py \
  --batch-scope core \
  --matrix-name curvy-mix2-clean-20260513a \
  --run-prefix curvy-mix2clean \
  --attempt-prefix try-mix2clean \
  --recipe-id r50-blank50 \
  --recipe-id r50-mid50 \
  --recipe-id r50-old50 \
  --recipe-id r50-scr50 \
  --recipe-id r50-blank25-scr25 \
  --recipe-id r50-mid25-old25 \
  --recipe-id r50-blank20-mid15-scr15 \
  --recipe-id recent100 \
  --recipe-id mid100 \
  --recipe-id old100 \
  --recipe-id blank100 \
  --recipe-id scr100
```

Do a manifest dry-run/readback before submit. The builder currently defaults to
`full`, which is the 228-row reserve plan; pass `--batch-scope core` and the
recipe allow-list explicitly.

## Current Launch Readout

`curvy-mix2-clean-20260513a` was launched at 2026-05-13 07:49 EDT. The first
full status sweep is a startup read, not a learning read and not yet a render
speed read.

| Readout | Count |
| --- | ---: |
| Rows in launch artifact | 156 |
| Rows with train call IDs | 156 |
| Rows with poller call IDs | 156 |
| Rows with `iteration_0` | 56 |
| Rows with trainer heartbeat | 83 |
| Rows with running poller | 128 |
| Rows with train root absent | 17 |

Render split at that moment:

| Render | Rows | `iteration_0` rows | Trainer heartbeat rows |
| --- | ---: | ---: | ---: |
| `body_circles_fast` | 78 | 34 | 47 |
| `browser_lines` | 78 | 22 | 36 |

Do not overread this split. The manifest order places fast rows before browser
rows inside each recipe, so early startup status is confounded by launch order.
The useful comparison is the time to the first `k10` checkpoint among matched
rows once both rows have actually started.

Later JSON status read:

| Readout | Count |
| --- | ---: |
| Rows at `iteration_10000` | 25 |
| Rows at `iteration_0` only | 76 |
| Rows without a checkpoint | 55 |
| Matched fast/browser pairs at `iteration_10000` | 6 |

Matched-pair render deltas so far are small and noisy, not a browser slowdown
smoking gun. In the six paired rows, median browser-minus-fast trainer elapsed
time was about `-70` seconds. Treat this as "not enough evidence that render is
the main blocker", not as "browser is faster." The sample is still too small.

Later matched-pair read:

| Readout | Count |
| --- | ---: |
| Rows at `iteration_10000` | 49 |
| Matched fast/browser pairs at `iteration_10000` | 15 |
| Median browser-minus-fast elapsed time | `+11` sec |
| Mean browser-minus-fast elapsed time | `-24` sec |

This still does not make render fidelity irrelevant, because learning quality
may differ by visual surface. But for checkpoint cadence, the current evidence
does not support "browser rows are the main reason checkpoints are slow." The
next batch should keep render pairing for learning/fidelity, not because
browser cadence is clearly broken.

Superseded elapsed read:

The first matched analyzer used latest `progress_latest.elapsed_sec`, which can
drift after a row passes the target checkpoint. It has been replaced with a
checkpoint-mtime read.

Corrected `k0 -> k10` checkpoint gap read:

| Readout | Value |
| --- | ---: |
| Rows at `iteration_10000` | 89 |
| Matched fast/browser pairs at `iteration_10000` | 38 |
| Fast median gap | `1285` sec |
| Browser median gap | `1395` sec |
| Median browser-minus-fast gap | `117` sec |

Browser is modestly slower, but not enough to explain missing checkpoints by
itself. Keep paired renders for learning/fidelity. Do not split checkpoint
cadence by render unless a later read shows a much larger gap.

Implication for the next batch: keep fast/browser pairing. If we want startup
or checkpoint timing to answer a fidelity question, generate the next manifest
with render rows interleaved or randomized instead of grouped fast-first.

## Mix3 Refinement Question

The 156-row mix2 clean launch was the first clean mixture wave, not the final
experiment design. It produced enough cadence/GIF/startup evidence to launch
`curvy-mix3-currentckpt-20260513a`. The next readout should use both mix2 and
mix3 results to settle:

- which baseline knob set is the anchor, rather than copying one baseline
  everywhere;
- how much of the grid stays paired across `body_circles_fast` and
  `browser_lines`;
- whether `rep0`, `repM`, and `repH` all stay, or whether one repeat level is
  clearly poor;
- which mixture recipes deserve more seeds;
- whether the main recipes should keep recent frozen checkpoint pressure near
  50% or move that fraction.

Now that mix2 has many `k10` checkpoints, the default stance is still
conservative: keep paired renders and do not choose future baselines from
`iteration_0` timing alone. Mix3 alternates fast/browser launch lead to reduce
that startup-order confusion.

## Mix3 Current-Checkpoint Matrix

This shape has launched as `curvy-mix3-currentckpt-20260513a`. Keep the old
draft details here so the launched matrix is easy to inspect.

Common settings:

| Setting | Value |
| --- | --- |
| Trainer | stock LightZero `train_muzero`, `--mode train` |
| Reward | `survival_plus_bonus_no_outcome` |
| Outcome | telemetry only |
| Episode cap | `source_max_steps=65536` |
| Base compute | L4/T4, sim8, C32/n32, B32 |
| Eval/GIF | CurvyZero background eval/GIF on |
| Stock LightZero eval | off |
| Passive rows | out of the main batch |

There are two clean 300-row shapes. The choice depends on what the current
156-row batch teaches.

Option A: more recipe repeat, fewer baseline probes.

| Block | Rows | Shape | Purpose |
| --- | ---: | --- | --- |
| Main mixture grid | 180 | 6 mixes x 2 renders x 3 repeat levels x 5 seeds | Rank opponent mixes with enough repeats to trust them. |
| Pure controls | 60 | 5 controls x 2 renders x 3 repeat levels x 2 seeds | Check whether mixes beat pure opponent sources. |
| Compute probes | 60 | 5 selected rows x 2 renders x 2 repeat levels x 3 compute probes | Check whether more search, more collectors, or larger learner batch changes the result. |

Option B: more baseline probes, slightly fewer recipe repeats.

| Block | Rows | Shape | Purpose |
| --- | ---: | --- | --- |
| Main mixture grid | 144 | 6 mixes x 2 renders x 3 repeat levels x 4 seeds | Rank opponent mixes while still improving over the current 3-seed batch. |
| Pure controls | 60 | 5 controls x 2 renders x 3 repeat levels x 2 seeds | Check whether mixes beat pure opponent sources. |
| Compute probes | 96 | 4 selected mixes x 2 renders x 2 repeat levels x 3 compute probes x 2 seeds | Learn whether search, collectors, or learner batch changes the result. |

Current preference: use Option B if the current 156-row readout stays uncertain
about baseline knobs. Use Option A if baseline settings look boring and the
main uncertainty is recipe variance.

Implementation status: the next-wave generator exists as `--profile next-wave`
in `scripts/build_curvytron_opponent_mixture_manifest.py`. The first draft
artifact was `curvy-mix3-nextwave-20260513a`; it remains dry-run only because
it used older preserved v1b checkpoint refs. The launched artifact is
`curvy-mix3-currentckpt-20260513a`, generated from the same 300-row shape but
with current `curvy-survive-bonus-large-20260513b` checkpoint refs.

It implements Option A:

- matrix name: `curvy-mix3-currentckpt-20260513a`;
- rows: 300;
- blocks: 180 main, 60 controls, 60 compute probes;
- passive rows: none;
- render order: matched fast/browser pairs with alternating lead render;
- launch status: submitted through the grouped trainer app at about
  2026-05-13 09:31 EDT.

Fresh startup read, 2026-05-13 10:13 EDT:

| Readout | Count |
| --- | ---: |
| Rows in manifest/status read | 300 |
| Train roots visible | 187 |
| Running pollers | 180 |
| Live trainer heartbeats | 36 |
| Rows with `progress_latest.json` | 35 |
| Rows at `iteration_0` | 32 |
| Rows at `iteration_10000` | 3 |
| Rows with eval manifests | 8 |
| Rows with GIF artifacts | 26 |

Plain read: the batch is launched and starting, with real checkpoints and some
first `k10` rows. It is not fully warm yet. Do not rank recipes or call rows
failed from this startup snapshot alone.

Follow-up read, 2026-05-13 10:22 EDT:

| Readout | Count |
| --- | ---: |
| Train roots visible | 190 |
| Running pollers | 186 |
| Live trainer heartbeats | 37 |
| Rows with `progress_latest.json` | 36 |
| Rows at `iteration_10000` | 33 |
| Rows at `iteration_20000` | 1 |
| Rows with eval manifests | 28 |
| Rows with GIF artifacts | 34 |

Plain read: mix3 is moving forward. It is still not mature enough for recipe
ranking, but it is no longer just iteration-zero startup evidence.

Artifact caveat: checkpoint eval/GIF workers briefly overloaded Modal volume
commits. The trainer/eval code now uses fewer commits plus retry/backoff with
jitter, and the trainer/eval apps were redeployed around 10:02-10:03 EDT. Old
already-started workers can still log `DataLossError` at old direct commit line
numbers until they drain; monitor new retry-labelled logs separately.

Website read after redeploy: the browser normal API can see a mix3 exact row
with both GIF variants at `iteration_0` and `iteration_10000`. The forced
`fresh=1` path is still slow under the current volume load and returned one
500; use normal exact checks for live spot checks unless a forced reload is
needed.

Artifacts:

- `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix3-currentckpt-20260513a.json`
- `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix3-currentckpt-20260513a.grouped_submit_dryrun.json`
- `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix3-currentckpt-20260513a.grouped_submit_launch.json`

Local gates passed for the draft:

- `uv run pytest tests/test_curvytron_opponent_mixture_manifest.py tests/test_opponent_mixture.py tests/test_curvytron_survivaldiag_submitter.py tests/test_curvytron_mixture_status_analysis.py -q`
- `uv run ruff check ...`
- `uv run ruff format --check ...`
- `uv run python -m py_compile ...`
- grouped submitter dry-run: 300 dry-run records targeting one deployed
  trainer app.

The old `curvy-mix3-nextwave-20260513a` artifact should not be launched. Use
`curvy-mix3-currentckpt-20260513a` for monitoring and future references.

Current learning read is still blocked: `curvy-mix2-clean` has many `k10`
checkpoints and `k10` GIFs, but the status snapshot only exposes
`iteration_0` eval checkpoints so far. Use this batch for cadence/startup/GIF
health now; wait for `k10` eval manifests before ranking mixture recipes.

Update: sparse `k10`/`k20` eval manifests have started to land. Early examples
show mean survival above the initial floor:

| Recipe | Eval sample | Mean steps read |
| --- | --- | ---: |
| `r50-mid50` | 3 rows at `k10`, 3 rows at `k20` | about 39 at `k10`, 47 at `k20` |
| `r50-scr50` | 2 rows at `k10`, 2 rows at `k20` | about 31 at `k10`, 45 at `k20` |
| `r50-blank50` | 1 row at `k10`, 1 row at `k20` | about 25 at `k10`, 36 at `k20` |

This is enough to say survival signal exists. It is not enough to rank recipes,
because sample counts are small and control rows are mostly not mature.

Scripted component check: at least one `r50-scr50` `iteration_10000` GIF summary
selected the scripted opponent component and completed cleanly:
`opponent_mixture_entry_name=scripted`,
`opponent_mixture_age_label=scripted_wall_avoidant`, `ok=true`, 68 physical
steps, no greedy action-collapse warning. Keep checking more samples before
calling scripted broadly healthy, but the basic selection/render path is
working.

Mix3 current-checkpoint main mixes:

| Mix name | Distribution | Why include it |
| --- | --- | --- |
| `r25-blank75` | recent 25, blank 75 | Easier survival pressure. |
| `r50-blank50` | recent 50, blank 50 | Carryover anchor from the current clean batch. |
| `r75-blank25` | recent 75, blank 25 | Tests stronger recent-checkpoint pressure. |
| `r50-scr50` | recent 50, scripted 50 | Trail-making pressure without passive rows. |
| `r50-mid25-old25` | recent 50, mid 25, old 25 | Frozen-checkpoint curriculum mix. |
| `r40-blank20-mid20-scr20` | recent 40, blank 20, mid 20, scripted 20 | Broader mix that is still readable. |

Mix3 current-checkpoint controls:

| Control | Distribution |
| --- | --- |
| `recent100` | recent 100 |
| `blank100` | blank 100 |
| `scr100` | scripted 100 |
| `mid100` | mid 100 |
| `old100` | old 100 |

Mix3 current-checkpoint repeat levels:

| Name | Meaning |
| --- | --- |
| `rep0` | no held-action repeat |
| `repM` | min/max 1/3, extra probability 0.20 |
| `repH` | min/max 1/3, extra probability 0.35 |

Mix3 current-checkpoint compute probes:

| Probe | Setting |
| --- | --- |
| Search | sim16, C32, B32 |
| Collectors | sim8, C64/n64, B32 |
| Learner batch | sim8, C32/n32, B64 |

Monitoring decisions after launch:

- If mix3 scripted opponent artifacts are unhealthy, do not promote scripted
  mixes from this wave.
- If `k10000` checkpoints take longer than about 30 minutes on healthy matched
  mix3 rows, consider `save_ckpt_after_iter=7500` in the next big matrix.
- If fast/browser rankings disagree, keep full 50/50 render pairing. If they
  agree strongly, later waves can move browser to a smaller anchor set.
- Keep future manifests interleaved or randomized by render so startup order
  does not bias speed reads.

## Checkpoint Sources

Use immutable checkpoint refs only.

Current launched source:

- recent:
  `training/lightzero-curvytron-visual-survival/curvy-survive-bonus-blank-fast-light-base-r063-s1111121/attempts/try-blank-fast-light-base-r063-s1111121/train/lightzero_exp/ckpt/iteration_105000.pth.tar`
- mid:
  `training/lightzero-curvytron-visual-survival/curvy-survive-bonus-blank-fast-light-base-r063-s1111121/attempts/try-blank-fast-light-base-r063-s1111121/train/lightzero_exp/ckpt/iteration_60000.pth.tar`
- old:
  `training/lightzero-curvytron-visual-survival/curvy-survive-bonus-blank-fast-light-base-r063-s1111121/attempts/try-blank-fast-light-base-r063-s1111121/train/lightzero_exp/ckpt/iteration_0.pth.tar`

Preferred source order for future launches:

1. Use a strong or late checkpoint from the currently running
   `curvy-survive-bonus-large-20260513b` batch if it matures enough before the
   mixture launch.
2. If the current batch has no clearly useful checkpoint yet, use preserved v1b
   refs as the provisional source. The old dense-run refs were cleaned from the
   volume and must not be used:
   - recent: `survivaldiag-v1b-20260513h-001-.../train/lightzero_exp/ckpt/iteration_20000.pth.tar`
   - mid: `survivaldiag-v1b-20260513h-001-.../train/lightzero_exp/ckpt/iteration_10000.pth.tar`
   - old: `survivaldiag-v1b-20260513h-001-.../train/lightzero_exp/ckpt/iteration_0.pth.tar`

Before launch, write the exact refs into the manifest. Do not resolve `latest`
inside the training job.

## Components

| Component | Simple meaning | Why include it | Main risk |
| --- | --- | --- | --- |
| `recent` | a very recent frozen checkpoint | closest trusted-stock-path version of self-play pressure | may be too strong, too weak, or too similar to ego |
| `mid` | somewhat recent frozen checkpoint | curriculum step below recent | may still saturate outcome or add little variety |
| `old` | old/random-ish checkpoint | easier moving learned opponent | may be mostly noise |
| `scripted` | proactive wall-avoidant hand-designed opponent | creates trails without dying immediately | can create artificial behavior if poorly tuned |
| `passive` | immortal fixed-straight dirty trail-maker | cheap obstacle source | conceptually dirty; may leave unrealistic trails |
| `blank` | no visible/collidable opponent | baseline survival pressure and reset diversity | too easy if overused |

Component critique rule: each component must have a reason to exist, a known
risk, and a test that proves it is actually the component named in the manifest.

## Component Decision Plan

Use a component only if it passes all four checks:

| Check | Question | Pass condition |
| --- | --- | --- |
| Reason | What pressure does this add? | The component fills a clear role in the table above. |
| Risk | How can it mislead us? | The risk is written down and is acceptable for a diagnostic batch. |
| Mechanical proof | Is this really the selected opponent? | Local tests and remote canary summaries record the component name. |
| Learning readout | What metric should move? | The expected readout is survival/reward/bonus/terminal mix, not vague "looks better". |

Do not add a component just because it is available. If a component cannot be
named simply, logged clearly, and tested directly, leave it out of this batch.

## Checkpoint Cadence

Live cadence audit, 2026-05-13 11:35 UTC:

Checkpoint directory pattern:
`training/lightzero-curvytron-visual-survival/{run_id}/attempts/{attempt_id}/train/lightzero_exp/ckpt`.

Fresh `curvy-mix2-canary-20260513b` sample:

| Row | Run id | Attempt id | Render | Knobs | Checkpoints in dir | `progress_latest.elapsed_sec` | `progress_latest.learner_train_iter` | Read |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `r001` | `curvy-mix2b-r50-blank25-scr25-rf-s8-c32-l32-repM-k10-c1-s2196011` | `try-mix2b-r50-blank25-scr25-rf-s8-c32-l32-repM-k10-c1-s2196011` | `body_circles_fast` | sim8, batch32, collectors32, repeat medium, k10000 | `iteration_0`, `iteration_10000` | 1539.7 | 12024 | first k10 checkpoint landed at about 21 min |
| `r002` | `curvy-mix2b-r50-blank25-scr25-rb-s8-c32-l32-repM-k10-c1-s2196011` | `try-mix2b-r50-blank25-scr25-rb-s8-c32-l32-repM-k10-c1-s2196011` | `browser_lines` | sim8, batch32, collectors32, repeat medium, k10000 | `iteration_0`, `iteration_10000` | 1515.6 | 10841 | first k10 checkpoint landed at about 23 min |
| `r003` | `curvy-mix2b-r50-mid25-old25-rf-s8-c32-l32-repM-k10-c1-s2197011` | `try-mix2b-r50-mid25-old25-rf-s8-c32-l32-repM-k10-c1-s2197011` | `body_circles_fast` | sim8, batch32, collectors32, repeat medium, k10000 | `iteration_0` | 1176.1 | 8820 | not enough checkpoint evidence yet |
| `r004` | `curvy-mix2b-r50-mid25-old25-rb-s8-c32-l32-repM-k10-c1-s2197011` | `try-mix2b-r50-mid25-old25-rb-s8-c32-l32-repM-k10-c1-s2197011` | `browser_lines` | sim8, batch32, collectors32, repeat medium, k10000 | `iteration_0` | 1419.8 | 9860 | close to first checkpoint; watch if it stays stuck |
| `r005` | `curvy-mix2b-r50-pass50-rf-s8-c32-l32-repH-k10-c1-s2195011` | `try-mix2b-r50-pass50-rf-s8-c32-l32-repH-k10-c1-s2195011` | `body_circles_fast` | sim8, batch32, collectors32, repeat high, k10000 | `iteration_0`, `iteration_10000` | 1644.3 | 12570 | first k10 checkpoint landed at about 21 min |
| `r006` | `curvy-mix2b-r50-scr50-rb-s8-c32-l32-repH-k10-c1-s2194011` | `try-mix2b-r50-scr50-rb-s8-c32-l32-repH-k10-c1-s2194011` | `browser_lines` | sim8, batch32, collectors32, repeat high, k10000 | `iteration_0` | 1358.9 | 9471 | close to first checkpoint; watch if it stays stuck |

Older `curvy-survive-bonus-large-20260513b` sample:

| Row | Run id | Render | Knobs | Checkpoints in dir | Observed cadence |
| --- | --- | --- | --- | --- | --- |
| `001` | `curvy-survive-bonus-blank-fast-steady-base-r001-s1110011` | `body_circles_fast` | sim8, batch32, collectors32, no repeat, k15000 | `iteration_0`, `15000`, `30000`, `45000` | about 29.0 min per 15000 iters |
| `002` | `curvy-survive-bonus-blank-browser-steady-base-r002-s1110011` | `browser_lines` | sim8, batch32, collectors32, no repeat, k15000 | `iteration_0`, `15000`, `30000`, `45000` | about 31.3 min per 15000 iters |
| `241` | `curvy-survive-bonus-blank-fast-medium-search16-r241-s1140744` | `body_circles_fast` | sim16, batch32, collectors32, repeat medium, k15000 | `iteration_0`, `15000`, `30000` | about 31.5 min per 15000 iters |
| `242` | `curvy-survive-bonus-blank-browser-medium-search16-r242-s1140744` | `browser_lines` | sim16, batch32, collectors32, repeat medium, k15000 | `iteration_0`, `15000`, `30000` | about 31.5 min per 15000 iters |
| `243` | `curvy-survive-bonus-blank-fast-medium-collect64-r243-s1140859` | `body_circles_fast` | sim8, batch32, collectors64, repeat medium, k15000 | `iteration_0`, `15000`, `30000` | about 26.5 min per 15000 iters |
| `244` | `curvy-survive-bonus-blank-browser-medium-collect64-r244-s1140859` | `browser_lines` | sim8, batch32, collectors64, repeat medium, k15000 | `iteration_0`, `15000`, `30000` | about 29.0 min per 15000 iters |
| `245` | `curvy-survive-bonus-blank-fast-medium-batch64-r245-s1140631` | `body_circles_fast` | sim8, batch64, collectors32, repeat medium, k15000 | `iteration_0`, `15000`, `30000` | about 33.5 min per 15000 iters |
| `246` | `curvy-survive-bonus-blank-browser-medium-batch64-r246-s1140631` | `browser_lines` | sim8, batch64, collectors32, repeat medium, k15000 | `iteration_0`, `15000`, `30000` | about 36.0 min per 15000 iters |

Facts from this sample:

- `browser_lines` is not the main explanation for slow checkpoint drops. Matched
  browser rows are usually about 0-3 minutes slower than matched
  `body_circles_fast` rows over a 15000-iteration checkpoint interval.
- First checkpoints are not instant. With k10000 in `mix2b`, first visible
  checkpoint drops were about 21-23 minutes. With k15000 in the older 300b
  rows, first visible checkpoints were about 26-35 minutes.
- Rows that crossed the save threshold did write the expected checkpoint file.
  No checkpoint-save bug is proven by this sample.
- Rows `r003`, `r004`, and `r006` in `curvy-mix2-canary-20260513b` were near
  the k10000 threshold but had not written `iteration_10000` at audit time. If
  those rows remain unchanged, treat that as a per-row health issue, not a
  render-cadence conclusion.
- `progress_latest.json` is useful for the fresh `mix2b` canary. It is often
  stale at iteration 0 in the older 300b rows, so use checkpoint file times for
  those older rows.

Recommended cadence:

| Launch type | Rows | `save_ckpt_after_iter` |
| --- | --- | ---: |
| Mixture canary | all rows | `5000` |
| Full mixture core | sim8, batch32, collectors32, either render | `10000` |
| Full mixture heavy sentinels | sim16 or batch64, either render | `7500` if matching core wall-clock cadence matters; otherwise `10000` |
| Full mixture collectors64 sentinel | sim8, batch32, collectors64, either render | `10000` |

Do not split checkpoint cadence by render mode yet. If the manifest path only
supports one global value, use `10000`. Use `5000` only for short canaries where
early visibility matters more than GIF/eval load. Do not use `1000`; that would
create too many checkpoint/eval/GIF jobs.

Rows that only show `iteration_0`, or no checkpoints, are not cadence evidence
by themselves. First check whether the trainer was queued, crashed, or still
starting.

Current read on the fidelity question: the sampled 300b rows do not show
`browser_lines` as the main reason some rows barely drop checkpoints. Browser
is usually only a little slower than the matched fast row in the samples above.
The bigger visible slowdowns are heavier settings like batch64 or rows that
have not yet proven they are actually training. Keep auditing with exact row
status before changing the next batch.

## Proposed Mixture Recipes

These names match the manifest builder. The older 20-recipe draft was
exploratory; do not launch it as-is.

Main recipes keep `recent` at 50%. The remaining 50% changes the opponent
pressure.

| Recipe | Rows in `core` | Weights | Decision |
| --- | ---: | --- | --- |
| `r50-blank50` | 18 | recent 50, blank 50 | Keep. |
| `r50-mid50` | 18 | recent 50, mid 50 | Keep. |
| `r50-old50` | 18 | recent 50, old 50 | Keep. |
| `r50-scr50` | 18 | recent 50, scripted 50 | Keep after scripted canary proves telemetry. |
| `r50-pass50` | 18 in core review, 0 in clean launch | recent 50, passive 50 | Drop from first launch. |
| `r50-blank25-scr25` | 18 | recent 50, blank 25, scripted 25 | Keep. |
| `r50-mid25-old25` | 18 | recent 50, mid 25, old 25 | Keep. |
| `r50-blank20-mid15-scr15` | 18 | recent 50, blank 20, mid 15, scripted 15 | Keep after scripted canary proves telemetry. |

Controls:

| Control | Rows in `core` | Decision |
| --- | ---: | --- |
| `recent100` | 6 | Keep. |
| `mid100` | 6 | Keep. |
| `old100` | 6 | Keep. |
| `blank100` | 6 | Keep. |
| `scr100` | 6 | Keep after scripted canary proves telemetry. |
| `pass100` | 6 in core review, 0 in clean launch | Drop from first launch. |

Distribution in `clean`:

- Main recipes: 7 recipes x 2 render modes x 3 repeat settings x 3 seeds = 126 rows.
- Controls: 5 controls x 2 render modes x 3 repeat settings x 1 seed = 30 rows.
- Total: 156 rows.

Do not add every combination of every knob. The first large mixture batch should
answer three plain questions: which opponent recipe helps, whether the fast
render path agrees with browser, and whether held-action repeat helps.

Critique of the 180-row core artifact:

- Keep the main rows with 50% `recent`; that is the meaningful opponent
  mixture distribution for this phase.
- Keep non-passive controls because they are cheap enough and make the readout
  easier to explain.
- Drop passive rows from the first full launch. Current passive immortal
  behavior is a trail-maker, not a clean opponent design.
- Paired render modes are justified because fast-vs-browser is still part of
  the readout and sampled cadence cost is small.
- Keep the full 2 render x 3 repeat sweep for this first clean batch so the row
  shape is simple and the manifest builder can be used directly.

Second wave only if the first large batch gives useful signal:

- add sim16/C64/B64 sentinels;
- add more seeds for noisy but promising recipes;
- add more scripted-opponent variants if scripted lanes clearly help.

## Naming

Use short, readable names:

```text
<run-prefix>-<recipe>-<render>-s<SIMS>-c<COLLECTORS>-l<BATCH>-<repeat>-k10-c<COPY>-s<SEED>
```

Examples:

```text
curvy-mix2clean-r50-blank50-rf-s8-c32-l32-repM-k10-c1-s2101011
curvy-mix2clean-r50-mid25-old25-rb-s8-c32-l32-rep0-k10-c2-s2107021
curvy-mix2clean-scr100-rb-s8-c32-l32-repH-k10-c1-s2113011
```

Do not encode every internal checkpoint filename in the run ID. Put exact refs
in the manifest.

## Validation Gates

These gates passed before the mix2-clean and mix3-currentckpt launches. Keep
them here as the checklist for any later relaunch.

Local tests:

- parse a mixture spec into normalized weights that sum to 1;
- reject empty specs, negative weights, unknown components, and missing
  checkpoint refs;
- sample opponent component per reset, not per step;
- prove two env instances with the same seed/reset sequence choose the same
  components;
- prove different reset indices can choose different components;
- prove the selected component is visible in `info`, telemetry, and run
  manifests;
- prove each component maps to the intended runtime/death/policy settings;
- prove no silent fallback to fixed-straight occurs.
- prove the readiness gate expects
  `learner_vs_weighted_episode_opponent_mixture` when a mixture is present.

Manifest/dry-run tests:

- every row uses the same base settings except recipe, seed, and run id;
- every row has `reward_variant=survival_plus_bonus_no_outcome`;
- every row has `source_max_steps=65536`;
- every row has background checkpoint eval/GIF enabled;
- every row has stock LightZero in-loop eval disabled;
- checkpoint refs are exact immutable paths, not `latest`;
- the grouped submitter rejects rows with missing mixture refs or invalid
  component weights.

Tiny remote canary before the mix2 clean launch:

1. Matched fast/browser rows for `r50-blank25-scr25`.
2. Matched fast/browser rows for `r50-mid25-old25`.
3. One passive-heavy fast row and one scripted-heavy browser row to prove the
   dirty/scripted components are actually carried through telemetry.

Canary pass means:

- trainer writes `run.json`, `attempt.json`, `status_heartbeat.json`, and
  `progress_latest.json`;
- `status_heartbeat.json` does not stop at `before_train_muzero`;
- compact output or summary proves `called_train_muzero=true`;
- readiness gate is `ok=true` and the opponent relation is
  `learner_vs_weighted_episode_opponent_mixture`;
- iteration 0 checkpoint exists;
- eval/GIF poller sees the checkpoint;
- eval/GIF summaries record selected opponent components;
- component counts over sampled resets are plausible for the weights;
- no trainer crash, no missing checkpoint ref, no unexpected fallback.

## Launch Order Status

1. Implementation in the trusted stock path: done.
2. Local tests, ruff, format check, and py_compile: done for the launched
   mixture surfaces.
3. Mixture canaries: done for mix2 after the relation fix.
4. `curvy-mix2-clean-20260513a`: launched and healthy-ish; keep monitoring eval
   maturity before ranking recipes.
5. `curvy-mix3-currentckpt-20260513a`: launched through the grouped app; monitor
   trainer roots, `iteration_0`, eval/GIF artifacts, and first `k10`
   checkpoints.
6. Keep `curvy-survive-bonus-large-20260513b` running unless it shows a concrete
   failure; it is the current frozen-checkpoint source.

## First Readout

Rank recipes by:

- latest mean survival;
- best mean survival;
- training reward curve;
- bonus pickup count;
- terminal cause mix;
- action collapse;
- component selection counts;
- eval/GIF health.

Do not drop a recipe solely because early greedy GIFs look collapsed. Use the
training reward/survival curves and component counts first.
