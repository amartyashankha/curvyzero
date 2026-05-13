# Next Overnight Matrix Plan

Purpose: describe the next CurvyTron training batch in plain language.

Status: historical rescue plan plus current pointer. The blank-canvas rescue
batch was launched as `curvy-survive-bonus-large-20260513b` and is now
background evidence. The active next-lane plan is the opponent-mixture wave in
[opponent_mixture_batch_plan_2026-05-13.md](opponent_mixture_batch_plan_2026-05-13.md).

Current live mixture batch:

- `curvy-mix2-clean-20260513a`
- launched 2026-05-13 07:49 EDT;
- 156 rows in one deployed Modal trainer app;
- paired `body_circles_fast` and `browser_lines`;
- `save_ckpt_after_iter=10000`;
- first status sweep was startup-only evidence, not a clean render-speed read.

Candidate next large matrix after the current batch has first `k10` checkpoints:

| Block | Rows | Plain shape |
| --- | ---: | --- |
| Main mixture grid | 180 | 6 opponent mixes x 2 renders x 3 repeat levels x 5 seeds |
| Pure controls | 60 | 5 pure opponent sources x 2 renders x 3 repeat levels x 2 seeds |
| Compute probes | 60 | selected mixes x 2 renders x 2 repeat levels x sim16/C64/B64 probes |

Alternate 300-row shape if baseline knobs remain uncertain:

| Block | Rows | Plain shape |
| --- | ---: | --- |
| Main mixture grid | 144 | 6 opponent mixes x 2 renders x 3 repeat levels x 4 seeds |
| Pure controls | 60 | 5 pure opponent sources x 2 renders x 3 repeat levels x 2 seeds |
| Compute probes | 96 | 4 selected mixes x 2 renders x 2 repeat levels x 3 compute probes x 2 seeds |

Do not launch this candidate until the current 156-row batch says whether
scripted opponents are healthy, whether fast/browser rows agree, and whether
`k10000` is close enough to the target checkpoint cadence.

## Rescue Order

1. Patch manifest/submission/trainer wrapper call shape.
2. Run focused tests and ruff.
3. Redeploy the canonical trainer app.
4. Rebuild the 300-row manifest.
5. Submit all 300 rows into the one deployed app.
6. Verify sampled rows have trainer-owned files, not only poller files.
7. Redeploy or confirm the GIF website.
8. Monitor early trainer heartbeats, markers, checkpoints, evals, and GIFs.

## One-App Rule

Do not create one Modal app per row. Deploy the canonical trainer app once:

```text
curvyzero-lightzero-curvytron-visual-survival-train
```

Then submit rows with:

```text
scripts/submit_curvytron_survivaldiag_manifest.py --allow-launch
```

For each row, the submitter spawns:

- checkpoint poller first;
- train function second.

This preserves checkpoint eval/GIF behavior while keeping the Modal dashboard
clean.

## Common Settings

| Setting | Value |
| --- | --- |
| Trainer | stock LightZero `train_muzero` |
| Mode | `train` |
| Env | `source_state_fixed_opponent` |
| Main opponent | `blank_canvas_noop` |
| Reward | `survival_plus_bonus_no_outcome` |
| Outcome reward | off; telemetry only |
| Episode cap | `source_max_steps=65536` |
| Compute | L4/T4 CPU40 |
| Main search | `num_simulations=8` |
| Main collector | `collector_env_num=32` |
| Main batch | `batch_size=32` |
| Render | matched `body_circles_fast` and `browser_lines` |
| Eval/GIF | CurvyZero background eval/GIF on |
| Stock LightZero eval | off, `lightzero_eval_freq=0` |
| Checkpoint cadence | `save_ckpt_after_iter=15000` |
| Expected checkpoint gap | about 30 minutes on active sim8 rows |
| Train cap | `max_train_iter=300000`, `max_env_step=30000000` |
| Poller lifetime | `background_eval_poller_max_runtime_sec=64800` |

## Batch Shape

The manifest has 300 rows and 150 matched fast/browser pairs.

| Plain block | Rows | Shape |
| --- | ---: | --- |
| Blank canvas all levels | 160 | 2 renders x 4 stochasticity levels x 20 copies |
| Blank canvas medium/high extra | 40 | 2 renders x medium/high x 10 extra copies |
| Passive immortal dirty controls | 40 | 2 renders x 4 stochasticity levels x 5 copies |
| Compute sentinels | 60 | 2 renders x medium/high x 5 copies x search16/collect64/batch64 |

No H100 rows are in this batch.

## Stochasticity Names

| Name | Policy repeat knobs |
| --- | --- |
| `steady` | min/max `1/1`, extra probability `0.0` |
| `light` | min/max `1/2`, extra probability `0.10` |
| `medium` | min/max `1/3`, extra probability `0.20` |
| `heavy` | min/max `1/3`, extra probability `0.35` |

These are held-action repeats inside one LightZero env step. They are not
separate no-op training transitions.

## Names

Run names are readable and encode the important axes:

```text
curvy-survive-bonus-<opponent>-<render>-<stochasticity>-<compute>-rNNN-sSEED
```

Examples:

```text
curvy-survive-bonus-blank-fast-steady-base-r001-s1110011
curvy-survive-bonus-passive-browser-heavy-base-r240-s1133041
curvy-survive-bonus-blank-browser-heavy-batch64-r300-s1141671
```

## Kept Out

These are not in the 300-row launch:

- H100 probes;
- ancestor/frozen checkpoint opponents;
- random-init frozen opponents;
- scripted wall-avoidant opponents;
- survival-only reward ablation;
- two-seat self-play.
- recent-checkpoint mixture opponents.

They can join later only after their own first-class wiring and tiny e2e
canaries.

## Follow-Up Mixture Batch

Current trusted stock path can train against one static frozen checkpoint
opponent per run. It does not yet support a weighted opponent pool with recent,
somewhat recent, old, scripted, blank, passive/immortal, and hand-designed
opponents. That is the next batch lane after the rescue launch is healthy
enough to monitor in the background.

## Validation Done

- Manifest tests: passed.
- Env/plumbing tests for the touched reward/opponent/checkpoint paths: passed.
- Grouped submitter dry run: passed for sample rows.
- Speed display fix: trainer now writes `train/progress_latest.json` on
  checkpoint save; local test covers it.

## Next After Launch

Monitor:

- train status and checkpoint count;
- checkpoint eval/GIF health;
- survival curve;
- trainer reward curve;
- bonus pickup count;
- terminal cause;
- action distribution/collapse.
