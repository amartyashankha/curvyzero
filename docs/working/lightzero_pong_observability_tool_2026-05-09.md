# LightZero Pong Observability Table Tool - 2026-05-09

## Tool

`scripts/summarize_lightzero_pong_scorecards.py` turns one or more dummy Pong
scorecard `summary.json` files into a checkpoint-curve table. It is local-first:
use Modal CLI to fetch summaries, then run the script over local JSON files.

Example:

```bash
uv run python scripts/summarize_lightzero_pong_scorecards.py \
  artifacts/local/lightzero-scorecards/iter8-summary.json \
  artifacts/local/lightzero-scorecards/iter64-summary.json
```

The default table is sorted by checkpoint label and opponent. Columns are:
source, checkpoint label, opponent, seating mode, episodes, wins/losses/
truncations, survival mean/median/p90, shaped loss-delay return mean, score
return mean, action histogram, and warnings.

Useful output variants:

```bash
uv run python scripts/summarize_lightzero_pong_scorecards.py \
  --format tsv \
  --output artifacts/local/lightzero-scorecards/checkpoint_curve.tsv \
  artifacts/local/lightzero-scorecards/*-summary.json

uv run python scripts/summarize_lightzero_pong_scorecards.py \
  --format json \
  artifacts/local/lightzero-scorecards/iter64-summary.json
```

## Modal Fetch

Fetch exact summary refs from the `curvyzero-runs` Volume with:

```bash
uv run --extra modal modal volume get curvyzero-runs \
  training/lightzero-dummy-pong/<run_id>/attempts/<attempt_id>/eval/<eval_id>/summary.json \
  artifacts/local/lightzero-scorecards/<label>-summary.json \
  --force
```

For standalone eval refs:

```bash
uv run --extra modal modal volume get curvyzero-runs \
  eval/lightzero-dummy-pong/<eval_dir>/summary.json \
  artifacts/local/lightzero-scorecards/<label>-summary.json \
  --force
```

The script can also resolve `ref:` paths if the Volume is already mounted
locally, for example:

```bash
CURVYZERO_RUNS_MOUNT=/runs uv run python scripts/summarize_lightzero_pong_scorecards.py \
  ref:training/lightzero-dummy-pong/<run_id>/attempts/<attempt_id>/eval/<eval_id>/summary.json
```

## Warnings

Warnings are intentionally blunt for fast checkpoint-curve reads:

- `zero_down`: checkpoint policy never chose action 2.
- `zero_up`: checkpoint policy never chose action 0.
- `single_action_gt_95pct`: one action accounts for more than 95 percent of
  recorded actions.
- `worse_than_random`: the checkpoint row scores below `random_uniform` in the
  same matchup.
- `no_survival_gain_if_comparable`: survival is no better than the available
  `random_uniform` baseline against the same opponent.

The survival-gain warning only fires when the same summary contains a comparable
`random_uniform` row for that opponent. Missing action histograms are reported
as `missing_action_hist`.

## Eval Trust Critique

Serious official Pong eval currently means:

- `max_eval_steps=2048`
- `max_episode_steps=2048`
- `num_simulations=50` MCTS simulations per action
- usually `16` fresh randomized starts per checkpoint for claim-seeking reads
- record the start/replay information needed to reproduce the panel later

Plain trust read:

| question | current answer |
| --- | --- |
| Is `16` games enough? | Enough to catch large survival moves, like hundreds of extra stock steps. Not enough for subtle deltas, stability claims, or "solved" language. If a run moves by only tens of steps, treat it as triage until repeated on another fresh randomized-start panel. |
| Is the `2048` step cap high enough? | High enough for the current early Pong question: does the policy get past the usual ~760-step loss wall? Not high enough to distinguish policies that can already survive to cap. Once rows hit `2048`, rerun with a higher cap before claiming durable play. |
| What do action histograms tell us? | A dominant action above about `0.95` is action collapse unless there is a specific mechanical reason. Broader histograms across actions are better telemetry, but not proof by themselves; survival and return still have to move. |
| Are current agents still one-action policies? | Mixed. Some rows are still collapsed or near-collapsed, such as one action around `0.99` or `1.00`. The better late stock/shaped rows are broader, with dominant actions closer to `0.39`-`0.44` and high entropy. |
| How long before reading signal? | `iteration_1000` has repeatedly been too early. Start treating `5000`/`7000` as the first useful read, and prefer later curves like `9000`/`12000`/`15000+` before judging stability. For shaped long runs, the clearest behavior appeared much later, around `51000`/`54000`. |

Report discipline:

- Lead with `stock_steps_survived` versus the same run's `iteration_0`.
- Put return and positive rewards after survival.
- Keep survival-shaped rows labeled side-lane telemetry.
- Use fresh randomized starts for new claim-seeking eval waves, and record the
  replay information needed to reproduce the exact starts later.
- Do not use one capped run, one small return bump, or one broader action
  histogram as proof by itself.

## MCTS Debug Rows

The optional debug mode is lazy and imports LightZero only when used:

```bash
uv run --extra modal python scripts/summarize_lightzero_pong_scorecards.py debug-mcts \
  --checkpoint lightzero:iter64=/path/to/iteration_64.pth.tar \
  --rows 12 \
  --num-simulations 8 \
  --max-env-step 4096 \
  --format md
```

It emits seed, seat, compact observation, action, and any adapter-exposed
policy logits, visit counts, or value fields. If LightZero changes the
eval-mode output shape, the row still records the output type instead of
failing the table path.

## Next Integrations

- Add the table command to Modal scoreboard result notes so each run prints the
  fetch command and local summarizer command together.
- Keep independent scorecard summaries as the source of truth for checkpoint
  quality; trainer-side `pong_scorecard` rows are supported but labeled
  `trainer-side`.
- If checkpoint curves become frequent, add a tiny Modal wrapper that fetches
  refs into a temp directory and calls this same script, rather than adding a
  second renderer.
