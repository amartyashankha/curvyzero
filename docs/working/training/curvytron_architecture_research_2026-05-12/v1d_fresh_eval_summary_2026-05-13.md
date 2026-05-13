# v1d Fresh Eval Summary - 2026-05-13

Purpose: record the fresh Modal `eval-summary` pull for the 32-row
`stock-tensor-v1d` matrix. This is old-run analysis, not a launch decision.

Command shape:

```text
modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids <32 v1d run ids> \
  --attempt-ids <32 v1d attempt ids> \
  --output eval-summary
```

## Plain Read

The old conclusion still holds:

- fixed-straight rows show weak real movement;
- old frozen rows show some movement but not stable enough;
- recent and mid frozen rows mostly stay around `8` mean steps;
- score/outcome often saturates even when survival is poor;
- action-collapse flags still appear in several rows.

## Key Rows

| Row | Label | Best mean steps | Latest mean steps | Latest outcome | Read |
| --- | --- | ---: | ---: | --- | --- |
| 01 | fixed sparse fast B32 sim8 | `23.000` | `21.750` | `loss:1,win:7` | Best simple fixed row now looks better than the older note. |
| 02 | fixed dense fast B32 sim8 | `17.375` | `17.250` | `win:8` | Improves score/survival, but action-collapse flag appears. |
| 03 | recent sparse fast B32 sim8 | `8.000` | `8.000` | `win:8` | Outcome saturated; no survival learning. |
| 04 | recent dense fast B32 sim8 | `8.000` | `8.000` | `draw:1,win:7` | Still flat and action-skewed. |
| 05 | mid sparse fast B32 sim8 | `8.375` | `8.000` | `win:8` | Tiny blip only. |
| 06 | mid dense fast B32 sim8 | `8.000` | `8.000` | `win:8` | Flat; latest greedy action fully collapsed. |
| 07 | old sparse fast B32 sim8 | `17.000` | `13.125` | `win:8` | Some movement, but latest falls back. |
| 08 | old dense fast B32 sim8 | `18.125` | `13.500` | `win:8` | Some movement, not stable. |
| 13 | fixed dense C64 B32 sim8 | `21.250` | `20.500` | `loss:1,win:7` | More collectors may help fixed control, but not proven broadly. |
| 21 | fixed sparse browser B32 sim8 | `17.375` | `16.500` | `win:8` | Browser is not obviously broken. |
| 22 | fixed dense browser B32 sim8 | `22.250` | `16.250` | `loss:1,win:7` | Big peak then drop. |
| 31 | fixed sparse straight005 B32 sim8 | `21.250` | `19.250` | `loss:1,win:7` | Stochasticity remains worth testing harder. |
| 32 | recent dense straight005 B32 sim8 | `8.000` | `8.000` | `win:8` | Tiny stochasticity does not fix recent frozen. |

## Axis Projection

| Axis | Fresh v1d evidence | Current decision |
| --- | --- | --- |
| Opponent | Fixed and old can move; recent/mid mostly flat at `8`. | Main next question should remove weak-opponent reward saturation. |
| Reward | Sparse and dense both move on fixed; neither fixes recent/mid. | Old reward sweep is not enough. Next reward should train survival plus bonus, outcome off. |
| Render | Fixed browser rows move and peak; not wildly worse than fast. | Keep matched fast/browser checks for serious future cells. |
| Stochasticity | Only `0.05` straight override was tested; fixed row stayed decent, recent still flat. | Sweep more levels if/when the next diagnostic is ready. |
| Search | sim16 does not rescue recent/mid; fixed sim16 is only mildly useful. | Keep sim8 default; sim16 only as sentinel. |
| Collector count | C64 fixed row improved, but recent/mid still flat. | C64 is secondary; not the main fix. |
| Learner batch | B64 does not rescue recent/mid. | B32 default remains reasonable until objective/opponent is fixed. |
| Episode cap | 1024 rows do not change the story. | Set high cap later; do not sweep cap. |

## Tooling Validation

The status/tooling bridge now works:

- raw Modal output:
  `artifacts/local/curvytron_status_snapshots/2026-05-13/stock-tensor-v1d-eval.json`
- cleaned JSON snapshot:
  `artifacts/local/curvytron_status_snapshots/2026-05-13/stock-tensor-v1d-eval.clean.json`
- curve scores:
  `artifacts/local/curvytron_status_snapshots/2026-05-13/stock-tensor-v1d-curve-scores.md`

Tooling result agrees with the manual read:

- fixed rows have survival lift and win-rate lift;
- old frozen rows have some survival lift but often peak and fall back;
- recent/mid frozen rows often have saturated win rate while survival stays
  flat near `8`;
- compute knobs do not rescue recent/mid frozen rows.

Tests for the status/tooling bridge passed:

```text
uv run pytest tests/test_eval_curves.py tests/test_curvytron_run_status.py -q
13 passed
```
