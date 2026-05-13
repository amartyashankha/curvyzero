# v1d Axis Projection

Purpose: project the old `stock-tensor-v1d` and fixed-opponent runs along the
knobs that matter before choosing the next matrix.

Read old v1d through outcome/score curves first. Those runs did not train on
the survival-first reward we now care about, so old survival is only a weak
observability signal. A win increase against weak frozen opponents can be a
low-bar result.

Local evidence used:

- manifest:
  `artifacts/local/curvytron_stock_train_manifests/stock-tensor-v1d.json`
- fresh eval snapshot:
  `artifacts/local/curvytron_status_snapshots/2026-05-13/stock-tensor-v1d-eval.clean.json`
- curve scores:
  `artifacts/local/curvytron_status_snapshots/2026-05-13/stock-tensor-v1d-curve-scores.md`

Known missing data: v1d has one seed per row, only tiny stochasticity
(`straight_override_0.05`), no survival-plus-bonus/no-outcome reward row, no
immortal/no-trail/blank-canvas opponent row, and browser rows used C16 while
the main fast rows used C32.

## Axis Projection

| Axis | Observed effect | Confidence | What to do next |
| --- | --- | --- | --- |
| Reward setting | `sparse_outcome` and `dense_survival_plus_outcome` both produced outcome lift on fixed/old rows. Neither fixed recent/mid saturation. Because outcome was part of the old objective, this does not answer the new survival-first question. | Medium for "old reward is inconclusive"; low for any sparse-vs-dense ranking. | Use `survival_plus_bonus_no_outcome` for the main stock `--mode train` lane. Keep outcome as eval telemetry only. |
| Opponent type/strength | Fixed-straight and old frozen opponents started mixed/lossy and later reached mostly wins. Recent/mid frozen rows often started near or at `win:8` while mean survival stayed around `8`. | High. | Main next rows must remove easy opponent-death saturation: blank-canvas/noop sanity first, then cleaner trail-maker or scripted/random opponent diagnostics. Keep fixed/old/recent only as controls. |
| Render mode | Browser fixed rows moved and peaked, so browser is not obviously broken. But browser rows also changed collector count (`C16` vs main `C32`), so render is confounded. | Medium-low. | Run matched fast/browser twins on serious settings with the same logical seed/copy and collector settings. Do not broad-sweep render. |
| Search sims | sim16 rows reached wins but did not rescue recent/mid survival. Fixed dense sim8 and sim16 both ended at `win_rate=1`, with similar survival (`17.25` latest). | Medium. | Default sim8. Add only a few sim16 sentinels on the strongest new cell. |
| Collector batch/envs | C64 fixed dense had the best fixed dense latest survival (`20.5`) but recent/mid still stayed near floor. Old frozen still peaked then fell. | Medium-low. | Default C32/n32. Use C64 as a sentinel only after objective/opponent are fixed. |
| Learner batch | B64 did not rescue recent/mid. Fixed dense B64 moved, but no clearer than B32 and had fewer eval points. | Medium. | Default B32. Add B64 only as a later sentinel, not a main axis. |
| Stochasticity | Only `straight_override_0.05` was tested. Fixed sparse row 31 looked decent (`0.25 -> 0.875` win rate; survival `10.25 -> 19.25`), but recent dense stayed saturated/flat. | Low. | Sweep meaningful levels later: none, low, medium, high. Repeat important stochastic rows across seeds. |
| Episode cap | `256` vs `1024` did not change the story; agents died far below either cap. | Medium-high. | Stop sweeping cap. Use a high cap such as `65536`; if episodes become long, that is the signal. |
| Seed/repeats | v1d rows were single-copy rows with changing row seeds, not repeated copies of the same logical setting. | High. | Future manifests should separate `training_seed`, `reset_seed`, `opponent_policy_seed`, `opponent_behavior_seed`, `eval_seed`, and `copy_id`. Use repeats for stochastic/random rows. |

## Promising Or Misleading Old Rows

| Rows | Looked promising because | Why that may mislead | Next use |
| --- | --- | --- | --- |
| 01 fixed sparse fast C32 B32 sim8 | Outcome improved from `loss:7,win:1` to latest `loss:1,win:7`; mean survival improved from `12.875` to `21.75`, best `23.0`. | Fixed-straight is a low-bar opponent and the reward was outcome-only. | Keep as a small fixed control; do not treat as proof of survival learning. |
| 02 fixed dense fast C32 B32 sim8 | Outcome improved to `win:8`; survival moved from `8.75` to `17.25`. | Dense reward included outcome, and action-collapse flag appeared. | Useful baseline for old-vs-new reward sanity, not a main lane. |
| 03/04 recent fast | Row 03 was `win:8` from first to latest; row 04 stayed near-saturated. | This is exactly the trap: outcome looks good while survival stays `8.0`. | Do not repeat unchanged as a main training row. Use as a saturation control only. |
| 05/06 mid fast | Row 05 reached `win:8`; row 06 was `win:8` from start. | Survival stayed flat near `8`; outcome cannot show learning once saturated. | Keep only minimal ancestor controls. |
| 07/08 old fast | Old frozen rows moved from mixed/lossy to `win:8`, with survival peaks `17.0` and `18.125`. | Latest survival fell back (`13.125`, `13.5`); may be curriculum weakness, brittle exploitation, or eval noise. | Keep a few old-opponent controls; require survival-first confirmation. |
| 13 fixed dense C64 | Best/latest survival (`21.25`/`20.5`) beat row 02, suggesting more collection can help the easy fixed case. | C64 did not fix recent/mid, and the row is still fixed-straight/outcome-contaminated. | C64 sentinel only, after the main objective/opponent cell works. |
| 17 fixed dense B64 | Reached `win:8` with survival `17.375` latest. | No clear win over B32; only 23 eval points in the curve snapshot. | B64 sentinel only if budget allows. |
| 21/22 fixed browser | Browser rows reached `win:8` or near it; row 22 peaked at `22.25` survival. | Browser rows used C16/n16, so this is not a clean render comparison. Row 22 peak fell to `16.25`. | Use matched render twins in the next serious cells. |
| 26/27 max1024 fixed | Higher cap rows still learned the easy fixed outcome. | Deaths were far below both caps; cap was not the limiter. | Set high cap everywhere; do not spend rows on cap. |
| 31 fixed sparse straight005 | Latest survival `19.25`, latest outcome `loss:1,win:7`; among the better fixed rows. | Only one tiny stochasticity level and still fixed-straight/outcome-only. | Sweep stochasticity harder under survival-first reward, with repeats. |
| 32 recent dense straight005 | Stayed `win:8` and survival `8.0`. | Tiny stochasticity did not break weak-opponent saturation. | Evidence that stochasticity alone is not the fix. |

## Bottom Line

v1d says the stock path can move outcome against easy opponents, but it does not
prove a useful CurvyTron survival-learning setup. The next matrix should spend
rows on the objective/opponent problem first, then use search, collectors,
batch size, render, and stochasticity as small sentinels around the strongest
cell.
