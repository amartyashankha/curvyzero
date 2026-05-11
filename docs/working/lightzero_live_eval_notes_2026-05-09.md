# LightZero Live Eval Notes - 2026-05-09

Purpose: evaluate Pong checkpoints while a Modal training job is still running.
Do not wait for the whole train to finish if checkpoints are already in the
Volume.

## Current Rule

- The training job writes checkpoints to the Modal Volume.
- The eval job reads checkpoints from that Volume by ref.
- Eval every new `iteration_*.pth.tar` once.
- Treat `ckpt_best.pth.tar` as suspicious unless its checkpoint state proves it
  is a real trained checkpoint. One earlier run produced a reset-looking
  `ckpt_best`.
- Use strict checkpoint load and no model fallback.
- Keep stock-ish Pong eval knobs fixed while comparing checkpoints:
  `num_simulations=50`, `evaluator_env_num=3`, `collector_env_num=8`,
  `batch_size=256`, `game_segment_length=400`, `max_env_step=200000`,
  `max_eval_steps=512`, and `max_episode_steps=512`.
- Read Pong eval with survival-first scoring, not win/loss alone. Official
  Atari eval rows must keep same-run baseline, stock return, manual/raw return,
  `steps_survived`, reward counts/timing, positive rewards, action entropy,
  and collapse verdict. Custom dummy Pong rows must keep survival
  mean/median/p90/std and shaped loss-delay beside true score return.
- Survival length is telemetry. It is the cleaner early signal for weak
  policies, but it is not the default LightZero training reward. See
  `docs/working/lightzero_pong_survival_reward_audit_2026-05-09.md`.
- Survival is first-class. Survival means raw `steps_survived`. Every eval
  note/table must include raw return, eval cap, episode steps survived,
  `survival_fraction` as the secondary normalized value
  (`steps_survived / eval_cap_steps`), whether the checkpoint survived to the
  cap when obvious, nonzero reward count, positive reward count, and whether
  the checkpoint survived longer than the same-run `iteration_0` baseline. Do
  not summarize a checkpoint as only win/loss return.
- `8192` final versus `32768 iteration_0` is not a regression comparison.
  Compare the active `32768` run only after a later normal `32768`
  checkpoint is evaluated against the same-run `iteration_0`.

## Seed 2 L4 Matrix Eval

- Run id: `lz-visual-pong-exact-installed-0.2.0-s2`
- Attempt:
  `train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath`
- Corrected seed-2 strict eval app: `ap-39Cn4c7Id0mant7cjFLPWp`.
- Eval id:
  `matrix-s2-l4-iter0-1000-2000-3000-stock2048-seed2`.
- Manifest:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s2/attempts/train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath/eval/matrix-s2-l4-iter0-1000-2000-3000-stock2048-seed2/manifest_custom_steps2048_seed2_20260510T013455Z.json`
- Local fetched manifest copy:
  `artifacts/local/lightzero-eval-manifests/matrix-s2-l4-iter0-1000-2000-3000-stock2048-seed2/manifest_custom_steps2048_seed2_20260510T013455Z.json`
- Local summary:
  `artifacts/local/lightzero-eval-manifests/matrix-s2-l4-iter0-1000-2000-3000-stock2048-seed2/summary_baseline_deltas.tsv`
- Claim: seed 2 L4/T4 checkpoints `iteration_0`, `iteration_1000`,
  `iteration_2000`, and `iteration_3000` all strict-loaded with no fallback
  and ran the stock evaluator under the 2048-step cap.
- Non-claim: this does not show learning progress through `iteration_3000`;
  survival, manual return, stock return, positive rewards, and
  negative/nonzero rewards are unchanged versus same-run `iteration_0`.
- Survival-first result: every row survived `762/2048`
  (`survival_fraction=0.37207`), did not survive to cap, and had
  `delta_steps_survived=0`. Manual return and stock return were `-21` for all
  four rows. Positive rewards stayed `0`; negative/nonzero rewards stayed
  `21`. Strict load was `true` and fallback was `false` for all rows.
  `iteration_0` was fully collapsed to action `1` (`share=1.0`,
  entropy `0`). `iteration_1000` was still action-1 dominated
  (`share=0.832021`, entropy `0.429028`). `iteration_2000` shifted into
  action-4 near-collapse (`share=0.94357`, entropy `0.313109`).
  `iteration_3000` became less pure but still action-0 dominated
  (`share=0.611549`, entropy `0.648552`).
- Plain read: this is a flat quality curve through `iteration_3000`. The
  action distribution changes, but it does not buy any survival, return, or
  reward improvement.

## Active 32768 Run

- Run id: `lz-visual-pong-exact-installed-0.2.0-s0`
- Clean post-patch spawned frequent-checkpoint attempt:
  `train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-spawn-relpath`.
- Clean spawned strict eval app:
  `ap-ys2xvPLPy17AON6HhTtnsu`.
- Clean spawned strict eval manifest:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-spawn-relpath/eval/live-32768-ckpt1000-spawn-iteration0-vs-1000-stockish512-stockeval-s0/manifest_custom_steps512_seed0_20260510T010212Z.json`
- Local fetched manifest copy:
  `artifacts/local/lightzero-eval-manifests/live-32768-ckpt1000-spawn-iteration0-vs-1000-stockish512-stockeval-s0/manifest_custom_steps512_seed0_20260510T010212Z.json`
- Local summary:
  `artifacts/local/lightzero-eval-manifests/live-32768-ckpt1000-spawn-iteration0-vs-1000-stockish512-stockeval-s0/summary_baseline_deltas.tsv`
- Claim: the clean post-patch spawned frequent-checkpoint eval loaded both
  `iteration_0` and `iteration_1000` strictly, with no model fallback, and both
  survived `512/512` eval steps under the stock evaluator path.
- Non-claim: this does not show learning progress; `iteration_1000` did not
  improve steps, manual return, stock return, reward count, positive reward
  count, or action collapse versus same-run `iteration_0`.
- Survival-first result: `iteration_0` survived `512/512`, return `-13`,
  stock return `-13`, `13` nonzero rewards, `0` positive rewards, dominant
  action `2` share `1.0`, entropy `0`, strict load true, fallback false,
  stock/manual match true. `iteration_1000` survived `512/512`,
  `delta_steps_survived=0`, return `-13` (`delta_return=0`), stock return
  `-13` (`delta_stock_return=0`), `13` nonzero rewards, `0` positive rewards,
  dominant action `5` share `1.0`, entropy `0`, strict load true, fallback
  false, stock/manual match true.
- Plain read: no quality gain at `iteration_1000`. It reaches the same cap,
  same return, same reward counts, and remains fully collapsed; only the
  collapsed action changes from `2` to `5`.
- Pre-patch detached frequent-checkpoint attempt:
  `train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath`.
- Five-point frequent-checkpoint strict eval app:
  `ap-6kQnKAUcF21SN5zqEAG3eZ`.
- Requested five-point eval id:
  `live-32768-ckpt1000-detached-iteration0-1000-2000-3000-4000-stockish512-stockeval-s0`.
  The emitted Modal eval directory was truncated to
  `live-32768-ckpt1000-detached-iteration0-1000-2000-3000-4000-stockish512-stockeva`.
- Five-point frequent-checkpoint strict eval manifest:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath/eval/live-32768-ckpt1000-detached-iteration0-1000-2000-3000-4000-stockish512-stockeva/manifest_custom_steps512_seed0_20260510T011234Z.json`
- Local fetched manifest copy:
  `artifacts/local/lightzero-eval-manifests/live-32768-ckpt1000-detached-iteration0-1000-2000-3000-4000-stockish512-stockeva/manifest_custom_steps512_seed0_20260510T011234Z.json`
- Local summary:
  `artifacts/local/lightzero-eval-manifests/live-32768-ckpt1000-detached-iteration0-1000-2000-3000-4000-stockish512-stockeva/summary_baseline_deltas.tsv`
- Claim: `iteration_0`, `iteration_1000`, `iteration_2000`,
  `iteration_3000`, and `iteration_4000` all strict-loaded with no fallback,
  survived `512/512`, and ran the stock evaluator under the stock-ish 512-step
  eval contract.
- Non-claim: this does not show durable learning progress through
  `iteration_4000`; survival, stock return, and positive rewards do not improve
  versus same-run `iteration_0`, and action behavior remains collapsed or near
  collapsed.
- Survival-first result: all five checkpoints survived `512/512`, so
  `delta_steps_survived=0` and `survival_fraction=1.0` for every row.
  `iteration_0`: return `-13`, stock return `-13`, `13` nonzero rewards, `0`
  positive rewards, dominant action `2` share `1.0`, entropy `0`, verdict
  `collapsed_action`. `iteration_1000`: return `-11`
  (`delta_return=+2`), stock return `-13` (`delta_stock_return=0`), `11`
  nonzero rewards, `0` positive rewards, dominant action `5` share
  `0.869141`, entropy `0.312069`, verdict `manual_stock_mismatch`.
  `iteration_2000`: return `-13`, stock return `-13`, `13` nonzero rewards,
  `0` positive rewards, dominant action `0` share `1.0`, entropy `0`, verdict
  `collapsed_action`. `iteration_3000`: return `-13`, stock return `-13`,
  `13` nonzero rewards, `0` positive rewards, dominant action `5` share
  `0.787109`, entropy `0.746972`, verdict `manual_stock_mismatch`.
  `iteration_4000`: return `-13`, stock return `-13`, `13` nonzero rewards,
  `0` positive rewards, dominant action `0` share `0.916016`, entropy
  `0.281851`, verdict `manual_stock_mismatch`.
- Plain read: the 512-step five-point curve is flat through `iteration_4000`
  on survival and stock return. `iteration_1000` has a small manual-return
  bump, but the stock evaluator does not confirm it and the later checkpoints
  return to the initialization-level `-13`.
- Three-point frequent-checkpoint strict eval app:
  `ap-pFmgqUPbjkDnal4bh7yHJE`.
- Three-point frequent-checkpoint strict eval manifest:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath/eval/live-32768-ckpt1000-detached-iteration0-vs-1000-vs-2000-stockish512-stockeval-s0/manifest_custom_steps512_seed0_20260510T010137Z.json`
- Local fetched manifest copy:
  `artifacts/local/lightzero-eval-manifests/live-32768-ckpt1000-detached-iteration0-vs-1000-vs-2000-stockish512-stockeval-s0/manifest_custom_steps512_seed0_20260510T010137Z.json`
- Local summary:
  `artifacts/local/lightzero-eval-manifests/live-32768-ckpt1000-detached-iteration0-vs-1000-vs-2000-stockish512-stockeval-s0/summary_baseline_deltas.tsv`
- Claim: `iteration_0`, `iteration_1000`, and `iteration_2000` all
  strict-loaded with no model fallback under stock-ish 512-step eval, with
  stock evaluator enabled.
- Non-claim: this does not show a quality gain by `iteration_2000`;
  survival, manual return, stock return, nonzero reward count, and positive
  reward count did not improve versus same-run `iteration_0`.
- Survival-first result: `iteration_0` survived `512/512`, return `-13`,
  stock return `-13`, `13` nonzero rewards, `0` positive rewards, dominant
  action `2` share `1.0`, entropy `0`, strict load true, fallback false.
  `iteration_1000` survived `512/512`, `delta_steps_survived=0`, return
  `-13` (`delta_return=0`), stock return `-10`
  (`delta_stock_return=+3`), `13` nonzero rewards, `0` positive rewards,
  dominant action `5` share `0.886719`, entropy `0.319986`, strict load true,
  fallback false, manual/stock mismatch. `iteration_2000` survived
  `512/512`, `delta_steps_survived=0`, return `-13`
  (`delta_return=0`), stock return `-13` (`delta_stock_return=0`),
  `13` nonzero rewards, `0` positive rewards, dominant action `0` share
  `1.0`, entropy `0`, strict load true, fallback false.
- Plain read: the curve point remains flat through `iteration_2000` on
  survival and manual/stock return. `iteration_1000` has a one-off stock
  evaluator improvement with manual/stock mismatch, but `iteration_2000`
  returns to the same `-13` stock/manual result as initialization and is fully
  action-collapsed.
- First frequent-checkpoint strict eval app:
  `ap-3JRlszZqkzuBCPyUvgXvdM`.
- First frequent-checkpoint strict eval manifest:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath/eval/live-32768-ckpt1000-detached-iteration0-vs-1000-stockish512-stockeval-s0/manifest_custom_steps512_seed0_20260510T005617Z.json`
- Local fetched manifest copy:
  `artifacts/local/lightzero-eval-manifests/live-32768-ckpt1000-detached-iteration0-vs-1000-stockish512-stockeval-s0/manifest_custom_steps512_seed0_20260510T005617Z.json`
- Local summary:
  `artifacts/local/lightzero-eval-manifests/live-32768-ckpt1000-detached-iteration0-vs-1000-stockish512-stockeval-s0/summary_baseline_deltas.tsv`
- Claim: the first pre-patch detached frequent-checkpoint eval loaded both
  `iteration_0` and `iteration_1000` strictly, with no model fallback, and both
  survived `512/512` eval steps.
- Non-claim: this does not show learning progress; `iteration_1000` did not
  improve steps, manual return, stock return, reward count, or positive reward
  count versus same-run `iteration_0`.
- Survival-first result: `iteration_0` survived `512/512`, return `-13`,
  stock return `-13`, `13` nonzero rewards, `0` positive rewards, dominant
  action `0` share `0.535156`, entropy `0.790535`, strict load true, fallback
  false. `iteration_1000` survived `512/512`,
  `delta_steps_survived=0`, return `-13` (`delta_return=0`), stock return
  `-13` (`delta_stock_return=0`), `13` nonzero rewards, `0` positive rewards,
  dominant action `5` share `0.90625`, entropy `0.275348`, strict load true,
  fallback false.
- Plain read: no quality gain at `iteration_1000`. It reaches the same cap and
  same return as initialization, while the policy becomes more collapsed around
  action `5`.
- Attempt:
  `train-faithful-short-installed-0.2.0-s0-32768-relpath`
- Train app: `ap-xiGLACKHPZLvL1eYgygqvm`
- Live eval app for the initial checkpoint: `ap-toWJGpg0nz4MnizyAoopVs`
- GPU live eval proof app for the initial checkpoint:
  `ap-3icJTrptdJEw38GZAoK5wx`
- Existing checkpoints at live-eval launch:
  `iteration_0.pth.tar` and `ckpt_best.pth.tar`
- Later checkpoint now visible: `iteration_9092.pth.tar`.
- Strict same-run eval app for `iteration_0` versus `iteration_9092`:
  `ap-cAxSfqSoXuE7k9q1lAZJPM`.
- Strict same-run eval manifest:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/eval/live-32768-iteration0-vs-9092-stockish512-stockeval-s0/manifest_custom_steps512_seed0_20260510T003157Z.json`
- Baseline-delta summary: `iteration_0` survived `512/512`, manual return
  `-13`, stock return `-13`, `13` nonzero rewards, `0` positive rewards,
  dominant action `0` share `0.503906`, entropy `0.766302`, strict load true,
  fallback false. `iteration_9092` survived `512/512`,
  `delta_steps_survived=0`, manual return `-8` (`delta_return=+5`), stock
  return `-10` (`delta_stock_return=+3`), `8` nonzero rewards, `0` positive
  rewards, dominant action `2` share `0.300781`, entropy `0.911185`, strict
  load true, fallback false.
- Plain read: this is not solved Pong, but it is a real same-run improvement
  over `iteration_0` on loss count and return. Survival cannot improve here
  because both rows already hit the 512-step cap. Manual/stock mismatch is
  still present in both rows, so keep reporting both values.
- Queue helper check for eval id `live-32768-later-stockish512-stockeval-s0`
  found `1` iteration checkpoint and `1` pending eval command, but that pending
  command was only for `iteration_0.pth.tar`; it was intentionally not run
  because the task is to evaluate later checkpoints beyond `iteration_0`.
- CPU live eval result for `iteration_0`: strict load true, fallback false,
  manual/raw return `-13`, stock return `-12`, `512` episode steps survived,
  `13` nonzero rewards, `0` positive rewards, survived longer than
  `iteration_0`: baseline/not applicable, dominant action `0` share
  `0.519531`, entropy `0.809641`, manual/stock mismatch still present.
- CPU eval manifest:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/eval/live-32768-iteration0-stockish512-stockeval-s0/manifest_custom_steps512_seed0_20260509T235146Z.json`
- GPU live eval proof result for `iteration_0`: compute `gpu-l4-t4`,
  strict load true, fallback false, CUDA true on NVIDIA L4, manual/raw return
  `-13`, stock return `-13`, `512` episode steps survived, positive rewards
  `0`, `13` nonzero rewards, survived longer than `iteration_0`:
  baseline/not applicable, action `2` for all steps, verdict
  `collapsed_action`.
- GPU proof eval manifest:
  `training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/eval/faithful-short-32768-live-gpu-stockeval-s0/manifest_low_steps512_seed0_20260509T235540Z.json`

Plain read: live eval is now useful for both `iteration_0` and
`iteration_9092`. The next useful live eval is the next later same-run normal
`iteration_*.pth.tar` once it appears in the checkpoint folder.

Plain read of the first same-run comparison: `iteration_9092` is weak but
better than the starting checkpoint. It reached the same 512-step cap as
`iteration_0`, lost fewer points, and improved both manual and stock return.
This is a small positive same-run learning signal, not a solve.

## Long-Horizon 2048-Step Eval

Repeat A first-checkpoint eval app:
`ap-jpok3tK4hpuMmScE4IHokr`.

Repeat A first-checkpoint eval id:
`repeatA-s0-32768-iteration0-vs-1000-stockish2048-stockeval-s0`.

Repeat A first-checkpoint manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath/eval/repeatA-s0-32768-iteration0-vs-1000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T012006Z.json`.

Repeat A first-checkpoint local manifest:
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-iteration0-vs-1000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T012006Z.json`.

Repeat A first-checkpoint local summary:
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-iteration0-vs-1000-stockish2048-stockeval-s0/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `761/2048`,
manual return `-21`, stock return `-21`, `21` nonzero rewards, `0` positive
rewards, strict load true, fallback false, dominant action `2` share `1.0`,
entropy `0`, verdict `collapsed_action`. `iteration_1000` survived
`761/2048`, `delta_steps_survived=0`, manual return `-21`,
stock return `-21`, `21` nonzero rewards, `0` positive rewards, strict load
true, fallback false, dominant action `1` share `1.0`, entropy `0`, verdict
`collapsed_action`.

Claim: repeat A `iteration_1000` gives no long-horizon improvement over its
same-run `iteration_0` baseline on survival, manual return, stock return, or
reward counts.

Non-claim: this does not prove the repeat A run cannot improve at later
checkpoints. It only says the first checkpoint is flat and action-collapsed
under strict 2048-step stock-evaluator eval.

Repeat A second-checkpoint eval app:
`ap-oNRa4eVcbLiIIg4h7qG8nw`.

Repeat A second-checkpoint eval id:
`repeatA-s0-32768-iteration0-vs-2000-stockish2048-stockeval-s0`.

Repeat A second-checkpoint manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath/eval/repeatA-s0-32768-iteration0-vs-2000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T012735Z.json`.

Repeat A second-checkpoint local manifest:
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-iteration0-vs-2000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T012735Z.json`.

Repeat A second-checkpoint local summary:
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-iteration0-vs-2000-stockish2048-stockeval-s0/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `761/2048`, manual return
`-21`, stock return `-21`, `21` nonzero rewards, `0` positive rewards, strict
load true, fallback false, dominant action `2` share `1.0`, entropy `0`,
verdict `collapsed_action`. `iteration_2000` survived `761/2048`,
`delta_steps_survived=0`, manual return `-21`, stock return `-21`, `21`
nonzero rewards, `0` positive rewards, strict load true, fallback false,
stock/manual match true, action histogram `{"0": 443, "2": 318}`, dominant
action `0` share `0.582129`, entropy `0.980449`, verdict `negative_return`.

Claim: repeat A `iteration_2000` has no survival, manual-return,
stock-return, reward-count, or positive-reward improvement over same-run
`iteration_0` under strict no-fallback stock-evaluator eval.

Non-claim: this is not evidence that the checkpoint is still fully
action-collapsed. Unlike `iteration_0` and `iteration_1000`, it uses actions
`0` and `2`; however, the survival-first quality metrics remain flat.

Repeat A compact latest-curve eval apps:
`ap-hKrDI5A93bIG3vGorxuEkT` and `ap-qHvUiewiwJ9nc4kJf7Z8kf`.

Repeat A compact latest-curve eval ids:
`repeatA-s0-32768-iteration0-1000-2000-9559-stockish2048-stockeval-s0`
and `repeatA-s0-32768-iteration5000-9000-stockish2048-stockeval-s0`.

Repeat A compact latest-curve manifests:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath/eval/repeatA-s0-32768-iteration0-1000-2000-9559-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T034656Z.json`
and
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-repeatA-ckpt1000-spawn-relpath/eval/repeatA-s0-32768-iteration5000-9000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T035104Z.json`.

Local compact summary:
`artifacts/local/lightzero-eval-manifests/repeatA-s0-32768-compact-curve-stockish2048-stockeval-s0-summary.tsv`.

Survival-first compact curve, all strict load true and fallback false:
`iteration_0` survived `761/2048`, baseline, not saturated, manual return
`-21`, stock return `-21`, positive rewards `0`, negative/nonzero rewards
`21/21`, action `2` only, collapsed. `iteration_1000` survived `761/2048`,
`delta_steps_survived=0`, not saturated, manual/stock return `-21/-21`,
positive rewards `0`, negative/nonzero rewards `21/21`, action `1` only,
collapsed. `iteration_5000` survived `761/2048`, `delta_steps_survived=0`,
not saturated, manual/stock return `-21/-21`, positive rewards `0`,
negative/nonzero rewards `21/21`, all six actions, dominant action `2` share
`0.529566`. `iteration_9000` survived `761/2048`,
`delta_steps_survived=0`, not saturated, manual/stock return `-21/-21`,
positive rewards `0`, negative/nonzero rewards `21/21`, all six actions,
dominant action `2` share `0.574244`. Latest `iteration_9559` survived
`973/2048`, `delta_steps_survived=+212`, not saturated, manual return `-21`,
stock return `-17`, positive rewards `0`, negative/nonzero rewards `21/21`,
all six actions, dominant action `1` share `0.34224`.

Claim: repeat A latest visible checkpoint `iteration_9559` gives the first
positive long-cap same-run signal for this repeat: `+212` manual survival
steps over `iteration_0` and stock return `-17` versus `-21`, with no fallback
and strict checkpoint load. `iteration_1000`, `iteration_5000`, and
`iteration_9000` remain flat on survival, return, and positive rewards.

Non-claim: this is not solved Pong, not cap survival, and not a clean
manual/stock-parity result. Manual return remains `-21`, positive rewards stay
`0`, and the later non-collapsed checkpoints have stock/manual action-prefix
mismatch.

First-priority eval app:
`ap-xeNj8SCLjyV7C42hLioF0C`.

First-priority eval id:
`live-32768-relpath-iteration0-vs-9092-stockish2048-stockeval-s0`.

First-priority manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/eval/live-32768-relpath-iteration0-vs-9092-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T010554Z.json`.

First-priority local summary:
`artifacts/local/lightzero-eval-manifests/live-32768-relpath-iteration0-vs-9092-stockish2048-stockeval-s0/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `761/2048`
(`survival_fraction=0.371582`), did not survive to cap, manual return `-21`,
stock return `-21`, `21` nonzero rewards, `0` positive rewards, strict load
true, fallback false, dominant action `2` share `1.0`, entropy `0.0`, verdict
`collapsed_action`. `iteration_9092` survived `821/2048`
(`survival_fraction=0.400879`), did not survive to cap,
`delta_steps_survived=+60`, manual return `-21` (`delta_return=0`), stock
return `-21` (`delta_stock_return=0`), `21` nonzero rewards, `0` positive
rewards, strict load true, fallback false, dominant action `5` share
`0.595615`, entropy `0.688619`, verdict `manual_stock_mismatch`.

Claim: the longer cap breaks the old 512-step survival saturation. Same-run
`iteration_9092` survives longer than same-run `iteration_0` by `60` steps
under strict no-fallback stock-evaluator eval.

Non-claim: this is not a return improvement beyond the 512-step read and not
solved Pong. Both checkpoints still lose the full `21` points, stock return and
manual return are both `-21`, and neither reaches the `2048` cap or gets a
positive reward.

Second-priority eval app:
`ap-9i3xjPKQ9814q3BuoHWwGT`.

Requested second-priority eval id:
`live-32768-ckpt1000-detached-iteration0-vs-1000-vs-2000-stockish2048-stockeval-s0`.
The emitted Modal eval directory was truncated to
`live-32768-ckpt1000-detached-iteration0-vs-1000-vs-2000-stockish2048-stockeval-s`.

Second-priority manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath/eval/live-32768-ckpt1000-detached-iteration0-vs-1000-vs-2000-stockish2048-stockeval-s/manifest_custom_steps2048_seed0_20260510T011028Z.json`.

Second-priority local summary:
`artifacts/local/lightzero-eval-manifests/live-32768-ckpt1000-detached-iteration0-vs-1000-vs-2000-stockish2048-stockeval-s/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `761/2048`, manual return `-21`,
stock return `-21`, `21` nonzero rewards, `0` positive rewards, strict load
true, fallback false, dominant action `2` share `1.0`, verdict
`collapsed_action`. `iteration_1000` survived `789/2048`,
`delta_steps_survived=+28`, manual return `-21`, stock return `-21`, `21`
nonzero rewards, `0` positive rewards, strict load true, fallback false,
dominant action `5` share `0.891001`, entropy `0.268514`, verdict
`manual_stock_mismatch`. `iteration_2000` survived `761/2048`,
`delta_steps_survived=0`, manual return `-21`, stock return `-21`, `21`
nonzero rewards, `0` positive rewards, strict load true, fallback false,
dominant action `0` share `1.0`, verdict `collapsed_action`.

Claim: in the pre-patch frequent run, `iteration_1000` has a small
long-horizon survival gain over same-run `iteration_0`; `iteration_2000` does
not.

Non-claim: the pre-patch frequent run does not show return learning under the
2048 cap. All three rows still end at `-21` stock/manual return, with zero
positive rewards and no survival to cap.

Six-point frequent-checkpoint eval app:
`ap-Y7hGN6KCX83rizv3ZjLm3N`.

Requested six-point eval id:
`live-32768-ckpt1000-detached-iteration0-1000-2000-3000-4000-5000-stockish2048-stockeval-s0`.
The emitted Modal eval directory was truncated to
`live-32768-ckpt1000-detached-iteration0-1000-2000-3000-4000-5000-stockish2048-st`.

Six-point manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath/eval/live-32768-ckpt1000-detached-iteration0-1000-2000-3000-4000-5000-stockish2048-st/manifest_custom_steps2048_seed0_20260510T011837Z.json`.

Six-point local summary:
`artifacts/local/lightzero-eval-manifests/live-32768-ckpt1000-detached-iteration0-1000-2000-3000-4000-5000-stockish2048-st/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `761/2048`,
manual return `-21`, stock return `-21`, `21` nonzero rewards, `0` positive
rewards, strict load true, fallback false, dominant action `2` share `1.0`,
entropy `0.0`, verdict `collapsed_action`. `iteration_1000` survived
`789/2048`, `delta_steps_survived=+28`, manual return `-21`, stock return
`-21`, dominant action `5` share `0.869455`, entropy `0.302628`, verdict
`manual_stock_mismatch`. `iteration_2000` survived `761/2048`,
`delta_steps_survived=0`, manual/stock return `-21`, dominant action `0`
share `1.0`, entropy `0.0`, verdict `collapsed_action`. `iteration_3000`
survived `761/2048`, `delta_steps_survived=0`, manual/stock return `-21`,
dominant action `5` share `0.784494`, entropy `0.751876`, verdict
`manual_stock_mismatch`. `iteration_4000` survived `761/2048`,
`delta_steps_survived=0`, manual/stock return `-21`, dominant action `0`
share `0.973719`, entropy `0.118432`, verdict `manual_stock_mismatch`.
`iteration_5000` survived `1001/2048`
(`survival_fraction=0.48877`), `delta_steps_survived=+240`, manual return
`-21`, stock return `-21`, `21` nonzero rewards, `0` positive rewards, strict
load true, fallback false, dominant action `0` share `0.92008`, entropy
`0.282697`, verdict `manual_stock_mismatch`.

Claim: in the pre-patch frequent run, `iteration_5000` has the best
long-horizon survival in this six-point eval, surviving `240` steps longer
than same-run `iteration_0` under strict no-fallback stock-evaluator eval.

Non-claim: this does not show return learning or solved Pong. All six rows
still finish at `-21` manual and stock return, with `21` nonzero negative
rewards, `0` positive rewards, no survival to the `2048` cap, and collapsed or
near-collapsed action usage.

Four-point continuation eval app:
`ap-dqqhBECIpveD2LP6aBKrOv`.

Four-point continuation eval id:
`detached-s0-0-5000-6000-7000-stockish2048-stockeval-s0`.

Four-point continuation manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath/eval/detached-s0-0-5000-6000-7000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T012812Z.json`.

Four-point continuation local manifest:
`artifacts/local/lightzero-eval-manifests/detached-s0-0-5000-6000-7000-stockish2048-stockeval-s0/manifest_custom_steps2048_seed0_20260510T012812Z.json`.

Four-point continuation local summary:
`artifacts/local/lightzero-eval-manifests/detached-s0-0-5000-6000-7000-stockish2048-stockeval-s0/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `761/2048`,
manual return `-21`, stock return `-21`, `21` nonzero negative rewards,
`0` positive rewards, strict load true, fallback false, stock/manual match
true, action histogram `{2: 761}`, dominant action `2` share `1.0`, entropy
`0.0`, verdict `collapsed_action`. `iteration_5000` survived `941/2048`
(`delta_steps_survived=+180`), manual return `-21`, stock return `-21`,
`21` nonzero negative rewards, `0` positive rewards, strict load true,
fallback false, stock/manual match false, action histogram
`{0: 877, 4: 15, 5: 49}`, dominant action `0` share `0.931987`, entropy
`0.259875`, verdict `manual_stock_mismatch`. `iteration_6000` survived
`821/2048` (`delta_steps_survived=+60`), manual return `-21`, stock return
`-21`, `21` nonzero negative rewards, `0` positive rewards, strict load true,
fallback false, stock/manual match false, action histogram
`{0: 803, 1: 1, 4: 10, 5: 7}`, dominant action `0` share `0.978076`,
entropy `0.08957`, verdict `manual_stock_mismatch`. `iteration_7000`
survived `821/2048` (`delta_steps_survived=+60`), manual return `-21`,
stock return `-21`, `21` nonzero negative rewards, `0` positive rewards,
strict load true, fallback false, stock/manual match false, action histogram
`{0: 668, 2: 12, 4: 15, 5: 126}`, dominant action `0` share `0.813642`,
entropy `0.425836`, verdict `manual_stock_mismatch`.

Claim: in the pre-patch detached frequent run, survival does not keep improving
past `iteration_5000`; under this strict no-fallback stock-evaluator
2048-step continuation eval, `iteration_6000` and `iteration_7000` both fall
back to `821/2048`, only `+60` versus same-run `iteration_0` and below
`iteration_5000` at `941/2048`.

Non-claim: this does not show return learning, positive rewards, a solved Pong
policy, or a stable exact magnitude for the earlier `iteration_5000` bump. The
previous six-point manifest measured `iteration_5000` at `1001/2048`; this
repeat measured it at `941/2048`. In both reads, later `6000/7000` checkpoints
do not exceed the `5000` survival bump, all rows stay at manual/stock return
`-21`, and action usage remains collapsed or near collapsed.

Final detached long-cap eval id:
`detached-s0-final-0-5000-7000-8000-9000-9086-stockish2048-stockeval-s0`.

Final detached eval root:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath/eval/detached-s0-final-0-5000-7000-8000-9000-9086-stockish2048-stockeval-s0`.

Final detached local per-checkpoint artifacts and summary:
`artifacts/local/lightzero-eval-manifests/detached-s0-final-0-5000-7000-8000-9000-9086-stockish2048-stockeval-s0/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `761/2048`, did not reach the
cap, manual return `-21`, stock return `-21`, `21` nonzero negative rewards,
`0` positive rewards, strict load true, fallback false, stock/manual match
true, action histogram `{2: 761}`, dominant action `2` share `1.0`, entropy
`0.0`, verdict `collapsed_action`. `iteration_5000` survived `1001/2048`
(`delta_steps_survived=+240`), did not reach the cap, manual return `-21`,
stock return `-21`, `21` nonzero negative rewards, `0` positive rewards,
strict load true, fallback false, stock/manual match false, action histogram
`{0: 927, 4: 16, 5: 58}`, dominant action `0` share `0.926074`, entropy
`0.275141`, verdict `manual_stock_mismatch`. `iteration_7000` survived
`821/2048` (`delta_steps_survived=+60`), did not reach the cap, manual/stock
return `-21`, `21` nonzero negative rewards, `0` positive rewards, strict load
true, fallback false, stock/manual match false, action histogram
`{0: 668, 2: 13, 4: 8, 5: 132}`, dominant action `0` share `0.813642`,
entropy `0.412921`. `iteration_8000` survived `761/2048`
(`delta_steps_survived=0`), manual/stock return `-21`, action histogram
`{0: 446, 1: 315}`, dominant action `0` share `0.586071`, entropy
`0.978518`. `iteration_9000` survived `761/2048`
(`delta_steps_survived=0`), manual return `-21`, stock return `-20`,
`21` nonzero negative rewards, `0` positive rewards, action histogram
`{0: 57, 1: 564, 3: 40, 4: 40, 5: 60}`, dominant action `1` share
`0.74113`, entropy `0.575417`. `iteration_9086` survived `761/2048`
(`delta_steps_survived=0`), manual/stock return `-21`, `21` nonzero negative
rewards, `0` positive rewards, action histogram `{0: 648, 3: 5, 5: 108}`,
dominant action `0` share `0.851511`, entropy `0.406866`. All final-detached
rows strict-loaded, used no model fallback, ran stock evaluator, and failed to
survive to the `2048` cap.

Claim: the pre-patch detached frequent run's later long-cap checkpoints do not
extend the best survival bump. In the final long-cap eval, `iteration_5000`
remains the high-survival point at `1001/2048` (`+240` versus same-run
`iteration_0`), while `iteration_7000` is only `821/2048` (`+60`) and
`iteration_9000`/`iteration_9086` return to baseline survival at `761/2048`.

Non-claim: this is not return learning, positive-reward learning, solved Pong,
or stable non-collapsed control. Manual return stays `-21` for every row,
positive rewards stay `0`, nonzero rewards are all negative, stock return only
shows a one-point `iteration_9000` bump (`-20`) with stock/manual mismatch, and
action use remains collapsed or dominated despite changing dominant actions.

## Matrix Seed 2 2048-Step Eval

Eval id:
`matrix-s2-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s2`.

App:
`ap-9FWkydIAb4ZXuFTaasudPl`.

Run:
`lz-visual-pong-exact-installed-0.2.0-s2`.

Attempt:
`train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath`.

Manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s2/attempts/train-faithful-short-installed-0.2.0-s2-65536-ckpt1000-spawn-relpath/eval/matrix-s2-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s2/manifest_custom_steps2048_seed2_20260510T011941Z.json`.

Local fetched manifest copy:
`artifacts/local/lightzero-eval-manifests/matrix-s2-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s2/manifest_custom_steps2048_seed2_20260510T011941Z.json`.

Local summary:
`artifacts/local/lightzero-eval-manifests/matrix-s2-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s2/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `762/2048`
(`survival_fraction=0.37207`), did not survive to cap, manual return `-21`,
stock return `-21`, `21` nonzero rewards, `0` positive rewards, strict load
true, fallback false, stock/manual match true, dominant action `1` share `1.0`,
entropy `0`, verdict `collapsed_action`. `iteration_1000` survived `762/2048`
(`delta_steps_survived=0`, `survival_fraction=0.37207`), did not survive to
cap, manual return `-21` (`delta_return=0`), stock return `-21`
(`delta_stock_return=0`), `21` nonzero rewards, `0` positive rewards, strict
load true, fallback false, stock/manual match false, dominant action `1` share
`0.838583`, entropy `0.430975`, verdict `manual_stock_mismatch`.

Claim: seed 2 L4/T4 first-checkpoint eval gives a valid strict no-fallback
stock-evaluator 2048-cap comparison for `iteration_0` versus `iteration_1000`.

Non-claim: no learning gain is visible at `iteration_1000`; survival, manual
return, stock return, nonzero reward count, and positive reward count are flat
against same-run `iteration_0`. Action collapse softens from a pure single
action to a dominant-action policy, but that is not a quality claim.

## Matrix Seed 3 2048-Step Eval

Eval id:
`matrix-s3-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s3`.

App:
`ap-bbpTz7z1tzXRgTNiG4QH2o`.

Run:
`lz-visual-pong-exact-installed-0.2.0-s3`.

Attempt:
`train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath`.

Manifest:
`training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s3/attempts/train-faithful-short-installed-0.2.0-s3-65536-ckpt1000-spawn-relpath/eval/matrix-s3-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s3/manifest_custom_steps2048_seed3_20260510T012148Z.json`.

Local fetched manifest copy:
`artifacts/local/lightzero-eval-manifests/matrix-s3-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s3/manifest_custom_steps2048_seed3_20260510T012148Z.json`.

Local summary:
`artifacts/local/lightzero-eval-manifests/matrix-s3-65536-l4t4-iteration0-vs-1000-stockish2048-stockeval-s3/summary_baseline_deltas.tsv`.

Survival-first result: `iteration_0` survived `762/2048`
(`survival_fraction=0.37207`), did not survive to cap, manual return `-21`,
stock return `-21`, `21` nonzero negative rewards, `0` positive rewards,
strict load true, fallback false, stock/manual match false, dominant action
`1` share `0.875328`, entropy `0.542643`, verdict
`manual_stock_mismatch`. `iteration_1000` survived `790/2048`
(`delta_steps_survived=+28`, `survival_fraction=0.385742`), did not survive
to cap, manual return `-21` (`delta_return=0`), stock return `-21`
(`delta_stock_return=0`), `21` nonzero negative rewards, `0` positive
rewards, strict load true, fallback false, stock/manual match false, dominant
action `4` share `0.663291`, entropy `0.630592`, verdict
`manual_stock_mismatch`.

Claim: seed 3 L4/T4 first-checkpoint eval gives a valid strict no-fallback
stock-evaluator 2048-cap comparison for `iteration_0` versus
`iteration_1000`. Because 512-step evals can saturate, the meaningful
survival signal is the non-saturated longer-cap result:
`iteration_1000` survives `790/2048`, which is `+28` steps over same-run
`iteration_0` at `762/2048`.

Non-claim: this is not solved Pong and not return learning. Both checkpoints
still finish at `-21` manual and stock return, with `21` negative rewards,
`0` positive rewards, and no survival to the `2048` cap. The action
distribution is less collapsed at `iteration_1000`, but that is only telemetry
unless paired with durable return or longer-horizon survival gains.

## Current Next Actions

1. Poll the active checkpoint directories for the next normal checkpoint after
   the already evaluated same-run points: old relpath `iteration_9092.pth.tar`
   and detached frequent `iteration_5000.pth.tar`.
2. Eval each new normal checkpoint once with the same strict no-fallback
   stock-ish 512-step contract. Do not use `ckpt_best` for quality unless its
   checkpoint state proves it is trained.
3. Summarize with baseline deltas against same-run `iteration_0` and lead with
   `steps_survived / eval_cap_steps`, saturation status, and
   `delta_steps_survived`; then report manual/raw return, stock return,
   positive and negative rewards, action collapse, CPU/GPU placement,
   checkpoint ref, and manifest ref. If a 512-step eval saturates, use/report
   the 2048-cap result as the meaningful survival signal.
4. State the current learning claim plainly: old relpath `iteration_9092`
   improved loss count and return versus same-run `iteration_0`; detached
   frequent `iteration_5000` improved long-cap survival by `240` steps, but
   still did not improve manual/stock return or positive rewards.

Tooling note: the eval manifest and
`scripts/summarize_lightzero_pong_eval_manifest.py` already expose
`eval_cap_steps`, `steps_survived`, `survival_fraction`, legacy
`survival_rate`, `survived_to_cap`, `nonzero_reward_count`,
`positive_reward_count`, and raw return. Survival means steps survived. Fraction
means `steps_survived / eval_cap_steps` and is secondary. `survival_rate` is
kept only as the old name for that same fraction, and old manifests can still be
summarized when they only have `survival_rate`. Use the manifest summarizer's
`--baseline-deltas` flag when comparing multiple same-run checkpoints; it adds
`delta_steps_survived`, `delta_survival_fraction`, `delta_stock_return`,
`delta_return`, and `delta_positive_rewards` against `iteration_0`, or against
the first row if `iteration_0` is absent.

Baseline-support note: official visual Pong eval support currently covers
strict learned-checkpoint eval, same-run baseline deltas against `iteration_0`,
arbitrary prior policy/checkpoint refs via `--checkpoint-refs`, and LightZero's
stock evaluator via `--run-stock-evaluator`. It does not currently expose an
ALE Pong random, no-op, or scripted track-ball baseline mode in
`curvyzero.infra.modal.lightzero_pong_eval_smoke`. Those random/track-ball
baseline scoreboards exist for custom dummy Pong tooling only. Minimal plan:
add an official visual Pong baseline mode that runs the same ALE env/caps/seeds
with named policies `random_uniform`, `noop`, and optionally a simple
screen/ram-based paddle heuristic if feasible, then write the same manifest
fields as checkpoint eval rows so `steps_survived`, return, stock/manual
comparisons, and action histograms are comparable. Until then, use same-run
`iteration_0`, prior checkpoint refs, and stock evaluator return as the
available baselines.

Queue helper note: use `scripts/lightzero_live_eval_queue.py` to list normal
`iteration_*.pth.tar` checkpoints, print strict no-fallback stock-evaluator eval
commands for checkpoints that do not already have an eval output directory, and
print the follow-up manifest fetch plus `--baseline-deltas` summary commands.
The manifest summarizer can now take a fetched eval directory and will read all
`manifest_*.json` files inside it, so one command can compare `iteration_0` and
later same-run manifests in a survival-first table.

Action-collapse note: if greedy eval always chooses one move, that is a
policy-quality failure, not by itself a Modal or reward-plumbing failure. The
dummy Pong shaped runs showed the pattern clearly: the 8-epoch final greedy eval
always chose down, while the 24-epoch final greedy eval always chose up. Tooling
should make this obvious by printing the action histogram for every eval, plus
per-opponent steps survived, raw score, shaped return, and a quick random-action
or non-greedy comparison when collapse is suspected.

Future-run note: the train wrapper now supports
`--save-ckpt-after-iter-override` for short rehearsals that need denser
checkpoint reads. Leaving it unset preserves stock cadence.

## Commands

List checkpoints:

```bash
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/train/lightzero_exp/ckpt
```

Run stock-ish strict eval for one checkpoint:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --parallel --eval-pass custom --eval-id live-32768-iteration0-stockish512-stockeval-s0 --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s0/attempts/train-faithful-short-installed-0.2.0-s0-32768-relpath/train/lightzero_exp/ckpt/iteration_0.pth.tar --run-id lz-visual-pong-exact-installed-0.2.0-s0 --attempt-id train-faithful-short-installed-0.2.0-s0-32768-relpath --seed 0 --max-env-step 200000 --collector-env-num 8 --evaluator-env-num 3 --num-simulations 50 --batch-size 256 --game-segment-length 400 --max-eval-steps 512 --max-episode-steps 512 --step-detail-limit 8 --no-allow-model-fallback --run-stock-evaluator
```

Summarize a fetched multi-checkpoint manifest with survival/return deltas:

```bash
uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
  --baseline-deltas \
  --format tsv \
  --output artifacts/local/lightzero-eval-manifests/<EVAL_ID>.tsv \
  artifacts/local/lightzero-eval-manifests/<MANIFEST_FILE>.json
```

Print the next live eval commands and the follow-up summary commands:

```bash
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id lz-visual-pong-exact-installed-0.2.0-s0 \
  --attempt-id train-faithful-short-installed-0.2.0-s0-32768-relpath \
  --eval-id live-32768-later-stockish512-stockeval-s0 \
  --compute gpu-l4-t4 \
  --eval-pass low \
  --low-detail-max-eval-steps 512 \
  --low-detail-step-detail-limit 8 \
  --max-episode-steps 512
```

Summarize a fetched eval directory instead of one manifest file:

```bash
uv run python scripts/summarize_lightzero_pong_eval_manifest.py \
  --baseline-deltas \
  --format tsv \
  artifacts/local/lightzero-eval-manifests/<EVAL_ID>
```

## GPU Path

The eval wrapper runs stock-ish eval on CPU unless the GPU path is explicitly
selected. The cheap GPU path has a proof result: app
`ap-3icJTrptdJEw38GZAoK5wx`, compute `gpu-l4-t4`, strict load true, fallback
false, CUDA true on NVIDIA L4, manual/raw return `-13`, stock return `-13`,
`512` episode steps survived, `13` nonzero rewards, `0` positive rewards,
survived longer than `iteration_0`: baseline/not applicable, all action `2`,
verdict `collapsed_action`.
