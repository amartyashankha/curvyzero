# LightZero Wave11 Diverse Training Wave - 2026-05-10

Last updated: `2026-05-10` full-curve rand8-e artifact read.

Purpose: keep official/control Pong training artifacts flowing while wave10
jobs continue. This is a diverse exploration wave, not seed chasing and not a
claim that Pong is solved.

## 2026-05-11 Control-Lane Checkpoint Audit

Task: check the four focus Pong runs for newer Modal Volume checkpoints beyond
the latest documented full-curve points. Result: no fresh eval launched. The
latest visible checkpoint for each focus run is already covered by the
documented `wave11-*-fullcurve-stockonly-rand8-e` curve.

Exact Modal Volume checkpoint listings used:

```text
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s73-wave11-l4cpu40/attempts/train-normal-wave11-s73-65536-ckpt1000-l4cpu40-relpath/train/lightzero_exp/ckpt
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s74-wave11-l4cpu16-mid131k/attempts/train-normal-wave11-s74-131072-ckpt1000-l4cpu16-relpath/train/lightzero_exp/ckpt
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s76-wave11-h100cpu16-long199k/attempts/train-normal-wave11-s76-199000-ckpt1000-h100cpu16-relpath/train/lightzero_exp/ckpt
uv run --extra modal modal volume ls curvyzero-runs training/lightzero-official-visual-pong/lz-visual-pong-survival-shaped-step0p001-s81-wave11-l4cpu16/attempts/train-survival-shaped-step0p001-wave11-s81-65536-ckpt1000-l4cpu16-relpath/train/lightzero_exp/ckpt
```

| run | latest visible checkpoint | documented latest curve point | decision |
| --- | ---: | ---: | --- |
| s73 normal | `iteration_18098` | `iteration_18098` in `wave11-s73-fullcurve-stockonly-rand8-e` | no launch; latest is covered |
| s74 normal | `iteration_37542` | `iteration_37542` in `wave11-s74-fullcurve-stockonly-rand8-e` | no launch; latest is covered |
| s76 normal | `iteration_53704` | `iteration_53704` in `wave11-s76-fullcurve-stockonly-rand8-e` | no launch; latest is covered |
| s81 shaped side lane | `iteration_20272` | `iteration_20272` in `wave11-s81-fullcurve-stockonly-rand8-e` | no launch; latest is covered |

No fresh eval id was created. If a later checkpoint appears later, use a new
stock-only serious eval via `scripts/lightzero_live_eval_queue.py` with
`--max-eval-steps 2048`, `--max-episode-steps 2048`,
`--num-simulations 50`, `--compute gpu-l4-t4-cpu40`,
`--eval-seed-count 8` or `16`, selected iterations including `0`, latest, and
one/two middle points, and `--max-parallel-launches 64`.

This is the Pong proof lane only. Stock visual Pong survival-time eval remains
the control lane and should not be mixed with CurvyTron scalar adapter or visual
adapter smoke results.

Main metric for eval: stock evaluator `stock_steps_survived` versus the same
run's `iteration_0`. The Pong eval signal leads with survival steps.
Score/return comes after survival, not first. Stock return and positive reward
count are secondary. Survival-shaped runs are side-lane telemetry and cannot
prove the normal proof lane.

Eval trust note: serious eval is `2048` game steps with `50` MCTS simulations
per action. `16` fresh randomized starts per checkpoint can catch large
survival moves, but it is not enough for subtle/stability claims. Action
histograms are collapse telemetry: one action above about `0.95` is suspect,
while broader histograms are encouraging only when survival/return also move.
Record replay information for each randomized-start eval wave.

Seed discipline: avoid fixed-seed obsession. Use reproducible random eval
panels, record the sampler seed and exact sampled seed list, and reuse a fixed
panel only for replay/debug.

## Late Stock-Only Rand16-D Artifact Read

Fetched:

```text
training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s76-wave11-h100cpu16-long199k/attempts/train-normal-wave11-s76-199000-ckpt1000-h100cpu16-relpath/eval/wave11-s76-late-0-15000-18000-stockonly-rand16-d
```

Summarized with:

```text
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --survival-curve --survival-aggregate --format tsv 'artifacts/local/lightzero-eval-manifests/wave11-s76-late-0-15000-18000-stockonly-rand16-d/**/*.json'
```

All rows are stock-only, `16` eval starts per checkpoint, serious
`50`-simulation eval, `max_eval_steps=2048`, and `max_episode_steps=2048`.

| train seed | eval id | eval starts per checkpoint | `iteration_0` mean stock steps | later checkpoint means and deltas | survival-first read |
| ---: | --- | ---: | ---: | --- | --- |
| 76 | `wave11-s76-late-0-15000-18000-stockonly-rand16-d` | `16` | `760.562` | `15000`: `1072.44` (`+311.875`); `18000`: `1093.19` (`+332.625`) | positive survival signal persists through `18000` |

Score is secondary: mean stock return moves from `-21` at `iteration_0` to
`-20.5` at `15000` and `-19.625` at `18000`. The real signal is survival
steps, not that return line.

## Full-Curve Eval Launches

Launched stock-only serious full-curve evals for the current useful Wave11 Pong
runs. These are curve reads, not final proof claims.

Common eval settings:

```text
stock-only: true
allow model fallback: false
eval cap: 2048
episode cap: 2048
search simulations per action: 50
eval starts per checkpoint: 8 fresh pseudo-random starts
compute: gpu-l4-t4-cpu40
group size: 4
max parallel launches: 64
```

| run | eval id | selected checkpoint spread |
| --- | --- | --- |
| s70 normal | `wave11-s70-fullcurve-stockonly-rand8-e` | `0,1000,5000,7000,8000,10000,13000,16000,17582` |
| s71 normal | `wave11-s71-fullcurve-stockonly-rand8-e` | `0,1000,5000,7000,8000,10000,13000,16000,18000,18287` |
| s72 normal | `wave11-s72-fullcurve-stockonly-rand8-e` | `0,1000,5000,7000,8000` |
| s73 normal | `wave11-s73-fullcurve-stockonly-rand8-e` | `0,1000,5000,7000,8000,10000,13000,16000,18000,18098` |
| s74 normal | `wave11-s74-fullcurve-stockonly-rand8-e` | `0,1000,5000,7000,8000,10000,13000,16000,20000,25000,30000,37000,37542` |
| s76 normal | `wave11-s76-fullcurve-stockonly-rand8-e` | `0,1000,5000,7000,10000,13000,16000,18000,20000,25000,30000,40000,50000,53704` |
| s80 shaped side lane | `wave11-s80-fullcurve-stockonly-rand8-e` | `0,1000,5000,7000,10000,13000,16000,17702` |
| s81 shaped side lane | `wave11-s81-fullcurve-stockonly-rand8-e` | `0,1000,5000,7000,10000,13000,16000,20000,20272` |
| s82 shaped side lane | `wave11-s82-fullcurve-stockonly-rand8-e` | `0,1000,5000,7000,10000,13000,14000` |

s75 only had `iteration_0`, so no curve was launched for it yet.

Interpretation rule for the next read: lead with mean `stock_steps_survived`
versus the same run's `iteration_0`. Return and point scoring are secondary.

## Full-Curve Rand8-E Artifact Read

Fetched root `manifest_*.json` files from the `curvyzero-runs` Modal Volume into
`artifacts/local/eval-manifest-files/`, without pulling the large per-episode
JSON trees. Summarized each run with:

```text
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --survival-curve --survival-aggregate --format tsv artifacts/local/eval-manifest-files/sXX_g*.json
```

All rows below are stock-only serious eval, `8` eval starts per checkpoint,
`50` search simulations per action, and a `2048` stock eval cap. Survival steps
are the main signal; return is secondary.

| run | reward type | best mean stock steps | latest mean stock steps | clear read |
| --- | --- | ---: | ---: | --- |
| s70 | normal | `1044.5` at `iteration_13000` | `761.75` at `iteration_17582` | transient lift, then back to baseline |
| s71 | normal | `1351.12` at `iteration_18287` | `1351.12` at `iteration_18287` | late positive survival signal after a long flat start |
| s72 | normal | `816.5` at `iteration_7000` | `759.5` at `iteration_8000` | small transient lift, then back to baseline |
| s73 | normal | `2048` at `iteration_18000` | `1823.38` at `iteration_18098` | strong late normal-run signal; latest stays far above baseline |
| s74 | normal | `2048` at `iteration_30000`/`37000`/`37542` | `2048` at `iteration_37542` | strong late normal-run signal; reaches cap |
| s76 | normal | `1905` at `iteration_40000` | `1786.25` at `iteration_53704` | strong late normal-run signal; latest remains high |
| s80 | survival-shaped `0.0005` | `1458.12` at `iteration_17702` | `1458.12` at `iteration_17702` | shaped side lane is positive but uneven |
| s81 | survival-shaped `0.001` | `2048` at `iteration_20000`/`20272` | `2048` at `iteration_20272` | shaped side lane reaches cap |
| s82 | survival-shaped `0.001` | `898.875` at `iteration_10000` | `857.875` at `iteration_14000` | modest shaped side lift, not a large curve |

Plain read: the normal proof lane is no longer just one favorable curve. s73,
s74, and s76 show large late survival gains on stock evaluator fields, with
s74 reaching the cap and s76 staying high at the latest checkpoint. s71 also
turns positive late. s70 and s72 stay marked unstable because their full curves
fall back near baseline. Do not mix the shaped lanes into the proof claim, but
s80/s81 are useful side telemetry.

## Late Stock-Only Rand16-C Artifact Read

Fetched the two completed late stock-only eval roots from the `curvyzero-runs`
Modal Volume into `/private/tmp/coach_wave11_late_c_fetch_20260510` and
summarized them with:

```text
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --survival-curve --survival-aggregate --format tsv '<local-root>/**/*.json'
```

All rows are stock-only, `16` eval seeds per checkpoint, serious
`50`-simulation eval, `max_eval_steps=2048`, and `max_episode_steps=2048`.

Late survival-first read:

| train seed | eval id | sampler seed | eval starts per checkpoint | `iteration_0` mean stock steps | later checkpoint means and deltas | survival-first read |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 73 | `wave11-s73-late-0-9000-11000-stockonly-rand16-c` | `2452693096008675152` | `16` | `760.375` | `9000`: `1029.25` (`+268.875`); `11000`: `764.125` (`+3.75`) | strong at `9000`, falls back near baseline at `11000` |
| 76 | `wave11-s76-late-0-12000-15000-stockonly-rand16-c` | `4342025071356809948` | `16` | `759.938` | `12000`: `965.375` (`+205.438`); `15000`: `1123.5` (`+363.562`) | strongest current continuing survival lift |

Interpretation: s76 is now the cleanest improving normal proof-lane curve.
s73 still matters because it showed large survival gains at `9000`, but it is
not monotonic and cannot be treated as stable by itself. Survival time remains
the first metric. Score/return remains secondary: s73 falls back to mean stock
return `-21` at `11000`; s76 improves only to mean stock return `-20.5` at
`15000` despite the much clearer survival-time lift.

Comparison note: s76 is not directly comparable to the shorter L4 normal runs.
It is the only Wave11 normal run on the costlier long lane
(`gpu-h100-cpu16`, `max env step=199000`), and the useful checkpoints are much
later (`12000`/`15000`) than the first `7000` gate used for seeds `70`-`74`.
The stock action histograms also look like a checkpoint-age effect: s76 moves
from collapsed `iteration_0` action `0 @ 1.00` to broad distributions at
`12000`/`15000` with dominant action share about `0.41`/`0.38`, while s70 falls
back by `10000` with action `3 @ 0.945`, and s73 falls back by `11000` with
survival near baseline. Current durable hypothesis: s76 may look better because
it has had more useful training horizon, not because H100 itself changes the
learned policy. Confirm by evaluating later checkpoints on s74/s75 or launching
matched long L4/H100 seeds.

## Late Stock-Only Rand16-B Artifact Read

Fetched the three completed late stock-only eval roots from the
`curvyzero-runs` Modal Volume into
`artifacts/local/lightzero-eval-manifests/` and summarized local JSON artifacts
with:

```text
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --survival-curve --survival-aggregate --format tsv '<local-root>/**/*.json'
```

All rows are stock-only, `16` eval seeds per checkpoint, serious
`50`-simulation eval, `max_eval_steps=2048`, and `max_episode_steps=2048`.
The artifacts are `eval_mode=stock_only`, `skip_manual_rollout=true`, and
`allow_model_fallback=false`.

Late survival-first read:

| train seed | eval id | sampler seed | eval starts per checkpoint | `iteration_0` mean stock steps | later checkpoint means and deltas | survival-first read |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 70 | `wave11-s70-late-0-7000-10000-stockonly-rand16-b` | `5204159071529909401` | `16` | `761.375` | `7000`: `790.562` (`+29.1875`); `10000`: `761.375` (`+0`) | transient lift at `7000`, back to baseline by `10000` |
| 73 | `wave11-s73-late-0-7000-9000-stockonly-rand16-b` | `6727270700486715752` | `16` | `760.75` | `7000`: `948.875` (`+188.125`); `9000`: `1035.69` (`+274.938`) | strong continued survival lift |
| 76 | `wave11-s76-late-0-7000-12000-stockonly-rand16-b` | `6116441944006179452` | `16` | `760.25` | `7000`: `847.875` (`+87.625`); `12000`: `919.812` (`+159.562`) | later survival lift on the long normal run |

Interpretation: the late read strengthens the survival-time case for s73 and
adds a positive long-run s76 signal. s70 remains unstable: the `7000` lift does
not persist at `10000` on the fresh rand16-b panel. Survival time remains the
first metric. Score/return remains secondary: s70 returns to mean stock return
`-21` at `10000`; s73 improves only to mean stock return `-20.375` at `9000`;
s76 remains `-20.625` at `12000`.

## 10:20 Wave11 Eval State

Early Wave11 strict stock evals at `iteration_1000` for normal seeds `70`-`74`
showed no survival gain. Across eval seeds `100`-`115`, stock steps survived
stayed around a mean of `760`, so these rows are not proof of learning.

Later checkpoints exist and matter more for the gate. `iteration_5000` and
`iteration_7000` are now being evaluated. Treat the early flat `1000` read as
triage only; the claim still depends on later stock evaluator survival versus
the same run's `iteration_0`.

Eval process correction: the slow path had been running both the manual rollout
and the stock evaluator. Queue eval now defaults to stock-only triage. Use
manual+stock only for debug cases such as parity, action traces, or
manual/stock disagreement.

New eval waves should sample fresh pseudo-random eval seed lists and record
both the RNG seed and exact list. Fixed seed lists are for replay/debug only.

## Final Stock-Only Rand16 Artifact Read

Fetched the five active stock-only eval roots from the `curvyzero-runs` Modal
Volume into `/private/tmp/coach_wave11_eval_fetch_final_20260510` and
summarized local JSON artifacts with:

```text
uv run python scripts/summarize_lightzero_pong_eval_manifest.py --survival-curve --survival-aggregate --format tsv '<wave11-root>/**/*.json'
```

Operator access row:

| date | eval roots | access path | note |
| --- | --- | --- | --- |
| 2026-05-10 | `wave11-s70` through `wave11-s74` stock-only rand16 roots | Fetch each complete eval root once, then summarize the local root with quoted `<local-root>/**/*.json` so root manifests and raw `iteration_*` JSONs are both valid inputs. | Do not resume the old partial-root fetch/list loop; record sampler seed/list for new waves before quoting results. |

This is the complete final `rand16` read for seeds `70`-`74`: all rows are
stock-only, `16` eval seeds per checkpoint, serious `50`-simulation eval
(`num_simulations=50`), `max_eval_steps=2048`, and
`max_episode_steps=2048`. The artifacts are `eval_mode=stock_only`,
`skip_manual_rollout=true`, and `allow_model_fallback=false`.

Survival-first read:

| train seed | eval starts per checkpoint | `iteration_0` mean stock steps | `iteration_1000` | `iteration_5000` | `iteration_7000` | survival-first read |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 70 | `16` | `760.25` | `760.25` (`+0`) | `760.25` (`+0`) | `796.25` (`+36`) | later survival lift at `7000` |
| 71 | `16` | `760.75` | `760.75` (`+0`) | `760.75` (`+0`) | `760.75` (`+0`) | flat |
| 72 | `16` | `761.125` | `761.125` (`+0`) | `775.812` (`+14.6875`) | `820.75` (`+59.625`) | later survival lift, stronger at `7000` |
| 73 | `16` | `760.188` | `760.188` (`+0`) | `803.375` (`+43.1875`) | `933.312` (`+173.125`) | strongest later survival lift |
| 74 | `16` | `760.938` | `760.938` (`+0`) | `760.938` (`+0`) | `760.938` (`+0`) | flat |

Interpretation: `3/5` runs show later survival lift and `2/5` are flat. The
signal is encouraging, especially s73, but it is not enough to declare stable
proof. Survival time remains the first metric; score/return remains secondary.
Stock return is still weak and mostly near `-21`, with only small mean-return
movement on the lifted runs.

Completed eval seeds in the artifacts:

| train seed | visible eval seeds |
| ---: | --- |
| 70 | `13389831,619352951,632646530,696452316,729729305,756034852,992283764,1142470502,1403820476,1505607169,1544267426,1748225240,1845678945,1962051939,2038049252,2129214856` |
| 71 | `142150559,397936544,411835385,754469000,907453940,910654531,1375456296,1403197855,1407265348,1429552667,1607517798,1668970351,1704594199,1746871467,1883166024,2083550031` |
| 72 | `67864893,208851451,330008153,446867830,571985149,572532948,631195095,677023721,718371596,766230552,823530378,884109344,1071906703,1438693512,1826854776,2022339502` |
| 73 | `163627071,303818443,323840513,561440143,767917608,858156884,1124114933,1171561106,1176448733,1213385226,1340300944,1346185523,1684288840,1698159987,1999686176,2065700808` |
| 74 | `62157637,275337055,351191188,357777922,366773960,424985860,676772454,710945619,786140772,991418272,995835331,1222626753,1334659666,1459190888,1492458743,1594674099` |

No eval seed RNG seed was visible in the fetched JSON fields or nearby docs
during this read; only the manifest filenames plus per-artifact `config.seed`
and `episode.seed` values were recorded. Do not promote this as stable proof
without another survival-first robustness read.

## 09:41 Shaped/Randomization Lane Decision

Do not launch more shaped or random-start training before consuming this wave's
first eval pass. The current shaped lane already covers seeds `30`-`37`,
`60`-`61`, and `80`-`82`, including lower shaping at `0.0005` and longer
shaped telemetry. The wrapper has a training seed knob, but no separate
random-start distribution knob, so more variation right now would mostly mean
more seeds and more bookkeeping.

Decision note and the exact fallback `wave12-micro` matrix:
[lightzero_shaped_randomization_lane_decision_2026-05-10.md](lightzero_shaped_randomization_lane_decision_2026-05-10.md).

## 09:46 Run/Eval Watch

Superseded by the final stock-only rand16 read above; retained as historical
launch/eval context.

This was the narrow Modal Volume read before the later `5000/7000` eval pass.
Checked narrow Modal Volume roots only. The five parent-owned local eval
sessions are still treated as external: `53597`, `44182`, `44341`, `47429`,
and `61693`. Do not poll those sessions directly unless their parent context is
available.

Early strict stock evals have been launched for normal training seeds `70`-`74`.
The Volume shows checkpoints through `iteration_5000` for all five. Only seed
`73` had a visible legacy eval root during this check. That eval used a fixed
panel launched before this correction. It is not a standing standard. The other
four eval roots were not visible yet, so treat them as launched/running or not
yet committed, not as failed.

Current checkpoint visibility:

| seed | lane | latest checkpoint visible | eval status |
| ---: | --- | --- | --- |
| 70 | normal | `iteration_5000` | early eval launched; no eval root visible yet |
| 71 | normal | `iteration_5000` | early eval launched; no eval root visible yet |
| 72 | normal | `iteration_5000` | early eval launched; no eval root visible yet |
| 73 | normal | `iteration_5000` | legacy fixed-panel eval root visible for `0,1000`; do not reuse as a default |
| 74 | normal | `iteration_5000` | early eval launched; no eval root visible yet |
| 75 | normal | `iteration_0` | not ready for early gate |
| 76 | normal | `iteration_6000` | checkpoint stream is past the first gate point |
| 80 | shaped | `iteration_5000` | side-lane telemetry only |
| 81 | shaped | `iteration_3000` | side-lane telemetry only |
| 82 | shaped | `iteration_4000` | side-lane telemetry only |

At this point in the day, the final gate still waited for completed strict
stock eval rows at `iteration_5000` or later, including the new
`iteration_7000` rows where available. Checkpoint files alone were not enough.
Use stock `stock_steps_survived` against the same run's `iteration_0` as the
first read.

Going forward, purge fixed-panel thinking. Each eval wave should sample a fresh
pseudo-random eval seed set. Record both the sampler seed and the exact sampled
eval seed list in the launch note or manifest summary so the wave can be
replayed. Do not reuse an old panel just because it appears in an early launch.

## Matrix

All runs use `save_ckpt_after_iter_override=1000`, Modal Volume-backed
artifacts, and fresh run/attempt ids. CPU64 is not used because this Modal
workspace caps function CPU at 40 cores.

| seed | lane | compute | max env step | shaping | run id | attempt id | purpose |
| ---: | --- | --- | ---: | ---: | --- | --- | --- |
| 70 | normal | `gpu-l4-t4-cpu16` | 65536 | 0 | `lz-visual-pong-exact-installed-0.2.0-s70-wave11-l4cpu16` | `train-normal-wave11-s70-65536-ckpt1000-l4cpu16-relpath` | cheap normal seed diversity |
| 71 | normal | `gpu-l4-t4-cpu16` | 65536 | 0 | `lz-visual-pong-exact-installed-0.2.0-s71-wave11-l4cpu16` | `train-normal-wave11-s71-65536-ckpt1000-l4cpu16-relpath` | cheap normal seed diversity |
| 72 | normal | `gpu-l4-t4-cpu40` | 65536 | 0 | `lz-visual-pong-exact-installed-0.2.0-s72-wave11-l4cpu40` | `train-normal-wave11-s72-65536-ckpt1000-l4cpu40-relpath` | CPU40 training throughput check |
| 73 | normal | `gpu-l4-t4-cpu40` | 65536 | 0 | `lz-visual-pong-exact-installed-0.2.0-s73-wave11-l4cpu40` | `train-normal-wave11-s73-65536-ckpt1000-l4cpu40-relpath` | CPU40 training throughput check |
| 74 | normal | `gpu-l4-t4-cpu16` | 131072 | 0 | `lz-visual-pong-exact-installed-0.2.0-s74-wave11-l4cpu16-mid131k` | `train-normal-wave11-s74-131072-ckpt1000-l4cpu16-relpath` | cheap longer-curve check |
| 75 | normal | `gpu-l4-t4-cpu40` | 131072 | 0 | `lz-visual-pong-exact-installed-0.2.0-s75-wave11-l4cpu40-mid131k` | `train-normal-wave11-s75-131072-ckpt1000-l4cpu40-relpath` | longer curve plus CPU40 |
| 76 | normal | `gpu-h100-cpu16` | 199000 | 0 | `lz-visual-pong-exact-installed-0.2.0-s76-wave11-h100cpu16-long199k` | `train-normal-wave11-s76-199000-ckpt1000-h100cpu16-relpath` | one costlier long normal curve |
| 80 | shaped | `gpu-l4-t4-cpu16` | 65536 | 0.0005 | `lz-visual-pong-survival-shaped-step0p0005-s80-wave11-l4cpu16` | `train-survival-shaped-step0p0005-wave11-s80-65536-ckpt1000-l4cpu16-relpath` | lower survival shaping side lane |
| 81 | shaped | `gpu-l4-t4-cpu16` | 65536 | 0.001 | `lz-visual-pong-survival-shaped-step0p001-s81-wave11-l4cpu16` | `train-survival-shaped-step0p001-wave11-s81-65536-ckpt1000-l4cpu16-relpath` | existing shaping side-lane comparison |
| 82 | shaped | `gpu-l4-t4-cpu40` | 131072 | 0.001 | `lz-visual-pong-survival-shaped-step0p001-s82-wave11-l4cpu40-mid131k` | `train-survival-shaped-step0p001-wave11-s82-131072-ckpt1000-l4cpu40-relpath` | longer shaped telemetry without H100 spend |

## Launch Results

All ten local launch commands returned `status=spawned`.

| seed | app id | function call id |
| ---: | --- | --- |
| 70 | `ap-DfpF4WNTLQXykc0itSQvZz` | `fc-01KR90ARTPH2VMWEH84ZYYQQMH` |
| 71 | `ap-lmYSL2sd7F4Jz6KUidUxUx` | `fc-01KR90BCVJP7864MTG23R3AKPN` |
| 72 | `ap-vJEWiJaLeedzgdApThJdU3` | `fc-01KR90BXG5S10EMV3ZM6RZ6YE3` |
| 73 | `ap-ohBzaqN74acp14Z03S81sn` | `fc-01KR90CMBCZ5JE44W9MN1B4ETA` |
| 74 | `ap-7VEZuzB134B9qhV8DdXKRA` | `fc-01KR90D8Z1KNBQ75P59N5C58BC` |
| 75 | `ap-MT9PhdiHh0LBYZR3mCodOn` | `fc-01KR90DZQMM95E6BCMPD6FY2PS` |
| 76 | `ap-UgFrq1KD8zvdCdgpm5xd8z` | `fc-01KR90EMHWJXAN6E55AEVXJYN0` |
| 80 | `ap-6PLRYEIoAwhsZUynUxjDyD` | `fc-01KR90GPGW5ZEXR628PFCJQXK4` |
| 81 | `ap-BzunYbSmSXjGHeyX6Q8yTw` | `fc-01KR90H7S92SHNBF12GZVWBV7C` |
| 82 | `ap-N24TWpibhpAq6BNkLiKrfD` | `fc-01KR90HSPGRV7CBC14YEX2SP2Q` |

## Eval Watch

Do not block this launch wave on eval. Poll checkpoints first and eval only
when enough rows exist to compare survival against same-run baseline.
When launching a fresh eval id, add `--skip-eval-root-listing` to the queue
command to avoid one rate-limit-prone Modal Volume listing. Leave it off for
reruns or partial panels so the queue can skip checkpoint/seed dirs that already
exist.

First eval target for every run:

- `iteration_0`, `iteration_1000`, `iteration_5000`
- add `iteration_7000` when visible
- stock evaluator, strict no-fallback checkpoint load
- `max_eval_steps=2048`, `max_episode_steps=2048`
- `gpu-l4-t4-cpu40`, `--group-size 1`, `--max-parallel-launches 64`
- `--update-per-collect -1`
- stock-only triage by default; manual+stock is debug only

If a normal run improves stock survival at `5000`, continue with
`8000,10000,13000,16000/latest`. For `131072` and `199000` runs, add
`20000/latest` if the curve is still improving. Shaped runs should be evaluated
on the same cadence, but reported as side-lane telemetry only.

## Claim Discipline

Claim: this wave tests whether normal stock-reward Pong survival improvements
persist across fresh seeds, longer caps, and CPU allocation choices.

Non-claim: this wave does not prove solved Pong, exact upstream replication, or
CurvyTron readiness. Shaped reward results do not count as normal proof-lane
success.

CurvyTron note: scalar survival-wrapper work is a contract check only. The next
training blocker is visual `[4,64,64]` stacking plus a bounded
collect/search/replay/sample/learner profile, not another adapter smoke.

## Worker D Gap Eval - s75 Baseline

At `2026-05-10 16:10 EDT`, Worker D checked s75 because it previously had no
curve. Only `iteration_0` was visible under the attempt checkpoint dir, so a
baseline-only stock survival panel was launched. No later s75 checkpoint curve
is possible until `iteration_1000` or later appears.

```text
run_id: lz-visual-pong-exact-installed-0.2.0-s75-wave11-l4cpu40-mid131k
attempt_id: train-normal-wave11-s75-131072-ckpt1000-l4cpu40-relpath
eval_id: workerD-gap-s75-stock2048-rand8-20260510
selected visible: 0
eval_seed_sampler_seed: 5797923286624521321
eval_seeds: 2044681077,232573702,1204649970,880497471,93702182,2135255345,180585572,253666493
app: ap-ZrJeAw9w84yUJuVa9DWki7
manifest_ref: training/lightzero-official-visual-pong/lz-visual-pong-exact-installed-0.2.0-s75-wave11-l4cpu40-mid131k/attempts/train-normal-wave11-s75-131072-ckpt1000-l4cpu40-relpath/eval/workerD-gap-s75-stock2048-rand8-20260510/manifest_custom_steps2048_seeds2044681077-232573702-1204649970-880497471-93702182-2135255345-180585572-253666493_20260510T200952Z.json
```

Survival-first read:

```text
iteration_0 stock steps survived: mean 761.250, median 761, min 759, max 764
stock return: -21 across the panel
```

Plain read: s75 now has a baseline survival panel but still lacks any later
checkpoint curve. Poll for `iteration_1000` and `iteration_5000` before
launching the next eval.

## Pong Eval-Curve Lane - Survival First

Current focused experiment note:
`docs/experiments/2026-05-10-lightzero-wave11-pong-survival-curves.md`.

Plain status: use the existing `rand8-e` Modal artifacts as the current curve
read. They already cover the useful Wave11 runs with stock-only strict eval,
`2048` step cap, and `50` MCTS simulations per action. Survival steps are the
lead metric; score/return is secondary.

Current normal proof-lane reads:

| run | current survival read |
| --- | --- |
| s71 | late lift: `759.625` at `iteration_0`, `1351.12` at `18287` |
| s73 | strong late lift: `761` at `0`, cap at `18000`, `1823.38` at `18098` |
| s74 | strongest current normal curve: `761.625` at `0`, cap at `30000`, `37000`, and `37542` |
| s76 | long-run lift: `761.125` at `0`, `1905` at `40000`, `1786.25` at `53704` |

Prepared simple follow-up curves are documented for s73, s74, and s76 with
`8` selected checkpoints and queue-generated reproducible random seed panels.
They were dry-run only. Launching all three would be `192` serious stock eval
jobs, so do not spend that unless we need a fresh confirmation panel. The next
tooling fix is to amortize checkpoint/env/policy setup across seeds or
checkpoints in a long-lived eval worker and batch artifact writes.
