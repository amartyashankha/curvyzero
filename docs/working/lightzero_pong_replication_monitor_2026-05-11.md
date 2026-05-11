# LightZero Pong Replication Monitor - 2026-05-11

Checked at 2026-05-11 11:44 EDT.

Goal: keep Coach from under-parallelizing without duplicating other workers'
live jobs.

Updated at 2026-05-11 11:48 EDT for upstream-exact scout follow-up.

Updated at 2026-05-11 11:56 EDT for current GitHub README-recommended MuZero
segment path dry/launch facts.

Updated at 2026-05-11 12:14 EDT for official v0.2.0 tag and model-card
source identity.

Updated at 2026-05-11 12:16 EDT for Agent96 s127 0/1k/2k/3k strict eval,
official HF pretrained96 strict-load/eval blocker, and the separate s128
Agent96 500k checkpoint-cadence launch.

Updated at 2026-05-11 12:21 EDT for current Agent96 s127/s128 checkpoint
inventory, s127 strict eval through `iteration_5000`, and pretrained96 stop
condition.

Updated at 2026-05-11 12:25 EDT for the first credible survival signal,
current-upstream live-run status, and realistic long-run timing.

Updated at 2026-05-11 12:26 EDT for speed/scale facts from live checkpoint
timestamps.

Updated at 2026-05-11 12:34 EDT for later checkpoint inventory and active
survival eval bundles.

Updated at 2026-05-11 12:39 EDT for completed survival curves through
`s122` 12k and comparison rows.

Updated at 2026-05-11 12:44 EDT for current GitHub upstream segment eval
tooling and strict tiny survival smokes on `iteration_0/5000/6000`.

Updated at 2026-05-11 12:40 EDT: launched one more H100 stock64 repeat,
`s142`, to test whether the `s122` survival curve repeats.

Updated at 2026-05-11 14:08 EDT for the mature stock64 eval sweep through
`s122` 26k, `s142` 15k, `s121` 17k, `s114/s120` late checkpoints, and exact
stock `s113/s123` 20k.

Updated at 2026-05-11 14:15 EDT: briefly launched higher-cap stock-only
serious evals for `s122` (`0/26000/26672`), `s142` (`0/15000/20000`), and
exact-control `s123` (`0/20000/30000`) at `max_eval_steps=4096`. These were
stopped locally after reorientation: the replication gate already has enough
survival signal, and more Pong eval detail is now optional, not the main lane.

## Latest Coach Read - 2026-05-11 14:08 EDT

Plain read: stock-like visual Pong is learning. The earlier “only `s122`
works” story was too pessimistic. Several runs looked flat at `7k-11k`, then
showed clear survival gains later.

Survival steps are still the lead metric. Score is moving too, but later and
with more noise.

| run | checkpoints | seeds | mean survival steps | mean score | plain read |
| --- | --- | ---: | --- | --- | --- |
| `s122` H100 100k | `0/12000/20000/26000` | 16 | `761.5 -> 1378.06 -> 1591.69 -> 1977.62` | `-21 -> -16.9375 -> -17.25 -> -13.5` | Strongest run. By `26000`, many games hit the `2048` eval cap, so the next eval needs a higher cap. |
| `s142` H100 100k repeat | `0/7000/12000/15000` | 16 | `761.5 -> 761.5 -> 839.938 -> 938.375` | `-21 -> -21 -> -20.3125 -> -19.9375` | Repeats the delayed-survival pattern, but weaker than `s122` at the same horizon. |
| `s114` L4/T4 50k | `0/10000/13000` | 8 | `761.25 -> 792 -> 1612.12` | `-21 -> -20.875 -> -15.625` | Recovered strongly by `13000`. Earlier flat read was too early. |
| `s120` L4/T4 50k | `0/11000/14000` | 8 | `761.25 -> 776.5 -> 961.375` | `-21 -> -21 -> -20.625` | Recovered by `14000`. Not a flat failure. |
| `s121` L4/T4 65k | `0/11000/17000` | 8 | `853.25 -> 772.5 -> 1579.38` | `-20.875 -> -21 -> -16.125` | Dropped early, then recovered strongly by `17000`. |
| `s113` exact L4/T4 200k | `0/10000/20000` | 8 | `761.25 -> 833.75 -> 917.125` | `-21 -> -21 -> -20.5` | Exact stock control shows modest survival improvement by `20000`. |
| `s123` exact H100 200k | `0/10000/20000` | 8 | `761.25 -> 851.25 -> 1145.12` | `-21 -> -21 -> -19.625` | Exact stock H100 control improves more clearly by `20000`. |

Main correction: do not call a stock Pong run failed from `1k`, `7k`, or even
`10k` alone. The current evidence says survival signal often appears between
roughly `12k` and `20k`, and sometimes later. The next serious eval for strong
runs should raise `max_eval_steps` above `2048`, because `s122` is hitting the
cap.

## Latest Coach Read - 2026-05-11 12:25 EDT

Lead with survival steps. Score can stay near `-21` while the policy is
beginning to avoid dying longer.

- Best current positive result: installed LightZero 0.2.0 64x64 Pong run
  `s122` reached `iteration_7000` and strict eval over 16 starts gave mean
  survival `934.062` steps, median `848.5`, min `779`, max `1236`, and mean
  score `-20.125`. Same lane was around the mid-700s earlier, so this is a
  real survival movement, not solved Pong.
- Flat comparison: installed run `s114` at `iteration_7000` gave mean survival
  `761.312` and mean score `-21.0` over 16 starts. It has not shown useful
  learning yet.
- Current GitHub upstream segment path with official `ALE/Pong-v5` is alive
  and now evalable. The short `s1-wait` run has written through
  `iteration_7000` under `lightzero_segment_exp`; strict tiny eval smokes
  loaded `iteration_0/5000/6000` and reported survival/score. The correct task
  id is `lightzero-official-visual-pong-github-upstream`. The long
  `s2-long500k` run has written `iteration_0` and is training, with checkpoint
  cadence `5000`.
- New checkpoint state: installed `s114` has through `iteration_10000`;
  installed `s120` and `s121` have through `iteration_11000`; installed
  `s122` has through `iteration_12000`.
- Fresh survival evals now running:
  - `s122-0-7k-10k-12k-stockonly-rand16-20260511b`: completed.
  - `s114-0-7k-10k-stockonly-rand16-20260511b`: completed.
  - `s120-0-7k-11k-stockonly-rand8-20260511b`: completed.
  - `s121-0-7k-11k-stockonly-rand8-20260511b`: completed.
- Do not wait for `500000` before learning anything. At observed speeds,
  `500000` iterations is roughly a one-to-two-day job. The near-term gate is
  survival curves at `5k`, `10k`, `20k`, `50k`, and then `100k` if still
  running.

Completed survival curves:

| run | checkpoints | seeds | mean survival steps | mean score | plain read |
| --- | --- | ---: | --- | --- | --- |
| `s122` H100 100k | `0/7000/10000/12000` | 16 | `761.5 -> 934.5 -> 988.562 -> 1295.06` | `-21 -> -20.25 -> -19.75 -> -17.9375` | Strong survival signal. Not solved, but this is the first useful learning curve. |
| `s114` L4/T4 50k | `0/7000/10000` | 16 | `761.5 -> 761.5 -> 807.375` | `-21 -> -21 -> -20.75` | Small late bump only. Not strong enough to trust alone. |
| `s120` L4/T4 50k | `0/7000/11000` | 8 | `761.25 -> 761.25 -> 761.25` | `-21 -> -21 -> -21` | Flat. |
| `s121` L4/T4 65k | `0/7000/11000` | 8 | `856.625 -> 768.25 -> 765.5` | `-21 -> -21 -> -21` | Worse than its own start. |

Important interpretation: survival is moving before score. The best row
survives much longer by `12k`, while still losing most games. Score alone
would hide the useful early signal.

New replication launch:

- `lz-visual-pong-replication-matrix-20260511-s142-h100-repeat` /
  `train-stock-surface-100k-ckpt1000-h100cpu40-detached-wait`
- Same installed 0.2.0 stock64 surface as `s122`, with the same 100k cap and
  1000-iteration checkpoint cadence for observability.
- Claim goal: does a second H100 stock64 run also show survival movement by
  roughly `7k-12k`?

Approximate speeds from observed checkpoint-iteration timestamps, not env-step
throughput:

| lane | rough speed | rough time to 20k | rough time to 50k | rough time to 100k | rough time to 500k |
| --- | ---: | ---: | ---: | ---: | ---: |
| installed 0.2.0 stock64 L4/T4 CPU40 (`s114/s120/s121`) | about 182-188 iter/min from `iteration_0` through `8000/9000` | 1.8 h | 4.4-4.6 h | 8.9-9.2 h | 44-46 h, but installed wrapper stock cap is 200k |
| installed 0.2.0 stock64 H100 CPU40 (`s122`) | about 222 iter/min from `iteration_0` through `9000` | 1.5 h | 3.7 h | 7.5 h | 37.5 h, but installed wrapper stock cap is 200k |
| current-upstream segment `ALE/Pong-v5` L4/T4 CPU40 (`s1`) | provisional lower-bound only: `iteration_3000` was visible within about 19 min of the `starting` marker; plan as roughly stock64 L4 until more timestamps exist | about 2 h | about 5 h | about 10 h | about 50 h |
| Agent96 H100 CPU40 (`s127`) | about 136 iter/min from `iteration_0` through `5000`; `s125` model-card cadence still only has `iteration_0` | 2.4 h | 6.1 h | 12.2 h | 61 h |

Live speed notes:

- Stock64 L4 rows now show `s114` through `iteration_9000`, `s120` and
  `s121` through `iteration_8000`; stock64 H100 `s122` shows through
  `iteration_9000`.
- Current-upstream segment `s1` now shows `iteration_0/1000/2000/3000` under
  `lightzero_segment_exp`; `s2-long500k` shows `iteration_0` only, with
  checkpoint cadence `5000`.
- Agent96 `s127` now shows `iteration_0/1000/2000/3000/4000/5000`; Agent96
  `s125` still shows only `iteration_0` plus `ckpt_best`.
- The H100 stock64 row is only about 1.2x faster than the L4/T4 CPU40 rows in
  this sample. Use H100 if it is idle or queue time matters, not because the
  current control loop scales dramatically with the GPU.

Next speed-conscious eval plan:

- Do telemetry triage at `num_simulations=5`, strict load, stock-only rollout,
  and the same survival-step artifact shape. Treat this as a routing probe,
  not a replication claim.
- Do serious claim eval at stock `num_simulations=50` on the same fixed 16
  seeds for any checkpoint that moves survival, plus same-run `iteration_0`.
- Next installed stock64 evals: `s114`/`s122` at `9000`; `s120`/`s121` at
  `5000` and `8000`; then all active stock64 rows at `20000`.
- Next current-upstream segment eval: wait for at least `5000`; evaluate
  `0/1000/3000/5000` only after `5000` exists.
- Next Agent96 eval: `s127` at `5000`, then `10000` if it appears; do not
  mix Agent96 numbers with stock64 numbers.
- Poll/sleep cadence: poll every 12 minutes for live checkpoints. If waiting
  specifically for `20000`, sleep about 60 minutes from the current `8k/9k`
  stock64 state before the next full eval sweep.

Obvious speed rules that preserve replication:

- Keep training settings stock for claim runs. Do not lower training search
  simulations on the official-control lanes just to make a faster but different
  experiment.
- Keep the current `1000` checkpoint cadence on short survival-scout rows;
  use `5000` or coarser for long `100k/500k` runs.
- Use quick, lower-cost telemetry evals only to decide where to look next.
  Use strict stock eval settings for claims.
- Parallel eval remains important, but the main search inside each game is
  still expensive. Do not mistake slow eval for no learning.
- The installed 0.2.0 wrapper currently rejects `max_env_step_override` greater
  than its stock `200000`. A true `500k` installed run needs either a clearly
  labeled extended-run patch or the current-upstream segment lane.
- Do not change these on replication/control runs: collector count `8`,
  evaluator count `3`, training `num_simulations=50`, observation shape,
  env id, reward, `game_segment_length`, batch size, or `update_per_collect`.

## Current Read

The earlier `s111/s112` names are not present in the Modal Volume. Newer runs
are present and active. After checking that matrix, I launched four more
L4/T4 stock visual Pong controls so Coach has more seeds and checkpoint
cadence instead of waiting on a narrow pool.

Live update at 2026-05-11 11:47 EDT:

- Reliable checkpoint-writing runs: `s113`, `s114`, `s120`, `s121`, `s122`,
  `s123`; `s134` has started but has no checkpoint root yet.
- `s114` has `iteration_0`, `1000`, `2000`, `3000`, `4000`, and `5000`.
- `s120` has `iteration_0`, `1000`, `2000`, and `3000`.
- `s121` has `iteration_0`, `1000`, `2000`, and `3000`.
- `s122` has `iteration_0`, `1000`, `2000`, `3000`, and `4000`.
- Exact stock runs `s113` and `s123` still only show `iteration_0` plus
  `ckpt_best`; this is expected because exact cadence is slower.
- Spawned rows `s130`-`s133` still have no visible Volume checkpoint roots.
  Treat them as no-evidence until a Volume path appears.
- Strict eval manifests visible:
  - `s114-0-1k-2k-stockonly-baseeval-rand16`
  - `s114-0-3k-stockonly-baseeval-rand16`
  - `s120-0-1k-stockonly-baseeval-rand16`
  - `s121-0-1k-stockonly-baseeval-rand16b`
  - `s122-0-1k-2k-stockonly-baseeval-rand16`
- The first `s121-0-1k-stockonly-baseeval-rand16` eval attempt failed during
  Modal image build because a file changed while the image was building. That
  is not a model result; it was relaunched as the `...rand16b` eval above.

Active Modal apps:

- `curvyzero-lightzero-pong-exact-reproduction`: six live tasks observed
  before the extra L4/T4 launches.
- `curvyzero-lightzero-tictactoe-tiny-train-smoke`: completed.
- `curvyzero-lightzero-connect4-tiny-train-smoke`: completed.

New extra launches used wrapper-level `Function.spawn`. The local Modal app
wrappers stopped after returning call ids, which is expected for detached
launches. Treat them as spawned until Volume progress appears.

Simple lane labels:

- Installed 0.2.0 64x64 `train_muzero`: the existing `s113/s114/s120-s123`
  visual Pong matrix below, plus any later same-surface rows. This is not
  current GitHub upstream exact.
- Installed 0.2.0 96x96 `MuZeroAgent`: the model-card-style Agent lane added
  below. It is materially different from the 64x64 `train_muzero` matrix.
- Current GitHub upstream MuZero segment: pinned GitHub image builds, dry
  capture passes with official `--env ALE/Pong-v5`, and one faithful-short
  train was spawned. The older plain config remains blocked by
  `PongNoFrameskip-v4` action-map drift.

## Replication Board

Latest Volume read in this worker pass: 2026-05-11 11:47 EDT.

| label | active run ids | speed/progress | eval status | plain read |
| --- | --- | --- | --- | --- |
| installed 64x64 stock | `s113`, `s114`, `s120`, `s121`, `s122`, `s123`; `s134` just started | `s113`: 2 checkpoint files after 1824s; `s114`: 7 through `iteration_5000` after 1811s; `s120`: 5 through `iteration_3000` after 1448s; `s121`: 5 through `iteration_3000` after 1449s; `s122`: 6 through `iteration_4000` after 1478s; `s123`: 2 after 1206s; `s134`: 0 at start. Spawned `s130`-`s133` still have no visible roots. | Completed strict rand16 manifests are visible for `s114-0-1k-2k-stockonly-baseeval-rand16`, `s114-0-3k-stockonly-baseeval-rand16`, `s120-0-1k-stockonly-baseeval-rand16`, `s121-0-1k-stockonly-baseeval-rand16b`, and `s122-0-1k-2k-stockonly-baseeval-rand16`; earlier rand8 evals also completed. | Running and evaluable. Early signal is weak or flat. |
| Agent96 model-card | `s125`, `s127`, `s128` | `s125`: `iteration_0` plus `ckpt_best`; `s127`: `iteration_0/1000/2000/3000/4000/5000` plus `ckpt_best` at `2026-05-11T16:19:20Z`; `s128`: `iteration_0` only at `2026-05-11T16:18:00Z`. | Strict Agent96 eval completed for `s127` `0/1000/2000/3000/4000/5000`: 4 seeds, 512 cap, strict load, no fallback; mean survival 512 for every checkpoint; mean return -13 through 4k and -12.25 at 5k. Official HF pretrained96 artifact strict eval attempted and blocked before rollout by checkpoint/model key mismatch. | Survival is still flat at the 512-step cap; 5k has a small return nudge, not a solved-Pong signal. `s128` is launched but not later-checkpoint/eval-ready. |
| current upstream MuZero segment | `lz-visual-pong-github-upstream-segment-20260511-s0-short` spawned | Dry capture passed for pinned GitHub `de74055298068f53b70e07bc38c41101fce51766` with official `--env ALE/Pong-v5`; faithful-short train spawned as call `fc-01KRBW0GVFD5N1K85ASWF98YAD`; immediate Volume poll found no progress file yet. | No eval. | Smallest current-upstream official MuZero path found; train is spawned, not yet evidenced by Volume progress. |
| board controls | `s200`, `s201`, `s202`, `s203` completed | TicTacToe smoke ~15s, Connect4 smoke ~12s, Connect4 progression ~20s, TicTacToe progression ~13s. | No Pong eval relevance. | Proves LightZero board-game plumbing only. |
| non-LightZero controls | none active | Not launched here. | No eval. | Still open. |

Early survival result summary:

- Claim: installed 64x64 stock LightZero Pong runs are alive, checkpointing,
  and strict eval can load their checkpoints with no fallback.
- Non-claim: these rows do not show solved Pong, current upstream exact
  replication, Agent96 model-card replication, or CurvyTron readiness.
- The rand16 stock evals are weak/flat: `s114` stayed `760.875` mean stock
  steps at `0/1000/2000`, and a separate `s114` `0/3000` read stayed
  `760.125`; `s120` stayed `760.3125` at `0/1000`; `s121` dropped from
  `842.4375` at `0` to `760.125` at `1000`; `s122` drifted from `764.5625`
  at `0` to `762.625` at `1000` and `760.8125` at `2000`. Stock return
  stayed `-21` in these reads.

## What We Know / Do Not Know Yet

What we know:

- installed 64x64 stock: running, writing checkpoints, and evaluable.
- Agent96 model-card: `s127` writes later checkpoints and strict eval is flat
  through `3000`; `s125` still only has `iteration_0` plus `ckpt_best`; `s128`
  is launched as a separate 500k H100 cadence lane.
- board controls: completed and useful as plumbing checks only.

What we do not know yet:

- Whether any installed 64x64 stock checkpoint improves survival later.
- Whether Agent96 improves beyond the flat `s127` strict panel, and whether
  `s128` reaches useful later cadence checkpoints.
- Whether current upstream exact works; it remains open.
- Whether non-LightZero controls add useful comparison signal; they remain open.

Official-source boundary for examples:

- LightZero upstream examples/configs only:
  - docs quick start Pong command:
    `python3 -u zoo/atari/config/atari_muzero_config.py`
    (`https://opendilab.github.io/LightZero/tutorials/installation/installation_and_quickstart.html`)
  - current GitHub config:
    `zoo/atari/config/atari_muzero_config.py`
    (`https://raw.githubusercontent.com/opendilab/LightZero/main/zoo/atari/config/atari_muzero_config.py`)
  - related current GitHub configs to replicate after the plain Pong blocker:
    `zoo/atari/config/atari_muzero_segment_config.py`,
    `zoo/atari/config/atari_gumbel_muzero_config.py`,
    `zoo/atari/config/atari_efficientzero_config.py`,
    `zoo/atari/config/atari_sampled_efficientzero_config.py`.
  - official `v0.2.0` release tag:
    tag object `709225a135b4fb8b7d6720e2997ac8fec12795e1`, peeled commit
    `44a23b532f8516b7dd5c61105d6e5dd28c79dc0a`. In that tag,
    `zoo/atari/config/atari_env_action_space_map.py` includes
    `'PongNoFrameskip-v4': 6`, and the default plain/segment Atari MuZero
    configs use `PongNoFrameskip-v4`. This older official release is internally
    consistent, but it is the installed-package lane, not current GitHub exact.
- OpenDILab model card/API lane:
  `https://huggingface.co/OpenDILabCommunity/PongNoFrameskip-v4-MuZero`.
  This is the source for the installed-package `MuZeroAgent` 96x96 lane and
  its `agent.train(step=int(500000))` snippet; it is not the same surface as
  `zoo.atari.config.atari_muzero_config`. The card identifies Gym `0.25.1`,
  DI-engine `v0.5.0`, PyTorch `2.0.1+cu117`, task `PongNoFrameskip-v4`, a
  `[4,96,96]` MuZero policy, and self-reported mean reward `20.4 +/- 0.49`.
- MiniZero official controls only:
  `https://github.com/rlglab/minizero`.
  Official quick-run examples include Go AlphaZero/Gumbel AlphaZero and Atari
  Ms. Pac-Man MuZero/Gumbel MuZero, e.g.
  `tools/quick-run.sh train atari mz 300 -n ms_pacman_mz_n50 -conf_str env_atari_name=ms_pacman:actor_num_simulation=50`.
  A Pong variant should be treated as a hypothesis until MiniZero docs/source
  confirm the exact `env_atari_name` spelling.
- PettingZoo/OpenSpiel are controls only, not MuZero reproduction sources:
  PettingZoo Atari Pong documents `pong_v3`, two agents, Parallel API,
  observation `(210,160,3)`, action values `[0,5]`, and +/- point reward.
  OpenSpiel AlphaZero documents actor/learner/evaluator structure and the
  illustrative command
  `python3 open_spiel/python/examples/alpha_zero.py --game connect_four --nn_model mlp --actors 10`.

## Plain Replication Answer

Are we failing to replicate? Not yet. The current stock-ish Pong runs are
writing checkpoints and the strict stock evaluator can return policy rows. The
early `iteration_1000`/`2000` rows do not yet show a convincing learning curve:
`s114` is flat, `s120` has one small survival bump, `s121` goes down, and
`s122` is basically flat. That is weak early signal, not proof of failure.

Claim: the installed-package 64x64 LightZero Pong path is alive, checkpointing,
and evaluable.

Non-claim: we have not yet replicated solved Pong, current GitHub upstream
exact, or the model-card 96x96 result.

Important blocker note: the failed `s114-0-1k-stockonly-rand16` bundle died
before policy results with ALE/subprocess reset pipe errors, including
`_pickle.UnpicklingError: invalid load key, 'A'`. Treat that as eval tooling
failure, not a model result. Later strict eval rows that saved `ok=True`,
`strict=True`, no fallback, and stock steps are usable as eval rows.

Completed strict rand8 eval facts from this monitor pass:

| run | eval id | checkpoints | stock steps survived, mean over 8 seeds | stock return | result path |
| --- | --- | --- | --- | --- | --- |
| `s114` | `repro-s114-0-1000-stock2048-rand8-20260511a` | `0`, `1000`, then appended `2000` | `761.25`, `761.25`, `761.25` | `-21` throughout | `training/lightzero-official-visual-pong/lz-visual-pong-faithful-short-20260511-s114-detached/attempts/train-stock-surface-50k-ckpt1000-l4cpu40-detached-wait/eval/repro-s114-0-1000-stock2048-rand8-20260511a` |
| `s120` | `repro-s120-0-1000-stock2048-rand8-20260511a` | `0`, `1000` | `761.25`, `768.25` | `-21` throughout | `training/lightzero-official-visual-pong/lz-visual-pong-replication-matrix-20260511-s120/attempts/train-stock-surface-50k-ckpt1000-l4cpu40-detached-wait/eval/repro-s120-0-1000-stock2048-rand8-20260511a` |
| `s121` | `repro-s121-0-1000-stock2048-rand8-20260511a` | `0`, `1000` | `835.875`, `761.25` | `-21` throughout | `training/lightzero-official-visual-pong/lz-visual-pong-replication-matrix-20260511-s121/attempts/train-stock-surface-65k-ckpt1000-l4cpu40-detached-wait/eval/repro-s121-0-1000-stock2048-rand8-20260511a` |
| `s122` | `repro-s122-0-1000-stock2048-rand8-20260511a` | `0`, `1000` | `768.75`, `768.375` | `-21` throughout | `training/lightzero-official-visual-pong/lz-visual-pong-replication-matrix-20260511-s122-h100/attempts/train-stock-surface-100k-ckpt1000-h100cpu40-detached-wait/eval/repro-s122-0-1000-stock2048-rand8-20260511a` |

Exact eval command shape used for these rows:

```bash
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id <run_id> \
  --attempt-id <attempt_id> \
  --eval-id <eval_id> \
  --compute gpu-l4-t4-cpu40 \
  --eval-pass custom \
  --eval-seeds 1250426297,611354690,2130530255,551302474,1777247108,1588563970,1580338265,995705522 \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --step-detail-limit 8 \
  --max-env-step <50000_or_65000_or_100000> \
  --update-per-collect -1 \
  --selected-iterations <same-run-iterations> \
  --group-size 8 \
  --max-parallel-launches 64 \
  --execute
```

## Stock Or Not Stock

| lane | stock or not stock | current status | plain read |
| --- | --- | --- | --- |
| Installed LightZero 64x64 `train_muzero` Pong | Stock installed `LightZero==0.2.0` surface for `zoo.atari.config.atari_muzero_config`; exact rows keep stock step budget, short rows only change step cap/checkpoint cadence for observability | Active: `s113`, `s114`, `s120`, `s121`, `s122`, `s123`; strict eval rows exist for `s114`, `s120`, `s121`, `s122` | Best current proof lane. Not solved yet. Keep comparing later checkpoints to same-run `iteration_0`. |
| Installed LightZero 96x96 `MuZeroAgent` / model-card-style | Not the 64x64 `train_muzero` stock lane. It is a separate installed-package Agent/model-card surface with `[4,96,96]` observations | Dry pass exists; H100 detached Agent rows have some progress/checkpoint signs in this doc | Useful compatibility lane for the older OpenDILab model-card shape. Do not mix its results with 64x64 `train_muzero`. |
| Current GitHub upstream exact | Stock target from pinned GitHub `opendilab/LightZero@de74055298068f53b70e07bc38c41101fce51766` | Dry-exact image build passed; config import blocked by upstream `PongNoFrameskip-v4` action-map `KeyError` | Do not train as exact until upstream env id/action-map mismatch is fixed or a clearly non-exact `ALE/Pong-v5` patch lane is approved. |
| Board-game LightZero controls | Stock LightZero examples, but not Atari Pong | TicTacToe and Connect4 completed tiny train/checkpoint smokes | Good framework sanity checks. They do not prove visual Pong replication. |
| Non-LightZero controls | Not stock LightZero | Mctx dependency/search smoke passed; OpenSpiel AlphaZero TicTacToe one-step Modal trainer control completed; MiniZero and `muzero-general` remain candidates but need wrappers/deps | Good outside comparison lane. OpenSpiel proves outside-LightZero actor/replay/learner/checkpoint plumbing, but not MuZero dynamics or visual Pong. |

Sources for lane identity: current LightZero README still lists Pong quick-start
commands under `zoo/atari/config/atari_muzero_config.py` and
`atari_muzero_segment_config.py`; current upstream `atari_muzero_config.py`
uses `PongNoFrameskip-v4`, observation shape `(4, 64, 64)`, `train_muzero`,
8 collectors, 3 evaluators, 50 simulations, batch size 256, and
`max_env_step=500000`. The OpenDILab Hugging Face Pong MuZero model card is a
separate 96x96/model-card-style lane with self-reported Pong mean reward
`20.4 +/- 0.49`.

## Next Official/Control Examples

1. Current GitHub LightZero MuZero segment progress poll.
   Why: this directly answers whether installed `0.2.0` is diverging from
   current upstream. The README-recommended MuZero segment path dry-captures
   cleanly with official `--env ALE/Pong-v5`, and a faithful-short L4/T4 run
   has been spawned. Next action is polling the recorded progress ref; do not
   evaluate or scale until Volume progress/checkpoints exist.

2. OpenDILab 96x96 pretrained/model-card Pong strict eval.
   Status: attempted on the official downloaded `policy_config.py` and
   `pytorch_model.bin`; blocked before rollout by strict-load mismatch
   `unexpected key representation_network.downsample_net.conv2.weight` under
   installed `LightZero==0.2.0`. Do not score this artifact until the exact
   compatible model code is identified or a separate non-strict compatibility
   lane is explicitly approved.

3. MiniZero official quick-run control.
   Why: this is an official full-system Zero framework control with server,
   self-play workers, optimization worker, storage, batched inference, and
   Atari support. Start from the documented Ms. Pac-Man MuZero command; treat
   any Pong variant as unverified until the official source confirms the exact
   `env_atari_name`.

4. PettingZoo/OpenSpiel controls only.
   Why: PettingZoo Atari Pong is the official parallel two-player visual Pong
   control; OpenSpiel AlphaZero Connect Four is the official actor/learner
   self-play control. Neither is a LightZero/MuZero reproduction target.

Completed OpenSpiel control added at 2026-05-11 11:55 EDT:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.openspiel_alphazero_quick_control \
  --run-id openspiel-alphazero-tictactoe-20260511-s300c \
  --attempt-id train-tictactoe-1step-mlp-cpu
```

Result: `ok=true`, app `ap-VKq1u4vuZFa1kU6F0M1JiB`,
`open_spiel==1.6.13`, `jax==0.10.0`, `flax==0.12.7`. It ran
OpenSpiel's official Python AlphaZero implementation on `tic_tac_toe` with
`actors=1`, `evaluators=0`, `max_steps=1`, `max_simulations=2`,
`train_batch_size=2`, and an MLP `16x1` model. Learner evidence:
`step=1`, `total_states=15`, `total_trajectories=2`,
`game_length.avg=7.5`, `loss.sum=1.8990381956100464`. Artifact refs:

```text
training/openspiel-alphazero-tictactoe/openspiel-alphazero-tictactoe-20260511-s300c/attempts/train-tictactoe-1step-mlp-cpu/train/summary.json
training/openspiel-alphazero-tictactoe/openspiel-alphazero-tictactoe-20260511-s300c/attempts/train-tictactoe-1step-mlp-cpu/train/openspiel_alpha_zero
```

Claim: outside-LightZero AlphaZero-style actor/replay/learner/checkpoint
plumbing runs in the Modal context. Non-claim: no MuZero dynamics model, no
visual Atari/Pong, no learning-strength claim from the one-step cap.

## Visual Atari Pong Control Matrix

All Pong rows use the installed LightZero Atari MuZero surface:
`zoo.atari.config.atari_muzero_config`, `PongNoFrameskip-v4`, stock visual
Atari env, stock sparse reward, and `train_muzero`.

Expected checkpoint dir for each Pong row:
`training/lightzero-official-visual-pong/<run_id>/attempts/<attempt_id>/train/lightzero_exp/ckpt`

| status | run_id | attempt_id | command | latest progress | what it proves |
| --- | --- | --- | --- | --- | --- |
| running | `lz-visual-pong-exact-repro-20260511-s113-detached` | `train-stock-exact-200k-l4cpu40-detached-wait` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu40 --seed 113 --run-id lz-visual-pong-exact-repro-20260511-s113-detached --attempt-id train-stock-exact-200k-l4cpu40-detached-wait --progress-interval-sec 300 --wait-for-train` | `running`, 304s elapsed at `2026-05-11T15:22:14Z`, `iteration_0` and `ckpt_best` visible | exact stock 200k L4/T4+CPU40 control; no checkpoint cadence override |
| running | `lz-visual-pong-faithful-short-20260511-s114-detached` | `train-stock-surface-50k-ckpt1000-l4cpu40-detached-wait` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu40 --seed 114 --run-id lz-visual-pong-faithful-short-20260511-s114-detached --attempt-id train-stock-surface-50k-ckpt1000-l4cpu40-detached-wait --progress-interval-sec 180 --max-env-step-override 50000 --save-ckpt-after-iter-override 1000 --wait-for-train` | `running`, 362s elapsed at `2026-05-11T15:23:13Z`, `iteration_0` and `ckpt_best` visible | near-stock short curve with useful checkpoint cadence on L4/T4+CPU40 |
| starting | `lz-visual-pong-replication-matrix-20260511-s120` | `train-stock-surface-50k-ckpt1000-l4cpu40-detached-wait` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu40 --seed 120 --run-id lz-visual-pong-replication-matrix-20260511-s120 --attempt-id train-stock-surface-50k-ckpt1000-l4cpu40-detached-wait --progress-interval-sec 180 --max-env-step-override 50000 --save-ckpt-after-iter-override 1000 --wait-for-train` | `starting`, no checkpoint yet at `2026-05-11T15:23:20Z` | second L4/T4 short seed for curve variance |
| starting | `lz-visual-pong-replication-matrix-20260511-s121` | `train-stock-surface-65k-ckpt1000-l4cpu40-detached-wait` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu40 --seed 121 --run-id lz-visual-pong-replication-matrix-20260511-s121 --attempt-id train-stock-surface-65k-ckpt1000-l4cpu40-detached-wait --progress-interval-sec 180 --max-env-step-override 65000 --save-ckpt-after-iter-override 1000 --wait-for-train` | `starting`, no checkpoint yet at `2026-05-11T15:23:20Z` | L4/T4 slightly longer near-stock curve |
| starting | `lz-visual-pong-replication-matrix-20260511-s122-h100` | `train-stock-surface-100k-ckpt1000-h100cpu40-detached-wait` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-h100-cpu40 --seed 122 --run-id lz-visual-pong-replication-matrix-20260511-s122-h100 --attempt-id train-stock-surface-100k-ckpt1000-h100cpu40-detached-wait --progress-interval-sec 180 --max-env-step-override 100000 --save-ckpt-after-iter-override 1000 --wait-for-train` | `starting`, no checkpoint yet at `2026-05-11T15:23:18Z` | longer curve if cheap lanes bottleneck |
| starting | `lz-visual-pong-replication-matrix-20260511-s123-h100-exact` | `train-stock-exact-200k-h100cpu40-detached-wait` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-h100-cpu40 --seed 123 --run-id lz-visual-pong-replication-matrix-20260511-s123-h100-exact --attempt-id train-stock-exact-200k-h100cpu40-detached-wait --progress-interval-sec 300 --wait-for-train` | `starting`, no checkpoint yet at `2026-05-11T15:23:15Z` | second exact 200k control on faster compute |
| spawned | `lz-visual-pong-replication-matrix-20260511-s130-l4-exact` | `train-stock-exact-200k-l4cpu40-detached` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu40 --seed 130 --run-id lz-visual-pong-replication-matrix-20260511-s130-l4-exact --attempt-id train-stock-exact-200k-l4cpu40-detached --progress-interval-sec 300` | spawned call `fc-01KRBTD4Z0PPTEZJCMVVPXACYK`; progress not visible on immediate poll | third exact 200k stock control, on L4/T4 |
| spawned | `lz-visual-pong-replication-matrix-20260511-s131-l4-50k` | `train-stock-surface-50k-ckpt1000-l4cpu40-detached` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu40 --seed 131 --run-id lz-visual-pong-replication-matrix-20260511-s131-l4-50k --attempt-id train-stock-surface-50k-ckpt1000-l4cpu40-detached --progress-interval-sec 180 --max-env-step-override 50000 --save-ckpt-after-iter-override 1000` | spawned call `fc-01KRBTD5376AA9TA06131AK5GC`; progress not visible on immediate poll | third short L4/T4 curve with `iteration_1000` cadence |
| spawned | `lz-visual-pong-replication-matrix-20260511-s132-l4-65k` | `train-stock-surface-65k-ckpt1000-l4cpu40-detached` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu40 --seed 132 --run-id lz-visual-pong-replication-matrix-20260511-s132-l4-65k --attempt-id train-stock-surface-65k-ckpt1000-l4cpu40-detached --progress-interval-sec 180 --max-env-step-override 65000 --save-ckpt-after-iter-override 1000` | spawned call `fc-01KRBTD503PMWP4P6QX24B2Z08`; progress not visible on immediate poll | second 65k L4/T4 curve with `iteration_1000` cadence |
| spawned | `lz-visual-pong-replication-matrix-20260511-s133-l4-100k` | `train-stock-surface-100k-ckpt1000-l4cpu40-detached` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode train --compute gpu-l4-t4-cpu40 --seed 133 --run-id lz-visual-pong-replication-matrix-20260511-s133-l4-100k --attempt-id train-stock-surface-100k-ckpt1000-l4cpu40-detached --progress-interval-sec 180 --max-env-step-override 100000 --save-ckpt-after-iter-override 1000` | spawned call `fc-01KRBTD57TADAPQ4APECNMSHWT`; progress not visible on immediate poll | 100k L4/T4 curve with `iteration_1000` cadence |

Exact checkpoint paths for the new L4/T4 rows:

- `s130`: `training/lightzero-official-visual-pong/lz-visual-pong-replication-matrix-20260511-s130-l4-exact/attempts/train-stock-exact-200k-l4cpu40-detached/train/lightzero_exp/ckpt`
- `s131`: `training/lightzero-official-visual-pong/lz-visual-pong-replication-matrix-20260511-s131-l4-50k/attempts/train-stock-surface-50k-ckpt1000-l4cpu40-detached/train/lightzero_exp/ckpt`
- `s132`: `training/lightzero-official-visual-pong/lz-visual-pong-replication-matrix-20260511-s132-l4-65k/attempts/train-stock-surface-65k-ckpt1000-l4cpu40-detached/train/lightzero_exp/ckpt`
- `s133`: `training/lightzero-official-visual-pong/lz-visual-pong-replication-matrix-20260511-s133-l4-100k/attempts/train-stock-surface-100k-ckpt1000-l4cpu40-detached/train/lightzero_exp/ckpt`

Exact progress refs for the new L4/T4 rows:

- `s130`: `training/lightzero-official-visual-pong/lz-visual-pong-replication-matrix-20260511-s130-l4-exact/attempts/train-stock-exact-200k-l4cpu40-detached/train/progress/latest.json`
- `s131`: `training/lightzero-official-visual-pong/lz-visual-pong-replication-matrix-20260511-s131-l4-50k/attempts/train-stock-surface-50k-ckpt1000-l4cpu40-detached/train/progress/latest.json`
- `s132`: `training/lightzero-official-visual-pong/lz-visual-pong-replication-matrix-20260511-s132-l4-65k/attempts/train-stock-surface-65k-ckpt1000-l4cpu40-detached/train/progress/latest.json`
- `s133`: `training/lightzero-official-visual-pong/lz-visual-pong-replication-matrix-20260511-s133-l4-100k/attempts/train-stock-surface-100k-ckpt1000-l4cpu40-detached/train/progress/latest.json`

## Compact Status Poller

Use the local status helper instead of manually polling every checkpoint,
progress, and eval directory:

```bash
uv run python scripts/lightzero_replication_status.py
```

It knows the active `s113/s114/s120-s123`, `s130-s133`, and Agent96
`s125/s127` rows from this monitor note. It prints run id, attempt id, source
family, train status if a progress JSON is readable, checkpoint count/latest
checkpoint, the latest known strict eval survival mean by checkpoint where this
monitor has one, and compact notes for visible eval roots or local summaries.

Default mode is intentionally local-only so it stays fast. It reads fetched
progress JSON under `artifacts/local`, mines local LightZero eval manifests
under `artifacts/local/lightzero-eval-manifests/` and
`artifacts/local/eval-manifest-files/`, and then falls back to strict eval
facts parsed from this monitor note.

For a live Modal Volume refresh:

```bash
uv run python scripts/lightzero_replication_status.py --live-modal
```

Live mode now parallelizes reads by run while keeping the default command
local-only. Use bounded workers and per-Modal-command timeout if the Volume is
slow:

```bash
uv run python scripts/lightzero_replication_status.py \
  --live-modal \
  --workers 8 \
  --modal-timeout-sec 8
```

The script is still a small status lane: live mode reads progress, checkpoint
visibility, and eval-root visibility from Modal; survival means come from local
manifests or this note unless fresh manifests have been fetched locally. If
Modal reads are slow, use the default command and treat `local-only` notes as
stale-until-refreshed.

## MuZeroAgent 96x96 Model-Card Lane

Label: Installed 0.2.0 96x96 `MuZeroAgent`.

Claim: this lane exercises the installed LightZero `lzero.agent.MuZeroAgent`
API and its bundled `PongNoFrameskip-v4` visual Atari config.

Non-claim: this is not current GitHub upstream exact, not CurvyTron evidence,
and not the same surface as the 64x64 `zoo.atari.config.atari_muzero_config`
`train_muzero` matrix.

Dry validation passed:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_muzero_agent_reproduction \
  --mode dry --compute cpu --seed 124 \
  --run-id lz-visual-pong-muzero-agent96-20260511-s124-dry \
  --attempt-id dry-agent96-model-card-surface
```

Dry result: `ok=true`, installed `LightZero==0.2.0`,
`PongNoFrameskip-v4`, `lzero.agent.MuZeroAgent`, visual shape `[4,96,96]`,
`downsample=True`, 8 collectors, 3 evaluators, 50 simulations, batch size
256, `update_per_collect=1000`, `game_segment_length=400`, and
`agent.train(step=500000)` as the model-card step shape. Summary ref:
`training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s124-dry/attempts/dry-agent96-model-card-surface/dry_summary.json`.

Active/suspicious Agent rows:

| status | run_id | attempt_id | command | latest progress | what it proves |
| --- | --- | --- | --- | --- | --- |
| suspicious-no-progress | `lz-visual-pong-muzero-agent96-20260511-s124-short` | `train-agent96-50k-l4cpu40-ckpt1000` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_muzero_agent_reproduction --mode train --compute gpu-l4-t4-cpu40 --seed 124 --train-step 50000 --run-id lz-visual-pong-muzero-agent96-20260511-s124-short --attempt-id train-agent96-50k-l4cpu40-ckpt1000 --progress-interval-sec 180 --save-ckpt-after-iter-override 1000` | returned call `fc-01KRBTRNW73WXBKTTDASJ9BPV5`; no Volume progress on poll | plain non-detached spawn is not reliable; do not count as active |
| suspicious-no-progress | `lz-visual-pong-muzero-agent96-20260511-s125-modelcard` | `train-agent96-modelcard-500k-h100cpu40` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_muzero_agent_reproduction --mode train --compute gpu-h100-cpu40 --seed 125 --train-step 500000 --run-id lz-visual-pong-muzero-agent96-20260511-s125-modelcard --attempt-id train-agent96-modelcard-500k-h100cpu40 --progress-interval-sec 300` | returned call `fc-01KRBTRNZ5MT3YRT8QRAYBQS8M`; no Volume progress on poll | plain non-detached spawn is not reliable; do not count as active |
| no-progress | `lz-visual-pong-muzero-agent96-20260511-s124-short-detached` | `train-agent96-50k-l4cpu40-ckpt1000-detached` | `uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_muzero_agent_reproduction --mode train --compute gpu-l4-t4-cpu40 --seed 124 --train-step 50000 --run-id lz-visual-pong-muzero-agent96-20260511-s124-short-detached --attempt-id train-agent96-50k-l4cpu40-ckpt1000-detached --progress-interval-sec 180 --save-ckpt-after-iter-override 1000` | app `ap-EaGoAZQA6U2AXzytZeNanF`, call `fc-01KRBTTM5WP7VV62RD6TX3S3H4`; no Volume progress on poll | L4/T4 Agent launch did not materialize; superseded by H100 short |
| no-progress | `lz-visual-pong-muzero-agent96-20260511-s126-short-detached` | `train-agent96-50k-l4cpu40-ckpt1000-detached` | `uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_muzero_agent_reproduction --mode train --compute gpu-l4-t4-cpu40 --seed 126 --train-step 50000 --run-id lz-visual-pong-muzero-agent96-20260511-s126-short-detached --attempt-id train-agent96-50k-l4cpu40-ckpt1000-detached --progress-interval-sec 180 --save-ckpt-after-iter-override 1000` | app `ap-fVm0wtG63qMvjXokt7Fj4E`, call `fc-01KRBTWR6ZSYGK9REEM8ZY2Y24`; no Volume progress on poll | solo L4/T4 Agent launch did not materialize; superseded by H100 short |
| running | `lz-visual-pong-muzero-agent96-20260511-s125-modelcard-detached` | `train-agent96-modelcard-500k-h100cpu40-detached` | `uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_muzero_agent_reproduction --mode train --compute gpu-h100-cpu40 --seed 125 --train-step 500000 --run-id lz-visual-pong-muzero-agent96-20260511-s125-modelcard-detached --attempt-id train-agent96-modelcard-500k-h100cpu40-detached --progress-interval-sec 300` | progress `starting` at `2026-05-11T15:35:02Z`; app `ap-UcZ6lCQbizAjjyulRFCUvS`; call `fc-01KRBTTAYD4SBJX4SM5K9KB6GH`; `agent_exp/ckpt/iteration_0.pth.tar` visible | model-card-step Agent-path 500k H100 control |
| running | `lz-visual-pong-muzero-agent96-20260511-s127-short-h100-detached` | `train-agent96-50k-h100cpu40-ckpt1000-detached` | `uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_muzero_agent_reproduction --mode train --compute gpu-h100-cpu40 --seed 127 --train-step 50000 --run-id lz-visual-pong-muzero-agent96-20260511-s127-short-h100-detached --attempt-id train-agent96-50k-h100cpu40-ckpt1000-detached --progress-interval-sec 180 --save-ckpt-after-iter-override 1000` | progress `starting` at `2026-05-11T15:36:55Z`; app `ap-RTYS4gh8K1FVpHe7FVAyVS`; call `fc-01KRBTXSAQR2WW6VTCA82RJMTR`; Agent exp config/log files visible, checkpoint not visible yet | short Agent-path H100 curve with checkpoint cadence override |
| spawned | `lz-visual-pong-muzero-agent96-20260511-s128-modelcard-ckpt5000-detached` | `train-agent96-modelcard-500k-h100cpu40-ckpt5000-detached` | `uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_muzero_agent_reproduction --mode train --compute gpu-h100-cpu40 --seed 128 --train-step 500000 --run-id lz-visual-pong-muzero-agent96-20260511-s128-modelcard-ckpt5000-detached --attempt-id train-agent96-modelcard-500k-h100cpu40-ckpt5000-detached --progress-interval-sec 300 --save-ckpt-after-iter-override 5000` | app `ap-vRptjltbIRoLipQe6UKggm`; call `fc-01KRBWZVV7N7HMSF4VK9PGH5BP`; progress `starting` at `2026-05-11T16:12:59Z`; immediate checkpoint-dir poll found no directory | separate 500k Agent96 H100 lane with useful learner-iteration checkpoint cadence; not a duplicate of `s125` because `s125` has no cadence override |

Agent checkpoint roots:

- `s125 modelcard`: `training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s125-modelcard-detached/attempts/train-agent96-modelcard-500k-h100cpu40-detached/train/agent_exp/ckpt`
- `s127 short`: `training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s127-short-h100-detached/attempts/train-agent96-50k-h100cpu40-ckpt1000-detached/train/agent_exp/ckpt`
- `s128 modelcard ckpt5000`: `training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s128-modelcard-ckpt5000-detached/attempts/train-agent96-modelcard-500k-h100cpu40-ckpt5000-detached/train/agent_exp/ckpt`

Agent progress refs:

- `s125 modelcard`: `training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s125-modelcard-detached/attempts/train-agent96-modelcard-500k-h100cpu40-detached/train/progress/latest.json`
- `s127 short`: `training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s127-short-h100-detached/attempts/train-agent96-50k-h100cpu40-ckpt1000-detached/train/progress/latest.json`
- `s128 modelcard ckpt5000`: `training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s128-modelcard-ckpt5000-detached/attempts/train-agent96-modelcard-500k-h100cpu40-ckpt5000-detached/train/progress/latest.json`

Source read for the model-card `agent.train(step=500000)` knob: LightZero
`MuZeroAgent.train` stops when `collector.envstep >= step`; checkpoint cadence
is controlled separately by the learner hook `save_ckpt_after_iter`. Treat
`step=500000` as collected environment steps, not learner iterations.

### Agent96 Strict Eval Update - 2026-05-11 11:50 EDT

Changed path:
`src/curvyzero/infra/modal/lightzero_pong_muzero_agent96_eval.py`.

This lane needs a separate config loader from the 64x64 eval smoke. The useful
reuse is only the stock-evaluator recording pattern. The Agent96 eval must load
`lzero.agent.config.muzero.supported_env_cfg["PongNoFrameskip-v4"]`, not
`zoo.atari.config.atari_muzero_config`; the captured eval surface is
`PongNoFrameskip-v4`, `atari_lightzero`, observation shape `[4,96,96]`,
`downsample=True`, `MuZeroPolicy`, action space 6.

Strict eval properties:

- no model fallback;
- no manual fallback rollout;
- strict policy model load only;
- stock Pong env through `lzero.worker.MuZeroEvaluator`;
- evaluator env manager patched to one in-process/base env only to avoid ALE
  subprocess pipe corruption; train surface remains stock subprocess;
- same-run comparison requires `iteration_0` and later checkpoints;
- selection/readout priority is stock survival steps first, stock return second.

Fresh checkpoint inventory from Modal Volume:

| run | checkpoint inventory | progress snapshot | eval status |
| --- | --- | --- | --- |
| `s125 modelcard` | `ckpt_best.pth.tar`, `iteration_0.pth.tar`; no later iteration checkpoint | `running` at `2026-05-11T15:50:05Z`, elapsed 903.6s, progress interval 300s | not eval-ready for same-run curve; next useful poll after `2026-05-11T15:55:05Z` |
| `s127 short` | `iteration_0.pth.tar`, `iteration_1000.pth.tar`; no higher later checkpoint visible | `running` at `2026-05-11T15:49:02Z`, elapsed 726.6s, progress interval 180s | strict Agent96 eval completed for `0` vs `1000`; next useful checkpoint poll after `2026-05-11T15:52:02Z` |

Strict eval command run:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_muzero_agent96_eval \
  --compute gpu-l4-t4-cpu8 \
  --run-id lz-visual-pong-muzero-agent96-20260511-s127-short-h100-detached \
  --attempt-id train-agent96-50k-h100cpu40-ckpt1000-detached \
  --selected-iterations 0,1000 \
  --eval-seed-count 4 \
  --eval-seed-rng-seed 20260511 \
  --max-eval-steps 512 \
  --summary-only
```

Strict eval summary:

| checkpoint | seeds | ok | mean stock survival steps | mean stock return | delta survival vs `iteration_0` | delta return vs `iteration_0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `iteration_0` | 4 | 4 | 512 | -13 | 0 | 0 |
| `iteration_1000` | 4 | 4 | 512 | -13 | 0 | 0 |

Per-seed eval facts: seeds `1250426297`, `611354690`, `2130530255`,
`551302474`; every row strict-loaded, used no fallback, reached the 512-step
eval cap, and returned `-13`.

Manifest:
`training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s127-short-h100-detached/attempts/train-agent96-50k-h100cpu40-ckpt1000-detached/eval/agent96_strict_stock_curve/manifest_steps512_seedsn4_a2587e3fa194_20260511T155055Z.json`.

Eval artifact root:
`training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s127-short-h100-detached/attempts/train-agent96-50k-h100cpu40-ckpt1000-detached/eval/agent96_strict_stock_curve`.

### Agent96 / Pretrained96 Update - 2026-05-11 12:21 EDT

Current checkpoint inventory:

| run | checkpoint inventory | progress snapshot | current eval/read |
| --- | --- | --- | --- |
| `s127 short` | `iteration_0/1000/2000/3000/4000/5000` plus `ckpt_best` | `running` at `2026-05-11T16:19:20Z`, elapsed 2544.5s, interval 180s | strict eval through `5000` completed; survival flat at 512/512 for all 4 seeds at every checkpoint |
| `s128 modelcard ckpt5000` | `iteration_0` only | `running` at `2026-05-11T16:18:00Z`, elapsed 300.8s, interval 300s | not later-checkpoint/eval-ready |

Strict survival-first s127 command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_muzero_agent96_eval \
  --compute gpu-l4-t4-cpu8 \
  --run-id lz-visual-pong-muzero-agent96-20260511-s127-short-h100-detached \
  --attempt-id train-agent96-50k-h100cpu40-ckpt1000-detached \
  --selected-iterations 0,1000,2000,3000,4000,5000 \
  --eval-seed-count 4 \
  --eval-seed-rng-seed 20260511 \
  --max-eval-steps 512 \
  --summary-only
```

| checkpoint | seeds | ok | mean stock survival steps | mean stock return | survival delta vs `iteration_0` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `iteration_0` | 4 | 4 | 512 | -13 | 0 |
| `iteration_1000` | 4 | 4 | 512 | -13 | 0 |
| `iteration_2000` | 4 | 4 | 512 | -13 | 0 |
| `iteration_3000` | 4 | 4 | 512 | -13 | 0 |
| `iteration_4000` | 4 | 4 | 512 | -13 | 0 |
| `iteration_5000` | 4 | 4 | 512 | -12.25 | 0 |

Every row strict-loaded, used no fallback, and reached the 512-step eval cap.
Manifest:
`training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s127-short-h100-detached/attempts/train-agent96-50k-h100cpu40-ckpt1000-detached/eval/agent96_strict_stock_curve/manifest_steps512_seedsn4_a2587e3fa194_20260511T161910Z.json`.

Iteration speed estimate: `s127` wrote `iteration_1000` through
`iteration_5000` over about 1788s, roughly 2.2 learner iterations/sec. Its
50k env-step run was still running after 2544.5s. Because `agent.train(step=...)`
stops on `collector.envstep >= step`, `500000` means collected env steps, not
learner iterations. At the current H100 evidence level, 500k looks like an
hours-scale run, not a multi-day run, though the wrapper timeout is 18h.

Official pretrained96 attempt:

| item | durable fact |
| --- | --- |
| source | `https://huggingface.co/OpenDILabCommunity/PongNoFrameskip-v4-MuZero`, self-reported mean reward `20.4 +/- 0.49` |
| local artifact refs | `training/lightzero-official-visual-pong/pretrained/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/pytorch_model.bin`; `training/lightzero-official-visual-pong/pretrained/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/policy_config.py` |
| strict-load probe | `training/lightzero-official-visual-pong-pretrained96/pretrained/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/probe/lightzero_visual_pong_pretrained96_strict_load_20260511T160646Z.json` |
| strict eval attempt | `training/lightzero-official-visual-pong-pretrained96/pretrained/OpenDILabCommunity/PongNoFrameskip-v4-MuZero/eval/lightzero_visual_pong_pretrained96_eval_20260511T160712Z.json` |
| blocker | installed `LightZero==0.2.0` strict policy/model load rejects the artifact with unexpected key `representation_network.downsample_net.conv2.weight`; no env rollout, no fallback, no scored survival/return |

Next command after resolving the compatibility mismatch:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_pretrained96_eval_smoke \
  --num-simulations 50 --batch-size 256 --update-per-collect 1000 \
  --game-segment-length 400 --max-env-step 500000 \
  --max-episode-steps 4096 --max-eval-steps 4096 \
  --step-detail-limit 0 --run-stock-evaluator
```

## Current GitHub Upstream MuZero

Pinned source identity for this lane:

```text
repo: https://github.com/opendilab/LightZero.git
commit: de74055298068f53b70e07bc38c41101fce51766
install: git+https://github.com/opendilab/LightZero.git@de74055298068f53b70e07bc38c41101fce51766
Modal package versions from dry: LightZero 0.2.0 from git, DI-engine 0.5.3, torch 2.11.0,
gym 0.25.1, gymnasium 1.3.0, ale-py 0.11.2, AutoROM 0.6.1
```

Implementation:
`src/curvyzero/infra/modal/lightzero_pong_github_upstream_dry_check.py`.

Eval implementation:
`src/curvyzero/infra/modal/lightzero_pong_github_upstream_eval.py`.

Current upstream checkpoint inventory checked at 2026-05-11 12:43 EDT:

```text
uv run --extra modal modal volume ls curvyzero-runs \
  training/lightzero-official-visual-pong-github-upstream/lz-visual-pong-github-upstream-segment-20260511-s1-wait/attempts/train-muzero-segment-ale-pong-v5-50k-ckpt1000-l4cpu40-wait/train/lightzero_segment_exp/ckpt
```

Visible checkpoints: `iteration_0/1000/2000/3000/4000/5000/6000/7000` plus
`ckpt_best`.

Strict eval wrapper compile check:

```text
python -m py_compile src/curvyzero/infra/modal/lightzero_pong_github_upstream_eval.py
```

Tiny strict evalability smoke, `iteration_0`:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_github_upstream_eval \
  --compute cpu --selected-iterations 0 --eval-seeds 0 \
  --max-eval-steps 8 --num-simulations 1 \
  --eval-id upstream-segment-smoke-iter0-steps8-sim1-20260511
```

Result: `ok=true`, strict load true, survival `8/8`, score `0.0`.

Manifest:

```text
training/lightzero-official-visual-pong-github-upstream/lz-visual-pong-github-upstream-segment-20260511-s1-wait/attempts/train-muzero-segment-ale-pong-v5-50k-ckpt1000-l4cpu40-wait/eval/upstream-segment-smoke-iter0-steps8-sim1-20260511/manifest_steps8_seedsn1_5feceb66ffc8_20260511T164235Z.json
```

Tiny strict evalability smoke, `iteration_5000/6000`:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_github_upstream_eval \
  --compute cpu --selected-iterations 5000,6000 --eval-seeds 0 \
  --max-eval-steps 64 --num-simulations 1 \
  --eval-id upstream-segment-smoke-5k-6k-steps64-sim1-20260511
```

Result: `ok=true` for both, strict load true for both, survival `64/64` for
both, scores `-6.0` at `iteration_5000` and `-5.0` at `iteration_6000`.
This is only a load/env/survival-smoke because it uses one seed and one MCTS
simulation. Use stock `--num-simulations 50`, more seeds, and a larger cap for
claims.

Manifest:

```text
training/lightzero-official-visual-pong-github-upstream/lz-visual-pong-github-upstream-segment-20260511-s1-wait/attempts/train-muzero-segment-ale-pong-v5-50k-ckpt1000-l4cpu40-wait/eval/upstream-segment-smoke-5k-6k-steps64-sim1-20260511/manifest_steps64_seedsn1_5feceb66ffc8_20260511T164328Z.json
```

Useful next strict claim command shape:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_github_upstream_eval \
  --compute gpu-l4-t4-cpu40 --selected-iterations 0,5000,6000,7000 \
  --eval-seed-count 16 --max-eval-steps 2048 --num-simulations 50 \
  --eval-id upstream-segment-strict-survival-rand16-stock50-20260511
```

Current blocker: no eval wrapper blocker remains for the segment lane. The
plain current-upstream config is still blocked by `PongNoFrameskip-v4` action
map drift, so this eval tool intentionally targets the official segment config
with `--env ALE/Pong-v5`.

Plain current config blocker, checked at 2026-05-11 11:48 EDT:
`zoo/atari/config/atari_muzero_config.py` still sets
`env_id = 'PongNoFrameskip-v4'`, but current
`zoo/atari/config/atari_env_action_space_map.py` only has newer ALE ids such
as `ALE/Pong-v5`. Importing the plain config from the pinned package fails
before env creation or trainer launch:

```text
KeyError: 'PongNoFrameskip-v4'
file: /usr/local/lib/python3.11/site-packages/zoo/atari/config/atari_muzero_config.py
line: action_space_size = atari_env_action_space_map[env_id]
```

Plain-config blocker artifact:

```text
training/lightzero-official-visual-pong-github-upstream/lz-visual-pong-github-upstream-exact-20260511-s0-dry/attempts/dry-exact-github-de740552-config-surface/dry_exact_github_upstream_summary.json
```

Smallest faithful current-upstream visual Pong MuZero path found:

- Current official README MuZero Pong command points to
  `python3 -u zoo/atari/config/atari_muzero_segment_config.py`.
- That official script exposes `--env`; using `--env ALE/Pong-v5` matches the
  current action map and avoids local source edits.
- Exact official command shape for a full run is:

```bash
cd LightZero
python3 -u zoo/atari/config/atari_muzero_segment_config.py --env ALE/Pong-v5 --seed 0
```

Dry gate passed at 2026-05-11 11:56 EDT:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_github_upstream_dry_check \
  --mode segment-dry --compute cpu --env-id ALE/Pong-v5 --seed 0 \
  --run-id lz-visual-pong-github-upstream-segment-20260511-s0-dry2 \
  --attempt-id dry-muzero-segment-ale-pong-v5-config-surface
```

Dry artifact:

```text
training/lightzero-official-visual-pong-github-upstream/lz-visual-pong-github-upstream-segment-20260511-s0-dry2/attempts/dry-muzero-segment-ale-pong-v5-config-surface/dry_segment_github_upstream_summary.json
```

Dry-captured official segment surface: `ALE/Pong-v5`, visual
`[4,64,64]`, 8 collectors, 3 evaluators, 50 simulations, `batch_size=256`,
`update_per_collect=None`, `replay_ratio=0.25`, `num_segments=8`,
`game_segment_length=20`, `train_start_after_envsteps=2000`,
`save_ckpt_after_iter=1000000`, CUDA on, and `max_env_step=500000`.

Faithful-short current-upstream run spawned at 2026-05-11 11:56 EDT:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_github_upstream_dry_check \
  --mode segment-train --compute gpu-l4-t4-cpu40 --env-id ALE/Pong-v5 --seed 0 \
  --run-id lz-visual-pong-github-upstream-segment-20260511-s0-short \
  --attempt-id train-muzero-segment-ale-pong-v5-50k-ckpt1000-l4cpu40 \
  --max-env-step-override 50000 \
  --save-ckpt-after-iter-override 1000
```

Spawn result:

```text
function_call_id: fc-01KRBW0GVFD5N1K85ASWF98YAD
progress_ref: training/lightzero-official-visual-pong-github-upstream/lz-visual-pong-github-upstream-segment-20260511-s0-short/attempts/train-muzero-segment-ale-pong-v5-50k-ckpt1000-l4cpu40/train/progress/latest.json
```

Immediate and follow-up Volume polls after spawn found no progress JSON and no
`lightzero_segment_exp` directory yet. Do not count it as running until a
Volume progress file appears.

Official older-release scout:

- `git ls-remote https://github.com/opendilab/LightZero.git refs/heads/main 'refs/tags/*'`
  showed current `main` at `de74055298068f53b70e07bc38c41101fce51766`.
- The official `v0.2.0` tag peels to
  `44a23b532f8516b7dd5c61105d6e5dd28c79dc0a`, and its Atari action map still
  contains `PongNoFrameskip-v4`. That means the literal older-release command
  `python3 -u zoo/atari/config/atari_muzero_config.py` is a real stock visual
  Pong/MuZero example for the release surface, but it does not answer current
  upstream exact.
- The official docs describe building a local image from the repository
  `Dockerfile`; no official prebuilt Docker image identity was found in the
  docs checked here.

This should not block the installed-package replication controls above.

## Board-Game LightZero Controls

These are quick controls for the same framework family on delayed terminal
PvP-style rewards. They are not visual Atari controls.

| status | run_id | attempt_id | command | checkpoint dir | what it proves |
| --- | --- | --- | --- | --- | --- |
| completed | `lz-tictactoe-control-train-20260511-s200` | `train-smoke-stock-lightzero-20260511` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_tictactoe_tiny_train_smoke --mode train --run-id lz-tictactoe-control-train-20260511-s200 --attempt-id train-smoke-stock-lightzero-20260511` | `training/lightzero-official-tictactoe/lz-tictactoe-control-train-20260511-s200/checkpoints/lightzero` | stock TicTacToe MuZero train path runs and mirrors `ckpt_best`, `iteration_0`, `iteration_10` |
| completed | `lz-connect4-control-train-20260511-s201` | `train-smoke-stock-lightzero-20260511` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_connect4_tiny_train_smoke --mode train --run-id lz-connect4-control-train-20260511-s201 --attempt-id train-smoke-stock-lightzero-20260511` | `training/lightzero-official-connect4/lz-connect4-control-train-20260511-s201/checkpoints/lightzero` | stock Connect4 MuZero train path runs and mirrors `ckpt_best`, `iteration_0`, `iteration_10` |
| completed | `lz-tictactoe-control-progression-20260511-s203` | `progression-stock-lightzero-20260511-s203` | `uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_tictactoe_tiny_train_smoke --mode progression --seed 203 --run-id lz-tictactoe-control-progression-20260511-s203 --attempt-id progression-stock-lightzero-20260511-s203` | `training/lightzero-official-tictactoe/lz-tictactoe-control-progression-20260511-s203/checkpoints/lightzero` | stock TicTacToe MuZero progression control completed on app `ap-EpptI16rpSAR6pKTpI708I`; mirrors `ckpt_best`, `iteration_0`, `iteration_10` |
| completed | `lz-connect4-control-progression-20260511-s202` | `progression-stock-lightzero-20260511-s202` | `uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_connect4_tiny_train_smoke --mode progression --seed 202 --run-id lz-connect4-control-progression-20260511-s202 --attempt-id progression-stock-lightzero-20260511-s202` | `training/lightzero-official-connect4/lz-connect4-control-progression-20260511-s202/checkpoints/lightzero` | stock Connect4 MuZero progression control completed on app `ap-NYWUtcn1zs6laSntxPpqce`; mirrors `ckpt_best`, `iteration_0`, `iteration_10` |

## Coach Follow-up Distinct Controls - 2026-05-11

No new duplicate installed `LightZero==0.2.0` 64x64 Pong train was launched in
this follow-up.

| target | source | why it matters for CurvyTron visual MuZero | command/run id | checkpoint/eval location | current status |
| --- | --- | --- | --- | --- | --- |
| Current GitHub LightZero Atari Pong exact | `opendilab/LightZero@de74055298068f53b70e07bc38c41101fce51766`, `zoo.atari.config.atari_muzero_config` | Tests whether current upstream stock visual MuZero Pong can be an exact reference separate from installed `0.2.0` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_github_upstream_dry_check --run-id lz-visual-pong-github-upstream-exact-20260511-s0-dry --attempt-id dry-exact-github-de740552-config-surface` | `training/lightzero-official-visual-pong-github-upstream/lz-visual-pong-github-upstream-exact-20260511-s0-dry/attempts/dry-exact-github-de740552-config-surface/dry_exact_github_upstream_summary.json` | Dry gate failed before env/train: `KeyError: 'PongNoFrameskip-v4'`; no train launched |
| Installed LightZero 96x96 MuZeroAgent model-card path | `LightZero==0.2.0`, `lzero.agent.MuZeroAgent`, `PongNoFrameskip-v4`, `[4,96,96]` | Separate visual MuZero surface from 64x64 `train_muzero`; closer to OpenDILab model-card usage | Already owned: `s125` model-card H100 and `s127` short H100 in this doc | `training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s125-modelcard-detached/attempts/train-agent96-modelcard-500k-h100cpu40-detached/train/agent_exp/ckpt`; `training/lightzero-official-visual-pong-muzero-agent96/lz-visual-pong-muzero-agent96-20260511-s127-short-h100-detached/attempts/train-agent96-50k-h100cpu40-ckpt1000-detached/train/agent_exp/ckpt` | Already active/owned; no duplicate Agent96 launch here |
| TicTacToe board-game plumbing | Stock LightZero `zoo.board_games.tictactoe.config.tictactoe_muzero_bot_mode_config` | Checks LightZero board-game search/terminal-reward plumbing; not visual | `uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_tictactoe_tiny_train_smoke --mode progression --seed 203 --run-id lz-tictactoe-control-progression-20260511-s203 --attempt-id progression-stock-lightzero-20260511-s203` | `training/lightzero-official-tictactoe/lz-tictactoe-control-progression-20260511-s203/checkpoints/lightzero`; summary `training/lightzero-official-tictactoe/lz-tictactoe-control-progression-20260511-s203/attempts/progression-stock-lightzero-20260511-s203/train/summary.json` | Completed; `ok=true`; mirrored `ckpt_best`, `iteration_0`, `iteration_10` |
| Connect4 board-game plumbing | Stock LightZero `zoo.board_games.connect4.config.connect4_muzero_bot_mode_config` | Checks larger board-game terminal-reward plumbing; not visual | `uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_connect4_tiny_train_smoke --mode progression --seed 202 --run-id lz-connect4-control-progression-20260511-s202 --attempt-id progression-stock-lightzero-20260511-s202` | `training/lightzero-official-connect4/lz-connect4-control-progression-20260511-s202/checkpoints/lightzero`; summary `training/lightzero-official-connect4/lz-connect4-control-progression-20260511-s202/attempts/progression-stock-lightzero-20260511-s202/train/summary.json` | Completed; `ok=true`; mirrored `ckpt_best`, `iteration_0`, `iteration_10` |
| Non-LightZero quick check | `mctx==0.0.6` JAX Gumbel MuZero search smoke | Confirms outside-LightZero search runtime can execute; not a trainer | Previously launched: `uv run --extra modal modal run -m curvyzero.infra.modal.mctx_dependency_smoke --kind cpu`, app `ap-E3GwfOaPOHu8Ce3IeFj5KA` | No checkpoint expected; stdout result recorded in `docs/working/non_lightzero_control_scout_2026-05-11.md` | Passed runtime smoke; no learning/eval claim |

## Repo-Native Visual Delayed-Reward Check

This is not stock Atari Pong and not CurvyTron. It is the closest repo-native
LightZero example already verified for delayed Pong-like play plus visual-ish
input.

| status | run_id | attempt_id | command | checkpoint dir | what it proves |
| --- | --- | --- | --- | --- | --- |
| verified | `lz-dpong-raster-flat-h120-lag1-s10` | `train-512x8-raster-h120` | `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt --mode progression --env dummy_pong_lag1 --feature-mode raster_flat --opponent-policy lagged_track_ball_1 --max-env-step 512 --pong-episode-max-steps 120 --max-train-iter 8 --num-simulations 8 --batch-size 32 --update-per-collect 8 --n-evaluator-episode 4 --collector-env-num 1 --evaluator-env-num 1 --n-episode 2 --game-segment-length 50 --td-steps 120 --num-unroll-steps 5 --discount-factor 1.0 --reward-support-min -5 --reward-support-max 6 --reward-support-delta 1 --value-support-min -5 --value-support-max 6 --value-support-delta 1 --seed 10 --run-id lz-dpong-raster-flat-h120-lag1-s10 --attempt-id train-512x8-raster-h120` | `training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/checkpoints/lightzero` | repo-native LightZero can train and mirror checkpoints for delayed dummy Pong with `raster_flat`; it is only a bridge smoke because the raster has no frame history |

## Polling Plan

Poll only, no tests:

```bash
uv run --extra modal modal volume get curvyzero-runs \
  training/lightzero-official-visual-pong/<run_id>/attempts/<attempt_id>/train/progress/latest.json \
  /tmp/latest.json

uv run --extra modal modal volume ls curvyzero-runs \
  training/lightzero-official-visual-pong/<run_id>/attempts/<attempt_id>/train/lightzero_exp/ckpt
```

## Eval Tooling Fix

The strict stock Pong eval bundle hit `_pickle.UnpicklingError: invalid load
key, 'A'` during ALE env reset, consistent with ALE banner bytes corrupting
DI-engine subprocess env-manager pipes. The smallest eval-only fix is now in
`src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`: the stock evaluator
probe deep-copies the stock create config, then forces the compiled evaluator
env manager to `base`, keeps `evaluator_env_num=1`, and still uses
`lzero.worker.MuZeroEvaluator`.

Exact non-claim: this does not claim that any checkpoint improves, that the
strict eval bundle now passes remotely, or that Pong learning has been shown.
It only makes stock evaluator eval instantiate one in-process/base env manager
to avoid the observed subprocess pipe corruption path. Training remains on the
stock subprocess env-manager surface.

Fresh eval now running under this posture:
`s114-0-1k-2k-stockonly-baseeval-rand16`.

Review note: the code path is scoped to `_run_stock_evaluator_probe` after
`compile_config`, so it does not alter train-time `create_config`. The only
artifact-safety improvement added here is that failure JSON now also includes
`env_manager_patch` if the run fails after the override is applied.

First eval posture once checkpoints exist:

- stock evaluator only
- strict checkpoint load, no fallback
- serious search: `50` MCTS simulations/action
- survival-first curve, with stock return as secondary metric
- eval cap and episode cap: `2048`
- reproducible random seed panel, rng seed `20260511`, seeds:
  `1250426297,611354690,2130530255,551302474,1777247108,1588563970,1580338265,995705522`

What proves learning:

- For stock Pong, compare same-run `iteration_0` against later checkpoints.
- The main signal is higher stock steps survived on the fixed seed panel.
- Stock return is secondary.
- Action histograms must not collapse to one bad action.
- A single lucky episode is not enough; the curve should move across
  checkpoints or across several launched seeds.

## Non-LightZero Control

No large non-LightZero framework was launched. The current safest optional
control remains OpenSpiel AlphaZero Connect Four, but it needs a clean local or
Modal command before it should enter the live matrix.

## Modal Pong Cleanup Audit

2026-05-11 14:51 EDT: Pong replication is considered validated enough for this
lane, so active Modal Pong jobs were stopped after a broad `modal app list
--json` inventory and spot log checks.

Stopped:

- `ap-bwtOn568WTejRiHrIW78G9` — `curvyzero-lightzero-pong-exact-reproduction`,
  created 2026-05-11 11:16 EDT, stopped 2026-05-11 14:51 EDT.
- `ap-JvwAnokQFvcL0XySLAGkqG` — `curvyzero-lightzero-pong-exact-reproduction`,
  created 2026-05-11 11:22 EDT, stopped 2026-05-11 14:50 EDT.
- `ap-UcZ6lCQbizAjjyulRFCUvS` —
  `curvyzero-lightzero-pong-muzero-agent-reproduction`, created 2026-05-11
  11:34 EDT, stopped 2026-05-11 14:51 EDT.
- `ap-l8lCW1sSR296pjwupy2B8C` —
  `curvyzero-lightzero-pong-github-upstream-dry-check`, created 2026-05-11
  12:11 EDT, stopped 2026-05-11 14:50 EDT.
- `ap-vRptjltbIRoLipQe6UKggm` —
  `curvyzero-lightzero-pong-muzero-agent-reproduction`, created 2026-05-11
  12:12 EDT, stopped 2026-05-11 14:51 EDT.

Left alone:

- `ap-pzRnD0oXuFYb4N7yWzORA3` —
  `curvyzero-lightzero-curvytron-two-seat-train-smoke`, active with 1 task.
- Older deployed non-Pong apps from the same Modal environment, including
  `dark-star`, benchmark, giphius, lovable, humanx, and imaginator services.

Uncertain:

- None for active Pong. No active deployed app name looked Pong-related outside
  the five stopped jobs.
