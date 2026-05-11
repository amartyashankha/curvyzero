# Eval Speed Investigation - 2026-05-11

Purpose: explain why Pong and CurvyTron eval bundles are slower than expected.
This is a runtime note only. It does not judge training quality.

## Short Answer

## Top-Line Correction: Pong Real-Time Concern

Important correction: the `3s` average / `10s` p95 number below is from
CurvyTron eval artifacts with short episodes. It is not the timing for a
serious Atari Pong eval.

Old Pong artifacts show the real scale:

- Hardware was Modal cheap GPU: `gpu-l4-t4`, `gpu-l4-t4-cpu8`, or
  `gpu-l4-t4-cpu40`, which requests an L4 or T4 GPU plus the listed CPUs.
- These artifacts did not prove whether every LightZero stock-evaluator model
  call actually stayed on GPU.
- The examples were not full 2048-step games. They died around `833-848`
  action steps and still took about `90-104s` in the stock evaluator.
- Older manual+stock Pong evals took about `216-245s` per checkpoint/seed
  because they ran two rollouts: our manual rollout plus LightZero's stock
  evaluator rollout.
- Stock-only speed from those examples is roughly `100s / 840 actions`, or
  about `0.12s/action` and `8 actions/sec`.
- Atari Pong with frame skip 4 needs about `15 actions/sec` to keep up with
  60 FPS game time. So the current serious 50-search eval mode is not real-time
  on those artifacts. If a game reaches the full `2048` action cap at the same
  speed, it could take roughly four minutes.

Measured fact: full 50-simulation MCTS eval was slower than real-time in these
old Pong artifacts. This does **not** measure direct policy-head action speed,
tiny-search action speed, batched search speed, H100 speed, or a deployment
serving path.

Likely reason: this is not just one GPU inference per action. It is LightZero
MCTS/search around the model, `50` simulations per action, largely one root at
a time, with Python/tree bookkeeping and small model calls. That can leave the
GPU underused even when a GPU worker is requested.

Bundle arithmetic: if one stock-only 2048-cap Pong game takes around
`90-104s` for ~840 actions, then a fully parallel 64-game eval bundle should
be near the slowest single game plus Modal startup/capacity delay. If observed
bundle wall time is much longer than that, possible explanations include hidden
serial grouping, capacity delay, or accidental manual+stock duplication. This
has to be measured per bundle, not assumed.

The current evals are not "one game, copied many times." They are many
independent checkpoint/seed jobs. Each job usually:

1. reads the checkpoint from the Modal Volume,
2. builds and strict-loads a LightZero MuZero policy,
3. builds one env,
4. runs one episode with `policy.eval_mode.forward(...)` on every step, which
   means MCTS/search, and
5. writes one JSON artifact and commits the Volume.

Standalone CurvyTron and Pong curve evals do use Modal `Function.starmap` across
checkpoint/seed jobs, so they are not simple local serial loops. But each job
still pays the full setup and search cost, and Modal does not make hundreds of
GPU jobs complete in exactly the time of the single fastest episode. The live
background CurvyTron checkpoint eval path is worse: it spawns one inspector job
per checkpoint, then loops over seeds serially inside that job.

Current likely bottleneck:

- Per episode: MCTS/search plus Python eval-loop overhead. Raw env stepping and
  frame stack work are much smaller in the probes we have.
- Per bundle wall time: repeated checkpoint/policy/env setup, Modal scheduling
  and cold-start/capacity effects, per-job Volume writes/commits, and the live
  path's serial seed loop.

## What Is Measured

Code path facts:

- CurvyTron standalone eval builds one job per `(seed, checkpoint)` and uses
  `eval_fn.starmap(...)` when there is more than one job. See
  `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py`.
- Each CurvyTron job times only two coarse clocks today:
  `episode.elapsed_sec` around the step loop, and `remote_elapsed_sec` around
  checkpoint load, policy/env setup, episode, artifact write, and Volume commit.
- The CurvyTron step loop calls `_policy_eval_action(...)` and then
  `env.step(...)` every step. It does not currently split policy forward,
  search, preprocessing, env step, or artifact timing.
- The live/background CurvyTron eval-and-inspect function accepts only CPU
  background evals and loops `for eval_seed in eval_seed_values`, calling
  `_run_eval(...)` once per seed. That reloads/rebuilds for every seed.
- Pong Agent96 strict eval also builds checkpoint/seed jobs and uses
  `starmap(...)`; it runs LightZero's stock `MuZeroEvaluator` per job.
- Older Pong eval smoke can run stock-only or manual-plus-stock. The queue helper
  defaults to `--stock-only`, which avoids the duplicate manual rollout. Serious
  Pong eval defaults to 50 MCTS simulations/action; telemetry mode uses 5 unless
  overridden.

Artifact timing:

- CurvyTron `s101_fixed64` manifest has 384 checkpoint/seed rows. Per-job
  `elapsed_seconds` mean is about `2.99s`, p50 `2.20s`, p95 `10.28s`, max
  `13.19s`.
- Fetched raw JSON subset for `s101_fixed64` has 378 jobs. Mean remote elapsed
  is `2.91s`; mean episode elapsed is `2.02s`; mean non-episode overhead is
  `0.90s`. Median overhead is only `0.27s`, but p95 overhead is `8.06s`.
- Fetched raw JSON subset for `s102_fixed64` has 381 jobs. Mean remote elapsed
  is `2.98s`; mean episode elapsed is `2.11s`; mean non-episode overhead is
  `0.87s`. Median overhead is `0.26s`, p95 overhead is `7.84s`.
- A representative `s101` job loaded a `~96 MB` checkpoint on CUDA and ran 112
  eval steps in `1.316s` episode time, `1.600s` total remote time.
- The slowest fetched `s101` job ran 312 steps in `4.745s` episode time but
  `13.186s` remote time, so setup/write/cold overhead was about `8.44s`.
- A local no-train debug visual adapter probe on this checkout ran 512 adapter
  transitions with stack+copy in `0.0766s` wall, with `env_step_total=0.0729s`
  and `frame_stack_update_copy=0.0031s`. This is not the exact source-state
  eval surface, but it shows raw env/render/stack work is not multi-second.
- Full Atari Pong examples found locally:
  - `normal-s27` `iteration_5000`, 2048-step cap: stock evaluator
    `103.98s`, manual rollout `119.18s`, total remote `244.66s`.
  - `sweep65k-s18` `iteration_13000`, 2048-step cap: stock evaluator
    `99.55s`, manual rollout `122.50s`, total remote `235.96s`.
  - `repeatA` `iteration_2000`, 2048-step cap: stock evaluator `90.59s`,
    manual rollout `112.61s`, total remote `216.44s`.
  These old artifacts did not have phase timing enabled, but they make the
  one-game Pong cost clear.

Measured but easy to miss:

- A 384-job CurvyTron eval over `~96 MB` checkpoints implies tens of GB of total
  checkpoint reads across the bundle because the same checkpoint is loaded once
  per seed. That can be parallel, but it is not free.
- Current manifests record per-job elapsed time. They do not record the local
  `starmap` wall time, queue/cold-start time before `_run_eval` starts, or an
  effective concurrency count. So the most important bundle-level number is
  currently missing.

## What Is Inferred

The per-step slowdown is mostly search, not env stepping. The strongest evidence
is the gap between no-policy adapter throughput and eval episode throughput:
thousands of raw adapter transitions/sec locally versus roughly tens of searched
eval decisions/sec in fetched CurvyTron artifacts. Existing optimizer notes also
show LightZero Pong training slices dominated by evaluator plus collector/env/MCTS
time, not learner update time.

GPU helps only when enough model/search work is batched or the worker stays
warm. Current standalone CurvyTron serious evals request `gpu-l4-t4-cpu40` and
the fetched artifacts show model device `cuda`. Pong Agent96 strict eval defaults
to a GPU worker too. But one env, one root, and one action at a time under
Python/MCTS is not a GPU-saturating workload. Background CurvyTron eval is CPU
only today, so it gets none of the GPU benefit.

Modal Volume access is a contributor, but not yet isolated. The coarse overhead
bucket contains checkpoint read, `torch.load`, config compile, policy init,
state-dict load, env creation, artifact write, Volume commit, and warm/cold
effects. We cannot honestly say which subpart dominates until the runner emits
phase timings.

Python/container startup and Modal scheduling probably explain part of the
"bundle wall time is not close to one game" feeling, but current JSON does not
measure it. `remote_elapsed_sec` starts inside the function body, after the
function has been invoked.

## Parallelism Answer

Current standalone bundles:

- CurvyTron visual survival eval: parallel across checkpoint/seed jobs with
  `starmap(...)`.
- Pong Agent96 strict stock eval: parallel across checkpoint/seed jobs with
  `starmap(...)`.
- Older Pong eval smoke: parallel when multiple refs or seeds are passed; queue
  helper groups checkpoints and can keep many `modal run` processes in flight
  with `--execute`.

Current accidental serial work:

- CurvyTron live checkpoint eval-and-inspect: one spawned job per checkpoint,
  seeds serial inside that job.
- Every checkpoint/seed job reloads the same checkpoint for each seed. This is
  parallel wall-clock work, but serial duplicated work inside the total bundle.
- If a human runs printed queue-helper groups one by one instead of using
  `--execute`, group launch becomes serial outside the Python eval runner.

Plain expectation:

- If one stock-only 2048-step Pong game takes around `90-104s`, then a fully
  parallel 64-game bundle should be roughly a couple of minutes plus Modal
  startup/capacity delay.
- If it takes much longer, the likely issue is capacity/cold-start delay,
  hidden serial grouping, or duplicated manual+stock rollouts.
- If CurvyTron live eval takes much longer than its single-game time, the known
  issue is the CPU inspector path serializing seeds inside one job.

## Simple Fixes To Own Now

1. Add phase timing to CurvyTron eval JSON.
   Required buckets: checkpoint path wait/reload, checkpoint read, `torch.load`,
   config build/compile, policy init, state-dict load, env make, reset,
   policy/search total/count, env step total/count, preprocessing/stack,
   artifact write, Volume commit, and total remote wall.

2. Add bundle-level timing to standalone eval manifests.
   Record local submit start/end, `starmap` wait elapsed, manifest write elapsed,
   job count, seed count, checkpoint count, sum/max/mean/p50/p95 remote elapsed,
   and a simple `sum_remote_elapsed / starmap_wall_elapsed` concurrency estimate.

3. Stop using the live background eval path for serious seed panels.
   Use it for a one-seed smoke, or change it so seeds are either submitted via
   `starmap` or evaluated in one worker after loading the checkpoint once.

4. Do not rerun duplicate Pong manual-plus-stock eval unless debugging parity.
   Keep `--stock-only` for scorecard/monitoring work.

5. Use telemetry evals deliberately.
   For quick trend checks, lower `num_simulations` and label them as telemetry.
   For final scorecards, keep the serious search setting and accept the cost.

## Deeper Work

- Build a batched eval worker that loads one checkpoint once, runs many seeds or
  envs, batches policy/search roots where LightZero allows it, then writes one
  manifest and commits once.
- Keep warm evaluator workers for a checkpoint bundle instead of starting many
  independent one-episode jobs.
- Split model forward time from MCTS tree bookkeeping and Python action/env
  overhead. If search remains the dominant bucket, optimizer can decide whether
  batching, Mctx, or a thinner search path is worth owning.
- For GPU eval, batch enough roots/games to make CUDA useful. One tiny eval
  decision at a time leaves GPU speed hidden behind Python and setup cost.

## Future Eval Contract

Every eval artifact should answer these questions without archaeology:

- What was the job? `run_id`, `attempt_id`, `eval_id`, checkpoint ref, checkpoint
  bytes/hash, seed, max steps, env variant, opponent kind, `num_simulations`,
  batch size, requested compute, actual model device, Modal task id.
- What was wall time? Function remote wall, episode wall, setup wall, artifact
  write wall, Volume commit wall.
- What happened per step? Step count, policy/search count, total/mean/p95
  policy/search sec, total/mean/p95 env step sec, preprocessing/stack sec,
  action histogram, terminal reason.
- What happened at the bundle level? Job count, checkpoint count, seed count,
  local submit/starmap/manifest wall time, sum/max/p95 remote job time, effective
  concurrency estimate, failed/cancelled jobs, and queue/cold-start caveat.
- What was missing? If a bucket is not measured, say `not_measured`, not `0`.

Until this is added, the honest answer is: current evals are slow because each
checkpoint/seed job repeats setup and then runs searched policy decisions; some
paths are parallel at Modal job level, but the live background path is serial
over seeds and no path currently reports enough phase timing to make the bundle
wall time self-explaining.
