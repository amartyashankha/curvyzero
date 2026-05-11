# LightZero Pong Scale Performance Critique - 2026-05-09

Role: Amdahl/performance critique. No code changes. No pytest.

## Short Read

Do not scale just to scale.

For the current LightZero dummy Pong lane, the biggest known blocker is not raw
environment speed. It is trust and learning quality: repeated seed telemetry
made trainer-side results look much better than held-out checkpoint eval, and
the independent MCTS scorecard still shows degenerate action use.

Performance-wise, tiny runs are dominated by fixed job/framework overhead. A
dry LightZero Pong config smoke took about `12.2s`, and the completed 512/8
dummy Pong train reported about `12.3s` of trainer elapsed. That makes env
micro-optimization a poor first target for the current job size.

## 1. Current Likely Bottleneck

Current end-to-end small jobs:

- Modal image/function startup, LightZero import/config, trainer setup, and
  artifact scan/copy are a large share of wall time.
- Artifact handling is not free: each mirrored LightZero checkpoint is around
  `27-55 MB`, and the current wrapper scans and copies checkpoints after the
  trainer returns.
- The actual hot loop is not instrumented enough to rank env step vs MCTS vs
  learner update.

Current trainer hot loop, best guess:

- Python dummy Pong env step is probably not the main bottleneck yet.
- LightZero collector/trainer/search overhead is more likely than env physics
  for this tiny `tabular_ego` MLP setup.
- `num_simulations=4` or `8`, `batch_size=16` or `32`, and one collector env
  are too small to make GPU throughput or vector env throughput the obvious
  limit.

So the answer is split:

| Scope | Likely bottleneck now |
| --- | --- |
| Tiny whole Modal job | Modal/LightZero startup plus artifact handling |
| Training hot loop | LightZero trainer/collector/MCTS overhead, not proven env step |
| Learning progress | Seed/eval trust and objective/wiring, not throughput |

Amdahl read: speeding env step by 10x does little if env step is only a small
slice of a `12s` job dominated by setup and framework overhead.

## 2. Does Env Vectorization Help Here?

Not much with the current LightZero adapter.

`DummyPongLightZeroEnv` is a DI-engine `BaseEnv`. It exposes:

```text
reset() -> one observation dict
step(action) -> one BaseEnvTimestep
```

The config uses `create_config.env_manager.type = "base"`. Raising
`collector_env_num` gives LightZero more env instances, but the env interface is
still Python one-env-at-a-time. LightZero may batch policy calls across ready
env ids, but the current dummy Pong env does not expose a true
`step_many(actions[B])` NumPy path to LightZero.

Vectorizing the internal dummy Pong core would help only after one of these
changes:

- a custom env manager calls a batched `step_many`;
- the LightZero adapter owns many Pong rows inside one env instance;
- we bypass LightZero collection for a project-owned actor loop.

Until then, vectorization work is useful for the future CurvyZero actor loop,
but it is not an easy speed win for the current LightZero trainer.

## 3. Easy Scale Knobs Now

Easy knobs already surfaced in the wrappers:

| Knob | Current examples | Use carefully |
| --- | --- | --- |
| `max_env_step` | `512`, `1024`, `4096` | Currently also feeds env `max_steps`; this mixes train budget and episode horizon. |
| `max_train_iter` | `8`, `16`, `64` | Simple whole-job scale knob. |
| `collector_env_num` | mostly `1` | More env instances, not true vector stepping. Measure before trusting. |
| `evaluator_env_num` | mostly `1` | Useful for eval throughput once seed offsets are honest. |
| `n_evaluator_episode` | `4`, `8` | Raises eval cost and score confidence. |
| `num_simulations` | `4`, `8` | Directly raises MCTS cost; best search-quality knob to sweep. |
| `batch_size` | `16`, `32` | Learner minibatch, not env vector batch. |
| `update_per_collect` | `1` | More learner work per collected segment. |
| seeds/sweeps | seed `1`, `2`, planned `3` | Most important next scale axis after seed fix. |

Also useful but secondary: `opponent_policy`, `feature_mode`, scorecard
episodes, and checkpoint-selection policy.

## 4. When GPU Helps

GPU is probably overhead right now.

The current LightZero dummy Pong run uses:

- `cuda: false`;
- `tabular_ego` observation shape `10`;
- `MuZeroModelMLP`;
- action space `3`;
- low `num_simulations`;
- small batch sizes.

That is not a good GPU workload. A GPU would mostly add device transfer,
synchronization, cold-start, and framework overhead unless the batch/search
work gets much larger.

GPU starts to make sense when one of these is true:

- many roots are searched together, e.g. batched MCTS/Mctx roots;
- `num_simulations` and hidden size are large enough to saturate the device;
- the model becomes visual/conv or otherwise much larger;
- learner batches are large enough that forward/backward time dominates;
- CPU actors can keep a central GPU inference/search worker busy;
- host-device transfer is measured and amortized.

The Mctx synthetic benchmark is encouraging for future batched GPU search, but
it is not evidence that this LightZero MLP dummy Pong setup needs GPU today.

## 5. Measurements To Add Before Optimizing

Add timing buckets before changing architecture:

- Modal/client elapsed, remote function elapsed, and trainer elapsed.
- LightZero import/config/setup time.
- Env steps/sec and episodes/sec.
- Collector time, evaluator time, learner/train-update time.
- MCTS action latency: p50/p95/p99, plus `num_simulations`.
- Policy/model forward time separate from tree search if possible.
- Replay/staging time, checkpoint save time, artifact scan/copy time.
- Seed counts: unique seeds, most common seed fraction, row source labels.
- Checkpoint eval time: load time, per-episode time, action latency.
- Eval scorecard cost by opponent and episode count.

Keep this simple. One JSON timing block in `summary.json` is enough to start.

## 6. Recommended Next Scale Experiments

First, run the planned post-seed-fix trust run:

```text
1024 max_env_step, 16 train iters, 8 simulations, batch 32, seed 3
```

If that still has repeated-seed dominance or zero `down` actions in independent
scorecard, stop scaling and debug wiring/objective/seed flow.

If it passes the trust gate, run these next:

1. Same-size seed sweep.
   Run `1024/16` over 3-5 seeds with the same independent held-out MCTS
   scorecard. Goal: variance, seed health, action diversity, not bigger compute.

2. Search-cost sweep.
   Fixed `1024/16`, compare `num_simulations` like `2`, `8`, `16` with timing
   and scorecard. Goal: learn whether MCTS helps or just burns time.

3. Collector/evaluator scale probe.
   Fixed total budget, compare `collector_env_num=1,4,8` and
   `evaluator_env_num=1,4` if LightZero accepts it cleanly. Goal: measure
   collector/evaluator throughput and seed diversity. Treat this as a timing
   probe, not a quality claim.

Only after those should we consider a larger `2048/32` or `4096/64` rerun.
The long `4096/64` run already showed that more length alone did not fix the
policy collapse.
