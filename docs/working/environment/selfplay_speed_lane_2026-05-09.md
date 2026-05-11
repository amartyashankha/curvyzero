# Self-Play Speed

Date: 2026-05-09

Status: working note for the speed lane. This does not change the environment
API.

## Short Answer

The real speed goal is not just a fast `env.step()`.

MuZero-style self-play needs a loop that can do all of this many times:

```text
many live env rows
  -> many ego observation rows
  -> one batched policy/search/model call
  -> wrapper action maps (`joint_action` sidecars) for each real env
  -> fast deterministic env stepping
  -> chunked replay rows for training
```

The current vector prototype is useful, but it is still synthetic. The next
serious speed work should measure a whole self-play loop: source-verified env
rows, policy/search decisions, resets, and replay staging.

Architecture rule: the end goal is one fast faithful environment.
`CurvyTronSourceEnv` is a temporary executable spec/oracle harness used to avoid
lying while the rules are ported into the fast path. Speed work is not a
separate game or an independent lane: every optimization must name the source
contract it matches.

Amdahl guardrail: optimize env-step only with measurements against the whole
self-play loop, including MCTS/search/model cost, observation packing, reset,
and replay.

Plain priority answer:

1. Make the real actor loop fast enough to produce useful games/minute.
2. Keep action latency low enough that rows are not waiting on giant batches or
   stale policy/search results.
3. Keep the CPU environment, observation packing, replay staging, and
   model/search boundary measured in one report.
4. Delay GPU env work until the full loop proves CPU env stepping is the
   production bottleneck and data can stay on device.

Current local/debug evidence supports that ordering. The toy object-env
parallel bridge serial run reported about `19,937` wall env steps/sec,
`p95=1.391ms`, `p99=1.678ms`, and `env_step=94.5%`; its thread mode hurt tail
latency. The fixture-seeded vector bridge at `B=32`, no-event,
`simulations=1`, reported `P2_K4 env_step=46.7%`, synthetic policy/search
`7.9%`, replay `1.7%`; and `P3_K4 env_step=66.2%`, synthetic policy/search
`9.8%`, replay `1.8%`. These are local debug timings with fixture cycling,
synthetic NumPy policy/search, no real MCTS, no learned model, no GPU, no
LightZero collector, and no production replay writer.

So the next optimization target is not "put the env on GPU." It is: replace the
fake policy/search with calibrated real model/MCTS timing, measure host/device
transfer if GPU search is used, and then choose the largest production-relevant
bucket.

Practical component order:

| Component | Priority | Why |
| --- | --- | --- |
| Full actor-loop report | First | It answers the real question: useful games/minute plus p95/p99 action latency. |
| Env step | Optimize if it remains largest after debug events are off and real search timing is in. | Every transition needs it, but env-only work is capped if MCTS/model dominates. |
| Model/search/MCTS | Measure immediately; optimize as soon as calibrated timing is large. | Synthetic policy/search is cheap today, but real search can become the main wall-time bucket. |
| Observation packing | Keep fixed-shape and simple; optimize when visible or copy-heavy. | Bad ego-row packing can erase env gains and add transfer cost. |
| Replay | Keep chunky array staging; measure production writes next. | In-memory replay is small in debug scouts, but disk/object-store/learner handoff is not proven. |
| Reset/autoreset | Make correct and timed before clever. | Terminal rows can stall batches or corrupt replay if final transition and reset state are mixed. |

Latency versus throughput: throughput is finished useful work per minute; latency
is how long a ready observation waits for its action. A giant batch can improve
rows/sec while making actions stale. A small batch can feel responsive while
underusing model/search hardware. The target is enough throughput without bad
p95/p99 action latency or policy staleness.

## Speed Target

The target is useful completed games per minute at acceptable policy/search
quality, plus p50/p95/p99 action latency. Raw steps/sec is a support metric, not
the goal by itself.

Latency matters because self-play is sequential. A huge batch can look fast in
throughput while actions wait too long or use stale policy/search results.

Document these exact metrics next:

| Metric | Why |
| --- | --- |
| Single-env step p50/p95/p99 | Shows the small-row baseline and tail cost. |
| Batch steps/sec | Shows array efficiency, but not enough alone. |
| Completed games/min | Measures useful self-play output. |
| Actor-loop p95 action latency | Catches stale huge batches. |
| Reset/terminal overhead | Finished games must recycle rows cleanly. |
| Policy/search timing | Separates env cost from decision cost. |
| Host/device transfer time | Decides whether GPU boundaries help or hurt. |

Plain options to compare:

| Option | Use it for | Watch out for |
| --- | --- | --- |
| Python/NumPy sync baseline | First honest end-to-end loop. | May hide tail latency if only steps/sec is reported. |
| Bigger sync batches | Better array throughput. | Can increase action latency and policy staleness. |
| CPU worker pool | Parallel env stepping after sync baseline. | IPC and reset coordination can dominate. |
| Async actor pool | More completed games/min. | Needs latency and policy-staleness checks. |
| Native/Numba hot loops | Branchy CPU env hot spots. | Only worth it after profiling. |
| JAX/GPU env | Env stays on device with model/search. | Bad fit if rows bounce CPU/GPU or branch heavily. |
| GPU model/MCTS with CPU env | Likely near-term boundary to test. | Transfer and batching policy decide whether it wins. |
| Central inference | Better model utilization. | Queueing can hurt action latency. |
| Fewer real steps | Better policy/search/curriculum efficiency. | Must preserve source-fidelity claims. |

Near-term top three:

1. Optimize the Python/NumPy sync actor loop with honest completed-games/min and
   latency metrics.
2. Test the CPU env plus GPU model/MCTS boundary with transfer timing included.
3. Try a CPU actor pool after the sync baseline is clear.

Work that can proceed in parallel:

- Fidelity lane: keep expanding source fixtures, event semantics, timer/reset
  behavior, and unsupported-case labels.
- Actor-loop lane: keep the CPU/NumPy vector bridge reporting latency,
  completed-game proxy, reset/autoreset, replay staging, and bucket shares.
- Model/search lane: replace the NumPy stand-in with real MCTS/Mctx or
  production-calibrated model/search timings on the same ego-row shapes.
- Replay lane: add trainer-facing chunk schemas and replay write timing without
  putting one JSON object per ego step in the hot loop.
- Modal lane: run coarse CPU/GPU sweeps only around whole jobs, not per action
  or per env step.

Do not present GPU env stepping as the obvious next move. For this small,
branchy, sequential game it may be wrong unless profiling proves the state can
stay on device and the latency is acceptable.

Worker D cheap follow-up on 2026-05-09 kept the same practical read. The
fixture-seeded vector actor bridge, with `B=32`, `simulations=1`, and debug
events disabled, reported env-step loop shares of `46.7%` for `P2_K4` and
`66.2%` for `P3_K4`; synthetic policy/search was `7.9%` and `9.8%`, and replay
staging was below `2%`. The separate toy object-env parallel bridge stayed
env-step dominated in serial mode
(`94.5%`) and threading worsened tail latency in the cheap run
(`p95=3.786ms`, `p99=6.953ms`). These are local debug timings only: fixture
cycling, synthetic NumPy policy/search, no real MCTS, no LightZero collector,
no GPU env step, and no production reset/autoreset or replay writer.

Worker SPEED follow-up on 2026-05-09 ran a small P2_K4 `B=128` actor-bridge
scout on two passing fixtures. In debug-event mode, env step cost was
`4.824 us/env row` and about `59.5%` of the loop; debug obs/reward packing was
`0.164 us/ego row`, synthetic policy/search was `0.615 us/ego row`, and replay
staging was `0.030 us/ego row`. In no-event mode, env step dropped to
`2.251 us/env row` and the actor-step p95 dropped from about `1.122 ms` to
`0.769 ms`. The simple read: debug events are a large hot-path tax, replay
staging is not the current bottleneck, and policy/search is still synthetic.
Next speed proof should rerun after replay v0 actor-bridge integration, final
observation/reward, and reset/autoreset integration land.

Source-env scout on 2026-05-09: `scripts/benchmark_source_env.py` times the
long 111-step 1v1/no-bonus wall-round-done lifecycle in the Python source env
and, with `--js`, the persistent original-JS worker. Latest local main-thread
run:
`uv run python scripts/benchmark_source_env.py --repeats 20 --js --js-repeats 3`.
It reported Python source env `0.000849s/rollout` and `130,689 steps/s`; the JS
worker reported `0.006148s/rollout` and `18,054 steps/s`. These are local scout
numbers for a narrow no-bonus lifecycle only. They are not full speed evidence,
and they are not a full fidelity claim.

## Local Inputs Read

- [Environment Performance And Vectorization Plan](../../research/environment/performance_vectorization_plan.md)
- [Environment Measurement Critique](measurement_critique_2026-05-09.md)
- [Vector State Schema Draft](vector_state_schema.md)
- [Modal Vectorization Integration Plan](modal_vectorization_integration_plan.md)
- [Training Architecture](../../design/training_architecture.md)
- [Training Loop Bottlenecks And Amdahl's Law](../../research/training_loop_bottlenecks_amdhals_law_2026-05-09.md)
- [MuZero On Modal Architecture](../../design/muzero_modal_architecture.md)
- [MuZero Architecture Deep Dive](../../research/muzero_architecture_deep_dive.md)
- [JAX/Mctx Integration Plan](../../research/mctx_integration.md)
- [Modal Mctx GPU Smoke Runbook](modal_mctx_gpu_smoke_runbook.md)
- [Multiplayer Self-Play And MuZero-Style Search](../../research/multiplayer_selfplay_muzero.md)
- `src/curvyzero/env/core.py`
- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/training/dummy_pong_selfplay_replay.py`
- `src/curvyzero/training/dummy_pong_selfplay_train.py`
- `scripts/benchmark_policy_search_batch_standin.py`

## External Sources Checked

- [MuZero paper, arXiv 1911.08265](https://arxiv.org/abs/1911.08265): MuZero
  plans with a learned model that predicts reward, policy, and value. The real
  environment is still needed to collect training data.
- [Mctx README](https://github.com/google-deepmind/mctx): Mctx is JAX-native,
  supports JIT, and runs search over batches. Its `recurrent_fn` uses learned
  embeddings, not the real simulator state.
- [JAX benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html)
  and [asynchronous dispatch note](https://docs.jax.dev/en/latest/async_dispatch.html):
  compile time, device transfer, and `.block_until_ready()` must be measured
  correctly before trusting GPU/JAX timings.
- [Modal GPU docs](https://modal.com/docs/guide/gpu), [Modal Queues](https://modal.com/docs/guide/queues),
  and [Modal Volumes](https://modal.com/docs/guide/volumes): Modal is useful for
  coarse jobs and stored artifacts. Queues add network latency and are not a
  per-step channel.

## What Has To Be Fast

Fast enough for training means these pieces work together:

| Piece | Why it matters |
| --- | --- |
| Environment stepping | Self-play needs huge numbers of real game transitions. This is the source of replay data. |
| Observation packing | Every live player becomes an ego row for policy/search. Per-agent Python crops or dict copies can erase env speed. |
| Batched policy/search/model calls | MuZero search and neural inference must run on large fixed batches, not one player at a time. |
| Action unpacking | Policy rows must be turned back into wrapper action maps for each real env without losing player order. |
| Replay staging | The hot loop should write chunky array data, not one JSON object per ego step. |
| Reset/autoreset | Finished env rows must be replaced without changing batch shape or losing seed history. |
| Determinism metadata | Rules hash, observation schema, reward schema, action schema, seed/RNG state, and model checkpoint id must follow every replay chunk. |

For CurvyTron specifically, speed cannot break source-visible order: reverse
player update order, trail point insertion before collision, own-body latency,
strict overlap, PrintManager order, death point insertion, and scoring order.

## How Policy Batches Connect To Env Batches

Use fixed env rows first:

```text
env_state[B_env, ...]
players P
alive_mask[B_env, P]
obs[B_env, P, ...]
```

Then build policy/search rows from live ego players:

```text
policy_obs[B_policy, ...]
policy_env_id[B_policy]
policy_player_id[B_policy]
legal_action_mask[B_policy, A]
```

Where `B_policy` is usually close to `B_env * live_players`, padded to a fixed
shape for JAX/Mctx.

After the batched policy/search call:

```text
search_action[B_policy]
search_action_weights[B_policy, A]
root_value[B_policy]
```

Map policy actions back into the wrapper action map:

```text
joint_action[B_env, P]
```

Here `joint_action` is the wrapper/replay sidecar name, not native source
state. Then step the real env batch once. Store replay by ego row, but keep the
env row and player id so targets can be rebuilt later.

Later, Mctx should sit only in the policy/search box. The real env arrays build
observations and masks. The model turns those observations into hidden states.
Mctx searches over learned hidden states with a fixed action count. Its
`recurrent_fn` should not call the CurvyTron simulator. Search outputs actions,
action weights, and root values. Those map back to env rows, then replay chunks
store the ego metadata, action weights, values, model/checkpoint id, and rules
hash beside the real environment rewards.

For v0, search one ego row or both 1v1 players as independent ego rows. Do not
start with full wrapper action-map search. With 3 wrapper actions and 6
players, one wrapper joint decision already has `3^6 = 729` children.

## Clean Boundary Contracts

The simplest architecture is a plain pipeline:

```text
verified source fixtures
  -> fixed array env transition
  -> observation/reward packer
  -> batched policy/search
  -> chunked replay/logging
```

Modal runs whole pipeline shards or benchmark shards around that loop. It is not
inside the loop.

Keep the handoffs small:

| Piece | Input | Output | Rule |
| --- | --- | --- | --- |
| Source-fidelity fixtures | Scenario JSON or promoted batch manifest. | Common trace, ordered events, pass/fail/blocked status, source evidence refs. | This is the behavior authority. It makes no speed claim. |
| Vector env arrays | Verified fixture seed or reset spec, fixed wrapper actions converted to source controls, explicit RNG state. | New fixed state arrays, fixed point/die/score event arrays, done masks, unsupported list. | Compare before timing. Unsupported source fields stay unsupported, not passed. |
| Observation/reward packing | State arrays, event arrays, alive/done masks, schema ids. | `obs[B_env,P,...]`, rewards, legal-action masks, ego-row ids, reward/obs schema hash. | No policy choices and no env mutation here. |
| Policy/search | Ego observation rows, legal masks, checkpoint id, fixed search shape. | Action, action weights, root value, search stats per ego row. | Mctx searches learned hidden states only; it does not call the CurvyTron simulator. |
| Replay/logging | Ego ids, obs/action/reward/done, search outputs, rules/schema/model ids. | Chunked array replay plus compact summaries and mismatch refs. | No one-JSON-row-per-ego-step hot loop. |
| Modal jobs | Sweep or equivalence spec, code ref, artifact run id. | Coarse artifacts with equivalence status, setup/compile/transfer/steady timing, hardware labels. | Whole jobs only: no per-step, per-player, per-node, or hot-loop Queue/Dict calls. |

Short next path from here: keep the broadened `B>1` event preflight green for
the 19 supported one-step fixtures, keep debug observation/reward packing
checked, then replace the debug packer with a trainer-facing schema and add
replay chunks on the same shapes. Keep Mctx as boundary runtime evidence until
the CPU bridge is boring and real CurvyTron rollouts exist. Current trail-gap
caveat: vector support covers the four forced trail-gap fixtures, but not a
broad natural PrintManager cadence or reset/autoreset trail-gap system.

## What Should Run On CPU First

Keep these on CPU first:

- Source-fidelity runners, JS/Python trace comparison, and fixture work.
- Fixture-to-vector-state seeding.
- NumPy fixed-shape env stepping with source-like arrays.
- Collision/body-buffer sweeps over `B`, `P`, and `K`.
- Observation packing and replay staging benchmarks.
- Local NumPy policy/search stand-ins for batch shape, padding, and copy budgets.
- Modal CPU sweeps after local runs are boring.

This is not anti-GPU. It is sequencing. CPU is easier to inspect while the rules
are still being pinned down.

## What Might Move To GPU Later

Move these to GPU when a benchmark proves the need:

- Mctx search over fixed `B_policy`, action count, hidden shape, and simulation
  count.
- Neural representation, prediction, dynamics, and training unrolls.
- Large eval inference batches.
- A JAX/PyTorch tensor env prototype only if CPU env stepping cannot keep the
  model/search/trainer fed.

The GPU justification should be concrete: a real Mctx/JAX or PyTorch model/search
profile is slow enough on CPU, large enough to use the device well, and the CPU
env/observation/replay bridge is no longer the obvious bottleneck.

Do not move these to GPU now:

- Source-fidelity runners.
- Common-trace JSON diffs.
- Public dict wrappers.
- Pure NumPy vector prototype runs.
- Replay JSON writing.
- Modal Queue/Dict calls.

## Local Experiments Run

Machine/runtime labels from the scripts: macOS arm64, Python 3.11.14, NumPy
2.4.0. This checkout has no visible git revision, and the working tree has
changes, so treat the numbers as local scout data.

### Local Dependency Probe

Command:

```sh
python3 - <<'PY'
import importlib.util
for name in ['jax', 'mctx', 'torch', 'numpy']:
    spec = importlib.util.find_spec(name)
    print(f'{name}: {"installed" if spec else "missing"}')
    if spec:
        mod = __import__(name)
        print(f'{name}_version: {getattr(mod, "__version__", "unknown")}')
PY
```

Result:

| Package | Local status |
| --- | --- |
| JAX | missing |
| Mctx | missing |
| PyTorch | installed, `2.9.1` |
| NumPy | installed, `2.4.0` |

Hard fact: local Mctx work cannot run here without adding dependencies. No
dependency was installed for this note. The new local search benchmark is
therefore a CPU/NumPy stand-in, not a fallback implementation.

### Concrete Modal GPU Gate

First GPU dependency proof:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_gpu_dependency_smoke
```

Worker U ran this on 2026-05-09. It passed on Modal app
`ap-k2iRqzGbvLshqsZW8jDVav` with `ok: true`, `jax.default_backend: gpu`,
device `cuda:0`, and `nvidia_smi: NVIDIA L4, 23034 MiB, 580.95.05`. Package
versions were `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6`, and
`numpy==2.4.4`. Timing was `3.364952897s` compile plus first run and
`0.0025687180000000254s` for the second run. Action weights were finite and
row sums were close to `1.0`.

This proves only that pinned JAX/Mctx can run a tiny policy/search job on a
Modal GPU. It is not CurvyTron environment speed, not source-fidelity evidence,
and not a training result. Cost note: the run did allocate a remote Modal L4 GPU
briefly, so normal Modal billing may apply.

Worker Y ran the first measured synthetic Mctx profile on 2026-05-09:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 8 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 4 \
  --warmup-runs 1 \
  --steady-runs 2
```

It passed on Modal app `ap-E6lvbtm5xPQQ21nAnE2HHM` with `ok: true`,
`jax.default_backend: gpu`, device `cuda:0`, and `nvidia_smi` reporting
`NVIDIA L4, 23034 MiB, 17140 MiB, 0 %, 580.95.05`. Package versions were
`jax==0.7.0`, `jaxlib==0.7.0`, and `mctx==0.0.6`. Timing was
`4.855567563s` compile plus first run, `0.0021835409999990674s` warmup, and
steady runs of `0.001847238999999945s` and `0.0017432480000003636s`, with
median `0.0017952435000001543s`. Median throughput was
`4456.220005809414` decisions/sec and `17824.880023237656` simulations/sec.
Action weights were finite and normalized, and `problems` was empty.

The dependency smoke, synthetic sweep, and observation-shaped synthetic run are
captured in the runbook and experiment log. See those notes for pass criteria
and failure triage. Worker V then added and ran the fixture-seeded CPU debug
packer boundary mode on Modal L4: it built `obs[4,2,9]`, filtered 4 live ego
rows into Mctx roots, and kept host setup, device placement, compile/first
search, and steady search timing separate. This is still not real CurvyTron
rollout throughput.

### Current Toy Single-Env Smoke

Command:

```sh
PYTHONPATH=src python3 scripts/benchmark_env.py --episodes 1000 --max-steps 500 --format json
```

Result:

| Metric | Value |
| --- | ---: |
| Env | simplified `curvyzero-v0`, not source fidelity |
| Steps | 23,423 |
| Elapsed | 0.663570s |
| Steps/sec | 35,298.5 |
| Coarse `step` timer | 0.613622s |
| Reset timer | 0.011029s |
| Random action dict timer | 0.034114s |

Hard fact: current toy object-env stepping is around 35k steps/sec in this local
run. It does not split movement, trail writes, collision, observation, and dict
output.

### Source-Fidelity Runner Surface

Command:

```sh
python3 scripts/benchmark_source_fidelity.py --repeat 10 --warmup 2 --format plain
```

Result:

| Metric | Value |
| --- | ---: |
| Scenarios | 20 |
| Measured scenario iterations | 200 |
| Elapsed | 0.031970s |
| Iterations/sec | 6,255.9 |
| Scenario load | 0.009158s |
| Inclusive runner calls | 0.005449s |
| Payload wrapping | 0.003509s |
| Common trace projection | 0.003493s |
| JSON encode | 0.005360s |

Hard fact: this is real source-fidelity runner-surface timing for selected
fixtures. It is not source-internal movement, point insertion, collision, or
PrintManager timing.

### New Bridge Scout

New script:

```sh
python3 scripts/benchmark_selfplay_batch_bridge.py --batch 128 --steps 200 --warmup 20 --replay-mode array --format plain
python3 scripts/benchmark_selfplay_batch_bridge.py --batch 128 --steps 200 --warmup 20 --replay-mode dict --format plain
python3 scripts/benchmark_selfplay_batch_bridge.py --batch 128 --steps 200 --warmup 20 --replay-mode dict_json --format plain
python3 scripts/benchmark_selfplay_batch_bridge.py --batch 512 --steps 200 --warmup 20 --replay-mode array --format plain
```

What it measures: current toy `CurvyTronEnv` objects, fixed env batch outside
the env API, one synthetic NumPy policy/model batch per env tick, action dict
assembly, sequential env stepping, autoreset, and optional replay staging.

What it does not measure: source fidelity, fixture-seeded vector state, real
MuZero/MCTS search, real JAX, real GPU, or production `reset_many`/`step_many`.

| Profile | Elapsed | Total env steps/sec | Env step bucket | Policy batch bucket | Replay bucket |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=128`, array replay | 0.703687s | 36,379.8 | 0.663730s | 0.006610s | 0.000874s |
| `B=128`, dict replay | 0.720601s | 35,525.9 | 0.659910s | 0.006411s | 0.019145s |
| `B=128`, JSON row scaffold | 0.847581s | 30,203.6 | 0.684829s | 0.007225s | 0.120468s |
| `B=512`, array replay | 3.126963s | 32,747.4 | 2.950987s | 0.022987s | 0.002959s |

Hard facts from this local scout:

- The current toy object-env `step` bucket dominates the bridge loop: about 94%
  of the `B=128` array run.
- The synthetic batched policy call is small in this toy setup: 0.006610s for
  51,200 ego rows in the `B=128` array run. This should not be read as a real
  model/search result.
- Array replay staging is cheap in this shape. Python dict rows are visible.
  JSON row encoding is much more visible: 0.120468s for 51,200 small rows.
- Scaling from `B=128` to `B=512` did not improve total throughput because the
  env is still a Python object loop. This is exactly why a real array backend
  matters.

### Policy/Search Batch Stand-In

New script:

```sh
python3 scripts/benchmark_policy_search_batch_standin.py \
  --env-batch 256 --players 2 --obs-dim 64 --hidden-dim 64 \
  --simulations 16 --decision-batches 20 --warmup 3 \
  --copy-mode copy --format plain

python3 scripts/benchmark_policy_search_batch_standin.py \
  --env-batch 512 --players 3 --obs-dim 64 --hidden-dim 128 \
  --simulations 32 --decision-batches 10 --warmup 2 \
  --copy-mode copy --format plain

python3 scripts/benchmark_policy_search_batch_standin.py \
  --env-batch 256 --players 3 --obs-dim 64 --hidden-dim 64 \
  --simulations 16 --decision-batches 10 --warmup 2 \
  --live-fraction 0.75 --copy-mode copy --format plain
```

What it measures: fixed `[B_env, P, obs_dim]` observation arrays, padded
`[B_policy, obs_dim]` ego rows, active-row masks, synthetic representation and
prediction heads, repeated recurrent-model-like matrix work, fake visit-count
targets, action unmapping to `[B_env, P]`, and NumPy host-array copies.

What it does not measure: JAX, Mctx, GPU kernels, device transfer, real MCTS tree
logic, real CurvyTron observations, rewards, env stepping, or training.

| Profile | Elapsed | Env decisions/sec | Policy rows/sec | Recurrent loop | Copy buckets | Hidden tree lower bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_env=256`, `P=2`, rows 512, hidden 64, sims 16 | 0.035542s | 144,054.6 | 288,109.2 | 0.033583s | 0.000086s | 2,228,224 bytes |
| `B_env=512`, `P=3`, rows 1536, hidden 128, sims 32 | 0.133663s | 38,305.3 | 114,915.9 | 0.130137s | 0.000150s | 25,952,256 bytes |
| `B_env=256`, `P=3`, rows 768, live 75%, hidden 64, sims 16 | 0.022509s | 113,734.4 | 341,203.2 | 0.021413s | 0.000041s | 3,342,336 bytes |

Hard facts from this local scout:

- The repeated recurrent/search-shaped work dominates these CPU runs. The
  NumPy copy buckets are tiny here.
- That copy result is not a GPU-transfer result. It only says local host-array
  copies are not the bottleneck in these shapes.
- Fixed padded batches still pay for padded rows unless the search profile
  compacts rows before the compiled call. Compacting to many sizes can cause JAX
  recompiles later.
- Mctx-style hidden tree memory scales as at least
  `B_policy * (simulations + 1) * hidden_dim * 4` bytes before tree statistics.
  This is already about 26 MiB for the larger stand-in profile.

### Synthetic Vector Prototype

Command:

```sh
python3 scripts/benchmark_vectorization_prototype.py --batch 128 --players 3 --body-capacity 512 --steps 200 --warmup 20 --format plain
```

Result:

| Metric | Value |
| --- | ---: |
| Trust level | source-like synthetic timing only |
| Env steps | 25,600 |
| Elapsed | 0.169638s |
| Env steps/sec | 150,909.3 |
| Movement | 0.006921s |
| Normal point mask | 0.003174s |
| Body append | 0.007629s |
| Collision scan | 0.144483s |
| Observation | 0.001301s |

Hard fact: in the current synthetic vector prototype, collision scan dominates.
This is useful for array-shape planning, not for production throughput claims.

### Fixture-Seeded Array Step Compare

New script:

```sh
python3 scripts/compare_vector_arrays_to_fidelity.py \
  scenarios/environment/source_body_canary_batch.json \
  scenarios/environment/source_borderless_wrap_step.json \
  scenarios/environment/source_normal_wall_death_step.json \
  scenarios/environment/source_print_manager_batch.json \
  scenarios/environment/source_trail_gap_hole_space_safe_step.json \
  scenarios/environment/source_trail_gap_stored_body_still_kills_step.json \
  scenarios/environment/source_trail_gap_print_to_hole_boundary_kills_step.json \
  --body-capacity 4 \
  --format plain
```

What it measures: seed JSON arrays from supported fixtures, run one
source-ordered NumPy array tick, project supported fields back to common trace
shape, and compare against the matching Python source runner. It now includes
compact fixed event rows for the existing supported fixtures.

Result:

| Metric | Value |
| --- | ---: |
| Fixtures | 20 |
| Passed | 20 |
| Failed | 0 |
| Unsupported in this run | 0 |
| Compared fields per passing fixture | State fields plus source/common-trace event rows |
| Skipped fields per passing fixture | 0 |

What is real now:

- opponent seeded-body tangent safe and strict-overlap kill checks from
  fixture-seeded arrays
- own-body latency safe and kill checks from fixture-seeded arrays
- same-frame normal point materialization before a lower-index collision check
- strict body overlap, reverse player order, normal/death body insertion,
  visible trail count/last point, body counters, and world body count
- simple source borderless wrap after movement for `source_borderless_wrap_step`
- normal-wall death, death point insertion, and the one-survivor score update
  for `source_normal_wall_death_step`
- PrintManager no-toggle bookkeeping, print-to-hole/hole-to-print/exact-zero
  toggles, narrow delayed-start timer/start rows, and active wall/body
  death-stop side effects
- forced trail-gap hole-space safety, stored-body-in-gap collision,
  print-to-hole boundary collision, and hole-to-print same-update emitted-body
  collision through `source-trail-gap-canary`
- compact event rows with `L=16`, count/mask/overflow arrays, and numeric codes
  for `position`, `point`, `die`, `score:round`, `score`, `round:end`, and
  the supported PrintManager `property` rows

What is still not real:

- broader event details such as `angle`; unsupported event detail should stay
  recorded instead of guessed
- broader reset/timer/autoreset beyond the narrow delayed-start comparator
  fixture, broader natural trail-gap variants, normal-wall same-frame draws,
  broader borderless body-skip and PrintManager-wrap cases, terminal scoring
  beyond the one-survivor wall-death fixture, rewards, observations, real
  replay/autoreset, policy batching, and MCTS/search
- production `B>1` self-play batching with real reset/autoreset, replay, and
  policy/search
- broad speed claims from this equivalence script

Hard fact: the vector lane now has a small checked fixture-seeded transition
set, including narrow delayed-start comparator support. It still does not have
the batched self-play loop.

### Fixture-Seeded Array Step Timing

New script:

```sh
python3 scripts/benchmark_vector_array_steps.py \
  --repeat 10000 \
  --warmup 500 \
  --body-capacity 4 \
  --format plain
```

What it measures: the benchmark script's fixture-seeded one-step default set
after one preflight source/common-trace comparison pass. Check the script
default before quoting the fixture count; this note predates later fixture-set
widening. The repeated hot loop does reset-copy plus the prepared in-place
NumPy transition only. Source runner calls, common-trace projection, and
comparison are counted outside the hot loop.

What it does not measure: production `B>1` self-play stacking, real reset/autoreset,
observation/reward packing, policy/search, replay writes, the newer
PrintManager death-stop and trail-gap comparator fixtures, or broad source
semantics. The timing table below was recorded before
the fixed event-row comparator change; rerun timing before quoting current
step/sec.

Local result on macOS arm64, Python 3.11.13, NumPy 2.4.4:

| Metric | Value |
| --- | ---: |
| Timed supported transitions | 80,000 |
| Setup | 0.001878s |
| Preflight source trace | 0.001287s |
| Preflight env step | 0.001016s |
| Preflight projection | 0.000053s |
| Preflight comparison | 0.000129s |
| Timed reset-copy bucket | 0.610331s |
| Timed env-step bucket | 6.726476s |
| Env steps/sec, step bucket only | 11,893.3 |
| Env steps/sec, reset-copy plus step | 10,903.9 |
| Hot-loop source/projection/comparison calls | 0 |

Hard fact: the old equivalence script no longer doubles as the speed number.
The honest number for the current fixture-backed array transition is about
11.9k B=1 supported transitions/sec for the env-step bucket on this local run.
That is far slower than the synthetic vector prototype and is the point: source
comparison, projection, and setup are now visible as separate costs instead of
being mixed into a vague bridge claim.

### Fixture-Seeded Batch-Row Timing

New script:

```sh
python3 scripts/benchmark_vector_batch_rows.py \
  --batch-sizes 1 8 32 128 \
  --repeat 3000 \
  --warmup 300 \
  --body-capacity 4 \
  --format plain
```

What it measures: the default fixture-seeded vector rows stacked along a real
leading `B` axis, with one batched NumPy step call per repeat. The default input
list is now the supported 19-fixture one-step slice: body canaries, simple
borderless wrap, normal wall death, selected PrintManager
no-toggle/toggle/death-stop rows, and all four forced trail-gap fixtures. It
deliberately does not include the delayed-start PrintManager fixture.

The batched call takes row-specific `source_moves[B,P]`, `step_ms[B]`, and
PrintManager mode rows, keeps the source reverse-player order loop over `P`,
and batches the row math, collision masks, body appends, wall checks,
PrintManager state changes, selected trail-gap state, scoring, and fixed event
rows over `B`. Rows are grouped by fixed array profile, so the one-player
PrintManager toggles, two-player border/wall fixtures, and three-player
body/death-stop/trail-gap fixtures are timed separately.

What it now carries: fixed event rows for the 19 default fixtures, with batch
preflight checking those event arrays against the single-row comparator output
inside the `P1_K4`, `P2_K4`, and `P3_K4` groups. The active PrintManager
death-stop rows are part of that `B>1` event preflight. The batched step now
has a focused pre-step timer path for the delayed-start PrintManager fixture:
after the first delayed-start tick, the second tick fires the timer in batched
rows, updates PrintManager state/body rows, and can run with debug events or
no-event mode. That is a timer-slice test, not default throughput coverage. The
default benchmark still does not measure the full delayed-start reset/timer
trace, broader natural trail-gap variants, trainer-facing observations,
rewards, done/truncated wrappers, reset/autoreset policy, replay writes,
policy/search, MCTS, or GPU execution. The batch rows are made by cycling the
current supported fixture seeds, not by running random production self-play.

Historical local result on macOS arm64, Python 3.11.14, NumPy 2.4.0, before
the default broadening from 8 to 18 fixtures:

The top visible phase is from lightweight timers inside the timed step loop, so
the reported step bucket includes that instrumentation overhead.

| Group | B | Timed rows | Step bucket | Event bucket | Event % | Rows/sec, step only | Rows/sec, reset+step | Top visible phase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `P=2,K=4` | 1 | 3,000 | 0.248414s | 0.027728s | 11.2% | 12,076.6 | 10,774.7 | body collision |
| `P=2,K=4` | 8 | 24,000 | 0.482273s | 0.149120s | 30.9% | 49,764.3 | 46,785.4 | event emit |
| `P=2,K=4` | 32 | 96,000 | 0.934707s | 0.492429s | 52.7% | 102,706.0 | 99,108.5 | event emit |
| `P=2,K=4` | 128 | 384,000 | 2.793376s | 1.913490s | 68.5% | 137,468.1 | 135,296.0 | event emit |
| `P=3,K=4` | 1 | 3,000 | 0.327110s | 0.035347s | 10.8% | 9,171.2 | 8,386.1 | body collision |
| `P=3,K=4` | 8 | 24,000 | 0.562106s | 0.148985s | 26.5% | 42,696.6 | 40,408.2 | event emit |
| `P=3,K=4` | 32 | 96,000 | 0.872724s | 0.445509s | 51.0% | 110,000.4 | 105,895.3 | event emit |
| `P=3,K=4` | 128 | 384,000 | 2.132799s | 1.614802s | 75.7% | 180,045.1 | 176,548.9 | event emit |

Do not quote the table above as current 28-fixture throughput; rerun the command
after broadening before using rows/sec numbers.

Default broadening smoke on 2026-05-09:

```sh
python3 scripts/benchmark_vector_batch_rows.py \
  --batch-sizes 1 \
  --repeat 1 \
  --warmup 0 \
  --body-capacity 4 \
  --format plain
```

Result: the widened one-step fixture slice and batch preflight were green. The
default excludes delayed-start because the comparator's delayed-start support
is a full two-tick reset/timer trace, while the speed default remains a
one-step timing slice. Treat this as correctness gating for the timing slice,
not a speed headline and not production self-play throughput.

SPEED2 larger local check on 2026-05-09:

```sh
python3 scripts/benchmark_vector_batch_rows.py \
  --batch-sizes 32 128 \
  --repeat 250 \
  --warmup 25 \
  --body-capacity 4 \
  --event-modes debug-event no-event \
  --format plain
```

Result: debug-event and no-event preflight matched for `P1_K4`, `P2_K4`, and
`P3_K4`. The `B=128` run was not slow locally, so no fallback-only `B=32`
report was needed.

| Group | B | Debug rows/sec, step | No-event rows/sec, step | No-event speedup | Debug event % | Top debug phase | Top no-event phase |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `P1_K4` | 32 | 269,467.7 | 335,944.6 | 1.247x | 10.0% | PrintManager update | PrintManager update |
| `P1_K4` | 128 | 818,128.6 | 1,196,618.9 | 1.463x | 11.3% | PrintManager update | PrintManager update |
| `P2_K4` | 32 | 119,146.5 | 204,624.1 | 1.717x | 40.2% | event emit | terminal score state |
| `P2_K4` | 128 | 224,279.7 | 487,509.2 | 2.174x | 49.4% | event emit | terminal score state |
| `P3_K4` | 32 | 78,311.8 | 129,174.9 | 1.649x | 24.4% | event emit | PrintManager death stop |
| `P3_K4` | 128 | 251,415.0 | 445,206.8 | 1.771x | 32.1% | event emit | PrintManager death stop |

Plain bottleneck notes:

- In debug-event mode, event-row emission is now the clearest local bottleneck
  for the multi-player groups, especially `P2_K4/B=128` at 49.4% of the step
  bucket and `P3_K4/B=128` at 32.1%.
- With event rows disabled, the next visible costs are not one-size-fits-all:
  `P2_K4` spends most visible time in terminal score state work, while `P3_K4`
  points at PrintManager death-stop work.
- Reset-copy is small in this shape. The next practical optimization target is
  the debug event representation and the per-group transition hot spots, not
  reset copying.
- This is still CPU/NumPy fixture-row timing, not GPU timing and not production
  self-play.

Hard fact: the vector lane now has a real B>1 batch-row timing prototype that
carries fixed event rows for a widened default slice. This is useful as a
shape/timing signal, but it is not a useful-games/minute claim until broad
reset/timer/autoreset, trainer-facing observations/rewards, replay,
policy/MCTS connection, and action-latency reporting exist.

## What We Need To Measure Next

First measure useful output, not only raw steps/sec:

- single-env step p50/p95/p99
- batch steps/sec
- completed games/min
- actor-loop p95 action latency
- reset/terminal overhead
- policy/search timing
- host/device transfer time when a GPU boundary exists

Near-term order:

1. Optimized Python/NumPy sync actor loop with honest actor-loop metrics.
2. CPU env plus GPU model/MCTS boundary, including host/device transfers.
3. CPU actor pool after the sync baseline is clear.

Then keep these domain items moving:

1. Broader timer/autoreset and broader trail-gap state/events:
   the current comparator covers fixed rows for the supported PrintManager
   toggles/death-stops, the narrow delayed-start fixture, and the four forced
   trail-gap canaries. The speed defaults intentionally exclude delayed-start
   today, though the batch-row tests now cover the delayed-start second-tick
   pre-step timer fire in debug-event and no-event modes. Next event work should
   broaden timer/reset/autoreset semantics or add broader natural gap rows only
   when a source fixture emits them.
2. Trainer-facing observation/reward:
   the debug packer exists. Replace it with a real schema only after source
   state/events settle.
3. Replay/autoreset:
   the actor-loop bridge stages fixed chunks over short fixture-reset rollout
   blocks, but it still uses debug observations and synthetic policy/search.
   Add real row reset/autoreset before broad claims.
4. Mctx synthetic search:
   the Modal L4 synthetic, observation-shaped debug, and host-built synthetic
   transfer runs exist. The fixture-seeded debug packer boundary also exists.
   Next, connect a real fixed-shape actor loop before claiming any real
   CurvyTron rollout path.
5. End-to-end actor loop:
   keep reporting completed games/min, action latency, env rows/sec, and ego
   decisions/sec while replacing synthetic policy/search and debug packing one
   piece at a time.

## 2026-05-09 SPEED Update: Actor-Loop Cost Report

Worker SPEED added a clearer fixed-shape cost report to
`scripts/benchmark_vector_actor_loop_bridge.py`.

Each timed batch now includes `fixed_shape_cost_report` with:

- fixed shape metadata: `B`, `P`, event mode, rollout blocks, env-step calls,
  env rows, ego rows, and active ego rows
- per-bucket seconds, percent of loop, microseconds per env row, microseconds
  per ego row, microseconds per env-step call, and row/sec rates
- a `policy_total` aggregate for synthetic root/search/action-select
- event cost fields: event overhead seconds, percent of env-step bucket,
  microseconds per env row, emitted event count, overflow attempts, and events
  per env row

The plain output now prints a `fixed_shape_cost=...` line for every batch. The
`event_compare=...` line also reports debug/no-event deltas for the debug
packer, synthetic policy/search, and replay staging buckets, not just env-step
time.

Smoke command run:

```sh
python3 scripts/benchmark_vector_actor_loop_bridge.py \
  --batch-sizes 4 \
  --repeat 5 \
  --warmup 1 \
  --rollout-steps 2 \
  --event-modes debug-event no-event \
  --format plain
```

Historical smoke result before the default broadening: for `P=2,K=4,B=4`, the
debug-event run reported env-step as 56.8% of loop, debug pack as 10.8%,
synthetic policy/search as 23.0%, replay as 2.8%, and event emission as 27.8%
of the env-step bucket. For `P=3,K=4,B=4`, the corresponding values were
60.0%, 9.6%, 22.2%, 2.4%, and 26.7%. No-event total-loop speedup was 1.275x
for `P=2` and 1.251x for `P=3`.

Default broadening smoke on 2026-05-09:

```sh
python3 scripts/benchmark_vector_actor_loop_bridge.py \
  --batch-sizes 1 \
  --repeat 1 \
  --warmup 0 \
  --rollout-steps 1 \
  --body-capacity 4 \
  --format plain
```

Result: the actor-loop bridge preflight was green for the widened default
fixture slice. This confirms that the actor-loop bridge now starts from the
same default slice as the batch-row and debug packing benchmarks; it is still
not broad throughput evidence because the loop uses debug observations,
synthetic policy/search, and in-memory replay staging.

SPEED2 small actor-loop check on 2026-05-09:

```sh
python3 scripts/benchmark_vector_actor_loop_bridge.py \
  --batch-sizes 32 128 \
  --repeat 100 \
  --warmup 10 \
  --rollout-steps 2 \
  --body-capacity 4 \
  --event-modes debug-event no-event \
  --format plain
```

Result: the preflight was green and the run was still practical at `B=128`, so
no fallback-only actor report was needed. Step 0 uses fixture source moves; step
1 feeds synthetic selected actions back into the vector path and is not
source-compared.

| Group | B | Debug env rows/sec | No-event env rows/sec | No-event loop speedup | Debug env-step % | No-event policy % | Debug event % of env-step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `P1_K4` | 32 | 138,711.4 | 160,799.6 | 1.159x | 46.5% | 39.5% | 14.2% |
| `P1_K4` | 128 | 381,991.4 | 436,389.7 | 1.142x | 40.8% | 51.7% | 15.6% |
| `P2_K4` | 32 | 93,286.5 | 119,242.9 | 1.278x | 57.0% | 36.9% | 33.3% |
| `P2_K4` | 128 | 189,900.2 | 259,916.2 | 1.369x | 54.6% | 50.4% | 43.4% |
| `P3_K4` | 32 | 62,230.8 | 82,215.0 | 1.321x | 65.0% | 32.9% | 25.5% |
| `P3_K4` | 128 | 165,573.8 | 217,848.2 | 1.316x | 55.0% | 47.4% | 31.1% |

Plain bottleneck notes:

- In debug-event mode, the env-step bucket remains the largest bucket for
  `P2_K4` and `P3_K4`; event emission is a large part of that bucket.
- In no-event mode, the synthetic policy/search stand-in becomes the largest
  bucket for `P1_K4/B=128` and `P2_K4/B=128`, and nearly catches the env step
  for `P3_K4/B=128`. This is only shape pressure, not MCTS, Mctx, learned-model,
  or GPU evidence.
- The debug obs/reward packer is visible but not dominant in this run, about
  5% to 8% of the `B=128` loop depending on group and event mode.
- The in-memory replay ring is small here, about 1% to 2% at `B=128`. That says
  fixed array staging is cheap in this scout, not that production replay writing
  is solved.
- `P1_K4` had zero active ego rows in this actor-loop check, so `P2_K4` and
  `P3_K4` are the better signal for ego-policy bridge pressure.

This smoke is intentionally tiny. It proves the reporting path and fixed shapes,
not production throughput. The real evidence remains the fixture-seeded NumPy
batch step, debug obs/reward packer, synthetic CPU policy/search stand-in, and
in-memory replay staging.

Optional heavier repeat-count follow-up for this lane:

```sh
python3 scripts/benchmark_vector_actor_loop_bridge.py \
  --batch-sizes 32 128 \
  --repeat 500 \
  --warmup 50 \
  --rollout-steps 2 \
  --event-modes debug-event no-event \
  --format plain
```

The local `repeat=100` run already says event-row emission is still a real
debug-event cost in the two-step actor bridge, while no-event mode exposes the
synthetic policy/search stand-in as the next large bucket. Before making bigger
claims, replace the debug event/obs/reward path with trainer-facing schemas and
add real reset/autoreset and replay writing semantics.

## 2026-05-09 ROLLOUT Update: Sample Contract Metadata

Worker ROLLOUT added deterministic contract-shaped metadata to
`scripts/benchmark_vector_actor_loop_bridge.py --sample-only`.

The sample output now includes `sample_contract_metadata` with:

- replay schema id/hash for the current debug in-memory chunk shape
- observation, action-space, and reward schema ids/hashes
- selected fixture `ruleset_id` plus a fixture-scoped `rules_hash`
- env implementation id/version and producer id
- explicit caveats for the missing production replay pieces

The `rules_hash` is intentionally narrow: it hashes the selected fixture ids,
ruleset ids, body capacity, step index, and event mode. It is useful as a local
sample compatibility guard, but it is not a full source rules hash and should
not be treated as learner/replay compatibility evidence.

Smoke command:

```sh
python3 scripts/benchmark_vector_actor_loop_bridge.py \
  --sample-only \
  --batch-sizes 2 \
  --rollout-steps 2 \
  --hidden-dim 4 \
  --simulations 1 \
  --format json
```

Result: the sample printed `sample_contract_metadata` with
`curvyzero_debug_actor_loop_replay_chunk/v0`,
`curvyzero_debug_global_player_obs/v0`,
`curvyzero_source_move_action_space/v0`, and
`curvyzero_debug_score_round_delta_death_penalty/v0`. Treat it as historical
sample-only evidence, not a speed headline.

## 2026-05-09 LATENCY Update: Actor-Loop Rates And Amdahl Buckets

The actor-loop benchmark now reports three extra plain-output lines for each
batch:

- `training_rate=...`: staged ego rows/sec, final transition env rows/min, and
  a clearly labeled completed-game-row proxy.
- `latency=...`: p50/p95/p99 actor-step latency plus env-step, synthetic
  policy/search, and replay staging latency.
- `amdahl=...`: percent of this timed loop spent in env step, debug pack,
  synthetic policy/search, replay staging, and non-env work.

Small smoke command:

```sh
uv run python scripts/benchmark_vector_actor_loop_bridge.py \
  --batch-sizes 2 \
  --repeat 1 \
  --warmup 0 \
  --rollout-steps 2 \
  --body-capacity 4 \
  --hidden-dim 4 \
  --simulations 1 \
  --chunk-steps 4 \
  --format plain
```

Result: the smoke completed with green preflight.

Read this smoke carefully. `completed_game_rows_per_min_proxy` counts final
transition env rows before debug internal autoreset. It is not production
self-play games/min. `synthetic_policy_pct_loop` is NumPy shape work, not real
MCTS or GPU model inference. The Amdahl bucket is still useful because it keeps
the question honest: if real policy/search later dominates, env-only speedups
will not move total training time much.

## 2026-05-09 Practical Parallelism Update

Runtime for the local commands below:

```sh
python3 - <<'PY'
import platform, numpy as np, sys
print(platform.python_version())
print(sys.executable)
print(np.__version__)
print(platform.platform())
print(platform.machine())
PY
```

Result: Python `3.11.14`, executable
`/opt/homebrew/opt/python@3.11/bin/python3.11`, NumPy `2.4.0`,
`macOS-15.6-arm64-arm-64bit`, machine `arm64`.

Latest smoke matrix:

```sh
uv run python scripts/run_environment_fidelity_matrix.py --run smoke --format plain
```

Result: correctness smokes were green. These checks guard the timing slice; they
are not the speed result.

Tiny smoke timing from that same actor-loop quick command, debug-event only:

| Group | Env rows/sec | Ego rows/sec | Actor step p50 | Env-step % loop | Event % env-step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `P1_K4`, `B=2` | 4,593.7 | 4,593.7 | 0.378 ms | 41.2% | 12.6% |
| `P2_K4`, `B=2` | 4,372.4 | 8,744.8 | 0.410 ms | 45.0% | 30.4% |
| `P3_K4`, `B=2` | 5,413.3 | 16,240.0 | 0.324 ms | 58.4% | 26.9% |

This is local debug timing only: `repeat=1`, `B=2`, debug observations,
synthetic policy/search, and in-memory replay staging. It is useful for smoke
regression and latency wiring, not for training throughput claims.

User-provided local no-event actor-loop timing:

```sh
uv run python scripts/benchmark_vector_actor_loop_bridge.py \
  --batch-sizes 2 32 128 \
  --repeat 20 \
  --warmup 2 \
  --rollout-steps 4 \
  --hidden-dim 16 \
  --simulations 2 \
  --chunk-steps 16 \
  --event-modes no-event \
  --format plain
```

Useful `B=128` numbers from that run:

| Group | Env rows/sec | Ego rows/sec | Actor step p50 | Env-step % loop | Synthetic policy/search % loop |
| --- | ---: | ---: | ---: | ---: | ---: |
| `P1_K4`, no-event | 378,000 | 378,000 | 0.333 ms | 33% | 8% |
| `P2_K4`, no-event | 227,000 | 453,000 | 0.561 ms | 47% | 13% |
| `P3_K4`, no-event | 267,000 | 801,000 | 0.461 ms | 47% | 25% |

This is the best current local debug signal for larger `B` actor-loop shape,
but it is still not training throughput: no debug events, debug observations,
synthetic NumPy policy/search, no real MCTS, no real model, no learner handoff,
and no production replay writer.

New practical parallelism script:

```sh
PYTHONPATH=src python3 scripts/benchmark_selfplay_parallel_bridge.py \
  --batch 128 \
  --steps 100 \
  --warmup 10 \
  --workers 2 \
  --modes serial serial-sharded thread process \
  --format plain
```

What it measures: current toy `CurvyTronEnv` object rows, synthetic NumPy policy
batch per shard tick, action dict assembly, sequential env stepping inside each
shard, autoreset, and preallocated array replay staging. It compares coarse
local actor sharding. It does not measure source fidelity, fixture vector rows,
MCTS, JAX, GPU, central inference, or per-step IPC.

Two-worker local result:

| Mode | Wall env steps/sec | Steady env steps/sec estimate | Speedup vs serial | Actor p50 | Actor p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `serial` | 23,086.0 | 26,109.3 | 1.000x | 4.813 ms | 5.528 ms |
| `serial-sharded` | 23,897.6 | 26,505.0 | 1.015x | 4.803 ms | 5.080 ms |
| `thread` | 20,610.9 | 23,009.0 | 0.881x | 5.707 ms | 8.607 ms |
| `process` | 32,402.0 | 50,859.6 | 1.948x | 2.490 ms | 2.890 ms |

Four-worker local result:

```sh
PYTHONPATH=src python3 scripts/benchmark_selfplay_parallel_bridge.py \
  --batch 128 \
  --steps 100 \
  --warmup 10 \
  --workers 4 \
  --modes serial serial-sharded thread process \
  --format plain
```

| Mode | Wall env steps/sec | Steady env steps/sec estimate | Speedup vs serial | Actor p50 | Actor p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `serial` | 21,896.9 | 24,430.3 | 1.000x | 4.839 ms | 5.164 ms |
| `serial-sharded` | 16,598.7 | 17,852.3 | 0.731x | 5.280 ms | 11.632 ms |
| `thread` | 19,098.8 | 21,133.4 | 0.865x | 9.946 ms | 18.320 ms |
| `process` | 41,151.7 | 88,323.5 | 3.615x | 1.485 ms | 3.108 ms |

Plain read:

- Threads did not help this Python object-env loop. They made latency worse in
  the 2-worker and 4-worker checks.
- Coarse process shards helped when there was enough work to amortize process
  startup. The script has no per-step IPC; it is measuring independent actor
  shards, not a central policy server.
- Serial sharding by itself is not an optimization. The 4-shard serial run was
  slower than one serial batch because it split one useful batch into smaller
  loops.
- The toy object-env bucket is still about 94% to 95% env step time in serial
  and process modes. That says process sharding can hide Python object-env
  cost, but it does not make the object env the right final backend.
- For real MuZero, MCTS/model inference can easily dominate once
  `num_simulations` is real. If search takes most of the loop, env process
  sharding helps only until the policy/search queue becomes the bottleneck.

Updated recommendation:

1. Keep the fixture-seeded NumPy batch path as the fidelity/speed baseline.
2. Use process sharding only as coarse actor parallelism after reset/autoreset,
   replay chunks, and action-latency reporting are real. Do not add per-step
   `Queue`, `Dict`, web, or Modal calls in the hot loop.
3. Use Modal CPU for coarse shard sweeps and Modal GPU for JAX/Mctx model/search
   smokes. Do not move the env to GPU unless profiling shows CPU env rows are
   starving a stable real search/training batch.
4. Treat PyTorch/JAX env rewrites as later options. The next higher-value work
   is a real policy/search timing boundary, because MCTS may dominate before
   env stepping does.

## Mistakes That Would Waste Time

- Treating the current synthetic vector prototype as production evidence.
- Optimizing toy-v0 internals before the source fixture bridge exists.
- Adding `reset_many` or `step_many` as a public API before semantics are stable.
- Moving source runners, JSON diffs, or public wrappers to GPU.
- Calling Modal Functions, Queues, Dicts, or web endpoints per env step, per
  action, per model batch, or per MCTS node.
- Writing one JSON file or one JSON object from the hot loop for every ego step.
- Letting JAX recompile because batch size, action count, player count, or
  hidden shape changes constantly.
- Timing JAX/GPU without separating compile time, data transfer, and steady-state
  execution.
- Treating the NumPy policy/search stand-in as Mctx, MCTS, or GPU evidence.
- Searching full wrapper action maps first. Start ego-row search and
  policy-only opponents.
- Hiding RNG. Future natural gaps, spawn variation, and bonuses need explicit
  per-env RNG state.

## Practical Next Step

Do the next CPU bridge before any GPU env work:

```text
verified source fixture batch
  -> fixed vector seed arrays
  -> one or more supported array ticks
  -> common-trace comparison with unsupported cases split out
  -> separate timing run for only the supported transition
  -> fixed event arrays for current position/point/die/score/round-end traces
  -> PrintManager and trail-gap array semantics
  -> add real observation packing, replay, and autoreset timers
  -> report completed games/min and p50/p95/p99 action latency
  -> run the NumPy policy/search stand-in on the same batch shapes
  -> replace the stand-in with real JAX/Mctx search when dependencies are present
```

Only after that should the lane run broad CPU sweeps or replace the synthetic
policy batch with a real Mctx search benchmark. Treat GPU env stepping as a
separate profiling question, not the default next step.

## Worker M Actor-Loop Stand-In Knob

Existing script support is enough for a cheap latency-vs-throughput scout before
real MCTS is wired in. Use the actor bridge's `--simulations` option as a
calibrated stand-in knob for heavier local NumPy recurrent/search-shaped work:

```sh
uv run python scripts/benchmark_vector_actor_loop_bridge.py \
  --batch-sizes 8 32 \
  --repeat 5 \
  --warmup 1 \
  --rollout-steps 2 \
  --body-capacity 4 \
  --hidden-dim 16 \
  --simulations 64 \
  --chunk-steps 8 \
  --event-modes no-event \
  --format plain
```

This is still not MuZero/MCTS, not Mctx, not a learned model, not GPU timing,
and not device-transfer timing. It is useful because it puts the heavier
stand-in inside the same fixture-seeded actor-loop buckets and latency report.

Worker M's local check on 2026-05-09 completed with green preflight. At `B=32`,
`hidden_dim=16`, `rollout_steps=2`, and `no-event`, moving from
`simulations=1` to `simulations=64` shifted P2/P3 from env-step-led to
policy/search-led. P2 moved from `env_step=45.7%`, synthetic policy/search
`9.0%`, actor p95 `0.412ms` to synthetic policy/search `71.3%`, actor p95
`1.491ms`. P3 moved from `env_step=64.5%`, synthetic policy/search `11.4%`,
actor p95 `0.435ms` to synthetic policy/search `76.8%`, actor p95 `2.026ms`.

Practical read: this command can expose the shape of the boundary, but the
implementation decision still needs real model/MCTS timing on the same ego-row
shapes. If real search lands in the heavy-stand-in regime, env-only optimization
has a smaller Amdahl payoff and batching policy must be chosen against p95/p99
action latency, not just staged ego rows/sec.
