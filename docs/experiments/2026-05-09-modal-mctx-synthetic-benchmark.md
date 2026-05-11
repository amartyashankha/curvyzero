# 2026-05-09 Modal Mctx Synthetic Benchmark

## Question

Can Modal run small and moderate GPU Mctx search benchmarks, not just an
import smoke?

## Commands

Tiny profile:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 8 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 4 \
  --warmup-runs 1 \
  --steady-runs 2
```

Larger profile:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 64 \
  --num-simulations 16 \
  --hidden-dim 64 \
  --max-depth 16 \
  --warmup-runs 2 \
  --steady-runs 5
```

Small batch/simulation sweep:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 8 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 8 \
  --warmup-runs 1 \
  --steady-runs 3
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 16 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 8 \
  --warmup-runs 1 \
  --steady-runs 3
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 8 \
  --num-simulations 8 \
  --hidden-dim 32 \
  --max-depth 8 \
  --warmup-runs 1 \
  --steady-runs 3
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 16 \
  --num-simulations 8 \
  --hidden-dim 32 \
  --max-depth 8 \
  --warmup-runs 1 \
  --steady-runs 3
```

CurvyTron-shaped debug observation profile:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --observation-mode curvytron_debug \
  --batch-size 4 \
  --player-count 2 \
  --obs-dim 9 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 4 \
  --warmup-runs 1 \
  --steady-runs 2
```

## Results

The flat profiles, small sweep, and observation-shaped debug profile passed on
Modal L4.

### Tiny Profile

- `ok: true`
- Latest Modal app: `ap-E6lvbtm5xPQQ21nAnE2HHM`
- JAX backend: `gpu`
- device: `cuda:0`
- GPU: L4
- `nvidia_smi`: `NVIDIA L4, 23034 MiB, 17140 MiB, 0 %, 580.95.05`
- packages: `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6`
- latest compile plus first run: `4.855567563s`
- latest warmup run: `0.0021835409999990674s`
- latest steady runs: `0.001847238999999945s`, `0.0017432480000003636s`
- latest steady median: `0.0017952435000001543s`
- latest decisions/sec median: `4456.220005809414`
- latest simulations/sec median: `17824.880023237656`
- action weights finite and normalized.

### Larger Profile

- `ok: true`
- Modal app: `ap-ULhQNpnV6a1lsn0uQLUbnX`
- JAX backend: `gpu`
- device: `cuda:0`
- GPU: L4
- compile plus first run: `8.080801095000002s`
- steady median: `0.005292786999998356s`
- decisions/sec median: `12091.928127850202`
- simulations/sec median: `193470.85004560323`
- action weights finite and normalized.

### Small Batch/Simulation Sweep

Question: what happens as batch size and simulation count move a little while
the synthetic model shape stays fixed?

All four profiles passed on Modal L4 with JAX backend `gpu`, device `cuda:0`,
packages `jax==0.7.0`, `jaxlib==0.7.0`, and `mctx==0.0.6`. The benchmark
separates compile plus first run from one warmup and three steady-state runs.
It does not report host/device transfer time; the synthetic params and
observations are built inside the remote JAX process.

| Batch | Simulations | Modal app | Compile + first run | Warmup | Steady median | Steady min/max | Decisions/sec median | Simulations/sec median |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | `ap-qnWS8Z1wdpQhxyNA5f95Uw` | `4.810899s` | `0.002078s` | `0.001872s` | `0.001807s` / `0.001911s` | `4273.17` | `17092.69` |
| 16 | 4 | `ap-TU5UI2zI2qTrBVBoloU8ba` | `3.732479s` | `0.001861s` | `0.001817s` | `0.001599s` / `0.001943s` | `8804.71` | `35218.82` |
| 8 | 8 | `ap-bJh7OgzkGdJvShVRl1SEhq` | `4.321889s` | `0.002404s` | `0.002248s` | `0.002151s` / `0.002354s` | `3559.40` | `28475.24` |
| 16 | 8 | `ap-dgbjDvRlGXr4LroUQOp5HO` | `4.123886s` | `0.002736s` | `0.003589s` | `0.003150s` / `0.004467s` | `4458.40` | `35667.22` |

Tiny readout:

- At `num_simulations=4`, moving from `B=8` to `B=16` roughly doubled
  decision throughput while steady wall time stayed near the same small-kernel
  floor.
- At `num_simulations=8`, moving from `B=8` to `B=16` improved median
  simulations/sec, but wall time rose and the three steady samples were noisier.
- Moving from `4` to `8` simulations increased measured work per decision.
  Decision/sec dropped, while simulations/sec improved for `B=8` and stayed
  near `35k` for `B=16`.
- This is a small smoke table, not a scaling law. The goal was to answer the
  next question without spending much GPU time.

## Interpretation

This proves the small benchmark path works repeatedly and that the next fixed
shape `B=64`, `num_simulations=16`, `hidden_dim=64`, `max_depth=16` has useful
steady-state throughput on Modal L4. The small sweep adds one more concrete
fact: tiny Mctx jobs are overhead-sensitive, but more batch/search work starts
to use the GPU better.

The flat profiles are still synthetic: no real Pong, no CurvyTron env, no
replay, no trainer, no learned checkpoint, no real observation tensor, no real
recurrent dynamics, and no CPU-vs-GPU comparison. They do not prove CurvyTron
self-play throughput. The `curvytron_debug` profile below adds the current
debug observation shape, but the tensor values and dynamics are still synthetic.

### CurvyTron-Shaped Debug Obs Profile

Question: can the same Modal/JAX/Mctx path consume a root batch made from a
synthetic tensor shaped like the current debug packer, `obs[B,P,9]`, instead of
the old flat `obs[B,hidden_dim]` stand-in?

Result: passed on Modal L4.

- `ok: true`
- Modal app: `ap-nBRyeqrFSrjHIYgehvLDZP`
- JAX backend/device: `gpu`, `cuda:0`
- GPU: `NVIDIA L4, 23034 MiB, 17140 MiB, 0 %, 580.95.05`
- packages: `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6`
- source tensor shape: `[4, 2, 9]`
- root/search obs shape after ego flattening: `[8, 9]`
- ego/mask shapes carried in metadata: `ego_mask [4,2]`,
  `legal_action_mask [4,2,3]`, `ego_row_id [4,2]`,
  `ego_env_id [4,2]`, `ego_player_id [4,2]`
- compile plus first run: `4.113220473s`
- warmup run: `0.0018798989999986304s`
- steady runs: `0.0014564580000016036s`,
  `0.0014556279999986543s`
- steady median: `0.001456043000000129s`
- median throughput labels: `2747.1716151237606` env rows/sec,
  `5494.343230247521` ego decisions/sec,
  `21977.372920990085` simulations/sec
- action weights finite and normalized; row sums ranged from
  `0.9999998807907104` to `1.0000001192092896`

Interpretation: this is the next small shape bridge. It verifies that the Modal
Mctx benchmark can start from a synthetic `obs[B,P,9]` debug-observation-shaped
tensor and flatten ego rows into Mctx roots. It is still not CurvyTron rollout
throughput, not source fidelity, not real replay, not reward learning, and not a
trainer. The values inside `obs[B,P,9]` are synthetic, although the feature names
match the debug packer.

### CurvyTron-Shaped Boundary Timing Profile

Question: can the debug-shaped synthetic benchmark report the setup and transfer
boundary separately from device-resident Mctx search timing, while keeping the
same tiny shape?

Command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --observation-mode curvytron_debug \
  --obs-dim 9 \
  --batch-size 4 \
  --player-count 2 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 4 \
  --warmup-runs 1 \
  --steady-runs 2
```

Result: passed on Modal L4.

- `ok: true`
- Modal app: `ap-cX21WujgqSjGAwMECD41cB`
- JAX backend/device: `gpu`, `cuda:0`
- GPU: `NVIDIA L4, 23034 MiB, 17140 MiB, 0 %, 580.95.05`
- packages: `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6`
- build path: `host_numpy_then_jax_device_put`
- source tensor shape: `[4, 2, 9]`
- root/search obs shape after ego flattening: `[8, 9]`
- counts: `4` env rows, `2` players/env, `8` candidate ego rows,
  `8` live ego rows, `8` Mctx search roots, `3` actions
- shape assertions: `passed`
- host observation setup: `0.0003469869999999098s`
- host-to-device placement: `0.21319387700000014s`
- compile plus first search run: `5.0306235169999995s`
- warmup run: `0.002236305999998578s`
- steady runs: `0.0026488490000016185s`,
  `0.0022923420000005024s`
- steady median: `0.0024705955000010604s`
- median throughput labels: `1619.0428582899478` env rows/sec,
  `3238.0857165798957` ego decisions/sec,
  `12952.342866319583` simulations/sec
- action weights finite and normalized; row sums ranged from
  `0.9999998807907104` to `1.0000001192092896`

Interpretation: this adds a useful boundary measurement, not a bigger search.
The observation tensor and action mask are now built as host NumPy arrays,
asserted against the debug shape, placed on the JAX device, and then reused for
the timed search calls. The `host_to_device_transfer_sec` value is a first
placement measurement in one fresh Modal process for a tiny synthetic tensor, so
treat it as a boundary smoke, not a scaling number.

This still does not use the real CurvyTron environment, real debug packer
output, real rewards, replay, a checkpoint, a trainer, or source-fidelity
fixtures. It only proves the synthetic benchmark can report the shape/count and
setup/placement/search timing boundaries for `obs[B,P,9]`.

### Fixture-Seeded CPU Debug Packer Boundary Profile

Question: can the Modal/JAX/Mctx boundary consume real CPU debug-packer output
from fixture-seeded vector arrays, instead of synthetic `obs[B,P,9]` values?

Command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --observation-mode curvytron_debug_packer \
  --batch-size 4 \
  --player-count 2 \
  --obs-dim 9 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 4 \
  --warmup-runs 0 \
  --steady-runs 1
```

Result: passed on Modal L4.

- `ok: true`
- Modal app: `ap-eBPR9uUVfJhXB7nbaItivH`
- JAX backend/device: `gpu`, `cuda:0`
- GPU: `NVIDIA L4, 23034 MiB, 17140 MiB, 0 %, 580.95.05`
- packages: `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6`
- packer source: `fixture_seeded_cpu_debug_packer`
- selected fixture group: `P2_K4`
- source tensor shape: `[4, 2, 9]`
- candidate ego rows: `8`
- live/search root rows after `ego_mask` filtering: `4`
- root/search obs shape: `[4, 9]`
- reward shape: `[4, 2]`
- reward die source: `event_rows`
- source preflight summary: `8` pass, `0` fail, `0` unsupported
- host observation setup: `0.02176979999999995s`
- host-to-device placement: `0.08496403299999988s`
- compile plus first search run: `3.9509587949999996s`
- steady run: `0.002627230999999952s`
- median throughput labels: `1522.515530610012` env rows/sec,
  `1522.515530610012` live ego decisions/sec,
  `6090.062122440048` simulations/sec
- action weights finite and normalized.

Interpretation: this is one step less fake. The observation and legal-action
mask now come from the existing fixture-seeded CPU debug packer, including done
rows, reward arrays, and die-event reward evidence. The benchmark filters live
ego rows before calling Mctx, so done/dead slots are not searched as roots.

This is still not a real environment rollout, not production self-play
throughput, not learned dynamics, not replay, not a trainer, and not a final
training observation/reward contract. The fixtures are still cycled inside a
fixed `P/K` group.

### Actor-Bridge Sample Boundary Mode

Question: can the Modal/JAX/Mctx boundary consume the same fixed-shape arrays
that the local actor bridge exposes after real vector env steps?

Status: code path added, local helper smoke passed, and one tiny remote Modal
L4 boundary run passed.

Local helper smoke:

```sh
python3 scripts/benchmark_vector_actor_loop_bridge.py \
  --sample-only \
  --batch-sizes 4 \
  --event-modes debug-event \
  --rollout-steps 2 \
  --hidden-dim 32 \
  --simulations 4 \
  --body-capacity 4 \
  --player-count 2 \
  --format json
```

Local result:

- Source preflight: `8` pass, `0` fail, `0` unsupported.
- Selected fixture group: `P2_K4`.
- Actor sample source: `fixture_seeded_cpu_actor_loop_bridge`.
- Final step source kind: `synthetic_feedback_moves`.
- Source tensor shape: `[4, 2, 9]`.
- Reward shape: `[4, 2]`.
- Done/truncated shapes: `[4]`, `[4]`.
- Ego id/mask shapes: `[4, 2]`.
- Legal action mask shape: `[4, 2, 3]`.
- Live ego rows: `4`.
- Done rows: `2`.
- Truncated rows: `0`.
- Reward die source: `event_rows`.

Remote Modal smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --observation-mode curvytron_actor_bridge_sample \
  --batch-size 4 \
  --player-count 2 \
  --rollout-steps 2 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 4 \
  --warmup-runs 0 \
  --steady-runs 1
```

Remote result:

- `ok: true`
- Modal app: `ap-WuWvnBbYYnnPAIhmgVXDxY`
- JAX backend/device: `gpu`, `cuda:0`
- GPU: `NVIDIA L4, 23034 MiB, 17140 MiB, 0 %, 580.95.05`
- packages: `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6`
- actor bridge source: `fixture_seeded_cpu_actor_loop_bridge`
- selected fixture group: `P2_K4`
- source preflight summary: `8` pass, `0` fail, `0` unsupported
- source tensor shape: `[4, 2, 9]`
- live/search root rows after `ego_mask` filtering: `4`
- root/search obs shape: `[4, 9]`
- reward/done/truncated shapes: `[4, 2]`, `[4]`, `[4]`
- legal action mask shape: `[4, 2, 3]`
- rollout source kinds: step 0 `fixture_source_moves`, step 1
  `synthetic_feedback_moves`
- host observation setup: `0.026884623000000385s`
- host-to-device placement: `0.08940860299999986s`
- compile plus first search run: `3.757359387s`
- steady run: `0.0019022039999994078s`
- median throughput labels: `2102.8238821920495` env rows/sec,
  `2102.8238821920495` live ego decisions/sec,
  `8411.295528768198` simulations/sec
- action weights finite and normalized; row sums ranged from
  `0.9999999403953552` to `1.0`

Interpretation: this moves the host-side boundary from a one-step debug packer
sample to a sample produced by the local actor bridge: fixture reset, real
vector env step 0, synthetic action feedback, real vector env step 1, debug
obs/reward/legal packing, live-ego filtering, then synthetic Mctx search. The
Modal mode still times host setup, device placement, compile/first search, and
steady search separately.

What remains fake: fixture reset/cycling, debug observation/reward schema,
synthetic feedback policy in the actor helper, synthetic JAX/Mctx model/search
dynamics, no checkpoint, no replay writer, no trainer, and no production
reset/autoreset contract.

## Follow-ups

- The fixture-seeded debug packer boundary and a small actor-bridge sample
  boundary now exist. The next interface needed is the same fixed-shape contract
  without fixture cycling and synthetic feedback.
- Keep host observation setup, device placement, compile/first-run, warmup, and
  steady-state search timing separate.
