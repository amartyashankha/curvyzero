# Modal Mctx GPU Smoke Runbook

Date: 2026-05-09

Status: first concrete GPU proof for the policy/search lane. Worker U ran it
successfully on 2026-05-09. This is not an environment benchmark and not a
trainer.

## Local Facts

Worker N found this local dependency state:

| Package | Local status |
| --- | --- |
| JAX | missing |
| Mctx | missing |
| PyTorch | installed, `2.9.1` |
| NumPy | installed, `2.4.0` |

No dependency was installed locally for that probe. Do not treat any local NumPy
or PyTorch stand-in as GPU or Mctx evidence.

## First GPU Command

Run this from the repository root:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_gpu_dependency_smoke
```

What it does:

- builds a Modal image with pinned `mctx`, `jax[cuda12]`, and `numpy`
- requests one cheap GPU, currently `L4` or `T4`
- imports `jax`, `jax.numpy`, `mctx`, and `numpy`
- records package versions, import status, JAX backend/devices, and `nvidia-smi`
- runs one tiny fixed-shape `mctx.gumbel_muzero_policy` search
- separates compile plus first run from a second steady-state run

What counts as a pass:

- `ok` is `true`
- every value in `imports` is `ok`
- `jax.default_backend` is `gpu` or `cuda`
- `jax.devices` contains a GPU device
- `nvidia_smi` is not `null`
- `timing.compile_plus_first_run_sec` and `timing.second_run_sec` are numbers
- `output.action_weights_finite` is `true`
- `output.action_weight_row_sums` are close to `1.0`

What a pass proves:

- Modal can build the pinned JAX/Mctx GPU dependency image.
- JAX sees the GPU inside Modal.
- Mctx can run a tiny real Gumbel MuZero search on a fixed batch.
- The timing harness blocks on device work before reporting.

What it does not prove:

- CurvyTron environment speed.
- Source-fidelity vector equivalence.
- End-to-end self-play throughput.
- GPU advantage over Modal CPU.
- Training correctness.

## If It Fails

If image build fails, the blocker is dependency resolution or CUDA wheel support.
Start by adjusting `MCTX_VERSION` or `JAX_VERSION` in
`src/curvyzero/infra/modal/mctx_dependency_smoke.py`; keep the change pinned and
document the reason.

If the function returns JSON with `ok: false`, read `imports`, `packages`,
`problems`, `jax`, and `nvidia_smi` first. A CPU backend with a non-null
`nvidia_smi` usually means the GPU exists but the JAX CUDA install is wrong.

If `nvidia_smi` is `null`, check the Modal GPU request and quota before changing
CurvyZero code.

## Benchmark Sequence After The Smoke Passes

These benchmark steps have each passed once on 2026-05-09. Rerun them only for
regression, dependency, or timing comparison.

Only after the dependency smoke passes, run the tiny synthetic policy/search
benchmark:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --batch-size 8 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 4 \
  --warmup-runs 1 \
  --steady-runs 2
```

This still is not environment evidence. It is only the first measured GPU
policy/search profile. Keep compile time, warmup time, steady-state time,
decisions/sec, simulations/sec, backend, device list, and package versions in
the result.

After the flat synthetic benchmark passes, run one observation-shaped synthetic
profile before attaching replay or trainer code:

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

This builds synthetic `obs[B,P,9]` with the current debug packer feature names,
adds ego/mask metadata, asserts the shape, flattens live ego rows into Mctx
roots, and reports host observation setup plus first device placement timing
separately from device-resident search timing. It is not CurvyTron rollout
throughput, not source-fidelity evidence, not real replay, not a trainer, and
not reward-learning evidence.

After the synthetic observation-shaped path passes, run the fixture-seeded debug
packer boundary profile:

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

This imports the existing CPU debug packer, builds fixture-seeded `obs[B,P,9]`,
reward, ego ids, and legal masks, filters live ego rows into Mctx roots, and
then times the same synthetic Mctx search boundary. It is still not a real
rollout, trainer, learned dynamics, replay, or final observation/reward
contract.

## Worker R Run Log

Worker R made the smoke report concrete but did not run the Modal GPU command.
Reason: running it would build a CUDA/JAX image and allocate a remote GPU. The
next real experiment was exactly the first GPU command above.

## Worker U Run Log

Date: 2026-05-09

Command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_gpu_dependency_smoke
```

Modal run:
<https://modal.com/apps/modal-labs/shankha-dev/ap-k2iRqzGbvLshqsZW8jDVav>

Result: pass. The run built the image, created `gpu_smoke`, allocated an NVIDIA
L4, and completed with `ok: true`. Cost note: this did allocate a remote Modal
GPU briefly, so normal Modal build/function/GPU billing may apply.

Key facts:

| Field | Result |
| --- | --- |
| Modal app | `ap-k2iRqzGbvLshqsZW8jDVav` |
| GPU | `NVIDIA L4, 23034 MiB, 580.95.05` |
| JAX backend/devices | `gpu`, `cuda:0` |
| Packages | `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6`, `numpy==2.4.4` |
| First run | `3.364952897s` |
| Second run | `0.0025687180000000254s` |
| Second-run throughput | `1557.1970142304294` decisions/sec, `6228.788056921718` simulations/sec |
| Output check | finite action weights, row sums `1.0`, `0.9999999403953552`, `1.0`, `1.0` |

Captured JSON:

```json
{
  "imports": {
    "jax": "ok",
    "jax.numpy": "ok",
    "mctx": "ok",
    "numpy": "ok"
  },
  "jax": {
    "default_backend": "gpu",
    "device_count": 1,
    "devices": [
      "cuda:0"
    ]
  },
  "nvidia_smi": "NVIDIA L4, 23034 MiB, 580.95.05",
  "ok": true,
  "output": {
    "action_histogram": [
      0,
      0,
      4
    ],
    "action_weight_row_sums": [
      1.0,
      0.9999999403953552,
      1.0,
      1.0
    ],
    "action_weights_finite": true,
    "actions": [
      2,
      2,
      2,
      2
    ]
  },
  "packages": {
    "jax": "0.7.0",
    "jaxlib": "0.7.0",
    "mctx": "0.0.6",
    "numpy": "2.4.4"
  },
  "problems": [],
  "profile": {
    "action_count": 3,
    "batch_size": 4,
    "hidden_dim": 8,
    "max_depth": 4,
    "num_simulations": 4,
    "policy_kind": "gumbel_muzero_policy"
  },
  "timing": {
    "compile_plus_first_run_sec": 3.364952897,
    "decisions_per_sec_second_run": 1557.1970142304294,
    "second_run_sec": 0.0025687180000000254,
    "simulations_per_sec_second_run": 6228.788056921718,
    "steady_state_second_run_sec": 0.0025687180000000254
  }
}
```

## Worker N CurvyTron-Shaped Obs Run Log

Date: 2026-05-09

Command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --observation-mode curvytron_debug \
  --batch-size 4 \
  --player-count 2 \
  --num-simulations 4 \
  --hidden-dim 32 \
  --max-depth 4 \
  --warmup-runs 1 \
  --steady-runs 2
```

Modal run:
<https://modal.com/apps/modal-labs/shankha-dev/ap-nBRyeqrFSrjHIYgehvLDZP>

Result: pass. The run allocated an NVIDIA L4 and completed with `ok: true`.
Cost note: this was deliberately tiny, but it did allocate a remote Modal GPU
briefly.

Key facts:

| Field | Result |
| --- | --- |
| Modal app | `ap-nBRyeqrFSrjHIYgehvLDZP` |
| GPU | `NVIDIA L4, 23034 MiB, 580.95.05` |
| JAX backend/devices | `gpu`, `cuda:0` |
| Packages | `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6` |
| Source obs shape | `[4, 2, 9]` |
| Root obs shape | `[8, 9]` |
| Ego/mask metadata | `ego_mask [4,2]`, `legal_action_mask [4,2,3]`, `ego_row_id [4,2]` |
| Compile plus first run | `4.113220473s` |
| Warmup | `0.0018798989999986304s` |
| Steady median | `0.001456043000000129s` |
| Median throughput | `2747.1716151237606` env rows/sec, `5494.343230247521` ego decisions/sec, `21977.372920990085` simulations/sec |
| Output check | finite action weights, normalized row sums from `0.9999998807907104` to `1.0000001192092896` |

## Worker R CurvyTron-Shaped Boundary Timing Run Log

Date: 2026-05-09

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

Modal run:
<https://modal.com/apps/modal-labs/shankha-dev/ap-cX21WujgqSjGAwMECD41cB>

Result: pass. This run kept the search size tiny and measured and reported
boundaries around the synthetic debug-shaped observation.

Key facts:

| Field | Result |
| --- | --- |
| Modal app | `ap-cX21WujgqSjGAwMECD41cB` |
| GPU | `NVIDIA L4, 23034 MiB, 580.95.05` |
| JAX backend/devices | `gpu`, `cuda:0` |
| Packages | `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6` |
| Build path | `host_numpy_then_jax_device_put` |
| Source obs shape | `[4, 2, 9]` |
| Root obs shape | `[8, 9]` |
| Counts | `4` env rows, `2` players/env, `8` candidate ego rows, `8` live ego rows, `8` search roots, `3` actions |
| Shape assertions | `passed` |
| Host obs setup | `0.0003469869999999098s` |
| Host-to-device placement | `0.21319387700000014s` |
| Compile plus first search run | `5.0306235169999995s` |
| Warmup | `0.002236305999998578s` |
| Steady median | `0.0024705955000010604s` |
| Median throughput | `1619.0428582899478` env rows/sec, `3238.0857165798957` ego decisions/sec, `12952.342866319583` simulations/sec |
| Output check | finite action weights, normalized row sums from `0.9999998807907104` to `1.0000001192092896` |

Boundary note: `host_to_device_transfer_sec` is the first explicit placement
measurement for a tiny synthetic host-built debug-shaped tensor in one fresh
Modal process. It is useful as a boundary smoke, not as a transfer scaling law.

This is runtime/shape evidence only. The observation values are synthetic, the
recurrent dynamics are synthetic, and the benchmark does not run or validate the
CurvyTron environment.

## Worker V Fixture-Seeded Debug Packer Boundary Run Log

Date: 2026-05-09

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

Modal run:
<https://modal.com/apps/modal-labs/shankha-dev/ap-eBPR9uUVfJhXB7nbaItivH>

Result: pass. This run replaced the synthetic host-built debug observation
values with existing fixture-seeded CPU debug packer output, then filtered live
ego rows into the same tiny synthetic Mctx search.

Key facts:

| Field | Result |
| --- | --- |
| Modal app | `ap-eBPR9uUVfJhXB7nbaItivH` |
| GPU | `NVIDIA L4, 23034 MiB, 580.95.05` |
| JAX backend/devices | `gpu`, `cuda:0` |
| Packages | `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6` |
| Build path | `seed_fixtures_source_preflight_batched_step_pack_debug_obs_reward_then_live_ego_filter_then_jax_device_put` |
| Packer source | `fixture_seeded_cpu_debug_packer` |
| Selected fixture group | `P2_K4` |
| Source obs shape | `[4, 2, 9]` |
| Root obs shape | `[4, 9]` |
| Counts | `4` env rows, `2` players/env, `8` candidate ego rows, `4` live ego rows, `4` search roots, `3` actions |
| Reward source | `event_rows` die evidence, reward shape `[4, 2]` |
| Host obs setup | `0.02176979999999995s` |
| Host-to-device placement | `0.08496403299999988s` |
| Compile plus first search run | `3.9509587949999996s` |
| Steady median | `0.002627230999999952s` |
| Median throughput | `1522.515530610012` env rows/sec, `1522.515530610012` live ego decisions/sec, `6090.062122440048` simulations/sec |
| Output check | finite action weights and normalized row sums |

This is one step less fake than the synthetic debug observation path. It is
still not a real rollout, trainer, learned dynamics, replay, or final
observation/reward contract.

## Worker DD Actor-Bridge Sample Boundary Run Log

Date: 2026-05-09

Command:

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

Modal run:
<https://modal.com/apps/modal-labs/shankha-dev/ap-WuWvnBbYYnnPAIhmgVXDxY>

Result: pass. This tiny L4 run consumed one fixed-shape local actor-bridge
sample, filtered live ego rows, placed them on the JAX device, and ran the same
synthetic Mctx search boundary.

Key facts:

| Field | Result |
| --- | --- |
| Modal app | `ap-WuWvnBbYYnnPAIhmgVXDxY` |
| GPU | `NVIDIA L4, 23034 MiB, 580.95.05` |
| JAX backend/devices | `gpu`, `cuda:0` |
| Packages | `jax==0.7.0`, `jaxlib==0.7.0`, `mctx==0.0.6` |
| Build path | `seed_fixtures_source_preflight_batched_step_actor_bridge_sample_then_live_ego_filter_then_jax_device_put` |
| Actor bridge source | `fixture_seeded_cpu_actor_loop_bridge` |
| Selected fixture group | `P2_K4` |
| Source preflight | `8` pass, `0` fail, `0` unsupported |
| Rollout source kinds | step 0 `fixture_source_moves`, step 1 `synthetic_feedback_moves` |
| Source obs shape | `[4, 2, 9]` |
| Root obs shape | `[4, 9]` |
| Counts | `4` env rows, `2` players/env, `8` candidate ego rows, `4` live ego rows, `4` search roots, `3` actions |
| Reward/source surfaces | reward `[4, 2]`, done `[4]`, truncated `[4]`, legal action mask `[4, 2, 3]`, reward source `event_rows` |
| Host obs setup | `0.026884623000000385s` |
| Host-to-device placement | `0.08940860299999986s` |
| Compile plus first search run | `3.757359387s` |
| Steady median | `0.0019022039999994078s` |
| Median throughput | `2102.8238821920495` env rows/sec, `2102.8238821920495` live ego decisions/sec, `8411.295528768198` simulations/sec |
| Output check | finite and normalized action weights; row sums from `0.9999999403953552` to `1.0`; actions `[1, 1, 2, 1]` |

This is boundary evidence only. The vector env steps in the actor-bridge sample
are real NumPy vector steps and the final debug obs/reward/legal masks are real
debug packer output, but the run is still fixture-seeded, uses synthetic action
feedback after step 0, uses synthetic JAX/Mctx model dynamics, and does not
exercise replay, a learner, a trainer, final observation/reward contracts, or a
production reset/autoreset loop.
