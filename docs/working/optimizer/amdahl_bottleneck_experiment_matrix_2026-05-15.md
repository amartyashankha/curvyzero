# Amdahl Bottleneck Experiment Matrix

Date: 2026-05-15

Purpose: propose a profiling matrix that isolates render, search, env-engine,
and learner bottlenecks without launching live training or polluting production
run state. This is an optimizer-lane plan, not a Coach learning claim.

## Current Read

Existing docs already answer part of the question:

- Env-only no-death `browser_lines + simple_symbols` rollouts are render-heavy:
  render is roughly `77%` of scalar env-only wall across 100 to 2,000 steps.
- Stock LightZero full-loop rows are not render-only. At C32/sim8/no-death,
  wall also pays collector orchestration, policy collect, MCTS, replay sample,
  and learner work.
- Scalar `policy_observation_backend=jax_gpu` is wired but currently slower
  than `cpu_oracle`; it is a diagnostic path, not a production backend.
- The isolated lab GPU renderer is promising, especially on H100, and now
  reaches exact CPU parity on the first checked smoke rows after the
  owner-priority fix. It is still not a trainer backend.
- A GPU semantic bug was found in the lab renderer: the old JAX `browser_lines`
  path connected to slot-1 instead of the previous active same-owner
  visual-trail point. The current worktree has a benchmark-only fix, but
  full-loop GPU timings are still renderer economics only until adversarial
  parity tests prove the fix and the production path consumes it safely.

The experiment design below keeps those facts separate. It measures component
fractions by denominator, then reruns only the cells that can change a decision.

## Non-Pollution Rules

All full-loop rows must use:

```text
--mode profile
--env-variant source_state_fixed_opponent
--lightzero-eval-freq 0
--skip-lightzero-eval-in-profile
--save-ckpt-after-iter 9999
--no-background-eval-enabled
--no-background-gif-enabled
--output-detail compact
```

Operational rules:

- Never use `--mode train` for this matrix.
- Use fresh `run_id`s and keep `--profile-allow-auto-resume` off.
- Prefix run ids with `opt-bneck-20260515-...`, not a live-training campaign.
- Keep `profile_volume_commit=false`; it is already the profile default.
- Keep opponent inference on CPU with `--no-opponent-use-cuda` unless a row is
  explicitly about opponent CUDA.
- Do not use historical `--mode two-seat-selfplay` rows as optimizer evidence.
- Use `--env-telemetry-stride 1` only for attribution rows; use `32` or `128`
  for throughput rows so telemetry does not become the experiment.
- Summarize through `scripts/summarize_curvytron_lightzero_profiles.py` and
  record denominators: env steps/sec, policy roots/sec, simulations/sec,
  learner updates/sec, and completed episodes where available.

## Shared Full-Loop Template

Use the canonical launcher:

```text
uv run --extra modal modal run --quiet \
  -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode profile \
  --compute COMPUTE \
  --seed SEED \
  --run-id RUN_ID \
  --attempt-id ATTEMPT_ID \
  --env-variant source_state_fixed_opponent \
  --reward-variant sparse_outcome \
  --opponent-policy-kind frozen_lightzero_checkpoint \
  --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-dense-ckpt1-iter10000-sanity-20260512a/checkpoints/lightzero/iteration_32.pth.tar \
  --no-opponent-use-cuda \
  --env-manager-type MANAGER \
  --collector-env-num C \
  --n-episode C \
  --batch-size BATCH \
  --num-simulations SIMS \
  --source-max-steps HORIZON \
  --source-state-trail-render-mode browser_lines \
  --source-state-bonus-render-mode simple_symbols \
  --policy-observation-backend BACKEND \
  --stop-after-learner-train-calls LEARN_CALLS \
  --env-telemetry-stride TELEMETRY_STRIDE \
  --lightzero-eval-freq 0 \
  --skip-lightzero-eval-in-profile \
  --save-ckpt-after-iter 9999 \
  --no-background-eval-enabled \
  --no-background-gif-enabled \
  --output-detail compact
```

For long/no-death rows add:

```text
--disable-death-for-profile
--opponent-runtime-mode blank_canvas_noop
```

Warmup rule for full-loop Modal rows:

- Screening: one seed per cell, `SEED=304`, `LEARN_CALLS=12`.
- Confirmation: three seeds per decision cell, `304/305/306`, same knobs.
- Throw away any first row after a container/image/config change if package
  import, JIT, or auto-resume checks dominate the summary. Replace it with a
  fresh run id; do not average warmup contamination into the cell.
- Keep `--profile-cuda-sync-enabled` off for throughput. Use it only for one
  timing-attribution rerun when CUDA async timing is itself the question.

## Matrix 0: GPU Parity Gate

Do this before any GPU full-loop comparison is allowed.

| row | state family | backend candidate | knobs | runs | decision |
| --- | --- | --- | --- | ---: | --- |
| P0 | adversarial toy slots | JAX GPU renderer | owners `[0,1,0]`, inactive holes, `break_before`, cursor wrap/reset, both controlled players | until exact | Fails if slot-1 connectivity differs from CPU oracle. |
| P1 | real env rollout | JAX GPU renderer | `browser_lines`, `simple_symbols`, `real_env_steps=128`, `trail_slots=256`, verify both players | 3 seeds | Promotes renderer from economics-only to candidate if pixel diff is zero or explicitly approved tolerance. |
| P2 | long active-prefix state | JAX GPU renderer | active buckets `256/512/1024/2048/4096`, chronological draw order preserved | 3 seeds | Decides whether active-prefix compaction is semantically safe. |

Hard gate: the previous-active-same-owner `browser_lines` bug must be fixed, or
GPU full-loop rows stay labeled `invalid-for-training-speedup`.

## Matrix 1: Toy Amdahl Rows

These rows are deliberately small and local. They answer what the env/renderer
can do when search and learner are absent.

Use:

```text
uv run python scripts/profile_curvytron_render_trajectory_lengths.py \
  --lengths 32 64 128 256 512 1000 2000 \
  --render-modes browser_lines body_circles_fast \
  --repeats 5 \
  --warmup-steps 50 \
  --source-max-steps 4096 \
  --opponent-runtime-mode blank_canvas_noop \
  --policy wall_avoidant \
  --output artifacts/local/curvytron_render_profiles/bneck_envonly_20260515.json \
  --markdown
```

| knob | values | reason |
| --- | --- | --- |
| `lengths` | `32/64/128/256/512/1000/2000` | Separates short reset-heavy rollouts from long trail-heavy rollouts. |
| `render_modes` | `browser_lines`, `body_circles_fast` | Bounds render sensitivity while keeping browser as the semantic target. |
| `repeats` | `5` | Enough for median/p95 without spending Modal budget. |
| `warmup_steps` | `50` per repeat | Warms env allocation and caches before timing the cell. |

Decisions:

- If browser render fraction is still near `75%` across long rows, renderer
  work remains justified for long-survival policies.
- If fast render only improves env-only but not full-loop rows, the bottleneck
  is outside scalar render.
- If vector step catches render at long horizons, split env-engine movement,
  collision, bonus, and autoreset before more renderer work.

## Matrix 2: Isolated Render Backend Economics

This is a renderer bakeoff, not trainer evidence.

Use:

```text
uv run --extra modal modal run --quiet \
  -m curvyzero.infra.modal.source_state_gpu_render_benchmark \
  --state-source real_env_rollout \
  --batch-size B \
  --controlled-player PLAYER \
  --trail-slots SLOTS \
  --compute COMPUTE \
  --render-mode browser_lines \
  --bonus-render-mode simple_symbols \
  --render-surface block_704_gray64 \
  --bonus-count 8 \
  --warmup-runs 5 \
  --steady-runs 20 \
  --real-env-steps HORIZON \
  --verify-rows 4 \
  --transfer-output
```

Core cells:

| axis | values |
| --- | --- |
| `COMPUTE` | `gpu-l4-t4`, `gpu-h100` |
| `B` | `1`, `8`, `64`, `256` |
| `SLOTS` | `64`, `256`, `1024` |
| `PLAYER` | `0` for timing, rerun best cells with `1` for parity |
| `HORIZON` | `64` for short, `512` for long/no-death-like state density |

Run count: one screening pass for all `2 * 4 * 3 * 2 = 48` timing cells; three
repeats only for the best B/SLOTS/GPU knee and for any cell that contradicts
prior H100-over-L4 evidence.

Decisions:

- `B=1` versus `B=64/256` quantifies launch/readback amortization.
- `SLOTS` slope estimates how much active-prefix bucketing can buy.
- L4 versus H100 decides whether GPU rendering belongs on cheap profile
  hardware or only on H100-class rows.
- If `transfer_output=false` is later added, compare it to `--transfer-output`
  to price the host round trip separately.

## Matrix 3: CPU Oracle Full-Loop Attribution

These are the first Amdahl-valid stock LightZero rows. Keep GPU observation out
of this matrix.

| cell | manager | C | sims | horizon/death | batch | telemetry | runs | decision |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| A1 | `base` | 1 | 2 | normal, `source_max_steps=256` | 32 | 1 | 3 | Clean scalar split of render/env/search/learner with short natural episodes. |
| A2 | `base` | 1 | 2 | no-death, `source_max_steps=512` | 32 | 1 | 3 | Scalar long-horizon render fraction inside real trainer plumbing. |
| A3 | `subprocess` | 32 | 8 | normal, `256` | 32 | 1 | 3 | Worker timing with realistic search and short trajectories. |
| A4 | `subprocess` | 32 | 8 | no-death, `512` | 32 | 1 | 3 | Worker timing with long trajectories and MCTS included. |
| A5 | `subprocess` | 128 | 8 | no-death, `512` | 32 | 32 | 1 screen, 3 confirm | Does render remain visible once env work is parallelized widely? |

Common knobs:

```text
BACKEND=cpu_oracle
COMPUTE=gpu-l4-t4-cpu40
LEARN_CALLS=12
source_state_trail_render_mode=browser_lines
source_state_bonus_render_mode=simple_symbols
```

Decisions:

- A1/A2 isolate short versus long trajectory effects without subprocess
  scheduling.
- A3/A4 show whether long/no-death changes wall fractions once search is real.
- A5 checks the Amdahl denominator shift: render can dominate workers while
  not dominating full-loop wall.

## Matrix 4: Search Fraction Sweep

Hold render/backend fixed and vary MCTS pressure.

| compute | C | sims | horizon/death | batch | telemetry | runs | decision |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `gpu-l4-t4-cpu40` | 64 | `2/8/16/32` | normal, `256` | 32 | 32 | 1 screen, 3 confirm at knee | Finds short-trajectory search fraction. |
| `gpu-l4-t4-cpu40` | 64 | `2/8/16/32` | no-death, `512` | 32 | 32 | 1 screen, 3 confirm at knee | Finds long-trajectory search fraction. |
| `gpu-h100-cpu40` | 64 | `16/32` | no-death, `512` | 32 | 32 | 1 screen, 3 confirm if H100 wins | Decides whether H100 is only useful once search pressure rises. |

Common knobs:

```text
BACKEND=cpu_oracle
MANAGER=subprocess
source_state_trail_render_mode=browser_lines
source_state_bonus_render_mode=simple_symbols
```

Decisions:

- If MCTS grows linearly and dominates at sim16/sim32, renderer work cannot
  rescue high-search runs alone.
- If H100 wins only at sim16/sim32, use L4/T4 for sim8 profiles and reserve
  H100 for search-heavy or batched-render rows.
- If sim2/sim8 have similar wall but different policy quality later, Coach owns
  the quality tradeoff; Optimizer only reports cost.

## Matrix 5: Batch And Collector Scaling

Hold search fixed, vary collection width and learner batch. This answers batch
and self-play scaling in the trusted stock shape: many simultaneous
fixed-opponent self-play-like episodes, not historical custom two-seat mode.

| axis | values |
| --- | --- |
| `C`, with `n_episode=C` | `32`, `64`, `128`, `256`, `384` |
| `batch-size` | `16`, `32`, `64` |
| `compute` | `gpu-l4-t4-cpu40`, `gpu-h100-cpu40` |
| `sims` | `8` |
| `horizon/death` | no-death `source_max_steps=512`; one normal-death `256` sentinel at C128 |
| `backend` | `cpu_oracle` |
| `telemetry` | `128` for throughput, `1` only on C128 sentinels |

Screening run count:

- Width sweep: `5 C * 2 compute * 1 batch32 = 10` rows.
- Batch sweep at the best L4 and best H100 width: `2 widths * 2 compute * 3
  batches = 12` rows.
- Confirm only the selected width/batch/hardware with three seeds.

Decisions:

- Finds the collector-width knee where subprocess overhead beats parallel env
  work.
- Tests whether learner batch affects wall; prior evidence says batch64 did not
  help, so batch64 is a sentinel, not a default.
- Separates hardware wins from width wins: H100 should not be credited if the
  real win was simply wider collection.

## Matrix 6: Learner Fraction Rows

Use these only if matrices 3-5 show learner or replay sample above `15%` of
wall, or if learner idle becomes the bottleneck.

| row | knobs | runs | decision |
| --- | --- | ---: | --- |
| L1 | C32/sim2/no-death512, `batch-size=16/32/64`, `LEARN_CALLS=24` | 1 screen | Makes learner/update visible by reducing search pressure. |
| L2 | C128/sim8/no-death512, `batch-size=16/32/64`, `LEARN_CALLS=24` | 1 screen, 3 confirm if learner >15% | Checks whether learner grows under the selected throughput shape. |
| L3 | Same as selected row, `save_ckpt_after_iter=1` | 1 | Prices checkpoint cadence only; do not mix into clean throughput rows. |

Decisions:

- If learner remains under `10-15%`, do not optimize learner for this lane yet.
- If replay sample dominates learner, split replay sample, target construction,
  and update before changing model hardware.
- If checkpoint cadence is expensive, keep profile/training checkpoint cadence
  separate in reports.

## Matrix 7: GPU After Parity

Run this only after Matrix 0 passes.

| row | backend | C | sims | horizon/death | compute | telemetry | runs | decision |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| G1 | `cpu_oracle` vs fixed `jax_gpu` | 1 | 2 | no-death `512` | `gpu-h100-cpu40` | 1 | 3 paired seeds | Scalar canary: does fixed active-prefix GPU beat CPU in base manager? |
| G2 | `cpu_oracle` vs batched candidate | 32 | 8 | no-death `512` | `gpu-h100-cpu40` | 1 | 3 paired seeds | First real full-loop speedup test. |
| G3 | `cpu_oracle` vs batched candidate | 128 | 8 | no-death `512` | `gpu-h100-cpu40` | 32 | 3 paired seeds | Does batching still help when collection is wide? |
| G4 | batched candidate only | 128 | 8 | no-death `1024` | `gpu-h100-cpu40` | 32 | 1 screen | Long-horizon stress after source1024 footguns are understood. |

Do not run scalar `jax_gpu` under `subprocess`; the existing canary failed
before collection with CUDA subprocess initialization errors.

Decisions:

- G1 decides whether active-prefix scalar GPU is even competitive with CPU.
- G2 decides whether the batched integration beats CPU in a realistic loop.
- G3 checks whether the speedup survives the Amdahl denominator shift.
- G4 is a stress row only; if it fails before learner, label it
  collection-only evidence.

## Final Decision Rules

Use medians across confirmed seeds. A cell changes direction only if the effect
is larger than noise and visible in the right denominator.

| question | accept evidence | decision threshold |
| --- | --- | --- |
| Render fraction | env-only render timers plus full-loop worker observation timers | Optimize render only if it is >50% of env-only long wall or >20% of full-loop wall after parallelism. |
| Search fraction | MCTS/policy collect timers and simulations/sec | Prioritize search batching/hardware if MCTS+policy collect is >35% of full-loop wall. |
| Env engine | vector step, opponent step, reset/autoreset, terminal/final obs | Split env engine if vector/reset/opponent exceeds render or search in confirmed rows. |
| Learner | learner sample/update/target timing | Optimize learner only if learner+sample is >15% of wall or actors wait on learner. |
| CPU oracle vs GPU | paired seeds, same commit, same observations, post-parity only | Promote GPU only if full-loop wall improves by at least 1.25x at C32/C128 and parity is trusted. |
| Batch/self-play scaling | env steps/sec, roots/sec, learner updates/sec, p95 action latency if exposed | Pick the smallest C within 10% of peak throughput unless Coach asks for maximum collection. |
| GPU type | matched L4/H100 rows at same C/sims/horizon | Use H100 only where it wins full-loop wall by >20% or is required for batched renderer throughput. |

The important negative result is also useful: if render is huge in toy rows but
small in confirmed C128/C256 full-loop wall, stop selling renderer work as a
whole-loop fix. Then the next optimizer bottleneck is search/collector
orchestration, not pixels.
