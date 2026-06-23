# Profile And Run Plan

Status: active run ladder. Local, Modal reference, Flash H100 raw-env, Flash
accelerated validate/parity, and one Flash PPO profile row have run. The
remaining cloud step would be a CurvyZero row through an existing accepted gate,
or a Flash DDP rerun if the recovered H100:8 headline needs fresh evidence.

## Rules

- Move in small, real steps: local check, CPU/reference Modal check, one H100
  raw-env control, one H100 PPO profile control.
- Keep PPO Flash results separate from MuZero CurvyZero speed claims.
- Label raw environment throughput, policy-training throughput, and whole-loop
  learning throughput as different denominators.
- For Modal launches, capture exact command, app/function, artifact path,
  hardware, and stop/cleanup boundary.

## Local First

| Run | Purpose | Command | Gate |
| --- | --- | --- | --- |
| Flash source compile check | Verify recovered Python files are syntax-clean in this checkout | From recovered app dir: `python -m py_compile accelerated/__init__.py accelerated/abi.py accelerated/benchmark.py accelerated/env.py accelerated/kernels.py training/__init__.py training/ddp.py training/models.py training/ops.py training/policies.py training/ppo.py training/reachability.py training/rewards.py modal_train.py` | Passed locally. |
| Flash local reference smoke/eval | Fidelity oracle sanity without Modal | From recovered app dir: `node reference/smoke.js`, `node reference/eval.js` | Passed locally; outputs saved under `artifacts/local/flash_controls/reference/`. |
| Flash local reference benchmark | CPU reference throughput and deterministic hashes | From recovered app dir: `node reference/benchmark.js --worker-counts 1,2,4 --episodes-per-worker 4 --ticks 600 --workloads scripted,random --json` | Passed locally; JSON saved under `artifacts/local/flash_controls/reference/`. |
| Flash Modal reference checks | Same reference path in Modal | From recovered app dir: `modal run --env david.wang-dev modal_reference.py::check`, `modal run --env david.wang-dev modal_reference.py::smoke`, `modal run --env david.wang-dev modal_reference.py::eval` | Passed; eval output saved under `artifacts/local/flash_controls/reference/`. |
| Flash Modal reference benchmark | Modal CPU reference throughput | From recovered app dir: `modal run --env david.wang-dev modal_reference.py::benchmark --worker-counts 1,2,4,8,16 --episodes-per-worker 16 --ticks 1200 --workloads scripted,random,lookahead --json-output` | Passed; not training speed. |
| Flash Modal CPU check | Repo-owned non-GPU syntax/policy sanity check | From recovered app dir: `MODAL_FUNCTION_RUNTIME=runc modal run --env david.wang-dev modal_train.py::check` | Not run in this packet; local py_compile and accelerated check covered syntax. |
| Flash accelerated validate/parity | Validate CUDA mechanics and semantic parity | From recovered app dir: `MODAL_FUNCTION_RUNTIME=runc modal run --env david.wang-dev modal_accelerated.py::validate`; `MODAL_FUNCTION_RUNTIME=runc modal run --env david.wang-dev modal_accelerated.py::parity` | Passed; logs saved under `artifacts/local/flash_controls/raw_env/`. |
| CurvyZero focused tests | Ensure current comparison anchors are not stale | `uv run pytest tests/test_reward_contracts.py tests/test_lightzero_config_builder.py tests/test_compact_owned_loop.py -q` | Passed: 59 tests. |
| CurvyZero second local slice | Compact search/replay, profiler, Wave A launch audit | `uv run --extra dev --extra lightzero pytest tests/test_compact_search_replay_contract.py tests/test_lightzero_phase_profiler.py tests/test_curvytron_wave_a_launch_packet_audit.py -q` | Passed: 71 tests. |
| CurvyZero direct-CTree exact compare | Local comparison anchor, no Modal | `uv run --extra dev --extra lightzero python scripts/compare_curvytron_direct_ctree_stock.py --seeds 2 --batch-rows 4 --num-simulations 8 --impls direct_ctree --action-mask-scenario mixed_legal_cycle --strict-exact` | Passed; JSON saved under `artifacts/local/curvyzero_comparison_controls/`. |

## Modal/H100 Candidates

| Candidate | Question | Comparable To | Risk |
| --- | --- | --- | --- |
| Flash `modal_accelerated.py::benchmark` | Raw accelerated env throughput | CurvyZero env/profile microbenchmarks only | Not a whole-loop training comparison. |
| Flash `modal_train.py::profile_training` | PPO rollout/update split | CurvyZero training profile rows only at coarse level | Different algorithm and batch semantics. |
| Flash `modal_train.py::train_preset` smoke | Does PPO produce/export usable bots quickly? | CurvyZero learning-retention campaign only qualitatively | Could consume GPU without answering speed target. |
| CurvyZero accepted manifest/coach speed row | Same-work MuZero whole-loop candidate only if it uses existing gates | OPT-104 speed ledger | Do not launch from this packet without an accepted manifest/gate. |

Candidate exact shapes:

```bash
# Flash raw env, raycast H100 control, no learning claim.
MODAL_FUNCTION_RUNTIME=runc modal run modal_accelerated.py::benchmark \
  --num-envs 8192 \
  --steps 2048 \
  --warmup-steps 256 \
  --max-trail 512 \
  --collision-mode grid \
  --print-holes \
  --include-observations \
  --observation-mode raycast_v1 \
  --json-output

# Flash PPO profile, one small update, no policy-quality claim.
MODAL_FUNCTION_RUNTIME=runc modal run modal_train.py::profile_training \
  --preset diagnostic \
  --run-name profile-diagnostic-curvyzero-compare \
  --updates 1 \
  --num-envs 2048 \
  --rollout-steps 128 \
  --opponent raycast_safety
```

Kill/ignore conditions:

- `grid_overflow != 0` on Flash raw env benchmark.
- Missing timing JSON under `/checkpoints/profiles/` for Flash PPO profile.
- Any PPO profile interpreted as production PPO without noting `update_epochs=1`,
  capped minibatches, and `env_step_s` includes observation generation.
- Any DDP headline without `world_size`, `num_envs` per rank, `rollout_steps`,
  `updates`, `ddp_timing_sync`, `grid_overflow_sum`, and rank-log evidence.
- Any no-sync DDP row used as a bottleneck breakdown. No-sync rows are
  throughput-only.
- Any CurvyZero profile-only result that lacks explicit `profile_only` or
  `calls_train_muzero` metadata.
- Any CurvyZero speed row that cannot point to the OPT-104 same-work gate.

## Immediate Recommendation

1. Treat the current Flash rows as controls, not CurvyZero speed proof.
2. If speed work continues, pick a CurvyZero H100 row only through the existing
   optimizer or reward-campaign gate.
3. Rerun Flash DDP only if we specifically need fresh evidence for the recovered
   H100:8 PPO/DDP headline.
4. Keep artifacts under `artifacts/local/flash_controls/{reference,raw_env,ppo_profile}/`.
