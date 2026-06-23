# Fixed-A3 Vendored CTree Status, 2026-05-23b

Scope: read-only scout of the current working tree. I did not touch live
training runs. The only write from this pass is this report.

## Short Answer

Flat-A3 is wired into the real stock `train_muzero` profile shell, but only
through the profile-only `direct_ctree_gpu_latent` collect-search monkeypatch
and only on isolated CPU40 optimizer images that build the vendored Cython
extension. It is not a default stock/train-mode backend and should not be Coach
launch advice.

The connection to the optimizer profile denominator already exists in this
working tree. The available full-loop evidence says it is capped for the tested
row: `direct LightZero CTree 516.55 steps/sec` vs `flat-A3 CTree 509.69
steps/sec` on `opt-flat-a3-ab-20260522a`, H100 C64/sim16/3 learner.

## Current Files Present

Repo-status caveat: the fixed-A3 surface is present in the dirty working tree,
but the top-level `src/curvyzero/vendor/`, `scripts/build_lightzero_ctree_a3.py`,
`scripts/benchmark_lightzero_ctree_no_model.py`, and
`src/curvyzero/infra/modal/lightzero_ctree_no_model_benchmark.py` showed as
untracked in `git status --short` during this scout.

Vendored extension source:

- `src/curvyzero/vendor/lightzero_ctree_a3/ctree_muzero/mz_tree_a3.pyx`
  exposes the LightZero-like Cython API plus
  `batch_backpropagate_flat_a3(...)` and `set_deterministic_tie_breaking(...)`.
- `src/curvyzero/vendor/lightzero_ctree_a3/ctree_muzero/lib/cnode.cpp` adds
  `CNode::expand_a3(...)`, deterministic tie-breaking for parity checks, and
  `cbatch_backpropagate_flat_a3(...)`.
- `src/curvyzero/vendor/lightzero_ctree_a3/ctree_muzero/lib/cnode.h` and
  `mz_tree_a3.pxd` declare the flat-A3 ABI.
- Generated local artifacts also exist:
  `mz_tree_a3.cpp` and `mz_tree_a3.cpython-311-darwin.so`. They are build
  outputs, not the source of truth; Modal builds fresh inside the image.

Build and no-model benchmark:

- `scripts/build_lightzero_ctree_a3.py` is an opt-in Cython build script for
  extension name
  `curvyzero.vendor.lightzero_ctree_a3.ctree_muzero.mz_tree_a3`.
- `scripts/benchmark_lightzero_ctree_no_model.py` has four backends:
  `ctree-list`, `ctree-torch-d2h`, `ctree-flat-a3`, and `fake-flat`.
- The flat backend imports the vendored module and calls
  `batch_backpropagate_flat_a3(...)` with contiguous `float32` reward `[N]`,
  value `[N]`, and policy `[N,3]` arrays.
- Its parity check compares vendored list backprop against flat-A3 with
  deterministic tie-breaking enabled only for the check.

Modal no-model runner:

- `src/curvyzero/infra/modal/lightzero_ctree_no_model_benchmark.py` builds an
  image with LightZero, Torch, NumPy, Cython, copies `src/` and `scripts/`, and
  runs `scripts/build_lightzero_ctree_a3.py build_ext --inplace`.
- The runner supports `ctree-flat-a3` and forwards `flat_a3_parity_check`.
- It is explicitly not a trainer.

## Train/Profile Wiring

The stock trainer module now defines:

- `COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT =
  "direct_ctree_gpu_latent"`.
- `COLLECT_SEARCH_CTREE_BACKEND_FLAT_A3 = "flat_a3"`.
- `_validate_collect_search_ctree_backend(...)` accepts `lightzero` or
  `flat_a3`.

Import/image handling:

- `_import_collect_search_tree_muzero("lightzero")` returns LightZero's normal
  `mcts_ctree.tree_muzero`.
- `_import_collect_search_tree_muzero("flat_a3")` prepends `/repo/src` and the
  nested repo package paths before importing the vendored
  `curvyzero.vendor.lightzero_ctree_a3.ctree_muzero.mz_tree_a3`.
- `ctree_a3_image` extends the normal trainer image with Cython, adds
  `scripts/build_lightzero_ctree_a3.py`, and builds the extension in-place.

The actual collect-search hook:

- `_install_lightzero_collect_search_backend_hook(...)` monkeypatches
  `lzero.policy.muzero.MuZeroPolicy._forward_collect` only for
  `collect_search_backend=direct_ctree_gpu_latent`.
- The hook still runs inside real stock `train_muzero`; `_run_visual_survival_train`
  installs it immediately before the LightZero trainer call.
- For `ctree_backend=flat_a3`, the hook requires `action_count == 3`.
- Root prep is still list-shaped: `roots.prepare(...)` receives root rewards,
  policy logits, noises, and legal actions as Python/list payloads.
- The per-simulation recurrent payload changes: model outputs are still copied
  D2H once per simulation, but flat-A3 converts reward/value/policy into
  contiguous arrays and calls `batch_backpropagate_flat_a3(...)` instead of
  nested-list `batch_backpropagate(...)`.
- The compact profile output records `collect_search_ctree_backend`,
  `collect_search_backend_flat_payload`, and `search_backend_proof` with
  observed runtime CTree backends.

Profile-only guardrails:

- `_run_visual_survival_train(...)` rejects any non-stock collect-search backend
  unless `mode == "profile"` and compute uses a GPU.
- It also rejects `collect_search_ctree_backend != "lightzero"` unless
  `collect_search_backend == "direct_ctree_gpu_latent"`.
- The CLI launcher routes `flat_a3` only to
  `lightzero_curvytron_visual_survival_gpu_cpu40_ctree_a3` or
  `lightzero_curvytron_visual_survival_h100_cpu40_ctree_a3`.
- CPU, CPU64, normal `gpu-l4-t4`, and H100x2 reject `flat_a3`; normal stock/live
  images do not build the Cython extension.

So the precise answer is: flat-A3 is wired into the real stock
`train_muzero` profile hook, not merely the no-model CTree benchmark, but it is
isolated to profile mode and the CTree-A3 optimizer images. It is not wired as a
live/train-mode stock backend.

## Tests and Tooling

Existing local tests cover the surrounding wiring:

- `tests/test_lightzero_phase_profiler.py` validates CTree backend values,
  stock no-op behavior, direct collect-search hook behavior, bad-mask rejection,
  output-fast-path semantics, and compact output proof fields for
  `collect_search_ctree_backend="flat_a3"`.
- `tests/test_curvytron_profile_grid_builder.py` checks that the profile grid
  emits both `--collect-search-backend direct_ctree_gpu_latent` and
  `--collect-search-ctree-backend flat_a3`, and rejects flat-A3 with stock
  search.
- `tests/test_summarize_curvytron_optimizer_profile_results.py` requires flat-A3
  runtime proof: observed `flat_a3` in `search_backend_proof` and a present flat
  payload timer.

Coverage gap: I did not find a focused unit test that calls the real
`_direct_ctree_gpu_latent_search_for_collect(..., ctree_backend="flat_a3")`
branch and asserts `batch_backpropagate_flat_a3(...)` is invoked. Current tests
prove CLI/manifest/summary proof plumbing and the direct hook shape, while the
no-model benchmark proves the vendored flat backprop semantics.

## Next Patch

No connection patch is needed to put flat-A3 into the optimizer profile
denominator; the current working tree already threads:

```text
CLI/main
-> _run_visual_survival_train(command)
-> _install_lightzero_collect_search_backend_hook(ctree_backend)
-> _import_collect_search_tree_muzero("flat_a3")
-> _direct_ctree_gpu_latent_search_for_collect(...)
-> batch_backpropagate_flat_a3(...)
-> compact search_backend_proof
-> summarizer attestation
```

The only tiny patch I would consider next is test-only, not training code:

```text
Add a focused test in tests/test_lightzero_phase_profiler.py that installs the
direct collect hook with ctree_backend="flat_a3", monkeypatches the vendored tree
import/direct search boundary, and asserts:

- all-3-action masks route through vendored Roots;
- ctree_backend="flat_a3" reaches _direct_ctree_gpu_latent_search_for_collect;
- action_count != 3 raises before search;
- compact proof still records observed flat_a3 and flat_payload_timer_present.
```

To prove the speed cap, no patch is required; run a matched profile A/B and let
the summarizer attestation fail if the row only echoes the command. Existing
docs already record the first cap row:

```text
opt-flat-a3-ab-20260522a, H100 C64/sim16/3 learner
direct LightZero CTree: 516.55 steps/sec
flat-A3 CTree:          509.69 steps/sec
```

That caps flat-A3 for that denominator. A stronger cap would repeat the same
comparison at the current larger denominator, especially sim16 and sim32 with
the same collector/batch/env-manager settings used for the compact-service
comparison.

## Smallest Commands

Local unit smoke, no Modal and no training:

```bash
uv run pytest \
  tests/test_lightzero_phase_profiler.py::test_collect_search_backend_validation_and_stock_noop \
  tests/test_lightzero_phase_profiler.py::test_curvytron_compact_output_uses_mcts_root_fallback_for_profile_steps \
  tests/test_curvytron_profile_grid_builder.py::test_profile_grid_builder_emits_collect_search_backend_flags \
  tests/test_curvytron_profile_grid_builder.py::test_profile_grid_builder_rejects_flat_ctree_without_direct_search \
  tests/test_summarize_curvytron_optimizer_profile_results.py::test_profile_attestation_requires_flat_a3_runtime_proof
```

Local vendored-extension parity/smoke, no env/model/training:

```bash
uv run --with Cython --with numpy python scripts/build_lightzero_ctree_a3.py build_ext --inplace
uv run python scripts/benchmark_lightzero_ctree_no_model.py \
  --roots 64 \
  --simulations 1,2,4,8 \
  --iterations 3 \
  --warmup 1 \
  --backends ctree-flat-a3 \
  --legal-profiles all3,mixed_2of3 \
  --flat-a3-parity-check
```

H100 no-model boundary check, still not training:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_ctree_no_model_benchmark \
  --compute h100 \
  --roots 1024 \
  --simulations 16 \
  --iterations 100 \
  --warmup 10 \
  --backends ctree-list,ctree-flat-a3 \
  --legal-profiles all3,mixed_2of3 \
  --flat-a3-parity-check
```

Profile-only denominator A/B. Use fresh ids; this launches new profile rows, not
live Coach training:

```bash
uv run python scripts/build_curvytron_profile_grid.py \
  --experiment-id opt-flat-a3-denom-check-20260523b \
  --run-prefix opt-flat-a3-denom-check-20260523b \
  --computes gpu-h100-cpu40 \
  --env-manager-types curvyzero_batched_profile \
  --collectors 64 \
  --batch-sizes 64 \
  --num-simulations 16 \
  --collect-search-backends direct_ctree_gpu_latent \
  --collect-search-ctree-backends lightzero,flat_a3 \
  --max-train-iter 32 \
  --stop-after-learner-train-calls 3 \
  --detached

uv run python scripts/run_curvytron_optimizer_profile_manifest.py \
  --manifest artifacts/local/curvytron_optimizer_profile_manifests/opt-flat-a3-denom-check-20260523b.json \
  --action launch-and-collect \
  --collect-timeout-sec 1800

uv run python scripts/summarize_curvytron_optimizer_profile_results.py \
  --results-dir artifacts/local/curvytron_optimizer_profile_results/opt-flat-a3-denom-check-20260523b \
  --require-attestation
```

