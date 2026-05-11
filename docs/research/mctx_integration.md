# JAX/Mctx Integration Plan

Status: Research note
Last updated: 2026-05-09

## Short Answer

Mctx is the fallback/comparison path now, not the immediate implementation
lane. The immediate lane is LightZero-first: make a custom dummy Pong env run
through LightZero's real MuZero trainer before writing trainer pieces
ourselves.

Mctx remains valuable because its public API is exactly the search piece we
would need if LightZero fails: batched, JIT-compatible tree search over a
learned recurrent model. It is not a trainer, replay buffer, environment
wrapper, league manager, or Modal orchestrator.

Fallback path after LightZero fails or after LightZero needs a comparison:

1. Use `mctx.gumbel_muzero_policy` first, with `mctx.muzero_policy` as a comparison point.
2. Keep the real CurvyZero simulator outside the search tree. Mctx search calls only learned `representation`, `dynamics`, and `prediction` functions.
3. Search one ego-perspective row per live player or per selected focal player. Do not start with joint-action search.
4. Keep action count fixed. For `curvyzero-v0`, use `A=3`: left, straight, right.
5. Treat `batch_size`, `num_actions`, `num_simulations`, `max_depth`, observation shape, and hidden-state shape as fixed benchmark profiles. Changing them should be expected to recompile.
6. Keep synthetic Mctx benchmarks labeled as search/runtime evidence until a
   real trainer with replay, update, checkpoint, and eval exists.

Current smoke boundary, 2026-05-09: the Mctx lane has proven dependency and
search mechanics only. It has not run training. The immediate new target is the
LightZero dummy Pong custom-env config/import smoke, then the tiny LightZero
dummy Pong trainer smoke:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 0
```

The project-owned Mctx trainer should start only if LightZero cannot preserve
the custom env, metadata, scorecard telemetry, or Modal artifacts. Until then,
synthetic Mctx benchmark passes must not be described as MuZero training.

## Mctx API Shape

Mctx exposes a low-level `search` function and high-level policies including `muzero_policy`, `gumbel_muzero_policy`, and `stochastic_muzero_policy`. The README states that search algorithms are JAX-native, support JIT compilation, and operate on batches in parallel. The high-level MuZero policies expect a root prediction plus a recurrent function.

Core root object:

```python
mctx.RootFnOutput(
    prior_logits=policy_logits,  # float32[B, A]
    value=value,                 # float32[B]
    embedding=hidden,            # pytree with leaves [B, ...]
)
```

Core recurrent output:

```python
mctx.RecurrentFnOutput(
    reward=reward,               # float32[B]
    discount=discount,           # float32[B]
    prior_logits=policy_logits,  # float32[B, A]
    value=value,                 # float32[B]
)
```

Core policy output:

```python
mctx.PolicyOutput(
    action=action,                # int32[B]
    action_weights=weights,       # float32[B, A]
    search_tree=tree,             # mctx.Tree with batch axis
)
```

Important shape facts from the Mctx source:

| Field | Shape | CurvyZero meaning |
| --- | --- | --- |
| `root.prior_logits` | `[B, A]` | Policy logits for each ego row and legal ego action. |
| `root.value` | `[B]` | Current value from the ego player's perspective. |
| `root.embedding` | `[B, ...]` | Latent state consumed by learned dynamics. |
| `action` into `recurrent_fn` | `[B]` | One selected discrete action per batch row. |
| `RecurrentFnOutput.reward` | `[B]` | Predicted immediate ego reward. |
| `RecurrentFnOutput.discount` | `[B]` | Predicted discount, usually `gamma` or 0 at predicted terminal states. |
| `RecurrentFnOutput.prior_logits` | `[B, A]` | Policy logits for the next latent state. |
| `RecurrentFnOutput.value` | `[B]` | Predicted next-state ego value. |
| `invalid_actions` | `[B, A]` | Ones are invalid, zeros are valid. |
| `Tree` | `[B, N, ...]` | `N = num_simulations + 1`; Mctx stores per-node stats and embeddings. |

The tree stores embeddings for every expanded node. That means hidden-state size is a first-order memory constraint. A spatial hidden state of `float32[B, N, 64, 8, 8]` uses about `B * N * 64 * 8 * 8 * 4` bytes before tree statistics. At `B=1024` and `num_simulations=50`, embeddings alone are about 855 MiB.

## Recurrent Function Contract

The project-owned model contract should stay thin:

```python
def make_root(params, obs):
    hidden = representation(params, obs)
    policy_logits, value = prediction(params, hidden)
    return mctx.RootFnOutput(
        prior_logits=policy_logits,
        value=value,
        embedding=hidden,
    )


def recurrent_fn(params, rng_key, action, hidden):
    reward, next_hidden, discount = dynamics(params, hidden, action)
    policy_logits, value = prediction(params, next_hidden)
    return (
        mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=policy_logits,
            value=value,
        ),
        next_hidden,
    )
```

Implementation rules:

- `recurrent_fn` must be pure JAX: no real environment calls, Python object mutation, logging, host I/O, Modal calls, or NumPy work.
- `action` is an integer array of shape `[B]`. One-hot it inside JAX if the dynamics network wants action planes or action features.
- Return fixed-shape pytrees every time. Use masks and `jnp.where` for terminal/dead cases.
- Keep RNG use explicit. Mctx passes an `rng_key`; the v0 deterministic model can ignore it, but stochastic later variants should consume it consistently.
- Test shape assertions locally before Modal: bad shapes in `prior_logits`, `reward`, `discount`, or `value` will fail inside the compiled search path.

## Batching

Mctx is already batch-first. A CurvyZero search batch should be assembled as:

```text
B = number of searched ego decisions
A = fixed action count
obs_batch[B, ...]
ego_metadata[B]          # env id, player id, alive/done masks, replay ids
invalid_actions[B, A]    # usually all zeros for curvyzero-v0
```

Recommended first batching policy:

- Batch many independent ego decisions together: many environments, many seeds, and optionally both players as separate ego rows.
- Keep `B` fixed inside a compiled profile. Pad the last batch and mask inactive rows rather than compiling many shapes.
- Run Mctx once per decision batch, not once per environment.
- Convert the resulting `PolicyOutput.action[B]` back into joint environment actions outside Mctx.
- For self-play v0, search one focal ego at a time or search both players as independent rows. Opponent actions in the real environment can come from policy-only inference, previous checkpoints, random, or heuristic opponents.

The simultaneous-move part of CurvyZero is the modeling wrinkle. Standard MuZero/Mctx recurrent dynamics receives only the searched action, not a joint action. For v0 this is acceptable if the learned dynamics is interpreted as predicting the next ego state under the opponent policy distribution present in the replay data. That is a smoke-test formulation, not a final multiplayer theory.

Alternatives to keep in reserve:

| Approach | Action count | Pros | Why not first |
| --- | ---: | --- | --- |
| Ego-only action | `A=3` | Small, matches Mctx cleanly, fastest benchmark. | Opponent behavior is folded into learned dynamics. |
| 1v1 joint action enumeration | `A=9` | Deterministic transition can condition on both players. | Search proposes joint actions, so opponent choice semantics become awkward. |
| Full n-player joint action | `A=3^N` | The tree sees all simultaneous choices. | Branching explodes immediately. |
| Stochastic MuZero | decision/chance split | Better for stochastic opponents, bonuses, or trail gaps. | More machinery than the first integration needs. |

## Action Spaces

Mctx assumes a fixed discrete action space. CurvyZero already has a compatible v0 action contract:

```text
0 = turn_left
1 = straight
2 = turn_right
```

The two-action left/right variant is also compatible, but it should be a separate ruleset because it changes trajectory semantics, replay meaning, and compiled shape. Do not switch `A` inside one run. If a later ruleset adds actions, compile and benchmark it as a separate profile.

For illegal actions:

- `curvyzero-v0` probably has no illegal root actions, so `invalid_actions = zeros[B, 3]`.
- If dead/padded rows appear in a fixed batch, either avoid searching them or give them a safe all-valid mask and ignore the output with an outer active-row mask.
- Do not create variable-length legal-action lists. Use the Mctx `[B, A]` invalid-action mask.

## Root Priors, Values, And Targets

At the root:

- `prior_logits` are unnormalized policy logits from the prediction head.
- `value` is the model's raw value estimate from the ego player's perspective.
- `embedding` is the latent state to store in the tree and pass to `recurrent_fn`.

After search:

- `policy_output.action` is the selected ego action.
- `policy_output.action_weights` is the policy-improvement target for the policy head.
- `policy_output.search_tree.summary()` can provide root visit probabilities, root value, and Q-values for diagnostics and target construction.

Store in replay:

```text
action
root_action_weights[A]
root_value_search
root_value_raw
legal_or_invalid_action_mask[A]
search_config_hash
model_step_used_for_search
```

Training should treat `action_weights` as a stopped-gradient target for policy cross-entropy. Value and reward targets should still come from environment rewards plus bootstrapping, with search value recorded for diagnostics and possible target variants. Keep raw model value and search value both visible so search improvement can be measured.

For terminal predictions inside the recurrent model, use `discount=0` for predicted terminal states and `gamma` otherwise. For the first synthetic benchmark, a constant discount is enough because the goal is measuring search mechanics.

## Gumbel MuZero Relevance

Mctx's README recommends `gumbel_muzero_policy`, and the Mctx source identifies it as Full Gumbel MuZero from "Policy improvement by planning with Gumbel." The paper targets better policy improvement with few simulations, which matters here because CurvyZero may need many real-time decisions and many ego rows per batch.

Use Gumbel MuZero as the default benchmark and integration path:

```python
policy_output = mctx.gumbel_muzero_policy(
    params=params,
    rng_key=rng_key,
    root=root,
    recurrent_fn=recurrent_fn,
    num_simulations=num_simulations,
    invalid_actions=invalid_actions,
    max_depth=max_depth,
    max_num_considered_actions=3,
    gumbel_scale=1.0,
)
```

For `A=3`, `max_num_considered_actions=3` is natural. Use `gumbel_scale=0.0` for deterministic evaluation sweeps if needed. Also run one `muzero_policy` comparison in the synthetic benchmark to separate API compatibility from Gumbel-specific behavior.

Stochastic MuZero is not the first CurvyZero path. Revisit it if:

- trail gaps, bonuses, or spawn randomization are modeled inside search rather than only in the real environment;
- opponent actions need explicit chance-node modeling;
- ego-only deterministic dynamics produces unstable planning behavior.

## JIT And Static Shapes

JAX and Mctx reward boring shapes. The compiled outer function should include root construction and the policy call:

```python
@functools.partial(
    jax.jit,
    static_argnames=("num_simulations", "max_depth", "policy_kind"),
)
def run_search(params, rng_key, obs_batch, invalid_actions, *,
               num_simulations, max_depth, policy_kind):
    root = make_root(params, obs_batch)
    if policy_kind == "gumbel":
        return mctx.gumbel_muzero_policy(
            params, rng_key, root, recurrent_fn,
            num_simulations=num_simulations,
            max_depth=max_depth,
            invalid_actions=invalid_actions,
            max_num_considered_actions=3,
        )
    return mctx.muzero_policy(
        params, rng_key, root, recurrent_fn,
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=invalid_actions,
    )
```

Shape policy:

- Static/compile-time: `num_simulations`, `max_depth`, `A`, model architecture, hidden shape, observation shape, and usually batch size.
- Dynamic/runtime arrays: params, RNG key, observation batch values, invalid-action mask values.
- Keep a small set of benchmark profiles, for example `(B, simulations) = (64, 16), (256, 32), (1024, 32)`.
- Pad incomplete batches. Avoid recompiling for tail batches, player-count variants, or action-count variants.
- Use `jax.lax.cond`, `jax.lax.fori_loop`, `jax.lax.while_loop`, masks, and `jnp.where` instead of Python control flow on traced values.
- Benchmark with `.block_until_ready()` because JAX dispatch is asynchronous.
- Report compile time separately from steady-state runtime.

JAX's persistent compilation cache may help repeated Modal runs if the same shapes are used. Configure it before the first compilation and put it on a local or mounted path that is appropriate for the run. This is an optimization after the initial measurement, not a prerequisite.

## Modal GPU Implications

For the first dependency smoke, use the cheaper contained module before the
benchmark profile below:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_gpu_dependency_smoke
```

That smoke pins `mctx==0.0.6` and `jax[cuda12]==0.7.0`, requests `["L4", "T4"]`,
and runs only one tiny `B=4`, `A=3`, 4-simulation `gumbel_muzero_policy` call.
It answers "can Modal import JAX/Mctx and execute Mctx search on a cheap GPU?",
not "is the real benchmark fast enough?"

Recorded 2026-05-09 evidence:

- Dependency smoke passed on Modal GPU.
- Tiny synthetic benchmark passed on Modal L4 with `B=8`, `num_simulations=4`,
  `hidden_dim=32`, and `max_depth=4`.
- Larger synthetic benchmark passed on Modal L4, app
  `ap-ULhQNpnV6a1lsn0uQLUbnX`, with `B=64`, `num_simulations=16`,
  `hidden_dim=64`, `max_depth=16`, `compile_plus_first_run_sec =
  8.080801095000002`, `steady_median_sec = 0.005292786999998356`,
  `decisions/sec median = 12091.928127850202`, and
  `simulations/sec median = 193470.85004560323`.
- The larger run reported GPU backend `gpu/cuda:0` and finite normalized
  `action_weights`.

Use Modal GPU jobs for future Mctx benchmark sweeps once the local CPU smoke
passes. Modal's current CUDA docs say GPU functions have the NVIDIA driver and
CUDA Driver API installed, and JAX's install docs currently recommend CUDA 13
wheels while still listing CUDA 12 as an alternative. For a fresh Modal image,
start with:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda13]",
        "mctx",
        "numpy",
    )
)
```

If the image or GPU class has CUDA compatibility issues, fall back to `jax[cuda12]` and record it in the benchmark artifact.

GPU choice:

- Start with `L40S` if available because it has enough memory for meaningful tree batches and is a reasonable cost/performance debugging target.
- Use `A100` or `H100` only after L40S results show the benchmark is GPU-bound and memory fits.
- For precise H100 benchmarking, Modal documents `H100!` to avoid automatic H200 upgrades.
- Do not start with multi-GPU. Mctx single-search batching is the first bottleneck to understand.

Runtime shape:

- Keep Mctx search, model inference, and any trainer step inside one Modal container/process group.
- Do not call Modal Queues, Dicts, remote Functions, or storage from inside `recurrent_fn` or per MCTS simulation.
- Use Queues only for coarse work dispatch, Dicts only for small metadata such as latest checkpoint pointers, and Volumes or bucket-backed storage for checkpoints/replay chunks.
- Watch GPU utilization, power, memory, compile time, and steady-state searches/sec. Modal's GPU metric docs warn that utilization alone does not measure FLOPS or memory-bandwidth saturation.
- Write one compact JSONL/CSV result per run, not thousands of tiny files.

## Smallest Synthetic Benchmark

Implement one script before any real MuZero self-play integration:

```text
scripts/benchmark_mctx_synthetic.py
```

The script should:

1. Import `jax`, `jax.numpy as jnp`, and `mctx`; print `jax.devices()`.
2. Build deterministic synthetic params for a tiny model:
   - observation shape: `obs[B, obs_dim]`, start with `obs_dim=32`;
   - hidden shape: `hidden[B, hidden_dim]`, start with `hidden_dim=64`;
   - action count: `A=3`;
   - representation: one linear/tanh projection;
   - dynamics: action embedding plus hidden MLP block, returns reward/value/logits;
   - prediction: small linear heads for logits and value.
3. Create `RootFnOutput` and `recurrent_fn` exactly as the real integration will.
4. JIT an outer `run_search` function with static `num_simulations`, `max_depth`, and `policy_kind`.
5. Sweep a tiny matrix:
   - `B`: `64`, `256`, `1024`;
   - `num_simulations`: `16`, `32`, `64`;
   - `policy_kind`: `gumbel`, plus one `muzero` comparison at `B=256`, `num_simulations=32`.
6. For each profile, report:
   - compile plus first-run wall time;
   - median steady-state wall time over at least 20 runs;
   - decisions/sec: `B / seconds`;
   - simulations/sec: `B * num_simulations / seconds`;
   - selected action histogram;
   - `action_weights` finite check and row-sum check;
   - peak GPU memory if easy to read, otherwise Modal GPU memory metric and `nvidia-smi` snapshot.
7. Use `.block_until_ready()` on `policy_output.action_weights` or a tree summary value before stopping timers.
8. Write one result file, for example:

```text
artifacts/mctx_synthetic/YYYYMMDD-HHMMSS/results.jsonl
```

Current pass/read criteria for moving toward real CurvyZero observations:

- The script runs locally on CPU for one tiny profile.
- The Modal GPU run sees a GPU in `jax.devices()`; this passed on L4 for
  `B=64`, `num_simulations=16`, `hidden_dim=64`, `max_depth=16`.
- `gumbel_muzero_policy` completes all sweep points without recompilation surprises other than the intentional profile changes.
- Steady-state timing is separated from compile timing.
- `action_weights` are finite and sum to approximately 1 per active row; this
  passed in both recorded Modal synthetic benchmark profiles.
- Tree memory is understood well enough to pick an initial real hidden shape and batch size.

Do not include the real simulator, replay buffer, or trainer in this benchmark. The benchmark's job is to answer whether Mctx plus JAX plus Modal can execute CurvyZero-shaped batched search.

## Integration Sequence

1. Add optional dependencies for the JAX/Mctx spike only after the synthetic script is ready to run.
2. Build the synthetic benchmark and run it on CPU.
3. Wrap the synthetic benchmark in a Modal GPU function.
4. Pick initial search profile from the benchmark. The first recorded usable
   L4 profile is `B=64`, `num_simulations=16`, `hidden_dim=64`, `max_depth=16`,
   `A=3`; do not jump to `B=256`, `num_simulations=32` until a separate sweep
   shows memory and compile behavior are still boring.
5. Implement `SearchPolicy` around Mctx with the same root/recurrent signatures.
6. Feed real CurvyZero observations through a toy randomly initialized model, still without training.
7. Add replay fields for `action_weights`, raw value, search value, model step, rules hash, observation hash, reward hash, and search config hash.
8. Only then connect training targets and self-play.

## Risks

| Risk | Mitigation |
| --- | --- |
| Hidden states make the tree too large. | Benchmark vector hidden first; add spatial hidden only after memory measurements. |
| JAX recompiles constantly. | Fixed profiles, padded batches, static configs, and no shape-changing action/rules variants. |
| GPU sits idle because batches are too small. | Sweep `B` and simulations; measure steady-state with `.block_until_ready()`. |
| Mctx action model hides simultaneous opponent choices. | Start ego-only for smoke; revisit stochastic or opponent-aware dynamics after baseline evidence. |
| Modal network primitives enter the hot loop. | Keep all MCTS/model work in one container; use Modal primitives only around coarse jobs/artifacts. |
| Synthetic benchmark overstates real throughput. | Add one follow-up benchmark with real observation tensors and realistic hidden shape before serious self-play. |

## Open Questions

- What hidden shape is the best first real model: vector latent, small spatial latent, or both behind a compile profile?
- Should the first searched self-play run search both players independently or search one focal player while the opponent is policy-only?
- Is the ego-only dynamics approximation good enough for 1v1, or do we need stochastic/opponent-conditioned dynamics earlier?
- What steady-state simulations/sec threshold makes Mctx worth integrating before a stronger PPO baseline?
- Should the persistent JAX compilation cache live on a Modal Volume for repeated benchmark jobs, or is compile time acceptable at this stage?

## Sources

Primary external sources:

- Mctx GitHub README: https://github.com/google-deepmind/mctx
- Mctx base types: https://github.com/google-deepmind/mctx/blob/main/mctx/_src/base.py
- Mctx policies: https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py
- Mctx search implementation: https://github.com/google-deepmind/mctx/blob/main/mctx/_src/search.py
- Mctx tree data structure: https://github.com/google-deepmind/mctx/blob/main/mctx/_src/tree.py
- Gumbel MuZero paper page: https://openreview.net/forum?id=bERaNdoegnO
- MuZero paper page: https://arxiv.org/abs/1911.08265
- JAX installation docs: https://docs.jax.dev/en/latest/installation.html
- JAX `jit` docs: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
- JAX `vmap` docs: https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html
- JAX JIT guide: https://docs.jax.dev/en/latest/jit-compilation.html
- JAX sharp bits, dynamic shapes: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#dynamic-shapes
- JAX benchmarking guide: https://docs.jax.dev/en/latest/benchmarking.html
- JAX persistent compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- Modal GPU docs: https://modal.com/docs/guide/gpu
- Modal CUDA docs: https://modal.com/docs/guide/cuda
- Modal GPU metrics: https://modal.com/docs/guide/gpu-metrics
- Modal Queues: https://modal.com/docs/guide/queues
- Modal Dicts: https://modal.com/docs/guide/dicts
- Modal Volumes: https://modal.com/docs/guide/volumes

Local CurvyZero sources:

- `docs/research/muzero_architecture_deep_dive.md`
- `docs/research/multiplayer_selfplay_muzero.md`
- `docs/research/performance_vectorization.md`
- `docs/design/training_architecture.md`
- `docs/design/modal_architecture.md`
- `docs/design/rulesets.md`
- `docs/decisions/0002-modal-hot-loop-locality.md`
- `curvyzero_env/config.py`
- `curvyzero_env/core.py`
