# Direct CTree GPU-Latent Train Hook Shape

Date: 2026-05-22

Status: doc-only. Do not mutate live runs, checkpoints, Modal volumes, trainer
defaults, or production launch defaults from this note.

## Current Block

Coach still enters stock LightZero. In
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`,
the launcher imports `lzero.entry`, chooses `train_muzero` or the RND wrapper,
builds CurvyTron configs, installs reversible hooks, then calls:

```python
output = train_muzero(
    [patched["main_config"], patched["create_config"]],
    seed=seed,
    max_train_iter=max_train_iter,
    max_env_step=max_env_step,
)
```

Exact current block: import/entrypoint at lines `5338-5343`, config build at
`5344-5375`, hook installation at `5595-5714`, and the call above at
`5719-5730`.

The config builder is not a search-backend selector. In
`src/curvyzero/training/lightzero_config_builder.py`,
`_build_visual_survival_configs_from_builder_kwargs(...)` copies the stock Atari
MuZero config and patches stock knobs: `policy.collector_env_num`,
`policy.num_simulations`, `policy.batch_size`, model shape/action size, env
metadata, reward targets, opponent config, and RND config (`1160-1475`). There is
no field for `direct_ctree_gpu_latent` or any alternate collect/search backend.

The installed LightZero collect path does the expensive crossing before MCTS:
`.venv/lib/python3.11/site-packages/lzero/policy/muzero.py` converts
`latent_state_roots` to CPU NumPy at lines `739-743`, builds CTree roots at
`745-759`, then calls `self._mcts_collect.search(...)` at `760`. The stock
collector consumes the returned dict fields directly:
`action`, `searched_value`, `predicted_value`, `visit_count_distributions`, and
`visit_count_distribution_entropy`
(`.venv/.../lzero/worker/muzero_collector.py:448-467` and
`muzero_segment_collector.py:474-493`).

The fast row is separate profile code:
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
declares `direct_ctree_gpu_latent` at `166-180`. Its `_run_direct_mcts_arrays`
path calls real LightZero CTree and delegates to
`_run_direct_ctree_gpu_latent_search(...)` when `keep_latents_on_device=True`
(`5032-5147`). That helper keeps a GPU latent pool and only reads CPU
reward/value/policy arrays for CTree expansion/backprop (`5542-5722`). It is
profile-only and returns compact arrays, not stock collect-mode output.

## Candidate Hook Points

1. Preferred smallest full-loop-profile hook:
   patch `MuZeroPolicy._forward_collect` during `mode="profile"` inside the
   existing launcher hook window. Keep stock `train_muzero`, stock collector,
   stock GameSegment/replay/target builder, RND hooks, checkpoint hooks, and env
   managers. The patched collect method should clone stock collect semantics but
   skip only the root latent CPU copy and call a shared GPU-latent CTree helper.

2. Smaller-looking but insufficient hook:
   patch `MuZeroMCTSCtree.search`. This is too late because stock
   `_forward_collect` already converted root latents to CPU NumPy before calling
   `search`.

3. Larger hook:
   replace collector/env-manager with the batched profile manager plus the
   direct search hook. This can produce a full-loop profile denominator with
   `called_train_muzero=true`, but it changes collection topology and should
   stay profile-only.

4. Trainer replacement:
   keep compact arrays through replay/targets or bypass stock GameSegment
   construction. That stops being stock LightZero collector/replay semantics and
   should be labeled a replacement trainer.

## Minimal Patch Shape

Add an opt-in search backend flag, default `stock`, and reject it outside
`mode="profile"` until parity gates pass:

```text
--collect-search-backend stock|direct_ctree_gpu_latent
```

Install a reversible hook next to the existing profiler hooks:

```text
_install_lightzero_collect_search_backend_hook(
    train_muzero=train_muzero,
    backend="direct_ctree_gpu_latent",
    profiler=profiler,
)
```

Patch `lzero.policy.muzero.MuZeroPolicy._forward_collect` only for classic
MuZero CTree collect mode:

- fall back to the original method unless `model_type in {"conv", "mlp"}`,
  `mcts_ctree=True`, `collect_with_pure_policy=False`, and CUDA/device support is
  present;
- run `initial_inference(data)` as stock does;
- keep `latent_state_roots` as a tensor, but still CPU-read `pred_values` and
  `policy_logits` because LightZero roots need them;
- build `legal_actions`, Dirichlet noises, and `roots.prepare(...)` exactly like
  stock;
- call a shared helper factored from `_run_direct_ctree_gpu_latent_search(...)`
  that mutates `roots`;
- after search, use `roots.get_distributions()` and `roots.get_values()` and the
  same `select_action(...)` logic as stock;
- return the same per-env dict as stock, including raw legal-action visit counts,
  not the profiler's normalized full-action compact arrays.

That is a small patch for a full-loop profile canary. Promoting it to train mode
is a small train-facing policy/search extension only after parity gates pass. It
is not a config-only patch.

## Risks

- Private API risk: LightZero version, `MuZeroPolicy._forward_collect`, CTree
  `tree_muzero`, and root APIs are not stable public extension points.
- Semantic drift risk: the profiler compact arrays normalize visits and expand
  to full action width; stock collector wants raw legal-action visit counts.
- RNG/action risk: Dirichlet noise, epsilon, temperature, legal-action indexing,
  and `ready_env_id` mapping must match stock behavior.
- Coverage risk: `collect_with_pure_policy`, eval mode, `conv_context`,
  Gumbel/sample variants, and non-CTree should fall back to stock.
- Performance risk: recurrent outputs still copy reward/value/policy logits to
  CPU each simulation; this only removes the repeated latent CPU round trip.

## Tests And Gates

Static/local tests:

- config-builder test: new flag defaults to `stock` and appears in surface only
  as metadata; stock built configs are unchanged.
- hook lifecycle test: install/restore leaves `MuZeroPolicy._forward_collect`
  identical after restore.
- collect parity smoke on a tiny real `MuZeroPolicy`: same output keys, env ids,
  legal actions, legal visit lengths, no illegal actions, and finite values.
- fallback tests: pure-policy, non-CTree, unsupported model type, and CPU path
  call the original method.

Remote/profile gate:

- fresh `mode=profile`, no auto-resume, no volume commit, same code/image;
- stock control vs direct hook with identical `collector_env_num`,
  `num_simulations`, `batch_size`, reward/RND/opponent/death settings;
- `called_train_muzero=true` and at least one learner train call;
- replay/target audit sees the same LightZero fields and visit-count shape;
- wall-clock full-loop denominator improves, not only roots/sec.

## Classification

Best next step: small full-loop-profile hook.

Train mode: small policy/search extension only after parity gates.

Not recommended as the next step: compact-array replay or custom collector
training. That is a trainer replacement.
