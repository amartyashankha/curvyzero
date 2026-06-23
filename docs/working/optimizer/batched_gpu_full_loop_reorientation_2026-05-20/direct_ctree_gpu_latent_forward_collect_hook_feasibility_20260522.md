# Direct CTree GPU-Latent Forward-Collect Hook Feasibility

Date: 2026-05-22

Status: read-only feasibility note. No code, live runs, trainer defaults,
checkpoints, or Modal volumes changed.

## Plain Read

The smallest useful next step is a profile-only patch around
`MuZeroPolicy._forward_collect` while the existing stock `train_muzero` run is
active.

This is feasible because the stock MuZero collectors do not require a special
collector object. They call `policy.forward(...)`, then consume a per-env-id
dict. If the patched `_forward_collect` returns the same dict fields in the
same shapes, stock collector, `GameSegment`, replay, target building, RND hooks,
and learner stay unchanged.

The patch should be profile-gated first. It should not become a production
default until a matched stock full-loop A/B proves that wall-clock training
throughput improves and replay rows stay identical in shape and meaning.

## Required Return Contract

For the normal MCTS collect path, `_forward_collect(...)` must return:

```text
{
  env_id: {
    "action": scalar_full_action_id,
    "visit_count_distributions": raw_legal_action_visit_counts,
    "visit_count_distribution_entropy": entropy_from_stock_select_action,
    "searched_value": root_search_value,
    "predicted_value": inverse_scaled_initial_value,
    "predicted_policy_logits": full_action_policy_logits,
  },
  ...
}
```

Important details:

- The outer keys must be the exact `ready_env_id` values passed by the
  collector.
- `action` must be a legal full action id, not an index inside the legal-action
  subset.
- `visit_count_distributions` must be the raw visit-count list returned by
  CTree for legal actions only, in the same order as stock
  `legal_actions = [i for i, x in enumerate(action_mask[row]) if x == 1]`.
  Do not return normalized full-width `[A]` arrays here.
- `searched_value` must come from `roots.get_values()[row]`.
- `predicted_value` must be the initial model value after
  `inverse_scalar_transform_handle(...)`, as stock does.
- `predicted_policy_logits` must be the initial model policy logits over the
  full action space.
- `timestep` is not required; both stock collectors default it to `-1` if
  absent.

The stock collectors then do:

- `actions[env_id] = output[env_id]["action"]`
- `store_search_stats(visit_count_distributions, searched_value, ...)`
- `append(action, next_observation, reward, previous_action_mask, previous_to_play, ...)`

So replay stays stock only if those meanings stay stock.

## Fallback And Fail-Closed Rules

Call the original stock `_forward_collect` for unsupported modes:

- `collect_with_pure_policy=True`
- `mcts_ctree=False`
- `model.model_type == "conv_context"`
- `sampled_algo=True` or `gumbel_algo=True`
- non-discrete/continuous action settings
- missing `_collect_model`, missing `_mcts_collect`, or missing CTree APIs
- CPU/non-CUDA data or model when the selected backend is explicitly
  `direct_ctree_gpu_latent`
- action-space size not matching the fixed CurvyTron action count expected by
  the helper

Fail closed, rather than silently falling back, for profile invariants that
would make an A/B row misleading:

- action-mask batch length does not match `data.shape[0]`
- an action mask has zero legal actions
- masks are non-binary when this profile says it is testing stock LightZero
  legal-action semantics
- `ready_env_id` length/order cannot be mapped one-to-one to the batch rows
- output action is illegal or visit-count length does not match legal-action
  length

In other words: fallback for unsupported LightZero modes, hard fail for a
supposed supported profile row that would hide a semantic mismatch.

## Reuse Or Factor The Existing Helper

Do not import
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
from the training hook just to reuse `_run_direct_ctree_gpu_latent_search`.

Reason: that module is a Modal sidecar and imports Modal, renderer benchmark
helpers, vector env code, JAX render utilities, and profile-only configuration.
Pulling it into the stock training launcher would blur the boundary and create
unwanted dependencies.

Better shape:

1. First implementation can keep a small local helper inside
   `lightzero_curvyzero_stacked_debug_visual_survival_train.py`, because this is
   profile-only and reversible.
2. If the full-loop A/B wins, factor the helper into a pure non-Modal module,
   for example under `curvyzero.training`, with no Modal imports and no renderer
   imports.
3. The helper can be copied from the current profile method conceptually:
   keep a GPU latent pool, use CTree `batch_traverse`, gather leaf latents on
   device, call `model.recurrent_inference`, read only reward/value/policy
   arrays back to CPU for CTree expansion/backprop, then mutate the same
   `roots` object.

The patched `_forward_collect` should still own stock-shaped output assembly,
because the existing profile helper returns compact arrays and normalized
full-action visits, which are not the collector contract.

## Minimal Local Tests Before Remote Full-Loop Profile

1. Hook lifecycle:
   install then restore the hook and assert `MuZeroPolicy._forward_collect` is
   back to the original object.

2. Output schema:
   run a tiny real or fake MuZero policy through patched collect and assert
   exact keys, `ready_env_id` preservation, legal actions, finite values, and
   legal-action visit-count lengths.

3. Stock fallback:
   verify original `_forward_collect` is called for pure-policy, non-CTree,
   CPU/non-CUDA, `conv_context`, sampled, and Gumbel cases.

4. Semantic canaries:
   forced single-legal-action masks produce one legal action and one visit row;
   clear biased logits choose the same top legal action as stock; illegal visit
   mass is zero after mapping.

5. Statistical comparison:
   stock versus patched direct GPU-latent over small fixed batches/seeds should
   have zero illegal actions and matching action/value/visit distributions for
   deterministic or clear-preference cases. Neutral/tie exact parity should not
   be the gate.

6. Full-loop smoke readiness:
   a local or tiny remote profile must still report `called_train_muzero=true`,
   one collector pass, one replay sample, one learner call, and unchanged replay
   row shapes.

## Silent Semantics Risks

Biggest ways this hook could accidentally change training:

- Returning normalized full-action visit probabilities instead of raw
  legal-action visit counts. That would directly change policy targets.
- Treating the selected action as a legal-subset index instead of a full action
  id.
- Mixing Dirichlet root noise differently from stock, especially under legal
  masks.
- Sampling actions with a different temperature, epsilon, or RNG order.
- Changing `to_play` ordering or `ready_env_id` ordering.
- Using recurrent reward/value transforms that differ from
  `MuZeroMCTSCtree.search`.
- Dropping `model.eval()` or changing `torch.no_grad()` behavior.
- Handling all-zero or fractional masks differently without saying so.
- Accidentally enabling the hook in eval mode, pure-policy mode, RND reward
  estimation, or production train mode.
- Importing the Modal profile module and bringing renderer/profile side effects
  into the training launcher.

## Recommended Next Experiment

Implement the hook only behind an explicit profile-only flag, then run one
matched full-loop A/B:

```text
A: stock train_muzero collect/search
B: stock train_muzero collect path with direct_ctree_gpu_latent _forward_collect hook
```

Keep the same seed, env variant, collector count, batch size, simulations,
RND setting, death/autoreset setting, checkpoint/eval/GIF sidecar settings,
and code image. The pass criterion is full-loop throughput improvement with
unchanged stock replay semantics, not another profile-only roots/sec win.
