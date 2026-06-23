# Dense Torch Fixed-Shape Audit

Scope: `_LightZeroArrayCeilingStackProbe.run` and `_run_dense_torch_mcts` in
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`.
I did not edit production code or touch live runs.

## Verdict

The fixed-shape patch removes the worst dynamic boolean slicing inside the
all-active dense search case, but it does not make sim16 likely to improve on
its own. The hot path is still an eager triangular selection/backprop loop with
dynamic integer gather/scatter, many per-depth tensor allocations, and per-call
tree/latent allocation. I also see two semantic issues the current focused test
does not cover: reward backup stores the wrong value, and partial-mask root
noise does not preserve the configured noise weight.

## Findings

1. **Backprop appears to store child bootstrap instead of edge-backed value.**
   Selection treats `edge_value_sum / visits` as Q at
   `source_state_batched_observation_boundary_profile.py:6596-6600`, but backup
   adds `bootstrap` into `edge_value_sum` at `:6699-6700` before computing
   `edge_reward + discount_factor * bootstrap` at `:6701-6704`. Root values are
   then assembled from the same unbacked sums at `:6716`. With nonzero
   recurrent reward/value-prefix, this omits the immediate edge reward and
   discount from the edge Q used by PUCT. The dense unit test uses zero reward
   fakes (`tests/test_source_state_batched_observation_boundary_profile.py:1998-1999`)
   and only asserts shape/legal actions (`:2053-2062`), so it will not catch this.
   Next fix: compute `backed_value = reward + discount * bootstrap` first, add
   `backed_value` to `edge_value_sum`, update min/max from `backed_value`, then
   carry `bootstrap = backed_value`. Add a tiny nonzero-reward fake-model test.

2. **Partial-mask root noise changes the effective noise weight.** Dense samples
   a full `ACTION_COUNT` Dirichlet at
   `source_state_batched_observation_boundary_profile.py:6520-6527`, zeros
   illegal actions in the mixture at `:6528-6531`, then renormalizes at `:6532`.
   For two legal actions out of three, the legal noise mass is usually below
   one, so the configured `root_noise_weight` is diluted and random. Direct
   CTree builds legal-action-only noise at `:5097-5106`. Next fix: mask the
   sampled noise, normalize the masked noise over legal actions, then mix with
   `(1 - w) * prior + w * legal_noise`.

3. **The path is only fixed-shape after active-root compaction.** `run()` still
   filters zero-mask roots with `flat_stack_all[legal_root_mask]` and
   `flat_mask_all[legal_root_mask]` at
   `source_state_batched_observation_boundary_profile.py:6073-6091`, and returns
   a zero-root telemetry fast path at `:6094-6104`. That is fine for no-death
   profile rows, but terminal/autoreset rows will still change shape and use CPU
   boolean compaction before dense search. Next fix: carry `total_root_count`
   plus a `root_active_mask` through dense mode, with a dummy safe action for
   inactive roots and masked output assembly.

4. **sim16 slowdown is structurally plausible.** The dense loop performs
   selection over `range(simulation_index + 1)` at
   `source_state_batched_observation_boundary_profile.py:6595`, and backprop over
   the same triangular depth count at `:6694`. That is 36 depth iterations at
   sim8 versus 136 at sim16, before counting the per-depth advanced indexing,
   `torch.where`, `masked_fill`, `argmax`, and scatter updates at `:6596-6625`
   and `:6695-6707`. Removing boolean slices does not remove this eager kernel
   launch/allocation shape. Next fix: either compile/fuse the fixed-shape search
   body, flatten gather/scatter indices into reusable buffers, or keep pursuing
   the direct CTree GPU-latent lane for sim16.

5. **Dense mode still pays for an unused root policy decode.** `run()` computes
   `_masked_policy_arrays(...)` at
   `source_state_batched_observation_boundary_profile.py:6125-6137` before
   calling dense mode, then `_run_dense_torch_mcts` recomputes root priors from
   `root_policy_logits` at `:6514-6519`. Next fix: skip the pre-dense
   `_masked_policy_arrays` work when `mode == dense_torch_mcts`; dense only needs
   logits, root value, and latent state.

6. **Phase telemetry can hide CUDA sync debt.** Dense intentionally avoids
   per-recurrent-call sync, but `recurrent_inference_sec` at
   `source_state_batched_observation_boundary_profile.py:6636-6652` and
   `search_update_sec` at `:6654-6708` are enqueue timings. The actual device
   drain happens at `:6718` and final numpy readback at `:6242-6245`. Next fix:
   use CUDA events or a diagnostic sync mode before deciding which subphase
   regressed.

7. **Mask validation is weaker than the direct collect path.** Array-ceiling
   masks are cast to float and treated as legal when `> 0.0` at
   `source_state_batched_observation_boundary_profile.py:6065-6075` and `:6514`,
   while collect-forward rejects non-binary masks at `:4498-4500`. Next fix:
   reject fractional positive masks in array-ceiling too, unless this probe is
   deliberately accepting soft masks.

8. **Zero-simulation root value fallback is dead.** `visit_totals` is clamped at
   `source_state_batched_observation_boundary_profile.py:6713`, so the fallback
   condition at `:6717` is always true. This is low priority for sim8/sim16, but
   the fix is to keep an unclamped `raw_visit_totals` for the condition.

## Next Fix Order

1. Fix reward/discount backup and partial-mask root noise first; these are
   semantic and can be covered with small CPU/GPU fake-model tests.
2. Reprofile sim8/sim16 only after skipping the unused pre-dense policy decode
   and adding CUDA-event timing, so the denominator tells us where time moved.
3. If sim16 still trails direct CTree GPU-latent, stop polishing eager Python
   Torch and move to a compiled/fused search body or the CTree GPU-latent path.
4. For any terminal/death profile claim, remove the active-root compaction and
   keep inactive roots in a masked fixed-shape tensor all the way through.
