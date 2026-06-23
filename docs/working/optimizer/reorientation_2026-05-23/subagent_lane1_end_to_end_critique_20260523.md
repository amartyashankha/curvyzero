# Lane 1 End-To-End Critique

Date: 2026-05-23

Scope: static review only. No live runs, tests, source-code edits, Modal state, or
checkpoint artifacts were touched.

## Short Verdict

The current patch is a practical narrow trainer path, not just another detached
profile row. `mode="train"` can request `collect_search_backend="direct_ctree_gpu_latent"`,
the launcher installs the hook before calling stock `train_muzero`, train mode
passes `allow_fallback=False`, and the final summary is marked not-ok when direct
collect calls are missing or fallback calls are observed.

I would call this "Lane 1 trainer-attached collect/search" rather than fully
proven "Lane 1 end-to-end" until the replay, target, and RND proof fields become
promotion gates instead of mostly passive artifacts.

## What Is Better Now

- Train-mode non-stock search is explicitly allowed only for GPU compute and
  LightZero CTree:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5590`.
- The hook install is in the real trainer setup path before `train_muzero`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6322`
  and `:6459`.
- Train mode is fail-closed because `allow_fallback=(mode == "profile")`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6327`.
- Hidden fallback now raises in train mode instead of silently calling the stock
  `_forward_collect`:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:2189`.
- The result adds `collect_search_backend_proof` and validates it for train:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:848`
  and `:886`.

## Top 5 Remaining Risks

1. The new local train test proves the launcher wiring, but not that real
   `train_muzero` consumes the actual monkeypatched `_forward_collect`.

   `tests/test_curvytron_live_checkpoint_eval_plumbing.py:1153` replaces
   `_install_lightzero_collect_search_backend_hook` with a fake installer that
   writes proof counters directly. That is useful for launcher plumbing, but it
   would still pass if the real monkeypatch stopped matching LightZero's actual
   policy class. The unit hook tests call `_forward_collect` manually
   (`tests/test_lightzero_phase_profiler.py:268`), not through the trainer.

   Needed: one local fake-LightZero trainer test that uses the real installer,
   then has fake `train_muzero` invoke `lzero.policy.muzero.MuZeroPolicy._forward_collect`
   on a minimal fake policy. Assert the hook, not the test, creates the proof
   counters.

2. The proof validates backend observation, but not enough completed-work fields.

   `_validate_train_collect_search_backend_proof` requires fallback policy,
   zero fallback calls, direct calls greater than zero, and an observed backend
   sample:
   `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:886`.
   It does not require `output_rows > 0`, `recurrent_inference_calls > 0`,
   `model_output_d2h_bytes > 0`, or observed CTree backend containing
   `lightzero`, even though those fields are already emitted at `:878`, `:879`,
   `:882`, and `:872`.

   Needed: tighten the train proof gate so a direct call alone is not enough.
   Require positive output rows, recurrent calls, D2H bytes, and observed CTree
   backend matching the requested LightZero CTree.

3. Target/replay proof is passive and does not affect `ok`.

   The target audit can record collector segments and replay samples
   (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:4981`),
   and it is installed in the trainer path at `:6420`. But its own caveat says
   it does not prove reward credit correctness or change learner inputs (`:5094`),
   and the launcher writes `target_audit_summary` without adding any target-audit
   failures to `problems` (`:6626` to `:6673`).

   Needed: for Lane 1 promotion, require a non-missing target audit in train
   mode with at least one collected GameSegment and, when learner sampling
   occurs, at least one replay sample. Add a stock-vs-direct audit comparing
   action, child visits, root value, action mask, and sampled policy target
   shape/order.

4. RND compatibility is optional unless `require_rnd_metrics=True`, and current
   tests do not combine RND with the direct collect backend.

   RND patching is separate from the collect-search hook:
   `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5216`.
   RND metrics become a failing gate only when both exploration bonus is enabled
   and `require_rnd_metrics` is true:
   `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6632`.
   The RND entrypoint test proves `train_muzero_with_reward_model` plumbing
   (`tests/test_curvytron_live_checkpoint_eval_plumbing.py:1068`), while the
   direct backend test proves non-RND hook plumbing (`:1132`). The combination
   remains unproven locally.

   Needed: one local direct-backend plus RND-meter train plumbing test requiring
   RND metrics, and one live-gated canary later that checks collect/train/estimate
   calls, predictor hash change, frozen target hash stability, and target reward
   delta semantics.

5. Train-mode telemetry still has profile-shaped holes.

   The generic phase profiler is installed only for profile mode or learner-call
   capped debugging:
   `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:842`.
   The direct hook records its own counts even when the profiler is disabled
   (`:1017` to `:1034`), but `train_muzero_wall_sec` is only timed in profile
   mode:
   `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6454`.
   Compact output computes `steps_per_sec` from that profile timer:
   `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:13462`.

   Needed: add a train-mode wall-time field to compact output, or explicitly
   label train-mode direct rows as semantic attachment proof rather than speed
   proof. Do not use this as a Coach speed claim until it has a clean train
   denominator and matched stock/direct rows.

## Promotion Bar

Call the current patch practical Lane 1 only with this narrower claim:

```text
stock train_muzero is invoked
direct_ctree_gpu_latent replaces collect/search in train mode
fallback is fail-closed
stock collector/replay/learner remain the consumer path
```

Do not call it fully proven end-to-end until the result also fails closed on:

```text
missing direct output rows
missing recurrent/D2H proof
missing observed LightZero CTree proof
missing target/replay audit
missing required RND metrics when RND is enabled
missing train-mode denominator for speed claims
```

