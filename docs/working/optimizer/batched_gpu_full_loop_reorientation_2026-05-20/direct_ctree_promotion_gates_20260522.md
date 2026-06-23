# Direct CTree Promotion Gates

Date: 2026-05-22

Status: optimizer contract. Profile-only until every mandatory gate passes. Do
not touch live Coach training runs.

## Plain Goal

Historical best LightZero-shaped train-facing speed probe was:

```text
collect_search_backend=direct_ctree_gpu_latent + all-actions-legal output fast path
```

Update 2026-05-23j: this is no longer the top optimizer speed row. Real
checkpoint-backed MCTX/JAX shadow profiling has a faster profile-only row:
B1024/A16/sim8 scalar-off at `19,334` active steps/sec versus direct CTree
`8,792`, speedup `2.20x`. That row has semantic deltas and is not a promoted
Coach backend.

It is faster in matched profile rows, but it is not automatically safe for
Coach. The gate is not "did one profile row look fast?" The gate is "does it
keep the same training meaning while making the stock `train_muzero`
denominator faster?"

## Mandatory Gates

| gate | proof needed | why it matters |
| --- | --- | --- |
| hook lifecycle | install only inside `mode=profile`, restore after run, reject unsupported compute/policy modes | prevents accidental production mutation |
| stock output schema | same per-env collect dict fields as stock LightZero | replay and target builder must see the same shape |
| all-actions-legal fast path | deterministic selector parity and stochastic distribution check | output fast path must not choose actions differently by accident |
| mixed-mask fallback | one illegal action, one legal action, masked clear preference, zero illegal visit mass | avoids legal-list index/full-action id bugs |
| fail-closed masks | fractional masks and empty masks reject or are filtered upstream | stock LightZero uses `x == 1` legality |
| packed output transfer | row/action sentinels prove reward/value/logit unpack order | avoids silent row or column swaps |
| GPU-latent search parity | stock facade versus direct GPU-latent over sim1/sim2/sim8/sim16 | catches latent-pool/search drift |
| replay/target canary | action, visit counts, searched value, predicted value/logits, reward/done/mask metadata preserved | fast collect output must train the same targets |
| terminal/live canary | normal death/autoreset with terminal and live rows | no-death profiles are not enough |
| semantic attestation | run summary includes backend, fallback calls, env-step source, death/RND mode, eval/GIF/checkpoint state | keeps speed tables honest |
| matched full-loop A/B | at least two same-shape stock/direct profile repeats, `fallback_calls == 0` | proves the win survives the real denominator |
| RND canary when enabled | predictor changes, target stays frozen, latest-frame source is correct, target rewards unchanged for meter mode | RND can hide a separate wall or semantic bug |

## Exact Versus Statistical

Do not use exact neutral/tie-heavy action parity as the approval ruler. Stock
LightZero itself can allocate visits differently in neutral rows.

Use exact checks for:

- forced masks;
- clear preferences;
- illegal-action and illegal-visit guarantees;
- output schema;
- replay/target fields;
- value/logit equality where the same model outputs are being compared.

Use statistical checks for ordinary stochastic collect rows.

## Fresh Local Compare

These were local validation harness rows, not live training runs.

```text
sim8, 8 seeds, 8 roots/seed, root_noise_weight=0.25:
  direct_ctree_arrays:      exact action/visit/value/logit agreement
  direct_ctree_gpu_latent:  exact action/visit/value/logit agreement
  illegal actions:          0
  gpu-latent enabled rows:  8/8
```

Strict sim16 neutral/tie-heavy exact gate:

```text
sim16, 4 seeds, 8 roots/seed, root_noise_weight=0.0:
  direct_ctree_gpu_latent mean_action_agreement: 0.65625
  mean_visit_l1: 0.04296875
  max_visit_l1: 0.125
  max value/logit diffs: 0
  illegal actions: 0
  gpu-latent enabled rows: 4/4
```

Plain read:

```text
This does not prove the direct path is wrong. It proves strict action/visit
equality on tie-heavy rows is the wrong promotion gate. The values/logits and
illegal-action checks stayed clean, and the candidate really used the
GPU-latent path. Keep exact gates for forced cases and schema; use
distributional gates for ordinary collect rows.
```

## 2026-05-22 Harness Tightening

The local compare script now has explicit action-mask scenarios:

- `random`
- `all_legal`
- `single_legal_cycle`
- `mixed_legal_cycle`

It also has `--require-gpu-latent-enabled`, so a GPU-latent row cannot pass by
silently falling back to the non-latent path.

Fresh tiny forced-mask result:

```text
single_legal_cycle, sim4, seeds=2, batch_rows=2, root_noise_weight=0.0:
  direct_ctree_arrays: exact
  direct_ctree_gpu_latent: exact
  illegal actions: 0
  gpu-latent enabled rows: 2/2
```

Fresh neutral/mixed-mask strict result:

```text
all_legal and mixed_legal_cycle strict action/visit equality can fail while
values, logits, illegal-action checks, and GPU-latent activation stay clean.
```

Plain read:

```text
Single-legal and clear-preference rows are exact gates.
Neutral and mixed ordinary rows are statistical gates.
```

The train-facing hook also now has a local schema smoke: it patches the actual
`MuZeroPolicy._forward_collect` hook, uses fake stock outputs, and checks that
the direct path returns the same per-env keys and values without fallback. That
is not enough for promotion by itself; the next P0 is a stronger hook-vs-stock
parity test over forced masks and clear preferences with the actual installed
hook.

## 2026-05-22 Train-Hook Forced-Mask Gate

The actual train-facing hook now has local forced-mask coverage in
`tests/test_lightzero_phase_profiler.py`.

New checks:

- mixed masks such as `[1,0,1]`, `[0,1,0]`, and `[1,1,0]`;
- single-legal rows cycling full action ids `0`, `1`, and `2`;
- direct hook returns full action ids, not legal-list indices;
- `visit_count_distributions` stays as raw legal-action visit counts, not
  normalized full-width arrays;
- all-actions-legal output fast path is not used on mixed masks;
- fractional masks, zero masks, and ready-env-id length mismatch fail closed
  before model/search.
- all-actions-legal stochastic fast-path sampling matches the stock selector
  distribution within a loose empirical tolerance.

Validation:

```text
uv run pytest -q -p no:cacheprovider tests/test_lightzero_phase_profiler.py \
  -k "collect_search_hook"
-> 8 passed, 7 deselected

uv run ruff check --ignore F401 tests/test_lightzero_phase_profiler.py
-> passed

uv run python -m py_compile tests/test_lightzero_phase_profiler.py
-> passed
```

Remaining P0 validation gaps:

- hook output to stock replay/target material;
- row/column sentinel for GPU-latent gather and reward/value/logit unpack;
- broader replay/death/RND promotion gates before Coach-facing recommendations.

## 2026-05-22 Direct-Row Attestation Gate

The optimizer profile summarizer now treats direct backend rows as
under-attested unless they carry direct-hook self-audit fields.

Required for `collect_search_backend=direct_ctree_gpu_latent`:

- command and semantic identity agree on `collect_search_backend`;
- `collect_search_backend_direct_ctree_gpu_latent_calls > 0`;
- `collect_search_backend_output_rows > 0`;
- `collect_search_backend_fallback_calls == 0`.

Validation:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_summarize_curvytron_optimizer_profile_results.py
-> 3 passed

uv run ruff check \
  scripts/summarize_curvytron_optimizer_profile_results.py \
  tests/test_summarize_curvytron_optimizer_profile_results.py
-> passed

uv run python -m py_compile \
  scripts/summarize_curvytron_optimizer_profile_results.py \
  tests/test_summarize_curvytron_optimizer_profile_results.py
-> passed
```

## Promotion Rule

For Coach-facing use, require all mandatory semantic gates plus at least two
matched full-loop A/B repeats showing a stable win and:

```text
collect_search_backend_fallback_calls == 0
called_train_muzero == true
evaluator_eval_calls == 0
checkpoint/GIF sidecars disabled unless explicitly profiled
```

Until then, the direct path is an optimizer probe.
