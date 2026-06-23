# Direct CTree Promotion Contract

Date: 2026-05-21

Status: draft contract for a profile-only speed lane. This is not Coach launch
advice.

## Plain Summary

`direct_ctree_arrays` is the active optimizer speed probe because it removes
the public `collect_mode.forward` wrapper while still calling the real
LightZero MuZero model and real CTree MCTS.

It must stay profile-only until it passes the gates below.

The important correction:

```text
Exact neutral/tie-heavy visit parity is not a reliable gate.
```

A local diagnostic showed stock LightZero CTree does not repeat identical
neutral visit allocations even stock-vs-stock under fixed Python/NumPy/Torch
seeds. So neutral rows can catch obvious decode bugs, but they should not be
the production approval ruler.

## Required Gates

### P0: Exact Forced-Case Gates

These must pass exactly:

- action-mask mapping: legal-list indexes map back to full action ids;
- illegal actions: selected action is always legal;
- illegal visit mass: zero on masked actions;
- single legal action: exact one-hot visit distribution and selected action;
- masked preference: if the preferred action is illegal, choose the best legal
  fallback;
- clear preference: if one legal action is made obviously best, choose it;
- searched value shape and finite values;
- predicted value/logit debug arrays when LightZero returns them;
- `to_play=-1` fixed-opponent semantics;
- output schema and byte/count telemetry;
- replay target-row compatibility for non-one-hot visit distributions.
- binary action masks only. Stock LightZero builds legal actions with
  `x == 1`; the direct/profile boundary must reject fractional masks instead
  of silently treating any positive value as legal.

Current status: mostly covered locally. Remaining work is to add the missing
support-transform/root-noise/eval-mode checks explicitly if we want this to
be a full production contract.

### P1: Statistical Collect Comparison

Run stock facade and direct CTree over many roots/seeds with ordinary non-tie
policies.

Compare:

- action frequency by action id;
- visit distributions after masking and normalization;
- searched values;
- predicted values/logits where present;
- illegal action count;
- illegal visit mass;
- replay-facing fields.

This should be a distributional check, not a one-row exact equality check.

### P2: Matched Profile Rows

Run the same shape in the same harness:

- stock facade;
- direct CTree host `uint8`;
- direct CTree pinned `uint8`;
- resident stale-input ceiling, clearly labeled as a ceiling only.

Use enough warmup and measured steps to avoid short-row noise. The current
minimum useful shape is H100 B512/A16/sim8 with about `60` measured steps and
`15` warmup steps.

### P3: Trainer Reconnection Gate

Before any Coach-facing recommendation:

- same observation contract;
- same action-mask contract;
- same reset/death/autoreset behavior;
- same RND mode or no-RND control;
- same checkpoint/eval/tournament compatibility story;
- compact arrays converted into the correct replay/target rows;
- short matched full-loop profile against trusted stock `train_muzero`.

## Current Speed Read

Longer H100 direct rows:

```text
stock facade anchor:              ~2419.81 roots/sec
direct CTree host uint8, 60/15:   ~4111.80 roots/sec
direct CTree pinned uint8, 60/15: ~4513.15 roots/sec
resident stale ceiling, 60/15:    ~5537.40 roots/sec
```

Plain read: there is real headroom, but this is still a profile lane until the
contract passes.

Fresh current-telemetry refresh, same H100 B512/A16/sim8 60/15 shape:

```text
stock facade host_uint8:            ~2473.11 roots/sec
direct CTree fresh host_uint8:      ~4564.03 roots/sec
direct CTree fresh pinned_uint8:    ~4113.52 roots/sec
direct CTree resident stale ceiling: ~4884.69 roots/sec
```

Plain read: the robust speed signal is direct CTree versus the public stock
facade, roughly `1.85x` in the fresh host row. Input-copy removal is not the
main remaining prize; the stale resident ceiling is only modestly above fresh
host input in this refresh. Keep the promotion gate focused on semantics and
the remaining search/root-prep/model-output path.

## Next Local Work

1. Add the missing exact forced-case tests that are still only listed in docs:
   support transform, no-noise eval mode, and root-noise collect mode.
2. Keep the binary-mask precondition tested anywhere the LightZero boundary is
   used. Fractional-mask rows are synthetic profile bugs, not valid stock
   parity rows.
3. Use the new statistical comparison runner
   `scripts/compare_curvytron_direct_ctree_stock.py` for stock facade versus
   direct CTree over many roots/seeds. The first smoke passed with no illegal
   actions and exact searched values in a tiny no-noise sim2 case, while
   showing expected action/visit drift in low-simulation tie-ish rows.
   Follow-up sim4/root-noise-on validation over 4 seeds and 8 roots per seed
   gave exact action/visit/value agreement and zero illegal actions.
   Follow-up sim8/root-noise-on validation over 8 seeds and 8 roots per seed
   also gave exact action/visit/value agreement and zero illegal actions.
4. Re-run the same-shape H100 profile rows with current telemetry fields.
5. Only then consider a short matched full-loop A/B.

## 2026-05-22 Gate Update

The stricter promotion checklist now lives in
[direct_ctree_promotion_gates_20260522.md](direct_ctree_promotion_gates_20260522.md).

Fresh local compare result:

```text
sim8, 8 seeds, 8 roots/seed:
  direct_ctree_arrays and direct_ctree_gpu_latent matched stock exactly on
  actions, visits, values, predicted values, and logits.

sim16, 4 seeds, 8 roots/seed, root noise off:
  direct_ctree_gpu_latent had clean values/logits and zero illegal actions, but
  action/visit equality failed on neutral/tie-heavy rows.
```

Plain read: this reinforces the original contract. Exact action/visit parity
belongs to forced cases and clear-preference cases, not neutral stochastic
rows.
