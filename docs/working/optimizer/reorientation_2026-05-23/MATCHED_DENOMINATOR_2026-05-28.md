# Matched Denominator Selection

Date: 2026-05-28

Status: selected, materialized, and captured remotely.

## Decision

Use one no-RND matched-denominator pair:

```text
denominator_id:
  curvytron-stock-vs-compact-owned-no-rnd-h100-20260528

stock role:
  stock_reference
  currency: stock_train_muzero_profile_env_steps_per_sec

compact role:
  compact_candidate
  currency: compact_profile_active_roots_per_sec
```

This pair is for honest orientation, not a promotion claim. The stock row calls
`train_muzero`; the compact row is the split compact-owned loop entrypoint and
still has `calls_train_muzero=false`.

## Materialized Manifests

Stock reference:

```text
artifacts/local/curvytron_optimizer_profile_manifests/optimizer-matched-denominator-stock-20260528.json
```

Shape:

```text
H100, stock train_muzero, C512/b64/sim8, seed 304
env_manager_type=subprocess
exploration_bonus_mode=none
source_max_steps=512
stop_after_learner_train_calls=12
disable_death_for_profile=true
detached profile-spawn result capture
```

Compact candidate:

```text
artifacts/local/curvytron_hybrid_observation_profile_manifests/optimizer-matched-denominator-compact-owned-20260528/manifest.json
```

Shape:

```text
H100, B1024/A16/sim8, 60 measured, 15 warmup
max_ticks=2000
root_noise=0.0
resident/device-only/native actor buffer
compact_torch_search_service
sample gate B512 every 8, replay capacity 4096
compact MuZero learner, train_steps=1, support_scale=300, num_unroll_steps=1
split compact-owned loop entrypoint
replay-store state capture
detached FunctionCall result capture
```

## Non-Claims

Do not divide the compact number by the stock number and call it Coach speedup.

The stock row measures stock `train_muzero` profile env steps/sec. The compact
row measures a profile-only compact loop boundary with active compact roots/sec
and compact learner-edge work. It does not own checkpoint/eval/tournament
semantics and does not enter stock LightZero Coach.

RND is explicitly off. Remote RND stock-profile raw collector-step verification
remains a separate P1 follow-up; compact RND parity is not selected here.

## Fail-Closed Checks Added

Stock manifest/runner checks now pin:

- denominator id, role, row purpose, counterpart ref, and `promotion_claim=false`;
- `collect_search_backend=stock`;
- `collect_search_ctree_backend=lightzero`;
- no-RND, H100, C512/b64/sim8, raw collector env-step currency;
- result payload fields proving `called_train_muzero=true`, profile mode,
  learner/sample calls, evaluator calls zero, and no fallback MCTS-root
  denominator.

Compact manifest/runner checks now pin:

- denominator id, role, row purpose, counterpart ref, and `promotion_claim=false`;
- split compact-owned entrypoint and replay-store capture;
- H100 B1024/A16/sim8, 60/15, `max_ticks=2000`, root noise `0.0`;
- resident/device-only/no-scalar compact Torch search;
- sample gate B512 every 8, support scale `300`, `num_unroll_steps=1`;
- nonterminal speed-row shape, positive compact speed fields, compact Torch
  backend, zero host fallback/observation H2D/replay payload D2H, and real
  compact MuZero updates.

## Run Commands

The captured pair was produced with:

```text
uv run python scripts/run_curvytron_optimizer_profile_manifest.py \
  --manifest artifacts/local/curvytron_optimizer_profile_manifests/optimizer-matched-denominator-stock-20260528.json \
  --collect-timeout-sec 7200

uv run python scripts/run_curvytron_hybrid_observation_profile_manifest.py \
  --manifest artifacts/local/curvytron_hybrid_observation_profile_manifests/optimizer-matched-denominator-compact-owned-20260528/manifest.json
```

## Captured Results

Stock reference result:

```text
artifacts/local/curvytron_optimizer_profile_results/optimizer-matched-denominator-stock-20260528/row_001_result.json
```

Result:

```text
status=complete
function_call_id=fc-01KSR704GN5HCTHX580YB7C41T
env_steps_collected=262144
env_steps_collected_source=collector_envstep_delta
steps_per_sec=642.9069738864584
steps_per_sec_currency=stock_train_muzero_profile_env_steps_per_sec
steps_per_sec_uses_fallback_denominator=false
learner_train_calls=12
replay_sample_calls=12
evaluator_eval_calls=0
```

Compact candidate result:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-matched-denominator-compact-owned-20260528/row_001_result.json
```

Result:

```text
status=complete
function_call_id=fc-01KSR75ZCTDJT7D54ZNTE05VVJ
steps_per_sec=23104.96230335242
physical_rows_per_sec=11552.48115167621
speed_currency=compact_profile_active_roots_per_sec
total_roots=122880
measured_sec=5.318338042999997
compact_owned_loop_entrypoint_enabled=true
resident_observation_used=true
resident_observation_host_fallback_count=0
obs_h2d_bytes=0
committed_replay_payload_d2h_bytes=0
replay_payload_d2h_bytes=0
python_rows_materialized=0
compact_rollout_slab_learner_gate_real_muzero_update=true
```

Preserved compact guardrail failure:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-matched-denominator-compact-owned-20260528/row_001_result_max_ticks_projection_failed.json
```

That first compact attempt failed closed because the compact projection did not
expose top-level `max_ticks`; the boundary now emits it.

## Read

The compact profile-only split loop is much faster in its own compact-root
currency than the stock `train_muzero` profile is in raw env-step currency. It
also preserves the resident/no-scalar/replay-state/lineage invariants.

This still does not prove a Coach speedup. The next blocker is promotion
semantics: a compact candidate must either actually enter the stock
`train_muzero`/Coach lifecycle or grow an explicitly validated compact trainer
with checkpoint, eval, tournament, reward/RND, death, and policy-refresh
contracts.
