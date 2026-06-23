# Optimizer Fallback Ledger

Date: 2026-05-28

Purpose: track intentional fallbacks, shims, and compatibility paths in touched
optimizer code. Hidden fallback is failure.

| ID | Path | Fallback / Shim | Keep Or Delete | Expiry / Reversal | Test |
| --- | --- | --- | --- | --- | --- |
| FB-001 | Hybrid profile runner | Missing payload `calls_train_muzero` / `touches_live_runs` labels fall back to manifest row labels for legacy/local test payloads. Contradictory payload labels fail closed. | Keep temporarily | Delete fallback once all hybrid payload fixtures and remote outputs always carry the full triad. | `test_hybrid_profile_runner_rejects_manifest_payload_label_mismatch` |
| FB-002 | Stock profile runner | Compact output without `ok=false` remains `complete`; explicit `ok=false` becomes `profile_failed`. | Keep | Tighten to require explicit `ok=true` once all stock compact outputs carry it. | `test_profile_manifest_status_marks_compact_ok_false_as_profile_failed` |
| FB-003 | Stock profile compact output | If raw collector env-step delta is zero and eval was skipped, compact output may report MCTS root-sum fallback as an explicitly separate speed currency. | Keep as diagnostic only | Remove from Gate A promotion path once all stock/RND profile rows reliably expose raw collector env-step deltas. | `test_profile_summary_marks_mcts_root_fallback_as_gate_a_ineligible` |
