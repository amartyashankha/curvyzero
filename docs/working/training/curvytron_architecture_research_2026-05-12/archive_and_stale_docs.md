# Archive And Stale Docs

Purpose: stop stale docs from steering new work while keeping the paper trail.

## Current Source

Use [current_source_of_truth.md](current_source_of_truth.md) first.

## Historical But Useful

- [high_signal_batch_matrix.md](high_signal_batch_matrix.md): launch ledger and
  v1d result history. Use it for evidence, not next-run defaults.
- [orchestration_plan.md](orchestration_plan.md): older research DAG. Some
  lanes are complete or superseded.
- [open_questions_and_hypotheses.md](open_questions_and_hypotheses.md): older
  hypothesis ledger. New active version is
  [hypotheses_and_evidence.md](hypotheses_and_evidence.md).
- [survival_readout_cache_plan_v1d_2026-05-12.md](survival_readout_cache_plan_v1d_2026-05-12.md):
  still useful for tooling design.
- [next_gates.md](next_gates.md): historical fixed/frozen gates plus a current
  warning at the top. Use `launch_gate_checklist.md` for active launch gates.
- [matrix_critique_2026-05-13.md](matrix_critique_2026-05-13.md): useful
  critique sketch, but its old 48-64 row shape is superseded by the current
  survivaldiag dry-run manifest.

## Treat As Historical Unless Revalidated

- Any doc that recommends `--mode two-seat-selfplay` for learning claims.
- Any doc that treats recent frozen opponents as automatically good curriculum.
- Any doc that recommends broad render/search/batch sweeps before fixing the
  reward/opponent issue.
- Any doc using `fast_gray64_direct` as a stock source-state train surface.

## Cleanup Rule

Do not delete the paper trail during active investigation. Instead:

1. Add a stale warning at the top of misleading docs.
2. Link back to `current_source_of_truth.md`.
3. Move obviously obsolete launch plans to an archive later if they keep
   confusing agents.
