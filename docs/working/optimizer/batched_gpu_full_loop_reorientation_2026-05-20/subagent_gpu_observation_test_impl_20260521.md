# GPU Observation Test Implementation - 2026-05-21

## Scope

Focused only on local GPU observation validation tests for the profile-only CurvyTron optimizer lane. No live training runs, Modal jobs, checkpoints, or run-management paths were touched.

## Changed Files

- `tests/test_source_state_batched_observation_boundary_profile.py`
  - Added a persistent framebuffer renderer request-order guard for full row-major `[row0/player0, row0/player1, ...]` requests.
  - Tightened exact-parity fail-closed coverage so a one-pixel drift must raise with the parity label in the failure path.
- `tests/test_source_state_hybrid_observation_profile.py`
  - Added renderer-backed `uint8` stack FIFO coverage across consecutive steps.
  - Added terminal autoreset coverage showing the returned timestep keeps the terminal stack while the manager's internal stack is reset to the post-reset frame.
  - Added action-mask order coverage through the batched stack probe and scalar LightZero timestep materialization.
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_gpu_observation_test_impl_20260521.md`
  - This note.

## Validation

Command:

```bash
uv run pytest tests/test_source_state_batched_observation_boundary_profile.py tests/test_source_state_hybrid_observation_profile.py
```

Result:

```text
76 passed in 0.41s
```

## Notes

- The new tests stay local and synthetic: fake renderers/probes exercise row/player order, stack FIFO/reset semantics, action-mask plumbing, and parity failure behavior without invoking GPU, Modal, or LightZero training.
- The target profile modules were reviewed for the tested contracts but were not edited.
