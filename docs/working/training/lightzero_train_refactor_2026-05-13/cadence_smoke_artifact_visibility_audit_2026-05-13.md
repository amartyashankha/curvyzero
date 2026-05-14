# Cadence Smoke Artifact Visibility Audit - 2026-05-13

Scope: focused follow-up on the CPU Modal `--mode train` smoke after changing
trusted CurvyZero LightZero train to one policy action per source physics frame.
No source was edited for this audit.

## Result

Most likely cause: missing explicit final Modal Volume commit for `mode="train"`,
with ordinary Modal background/final commit timing as the immediate visibility
mechanism.

This does not look like a wrong path or hidden LightZero artifact path in the
current code. The current code already scans sibling `lightzero_exp*` roots and
the returned smoke result had `ok=true`, `called_train_muzero=true`,
`problems=[]`, and telemetry `row_count=128`. If final artifact discovery or
mirroring had failed in-process, the train summary would have added problems.

## Facts

- The smoke returned:
  - `ok=true`
  - `called_train_muzero=true`
  - `problems=[]`
  - telemetry `row_count=128`
- Immediate/repeated `modal volume ls` only showed early files:
  - `run.json`
  - `latest_attempt.json`
  - `attempt.json`
  - `train/status_heartbeat.json`
- Missing from the listing:
  - `train/summary.json`
  - `train/action_observability.json`
  - `train/lightzero_artifacts_manifest.json`
  - `checkpoints/lightzero/iteration_*.pth.tar`

## Code Evidence

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py::_run_visual_survival_train`
  writes the final summary and related train artifacts at lines 3886-3914.
- The same function writes final attempt metadata and final heartbeat at lines
  3916-3947.
- The only explicit final commit in this function is gated to profile mode:
  lines 3948-3961 commit only when `mode == "profile"` and
  `profile_volume_commit` is true.
- Train mode therefore returns with `final_volume_commit={"attempted": false}`
  in the result at lines 3962-3989.
- Checkpoint discovery happens before the final writes at lines 3756-3760.
  Train mode adds problems if no checkpoint artifacts or mirrors are found at
  lines 3771-3777. The smoke had `problems=[]`, so in-process discovery and
  mirroring succeeded.
- `_scan_lightzero_artifacts` now scans the configured exp root and sibling
  `lightzero_exp*` roots via `_lightzero_exp_sibling_roots` at lines 5524-5571.
- `src/curvyzero/training/lightzero_checkpoints.py::lightzero_exp_sibling_roots`
  explicitly returns the configured exp dir plus timestamped sibling dirs at
  lines 54-75.
- `_mirror_lightzero_checkpoints` copies discovered checkpoints into
  `training/<task>/<run>/checkpoints/lightzero/` at lines 5574-5616.
- `src/curvyzero/infra/modal/run_management.py` defines the expected refs:
  `run.json` at lines 81-82, `latest_attempt.json` at lines 89-90,
  `attempt.json` at lines 93-98, `train/` at lines 101-102, and
  `checkpoints/` at lines 116-117.

## Modal Semantics

Modal docs say Volume changes made inside a container must be committed before
they are visible outside that container. Background commits run every few
seconds while a function executes, and Modal also performs a final snapshot and
commit on container shutdown:

- https://modal.com/docs/guide/volumes
- https://modal.com/docs/reference/modal.Volume

That matches the observed shape. The early files were written before and during
training, so a background commit could publish them. The final files and mirror
copies are written after `train_muzero` returns. Without a train-mode explicit
commit, an immediate CLI listing can see only the earlier committed tree. Also,
`status_heartbeat.json` being visible is not proof that the final heartbeat was
visible, because the same file is written early at stage `auto_resume_checked`.

## Classification

- Propagation delay: partly, but not a satisfying root cause. Modal background
  and shutdown commits are asynchronous enough that immediate listing can lag.
- Missing final volume commit: yes, this is the likely code-level cause.
- Wrong path: unlikely. The refs and local writes line up with
  `training/<task>/<run>/attempts/<attempt>/train/summary.json`.
- Hidden LightZero artifact path: unlikely in the current code. Sibling
  `lightzero_exp*` discovery is present, and `problems=[]` means checkpoint
  discovery and mirroring passed in-process.
- Another bug: yes, as an artifact durability/visibility bug in the trainer
  scaffold, not in the cadence change or LightZero training call.

## Minimal Test

Add or extend a focused trainer plumbing test, likely
`tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_stock_train_mode_calls_lightzero_train_muzero_entrypoint`:

- monkeypatch `train_mod.runs_volume` to a fake object with `commit_count`;
- keep the fake `train_muzero`, fake artifact scan, fake mirror, and fake
  telemetry already used by that test;
- run `_run_visual_survival_train(mode="train", ...)`;
- assert `result["final_volume_commit"]["attempted"] is True`;
- assert `result["final_volume_commit"]["ok"] is True`;
- assert `fake_volume.commit_count == 1`;
- assert the local `summary_ref` and mirror entries still exist before the
  commit assertion.

Keep the existing tests that assert background checkpoint eval/GIF hooks do not
commit. Those tests protect the no-commits-inside-training-loop rule.

## Minimal Patch

After final summary, artifacts, attempt state, latest attempt, and final
heartbeat are written, call:

```python
_commit_runs_volume_with_backoff(label="train_final_commit")
```

for `mode == "train"` when `runs_volume` has `commit`.

Reuse the existing `final_volume_commit` result shape. Do not add commits inside
`BaseLearner.save_checkpoint`, `_install_checkpoint_progress_writer`,
`_install_live_checkpoint_publisher`, `_spawn_checkpoint_eval_triggers`, or the
poller loop.

## No-Commits-Inside-Training-Loop Rule

Adding one final commit after `train_muzero` has returned would not violate the
no-commits-inside-training-loop rule. It is outside LightZero collection,
replay, learner updates, and checkpoint save hooks. It only publishes final
artifacts that have already been written.

Committing from the save-checkpoint wrapper, live checkpoint publisher,
progress writer, background eval trigger, or checkpoint poll loop would violate
the spirit of that rule and would risk slowing or perturbing training.
