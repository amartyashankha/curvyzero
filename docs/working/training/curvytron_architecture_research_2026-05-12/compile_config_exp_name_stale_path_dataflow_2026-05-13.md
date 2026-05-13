# DI-engine `exp_name` Renewal vs CurvyZero Stale Path Closures

Date: 2026-05-13

## Summary

The smoking-gun hypothesis is credible and path-specific. CurvyZero builds a desired LightZero experiment path once as `.../train/lightzero_exp`, passes that into `main_config.exp_name`, and also closes over the same `Path` in progress, resume-sidecar, live eval, poller, scanner, and mirror helpers. LightZero then calls DI-engine `compile_config(..., save_cfg=True)` inside `train_muzero`. If that directory already exists and `renew_dir=True` (the default), DI-engine mutates the compiled `cfg.exp_name` by appending a timestamp. From that point onward, stock LightZero/DI-engine workers write logs and checkpoints under the timestamped directory, while CurvyZero wrappers keep watching the original directory.

This only bites when the pre-compile `exp_name` directory exists before `train_muzero` reaches DI-engine `compile_config`. If the directory is absent, DI-engine creates the requested directory and CurvyZero paths remain aligned.

## Source Links

- Local CurvyZero trainer: [`lightzero_curvyzero_stacked_debug_visual_survival_train.py`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py)
- Upstream LightZero v0.2.0 `train_muzero`: https://github.com/opendilab/LightZero/blob/v0.2.0/lzero/entry/train_muzero.py#L72-L127
- Upstream DI-engine v0.5.3 `compile_config`: https://github.com/opendilab/DI-engine/blob/v0.5.3/ding/config/config.py#L464-L471
- Upstream DI-engine `BaseLearner`: https://github.com/opendilab/DI-engine/blob/v0.5.3/ding/worker/learner/base_learner.py#L75-L104
- Upstream DI-engine `SaveCkptHook`: https://github.com/opendilab/DI-engine/blob/v0.5.3/ding/worker/learner/learner_hook.py#L127-L169

## Exact Dataflow

1. CurvyZero computes the intended experiment ref:
   - `exp_name_ref = attempt_train_ref / "lightzero_exp"`
   - `exp_name = Path(exp_name_ref.as_posix())`
   - Source: [`lightzero_curvyzero_stacked_debug_visual_survival_train.py:3192`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L3192)

2. CurvyZero writes `main_config.exp_name = str(exp_name)` while building patched LightZero configs.
   - Source: [`...:4458-4460`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L4458-L4460)

3. CurvyZero runs a preflight `_compile_config_summary`, but it uses `save_cfg=False`, so it does not exercise the DI-engine directory-renewal branch.
   - Source: [`...:3506-3512`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L3506-L3512)

4. CurvyZero installs runtime wrappers before calling `train_muzero`, passing the original `exp_name` into each wrapper:
   - full resume sidecar hooks: [`...:3539-3547`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L3539-L3547)
   - progress writer hook: [`...:3548-3555`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L3548-L3555)
   - live checkpoint publisher hook: [`...:3556-3567`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L3556-L3567)

5. CurvyZero calls `train_muzero([patched["main_config"], patched["create_config"]], ...)`.
   - Source: [`...:3596-3601`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L3596-L3601)

6. LightZero immediately rebinds `cfg` to DI-engine's compiled config.
   - Upstream source: `cfg = compile_config(..., save_cfg=True)` at https://github.com/opendilab/LightZero/blob/v0.2.0/lzero/entry/train_muzero.py#L72

7. DI-engine deep-copies the input config, merges defaults, sets `cfg.seed`, then enters the save-config branch. If `cfg.exp_name` already exists and `renew_dir=True`, it appends a timestamp before creating the directory and saving config files.
   - Upstream source: https://github.com/opendilab/DI-engine/blob/v0.5.3/ding/config/config.py#L369-L471

8. After compile, stock LightZero constructs runtime objects with the compiled `cfg.exp_name`.
   - TensorBoard writer path: https://github.com/opendilab/LightZero/blob/v0.2.0/lzero/entry/train_muzero.py#L102
   - `BaseLearner(..., exp_name=cfg.exp_name)`: https://github.com/opendilab/LightZero/blob/v0.2.0/lzero/entry/train_muzero.py#L103
   - `Collector(..., exp_name=cfg.exp_name)`: https://github.com/opendilab/LightZero/blob/v0.2.0/lzero/entry/train_muzero.py#L112-L118
   - `Evaluator(..., exp_name=cfg.exp_name)`: https://github.com/opendilab/LightZero/blob/v0.2.0/lzero/entry/train_muzero.py#L119-L127

9. DI-engine checkpoint saving follows the compiled learner `exp_name`: `SaveCkptHook` builds `./{engine.exp_name}/ckpt` and writes `iteration_*.pth.tar`.
   - Upstream source: https://github.com/opendilab/DI-engine/blob/v0.5.3/ding/worker/learner/learner_hook.py#L127-L169

## CurvyZero Closures That Keep The Original Path

- `_install_checkpoint_progress_writer` closes over `exp_name` and calls `_write_checkpoint_progress_latest(..., exp_name=exp_name)`. That helper looks for `exp_name / "ckpt"`, so after DI-engine renewal it can report learner iteration but miss checkpoint metadata.
  - Source: [`...:1829-1865`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L1829-L1865), [`...:1868-1918`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L1868-L1918)

- `_install_lightzero_full_resume_state_hooks` closes over `exp_name`; its `SaveCkptHook.__call__` wrapper saves sidecar state under `Path(exp_name) / lightzero_resume_state` only if the matching checkpoint exists under `Path(exp_name) / ckpt`.
  - Source: [`...:1926-2070`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L1926-L2070)

- `_install_live_checkpoint_publisher` closes over `exp_name` and scans that path for new checkpoints after `BaseLearner.save_checkpoint`.
  - Source: [`...:1742-1794`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L1742-L1794), [`...:5693-5735`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L5693-L5735)

- The default background eval poller is spawned with `exp_name_ref` set to the unsuffixed `.../train/lightzero_exp` and scans that exact directory.
  - Source: [`...:9930-9943`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L9930-L9943), [`...:6193-6335`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L6193-L6335)

- Post-training artifact discovery and mirroring scan the original `exp_name`, then mirror whatever checkpoints that stale scan finds.
  - Source: [`...:3672-3676`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L3672-L3676), [`...:5432-5513`](../../../../src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py#L5432-L5513)

## Ranked Implications

1. **Highest: false "no checkpoints" / failed mirror after successful training.** Stock LightZero may successfully save to `lightzero_exp_YYMMDD_HHMMSS/ckpt`, while CurvyZero scans `lightzero_exp/ckpt`, finds nothing, and appends "no LightZero checkpoint artifacts" or "no LightZero checkpoints were mirrored".

2. **High: live eval and GIF poller can starve.** The poller watches the unsuffixed path. If checkpoints land under the renewed path, `last_scan_count` stays zero and no eval/GIF jobs launch.

3. **High: progress files can lose checkpoint refs.** The progress hook can still record `learner_train_iter`, but `_latest_lightzero_iteration_checkpoint(exp_name)` misses the actual checkpoint, so browser/status consumers may see progress without `checkpoint_ref`.

4. **Medium: CurvyZero full-resume sidecars may not be written.** The sidecar save path checks for a matching checkpoint under the stale path first. With renewed `cfg.exp_name`, the checkpoint exists elsewhere, so the sidecar write can return `matching_iteration_checkpoint_not_found`.

5. **Medium: auto-resume is partially protected by stable mirror, but current-attempt discovery is stale.** `_prepare_lightzero_auto_resume` scans current/prior attempts at literal `train/lightzero_exp/ckpt`, then the stable run checkpoint mirror. If mirroring was already broken by the stale scan, later resume also degrades.

6. **Low: model learning itself is probably not impaired by this path split.** LightZero creates envs, policy, learner, collector, evaluator, replay buffer, and training loop state from the compiled `cfg`. The likely damage is observability, checkpoint publication, eval/GIF scheduling, and resume continuity, not gradient updates.

## Practical Diagnostic Signature

Look for a training attempt whose summary reports missing checkpoints while logs mention `learner save ckpt in ...lightzero_exp_<timestamp>/ckpt/iteration_*.pth.tar`, or whose volume contains both `lightzero_exp/` and `lightzero_exp_YYMMDD_HHMMSS/`. That pattern would directly match the DI-engine renewal branch plus CurvyZero stale closures.
