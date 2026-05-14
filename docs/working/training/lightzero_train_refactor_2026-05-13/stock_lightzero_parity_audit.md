# Stock LightZero Parity Audit

Date: 2026-05-13

Purpose: record what is still stock LightZero in `--mode train`, what CurvyZero
adds around it, and what still needs proof before a larger cleanup.

## Plain Summary

Fresh `--mode train` is still close to stock LightZero in the important training
loop sense:

- CurvyZero builds a patched Atari MuZero config.
- CurvyZero registers a CurvyTron env.
- CurvyZero calls `lzero.entry.train_muzero(...)`.
- LightZero still owns collector, search, replay buffer, learner, and stock
  checkpoint creation.

The second CurvyTron player is hidden inside the env wrapper. LightZero sees one
scalar action per env step. The wrapper turns that into a two-player source
action, computes the reward, and returns one LightZero timestep.

## What LightZero Owns

| Piece | Current status |
| --- | --- |
| Entrypoint | `lzero.entry.train_muzero` is called in `--mode train`. |
| Collector | Stock `MuZeroCollector` drives env stepping. |
| Search | Stock MuZero policy/search runs inside collect/eval forward calls. |
| Replay | Stock MuZero game buffer receives collected game segments. |
| Learner | Stock `BaseLearner.train(...)` drives policy updates. |
| Stock checkpoint creation | Original LightZero/DI-engine save hooks run before CurvyZero side effects. |

## What CurvyZero Adds

| Addition | Why it exists | Risk |
| --- | --- | --- |
| CurvyTron env config | Makes LightZero train on the source-state visual env. | Expected semantic difference from Pong/Atari. |
| Reward and target-range patches | Keeps model supports aligned with CurvyTron reward scale. | Must stay visible in summaries. |
| Checkpoint progress writer | Makes website/status know which iteration exists. | Observability side effect. |
| Resume sidecar hooks | Tries to continue interrupted runs more usefully. | Not exact stock resume; replay objects are not fully portable. |
| Target/profiling audit hooks | Records collector/replay/learner evidence. | Passive for fresh train: wrappers return original collect/replay results. |
| Background eval/GIF workers | Human inspection artifacts. | Should not feed learner data. |

## Discrepancies To Keep Honest

| Area | Difference from stock Pong/Atari | Current judgment |
| --- | --- | --- |
| Env semantics | One LightZero action becomes an ego action plus an env-owned opponent action. | Acceptable: this is the env contract. |
| Action repeat | Defaults are no repeat, but config can repeat one policy action across physical ticks. | Training semantics change when enabled; keep explicit. |
| Reward variants | CurvyTron can use dense survival or bonus shaping. | Expected, but not a stock Atari reward. |
| Periodic eval | Default `lightzero_eval_freq=0` maps to after `max_train_iter`; fresh train still gets stock initial eval. | Acceptable for throughput, but document clearly. |
| Auto-resume | Reused `run_id` can inject `load_ckpt_before_run`. | Fresh run is stock-like; resumed run is not a fresh learning curve. |
| Resume sidecar | Saves extra state, but raw replay is metadata only. | Operational continuity, not exact uninterrupted training. |
| Always-on target audit | Wraps stock methods and records outputs. | Should remain passive; test/monitor for side effects. |
| Background artifacts | Extra workers read checkpoints for eval/GIF. | Observability only. |

## Evidence So Far

- Focused local tests pass for checkpoint discovery, trainer plumbing, run
  status, opponent mixtures, and opponent assignment snapshots.
- A tiny Modal CPU `--mode train` smoke succeeded after the refactor. It called
  stock `train_muzero`, wrote iteration checkpoints and resume sidecars, and
  run status found the latest checkpoint.
- A tiny Modal CPU artifact smoke with background eval/GIF enabled completed.
  The poller saw three checkpoints, completed three inspections and three GIF
  jobs, and run status found greedy `raw.gif` plus sampled `collect_t1.gif`
  for all three checkpoint cards.
- Read-only audit found the simultaneous-action detail is still hidden inside
  the env. The trainer does not implement custom replay or learner behavior for
  `source_state_fixed_opponent`.
- New local boundary test:
  `test_stock_source_state_mixture_config_instantiates_registered_env_and_steps_scalar_action`.
  It builds the trainer config, instantiates the registered LightZero env, and
  proves one scalar ego action steps the mixed blank-canvas opponent path.
- New local entrypoint test:
  `test_stock_train_mode_calls_lightzero_train_muzero_entrypoint`.
  It fakes `lzero.entry.train_muzero` and proves fresh `--mode train` calls
  that stock entrypoint with the patched config pair.
- Sidecar failure guard:
  `test_resume_sidecar_save_failure_does_not_fail_stock_checkpoint_hook`
  proves CurvyZero resume-sidecar I/O failures are logged without failing the
  already-completed stock checkpoint hook.
- Hook-passivity guards:
  `test_target_audit_hooks_return_original_collect_and_replay_results` and
  `test_live_checkpoint_publisher_calls_original_save_before_spawning` prove
  the main fresh-run audit/artifact hooks preserve original LightZero return
  values and run stock checkpoint save before side effects.
- Fresh resume hook guard:
  `test_fresh_resume_hooks_preserve_original_call_hook_eval_and_random_collect`
  proves `call_hook`, `eval`, and `random_collect` still return the stock
  results when resume is inactive.

## Not Proven Yet

- A live Modal run where DI-engine creates `lightzero_exp_*` after the patch.
- GPU train after this refactor.
- Hook-on/off checkpoint equality for a deterministic tiny run.
- Exact resumed-vs-uninterrupted equivalence. Current sidecar design should not
  claim exact equivalence.

## Next Gates

1. Keep the focused local test gate green.
2. Run the tiny Modal CPU stock-train smoke before trusting any deploy.
3. Run one tiny Modal artifact smoke with background eval/GIF enabled.
4. Add a local all-readers coherence test over one fake timestamped run tree if
   checkpoint/gif/status code is touched again.
5. Do not wire tournament-fed opponent selection into the trainer until a
   trainer-consumption test exists for assignment snapshots.
