# Training Setup Red-Team - 2026-05-09

Scope: assume Coach messed up setup. Top 10 plausible simple mistakes across
LightZero official Atari Pong, custom dummy Pong, and repo-native PPO. Local
docs/source only. No code changes. No pytest.

Blunt read: the setup is mostly plumbing-rich and quality-poor. The easy
failure mode is calling smokes, capped evals, or telemetry rows "learning."

## Sources Read

- `docs/working/training_state_index_2026-05-09.md`
- `docs/working/lightzero_official_atari_collapse_investigation_2026-05-09.md`
- `docs/working/lightzero_official_atari_settings_audit_2026-05-09.md`
- `docs/working/lightzero_official_atari_next_run_plan_2026-05-09.md`
- `docs/working/lightzero_stock_evaluator_action_mask_gap_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_root_cause_red_team_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_official_parity_gap_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_support_calibration_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_replay_sampling_error_2026-05-09.md`
- `docs/working/repo_native_ppo_learner_boundary_2026-05-09.md`
- `docs/working/repo_native_actor_loop_next_step_2026-05-09.md`
- `docs/working/shared_training_reporting_contract_2026-05-09.md`
- `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py`
- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- `src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py`
- `scripts/repo_native_ppo_actor_loop_dry_run.py`
- `scripts/repo_native_ppo_learner_smoke.py`
- `pyproject.toml`

## Top 10 Setup Mistakes

### 1. Official Atari: calling a tiny off-recipe run "stock Pong training"

Symptom: strict eval flatlines, e.g. return `-6`, no positive rewards, action
collapse by early checkpoints.

Evidence for: local docs say official Atari controls used tiny budgets versus
stock: `4096` env steps / `10` sims / batch `32` / `2` collectors in the
largest rung, while stock is hundreds of thousands of env steps, `50` sims,
batch `256`, and `8` collectors. Source wrapper explicitly patches those knobs.

Evidence against: infrastructure did work. CUDA, ALE Pong, checkpoints, strict
load, and eval plumbing are real.

Quickest falsifying check: read the train summary/config surface and compare
`max_env_step`, `num_simulations`, `batch_size`, `collector_env_num`,
`update_per_collect`, and `game_segment_length` to the official reference table
in `lightzero_official_atari_settings_audit`. If they are still tiny, do not
call it stock-quality training.

### 2. Official Atari: treating the capped eval number as full Pong score

Symptom: every checkpoint looks like exactly `-6` and someone over-interprets
that as a full-game score.

Evidence for: collapse investigation says the `-6` came from a manual
256-step eval cap; losses happened inside that small window. The train wrapper
also patches `collect_max_episode_steps` and `eval_max_episode_steps`.

Evidence against: the bad policy is still bad. Longer 512-step trainer-side
evals were also poor, just not the same exact number.

Quickest falsifying check: read the eval report for `max_eval_steps`,
`eval_max_episode_steps`, `n_evaluator_episode`, and nonzero reward timestamps.
If the window is 256 or 512 steps with one episode, the number is a capped
smoke read, not an Atari benchmark score.

### 3. Official Atari: using the wrong evaluator/checkpoint/config path

Symptom: eval errors about missing `action_mask`, uses model-greedy fallback,
or strict-load fails on observation/model shape.

Evidence for: the stock-evaluator note found `ding.worker.InteractionSerialEvaluator`
is wrong for MuZero parity because it does not pass the required action mask.
The pretrained OpenDILab path is also separately blocked by older 96x96/downsample
checkpoint surface versus the current 64x64 config path.

Evidence against: the current local checkpoint path is much cleaner now:
manual strict eval and the stock `lzero.worker.MuZeroEvaluator` parity smoke
matched actions for the matching 64x64 checkpoint.

Quickest falsifying check: open the eval artifact and confirm:
`stock_evaluator.path == lzero.worker.MuZeroEvaluator`,
`model_fallback_used == false`, strict load is true, policy observation shape
matches the checkpoint, and action mask values are present.

### 4. Custom dummy Pong: training with too few MCTS simulations

Symptom: learner physically executes all actions, but held-out MCTS never picks
the action that solves the state.

Evidence for: local target notes found `num_simulations=2` can produce root
visits like `[1, 1, 0]` in states where `down` is the useful action. LightZero
trains the policy target from root visit distribution, not from the executed
exploration action.

Evidence against: action ids are not obviously inverted. `ACTION_LABELS` is
`("up", "stay", "down")`; the env action mask is all ones; baselines can emit
`down`.

Quickest falsifying check: read `target_replay_steps.jsonl` for a tiny run and
compare `action_label` against `child_visit_segment`. If useful/executed actions
regularly receive zero target mass, the setup is teaching the wrong policy
target.

### 5. Custom dummy Pong: believing support ranges changed the compiled support

Symptom: reward/value heads look flat or nearly useless on a `-1/0/+1` toy,
even when summaries claim small support ranges.

Evidence for: support audit says pinned `LightZero==0.2.0` uses
`policy.model.support_scale`, default `300`, and support sizes around 601.
Older local code recorded requested `reward_support_range`/`value_support_range`
fields that may not be decisive for v0.2.0.

Evidence against: current config-import smoke now exposes compiled fields and
even checks patched `support_scale` against compiled `policy.model.support_scale`
when requested.

Quickest falsifying check: inspect the dry config/import output for compiled
`support_scale`, `reward_support_size`, `value_support_size`,
`reward_support_range`, and `value_support_range`. Ignore requested fields until
the compiled fields agree.

### 6. Custom dummy Pong: making segments shorter than MuZero's target window

Symptom: replay sampling fails with `'a' and 'p' must have same size`, or the
learner gets weird tiny-batch failures.

Evidence for: replay sampling note traced the failure to `game_segment_length=4`
with default `num_unroll_steps=5` and `td_steps=5`; non-terminal segments can
produce negative valid length and broken priority bookkeeping. Batch size `1`
then exposed a separate batchnorm single-row learner failure.

Evidence against: `game_segment_length=16` and `batch_size=2` completed the
bounded telemetry smoke and wrote target sidecars.

Quickest falsifying check: compare `game_segment_length` against
`num_unroll_steps + td_steps`, and check `batch_size`. If segment length is
smaller than the target window or batch is `1`, this is a setup bug, not a
learning result.

### 7. Custom dummy Pong: mixing run provenance across horizons/features/checkpoints

Symptom: a checkpoint looks bad or good only because eval used the wrong
episode horizon, feature mode, simulation count, state key, or checkpoint ref.

Evidence for: prior docs mention a 512-step checkpoint scored with mismatched
64/120 horizon values, feature modes that must be explicit, heuristic state-dict
loading, and manual artifact refs. The train wrapper has many knobs:
`pong_episode_max_steps`, feature mode, support, opponent, checkpoint state key,
and simulation count.

Evidence against: strict-load and scorecard plumbing are much better now, and
the reports include more provenance than early runs did.

Quickest falsifying check: require the train summary and scorecard to agree on
checkpoint SHA/ref, `feature_mode`, observation shape, `pong_episode_max_steps`,
reset profile, opponent policy, support fields, `num_simulations`, and state
key. Any mismatch invalidates the quality read.

### 8. Custom dummy Pong: treating trainer telemetry as checkpoint quality

Symptom: trainer-side episodes show all three actions or a positive terminal
reward, but independent held-out scorecards still lose or collapse.

Evidence for: training state says trainer rows used all actions, while held-out
MCTS rows for contact-pressure checkpoints still had `down=0` and final
checkpoints got worse than initialization. The environment records
`shaped_loss_delay_return` as telemetry, but LightZero receives sparse score
reward only.

Evidence against: telemetry is still useful for plumbing. It proves collection,
seeding, action mapping, and sidecar writing.

Quickest falsifying check: look for an independent no-fallback MCTS scorecard
against fixed baselines with action histograms. If only trainer sidecars or
episode summaries exist, quality is unproven.

### 9. Repo-native PPO: thinking the optional Torch smoke actually ran

Symptom: report exists, but no PPO update happened.

Evidence for: `pyproject.toml` depends only on NumPy; Torch is not a project
dependency. The learner script has a script-local optional Torch import and
writes a skipped report if Torch is absent.

Evidence against: the local smoke on this machine did import Torch and wrote
`checkpoint_step_000001.pt`, `ppo_metrics.jsonl`, and a learner report.

Quickest falsifying check: open `learner_report.json` and verify
`status != skipped`, `collection_policy_kind == tiny_actor_critic_sampled`,
`active_policy_rows > 0`, `ppo_metrics.jsonl` exists, and a checkpoint exists.

### 10. Repo-native PPO: calling a one-rollout toy learner "training progress"

Symptom: PPO report has loss/entropy/masked-action metrics, but no held-out
scorecard and no source/vector fidelity claim.

Evidence for: docs and source both label the PPO learner as no-quality:
one rollout, one tiny local MLP, one clipped PPO update, scalar toy env rows,
optional Torch, no scorecard, no source fidelity, no vector runtime claim.

Evidence against: the shape contract is valuable. It preserves `[T,B,P]`,
masked actions, final observations, reset seeds, and profiling buckets.

Quickest falsifying check: read the report `status`, `non_claims`, scorecard
section, and environment backend. If there is no held-out scorecard and the
backend is scalar toy rows, it is a wiring smoke only.

## Fastest Sanity Order

1. Official Atari: verify config scale and eval cap before reading returns.
2. Official Atari: verify strict no-fallback MuZeroEvaluator/checkpoint shape.
3. Custom Pong: inspect target replay `child_visit_segment`, not executed action.
4. Custom Pong: inspect compiled support fields, not requested support fields.
5. Custom Pong: reject segment/window/batch settings that break MuZero sampling.
6. Repo PPO: verify the report did not skip Torch.
7. Repo PPO: require a scorecard before any learning claim.

Bottom line: setup can be "working" and still be useless as learning evidence.
Most likely Coach mistake is over-claiming from the wrong report surface.
