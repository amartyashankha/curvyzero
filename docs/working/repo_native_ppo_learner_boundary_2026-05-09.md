# Repo-Native PPO Learner Boundary

Date: 2026-05-09
Status: tiny optional-Torch on-policy learner smoke added

## Decision

The smallest safe next step from the masked-uniform actor dry run is a separate
script-local optional Torch on-policy learner smoke:

```text
scripts/repo_native_ppo_learner_smoke.py
```

It does not replace `scripts/repo_native_ppo_actor_loop_dry_run.py`. The dry-run
script remains the clean masked-uniform actor-loop contract probe. The learner
smoke now initializes one tiny actor-critic, collects the rollout with that same
model, keeps the `[T,B,P]` storage boundary, then applies one PPO update to the
same model instance:

- shared ego-row MLP policy/value model;
- masked categorical logits over action ids `0..2` during collection and update;
- recorded behavior `action_logprob`, `action_probs`, and `value` from the
  collecting policy;
- GAE/returns with shapes `[T,B,P]`;
- one clipped PPO update over live ego rows;
- `ppo_metrics.jsonl`;
- `checkpoint_step_000001.pt`;
- `learner_report.json` with original dry-run profile buckets plus
  `target_construction_sec`, `learner_update_sec`, and
  `checkpoint_publish_sec`.

This is a no-quality smoke. It proves wiring, shape preservation, mask handling,
artifact writing, and optional Torch import behavior only.

## Torch Availability

`pyproject.toml` currently declares only:

```text
numpy>=1.26
```

There is no required or optional Torch dependency in project metadata. Local
import probe on this machine found Torch `2.9.1`, so adding a script-local
optional import is safe. If Torch is absent, the learner smoke writes a skipped
`learner_report.json` instead of making Torch a repository requirement.

## Boundary Preserved

The smoke keeps the actor-loop contract parallel to LightZero replication:

- LightZero remains the MuZero replication/control lane.
- Repo-native PPO remains a transparent CurvyTron `[B,P]` actor/learner lane.
- PPO success or plumbing success must not retire LightZero.
- LightZero plumbing success must not skip repo-native timing, buffer, mask, and
  reset/final-observation checks.

Current learner-smoke limitation: the actor/learner loop is on-policy but still
no-quality and smoke-scale. It uses scalar toy env instances, a tiny local MLP,
one rollout, and one PPO update only. There is no scorecard, no source-fidelity
claim, no vector-runtime claim, and no LightZero replacement claim.

## Local Smoke

No pytest was run. Direct smoke command:

```sh
python3 scripts/repo_native_ppo_learner_smoke.py \
  --batch-size 4 \
  --rollout-steps 8 \
  --artifact-root /private/tmp/curvy-repo-native-ppo-on-policy-smoke \
  --format plain
```

Observed output:

```text
repo_native_ppo_learner_smoke B=4 T=8 active_policy_rows=64 loss=0.159628 entropy=1.094129 masked_action_violations=0 artifact_root=/private/tmp/curvy-repo-native-ppo-on-policy-smoke
```

Artifact directory:

```text
checkpoint_step_000001.pt
learner_report.json
ppo_metrics.jsonl
report.json
rollout_buffer.npz
```

Shape check from `learner_report.json`:

```text
observation        [8, 4, 2, 106]
legal_action_mask  [8, 4, 2, 3]
live_mask          [8, 4, 2]
action             [8, 4, 2]
action_logprob     [8, 4, 2]
value              [8, 4, 2]
reward             [8, 4, 2]
done               [8, 4]
advantage          [8, 4, 2]
return             [8, 4, 2]
```

Collection policy and value check:

```text
collection_policy_kind  tiny_actor_critic_sampled
behavior_value_abs_mean 0.19575341045856476
```

Profile buckets present:

```text
reset_autoreset_sec
observation_packing_sec
row_compaction_sec
policy_forward_sec
action_scatter_sec
env_step_sec
replay_or_rollout_stage_sec
replay_write_or_learner_handoff_sec
target_construction_sec
learner_update_sec
checkpoint_publish_sec
```

## Exact Next Slice

This slice completed the previous acceptance criteria:

- rollout collection writes model `value` predictions instead of zeros:
  `behavior_value_abs_mean=0.19575341045856476`;
- recorded `action_logprob` comes from the behavior policy used to sample;
- masked-action violation count remained zero;
- all existing `[T,B,P]` artifact fields and profile buckets remained present;
- `learner_report.json` still includes dry-run-style artifact/profile
  fields for comparison with LightZero;
- the run is still labeled no-quality until held-out scorecards exist.

The next smallest implementation slice is to add a tiny deterministic
post-update sanity eval over the same scalar env contract, still no-quality, so
the checkpoint/report can prove load-and-act wiring without implying strength.

Do not add Torch to `pyproject.toml` until the repo chooses to make the
repo-native learner an official dependency surface. Until then, keep this lane
optional and smoke-scale.
