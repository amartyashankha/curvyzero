# Training Eval Rules

Status: v0 draft
Date: 2026-05-09

This note turns the current evaluation critique into rules for dummy survival,
Tiny Line Duel, and the first future MuZero runs on Modal. The goal is honest
progress measurement, not a leaderboard.

Current Pong status, 2026-05-09: self-play replay/train/eval exists locally,
but generation 2 lost to its parent and won 0 games against `track_ball`.
Treat generations and promotions as guardrails, not the main strategy. The next
decision is repair the crude trainer or switch to a known simple
baseline/curriculum, with fixed-baseline rows first.

Source context:
[training_evaluation.md](../research/training_evaluation.md), especially the
2026-05-09 critique. That note also anchors the broader precedent: AlphaZero
and MuZero report checkpointed evaluation curves against reference opponents;
AlphaGo Zero used old-vs-new promotion gates; OpenSpiel keeps evaluators,
checkpoints, configs, and logs separable. CurvyZero should borrow those habits,
not their scale.

## Non-Goals

- No Elo, league service, exploitability dashboard, or population training in
  v0.
- No claim that a dummy checkpoint is "MuZero quality."
- No per-step Modal calls. Modal runs whole training/eval/sweep jobs and stores
  saved outputs.
- No final-score-only reporting. Every selected checkpoint is shown next to
  `latest` and baselines.

## Evaluation Words

Use these words consistently:

- `episode`: one single-agent survival rollout, or one seated Tiny Line Duel
  game.
- `paired_seat_group`: two Tiny Line Duel games with the same environment seed
  and swapped seats. Use this when comparing different policies.
- `checkpoint`: a saved model state plus metadata.
- `policy_execution_spec`: the full policy run setup, including model, planner,
  safety masks, temperature/epsilon, search simulations, and fallback behavior.
- `selection_record`: the saved rule and file path showing how a best checkpoint
  was picked.

## Seed Splits

Every eval artifact, meaning each saved output file from an eval job, must name
the split and identify its seed list. A base seed alone is acceptable for v0 only
if the deterministic seed generator is also recorded.

| Split | Intended use | May select checkpoints? | May support final claims? |
| --- | --- | --- | --- |
| `train` | Rollout, exploration, curriculum, replay, online training noise. | No | No |
| `monitor` | Small check during training to catch regressions. It may become familiar to us. | No | No |
| `selection` | Fixed seed set used to rank periodic checkpoints with a metric chosen before the run. | Yes | Only as "selected on this split" |
| `heldout` | Fixed confirmation set run after the selected checkpoint is frozen. | No | Yes, with selection result shown |
| `debug` | Hand-picked failures, replayable seeds, and reproduction cases. | No | No |

The current seed-123 dummy survival sweeps are `monitor` or `debug`, not
`heldout`, because we have already used them to interpret the planner and make
patches.

Minimum split metadata:

```text
split_id: dummy_survival_monitor_v0
split_role: monitor | selection | heldout | debug | train
seed_generation: explicit_list | rng_integers
base_seed: optional integer
seed_count: integer
seed_list_hash: stable hash of explicit env seeds
paired_seat: true | false
created_at
notes
```

## Baselines And Ablations

A learned checkpoint is not a policy by itself. It is a model plus the code and
settings used to run it. Report those parts separately.

Required for dummy survival when evaluating learned checkpoints:

- `random_uniform`: floor baseline.
- `one_step_safe`: scripted heuristic gate.
- `planner_only` or `untrained_model_same_planner`: same planner, same safety
  rules, same tie-breaks, no learned table/model contribution.
- `latest` checkpoint: always present even when a `best` checkpoint exists.
- `selected_best` checkpoint: only after a selection split and selection record
  exist.

Required for Tiny Line Duel:

- `random_uniform`
- `random_sticky`
- `one_step_safe`
- `planner_only` or `untrained_model_same_planner` for learned-policy evals
- paired seats for every asymmetric matchup

Required for Pong while it is the visual toy:

- `random_uniform`: weak-opponent floor.
- `track_ball`: current scripted gate.
- `track_ball` versus `random_uniform`: baseline sanity row.
- `track_ball` versus `track_ball`: timeout and geometry canary.
- `latest` learned policy checkpoint when a training run saves policies.
- `previous` and `selected_best` checkpoint rows once periodic checkpoints
  exist.

Pong debug rows such as off-center contacts, contact-outcome probes, raster
frames, value loss, and action histograms explain the scoreboard. They are not
the scoreboard.

A Pong child checkpoint is only a candidate if it beats its parent and preserves
or improves the fixed-baseline rows. A monitor bump alone does not promote it.
Heldout is required before any quality claim.

Preferred comparison before any "model update helped" claim:

- `learned_no_safety_tiebreak` or equivalent. If this is not implemented yet,
  summaries must say the planner and safety rules may explain the result.

Allowed wording until the ablations exist:

```text
Checkpoint plus planner scored X on split Y.
```

Disallowed wording:

```text
Training learned a robust survival/winning policy.
```

## Checkpoint Selection

Picking a best checkpoint is allowed because final checkpoints can get worse,
but the report must still show that decline.

Use this v0 rule:

1. Choose one ranking metric per sweep before running it.
2. Rank periodic checkpoints on `selection`.
3. Break ties with secondary sanity metrics, not a second hidden score.
4. Freeze the selected checkpoint.
5. Run `heldout` once for that selected checkpoint, `latest`, and all required
   baselines.
6. Report both selection and heldout tables. If heldout does not confirm the
   selection ranking, mark the run inconclusive.

Default ranking metrics:

- Dummy survival: survival rate, then mean steps, then mean terminal reward,
  then lower crash count.
- Tiny Line Duel: paired-seat win rate against the opponent set, then lower
  loss rate, then lower timeout/draw farming, then action/terminal sanity.
- Pong: win rate against `track_ball`, then win rate against `random_uniform`,
  then lower truncation rate, then action/contact sanity. Latest-vs-old
  checkpoint rows are regression checks, not a replacement for the fixed
  baselines.
- Future MuZero: task-specific primary score plus sample count and wall-clock
  budget, with the exact planner/search config in the saved output.

Required visibility:

- Include `latest` and `selected_best` in the same summary.
- Include all evaluated checkpoint rows, not only the winner.
- Include the checkpoint iteration/update number and parent train run id.
- Record the selection split separately from heldout.
- For Pong training attempts, include iteration metrics, action histograms by
  seat, entropy/collapse metrics, terminal causes, and a few failure examples.

## Multiplayer Paired-Seat Rule

For Tiny Line Duel and later multiplayer CurvyTron tasks, do not average raw
seated games as if seats were independent.

For a matchup `A` vs `B`:

1. Generate one environment seed.
2. Run `A` as `player_0`, `B` as `player_1`.
3. Run `B` as `player_0`, `A` as `player_1` with the same environment seed.
4. Aggregate the two episodes as one `paired_seat_group`.

Report:

- paired win/loss/draw counts from A's perspective;
- per-seat win rates;
- seat delta: `player_0_win_rate - player_1_win_rate`;
- terminal causes by seat and policy;
- truncation/timeout rate;
- action histograms by policy and seat.

A learned policy cannot claim improvement if its gains come from seat order,
timeout farming, or action collapse.

## Minimum Modal Artifact Schema

Local and Modal eval jobs should write the same kind of saved outputs. Modal may
add run URLs, Volume refs, retry ids, and image/package metadata. A Modal
`Volume` is Modal's shared persistent storage for files created by remote jobs.

### Eval Job Summary

`summary.json` minimum:

```text
kind
schema_id
run_id
attempt_id
created_at
code_ref: git sha or working-tree marker
modal_ref: app/function/run url, image id, volume name, optional GPU type
task: dummy_survival | tiny_line_duel | future_muzero_task_id
ruleset_id/hash
observation_schema_id/hash
reward_schema_id/hash
action_schema_id/hash
eval_split: split_id, split_role, seed_count, seed_list_hash
episode_count
paired_seat_group_count
policy_specs[]
opponent_specs[]
policy_execution_specs[]
checkpoint_specs[]
aggregate_table[]
sanity_table: action histograms, terminal causes, truncations, seat deltas
intervals: confidence/bootstrap/binomial fields when claims are made
artifacts: summary, episodes, logs, checkpoint_eval, selection_record
```

`episodes.jsonl` minimum row:

```text
run_id
episode_id
pair_group_id
split_id
env_seed
seat
ego_policy_id
opponent_policy_id
checkpoint_id/path
policy_execution_spec_id
steps
terminal_reward
outcome: win | loss | draw | survived | crashed
terminal_cause
truncated
truncation_reason
action_histogram
reward_by_agent
death_cause_by_agent
elapsed_ms
```

### Checkpoint Sweep Summary

`summary.json` for a sweep must add:

```text
sweep_id
train_run_id
checkpoint_dir/ref
checkpoint_count
checkpoint_ids[]
selection_split_id
selection_metric
selection_tie_breaks
baseline_table
checkpoint_table
latest_checkpoint
selected_checkpoint
selected_checkpoint_path
heldout_required: true | false
heldout_summary_path: optional, filled after confirmation
```

`checkpoint_eval.jsonl` row:

```text
sweep_id
checkpoint_id
checkpoint_iteration
checkpoint_path/ref
split_id
episodes_or_pair_groups
primary_metric
secondary_metrics
baseline_deltas
planner_only_delta
action_histogram
terminal_causes
rank
selection_eligible
```

`selection_record.json`:

```text
selection_record_schema_id
sweep_id
selected_checkpoint_id
selected_checkpoint_path/ref
selection_split_id
selection_metric
selection_metric_value
tie_break_values
latest_checkpoint_id
latest_metric_value
required_baselines
heldout_command_or_job_spec
claim_status: selected_pending_heldout | confirmed | inconclusive
```

## Episode Counts And Allowed Claims

These are minimum rules for reading results. They are not statistical
guarantees.

| Episodes or paired groups | Allowed claim | Blocked claim |
| ---: | --- | --- |
| 5-10 | CLI runs, saved files are written, obvious schema/config bugs are visible. | Any quality, robustness, or learning claim. |
| 25-50 | Baseline sanity and obvious regressions. A floor like random crashing constantly may be noted. | Best checkpoint is robust; method improved. |
| 100 | Candidate selection on `selection`, if baselines and planner-only controls are present. | Final claim without heldout. |
| 400+ | Toy-task improvement claim, if selected on `selection`, confirmed on `heldout`, paired correctly for multiplayer, and intervals/sanity metrics are reported. | Claim that one training method is better than another without multiple independent training seeds. |

For binary survival/win rates, remember the rough rule of three: zero observed
failures in `n` trials still leaves an approximate `3/n` upper bound on the
failure rate. At 10 episodes that is about 30%; at 50 it is about 6%; at 100 it
is about 3%. Once summaries make quality claims, add exact binomial or bootstrap
intervals.

For method comparisons, separate eval episodes from independent training seeds.
One training seed with many eval episodes can identify a candidate. It cannot
show that one training method beats another. Use at least 3 independent training
seeds for internal decisions and 5-10+ for broader claims.

## Task-Specific Gates

### Dummy Survival

Before scaling local or Modal jobs:

- Learned checkpoint plus planner beats `random_uniform` on `selection`.
- Learned checkpoint plus planner beats `planner_only` or
  `untrained_model_same_planner` on `selection`.
- Selected checkpoint remains above those floors on `heldout`.
- Result is compared with `one_step_safe`; matching it on 10 monitor episodes
  is not enough.
- Action histogram and terminal causes do not show collapse or timeout tricks.

### Tiny Line Duel

Before treating a learned checkpoint as useful:

- Baselines behave plausibly: `one_step_safe` beats random/sticky on paired
  seats.
- Learned checkpoint beats random/sticky in paired-seat groups on `selection`.
- Learned checkpoint does not win through seat skew, timeout farming, or action
  collapse.
- `heldout` confirms the selected checkpoint against required baselines.

### Future MuZero

Before a MuZero run can claim improvement:

- The policy execution spec records search simulations, temperature, Dirichlet
  noise, value/reward transforms, action masks, safety masks, and fallback
  behavior.
- `planner_only` has a MuZero equivalent: same search code with an untrained or
  frozen-random network.
- Eval curves show `latest`, `selected_best`, and baselines across checkpoints.
- Modal saved outputs include image/package metadata and enough checkpoint refs
  to rerun the exact eval.

## Historical Implementation Tasks

This list is already mostly implemented or superseded. It is kept as context,
not as the active next-action list.

Keep the next changes small:

1. Add split metadata fields to dummy eval summaries: `split_id`,
   `split_role`, `seed_count`, and `seed_list_hash`.
2. Add `planner_only` or `untrained_model_same_planner` to dummy survival
   learned-checkpoint eval.
3. Extend dummy survival checkpoint sweep with `selection_record.json`,
   `latest_checkpoint`, `selected_checkpoint`, and `heldout_required`.
4. Add a heldout eval command shape for survival that reruns selected, latest,
   random, one-step-safe, and planner-only on a separate split.
5. Update Tiny Line Duel summaries to aggregate `paired_seat_group` rows and
   report seat deltas.
6. Add action histogram, terminal cause, truncation, and seat-bias checks to
   sweep summaries before adding any larger Modal jobs.
7. Mirror the same summary fields in Modal wrappers. Serious train/eval runs
   should use Modal as the durable execution path; local runs are tiny debug
   only.

The current active Pong decision lives in the coach docs: repair the crude
self-play trainer or switch to a simpler known baseline/curriculum, with fixed
baseline rows first.
