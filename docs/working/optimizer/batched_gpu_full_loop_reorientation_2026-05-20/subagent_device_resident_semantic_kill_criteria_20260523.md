# Device-Resident Search Semantic Kill Criteria - 2026-05-23

Status: read-only critique output. I did not touch live runs. This is for any
future aggressive path where search, render input, replay rows, or RND data are
kept resident on device or passed through a search service.

## Plain Rule

Speed does not count until the data still means the same thing.

The risky move is not "use the GPU." The risky move is changing who owns the
policy row while search, env stepping, replay, RND, and learner sampling are
now separated by queues, batches, or resident buffers.

Any candidate must prove:

```text
observation at time k
-> search root k
-> selected action k
-> env transition k+1
-> replay target row k
-> learner-visible sample
```

with the same `env_row`, `player`, `policy_env_id`, controlled-player view,
legal mask, terminal/final-observation rule, reward rule, and RND latest frame.

## What Can Go Wrong

| Risk | Failure Mode | Why It Is Dangerous |
| --- | --- | --- |
| Action feedback | Search chooses action A, but env steps action B because of stale arrays, queue reorder, row compaction, or fallback. | The policy learns from a transition it did not cause. |
| Root identity | `root_index`, `env_row`, `player`, or `policy_env_id` changes across root batch, search result, replay rows, or trainer sample. | Replay rows look plausible but point to the wrong physical game/player. |
| Player perspective | The observation is for player 0, but the action/reward belongs to player 1, or tournament gets a different view than training. | The policy learns a mismatched visual/action contract. |
| Terminal final observation | A terminal transition stores the post-reset observation instead of the terminal final frame. | The learner sees a fake next state after death. |
| Inactive roots | Terminal or no-legal roots still get searched, or poison values leak from inactive slots. | Visit targets and values silently include rows that should not exist. |
| RND latest frame | RND reads the wrong player, stale stack channel, or reset frame; meter mode mutates target rewards. | Exploration metrics stop describing the same observation stream. |
| Replay target parity | Compact/index rows materialize to different target rows than the trusted builder. | Speed path trains a different algorithm by accident. |
| Root noise and ties | Root noise, seeds, or tie-breaking differ between stock and candidate but results are compared as exact. | A valid stochastic difference gets mistaken for a bug, or a bug hides behind noise. |
| Stale policy | A queue or service uses model version N while replay metadata says model version N+1, or learner samples before rows are complete. | The denominator is no longer the claimed synchronous training loop. |
| Queue ordering | Producer results arrive out of order and are attached by position instead of stable ids. | The highest-throughput case is the easiest one to corrupt. |
| Batching cohorts | Roots are grouped by sim count, active mask, or device bucket and then returned in cohort order. | Non-prefix active roots and mixed terminal/live batches break silently. |
| Learner visibility | Replay rows become sample-visible before visit policy, root value, final obs, RND, or checksums are written. | The learner consumes partial rows. |
| Sync timing | Action readback is measured, but replay payload readback or row materialization happens later and is omitted from the denominator. | The speedup is fake. |

## Kill Criteria Checklist

Stop promotion immediately if any item is true:

- Search-selected action is not proven to be the next env `joint_action`.
- `root_index`, `env_row`, `player`, or `policy_env_id` changes across root,
  search, replay, and sample boundaries.
- Any selected action is illegal under its own legal mask.
- Visit policy or raw visit counts put mass on illegal actions.
- Active root order is inferred from array position instead of stable ids.
- Non-prefix active roots fail, reorder, or lose rows.
- Terminal roots are searched as live roots.
- Terminal next observation comes from an autoreset frame.
- Final-observation masks, terminal masks, and done masks disagree.
- Controlled-player perspective is chosen by optimizer code instead of the
  training/tournament contract.
- Player 0 and player 1 views are not both tested.
- RND meter mode changes target rewards.
- RND-enabled rows lack proof of reward-model entrypoint and metrics.
- RND latest frame is not tied to the same `(env_row, player)` as the policy
  observation.
- Root-noise state is unknown in a parity claim.
- Exact parity is claimed on noisy or tie-heavy roots without a deterministic
  fixture.
- Queue/service result attaches by batch position without id checks.
- A row is sample-visible before action, reward, done, visit policy, root value,
  final observation, and RND sidecars are complete.
- Fallback count is nonzero in a promoted row.
- The summary cannot distinguish Coach training speed, stock full-loop profile
  speed, and optimizer probe speed.

## Smallest Local Tests

These should pass before any remote profile is allowed to matter.

```sh
uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  tests/test_compact_torch_search_service.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_profile_cpu.py \
  tests/test_exploration_bonus.py \
  tests/test_lightzero_phase_profiler.py
```

Add or strengthen these cases if missing:

1. **Real closed-loop action feedback**
   - Build roots from a real profile env step.
   - Run the candidate search service.
   - Feed `selected_action` into the next env step.
   - Assert `applied_joint_action_checksum == selected_action_checksum`.
   - Include non-prefix active roots.

2. **Stable id poison test**
   - Use non-identity `policy_env_id`.
   - Shuffle/cohort roots internally.
   - Return results in a different order.
   - Assert replay attaches by ids, not position.
   - Poison one stale `env_row` and one stale `policy_env_id`; both must fail.

3. **Both-player perspective sentinel**
   - Same physical state, two policy rows, players 0 and 1.
   - Swap observations while keeping actions/rewards fixed.
   - Test must fail.
   - Metadata must cite
     `curvyzero_policy_observation_controlled_player_perspective/v1`.

4. **Mixed terminal/live final-observation test**
   - One row terminates, one row stays live.
   - Autoreset is enabled.
   - Terminal next observation must equal terminal `final_observation`, not the
     reset frame.
   - Terminal root must not be searched.
   - Live row must still be searched.

5. **Inactive-root poison test**
   - Fill inactive roots with absurd logits, values, and selected actions.
   - Assert no replay target, RND row, or learner sample sees those values.

6. **RND latest-frame and reward test**
   - Enable `rnd_meter_v0` with zero weight.
   - Assert target rewards are unchanged.
   - Assert `train_muzero_with_reward_model` path is used.
   - Assert `train_cnt_rnd > 0`, predictor changed, target network unchanged.
   - Assert RND latest frame equals the policy latest frame for the same
     `(env_row, player)`.
   - Repeat with terminal/autoreset row.

7. **Root-noise off exact gate**
   - `root_noise_weight=0.0`.
   - Single-legal and clear-preference fixtures.
   - Exact actions, legal visit mass, and root ids must match.
   - Tie-heavy fixtures must be labelled statistical, not exact.

8. **Queue/service ordering gate**
   - Simulate two producers and delayed out-of-order service results.
   - Attach results by stable ids.
   - Assert learner-visible rows are identical to ordered baseline.

9. **Learner sample visibility gate**
   - Insert incomplete compact rows into replay.
   - Sampler must reject or hide them.
   - Rows become visible only after action, reward, done, visit policy, root
     value, final observation masks, and RND sidecars are complete.

## Smallest Remote Profiles

Run only after the local gates above pass.

### P0 Same-Denominator Smoke

Purpose: prove the candidate can run the same denominator as the trusted path.

Required knobs:

```text
root_noise_weight=0.0
num_simulations=8
collectors=64 or 128
normal death and no-death rows
RND none and rnd_meter_v0 zero-weight rows
fallback_count must be 0
checkpoint/eval/GIF sidecars off or matched
```

Rows:

```text
stock
direct_ctree_gpu_latent
candidate_device_or_service_path
```

Required output fields:

```text
called_train_muzero or profile-only label
backend
input mode
seed
root_noise_weight
num_simulations
collectors
death mode
RND mode
fallback_count
illegal_action_count
replay proof passed
action feedback checksum passed
observation contract id
root id digest
player perspective digest
terminal/final observation digest
RND metrics digest
sample visibility proof
obs_h2d_bytes
action_d2h_bytes
replay_payload_d2h_bytes
root_copy_bytes
python_rows_materialized
rnd_materialized_rows
```

Kill if any required field is absent.

### P1 Noisy Statistical Smoke

Purpose: test normal stochastic training conditions after exact gates pass.

Required knobs:

```text
seed list shared across backends
normal root noise
num_simulations=16 and 32
collectors=256 and 512
RND none plus rnd_meter_v0 compatibility row
```

Do not require exact action parity on tie-heavy roots. Compare legality,
identity, replay completeness, timing, and distribution-level summaries.

### P2 Service Queue Smoke

Purpose: test the architecture that could actually scale.

Shape:

```text
N producers
one search service
out-of-order result completion allowed
stable id attachment required
fixed model version per batch
rows hidden from learner until complete
```

Required proof:

- result reorder count is nonzero;
- id-based attachment still matches baseline;
- model version is recorded per root batch and per replay row;
- learner-visible count equals complete-row count;
- action-critical sync time is separated from replay-payload readback.

## Promotion Rule

A device-resident or service-based path can be called Coach-facing only after:

1. local P0 semantic tests pass;
2. same-denominator remote smoke passes with root noise off;
3. RND meter compatibility passes if the recommended training config uses RND;
4. normal noisy smoke has zero fallback, zero illegal actions, and complete
   replay/terminal/perspective digests;
5. speed is reported in the correct currency.

Until then, it is an optimizer probe, not a training recommendation.
