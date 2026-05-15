# Run Reward Control Design

Date: 2026-05-14

## Plain Goal

For each training run, we want external control over the reward recipe, just as
we want external control over opponent slots.

The reward recipe has three plain parts:

| Part | Meaning | Current examples |
| --- | --- | --- |
| `survival` | Reward for still being alive after a source step. | `dense_alive_helper = 1.0` |
| `bonus` | Reward for picking up a bonus on that step. | `bonus_pickup_reward_per_catch = 1.0` |
| `final_outcome` | Reward or penalty at the end of the round. | win `+1`, loss `-1`, draw `0`, or all zero |

Today this is mostly controlled by named variants:

| Variant | Survival | Bonus | Final outcome |
| --- | --- | --- | --- |
| `sparse_outcome` | off | off | on |
| `dense_survival_plus_outcome` | on | off | on |
| `survival_plus_bonus_no_outcome` | on | on | off |
| `survival_plus_bonus_plus_outcome` | on | on | on, scaled by episode source-step count |

The target is not to invent a large reward language. The target is a small,
auditable run control record that can say which reward profile a run should use
and what the important weights are.

## Important Difference From Slots

Changing opponent slots changes who the policy plays next.

Changing reward weights changes what the learner is trying to predict and
optimize. If one run mixes old samples scored with one reward recipe and new
samples scored with another recipe, the replay buffer contains mixed reward
definitions. That may be intentional later, but it should never happen silently.

First rule:

```text
Reward changes are launch-time or new-attempt changes unless we explicitly
decide what to do with old training samples.
```

This is why reward recipes need a schema id, hash, generation, and durable
snapshot. The live Modal Dict is operator intent; the trainer should consume a
frozen reward recipe recorded in the launch/attempt artifacts.

## Recommended Control Shape

The slot Dict should evolve into a more general run-control Dict.

```text
dict name: curvyzero-training-run-control
key: run_control:<training_run_id>
```

Value:

```json
{
  "schema_id": "curvyzero_training_run_control/v0",
  "run_id": "curvy-mix-example",
  "generation": 7,
  "updated_at": "2026-05-14T00:00:00Z",
  "updated_by": {"kind": "operator", "id": "manual"},
  "enabled": true,
  "opponents": {
    "schema_id": "curvyzero_run_slot_recipe/v0",
    "generation": 3,
    "profile": "stable_5"
  },
  "reward": {
    "schema_id": "curvyzero_reward_recipe/v0",
    "generation": 2,
    "profile": "survival_bonus_no_outcome",
    "weights": {
      "survival": 1.0,
      "bonus": 1.0,
      "final_outcome": 0.0
    },
    "final_outcome": {
      "winner": 0.0,
      "loser": 0.0,
      "draw": 0.0,
      "timeout": 0.0
    },
    "change_policy": "new_attempt_only"
  },
  "notes": "operator-written human note"
}
```

The `opponents` section can either inline the existing slot recipe or point to
the existing slot recipe key. The important design change is that one run id can
now have one control record for all run-level choices.

## Profiles

Keep profiles small and obvious.

| Profile | Use | Weights |
| --- | --- | --- |
| `outcome_only` | Normal two-player win/loss learning. | survival `0`, bonus `0`, final outcome `1` |
| `survival_outcome` | Learn to survive while still caring about wins. | survival `1`, bonus `0`, final outcome `1` |
| `survival_bonus_no_outcome` | One-player or invincible-opponent sanity runs. | survival `1`, bonus `1`, final outcome `0` |
| `survival_bonus_outcome` | Tonight probe where survival and bonus are dense, and the terminal result can double or cancel the survival signal. | survival `1`, bonus `1`, final outcome `episode_source_step_count` |
| `bonus_probe_no_outcome` | Check whether bonus pickup creates useful exploration pressure. | survival `1`, bonus greater than `1`, final outcome `0` |

The user-facing choice should usually be the profile. The raw weights are there
for controlled experiments.

## Survival And Final Outcome

Survival and final outcome are related, but not the same.

Survival is dense. It gives the learner feedback every step: "being alive now is
good."

Final outcome is terminal. It gives feedback at the end: "winning was good" or
"losing was bad."

For normal two-player training, keeping both can make sense.

For invincible-opponent or blank-canvas runs, final outcome can be meaningless.
In those runs, the clean reward is usually:

```text
survival = on
bonus = on
final_outcome = off
```

That matches the existing `survival_plus_bonus_no_outcome` direction.

## Durable Artifacts

At launch or new attempt, write a frozen reward snapshot next to the command and
assignment artifacts.

Suggested file:

```text
train/<run_id>/<attempt_id>/control/reward_recipe.json
```

Suggested audit fields:

- Dict name and key;
- run-control generation;
- reward recipe generation;
- reward recipe hash;
- selected reward profile;
- exact weights;
- matching trainer `reward_variant` when it maps to an existing variant;
- whether this is launch-time, resume-time, or new-attempt-time;
- the rule for old training samples.

The command summary should also record:

- `reward_profile`;
- `reward_schema_id`;
- `reward_schema_hash`;
- `reward_recipe_hash`;
- `reward_change_policy`.

Training sample files already carry reward schema metadata. The run-control path
should make that metadata explicit enough that a later reader can tell which
reward recipe created the samples.

## First Implementation Direction

Docs only for now. When implemented, keep the first slice small:

1. Add a pure reward recipe validator.
2. Map the safe profiles to current trainer variants.
3. Store the chosen recipe in launch/attempt artifacts.
4. Reject unknown weights, negative weights, non-finite weights, and unsupported
   profile/variant combinations.
5. Do not allow mid-attempt reward changes.
6. Add tests that prove the existing profiles map to the current variants.

Do not wire this into the learner loop as a live Dict read.

## Failure Policy

Fail closed for new launches:

- missing run-control record when one was explicitly requested;
- invalid reward recipe;
- unsupported reward profile;
- unsupported weight combination;
- recipe hash mismatch;
- attempt trying to resume with a different reward recipe without an explicit
  new-attempt rule.

For an already-running attempt:

- keep the frozen reward recipe already recorded for that attempt;
- record that a newer Dict generation exists if useful;
- do not silently switch rewards.

## Open Questions

- Should custom weights be supported immediately, or should V0 allow only named
  profiles? Recommendation: named profiles first, custom weights after tests.
- Should `survival` and `final_outcome` have a tied preset? Recommendation:
  profile presets can tie them, but the stored weights should remain explicit.
- Should reward refresh ever happen at checkpoint boundary within one attempt?
  Recommendation: not in V0. Use a new attempt until replay handling is designed.
- Should reward control live in the same Dict as slot control? Recommendation:
  yes, as `curvyzero-training-run-control`, with `opponents` and `reward`
  sections.

## Anti-Patterns

- trainer reads reward weights from Modal Dict every episode;
- reward weights change without a new recipe hash;
- old and new reward definitions are mixed without an explicit rule for old
  training samples;
- slot recipe generation and reward recipe generation are collapsed into one
  vague number with no audit trail;
- reward profiles become a broad scripting language.
