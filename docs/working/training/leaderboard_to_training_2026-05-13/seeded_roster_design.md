# Seeded Tournament Roster And Hand-Coded Policies

## Problem

A new clean leaderboard should probably not start from neural checkpoints alone.
It may need:

- current best checkpoints from the latest-212 tournament;
- anchors from older tournaments;
- scripted or hand-coded sentinels;
- blank/no-op controls;
- explicit `opponent_immortal` overlays on training opponents.

These are not all the same kind of object. Mixing them carelessly can make the
leaderboard hard to interpret.

## Roster Types

| Type | Example | Should be leaderboard player? | Should be training opponent? |
| --- | --- | --- | --- |
| Neural checkpoint | exact `iteration_N.pth.tar` | Yes | Yes |
| Scripted wall-avoidant sentinel | `proactive_wall_avoidant` | No for official checkpoint Elo | Yes, immortal |
| Blank/no-op opponent | `blank_canvas_noop` | Diagnostic only | Yes, immortal |
| Immortal checkpoint slice | `opponent_immortal=true` on a frozen checkpoint entry | No, modifier not player | Yes, as a small explicit slice |
| Source bonus invincibility | SelfMaster/Godzilla-like source effect | No, unless modeled as environment mode | Maybe later |

## Current Code Reality

Today the tournament/rating pipeline is checkpoint-only:

- `normalize_checkpoint_spec` requires a `checkpoint_ref` or string ref;
- `rating_roster_by_checkpoint` stores `checkpoint_ref` plus model/render
  metadata;
- game workers load policies through checkpoint loading;
- resume-from-latest reconstructs players from checkpoint refs and drops rows
  without refs.

Training opponent mixtures are more general:

- `fixed_straight`;
- `proactive_wall_avoidant`;
- `frozen_lightzero_checkpoint`;
- `blank_canvas_noop` runtime;
- public `opponent_immortal` intent, with runtime death mode derived at the env
  boundary.

Therefore: non-checkpoint policies can be used as **training opponents** today,
but need tournament schema/loader work before they can be ranked as
**leaderboard players**.

## Representation Options

### Option A: Checkpoint-Only Leaderboard

Only exact neural checkpoints appear as leaderboard rows.

Pros:

- simplest;
- current code already assumes checkpoint refs;
- no fake checkpoint ids.

Cons:

- hand-coded policies cannot be rated directly;
- training assignment needs separate scripted sentinel entries.

Use for: next clean leaderboard if speed matters.

### Option B: General Player Spec

Extend tournament roster from `checkpoint_ref` to `player_spec`:

```json
{
  "player_id": "scripted-proactive-force-field-margin20-v0",
  "player_kind": "scripted_policy",
  "label": "scripted wall avoidant margin20",
  "scripted_policy_kind": "proactive_wall_avoidant",
  "safe_margin": 20,
  "rating_eligible": true,
  "training_eligible": true
}
```

Pros:

- leaderboard can rate neural and non-neural policies together;
- scripted baselines become visible anchors.

Cons:

- requires tournament loader and rating roster changes;
- must prevent checkpoint-specific code from assuming `checkpoint_ref`;
- needs clear context hashing.

Use for: durable public leaderboard once contracts are ready.

Required changes:

- allow non-checkpoint player specs during normalization;
- include `player_kind`, scripted policy version, and parameter hash in roster
  identity;
- include scripted identity in pool/context hash;
- dispatch game loading to scripted policy factories as well as checkpoint
  loaders;
- update website/checkpoint drilldown language from checkpoint-only to player
  when needed;
- preserve scripted rows across continuation from `latest.json`.

### Option C: Scripted Sentinels Outside Leaderboard

Leaderboard remains neural-only; assignment selector appends scripted, blank, and
small immortal checkpoint-slice entries to training mixtures.

Pros:

- lowest risk for leaderboard correctness;
- avoids non-checkpoint player schema work before overnight runs.

Cons:

- scripted opponents are not ranked;
- less visibility into how strong scripted policies are.

Use for: immediate next overnight if training needs these pressures before
tournament schema generalization.

## Immortal Opponent Design

Important distinction:

- A **policy** is the thing that chooses actions.
- `opponent_immortal` is a separate death-immunity flag on a training entry.

For training, the clean near-term design is a separate weighted slot entry:

```json
{
  "name": "leaderboard_champion_immortal_pressure",
  "weight": 10,
  "opponent_policy_kind": "frozen_lightzero_checkpoint",
  "opponent_checkpoint_ref": "training/.../iteration_270000.pth.tar",
  "opponent_immortal": true,
  "tags": ["leaderboard", "champion", "immortal_pressure"]
}
```

Current caveat: `opponent_immortal=true` suppresses opponent death but does not
make a clean source-faithful opponent; the opponent can still move out of
bounds and leave trails. The lower-level env may report derived
`opponent_death_mode=immortal` telemetry, but slot specs should not use that as
their public intent field.

There is no clean global "make any selected policy immortal X% of the time" knob
today. The clean current path is to duplicate mixture entries with different
`opponent_immortal` values and weights. Keep total immortal exposure around
`20-30%`, generally not above about `30%`.

## Recommended Near-Term Plan

For the next overnight run:

1. Do **not** make immortality a leaderboard player yet.
2. Keep the new public leaderboard checkpoint-only or neural-checkpoint-first.
3. Add scripted/blank/immortal pressures through assignment snapshots or
   manifest-defined opponent mixtures. Blank and hard-coded sentinels must be
   immortal; frozen checkpoint opponents should be mortal unless a small explicit
   immortal slice is intended.
4. Include explicit metadata in assignment audit:
   - `opponent_policy_kind`;
   - `opponent_runtime_mode`;
   - `opponent_immortal`;
   - `opponent_trail_mode`;
   - probability/weight;
   - whether the entry is leaderboard-rated or scripted.

For a future clean public leaderboard:

1. Start checkpoint-only with best existing neural policies and anchors.
2. Add scripted policies only after general player specs are implemented.
3. Treat immortal/modifier rows as diagnostics unless
   their exact game/evaluator semantics are documented and hashed.

## Open Questions

- Should the first clean public leaderboard include scripted wall-avoidant
  players, or should those stay assignment-only?
- Should immortal rows be allowed in public ratings, or only diagnostic
  tournaments/training assignments?
- Should we later add a first-class "immortal overlay probability" knob, or keep
  explicit duplicated mixture entries?
- How should a non-checkpoint player appear in website labels and rating rows?

## Test Requirements Before General Player Specs

- tournament accepts a scripted `player_spec`;
- rating row has stable `player_id`, `player_kind`, label, and context fields;
- battle/game summaries record scripted policy settings;
- website can display non-checkpoint players;
- assignment selector can include both checkpoint and scripted rows;
- context hash changes when scripted policy parameters change.
