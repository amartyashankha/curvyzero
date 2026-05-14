# Non-Neural Opponent Contracts

## Current State

Training can already express several non-neural opponent modes through opponent
mixtures and env config. Tournament/rating cannot yet represent those modes as
first-class players.

## Existing Training Opponent Kinds

| Kind / mode | Current meaning | Implementation notes |
| --- | --- | --- |
| `fixed_straight` | opponent always goes straight | Training mixture entry policy kind. |
| `proactive_wall_avoidant` | scripted policy steers away from walls | Uses safe-margin geometry. |
| `frozen_lightzero_checkpoint` | neural checkpoint opponent | Requires exact immutable checkpoint ref. |
| `blank_canvas_noop` | opponent hidden/inert/no-trail/no-collision/no-bonus | Runtime mode; requires `fixed_straight`. |
| `immortal` | opponent death immunity | Death mode; dirty diagnostic, not source-faithful. |

## Current Tournament Limitation

Tournament player specs are checkpoint-centric:

- normalized specs require `checkpoint_ref`;
- ratings are keyed by checkpoint id;
- loaders call checkpoint policy loading;
- roster restoration expects checkpoint refs.

Therefore scripted policies cannot appear as official leaderboard rows today
without schema and loader work.

## Proposed General Player Contract

Future tournament participants should be a tagged union.

### Neural Checkpoint Player

```json
{
  "participant_id": "ckpt-...",
  "participant_kind": "neural_checkpoint",
  "label": "blank-fast-medium i270000",
  "checkpoint_ref": "training/.../iteration_270000.pth.tar",
  "model_env_variant": "source_state_fixed_opponent",
  "policy_trail_render_mode": "browser_lines"
}
```

### Scripted Policy Player

```json
{
  "participant_id": "scripted-proactive-wall-avoidant-v0-margin20",
  "participant_kind": "scripted_policy",
  "label": "scripted wall avoidant margin20",
  "scripted_policy_kind": "proactive_wall_avoidant",
  "scripted_policy_version": 0,
  "scripted_policy_params": {
    "safe_margin": 20
  }
}
```

### Diagnostic Control

```json
{
  "participant_id": "control-blank-canvas-noop-v0",
  "participant_kind": "diagnostic_control",
  "label": "blank canvas no-op",
  "opponent_policy_kind": "fixed_straight",
  "opponent_runtime_mode": "blank_canvas_noop",
  "leaderboard_eligible": false
}
```

## Invincibility Semantics

Do not conflate:

- **source bonus invincibility**: in-game state/effect;
- **opponent death immunity**: `opponent_death_mode=immortal`;
- **invincible policy identity**: a policy that cannot die by construction;
- **training pressure modifier**: a wrapper applied to an opponent in some
  percentage of episodes.

Recommended representation for near-term training:

- use duplicate assignment entries for mortal vs immortal variants;
- record the variant in `assignment.json` and `audit.json`;
- keep invincible variants out of official Elo unless the tournament is explicitly
  diagnostic.

## Assignment Representation

Example assignment entry:

```json
{
  "name": "champion_anchor_immortal",
  "weight": 2,
  "opponent_policy_kind": "frozen_lightzero_checkpoint",
  "opponent_checkpoint_ref": "training/.../iteration_170000.pth.tar",
  "opponent_death_mode": "immortal",
  "tags": ["leaderboard", "champion", "immortal_pressure"]
}
```

Blank/no-op entry:

```json
{
  "name": "blank_canvas_noop_anchor",
  "weight": 4,
  "opponent_policy_kind": "fixed_straight",
  "opponent_runtime_mode": "blank_canvas_noop",
  "tags": ["blank", "anchor", "survival_baseline"]
}
```

## Tests Required Before Tournament Support

- non-checkpoint player specs normalize without fake checkpoint refs;
- roster hash includes participant kind and scripted params;
- game loader dispatches scripted policy vs checkpoint loader;
- resume-from-latest preserves non-checkpoint rows;
- website displays participant kind and tags;
- battle summaries record scripted/diagnostic settings;
- context hash changes when scripted params or invincibility settings change.

## Near-Term Recommendation

Use non-neural policies as assignment/training opponents now. Do not put them in
the official leaderboard until the general participant contract is implemented.
