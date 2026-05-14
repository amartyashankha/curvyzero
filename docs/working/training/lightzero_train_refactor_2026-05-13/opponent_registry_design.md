# Opponent Registry Design

Purpose: keep future frozen-opponent selection outside the trainer while still
letting training use tournament-produced opponents.

## Plain Goal

The training loop should not know how opponents were chosen. It should receive
a small resolved opponent description and pass it into the existing environment
configuration.

The tournament/control plane can decide which checkpoints are strong, recent,
old, scripted, blank, or otherwise useful. The trainer should only consume a
frozen assignment snapshot.

Do not make a running `train_muzero` job poll a live tournament or Modal Dict
for changing opponents. If opponents need to refresh, do it at an attempt
boundary or launch a new attempt with a new assignment snapshot.

## Naming

Use these terms to avoid confusing this with the neural network policy:

| Term | Meaning |
| --- | --- |
| Leaderboard | public tournament output: ranked checkpoint refs plus metadata |
| Opponent selection strategy | run-level rule for choosing opponents from the leaderboard |
| Assignment snapshot | concrete frozen opponent set chosen from a strategy at one time |
| Assignment refresh | creating a new snapshot, usually at an attempt/checkpoint boundary |

## Proposed Ownership

| Owner | Responsibility |
| --- | --- |
| Tournament/control plane | ranks policies and writes leaderboard records |
| Selection controller | applies one run's opponent selection strategy to the leaderboard |
| Opponent registry | stores named opponent descriptors and immutable checkpoint refs |
| Assignment snapshot | durable record of the exact opponent set a run consumes |
| Trainer scaffolding | reads one assignment snapshot or explicit spec, validates it, passes it into env config |
| `lightzero_checkpoints.py` | resolves checkpoint candidates and latest/immutable checkpoint path rules |
| Environment/opponent provider | executes the selected opponent behavior during episodes |
| Eval/GIF jobs | receive the same resolved opponent spec as training |

## Registry Shape

Keep the registry as data. A future Modal Dict entry can point at volume refs,
but the in-process trainer should first see ordinary Python data. The durable
training record should be a volume artifact or launch payload, not a mutable
Dict value that can change while training is running.

Example descriptor:

```json
{
  "schema_id": "curvyzero_opponent_registry/v0",
  "registry_id": "curvytron-main-overnight",
  "entries": [
    {
      "id": "recent_tournament_winner_001",
      "kind": "frozen_lightzero_checkpoint",
      "checkpoint_ref": "training/.../iteration_120000.pth.tar",
      "weight": 50,
      "tags": ["recent", "tournament_winner"]
    },
    {
      "id": "blank_canvas",
      "kind": "passive_immortal",
      "weight": 25,
      "tags": ["scripted", "no_trails"]
    }
  ]
}
```

The exact schema can change, but the boundary should not.

## Trainer Interface

Preferred future input:

```text
--opponent-assignment-ref <volume json ref>
--opponent-registry-ref <registry id or volume json ref>  # optional builder input
--opponent-registry-selection <small selection name>       # optional builder input
```

The trainer resolves that into one `ResolvedOpponentSpec` before building the
LightZero config, then records the exact consumed assignment in command metadata.

Plain resolved shape:

```text
schema_id
entries
selection_policy
assignment_id / source_epoch
source_ref
resolved_at
```

For checkpoint entries:

```text
opponent_policy_kind = frozen_lightzero_checkpoint
opponent_checkpoint_ref = immutable iteration_N.pth.tar ref
opponent_checkpoint_path = resolved mounted path
opponent_checkpoint_file = file summary
```

For scripted entries:

```text
opponent_policy_kind = fixed_straight | proactive_wall_avoidant | passive_immortal
scripted metadata only
```

## What Not To Put In The Trainer

- tournament ranking;
- Modal job spawning for tournaments;
- checkpoint loading beyond path validation;
- dynamic policy selection during a LightZero step;
- environment movement rules;
- GIF-only metadata enrichment;
- hard-coded recent/mid/old checkpoint strings.
- live Dict polling during training;
- tournament eligibility or rating logic;
- opponent refresh cadence.

## Current Code Pressure Points

Today these functions are the boundary risk:

```text
_resolve_opponent_checkpoint_for_env
_resolve_opponent_mixture_for_env
_reject_mutable_frozen_opponent_checkpoint_ref
```

Important current risk: mixture entries reject mutable frozen refs, but the
top-level frozen-opponent path should enforce the same immutable
`iteration_N.pth.tar` rule before external assignment control makes mutable refs
more dangerous.

Status: fixed for the trainer entry points. Mixture entries and top-level
`opponent_checkpoint_ref` now both reject mutable refs and non-iteration names.

The manifest scripts also hard-code checkpoint refs:

```text
scripts/build_curvytron_opponent_mixture_manifest.py
scripts/build_curvytron_stock_train_manifest.py
scripts/build_curvytron_survivaldiag_manifest.py
```

Do not push registry behavior into the 10k-line Modal trainer. Add a small
training helper module first, then make the trainer call it.

## Current Active Hard-Coded Refs

The clearest active problem is in:

```text
scripts/build_curvytron_opponent_mixture_manifest.py
```

It defines:

```text
DEFAULT_RECENT_OPPONENT_CHECKPOINT_REF
DEFAULT_MID_OPPONENT_CHECKPOINT_REF
DEFAULT_OLD_OPPONENT_CHECKPOINT_REF
```

Those point at one concrete historical run under
`train/lightzero_exp/ckpt/iteration_N.pth.tar`. They enter the script as CLI
defaults, become `recent` / `mid` / `old` component refs, and are inserted into
mixture entries.

The trainer path is less wrong than the manifest defaults: it already accepts
an `opponent_mixture_spec`, validates exact immutable `iteration_N.pth.tar`
refs, resolves mounted refs, and passes the resolved mixture through train,
eval, and GIF metadata. Keep that env-config contract stable while replacing
where the refs come from.

## Replacement Target

Smallest useful replacement:

1. registry/control-plane produces named records for `recent`, `mid`, `old`,
   scripted, blank, and passive entries;
2. manifest builder reads those records instead of embedding default checkpoint
   refs;
3. output rows still contain a fully resolved immutable `opponent_mixture_spec`;
4. trainer continues to consume the resolved spec.

This avoids making the trainer query tournaments or reason about which
opponents are good.

## Candidate Module

```text
src/curvyzero/training/opponent_registry.py
```

Implemented first slice:

```text
parse_opponent_assignment_snapshot(data) -> dict
```

This parser accepts a frozen assignment snapshot and returns the existing
opponent-mixture contract under `opponent_mixture`. It rejects mutable or
non-iteration frozen checkpoint refs. It does not read Modal Dict, reload
volumes, rank policies, or load checkpoints.

A later Modal adapter can read a Modal Dict or volume JSON and pass the raw data
into the pure helper.

## Refresh Design

Simplest safe design:

1. tournament writes a leaderboard;
2. each training run has an opponent selection strategy;
3. an external selection controller periodically reads the leaderboard and
   writes a new assignment snapshot;
4. a new training attempt starts or resumes from a checkpoint using that
   snapshot;
5. the trainer records the exact snapshot it consumed.

This keeps long-running `train_muzero` jobs reproducible. If we later need
in-run refresh, make it an explicit checkpoint-boundary feature with tests and
metadata, not a hidden per-step poll.

## Test Gates Before Implementation

- registry parser accepts frozen checkpoints and scripted entries;
- registry parser rejects mutable refs like `latest` or `ckpt_best`;
- assignment parser converts to the existing opponent-mixture contract;
- manifest builder has no baked historical run refs for recent/mid/old;
- manifest builder can produce rows from registry records;
- trainer config receives the resolved spec without changing LightZero train
  semantics;
- eval/GIF receive the same resolved spec as training;
- broad `lightzero_exp*` checkpoint refs are resolved through
  `lightzero_checkpoints.py`, not hand-built strings.

## Design Decision

Use a registry/control-plane interface to remove hard-coded opponent refs from
training manifests. Do not make the trainer responsible for choosing good
opponents. The trainer should consume a resolved opponent spec and remain close
to stock LightZero.
