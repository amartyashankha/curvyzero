# End-To-End Dataflow

## Intended Flow

```mermaid
flowchart TD
  trainRun["Training Run"] -->|"writes exact iteration checkpoints"| checkpointVolume["curvyzero-runs Volume"]
  checkpointVolume -->|"scan train/lightzero_exp*/ckpt"| intakeScanner["Intake Scanner"]
  intakeScanner -->|"checkpoint_seen event"| intakeQueue["Modal Queue"]
  intakeScanner -->|"manifest and dedupe state"| intakeDict["Modal Dict"]
  intakeQueue --> intakeDrain["Intake Drain"]
  intakeDrain -->|"spawn rating/adaptive run"| tournamentRating["Tournament Rating"]
  tournamentRating -->|"latest.json, pair_history, scheduler_state"| tournamentVolume["Tournament Volume"]
  tournamentVolume --> publicLeaderboard["Immutable Public Leaderboard Snapshot"]
  publicLeaderboard --> livePointer["Modal Dict Pointer Cache"]
  publicLeaderboard --> selector["Opponent Assignment Selector"]
  selector --> assignment["assignment.json and audit.json"]
  assignment --> nextTrain["Next Training Launch"]
```

## Durable Truth

| Artifact | Durable? | Owner |
| --- | --- | --- |
| Training checkpoints | Yes | Training Volume |
| Intake manifest artifact | Yes | Tournament Volume |
| Rating snapshots | Yes | Tournament Volume |
| Public leaderboard snapshot | Yes | Tournament publisher |
| Assignment snapshot | Yes | Training attempt |
| Modal Dict pointer | No, cache only | Publisher/selector |
| Modal Queue event | No, coordination only | Intake scanner/drain |

## Writer And Reader Responsibilities

| Component | Writes | Reads | Must not do |
| --- | --- | --- | --- |
| Trainer | checkpoints, training artifacts | assignment snapshot | poll live leaderboard |
| Intake scanner | intake manifest, Queue events | checkpoint Volume | rank policies |
| Tournament reducer | rating snapshots | checkpoint refs, battle summaries | write training assignments |
| Public leaderboard publisher | immutable leaderboard snapshots, Dict pointer | rating snapshots | select per-run opponents |
| Assignment selector | `assignment.json`, `audit.json` | leaderboard snapshot | mutate ratings |
| Trainer launch wrapper | attempt metadata | assignment snapshot | rank/sample live |

## Current Implemented Flow

```text
checkpoint Volume -> intake manifest/Queue -> rating loop -> latest.json ->
public leaderboard snapshot -> stable_slots_v1 assignment -> trainer launch
```

This has been proven manually in tiny remote smokes. It is not production
automation yet.

Launch lifetime caveat:

```text
intake/drain command returns
-> non-detached modal run app stops
-> child game workers can be killed
-> progress exists but latest.json/summaries do not advance
```

For child tournament work that must continue after the command returns, use
`modal run --detach`, use a deployed path that keeps work alive correctly, or
wait for child completion.

## Remaining Flow Gaps

- rerun the smoke after the checkpoint-recency metadata fix;
- write the production pointer-repair runbook; a tiny remote pointer-repair
  smoke passed, but queue/stale-claim repair still needs remote proof;
- prove continuation at bounded scale;
- add safe assignment refresh policy for long-running trainers.
