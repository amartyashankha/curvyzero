# LightZero Dummy Pong Target Policy Persistence Audit

## Question

The new oracle result says action id `2` is truly `down`, and `down` is the
score-winning action in sampled contact-pressure states. At low MCTS sims,
`iteration_0` can produce root visits like `[1, 1, 0]`, giving useful `down`
zero target mass.

We need to know where actual training targets are stored, how to inspect them,
and whether collect targets also zero out useful `down`.

## Storage Read

LightZero stores collect policy targets in memory during collection:

- Stored policy target: `GameSegment.child_visit_segment`
- Executed collect action: `GameSegment.action_segment`
- Source: `lzero/worker/muzero_collector.py` calls `store_search_stats(...)`
  before appending the executed action.
- Source: `lzero/mcts/buffer/game_segment.py` normalizes visits inside
  `GameSegment.store_search_stats`.
- Training source: `MuZeroGameBuffer` samples `child_visit_segment` as the
  non-reanalyzed policy target, and may overwrite it during reanalyze.

Current CurvyZero Modal train outputs do not persist these game segments. The
wrapper mirrors summaries, episode telemetry, training signal JSON, artifact
manifest, config/command/stdout/stderr, and checkpoints. It does not currently
save the replay buffer or `GameSegment` objects.

## Script

Added:

```sh
scripts/audit_lightzero_dummy_pong_target_policy.py
```

Main artifact-inspection command:

```sh
PYTHONPATH=src uv run python scripts/audit_lightzero_dummy_pong_target_policy.py inspect-stored \
  --path artifacts/local/contact-pressure-modest-rung-2026-05-09 \
  --path artifacts/local/lightzero-dummy-pong-scorecard-summaries-2026-05-09 \
  --state-seed 20260510 \
  --state-seed 20260515 \
  --state-seed 20260523 \
  --max-env-step 64 \
  --format md
```

Local result:

```text
Files inspected: 7
Records with stored policy targets: 0
Stored location in LightZero: GameSegment.child_visit_segment
Executed action location in LightZero: GameSegment.action_segment
Current CurvyZero mirror: ... no replay buffer/game segment artifact is mirrored by the current wrapper.
```

So, for existing mirrored train artifacts, there is no persisted
`visit_count_distributions` replay artifact to inspect yet.

## Live Collect Fallback

This is not stored replay evidence. It reruns collect-mode on the sampled
states to check what the collection API returns at low sims.

```sh
PYTHONPATH=src uv run --with LightZero==0.2.0 \
  python scripts/audit_lightzero_dummy_pong_target_policy.py \
  --checkpoint lightzero:contact3=/private/tmp/curvy-lz-contact-modest-iter3.pth.tar \
  --state-seed 20260510 \
  --state-seed 20260515 \
  --state-seed 20260523 \
  --rows 3 \
  --max-env-step 64 \
  --num-simulations 2 \
  --collect-repeats 1 \
  --format md
```

Local output:

| State | Oracle | Eval visits | Collect output |
| --- | --- | --- | --- |
| `20260510` | `down` | `[1,1,0]` | `up:[1,1,0]` |
| `20260515` | `down` | `[1,1,0]` | `up:[1,1,0]` |
| `20260523` | `down` | `[1,1,0]` | `up:[1,0,1]` |

Read: live collect-mode at `num_simulations=2` can also give useful `down`
zero mass. In this fallback run, two of three collect targets zeroed out
`down`; eval-mode zeroed out `down` in all three. The executed action was also
not `down`.

## Answer

Actual training targets are stored in LightZero `GameSegment.child_visit_segment`
while training is running. They are not currently saved by our Modal wrapper, so
the existing artifacts cannot answer historical stored-target questions.

To make this audit complete, persist collected `GameSegment` summaries or a
compact JSONL sidecar with:

- encoded observation or raw Pong observation
- `visit_count_distributions`
- selected/executed action
- action mask and `to_play`
- env seed, episode, step, and reset metadata

Then run `inspect-stored` on that sidecar and gate the contact-pressure
curriculum on target mass for useful `down`, not on executed-action histograms.
