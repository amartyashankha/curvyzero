# Operating Patterns

Status: active operating contract for the reward/RND H100 campaign.

## Core Posture

Move aggressively, but keep every claim small enough to audit. The H100 budget
lets us run many lanes at once; it does not let us merge their conclusions.

Default to broad, embarrassingly parallel sweeps when the code path has a
credible launch surface. Canaries are health sentinels and kill-switch tests,
not a reason to serialize every scientific question. Prefer one broad manifest
with stock/meter/positive controls, replicas, and early stop criteria over a
long chain of tiny one-off runs.

Runtime capacity is tiered by intended wall-clock, not one flat number:

| Intended runtime | Default active H100 posture |
| --- | --- |
| `<=2h` | Broad, timeboxed sweep. The prepared 90-row packet is reasonable if capacity is explicitly clear. More than 100 simultaneous H100s requires explicit operator override and a short timeout. |
| `2h-8h` | Narrow to at most 40 active H100 rows. Keep RND, stock/meter, and non-RND controls represented. |
| `8h+` | Narrow to 10-20 active H100 rows. Spend only on rows with clean health and the most informative controls. |

Broad means fill available capacity with interpretable lanes for the chosen
runtime tier. The capacity auditor is a conservative task-count proxy, not a
resource allocator. If current capacity is ambiguous, stage at or below the
capacity audit's conservative room or wait for an explicit operator capacity
decision.

Parallelism is a rule at every level:

- parallel lanes: reward/RND, non-RND controls, checkpoint anchors, monitoring,
  temporal abstraction, and bounded PPO/Puffer/planner spikes
- parallel rows: weights, replicas, reward arms, opponent recipes, and cadence
  knobs inside one audited manifest where possible
- parallel readouts: status, eval curves, RND metrics, action histograms, and
  artifact integrity checks prepared before launch
- parallel agents: delegate bounded audits or implementation slices with
  disjoint scopes, then integrate through this doc set

Do not use "parallel" to mean uncontrolled factorial explosion. Each extra axis
must name the claim it answers, the controls that make it interpretable, and the
runtime tier it can afford.

Non-RND testing is mandatory, not optional. Every RND claim needs matched
`none` and `rnd_meter_v0` controls, and the overall campaign must keep
independent non-RND extrinsic reward, exact-ref curriculum, and cadence/support
lanes alive. A strong RND result is uninterpretable if the non-RND baselines
were not healthy, long enough, and artifact-complete.

Every live lane must name its claim class:

| Claim class | Meaning |
| --- | --- |
| implemented | Code path exists. |
| test-passed | Unit/focused tests passed. |
| manifest-ready | Local manifest validates and names exact rows. |
| launched | Modal work was submitted. |
| healthy | Heartbeats, progress, checkpoints, eval, and required metrics exist. |
| learning-signal | Best/AUC/retention moved versus lane-local controls. |
| promoted | Reproducible result survives the promotion gate. |

Do not skip classes in language. "RND is implemented" is not "RND works."
"A row is faster" is not "the compact trainer is done." "Tournament rank moved"
is not "the reward is better."

## Source Of Truth

Use current repo contracts as the source of truth before launching:

- reward constants and compact reward contract:
  `src/curvyzero/contracts/curvytron.py`
- reward support mapping:
  `src/curvyzero/training/reward_contracts.py`
- source-state reward/runtime behavior:
  `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- RND implementation:
  `src/curvyzero/training/exploration_bonus.py`
- grouped submitter:
  `scripts/submit_curvytron_survivaldiag_manifest.py`
- Wave A packet auditor:
  `scripts/audit_curvytron_wave_a_launch_packet.py`
- Wave A capacity auditor:
  `scripts/audit_curvytron_wave_a_capacity.py`
- checkpoint anchor policy auditor:
  `scripts/audit_curvytron_checkpoint_anchor_policy.py`
- staged launch planner:
  `scripts/plan_curvytron_wave_a_staged_launch.py`

Older docs are evidence and memory, not launch contracts. If a prior note
conflicts with the current manifest or source contract, resolve it explicitly in
the active doc set before spending GPUs.

## Lane Separation

Run lanes in parallel, read them separately:

- RND lane: intrinsic reward hypothesis.
- Static reward lane: extrinsic reward hypothesis.
- Cadence/support lane: training-dynamics hypothesis.
- Static exact-ref recipe lane: production-style curriculum without live
  leaderboard feedback.
- Tournament lane: head-to-head selection hypothesis.
- Compact speed lane: throughput hypothesis.
- Pure/raw-tick MCTS lane: quarantined search research and profiling hypothesis,
  not a production learning lane.
- Long-term planning lane: recurrence, macro-actions, dense trajectory
  planners, reanalysis, or Gumbel/MuZero search under separate denominators.

Never use a win in one lane to close another lane unless a bridge experiment
was designed for that purpose.

## Pure MCTS Quarantine

The current conceptual read makes raw-tick pure MCTS a suspected misfit for
production learning spend. It may still be useful as a profiling/control lane,
but only with an explicit profile split:

- policy only
- recurrent/model only
- tree only
- full search

Do not launch or keep long production-style H100 runs whose only thesis is
"more raw-tick MCTS will learn" unless they already have policy-only controls,
equal GPU-second/action-latency comparisons, D2H/H2D copy counts, root batch,
simulation count, max depth, and action-latency distribution.

Before stopping anything, prove it is actually a CurvyTron pure/raw-tick MCTS
run. Required evidence:

- Modal app description or run id maps to CurvyZero/CurvyTron.
- Recent logs or manifest metadata mention CurvyTron, LightZero/MuZero,
  `train_muzero`, MCTS/search settings, or the source-state env.
- It is not an unrelated timing, LLM, dashboard, benchmark, or Flash playable
  service.

If a verified live pure MCTS run lacks the gates, stop it after preserving the
latest checkpoint, profile artifact, exact app id, run id, attempt id, and stop
reason. Never infer from capacity pressure alone.

This does not kill all search work. It moves search into measured research:
macro-action search, dense trajectory planning, reanalysis/distillation, or
Gumbel/joint-action MuZero can return only through separate manifests and
promotion gates.

## Long-Term Planning And RND

Use `LONG_TERM_PLANNING_RND_STRATEGY.md` before adding planning or curiosity
rows. The standing split is:

- recurrence and value learning carry most long-horizon memory
- strategic observations and auxiliary targets expose territory and reachable
  space
- macro-actions/action repeat make explicit decisions cover meaningful geometry
- dense planners search over macro-action sequences or selected states
- RND is training-only exploration pressure, not a production reward

For the policy-first baseline branch, use
`../curvytron_flash_comparison_2026-06-23/PUFFERLIB_STRATEGY.md` as the current
PPO/Puffer gap analysis. Puffer/Flash rows remain PPO or raw-env controls unless
a bridge experiment explicitly makes them comparable to CurvyZero MuZero/search
or reward/RND quality evidence.

Do not combine planner actions with vanilla PPO old-policy logprobs. If search
chooses actions, the bridge is imitation, reanalysis, or a separate search
training objective.

Any PPO/self-play/planner-hybrid row must declare its behavior-policy contract:
`ppo`, `imitation`, `off_policy_mixture`, or `search_training`. It must also
declare reward perspective, value perspective, action-mask/logprob convention,
entropy or top-action collapse threshold, and opponent-pool/frozen-checkpoint
policy if it makes a self-play claim.

Do not use RND as proof of planning. A positive RND result means novelty
pressure helped the training lane only after stock, meter, survival, retention,
and action-distribution controls agree.

## Before Launch

For each lane, create a small run note or add a section to this doc set with:

- manifest path
- row count
- selected row ids
- run-id prefix
- attempt-id prefix
- exact submit command
- expected H100 count
- expected first health artifact
- lane-local controls
- stop condition
- first decision horizon

Hard launch gates:

- For Wave A, rerun the packet auditor and require `ok=true` with
  `launch_artifacts=[]` before any capacity check or approval request.
- For Wave A, rerun the capacity auditor and require explicit operator review
  if it reports `operator_capacity_review_required`.
- Dry-run grouped submitter.
- Confirm row count and row ids.
- Confirm intended runtime tier and the active H100 row cap for that tier.
- Confirm initial policy seed policy: historical best-known seed versus
  current launchable repair seed, with an audit artifact.
- Confirm the two checkpoint roles separately: opponent rank slots come from
  `--checkpoint-refs-file`; learner seed comes from
  `--initial-policy-checkpoint-ref` or the documented fallback.
- Confirm compute is `gpu-h100-cpu40` where intended.
- Confirm at least one active non-RND lane is launching or already healthy
  unless the launch is explicitly an RND plumbing-only preflight.
- Confirm no-refresh controls have no refresh refs and interval `0`.
- Confirm RND enabled rows require RND metrics.
- Confirm stock RND control rows do not require RND metrics.
- Confirm seeded checkpoint refs are exact immutable `iteration_N.pth.tar`
  refs, not `latest` aliases.
- For PPO/self-play/planner-hybrid rows, confirm behavior-policy contract,
  reward/value perspective, action mask convention, and planner-action training
  contract.
- For environment, macro-action, planner, PPO/Puffer, or self-play branches,
  confirm the relevant `CURVYTRON_GAME_MECHANICS_GATES.md` checks are satisfied
  or explicitly relabel the row as a diagnostic/non-source-faithful control.
- Confirm the launch is full-manifest or explicitly uses
  `--allow-partial-launch` with pasted row ids.

Large Wave A launches also require:

- One command that prints the intended active row count before launch.
- A submitter dry-run whose summary says `dry_run=true`.
- Zero assignment writes for no-refresh/static lanes.
- Zero refresh pointer writes for no-refresh/static lanes.
- A syntax audit for every checkpoint-ref manifest.
- A Modal existence audit for exact checkpoint refs before launch.
- A checkpoint-anchor audit showing whether non-RND rows use the historical
  r18fresh rank-1 seed or the top4nz repair seed.
- A staged launch plan artifact for the chosen runtime tier.
- A chosen status command and note path before the first FunctionCall is
  submitted.
- A known stop/cleanup procedure keyed by run-id prefix.

Do not use live leaderboard slots, `latest` aliases, or moving assignment
pointers in Wave A static quality rows. Leaderboard feedback belongs after
nonzero candidate checkpoints exist.

## Wave A Launch Contract

Current prepared candidates are recorded in `WAVE_A_MANIFESTS.md`.

Launch order should favor interpretability:

1. Static exact-ref reward isolate and RND blank sweep are independent and
   should launch broadly if capacity is clear.
2. Long-horizon pretrained replicas should reuse the same exact initial
   checkpoint; vary only run ids, selected rows, or explicit knob settings.
3. Cadence/support rows should be launched as a support question, not as a
   reward variant conclusion.

For short health/breadth sweeps, keep a small H100 buffer for relaunches, failed
startup, or immediate RND bridge rows. For medium and long runs, narrow before
the runtime crosses the tier boundary; do not let a short 90-row launch quietly
become a 4-hour or 8-hour campaign.

When a path still needs a canary, include the canary as:

- a small meter/stock row group inside the broad manifest, or
- a same-hour preflight group whose launch command and stop criteria are already
  paired with the broad manifest.

Do not let a healthy canary wait idle overnight while the broad sweep remains
unlaunched.

Non-RND lanes have their own priority. If capacity forces a split launch, do
not spend all capacity on RND positives first. Keep at least the static
exact-ref reward isolate, or a documented subset of it, moving alongside the
RND lane.

## During Launch

Do not wait passively. While Wave A is starting, prepare the first fallback
manifest for the highest-risk lane.

Monitoring cadence:

| Time or horizon | Read |
| --- | --- |
| 0-30 minutes | health only: heartbeat, progress, checkpoint/eval poller, RND metrics |
| 30k-50k | first weak learning read, action collapse, eval wiring |
| 100k-170k | first useful AUC/best/retention read |
| 240k-300k | retention and promotion read |

At each pass, write a compact note:

```text
lane:
manifest:
rows:
health:
best/AUC/latest:
controls:
failure class:
decision:
next artifact:
```

## Decision Hygiene

Use these rules even when the first graph looks exciting:

- Compare raw trainer reward only within the same reward variant and RND weight.
- Use survival AUC, best-so-far, and latest-vs-best retention across reward
  variants.
- Ignore raw tournament rank until exposure is shown.
- Exclude or label `iteration_0` seed checkpoints in learned-policy reads.
- Treat GIFs as qualitative failure detectors, not promotion evidence.
- Preserve best checkpoints when latest regresses.
- Relaunch with one changed variable.
- Interpret RND only after checking stock `none`, `rnd_meter_v0`, and the
  independent non-RND reward/cadence lanes for health and horizon.
- Let healthy rows reach the planned long horizon before treating flat early
  survival as evidence of no learning.
- If a lane fails health, fix health before interpreting learning.
- If every reward lane improves then regresses, move to cadence/support instead
  of inventing another reward.

## RND Operating Rules

RND is now first-class for experiments and still experimental for conclusions.

Required controls:

- `none`
- `rnd_meter_v0`
- positive `rnd_replay_target_v0` weights

Required RND readout:

- `rnd_reward_model_metrics.jsonl`
- latest valid metrics snapshot or JSONL fallback
- predictor loss finite
- target network hash stable
- target reward unchanged for meter rows
- target reward changed for positive rows in the expected order of magnitude
- positive rows compared against both stock and meter controls

RND promotion ladder:

1. Blank-canvas positive weight beats stock and meter on AUC or best-so-far.
2. The result repeats across at least two replicas.
3. Latest does not collapse relative to best.
4. Action distribution does not show degenerate novelty seeking.
5. Fixed-opponent RND extension preserves the gain.
6. Tournament exposure does not contradict the survival read.

Historical RND "blocked" means blocked for recommendation/promotion, not
blocked for a controlled H100 experiment under this doc set.

## Modal And Artifact Rules

- Volume files are durable truth.
- Dicts and Queues are coordination, not history.
- FunctionCall ids prove submission, not completion.
- Scheduling a tournament or poller is not proof that it reduced to durable
  results.
- A status tool is the first operator surface; raw Volume reads are for
  forensics.
- A stale dashboard can be useful for noticing problems but should not decide
  promotion.
- Every large launch needs a path from row id to run id to checkpoint refs to
  eval rows to tournament rows.

## Claim Ledger

Every promoted claim should eventually have a one-line ledger row:

| Date | Lane | Claim | Evidence | Residual risk | Decision |
| --- | --- | --- | --- | --- | --- |

Use this to prevent partial proof from becoming folklore. If the evidence is
"tests only," the decision cannot be "promote." If the evidence is "one seed,"
the residual risk must say seed noise.

## Subagent Pattern

Use subagents for bounded, parallel questions:

- RND archaeology or metrics readout.
- Reward curve analysis.
- Tournament exposure audit.
- Manifest/submitter guard audit.
- Fake-progress critique.

Each subagent packet must say:

- question
- scope
- files or artifacts to inspect
- output shape
- no-edit or owned write scope

The main thread integrates. Subagent output that is not written into the active
doc set has not reduced future confusion.

## Stop And Preserve

Stop or pause a lane only after preserving:

- manifest
- submit dry-run or launch record
- status output
- latest progress
- latest eval curve JSON
- RND metrics, if relevant
- exact reason for stop

Stopping is not failure when it prevents ambiguous spend. The bad outcome is a
large run whose artifacts cannot answer why it won or lost.
