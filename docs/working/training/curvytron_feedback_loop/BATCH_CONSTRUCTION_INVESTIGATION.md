# Batch Construction Investigation

Created: 2026-05-16.

Purpose: explain exactly how a training batch is constructed, how opponent
slots are sampled, and how launch manifests define those slots.

## Questions

- Where are opponent slots represented?
- Are opponent slots sampled randomly, deterministically, or on a fixed cycle?
- Is sampling per episode, per environment reset, per collector, or per learner
  batch?
- How do weights, immortality, runtime mode, and frozen checkpoint refs affect
  the selected opponent?
- Where is the actual learner minibatch assembled from collector data/replay?
- Does each learner batch have the same opponent-slot distribution, or only the
  long-run expected distribution?
- How do manifest builders construct the slot recipe and refresh pointer?

## Active Delegations

| Agent | Lane | Ask |
| --- | --- | --- |
| Ramanujan | opponent slot sampling | Completed initial map and deterministic split follow-up. |
| Harvey | learner batch construction | Completed initial map and batch-size follow-up. |
| Lagrange | manifest batch construction | Completed initial map and manifest-shape follow-up. |

## Current Notes

- Do not confuse three different “batches”:
  - launched experiment batch: the manifest rows we start on Modal;
  - opponent slot mixture: the weighted recipe sampled by environments;
  - learner minibatch: the LightZero/replay data passed into learner updates.
- The current proof canary now has a `rank1` frozen slot plus an immortal blank
  slot so leaderboard refresh can prove a real frozen-checkpoint replacement.

## Decision: 64 vs 256

Decision for the current broad lane:

- Use 256 as the actual opponent quota unit because
  `collector_env_num=256` and `n_episode=256` define the rollout production
  wave.
- Keep `batch_size=64` as the learner replay minibatch size. Do not enforce
  exact opponent composition inside every learner minibatch unless we later
  deliberately build stratified replay sampling.
- Author recipes as power-of-two slot-count bags that divide 256. Current
  recipes use 64-slot bags for readability, then runtime repeats each bag four
  times to assign all 256 collector envs.

Reasoning:

- Opponents enter the data at collection/reset time, not at learner sampling
  time.
- The learner batch is sampled from replay, which may contain old assignment
  data. Forcing every 64-sample update to be balanced would require a separate
  replay-sampler design and could bias sampling.
- Exact collector-wave quotas remove avoidable opponent-count noise while
  leaving normal replay/gradient noise alone.

## Exact Sizes

- Broad launch batch: 18 trainers.
  - 3 reward variants.
  - 3 opponent recipes.
  - 2 noise modes.
- Broad per-trainer collector batch: 256 collector environments.
  - `collector_env_num=256`.
  - `n_episode=256`, so the intended collection wave is one completed episode
    per collector env.
- Broad per-trainer learner batch: 64 replay samples per learner update.
  - `batch_size=64`.
  - This is sampled from replay by LightZero. It is not a fresh exact split of
    opponent slots.
- Canary/default proof settings may be smaller, but those are proof harness
  sizes, not the broad training defaults.

## Current Implementation Change

Implemented 2026-05-16:

- Assignment refresh now resolves an explicit slot-count bag into exact
  deterministic per-collector-env quotas.
- The split unit is collector env, not learner minibatch.
- The split mode is explicit slot counts plus a deterministic SHA256 shuffle.
- The authored recipe total must be a power of two, must be no larger than
  `collector_env_num`, and must evenly divide `collector_env_num`.
- Each collector env receives a singleton opponent mixture for the assigned
  slot, rather than all envs receiving the same weighted mixture.
- The refresh hook now rebalances the initial assignment at the first collect,
  even when the pending assignment SHA matches the already loaded initial
  assignment. This prevents a run from starting in the old expected-probability
  mode and staying there until the leaderboard changes.
- Refresh telemetry now records the split plan SHA and observed slot counts in
  the ready report. Env step info also records split metadata:
  `opponent_split_plan_sha256`, `opponent_split_env_index`,
  `opponent_split_env_num`, `opponent_split_entry_name`, and
  `opponent_split_entry_count`.

Current broad contract:

- `collector_env_num=256`.
- `n_episode=256`.
- `batch_size=64`, kept as the learner replay minibatch size.
- Recipe slot bags currently use 64 authored slots because 64 divides 256.
  Runtime repeats the bag four times and shuffles the resulting 256 env
  assignments.

Example recipe resolution:

- Authored bag `8 blank / 8 wall / 16 rank2 / 32 rank1` sums to 64.
- Runtime materialization over 256 collector envs becomes
  `32 blank / 32 wall / 64 rank2 / 128 rank1`.
- The exact env order is shuffled deterministically from assignment SHA and
  refresh index.

Important boundary:

- This makes the collection wave deterministic. It does not force every
  learner minibatch of 64 replay samples to have the same opponent split,
  because replay deliberately contains transitions from prior collection
  waves.

## Findings

### Launch Manifest Construction

Reported by Lagrange, 2026-05-16:

- The current broad launch is built by
  `scripts/build_curvytron_tonight18_manifest.py`.
- It creates 18 rows from:
  - 3 reward variants;
  - 3 opponent recipes;
  - 2 noise modes.
- Each row has deterministic ids, train kwargs, poller kwargs, artifact refs,
  and a deployed app/function target.
- `scripts/submit_curvytron_survivaldiag_manifest.py` loads the manifest,
  writes only the assignment artifacts and refresh pointers needed by selected
  rows, then spawns poller before trainer.
- The broad recipes are explicit 64-slot bags:
  - `8/8/16/32`;
  - `8/8/6/8/12/20/2`;
  - `12/4/46/2`.
- Slot types include immortal blank canvas, immortal wall-avoidant opponent,
  mortal rank slots, and optional rank1 immortal slot.
- Refresh pointers are per recipe and normally live on the control volume.
- Runtime refresh detects assignment SHA changes and resets envs with the new
  opponent assignment. As of 2026-05-16, the refresh path repeats the explicit
  slot-count bag to the collector-env count and expands it into deterministic
  per-env slot assignments.

Important caveats:

- The E2E canary proves wiring, not survival improvement or ranking quality.
- If submitter is called with a row limit, it only publishes assignments and
  refresh pointers for selected rows. That is intentional but easy to misread.
- `--own-checkpoint-opponent-refresh` is a separate diagnostic lane and does not
  use the normal control-volume assignment pointer path.

### Learner Minibatch Construction

Reported by Harvey, 2026-05-16:

- The learner batch is built by stock LightZero, not custom CurvyTron code.
- CurvyTron sets LightZero config values such as `policy.batch_size`,
  `collector_env_num`, and `n_episode`; LightZero then collects segments,
  pushes them to replay, samples a minibatch, and calls `learner.train`.
- Replay sampling is random transition-level sampling from the whole retained
  buffer. It is not opponent-stratified.
- Opponent assignment/mixture is selected at env reset, before collection for
  that episode.
- Raw `opponent_mixture_spec` still uses weighted pseudo-random selection from
  mixture seed, episode seed, and reset index when it is used directly.
- The normal leaderboard assignment refresh path now overrides that broad
  mixture with per-env singleton mixtures, giving exact collector-env quotas
  for the current assignment.
- Learner seat defaults to random per episode.
- Assignment refresh happens just before a `Collector.collect` call. It can
  reset collector envs with a new mixture, but it does not purge or stratify the
  replay buffer.

Important implication:

- Slot counts now control exact collector-env quotas at assignment refresh.
  They still do not guarantee each learner update/minibatch has the same slot
  distribution. After a refresh, learner batches may still include old-opponent
  transitions already in replay.

### Opponent Slot Sampling

Reported by Ramanujan, 2026-05-16:

- Slots are assignment/mixture entries, not separate live trainer code.
- Assignment parsing converts `entries` directly into a validated opponent
  mixture.
- Entry `weight` values are still the serialized field name, but for the normal
  leaderboard assignment path they are authored and validated as integer slot
  counts, not percentages.
- Raw mixture sampling still draws from those weights at episode reset.
- The normal refresh path now turns those integer slot counts into deterministic
  collector-env quotas before env reset.
- The env selects the opponent on `reset()`.
- Training uses dynamic env seeds, so different resets normally see different
  episode seeds/reset indices.
- Important slot fields:
  - `opponent_policy_kind`: fixed straight, proactive wall avoidant, or frozen
    LightZero checkpoint.
  - `opponent_immortal`: author-facing immortality flag.
  - `opponent_runtime_mode=blank_canvas_noop`: blank-board sentinel behavior;
    requires fixed-straight plus immortal.
  - frozen checkpoint refs must be exact immutable refs such as
    `iteration_N.pth.tar`, not mutable latest/best refs.
- `opponent_assignment_ref` and raw `opponent_mixture_spec` are mutually
  exclusive in trainer setup. Either path becomes `opponent_mixture` in env
  config.
- Refresh compares assignment SHA; if changed, it resets collector envs with the
  new mixture/context.

Risks:

- `stable_slots_v1` accepts `checkpoint_death_mode`, but current checkpoint
  slots are forced mortal in that helper. Immortal frozen slices need explicit
  mixture recipes.
- Frozen policy cache keys use entry name, so refresh relies on the refresh
  path clearing/resetting env state. Normal reset alone would not be enough if a
  same-name entry silently changed under an existing env object.

Pending:

- Confirm whether we need an operator readout for realized opponent-slot
  distribution in recent collected episodes.
