# Training Lessons For CurvyTron - 2026-05-09

Purpose: collect the Pong / LightZero / Modal lessons that should transfer to
CurvyTron. This is not a new experiment log and not a claim that Pong is solved.

## Short Read

CurvyTron should inherit the discipline, not the confusion.

Keep two separate lanes:

- official reference lane: stock LightZero/ALE Atari Pong, used to check how the
  library behaves in its native visual setup;
- custom bridge lane: project-owned games like dummy Pong and later CurvyTron,
  used to prove our env contract, telemetry, checkpointing, and scorecards.

Do not mix their results. Official Atari can justify library and visual-stack
confidence. It cannot prove the CurvyTron wrapper is correct. Custom dummy Pong
can expose adapter and target bugs. It cannot claim official Atari parity.

## Two-Lane Control And Custom Bridge Pattern

Use the official lane as the outside control.

For Pong, this meant stock LightZero Atari Pong:

- ALE/Gym environment;
- four stacked grayscale frames;
- convolutional MuZero;
- six Atari actions;
- official-style search, batch, replay, and checkpoint patterns.

Use the custom lane as the bridge into CurvyTron.

For dummy Pong, this meant:

- project-owned env step/reset contract;
- one LightZero ego action with the opponent supplied by the wrapper;
- explicit action ids;
- sparse game reward;
- project-owned sidecars and scorecards;
- Modal checkpoints and eval jobs that can be rerun outside the trainer.

For CurvyTron, follow the same split. Keep one official LightZero reproduction
lane alive as a sanity check, but make CurvyTron quality claims only from the
CurvyTron lane.

## Target-Sidecar Observability

Do not infer training targets from executed actions.

The key LightZero lesson is simple: MuZero trains the policy head toward MCTS
root visit distributions, not toward the exploratory action that was finally
executed in the environment.

So CurvyTron target sidecars need row-level data:

- executed action;
- MCTS visit distribution / normalized policy target;
- root value and searched value when available;
- reward and done flag;
- state id or compact observation hash;
- config snapshot;
- seed, episode index, and step index;
- oracle or diagnostic action labels when a local oracle exists.

The question to answer is not "did collection ever try left/right/straight?"
The question is "did the target give useful mass to the action that should have
been searched or learned in this state?"

## Survival Telemetry

Keep environment reward honest.

For dummy Pong, the right eval contract was:

```text
score:       +1
lose:        -1
no score:     0
timeout:      0 with truncation telemetry
```

CurvyTron should follow the same rule. Death, survival, and timeout need to be
visible, but survival should not quietly become the environment reward unless a
separate ablation explicitly says so.

Every CurvyTron scorecard should report:

- wins/losses or terminal outcome;
- raw game return;
- survival steps;
- mean, median, p90, max, and standard deviation of survival;
- truncation rate;
- shaped loss-delay or survival diagnostic if used;
- action histograms;
- terminal cause.

Never reduce an early sparse-control read to only "0 wins." If a checkpoint is
not winning, survival and terminal-cause telemetry say whether it is learning
anything useful or only failing differently.

## Official-Vs-Custom Discrepancy Mapping

Before scaling CurvyTron, map the ways the custom setup differs from the
official control.

For Pong, the big differences were:

- stacked visual frames versus small tabular or flat raster features;
- conv model versus MLP model;
- six Atari actions versus three project actions;
- official search counts versus tiny smoke search counts;
- official replay/update scale versus diagnostic caps;
- stock evaluator logs versus project sidecars and scorecards;
- official support/config defaults versus custom sparse rewards;
- true ALE Pong dynamics versus project-owned toy dynamics.

For CurvyTron, make this map early. It should include observation shape, frame
history, action ids, terminal semantics, reward scale, support scale, MCTS
settings, replay settings, model family, checkpoint format, and eval path.

If a custom run fails, use the map to decide whether the failure is likely from
the game, the wrapper, weak search, wrong support scale, missing history, or
too little data.

## Visual History And Frame Stack

Single-frame raster is a smoke test, not a serious visual-control claim.

Dummy Pong showed why: one flat raster can show where the ball is, but not
where it is going. CurvyTron will be even more history-dependent because trails,
gaps, heading, speed, and opponent motion matter.

CurvyTron observations should be designed around:

- stacked recent frames or another explicit temporal encoding;
- ego heading or ego-aligned crop;
- separate semantic channels instead of magic integer pixels;
- fixed action ids, likely left / straight / right;
- schema ids attached to every checkpoint and replay row.

Use simple tabular or oracle features only as debugging lanes. The main visual
bridge should have history before it is treated as a learning result.

## Support Scale

For small sparse custom rewards, prove the compiled support scale before
training.

The dummy Pong risk was that our requested support ranges could be logged while
the pinned LightZero version still compiled a broad `support_scale=300` and
large reward/value heads. That is a bad match for rewards near `-1`, `0`, and
`+1`.

Transfer rule for CurvyTron:

- inspect the compiled policy/model config, not only the patch surface;
- log both requested fields and compiled fields;
- include `support_scale`, reward support size/range, and value support
  size/range in every train summary;
- fail or mark the run as config-invalid if compiled support does not match the
  environment reward/value scale;
- keep official Atari separate, because broad visual Atari defaults are not a
  free pass for tiny custom sparse rewards.

Do this before spending training budget. Scaling a run with the wrong support
scale can make weak values and bad MCTS targets look like a game problem.

## MCTS Target Quality

Search quality is part of the label quality.

The important dummy Pong failure was not that `down` was illegal. It was legal,
but low-simulation MCTS could assign zero visit mass to `down` in states where
`down` was the useful action. Training on that target teaches the wrong policy.

CurvyTron should not use two-simulation search as a learning result. Treat it as
an import or plumbing smoke only.

Before a serious custom CurvyTron run:

- run root-target probes on known scoreable or survivable states;
- sweep MCTS simulations, for example 2, 8, 16, 25, 50;
- report whether the oracle or diagnostic action gets nonzero target mass;
- report tie rates and top-1/top-2 visit gaps;
- check whether higher simulations make targets better or only make a bad
  action more confident.

If root targets are wrong, do not run longer. Fix target quality first.

## Modal Volumes And Checkpoints

Use Modal for whole jobs, not for the hot training loop.

The stable pattern is:

- one Modal function owns one train or eval job;
- framework output can go to local temp storage inside the job;
- useful artifacts are mirrored into the durable `curvyzero-runs` Volume;
- the Volume is committed after summaries, logs, checkpoints, and eval outputs
  are written;
- independent eval jobs load checkpoints from the Volume.

For CurvyTron checkpoints, write immutable payloads and small pointer files:

- `latest.json`;
- `best.json` when there is a real selection rule;
- checkpoint hash;
- run id and attempt id;
- config ref;
- observation schema;
- action schema;
- reward schema;
- support-scale fields;
- eval status.

Do not rely on Modal retries until resume from a committed checkpoint pointer is
explicit and tested. Failed attempts should stay visible as failed attempts.

## Eval Curves And Baselines

Trainer-side reward is not enough.

For dummy Pong, independent scorecards exposed constant-action policies even
when trainer-side telemetry looked alive. CurvyTron needs the same separation:
train logs for debugging, independent eval for claims.

Every serious CurvyTron checkpoint family should score:

- `iteration_0`;
- at least one middle checkpoint;
- final checkpoint;
- `ckpt_best` only if the selection rule is clear;
- fixed baselines;
- older frozen checkpoints once available.

Early baseline ladder:

- random policy as the sanity floor;
- simple scripted survival policy;
- simple scripted scoring or pressure policy if one exists;
- older frozen checkpoint;
- best-so-far checkpoint.

Report curves, not only a final row. A single lucky checkpoint is not enough.
Progress should appear across score, survival, action diversity, root quality,
and at least two adjacent checkpoints.

## Stop Rules

Stop a run family when the evidence says the next longer run will only scale
the same failure.

Stop if:

- compiled support scale does not match the custom reward scale;
- target sidecars show useful actions getting zero or near-zero target mass;
- higher MCTS simulations only make the wrong action more confident;
- held-out action histograms collapse to one action across opponents;
- trainer-side diversity improves but independent eval stays collapsed;
- survival improves only through stalling or timeouts;
- raw score, survival, shaped diagnostics, and action entropy are all flat;
- checkpoint curves do not improve from `iteration_0` to middle to final;
- eval uses fallback actions or cannot strict-load the checkpoint;
- a custom run is being compared directly to official Atari without naming the
  lane difference.

Go only when:

- support fields are verified in the compiled config;
- root targets look sane on known diagnostic states;
- independent eval improves over run-local `iteration_0`;
- fixed baselines do not regress;
- action use is state-dependent, not constant;
- the result survives at least one held-out seed or checkpoint-curve check.

## CurvyTron Transfer Checklist

Before the first serious CurvyTron training run:

- define the lane: official control or custom CurvyTron;
- write the observation, action, reward, terminal, and support schemas;
- add target sidecars before trusting training;
- verify compiled support scale and log patched plus compiled fields;
- build independent scorecards before scaling;
- include survival and terminal-cause telemetry from the first eval;
- run root-target probes on known states;
- keep Modal jobs whole, durable, and checkpoint-addressable;
- require eval curves and baseline rows before any quality claim.

The lesson is not "Pong failed." The lesson is that custom sparse games need
visible targets, verified config, honest survival telemetry, durable artifacts,
and stop rules before scale can mean anything.
