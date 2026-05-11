# Pong Training Plan

Status: historical plumbing and diagnostics. The current active spine is
`docs/working/pong_selfplay_training_plan_2026-05-09.md` plus
`docs/working/pong_training_critique_wave_2026-05-09.md`. The self-play lane is
a hypothesis under critique, not a proven next path.

## Current Truth / No More Pretending

- We validated stock LightZero CartPole MuZero progression.
- We validated an Mctx search benchmark.
- We validated a CEM-v2 Pong baseline.
- We validated a raster-only MLP Pong baseline.
- We have not run an actual project-owned MuZero/Mctx train loop for Pong.
- We have not run an actual project-owned MuZero/Mctx train loop for Curvy.
- CEM-v2 and the MLP are baselines and scaffolding only. They are not MuZero
  progress.
- The next main lane is LightZero-first: adapt dummy Pong as a custom env and
  run a capped LightZero MuZero trainer on Modal. Project-owned Mctx is fallback
  if LightZero fails or hides required telemetry/artifacts.

Prevention rules:

- Prove the target is scoreable before scaling it.
- Keep baselines separate from MuZero.
- Name the algorithm in every experiment title, command, and summary.
- Distinguish stock LightZero MuZero from project-owned MuZero/Mctx.
- Do not describe CEM, imitation, or MLP results as MuZero progress.

Use Pong either for explicitly named visual/raster baselines or for the next
real LightZero custom dummy Pong MuZero smoke. Keep the current evidence honest:
the active linear checkpoint encoding is raster-fed and geometry-augmented, not
visual-only. The geometry-suffix policies should not be called truly visual.
Do not read this historical plan as a mandate to add more generations or as
proof of MuZero progress.

Eval rule: never judge Pong by wins alone. In the default geometry,
`track_ball` is now a survival/tie floor rather than a hard win gate, because
exact search found no scoring sequence from normal resets. Every
learned-vs-`track_ball` read must include survival/loss-delay metrics and the
shaped proxy: episode length, truncation rate, and shaped return/loss-delay
proxy from `episodes.jsonl`.

Coach correction after CEM-v2: the failed lane judged by wins too early, did
not prove `track_ball` was beatable soon enough, and spent runs on an
impossible hard gate. The recovery was survival audit first, exact beatability
probe second, target ladder third, then a CEM-v2 score-pressure monitor on the
proven-scoreable `lagged_track_ball_1` target. New rule: prove the target is
scoreable and name the fine-grained metric before scaling training.

## Plain-Language Glossary

- Run/probe: one bounded experiment with its own command, seed range, target,
  checkpoint, and output folder. The recent Pong numbers came from separate
  runs/probes, not from one continuous training run.
- Checkpoint: saved policy weights. Checkpoints are comparable when the
  scoreboard loads them under the same env config, opponent, episode count,
  split/seed role, and metric bundle. A checkpoint is continued only when a
  later command explicitly resumes from it or uses it as the parent/behavior
  policy.
- `track_ball`: deterministic script that moves the paddle toward the current
  ball row. In the current default geometry, exact search says it is a
  survival/tie floor from normal resets, not a proof of optimal Pong play in
  every geometry or reset distribution.
- `lagged_track_ball_1`: the same tracking idea, but the opponent tracks the
  previous ball row instead of the current one. This one-tick lag makes the
  target scoreable from normal resets, so it is the current score-pressure rung.
- Shaped proxy: a separate loss-delay diagnostic/training target. It gives
  partial credit for losing later. It is not the environment reward.

## Why The Metrics Moved

The metrics shifted because the experiments changed, not because one long run
smoothly improved. The main changes were:

- Target/opponent: default `track_ball` was replaced by
  `lagged_track_ball_1` for score-pressure training and eval.
- Data/replay: old self-play replay, exact trace replay, mirrored/oversampled
  rows, DAgger-style appended rows, frame stacks, and uploaded Modal replay refs
  are different datasets.
- Policy architecture: the baseline lane moved from linear checkpoints to a
  stack-2 one-hidden-layer MLP for the visual-only supervised result.
- Feature mode: `raster_plus_geometry` includes decoded geometry helpers;
  `raster_only` removes those helpers and uses raster cells plus minimal ego
  routing.
- Modal wrapper: later runs used whole-job Modal train/eval wrappers and Volume
  refs. That changes execution/storage plumbing, not Pong's reward.

The env reward did not change: score events are still +1/-1/0. For
learned-vs-`track_ball`, report the metric bundle: wins, mean/median survival
steps, truncation rate, `track_ball` wins/losses, and shaped proxy. Do not
collapse this back to a bare win fraction while default `track_ball` is a
survival/tie floor.

Variance and exploration are more than a chart, but still not the environment
reward. Track survival standard deviation, p90 survival, max survival, rare
wins, and shaped-return standard deviation. Early checkpoint selection or
replay sampling may use a small bounded bonus for wider survival tails when
mean score/survival is similar. Remove or shrink that bonus if it rewards
stalling, random instability, or worse `random_uniform` performance.

## What Pong Currently Proves

- The repo has a tiny project-owned two-player Pong-like environment:
  `dummy_pong_v0`.
- The environment supports simultaneous actions, terminal win/loss rewards, and
  max-step truncation.
- Fixed baselines exist:
  - `random_uniform`
  - `track_ball`
- The eval harness can compare fixed baselines and write `summary.json` plus
  `episodes.jsonl`.
- The observability harness can write game rows, step rows, and raster frames.
- The artifact inspector can summarize Pong replay/trace directories and flag
  missing reward signal or truncation-only data.
- A tiny supervised raster learner can train from the imitation replay and save
  `checkpoint.npz`.
- Pong eval can now load `learned:<checkpoint.npz>` and run the checkpoint from
  raster observations inside games.
- A tiny value-target learner can train from all-ego scoring replay and save a
  reloadable value checkpoint. This proves target plumbing, not policy
  improvement.
- `PongEnv.raster_observation()` exposes the visual path that a later learner
  should use.
- Paddle hits already have simple angle control: hitting above the paddle
  center sends `ball_vy=-1`, hitting the center sends `ball_vy=0`, and hitting
  below center sends `ball_vy=1`.
- A first scripted `angle_control` probe can deliberately create off-center
  contacts and beats `random_uniform`, but it still only times out against
  `track_ball`.
- The contact-outcome probe can create top/center/bottom contact rows. The
  first smoke showed different outgoing `ball_vy`, but flat score-delta
  returns against `track_ball`.
- A width-9 contact-outcome probe also stayed flat: outgoing `ball_vy` differed
  on all 64 states, but every candidate score-delta return was still `0.0`.
  Smaller width alone did not create score-bearing contact labels.
- The checkpoint scoreboard command now exists. The first smoke used the same
  checkpoint as `latest` and `previous`, so it proves scoreboard plumbing only.
- A distinct-checkpoint scoreboard compared the three existing learned policy
  attempts. They beat random more often than not, but none passed the
  old `track_ball` gate. That gate is now known to be invalid for the default
  geometry because exact search found no scoring sequence from normal resets.
- The imitation trainer now saves periodic policy checkpoints with
  `--checkpoint-every-epochs`. The scoreboard can load those epoch checkpoints,
  so old-vs-new eval plumbing is ready.
- Important naming correction: `DummyPongImitationPolicy` receives
  `raster_grid` at eval time, but the feature encoder is
  `dummy_pong_raster_one_hot_plus_geometry_v0`. It concatenates one-hot raster
  cells with six decoded geometry features: ball x/y, both paddle centers, and
  ball-minus-paddle-center y offsets for both players. Existing "visual" trace
  checkpoints therefore prove raster-fed checkpoint plumbing and supervised
  learning from raster rows, not a raster-only policy.
- The imitation trainer now has an explicit backward-compatible feature mode:
  default `--feature-mode raster_plus_geometry` preserves old checkpoints and
  the six decoded geometry helpers, while `--feature-mode raster_only` writes
  `dummy_pong_raster_one_hot_v0` checkpoints whose logits use only one-hot
  raster cells plus the per-ego policy head. Eval reloads the checkpoint mode
  from metadata.
- The first periodic checkpoint scoreboard compared epoch 1 and epoch 3 from
  one tiny run. Epoch 3 did slightly better against random, but both checkpoints
  still won 0/4 against `track_ball`, and the learned-vs-learned row tied. This
  proves plumbing only.
- A longer 1000-epoch imitation run wrote epoch 250/500/750/1000 snapshots.
  Epoch 1000 beat random 42/64 and beat most earlier checkpoints, but still won
  0/64 against `track_ball`.
- A local selection-record helper now selects checkpoints from scoreboard
  summaries. On the 1000-epoch imitation selection split, it selected epoch
  1000 while marking heldout confirmation as still required.
- Heldout for selected epoch 1000 is mixed but useful: it beat previous 50/64
  and survived longer against `track_ball`, but still won 0/64 against
  `track_ball`. Previous beat random slightly more often on heldout.
- A CPU Modal scoreboard wrapper now runs the same Pong scoreboard remotely
  using `curvyzero-runs` checkpoint refs and Volume eval outputs.
- A short-lookahead replay builder now exists. It samples raster states against
  a fixed `track_ball` opponent, tries all three immediate ego actions, rolls
  out score-delta returns, and writes the best action as `target_action`.
- The first tiny lookahead smokes were mixed: strict score-separated rows
  produced only 9 labels and all matched `track_ball`; adding an explicit
  `angle_control` tie-break over equal-return states produced 131 labels with
  41 non-`track_ball` targets, but the trained checkpoint still scored 0 wins
  against `track_ball`.
- A larger angle-tie lookahead run produced 1,669 labels with 442
  non-`track_ball` targets, but all lookahead checkpoints still won 0/64
  against `track_ball`; the selector kept imitation epoch 1000.
- Loss-delay lookahead is implemented as a training-label option:
  `--loss-delay-alpha` gives losing candidate rollouts small credit for losing
  later. The first smoke with alpha `0.05` produced 251 rows but 0 targets
  different from `track_ball`; the trained checkpoint won 0/32 against
  `track_ball` and lost 27/32. This is not worth scaling as-is.
- Depth-2 ego-sequence lookahead is implemented with `--ego-sequence-depth 2`.
  The first strict smoke emitted 10 non-tied avoided-loss rows from 65 sampled
  states, but every target first action still matched `track_ball` and every
  chosen target return was `0.0`. Do not train from that small strict replay.
- Pong self-play replay/train/eval exists locally, but generation 2 lost to the
  parent and won 0 games against `track_ball`. This supersedes the older
  "try another generation" framing. See
  `docs/experiments/2026-05-09-dummy-pong-selfplay-gen2-smoke.md` and
  `docs/working/pong_training_critique_wave_2026-05-09.md`.
- Modal self-play repair and the parallel sweep have now been audited with
  raw `episodes.jsonl` survival metrics. Wins alone were an incomplete summary:
  repair ckpt25 was the best learned-vs-`track_ball` row with 47.30 mean steps,
  20/64 truncations, 44/64 `track_ball` wins, and a -0.6467 learned shaped
  loss-delay proxy. The best parallel row, higher-diversity e25, reached 46.41
  mean steps, 19/64 truncations, 45/64 `track_ball` wins, and -0.6582 shaped
  proxy. The sweep did not beat the repair run, and future decisions must
  report survival/loss-delay before judging the learner.
- The one fresh-data undertraining probe is done. The 512-game Modal run
  scored e25/e50/e75 with `--episodes 64`; the best row was e25 with 36.19
  mean steps, 22/128 truncations, 106/128 `track_ball` wins, 0/128 learned
  wins, and a -0.7633 shaped proxy. This did not beat repair ckpt25, and later
  checkpoints degraded further.
- A tiny survival/loss-delay curriculum trainer now exists at
  `src/curvyzero/training/dummy_pong_survival_curriculum_train.py`. It trains
  on raster observations, writes the existing scoreable checkpoint shape, and
  reports wins, episode length, survival fraction, truncation rate, score
  return, and shaped training return. The first smoke is a shape check, not a
  quality claim.
- A tiny geometry-CEM baseline now exists at
  `src/curvyzero/training/dummy_pong_cem_train.py`. It searches only the six
  geometry features appended to the raster encoding, writes the normal
  scoreboard-loadable linear checkpoint, and selects on a paired-seat
  survival-aware proxy against `random_uniform` and `track_ball`. The first
  smoke beat random 16/16 on the checkpoint scoreboard and forced 16/16
  full-length truncations against `track_ball`, with 120.0 mean steps, 0
  learned wins, 0 `track_ball` wins, and 0.0 learned shaped proxy. Treat it as
  the new survival floor and failure probe, not as evidence of win pressure.
- An exact beatability probe now answers the hard geometry question for
  `PongConfig(width=15,height=9,paddle_height=3,max_steps=120)`: no legal ego
  action sequence scores against deterministic `track_ball` from any normal
  reset support state. The probe checked all 20 reset states, both ego seats,
  and all legal ego action branches to the 120-step cap; transition parity
  against `PongEnv.step` passed on the reset support. This means `track_ball`
  is a full-survival/tie baseline in the current toy, not a valid hard win
  target.
- A compact exact target ladder now identifies the next score-pressure rung:
  keep the default geometry and normal resets, but evaluate against
  `lagged_track_ball_1`, where the opponent tracks the previous ball row. It is
  scoreable in 40/40 normal reset/player cases with median 19.0 winning steps.
  CEM-v2 is now wired for this target with score-primary selection and
  survival/loss-delay only as a tie-breaker.
  The same sweep found default `track_ball` at 0/40, symmetric paddle height 2
  at 38/40, paddle height 1 at 20/40, width 9 at 0/40, height 11 at 0/56, and
  biased near-contact starts at only 4/20. Use lag-1 as the main ladder target;
  keep near-contact starts diagnostic only.
- CEM-v2 now gives the first score-pressure baseline checkpoint for dummy Pong.
  It is project-owned baseline code, not project-owned MuZero. The documented
  local monitor scored 25/32 final-eval wins against `lagged_track_ball_1`,
  30/32 wins against `random_uniform`, and 32/32 truncation ties against
  default `track_ball`. The checkpoint scoreboard then confirmed 53/64 learned
  wins versus lag-1, 60/64 versus random, and 64/64 full-length ties versus
  default `track_ball`.
- CEM-v2 is not a visual learner. It uses the same
  `dummy_pong_raster_one_hot_plus_geometry_v0` checkpoint/eval shape, but its
  search vector only controls the appended six geometry features and per-agent
  action biases; all one-hot raster-cell weights are zero-filled when the
  checkpoint is written. It is also not MuZero.
- CEM-v2 is now Modal-backed baseline work. `dummy_pong_cem_train_attempt` runs
  the same trainer as one CPU Modal Function, writes `summary.json`,
  `checkpoint.npz`, `cem_rows.jsonl`, run/attempt manifests, and
  `checkpoints/latest.json` to `curvyzero-runs`, and the existing Modal
  scoreboard can load that checkpoint by Volume ref. The first Modal run
  reproduced the monitor result: 25/32 train final-eval wins versus lag-1, then
  53/64 learned wins versus lag-1 on the Modal scoreboard. This is not a
  project-owned MuZero/Mctx train loop.
- The first post-CEM raster-fed trace lane now exists as a tiny exact trace
  replay builder:
  `scripts/build_dummy_pong_lag1_trace_replay.py`. It converts target-ladder DP
  winning traces against `lagged_track_ball_1` into supervised raster rows for
  the existing imitation trainer. Because the trainer appends decoded geometry
  features to one-hot raster cells, this is not yet a visual-only/raster-only
  policy. The first smoke produced 1,332 raster rows from 40 exact traces,
  trained a reloadable raster-fed geometry-augmented checkpoint, and scored it
  at 5/16 wins versus lag-1, 10/16 versus random, and 0/16 versus default
  `track_ball` with 5/16 truncations. This starts the supervised raster-input
  lane but does not beat CEM-v2.
- The raster imitation trainer now has a backward-compatible
  `--class-weighting balanced` option. On the same lag-1 exact-trace replay,
  balanced loss changed the all-row predicted action histogram from
  1,311/0/21 up/stay/down to 1,063/55/214 and improved rare-class supervised
  accuracy, but scoreboard quality stayed weak: 6/16 wins versus
  `lagged_track_ball_1`, 10/16 versus random, and 0/16 versus default
  `track_ball` with only 2/16 truncations. Class weighting alone is diagnostic,
  not a pass.
- The smallest closed-loop relabeling repair now exists:
  `scripts/build_dummy_pong_lag1_dagger_replay.py`. It rolls out a learned
  raster-fed checkpoint against `lagged_track_ball_1`, exact-labels visited
  states from the current lagged-opponent memory, appends those labels to the
  supervised replay, and leaves the trainer/eval path unchanged. It now has
  controls for repeated seed blocks, repeated rollouts, both or selected seats,
  epsilon behavior, multiple source checkpoints, behavior histograms, and
  capped unlabelable-state examples. The first tiny smoke added 22 rows from
  four rollout episodes, retrained, and scored 5/16 wins versus lag-1, 11/16
  versus random, and 0/16 versus default `track_ball` with 4/16 truncations.
  The broader follow-up appended 1,200 labeled closed-loop rows with 392
  unlabelable visited states; unweighted training scored 5/16 versus lag-1,
  10/16 versus random, and 5/16 `track_ball` truncations, while balanced
  training matched mirror-only at 6/16 versus lag-1 but fell to 6/16 versus
  random and 2/16 `track_ball` truncations. This proves and broadens the
  closed-loop relabeling lane, but it still does not improve over mirror-only
  replay.
- The first truthful raster-only ablation is implemented and scored on the
  lag-1 exact-trace replay. With balanced loss, the checkpoint used a 675-wide
  one-hot raster feature axis instead of the 681-wide raster-plus-geometry axis,
  trained to 0.8180 train accuracy and predicted 848/43/175
  up/stay/down on the train split. Scoreboard: 6/16 wins versus
  `lagged_track_ball_1`, 10/16 versus `random_uniform`, and 0/16 versus default
  `track_ball` with 2/16 truncations and 29.56 mean steps. This proves the
  ablation is honest; it is not yet a viable lag-1 policy.
- Frame stack is implemented and loadable across replay, training checkpoint
  metadata, and eval. In the first linear smoke, stack-2 improved lag-1 wins to
  13/32 and default-`track_ball` truncations to 6/32, but it still failed the
  >50% lag-1 gate and lost random sanity at 15/32 versus `random_uniform`.
  This is weak positive evidence for temporal input, not a solved visual lane.
- A separate one-hidden-layer NumPy MLP checkpoint path now exists for the
  supervised imitation trainer and loads through the existing Pong eval
  `learned:<checkpoint.npz>` path. The first stack-2 `raster_only` MLP smoke
  used hidden dim 128, 800 epochs, balanced class weighting, and learning rate
  0.005. It trained to 0.9814 all-row accuracy and passed the cheap closed-loop
  gate: 26/32 wins versus `lagged_track_ball_1`, 19/32 versus
  `random_uniform`, and 10/32 truncations with 61.91 mean steps versus default
  `track_ball`. This is the first truthful raster-only lag-1 policy to clear
  the local >50% score-pressure gate. Read it as support for the
  observation/policy mismatch diagnosis: single-frame input was missing useful
  history, and the linear softmax was too weak for the stack-2 label map. It is
  now confirmed on heldout seed 29: 43/64 wins versus
  `lagged_track_ball_1`, 36/64 versus `random_uniform`, and 23/64 truncations
  with 62.1719 mean steps versus default `track_ball`. The decision is plain:
  the raster-only MLP baseline is real enough to Modalize, but it is still
  weaker than CEM-v2, it is supervised imitation, and default-track survival is
  not solved.
- The stack-2 `raster_only` MLP baseline is now Modal-backed. The
  `dummy_pong_imitation_train_attempt` wrapper trained from the replay Volume
  ref as one CPU Modal Function, wrote the copied replay, summary, checkpoint,
  manifests, and latest checkpoint pointer to `curvyzero-runs`, then the Modal
  scoreboard loaded the returned checkpoint by `ref:`. The scoreboard scored
  49/64 wins versus `lagged_track_ball_1`, 34/64 versus `random_uniform`, and
  25/64 truncations with 66.1875 mean steps versus default `track_ball`. This
  is supervised visual-only baseline evidence, not MuZero progress. It remains
  below CEM-v2's 53/64 lag-1 row and does not win against default
  `track_ball`.
- Observation sufficiency audit for the lag-1 visual lane: a single
  `PongEnv.raster_observation()` frame contains only current paddle cells and
  the current ball cell. It does not encode `ball_vx`, `ball_vy`, or the
  lagged opponent memory used by the exact teacher. The replay rows do carry
  tabular observations with velocity, but `DummyPongImitationPolicy` ignores
  those rows and predicts from the current raster grid plus ego head. Therefore
  the exact-DP first-action label is not guaranteed to be a function of one
  raster frame. `raster_plus_geometry` does not fix this: its six decoded
  helpers are only ball x/y, both paddle centers, and ball-minus-paddle-center
  y offsets, with no velocity or previous-frame channel.
- Current policy-class audit: the imitation checkpoint is per-ego multinomial
  logistic regression over one current frame. CEM-v2 proves a linear
  geometry-only return-optimized policy can win the lag-1 game, but the
  supervised exact-DP label map is stateful/aliased under the current input and
  likely piecewise even after velocity is exposed. Treat static BC, balancing,
  mirroring, and same-shape DAgger appends as negative evidence for this
  one-frame linear policy, not as proof that raster learning is impossible.

## Mini North Star: Off-Center Returns

Do not define Pong progress as merely copying `track_ball`. `track_ball` moves
the paddle center toward the current ball row, which is enough to beat random
but not a strategy.

The old mini north star was to learn when to choose off-center paddle contacts
so the return angle makes `track_ball` late or mispositioned. The exact
beatability probe closes that framing for the current default geometry:
off-center contacts exist, but no legal sequence can convert them into a score
against `track_ball` from normal resets within 120 steps.

Important caveat: off-center contact rate alone is not enough. The scripted
`track_ball` baseline also produces many off-center hits because simultaneous
paddle and ball movement creates one-step lag. Measure whether the chosen
contact creates score pressure, not just whether the hit is off-center.

## Eval Reset

Keep Pong eval simple. The scoreboard is not the contact-outcome probe.

Scoreboard metrics:

- learned checkpoint versus `random_uniform`;
- learned checkpoint versus `track_ball` as a survival/tie floor, not as a
  beatability gate in the current default geometry;
- latest checkpoint versus previous checkpoint;
- latest checkpoint versus best-so-far checkpoint;
- win/loss/truncation counts on a fresh recorded pseudo-random eval seed list
  before any stronger claim.
- for learned-vs-`track_ball`, always report mean/median episode steps,
  truncation rate, `track_ball` win rate, and a shaped loss-delay proxy when
  `episodes.jsonl` has `steps`, `winner`, and `truncated`.

Debug and observability metrics:

- action histograms;
- off-center contact counts;
- contact-outcome rows;
- value-target loss;
- sample frames or traces.

Use the debug metrics to explain the scoreboard. Do not promote them into the
scoreboard. A policy that makes more off-center hits but still times out
against `track_ball` has not improved yet. Conversely, do not compress
learned-vs-`track_ball` eval to 0 wins alone; survival length and delayed losses
are part of the scoreboard diagnosis.

## Main Gaps

- The next serious lane is a real LightZero custom dummy Pong MuZero smoke, or an
  explicitly named baseline lane. Baseline work may include a truly
  visual-only/raster-only policy against `lagged_track_ball_1`, or a
  deliberately named raster-fed geometry-augmented baseline. A true
  visual-only toy policy means action
  logits are computed from raw raster pixels or raster one-hot cells plus
  minimal non-visual routing such as `ego_agent`; no decoded ball coordinate,
  paddle center, distance-to-paddle, observation struct field, DP trace state,
  or other hand-built geometry helper can enter the policy input or searched
  parameter slice.
- Static exact-trace BC with the old linear policy is weak because one frame
  sees too little history and the linear softmax cannot express the useful
  stack-2 label map. Static trace BC, class weighting, mirror/oversampling, the
  first tiny DAgger append, broad DAgger, raster-only linear, and stack-2 linear
  all failed the >50% lag-1 win gate. The stack-2 raster-only MLP now passes the
  local lag-1 gate at 26/32 and heldout seed 29 at 43/64, with 36/64 versus
  random. That is enough to Modalize the MLP baseline if we are explicitly
  staying in baseline land. It is not enough to promote over CEM-v2 or call it
  MuZero progress: default `track_ball` survival stayed weak at 23/64
  truncations and 62.1719 mean steps, while CEM-v2 remains stronger at 53/64
  lag-1 wins and 64/64 default-track truncation ties. The next question is a
  real LightZero custom dummy Pong MuZero smoke, or, if labeled as baseline
  work, Modal execution and whether a smaller or regularized nonlinear policy can keep
  lag-1 wins without the strong seat/action bias.
  Do not spend another run on static balancing or more same-shape supervised
  append.
- Pong has monitor and selection split metadata in scoreboard outputs. The
  selected imitation epoch has one mixed heldout check, but new selections still
  need a standard heldout gate.
- Pong has periodic policy checkpoints and a local selection record, but no
  `best` pointer yet.
- The first learned checkpoint is weaker than scripted `track_ball`.
- Scoring replay can now emit all ego rows. Use all-ego rows for value targets,
  but not for plain expert-action copying because they include random actions.
- The first value checkpoint has weak fit. A single raster frame plus ego id is
  enough to prove the file path, but not enough to claim strategic value
  prediction.
- Pong now has a direct off-center-contact eval probe. The gap is that simple
  off-center contacts do not yet create scoring pressure against `track_ball`.
- Pong now also has an exact `track_ball` beatability probe, and the default
  geometry is not scoreable against `track_ball` from normal resets. Future
  win-pressure work needs a different target, reset distribution, or geometry.
- A one-step lookahead relabeler exists, but the default score-separated labels
  mostly say "`track_ball` was already best." The angle-control tie-break can
  create non-`track_ball` labels, but the first smoke only forced 2/8
  truncations against `track_ball`, with 0 learned wins.
- A loss-delay variant exists for training labels, not eval reward. The first
  run did not create labels different from `track_ball`, so it does not solve
  the objective gap by itself.
- A depth-2 sequence variant exists. The first smoke can detect losing
  sequences, but it still did not produce non-`track_ball` first-action labels.
- Pong has no project-owned MuZero/Mctx actor/search/trainer loop yet.
- LightZero Pong remains blocked at the ROM/license gate, and Mctx remains an
  isolated benchmark/search spike until there is a clear reason to connect it
  to replay or environments.

The important gap is not physics. The important gap is a stable training input
and then one small learner that can show improvement against `random_uniform`
without hiding behind scripts or seed luck.

## Reward Rule

Use score changes as the environment and eval reward:

```text
ego scores:       +1
opponent scores:  -1
no score event:    0
```

For training loops, use a separate shaped episode return that gives explicit
credit for survival/loss delay. This is a training target and diagnostic, not a
replacement for score/outcome reward:

```text
win:       +1.0
loss:      -1.0 + 0.5 * episode_steps / max_steps
timeout:    0.0
```

Log rally length, paddle hits, and timeout count so we can catch stalling. The
scoreboard should still answer the hard question, does the policy win more, but
run summaries must not hide survival/loss-delay when wins are still zero.

The research note is `docs/research/pong_reward_design.md`.

## Smallest Useful Next Step

Stop the old self-play trainer as the main lane. The survival/loss-delay audit
is accounted for: repair ckpt25 remains the best old baseline, and the 512-game
fresh-replay probe got worse. The exact beatability probe also shows the
current default `track_ball` target cannot be scored on from normal resets, so
a new Pong learner or curriculum must change the target/reset/geometry rather
than blindly optimize the same invalid hard gate.

The first CEM-v2 monitor used `lagged_track_ball_1`, not default `track_ball`:

```sh
uv run python -m curvyzero.training.dummy_pong_cem_train \
  --width 15 \
  --height 9 \
  --paddle-height 3 \
  --max-steps 120 \
  --generations 8 \
  --population-size 32 \
  --elite-count 8 \
  --eval-games 16 \
  --seed 8050913 \
  --opponent-weight lagged_track_ball_1=1.0 \
  --opponent-weight random_uniform=0.10 \
  --opponent-weight track_ball=0.10 \
  --target-opponent-id lagged_track_ball_1 \
  --loss-delay-weight 0.5 \
  --truncation-value 0.0 \
  --output-dir artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-2026-05-09
```

Then scored the checkpoint with `scripts/run_dummy_pong_checkpoint_scoreboard.py`
on monitor seeds:

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 32 \
  --seed 9050913 \
  --split-id dummy_pong_cem_v2_lagged_track_ball_1 \
  --split-role monitor \
  --checkpoint cem_v2=artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-cem-v2-lagged-track-ball-1-scoreboard-2026-05-09
```

Result: pass. Training final eval scored 25/32 learned wins against
`lagged_track_ball_1`, 30/32 versus `random_uniform`, and 32/32 truncation ties
against default `track_ball`. The scoreboard confirmed the target pressure on
fresh pair seeds: 53/64 learned wins versus lag-1, 60/64 versus random, and
64/64 full-length ties versus default `track_ball`. Survival against default
`track_ball` remains a diagnostic only.

The Modal-backed equivalent is now:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_cem_train_attempt \
  --width 15 \
  --height 9 \
  --paddle-height 3 \
  --max-steps 120 \
  --generations 8 \
  --population-size 32 \
  --elite-count 8 \
  --eval-games 16 \
  --seed 8050913 \
  --opponent-weights lagged_track_ball_1=1.0,random_uniform=0.10,track_ball=0.10 \
  --target-opponent-id lagged_track_ball_1 \
  --loss-delay-weight 0.5 \
  --truncation-value 0.0
```

Observed Modal run:
`pong-cem-20260509T045950Z-e8b06974a402` /
`attempt-20260509T045950Z-f16d342d760b`, with checkpoint ref
`training/dummy-pong/pong-cem-20260509T045950Z-e8b06974a402/attempts/attempt-20260509T045950Z-f16d342d760b/train/checkpoint.npz`.
The remote scoreboard run
`pong-scoreboard-20260509T050220Z-84b0c61e5ab9` confirmed 53/64 learned wins
versus lag-1, 60/64 versus random, and 64/64 truncation ties versus default
`track_ball`.

Why:

- Real project-owned MuZero/Mctx work must include learned dynamics/search,
  replay, training, checkpointing, and eval. Permanent teacher copying is
  baseline work, not MuZero.
- `track_ball` cannot teach us win pressure, and in the current default
  geometry it cannot be beaten from normal resets by any legal action sequence
  within the 120-step cap.
- The raster observation, checkpoint, and scoreboard plumbing already exist.
- The Modal self-play sweep showed that blind scaling by variants is not
  enough, but the audit found a meaningful survival/loss-delay ordering.
- The next run should answer a new learner/curriculum question, or test a
  concrete bug fix in the old trainer.

Current trainer read:

- The shaped return is actually used by the current trainer. Replay rows
  contain `shaped_return`; `_dataset_from_rows()` loads it into
  `dataset["shaped_returns"]`; `_advantages()` centers/scales that target; and
  `_train_epoch()` uses the resulting advantages for the policy update while
  fitting the value head directly to `shaped_returns`.
- Survival is available in artifacts. Replay rows carry `episode_steps`,
  `episode_truncated`, `episode_winner`, and `episode_terminal`; replay
  `summary.json` carries per-game `steps`, `winner`, and `truncated`.
  Scoreboard summaries already include `mean_steps`, `median_steps`, and
  `truncations`; raw scoreboard `episodes.jsonl` has per-episode `steps`,
  `winner`, `truncated`, and rewards, which is enough to compute the shaped
  loss-delay proxy.
- There is no obvious simple wiring break in the shaped-return path. The weak
  trend is more likely a mix of objective/data/learner limits than a missing
  target field. The data are still tiny by policy-learning standards, the
  objective is score/loss-delay labels from random-vs-random self-play rather
  than a curriculum that creates pressure against `track_ball`, and the learner
  is a linear raster policy/value update.
- Longer epochs on the same replay are not the likely fix. In the Modal sweep,
  early checkpoints were usually best, and later checkpoints often had fewer
  truncations and worse shaped proxy.
- The undertraining hypothesis has now been tested once by scaling fresh replay
  first and keeping epochs conservative. It did not beat the repair ckpt25
  survival baseline, so another 0-win fresh-data checkpoint is not useful by
  itself.

The one cheap Modal feasibility probe has now been run:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_train_attempt \
  --run-id pong-selfplay-512g-feasibility-20260509 \
  --attempt-id train \
  --games 512 \
  --max-steps 120 \
  --policy random_uniform \
  --epsilon 0.10 \
  --epochs 75 \
  --policy-learning-rate 0.03 \
  --value-learning-rate 0.001 \
  --action-diversity-beta 0.05 \
  --validation-fraction 0.2 \
  --checkpoint-every-epochs 25 \
  --seed 8050901
```

Result: failed the pass bar. The best checkpoint, e25, scored 36.19 mean steps,
19.0 median steps, 22/128 truncations, 106/128 `track_ball` wins, 0/128 learned
wins, and -0.7633 shaped proxy. Repair ckpt25 remains better at 47.30 mean
steps, 20/64 truncations, and -0.6467 shaped proxy. Do not run more fresh-data
scales of this old trainer unless a concrete learner/objective bug is found.

The tiny new survival-curriculum trainer remains a fallback experiment shape:

```sh
uv run python -m curvyzero.training.dummy_pong_survival_curriculum_train \
  --epochs 3 \
  --games-per-epoch 4 \
  --eval-games 2 \
  --seed 6050911 \
  --max-steps 120 \
  --learning-rate 0.08 \
  --output-dir artifacts/local/dummy-pong-survival-curriculum-smoke-2026-05-09
```

The tiny smoke beat random 6/8 on the checkpoint scoreboard, still won 0/8
versus `track_ball`, and forced 2/8 `track_ball` truncations. Treat this as a
better artifact shape, not proof of policy quality, and do not broaden it until
the old-trainer feasibility check is decided.

The current smallest useful replacement for the old self-play lane is the
geometry-CEM baseline:

```sh
uv run python -m curvyzero.training.dummy_pong_cem_train \
  --generations 2 \
  --population-size 8 \
  --elite-count 3 \
  --eval-games 4 \
  --seed 8050911 \
  --max-steps 120 \
  --output-dir artifacts/local/dummy-pong-geometry-cem-smoke-2026-05-09
```

Score it with the existing checkpoint scoreboard:

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 9050911 \
  --split-id dummy_pong_geometry_cem_smoke \
  --split-role smoke \
  --checkpoint cem=artifacts/local/dummy-pong-geometry-cem-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-geometry-cem-scoreboard-smoke-2026-05-09
```

Result: `learned_cem` beat `random_uniform` 16/16 and tied `track_ball` by
surviving to the 120-step cap in all 16 paired-seat episodes. This supersedes
the old blind self-play lane as a survival-aware baseline. It does not solve
the strategic objective because it still wins 0/16 against `track_ball`.

The exact beatability check for this geometry is now:

```sh
uv run python scripts/probe_dummy_pong_track_ball_beatable.py \
  --width 15 \
  --height 9 \
  --paddle-height 3 \
  --max-steps 120 \
  --output-dir artifacts/local/dummy-pong-track-ball-beatable-probe-2026-05-09
```

Result: 20/20 normal reset states and 40/40 reset/player cases were searched
exactly to the 120-step cap; 0 cases can score against `track_ball`. Treat
future learned-vs-`track_ball` rows as survival/tie diagnostics only.

The next project-owned target after CEM is not another survival run. The score
pressure trace now exists:

```sh
uv run python scripts/probe_dummy_pong_target_ladder.py \
  --max-steps 120 \
  --output-dir artifacts/local/dummy-pong-target-ladder-probe-2026-05-09
```

Result: `lag1_track_ball_normal` scored in 40/40 normal reset/player cases
while default `track_ball` stayed 0/40. This is the smallest useful target
because it preserves board geometry, paddle height, action space, reset
support, and visual observation shape. It changes only the opponent timing by
one state, which should be easy to encode as a deterministic baseline policy
and easy to evaluate with the existing scoreboard shape.

The smallest baseline target is now answered for CEM-v2. It preserved the
current checkpoint/scoreboard shape, scored against `random_uniform`,
`lagged_track_ball_1`, and default `track_ball`, and passed the score-pressure
bar with 53/64 scoreboard wins versus lag-1. The next learner should either
compare itself against this CEM-v2 checkpoint, or the next main lane should be
a real LightZero custom dummy Pong MuZero smoke. Do not return to default `track_ball`
as a hard win target.

The smallest visual-lane baseline fix is now a positive local baseline smoke.
Two-frame stack is implemented and loadable, and the first linear smoke helped
but did not pass:
stack-2 reached 13/32 wins versus `lagged_track_ball_1` and 6/32 truncations
versus default `track_ball`, while still failing the >50% lag-1 gate and random
sanity. Replacing the linear softmax with a tiny stack-2 `raster_only` MLP
passes the cheap gate at 26/32 lag-1 wins, 19/32 random wins, and 10/32 default
`track_ball` truncations. Heldout seed 29 confirmed the baseline at 43/64
versus `lagged_track_ball_1` and 36/64 versus random, but default-track
survival stayed open at 23/64 truncations and 62.1719 mean steps. Decision
rule: this is
evidence for missing history plus weak linear policy as the failure mode, and
it is real enough to Modalize as a baseline. It is still weaker than CEM-v2,
which has 53/64 lag-1 wins and 64/64 default-track truncation ties. Explicit
velocity channels remain a diagnostic if we accept engineered dynamics.
On-policy RL or more imitation only belongs in baseline land unless the work
becomes a real LightZero custom dummy Pong MuZero smoke.

The smallest post-CEM raster-fed supervised experiment is now the target-ladder
trace replay lane. It uses exact winning traces as labels and passes
`raster_grid` plus `ego_agent` through the existing softmax checkpoint path,
but that path currently expands each raster into one-hot cells plus six decoded
geometry helper features. This is useful, but it is not visual-only:

```sh
uv run python scripts/build_dummy_pong_lag1_trace_replay.py \
  --max-steps 120 \
  --repeats 1 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-replay-smoke-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-lag1-trace-replay-smoke-2026-05-09 \
  --epochs 300 \
  --learning-rate 1.0 \
  --validation-fraction 0.2 \
  --seed 7050913 \
  --output-dir artifacts/local/dummy-pong-lag1-trace-visual-policy-smoke-2026-05-09
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 8 \
  --seed 8050914 \
  --split-id dummy_pong_lag1_trace_visual_policy \
  --split-role smoke \
  --checkpoint lag1_trace_visual=artifacts/local/dummy-pong-lag1-trace-visual-policy-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-lag1-trace-visual-policy-scoreboard-smoke-2026-05-09
```

Smoke read: replay construction and checkpoint loading passed. The learned
raster-fed geometry-augmented checkpoint won 5/16 against
`lagged_track_ball_1`, 10/16 against `random_uniform`, and 0/16 against default
`track_ball`; the default
`track_ball` diagnostic row had 5/16 truncations, 47.125 mean steps, and a
-0.6474 shaped loss-delay proxy. Stop signal for this exact first setting:
trace labels are highly imbalanced toward `up`, and the trained policy's eval
actions are also almost all `up`. Pass signal for the next iteration: beat
random clearly, exceed 50% wins versus `lagged_track_ball_1`, and improve
survival/tie against default `track_ball` without dropping below CEM-v2's lag-1
score-pressure baseline.

Current naming rule: CEM-v2 is the score-pressure baseline to beat or
imitate, but it is geometry-only search over the geometry suffix of a
raster-compatible checkpoint and it is not MuZero. Static exact-trace BC is a
valid supervised
raster-input lane, but with the current encoder it is raster-fed
geometry-augmented, not visual-only. The whole current trace-policy family is
weak: the first checkpoint scored 5/16 versus `lagged_track_ball_1`, class
weighting reached only 6/16, mirror-only replay reached only 6/16, mirror plus
oversampling stayed at 5/16, the tiny DAgger append stayed at 5/16, and the
broad DAgger append only matched 6/16 in the balanced diagnostic while failing
random sanity. The raster-only ablation also stayed at 6/16 versus lag-1 and
weak default-`track_ball` survival. Frame-stack linear improved to 13/32 lag-1
wins and 6/32 default-`track_ball` truncations, but still failed the >50% gate
and random sanity. The stack-2 raster-only MLP is now the first visual-only
supervised baseline to pass locally, with 26/32 lag-1 wins, and heldout seed 29
confirmed 43/64 lag-1 wins plus 36/64 random wins. It still does not approach
CEM-v2's 53/64 lag-1 row or 64/64 default-`track_ball` survival tie; heldout
default-track survival was only 23/64 truncations with 62.1719 mean steps. So
CEM-v2 remains the stronger geometry baseline. The next visual-only learner
work is baseline work unless it becomes a real LightZero custom dummy Pong MuZero smoke.
Do Modalization, CEM-v2 comparison, and regularization only with that label,
not as more copies of the same replay.

Data-side replay augmentation is now tested. The builder can add valid
top/bottom mirrored rows with `--include-vertical-mirror`, swapping `up` and
`down`, and can oversample rare labels per ego agent with
`--balance-actions oversample`. Mirror-only augmentation is the better variant:
on the same seed as the original trace smoke, it moved lag-1 from 5/16 to 6/16
and random from 10/16 to 11/16, but default `track_ball` survival fell from
5/16 truncations and 47.125 mean steps to 4/16 truncations and 40.125 mean
steps. Full oversampling made the replay exactly balanced and the checkpoint
predictions roughly balanced, but stayed at 5/16 versus lag-1 and also worsened
survival. Keep mirroring as a cheap symmetry; do not treat duplicated rare
`stay` rows as the missing data.

Closed-loop relabeling has now been tested as both a tiny and broad
raster-fed lane. The first DAgger-style smoke rolled out the mirror checkpoint,
appended 22 exact-labeled visited states, retrained at LR `0.05`, and scored
5/16 versus lag-1, 11/16 versus random, and 4/16 truncations versus default
`track_ball`. That was a plumbing proof plus drift signal, not a promotion
result. Ten visited player-0 states were not exact-scoreable from the current
lagged-opponent memory. The broad follow-up used two behavior checkpoints,
both seats, repeated seed/rollout collection, and `--exploration-epsilon 0.20`,
capped at 1,200 appended rows. It reported 392 unlabelable visited states and
still did not beat mirror-only: the unweighted checkpoint scored 5/16 versus
lag-1 and 10/16 versus random, while the balanced checkpoint matched 6/16
versus lag-1 but dropped to 6/16 versus random. Broad DAgger is now closed as
a failed replay-scaling attempt for this setup. Do not keep scaling this exact
supervised append as the main bet unless the observation or data source changes
qualitatively.

The old imitation command below is retained as historical plumbing. It is not
the current training objective.

The first helper is now:

```sh
uv run python scripts/build_dummy_pong_imitation_replay.py \
  --games 32 \
  --seed 0 \
  --max-steps 120 \
  --output-dir artifacts/local/dummy-pong-imitation-replay-v0
```

Expected files:

- `summary.json`
- `replay_rows.jsonl`

Each replay row has:

- `raster_grid`
- `ego_agent`
- `target_action_id`
- `target_action_label`
- `reward_after_step`
- `next_raster_grid`

## Current Replay Result

A 32-game replay exists at:

```text
artifacts/local/dummy-pong-imitation-replay-v0
```

It contains 7,680 rows over 3,840 environment steps. This is enough for a first
supervised copying run.

Important warning: `track_ball` versus `track_ball` timed out in all 32 games
with zero score reward. That is acceptable for imitation because the target
action exists on every frame. It is not useful evidence for reward learning or
self-play progress yet. For reward-learning tests we need matchups or starts
that produce score events.

Inspect this artifact with:

```sh
uv run python scripts/inspect_dummy_pong_artifacts.py \
  artifacts/local/dummy-pong-imitation-replay-v0 \
  --sample-frames 1
```

## Historical Plumbing Path

1. Build imitation replay from `track_ball`. Done.
2. Add a tiny supervised raster policy learner. Done.
3. Save `checkpoint.npz`. Done.
4. Add Pong eval support for `learned:<checkpoint.npz>`. Done.
5. Compare learned policy against `random_uniform` and `track_ball`. Done once:
   learned beat random 43/64, while `track_ball` beat random 64/64.
6. Add a scoring replay/export path using `track_ball` versus `random_uniform`
   in both seats. Random opponents already produce score events, so no reward
   shaping or biased starts are needed for v0. Done, with both track-ball-only
   and all-ego modes.
7. Train from score-bearing random-opponent replay and repeat checkpoint eval.
   Done once; plain action cloning barely helped from expert-only scoring rows
   and got worse from all-ego rows.
8. Add a value/reward-target smoke from all-ego scoring replay. This should use
   terminal wins and losses as labels, not random actions as expert targets.
   Done once; it proves score-delta return targets and value checkpoint
   save/load, but not policy improvement.
9. Add an angle-control eval probe. Done once: it creates off-center contacts
   and beats random, but `track_ball` still forces truncations.
10. Add a simple Pong checkpoint scoreboard. Done once; it compares latest, older,
   and best-so-far learned checkpoints against `random_uniform` and
   `track_ball` on a recorded seed list when distinct checkpoints are available.
11. Add periodic policy checkpoints to the Pong policy-training attempt. Done;
   the smoke proves reloadable epoch checkpoints and scoreboard compatibility.
12. Run a meaningful longer Pong policy-training attempt and score its periodic
   checkpoints as `previous`, `latest`, and candidate `best`. Done once for
   imitation v0; epoch 1000 is selected on the selection split but still does
   not fully survive or score against `track_ball`.
13. Add a small selection record and heldout command shape for Pong. Selection
   record exists; the first imitation heldout check is done, but a standard
   heldout gate for every new selection is still pending.
14. Add a CPU Modal Pong scoreboard wrapper so remote eval uses the same
   artifacts and Volume pattern as the rest of the stack. Done once with manual
   checkpoint upload to `curvyzero-runs`.
15. Improve the learner itself with self-play policy/value updates. The older
   short-lookahead relabeling from score-delta returns can stay as a diagnostic,
   but first smokes did not score against `track_ball`.
16. Keep contact-outcome and angle-control probes as observability. Use them to
   explain scoreboard failures, not as progress metrics.
17. Historical note: do not bolt MuZero-style search onto the old self-play
   loop. The next real MuZero work should be a separate smoke with model,
   search, replay, training, checkpointing, and eval.

## Recommendation

Do not start with a full project-owned MuZero implementation. Start with a tiny
LightZero custom dummy Pong MuZero smoke, or say plainly that the work is still
baseline land. The Pong checkpoint scoreboard, periodic policy checkpoints, local
selection record, CPU Modal scoreboard wrapper, self-play loop, first lookahead
relabeler, lag-1 closed-loop relabeler, and raster-only ablation are present as
baseline/scaffold work. The current work is to stop replay scaling and check
observation/policy mismatch only if we are staying in baseline land. The
leading baseline hypothesis is that the single-frame raster does not encode
velocity, especially ball direction and the lagged-opponent memory needed by
the `lagged_track_ball_1` target. The 512-game survival audit already says the
old self-play trainer should not continue as the main lane.

The scoreboard should answer: are later checkpoints beating random, tying
`track_ball` by survival, improving against older checkpoints, and improving
survival/loss delay while wins are still zero? The current answer is limited:
existing learned checkpoints beat random sometimes, but `track_ball` wins are
not a valid promotion gate in the default geometry because exact search found
no scoring sequence. In the first periodic checkpoint smoke, epoch 3 was the
best tiny snapshot but still scored 0 wins against `track_ball`. In the longer
imitation run, epoch 1000 selected best on the selection split and beat
previous on heldout, but still scored 0/64 wins against `track_ball`. Those
rows now mean "not full survival yet" or "survival tie", not "failed to find a
winnable strategy."

The angle-control probe is done and gave a useful debug answer: deliberate
off-center hits are possible, but a naive script still cannot score on
`track_ball`. The lookahead relabel smoke reinforced that: plain
score-separated one-step labels collapse back to `track_ball`, while
angle-control tie-break labels can force some truncations but still lose.
The loss-delay smoke added a shaped training target for losing later, but with
`track_ball` tie-break it still produced no labels different from `track_ball`.
Keep the code path, but do not scale that exact setting.
The depth-2 smoke separates some losing action sequences from safe ones, but it
still did not produce target first actions different from `track_ball`.

The contact-outcome dataset is an observability tool. For each near-contact
state, compare top, center, and bottom contact choices over a short horizon.
Record which choice creates better score pressure. Use it only if it helps
explain or improve the scoreboard.

Rows should include intended offset, predicted contact row, whether the target
center was reachable, actual offset, outgoing `ball_vy`, and short score-delta
return. If every choice still returns zero against `track_ball`, change the
geometry before adding a bigger learner: try a stronger geometry change than
width 9 alone, smaller paddle, faster ball, or a slightly deeper ego
action-sequence lookahead.

Current concrete next coach action: stop scaling the old Pong self-play trainer
and use the CEM-v2 lag-1 checkpoint as the first score-pressure baseline. The
survival audit and fresh-replay Modal learner probe are done;
both failed to beat repair ckpt25. The exact beatability probe says the default
`track_ball` hard gate is unwinnable, the target ladder says
`lagged_track_ball_1` is the smallest normal-reset scoreable replacement, and
CEM-v2 has now scored on that replacement. The current geometry-suffix policy
family is not truly visual: no purge is needed for dated logs, but working docs
and future experiment titles should say either `raster-fed geometry-augmented`
or `geometry-CEM` unless the geometry helpers are removed. The raster-fed
static BC comparison has finished: exact trace, class weighting, mirroring,
and oversampling are all weak and below the >50% lag-1 gate. Tiny DAgger,
broad DAgger, and the first raster-only ablation also failed the same gate;
broad DAgger did not improve over mirror-only and raster-only did not rescue
the policy. Frame stack is now implemented/loadable and stack-2 helped
survival and lag-1 wins, but it still failed the >50% gate and random sanity.
A real visual-only baseline should use pixels/one-hot cells only; the first
tiny stack-2 `raster_only` MLP now passes the local lag-1 gate at 26/32 wins
and also beats random 19/32. Heldout seed 29 confirmed 43/64 lag-1 wins and
36/64 random wins, so Modalize the MLP baseline only if we are explicitly
staying in baseline land. Keep the decision scoped: it is still weaker than
CEM-v2, it is not MuZero, and default-track survival is not solved at 23/64
truncations and 62.1719 mean steps. The main work should now be a real
LightZero custom dummy Pong MuZero smoke, or plainly labeled baseline execution.
Stop scaling one-step angle-tie labels, static replay balancing, DAgger append data,
or CEM against default `track_ball` unless the task definition changes.

Detailed next-step notes live in
`docs/working/pong_angle_learning_next_steps_2026-05-09.md`.

Keep all-ego replay for value labels. Do not treat random-action rows as expert
policy targets. Use expert-only rows only when copying scripted behavior is
still useful.
