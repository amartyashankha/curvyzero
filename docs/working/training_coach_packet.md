# Training Coach Packet

Date: 2026-05-08

Scope: durable, short planning packet for the coach/main-thread role in
`/Users/shankha/curvy`. This summarizes the latest user directives and current
training plan. Stable behavior still lives in `docs/design`, `docs/decisions`,
`docs/runbooks`, and source-backed research notes.

First read after memory wipe:
`docs/working/training_coach_handoff_2026-05-09.md`.

## Current Truth / No More Pretending

- We validated stock LightZero CartPole MuZero progression.
- We validated an Mctx search benchmark.
- We validated a CEM-v2 Pong baseline.
- We validated a raster-only MLP Pong baseline.
- We have not run an actual project-owned MuZero/Mctx train loop for Pong.
- We have not run an actual project-owned MuZero/Mctx train loop for Curvy.
- CEM-v2 and the MLP are baselines and scaffolding only. They are not MuZero
  progress.
- Repo choice truth lives in `docs/research/muzero_reference_examples.md`:
  LightZero is now the next practical path for a real dummy Pong MuZero attempt
  because it is the only complete MuZero trainer already proven on Modal.
  Project-owned Mctx is fallback/comparison if LightZero cannot handle the
  custom env cleanly or hides required telemetry/artifacts.
- The next main lane must be a real LightZero custom dummy Pong MuZero smoke on
  Modal. If a command does not call LightZero's trainer, say plainly which
  non-LightZero lane ran.

Prevention rules:

- Prove the target is scoreable before scaling it.
- Keep baselines separate from MuZero.
- Name the algorithm in every experiment title, command, and summary.
- Distinguish stock LightZero MuZero from project-owned MuZero/Mctx.
- Do not describe CEM, imitation, or MLP results as MuZero progress.
- When a Pong summary says `0/N wins`, it must also report survival length,
  truncation rate, shaped loss-delay return, score return, and survival
  variance. A flat win rate alone is not a usable learning signal.

## Role And North Star

- Main thread role: plan, delegate, and decide only. Workers do edits, runs,
  and docs. The main thread keeps constraints visible, assigns narrow work, and
  turns worker output into decisions.
- Coach north star: build a mature Modal-backed training stack. Modal is the
  real execution target for serious runs. Local is for tiny debug runs only.
- Current eval north star: a boring checkpoint scoreboard. For Pong, ask
  whether learned checkpoints beat `random_uniform`, survive or tie the default
  `track_ball` floor, and improve against older/best checkpoints over fixed
  seeds.
- Current focus: move the main toy work toward Pong because it has a visual
  input path. Keep survival as a small diagnostic tool, not the main project.
- Current Pong path: CEM-v2 remains the score-pressure geometry baseline on
  `lagged_track_ball_1` and is Modal-backed baseline scaffolding, not MuZero
  progress. The current
  trace-policy family should not be called truly visual: it is raster-fed but
  still uses decoded geometry suffix features unless explicitly run in
  `raster_only` mode. Static trace BC, class weighting, mirror-only replay,
  mirror-plus-oversampling, tiny DAgger, broad DAgger, and the first raster-only
  ablation all failed the >50% lag-1 win gate and did not improve over the
  mirror-only 6/16 diagnostic. Frame stack is now implemented and loadable:
  stack-2 linear improved lag-1 wins to 13/32 and default `track_ball`
  truncations to 6/32, but still failed the >50% lag-1 gate and random sanity.
  Stack-2 raster-only MLP is the first truthful visual-only supervised
  baseline to pass the lag-1 gate: local smoke scored 26/32 wins versus
  `lagged_track_ball_1`; heldout seed 29 confirmed 43/64 versus
  `lagged_track_ball_1` and 36/64 versus `random_uniform`.
  Default `track_ball` remains a survival/tie floor, not a hard win gate; the
  same heldout scoreboard had only 23/64 truncations and 62.1719 mean steps
  versus default `track_ball`.
- Current Pong next decision: the stack-2 linear failure plus stack-2 MLP
  result supports the observation/policy mismatch diagnosis. Missing history
  and a too-weak linear policy were key issues. Heldout confirmed the
  raster-only MLP baseline is real enough to Modalize, but it is still weaker
  than CEM-v2 and still not MuZero: CEM-v2 remains the stronger geometry
  baseline at 53/64 lag-1 wins and 64/64 default-`track_ball` truncation ties.
  Default-track survival is not solved.
- Reader-facing Pong explanation: the recent numbers came from separate
  runs/probes, not one continuous run. Comparable checkpoint rows are those
  scored by the same scoreboard setup: same geometry, opponent/target, episode
  count, split/seed role, and metric bundle. A checkpoint is continued only
  when a later command explicitly resumes from it or uses it as the behavior
  policy for new replay. The changes that moved metrics were target/opponent,
  data/replay, policy architecture, feature mode, and Modal wrapper; the env
  reward stayed score-delta +1/-1/0.
- Glossary for Pong questions: default `track_ball` follows the current ball
  row and is a survival/tie floor in the current default geometry, not an
  optimality proof for every Pong setup. `lagged_track_ball_1` follows the
  previous ball row, giving the opponent a one-tick lag and making normal-reset
  scoring possible. Shaped proxy means loss-delay telemetry/training target,
  not the environment reward. Report wins, mean/median survival steps,
  truncation rate, opponent wins/losses, and shaped proxy together; do not
  collapse learned-vs-`track_ball` back to a bare fraction. Variance is not the
  environment reward, but it can be used as a small, bounded exploration or
  checkpoint-selection signal when mean score/survival is similar; record it
  instead of forgetting it.
- Latest Pong Modal result: a small parallel CPU Modal sweep completed
  remotely. All checkpoints still won 0/64 against `track_ball`, but the raw
  `episodes.jsonl` audit shows why wins alone were an incomplete summary:
  repair ckpt25 remains best with 47.30 mean steps, 20/64 truncations, 44/64
  `track_ball` wins, and a -0.6467 learned shaped proxy. The best sweep row,
  higher-diversity e25, reached 46.41 mean steps, 19/64 truncations, 45/64
  `track_ball` wins, and -0.6582 shaped proxy. This blocks win-based promotion
  and makes survival/loss-delay telemetry mandatory for the next diagnosis.
- `track_ball` is now an eval and curriculum baseline, not the main teacher.
  Imitation, lookahead, angle probes, and contact probes are diagnostics and
  plumbing only.
- Old Pong self-play was not merely weak; it was being asked to beat an
  impossible default `track_ball` target. Future win-pressure work needs a
  weaker or changed target ladder first.
- CEM-v2 corrected that mistake by training against the proven-scoreable
  `lagged_track_ball_1` rung and monitoring score pressure directly. New rule:
  before scaling a target, prove it is scoreable and define the fine-grained
  metric that will show progress before headline wins move.
- Environment reconstruction is separate active work. Respect its API and never
  call `curvyzero-v0` source-faithful, but do not let source-matching work
  distract this training work.
- Current training loop: debug tiny locally, then run serious train/eval jobs
  on Modal as whole jobs. Every run writes a structured summary/output file and
  gets critiqued before more compute.
- Coaching stance: prefer runnable proof over prose. If a Modal pattern
  matters, make or find a small quick run that proves it works.

## Reorientation - 2026-05-09

Read `docs/working/training_coach_reorientation_2026-05-09.md` before resuming
this lane. Short version: survival is now a side diagnostic, not the core
learning story; Modal is the serious-run target, while local is only for tiny
debug; Pong should be emphasized as the LightZero-first custom-env MuZero lane,
with baselines kept clearly separate; Tiny Line Duel remains the
CurvyTron-shaped multiplayer scaffold. The next claim should separate learned
behavior from scripts, planner rules, fixed-seed luck, and best-checkpoint
luck.

Read `docs/working/pong_selfplay_training_plan_2026-05-09.md` as an active Pong
working hypothesis under critique. Read
`docs/working/pong_training_critique_wave_2026-05-09.md` before scaling it; the
generation/promotion framing is under review and may be the wrong abstraction.

## Current Active Lanes

Keep the active work in this order until the facts change:

1. LightZero dummy Pong custom-env config smoke: prove LightZero can
   create/reset/step a tiny Pong ego-view env with `A=3`, observations, seeds,
   rewards, done/truncated behavior, and a tiny MuZero config. This must not
   train.
2. LightZero dummy Pong tiny train smoke: after config works, run a brutally
   capped real LightZero MuZero trainer on Modal and emit CurvyZero scorecard
   telemetry: wins, survival, shaped loss-delay, variance, seeds, actions, and
   artifact refs.
3. Fallback project-owned MuZero/Mctx smoke: use only if LightZero fails or
   hides required telemetry/artifacts. A valid fallback smoke must include
   project-owned model/search, replay, training, checkpointing, and eval.
4. CEM-v2 baseline: keep the recovered lag-1 score-pressure learner as the
   geometry baseline, now Modal-backed with structured monitor and scoreboard
   artifacts. This is not a MuZero lane.
5. Raster-fed/visual Pong baselines against `lagged_track_ball_1`: static
   trace BC, class weighting, mirror/oversampling, tiny DAgger, broad DAgger,
   and the first raster-only ablation are all below the >50% lag-1 gate. Frame
   stack loadability is proven, stack-2 linear helped but still failed the gate
   and random sanity, and stack-2 raster-only MLP passed the local lag-1 gate at
   26/32. Heldout seed 29 then confirmed the supervised baseline at 43/64
   versus lag-1 and 36/64 versus random, with 23/64 truncations and 62.1719
   mean steps versus default `track_ball`. Do not label the current
   geometry-suffix checkpoints truly visual. Do not scale more of the same
   replay append/balancing lane.
   Decision: treat the MLP work as a Modal-backed supervised baseline, keep
   CEM-v2 as the stronger geometry baseline, and keep default-track survival
   open. Use score wins as the hard metric and survival/loss-delay only as
   supporting telemetry. Do not call either baseline MuZero.
6. Pong survival and exploration audit: every learned-vs-`track_ball` read
   must include wins, mean/median/p90 episode steps, truncation rate, score
   return, shaped loss-delay proxy, and survival variance. Variance can be a
   small early exploration/selection tie-breaker, not a replacement objective.
7. Existing LightZero/stock MuZero example validation: keep stock Atari Pong
   blocked at the ROM/license gate and do not let Atari ROM work distract the
   dummy Pong custom-env lane.
8. Pong learner/curriculum replacement: the old self-play trainer is stopped
   as the main lane after the 512-game survival audit got worse. Treat repair
   ckpt25 as the best old survival baseline.
9. Modal/Mctx benchmark: keep Mctx isolated and measure the fixed-shape
   synthetic path before wiring it to environments, replay, or trainers.

## Directives

- Do not revert or overwrite others' edits. Respect owned files and concurrent
  agents.
- Keep language simple. Avoid new categories or labels unless they remove real
  confusion. If a technical word is needed, explain it.
- Source, command output, and tiny runnable quick runs beat memory, chat summaries,
  and handoff prose.
- Save new knowledge continuously: research conclusions, run results, rejected
  paths, and useful critique must land in the right doc/index/runbook, not stay
  only in chat or worker handoffs.
- Do not treat `curvyzero-v0` as source-faithful. It is a simplified training
  ruleset.
- Keep `source-kinematics` movement-only; do not let it become a second
  source clone.
- Stay parallel. Use workers for Modal research, toy experiments, docs updates,
  MuZero/library critique, and small code slices when the scope is clear. The
  main thread should not do worker jobs except for tiny coordination fixes.
- Do not run pytest unless explicitly redirected. For this lane, use training
  quick-run commands, import checks, Modal dry/small runs, and output-file
  inspection.
- Avoid browser hosting, pixels, and source-matching rabbit holes unless
  they directly unblock a training run.

## Current Architecture Truth

- `src/curvyzero/` should stay algorithm-neutral. Simulator, wrappers, search,
  training, and infra remain separate.
- The simulator boundary is protected. Training libraries and wrappers use it;
  they do not define its core API.
- Rulesets are distinct:
  - `curvyzero-v0`: first 1v1 no-bonus training toy.
  - `curvytron-v1-reference`: source-derived behavior from the local original.
  - `curvytron2-reference`: later public reference if needed.
- The useful comparison layer is common trace, not raw JS/Python trace layout.
- Local single-scenario and local batch source checks come before remote batch
  wrappers.
- Current probe backlog favors movement batch, forced two-player common trace,
  multi-step motion, normal-wall death, borderless wrap, trails, same-tick
  death, and scoring.

## Modal Rules

- Use Modal early for whole-job quick runs, benchmarks, GPU checks, training,
  eval, and file storage. Serious runs belong on Modal.
- Use local runs only for tiny debug, import checks, and quick artifact-shape
  checks.
- Never call Modal inside `env.step()`, JS ticks, MCTS node expansion, action
  selection, normalization, or trace diff hot loops.
- Modal Functions should run whole jobs: tests, benchmarks, self-play shards,
  training runs, evaluation runs, replay conversion, file packaging, and
  source-check batches.
- Use Modal Volumes or buckets for checkpoints, replay chunks, logs, profiles,
  videos, trace folders, and small JSON pointer files.
- Make long Modal runs resumable, checkpointed, idempotent, retry-safe, and
  cost-aware.
- Return structured summaries and file refs, not giant logs.
- Single-node comes first. Multi-node Modal clusters are a later pattern for
  large jobs, not part of the dummy training run.
- Every Modal pattern we plan to rely on needs either a local example, official
  doc link, or a tiny CurvyZero smoke.
- Current Modal storage status: `curvyzero-runs` Volume works for dummy
  survival and Pong. Pong now has a CPU Modal train wrapper and a CPU Modal
  scoreboard wrapper. The 2026-05-09 Pong parallel sweep trained and scored
  remotely on Modal CPU. This proves the remote train-to-eval path, not policy
  quality. Remote resume is still not proven. The real execution target remains
  Modal; see
  `docs/research/modal_training_execution_plan.md`.
- Modal simplification correction: use one training app, a tiny set of coarse
  Functions, `curvyzero-runs` for run artifacts, and an optional
  `curvyzero-cache` Volume for Hugging Face/torch/JAX caches. Do not add more
  one-off wrappers unless they prove a new primitive. Current Pong should stay
  CPU until a JAX/Mctx or GPU-model learner exists. See the blunt critique and
  exact lane in `docs/research/modal_training_execution_plan.md`.

## First Experiments

- First local command: `scripts/run_toy_baseline.py` runs random-vs-random and
  a privileged heuristic-vs-random matchup and emits JSON summaries.
- Current toy command proves code runs; it does not prove learning. The first
  100-episode quick run had the heuristic losing to random, so improve the toy
  environment/control/heuristic before claiming progress.
- First dummy single-player loop: `scripts/run_dummy_survival_train.py` writes
  summary, checkpoint, and metrics files around a simple tabular MuZero-like
  learner.
- First dummy two-player loop: `scripts/run_dummy_line_duel_train.py` exercises
  simultaneous actions, same-cell/cross-swap deaths, sparse win/loss/draw
  rewards, and one replay row per player view. The first quick run ended in all
  draws in final eval, so treat it as a code-path check, not learning progress.
- First eval tables:
  - `scripts/run_dummy_survival_eval.py` gives EVAL1 random-vs-scripted floors.
    First smoke: random crashes every episode; one-step-safe survives every
    episode.
  - `scripts/run_dummy_line_duel_eval.py` gives EVAL2 random/sticky/scripted
    paired-seat baseline matrix. First smoke: one-step-safe beats random and
    sticky in both seats, while mirror rows expose deterministic draw cases.
- Learned-checkpoint eval is wired for both dummy tasks. Current learned
  checkpoints are not good yet: survival still crashes every eval episode, and
  Line Duel is collision-prone and loses to random/one-step-safe in the first
  checkpoint smoke. That is acceptable; the evaluator is now honest enough to
  show lack of progress.
- A small dummy survival learning-curve probe showed more data is not enough:
  two 20x50 local runs still crashed every diagnostic eval episode. Do not scale
  that dummy learner blindly; inspect checkpoint selection/trainer/planner
  behavior first.
- Dummy survival had an early positive checkpoint result: adding a
  tiny planner safety mask produced iteration-2 and iteration-4 checkpoints
  with 100% survival on a single diagnostic eval list, matching `one_step_safe`.
  Later checkpoints degraded, so best-checkpoint selection matters. This has
  since been downgraded: an untrained model with the same planner also solves
  the tiny repeated diagnostic list, so the current result comes from planner/checkpoint
  behavior, not clean learned-policy evidence.
- Survival eval defaults now include `untrained_model_same_planner`, and sweep
  files include selected/latest checkpoint visibility plus a
  `selection_record.json`.
- Tiny Line Duel eval now records split metadata and `pair_groups`; use paired
  groups, not raw seated rows, as the multiplayer claim unit.
- Dummy Pong exists as a sidecar two-player eval scaffold. Its tabular
  observations are for debugging; a future real MuZero/Mctx Pong path can use
  the tiny visual/raster observation on `PongEnv.raster_observation()` plus
  trace artifacts via `frames.jsonl`.
- Pong should now get either a real project-owned MuZero/Mctx training smoke or
  be named plainly as baseline work. Existing raster replay, imitation
  training, CEM, checkpoint loading, and scoreboards are plumbing evidence, not
  MuZero progress.
- Pong eval reward should stay score/outcome based. The first self-play trainer
  should also log a shaped training return: wins are `+1`, fast losses are near
  `-1`, and longer losses can recover up to half a point. See
  `docs/working/pong_selfplay_training_plan_2026-05-09.md` and
  `docs/research/pong_reward_design.md`.
- Pong scoring data check showed random opponents are enough for score events:
  `track_ball` beat `random_uniform` in all 64 seated games with no truncations.
  Use this before adding any reward shaping or biased starts.
- A first scoring replay exists, but it stores only winning `track_ball` ego
  rows. It is useful for a first policy/value smoke, not a balanced value set.
- All-ego scoring replay now exists and includes positive and negative reward
  rows. Use it for value targets. Do not use random-action rows as expert policy
  targets.
- Pong value-target smoke now exists. It proves score-delta return labels,
  tiny value fitting, checkpoint save/load, and summary writing. It does not
  prove policy improvement.
- Pong paddle-angle smoke confirmed the important mini North Star: off-center
  paddle contacts change return angle. The next Pong work should measure
  whether a policy can use that against `track_ball`.
- Pong angle-control probe now exists. The scripted probe made off-center
  contacts and beat random, but it still timed out against `track_ball`. This
  says the next target is useful return sequences, not the raw bounce rule.
- Audit note: `track_ball` already creates many off-center contacts due to lag,
  so do not optimize or celebrate off-center rate alone. Use post-contact
  score pressure or short return.
- Pong contact-outcome probe now exists. Top/center/bottom contacts produce
  different outgoing `ball_vy`, but all short score-delta returns stayed zero
  against `track_ball` in the default geometry. Treat it as observability, not
  scoreboard progress.
- Width-9 contact-outcome probe also stayed flat: every candidate score-delta
  return was `0.0` even though outgoing `ball_vy` differed. Width alone did not
  create the missing training signal.
- Pong checkpoint scoreboard now exists. It writes baseline rows,
  checkpoint-vs-baseline rows, optional learned-vs-learned rows, split metadata,
  checkpoint specs, `summary.json`, and `episodes.jsonl`. The first smoke used
  the same checkpoint as `latest` and `previous`, so it proves plumbing only.
- Pong imitation training now supports periodic policy checkpoints via
  `--checkpoint-every-epochs`. The first periodic smoke wrote epoch 1, 2, and 3
  policy snapshots plus the final root `checkpoint.npz`.
- The checkpoint scoreboard can consume those periodic checkpoints. A tiny
  epoch-1-vs-epoch-3 smoke proved the path, but quality was bad: both
  checkpoints still lost to `track_ball`, and the learned-vs-learned row tied.
- A longer 1000-epoch imitation run wrote epoch 250/500/750/1000 checkpoints.
  The selector picked epoch 1000 on the selection split because it beat random
  42/64 and beat most earlier checkpoints, but it still won 0/64 against
  `track_ball`.
- Heldout for selected epoch 1000 is mixed but useful: selected beat previous
  50/64 and lost less often to `track_ball`, but still won 0/64 against
  `track_ball`. Previous beat random slightly more often on heldout, so do not
  claim broad policy quality.
- The distinct-checkpoint scoreboard compared `imitation_v0`, `scoring_expert`,
  and `scoring_all_ego`. All three beat random more often than not on the
  monitor smoke, but none scored against `track_ball`.
- CPU Modal Pong scoreboard now exists and has one remote smoke. It reads
  checkpoint refs from `curvyzero-runs`, writes eval outputs back to the Volume,
  and returns refs/hashes. It proves remote eval plumbing, not policy quality.
- Short-lookahead relabeling now exists. The larger angle-tie run produced 442
  labels different from `track_ball`, but all lookahead checkpoints still won
  0/64 against `track_ball`; selection stayed with imitation epoch 1000. Do not
  scale one-step angle-tie labels further unless a bug is found.
- Loss-delay lookahead exists as an old training-label diagnostic:
  `--loss-delay-alpha` gave losing candidate rollouts small credit for losing
  later. The first smoke with alpha `0.05` produced 0 labels different from
  `track_ball`, and the trained checkpoint still won 0/32 against `track_ball`.
  Do not scale that exact lookahead setting. Reuse the core idea only in the
  new self-play shaped episode return.
- Pong self-play replay and training now exist locally. First smoke:
  16 random-vs-random games produced 498 replay rows, trainer wrote epoch 25
  and epoch 50 checkpoints, and the scoreboard loaded them. Quality is weak:
  `selfplay50` tied random 8/16, beat `selfplay25` 10/16, and won 0 games
  against `track_ball`.
- Manual Pong self-play generation 2 also exists. It collected from
  `selfplay50` with epsilon exploration and wrote epoch 25/50/75 checkpoints.
  `gen2_50` improved slightly versus random on one monitor split, but every
  gen2 checkpoint lost to the parent and won 0 games against `track_ball`. Do
  not promote it. Also do not use the 0-win row alone as the full diagnosis;
  compare survival/loss-delay shaped metrics before deciding whether the
  objective is moving.
- Current Pong correction: decide whether to keep repairing the crude self-play
  trainer or switch to a known simple baseline/curriculum such as PPO,
  actor-critic, or CEM. Keep fixed-baseline scoreboards first either way. The
  small `policy_grad = probs.copy()` aliasing repair is done, but gen2 still
  missed the old invalid win gate, so do not scale the objective blindly.
- Current observability gap: scoreboards show outcomes, but not enough run
  health. Every new Pong run should emit iteration metrics, action histograms
  by seat, entropy/collapse metrics, terminal causes, failure examples, and
  heldout results after selection. Against `track_ball`, it must also report
  episode length, truncation rate, and shaped return/loss-delay proxy.
- 2026-05-09 Modal Pong parallel sweep: three CPU Modal variants completed
  through `dummy_pong_train_attempt` and `dummy_pong_scoreboard_attempt`. All
  still had 0/64 wins against `track_ball`; best truncation pressure was 19/64,
  below the prior 20/64 repair run. The raw episode audit confirmed the same
  ordering by mean steps and shaped loss-delay proxy: repair ckpt25 had 47.30
  mean steps and -0.6467 proxy, while the best sweep row, higher-diversity e25,
  had 46.41 mean steps and -0.6582 proxy. Recommendation: stop blind self-play
  scaling, keep the CPU Modal lane, and never report learned-vs-`track_ball`
  wins without survival/loss-delay metrics.
- Longer-run feasibility read: the existing self-play trainer is not obviously
  broken. Its shaped-return fields exist, and the policy-gradient sign is sane.
  The one cheap fresh-replay Modal probe is now done: 512 games, 75 epochs,
  checkpoint every 25, low policy LR, then survival-audited scoring. It got
  worse than repair ckpt25, so stop using the old self-play trainer as the main
  lane unless a concrete learner/objective bug is found. A new Pong learner or
  curriculum is a separate lane, not more blind generation scaling.
- Depth-2 ego-sequence lookahead now exists. The first strict smoke emitted 10
  non-tied avoided-loss rows, but all target first actions still matched
  `track_ball`, so it was not trained.
- Modal run helper functions now exist in
  `curvyzero.infra.modal.run_management` for run/attempt ids, Volume refs,
  file summaries, and latest/best checkpoint pointer files. A dummy-survival
  train-attempt wrapper also exists. It does not yet implement resume.
- Dummy survival degradation is diagnosed: later checkpoints learn a
  crash-heavy, non-positive value landscape where lower-clearance or unknown
  actions can look less negative than safer known actions, overriding the
  planner safety prior.
- Basic checks before serious MuZero: random stress, heuristic-vs-random, and
  one simple policy baseline.
- Start with sparse terminal rewards for environment/eval. Training-only shaping
  is allowed as an experiment, but only scale it if it creates action labels or
  value targets that actually improve the scoreboard.
- Use compact egocentric rays first; consider heading-aligned local rasters
  later for CNN/MuZero.
- Keep fixed debug seeds diagnostic. Eval waves should sample fresh
  pseudo-random eval seeds, record the generator seed/list, and use many starts
  for claims. Do not tune against one reused eval seed list.
- First project-owned MuZero/search spike is JAX/Mctx after a synthetic smoke;
  LightZero is the stock-example reference lane. The stock LightZero CartPole
  progression smoke is validated; stock Pong is still blocked at the ROM gate.
  Do not jump from dummy Pong baselines to "MuZero".
- User preference for first algorithm experiments: start single-player/dummy
  before multiplayer. Map back to multiplayer later through ego-perspective
  rows and opponent policies.
- Do not make training claims until the source-matching bar is explicit and met.

## Open Questions

- Which exact `curvyzero-v0` differences must close before the first serious
  learning run?
- What is the minimum common-trace field set for wall death, scoring, trail
  events, and same-frame multiplayer deaths?
- Should the old CurvyTron build be tested in a disposable local copy or pinned
  Modal image?
- What artifact layout becomes standard across local and Modal batches?
- When do 3-player and 4-player canaries become blocking rather than advisory?
- What cost fields belong in experiment logs before larger Modal GPU/MCTS runs?
- What is the smallest single-player dummy MuZero-style task that exercises
  replay, model, search, checkpointing, and Modal artifacts without inheriting
  multiplayer complexity?
- Which Modal patterns are validated locally versus only read from docs?
- What is the smallest LightZero dummy Pong custom-env smoke that proves the
  framework can train without hiding scorecard telemetry or traces?

## Worker Memory

Workers should take narrow assignments from the active lanes above and leave a
dated experiment log or doc update. The main thread should use worker output to
make the next decision, not start a second hidden queue.

Recently closed:

- Pong eval review: fixed baselines are the main scoreboard; latest-vs-old
  checkpoint eval is useful after baseline rows are sane.
- Pong contact-outcome probe: useful diagnostics, but flat score returns
  against `track_ball` in the default geometry.
- Pong checkpoint scoreboard: implemented as a command and smoke artifact; real
  old-vs-new eval now needs meaningful periodic policy checkpoints from a
  stronger training attempt, not just the tiny artifact smoke.
- Pong distinct-checkpoint scoreboard: useful current baseline table; it says
  the learned policy attempts are still below `track_ball`.
- Pong angle-control audit: contact detection is plausible; off-center rate by
  itself is weak.
- Modal setup critique: Modal whole-job and Volume/run-attempt patterns are
  correct where they exist, no current training job uses Modal GPUs, and the
  blocker is learner quality rather than compute.
- Setup critique refresh: the current patterns are right for this stage. Pong
  train/eval runs are real Modal CPU jobs. GPU is only for JAX/Mctx smokes and
  the synthetic benchmark until the next learner actually uses GPU work.
- CPU Modal Pong scoreboard wrapper: implemented and remote-smoked against two
  Volume checkpoint refs. It proves remote eval artifacts, not policy quality.
- Pong selection record: implemented locally. The 1000-epoch imitation run
  selects epoch 1000 on the selection split, but that checkpoint still won 0/64
  against `track_ball`.
- Pong heldout scoreboard: selected epoch 1000 beats the previous checkpoint
  but still misses the `track_ball` gate.

Future workers should be bounded by file ownership and should leave durable
docs or artifacts. Do not start broad research if the next scoreboard step is
clear.

## Documentation Hierarchy

Active spine. Read these first and keep them consistent:

- [training_coach_packet.md](training_coach_packet.md): current coach state,
  north star, active priorities, and next actions.
- [training_experiment_backlog.md](training_experiment_backlog.md): active lane
  list and working memory.
- [pong_training_critique_wave_2026-05-09.md](pong_training_critique_wave_2026-05-09.md):
  current Pong repair-vs-baseline critique.
- [pong_selfplay_training_plan_2026-05-09.md](pong_selfplay_training_plan_2026-05-09.md):
  active Pong self-play hypothesis under critique and shaped training return.
- [../research/modal_training_execution_plan.md](../research/modal_training_execution_plan.md):
  Modal/Mctx execution pattern and benchmark lane.
- [../research/muzero_reference_examples.md](../research/muzero_reference_examples.md):
  LightZero/stock MuZero reference validation lane.
- [../runbooks/training_smokes.md](../runbooks/training_smokes.md): exact local
  commands to rerun smokes; self-play commands are reproduction only.
- [../design/training_eval_protocol.md](../design/training_eval_protocol.md):
  stable eval rules.
- [../experiments/README.md](../experiments/README.md): dated evidence index.

Historical or supporting docs. Do not let their old next-action lists compete:

- [training_loop_agenda.md](training_loop_agenda.md): broader state and
  historical queue.
- [pong_training_plan.md](pong_training_plan.md): historical Pong plumbing and
  diagnostics. Do not follow its old imitation-first path.
- [modal_training_setup_critique_2026-05-09.md](modal_training_setup_critique_2026-05-09.md):
  historical Modal/training setup critique.
- [../research/training_evaluation.md](../research/training_evaluation.md):
  background and eval rationale.

## Next 10 Actions

1. Keep the Pong survival audit first: default `track_ball` is a survival/tie
   floor, not a hard win gate. Loss-delay metrics decide whether a 0-win row is
   moving.
2. Keep fixed-baseline scoreboards first: `random_uniform`, `track_ball`,
   parent/previous, and selected-best rows when checkpoints exist.
3. Keep the existing LightZero/stock MuZero reference facts separate from
   project-owned work; do not start real stock Pong training yet.
4. Stop the old Pong self-play trainer as the main lane. The 512-game
   feasibility probe got worse than repair ckpt25, which remains the best old
   survival baseline.
5. Run or document the fixed-shape Modal/Mctx benchmark before wiring Mctx to
   real env, replay, or trainer code.
6. Add run-health observability to any new Pong run: iteration metrics, action
   histograms by seat, entropy/collapse metrics, terminal causes, and failure
   examples.
7. Use selection records after meaningful scoreboard runs; run heldout only
   when a candidate is worth confirming.
8. Keep local runs tiny. Serious train/eval attempts go on Modal CPU unless the
   learner actually uses JAX/Mctx or GPU model training.
9. Keep angle-control and contact-outcome probes as debug tools only.
10. Before scaling more replay, run the LightZero dummy Pong custom-env config
   and tiny-train smokes, or say plainly that the work is fallback/baseline
   work. Keep any new Pong learner or curriculum separate from the old
   generation loop, with fixed-baseline evals and survival/loss-delay reporting
   from the first run.
See [Training Experiment Backlog](training_experiment_backlog.md) for the active experiment lanes and current priority.
