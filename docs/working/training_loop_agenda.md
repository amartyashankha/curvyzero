# Training Loop Agenda

Date: 2026-05-08

Scope: current coach notes for building the CurvyZero training loop. Keep this
short and plain. Move stable decisions into `docs/design/` or `docs/decisions/`
only after we have run evidence.

Language rule: use plain words. Keep technical terms only when they name real
files, commands, code fields, or known tools. When a term stays, explain it.

Current execution rule: local is for tiny debug only. Serious train/eval runs
belong on Modal as whole jobs with saved artifacts.

## Current State

- Current focus: Pong should get more attention now because it has the visual
  input path we want for MuZero-style work. Survival stays useful as a small
  trainer/eval diagnostic, but it should not keep pulling the main plan away
  from visual training.
- Current LightZero Pong truth: Modal training, checkpoint mirroring, and
  independent MCTS eval work, but reliable baseline improvement has not been
  shown. The repeated-seed trainer telemetry was misleading, and the
  post-seed-fix 1024/16 validation still failed seed trust: seed `3` had
  129/148 train rows, `seed_dominance_warning=true`, unique seeds 20. The
  independent paired MCTS scorecard stayed weak: random `31-33`, lagged
  `27-37`, track `0-57`, with learned actions `[0,1677,8005]`.
- Current LightZero Pong is not true self-play. It is learner ego versus a
  scripted opponent. The first real self-play step should be learner versus a
  frozen older checkpoint-backed opponent, with paired seats and held-out eval.
  Full multiplayer and joint-action search remain later CurvyTron work.
- Current next evidence: patch LightZero seed config so dynamic mode wins
  unless config is explicitly fixed, then add a `player_0`-only scorecard
  option. Do not scale the current lane. The first dynamic-seed patch appears
  not to have survived the LightZero env-manager path, so this is likely a
  deeper seed plumbing issue.
- Current reset/randomization read:
  `docs/working/lightzero_pong_reset_randomization_critique_2026-05-09.md`
  is superseded by the failed post-seed-fix run: dynamic seeding must win in
  the LightZero env-manager path before broader reset profiles. Mild paddle-y
  jitter remains the first later reset-profile knob.
- Current performance read:
  `docs/research/training_loop_bottlenecks_amdhals_law_2026-05-09.md` says to
  measure Python env steps, LightZero MCTS/search, trainer loop, replay
  staging, and Modal startup before jumping to GPU/vectorization or larger
  Modal jobs.
- Core environment: `CurvyTronEnv` is a tiny 1v1 NumPy toy game with sparse
  terminal rewards and flat global observations.
- First local command exists: `scripts/run_toy_baseline.py`.
- Toy result so far: random-vs-random runs. The privileged heuristic lost to
  random in the first 100-episode quick run. That proves the command works; it
  does not prove the game is learnable.
- First dummy training loop exists: `scripts/run_dummy_survival_train.py`
  drives `curvyzero.training.dummy_survival`, writes output files
  (`summary.json`, checkpoints, metrics), and uses MuZero-like function names
  around a simple tabular NumPy learner.
- First two-player dummy loop exists: `scripts/run_dummy_line_duel_train.py`
  drives `curvyzero.training.dummy_line_duel`, writes output files, and checks
  simultaneous two-player steps plus one replay row per player. The first final
  eval was all draws, so it proves the code path runs; it does not show
  learning progress.
- Evaluation research exists in `docs/research/training_evaluation.md`; near
  term evals should be small recorded-seed tables against random/scripted/frozen
  checkpoint opponents, not a league. Each new eval wave should sample fresh
  pseudo-random eval seeds and record the generator seed/list.
- Eval rules v0 now live in `docs/design/training_eval_protocol.md`.
- First recorded-seed eval tables exist:
  - `scripts/run_dummy_survival_eval.py` compares random, one-step-safe, and
    untrained-model-same-planner. First quick run: random crashed in all 50
    episodes; one-step-safe survived all 50 episodes. The planner-only baseline
    now also solves the tiny repeated diagnostic seed list. That means we cannot say the
    learned model caused the survival result yet.
  - `scripts/run_dummy_line_duel_eval.py` compares random/sticky/one-step-safe
    across seat assignments. First smoke: one-step-safe beat random and sticky
    in both seats; mirrors expose useful deterministic draw cases.
- Learned-checkpoint eval now works for both dummy tasks:
  - Survival checkpoint from `artifacts/local/dummy_survival_smoke` loaded and
    lasted longer than random on a 10-episode quick run, but still crashed every
    episode.
  - Tiny Line Duel checkpoint from `artifacts/local/dummy_line_duel_smoke`
    loaded and was weak/collision-prone: lost to random_uniform and
    one_step_safe, mixed with random_sticky on a 5-episode smoke.
- Dummy survival learning-curve probe tried two larger 20x50 local runs. More
  data alone did not improve learned-checkpoint eval: both checkpoints still
  crashed every single-seed diagnostic eval episode, with mean steps 10.0 and 25.0
  versus the prior 25.5 smoke.
- Diagnosis found the dummy planner was optimistic about unknown/zero-valued
  actions and collapsed deterministic ties. A small safety-aware planner patch
  now penalizes zero-clearance actions and ties toward safer/straight actions.
- First safety-planner quick run showed a useful checkpoint pattern:
  periodic checkpoints at iterations 2 and 4 reached 100% survival on the
  single-seed diagnostic eval, matching `one_step_safe`, while later checkpoints
  got worse. A planner-only baseline also solves that replay list, so this is not clean
  evidence that the model learned.
- Dummy Pong now exists as the main visual toy task. Treat tabular Pong
  observations as debugging data; the learning-facing Pong path uses
  `PongEnv.raster_observation()` and `frames.jsonl`.
- Pong working hypothesis now lives in
  `docs/working/pong_selfplay_training_plan_2026-05-09.md`: self-play games,
  replay rows, shaped training returns, policy/value update, checkpoints, and
  scoreboard eval. It is under active critique in
  `docs/working/pong_training_critique_wave_2026-05-09.md`. Treat it as a
  scaffold, not a source of truth.
- Pong imitation replay now exists. The larger v0 replay has 7,680 raster rows
  from `track_ball` targets. It is useful for copying a scripted policy, but
  all games timed out with zero score reward, so it is not reward-learning
  evidence.
- Pong imitation training now writes a first learned raster checkpoint at
  `artifacts/local/dummy-pong-imitation-train-smoke/checkpoint.npz`. It copies
  `track_ball` with about 99.0% validation accuracy on replay rows. This is a
  supervised copying smoke, not reward learning.
- Pong learned-checkpoint eval now works. On a 32-episode eval, the checkpoint
  beat random 43/64, but scripted `track_ball` beat random 64/64. The checkpoint
  is runnable, but not yet a strong copy.
- Pong eval reward should use score changes only: +1 when ego scores, -1 when
  the opponent scores, and 0 otherwise. The first self-play trainer should use
  a separate shaped episode return that gives partial credit for longer losses.
  Rally length, paddle hits, and ball distance stay debug logs.
- Pong scoring smoke showed random opponents produce clean score events:
  `track_ball` versus `random_uniform` scored every game in both seats. Use
  random opponents for the first reward-data path.
- Pong scoring replay now supports `--row-policy all`, so it can emit both
  winning and losing ego rows. The inspector now reports positive/negative
  reward row counts so cancellation is not mistaken for zero reward data.
- Training a plain action clone from all-ego replay was worse than expert-only
  cloning because it learned from random action rows too. All-ego replay is
  good for value labels because it has wins and losses. It is bad for expert
  action cloning because it has random actions.
- Pong value-target smoke now exists. It backs up score-delta rewards into
  scalar return labels, fits a tiny raster value regressor, saves
  `checkpoint.npz`, and reloads it. This proves target/checkpoint plumbing; it
  does not improve a policy.
- Pong paddle-angle smoke confirmed the key mechanic: top, center, and bottom
  paddle contacts produce outgoing `ball_vy` values of `-1`, `0`, and `1`.
  The next useful Pong eval should measure deliberate off-center returns
  against `track_ball`.
- Pong angle-control probe now exists. The scripted `angle_control` policy made
  only off-center contacts on its own hits and beat random in both seats, but
  it still produced only truncations against `track_ball`. The next gap is
  choosing useful return sequences, not creating off-center hits.
- Audit note: `track_ball` itself makes many off-center hits because of
  movement lag. The next metric must be post-contact score pressure or short
  return, not off-center rate by itself.
- Pong contact-outcome probe now exists. It produces inspectable rows for top,
  center, and bottom contacts. In the first smoke, contact choice changed
  outgoing `ball_vy` but not score-delta return; all rows truncated against
  `track_ball`. This is observability, not eval progress.
- Pong eval reorientation: the main scoreboard is fixed baselines and
  checkpoint history. Use `random_uniform` and `track_ball` first, then compare
  older and newer checkpoints now that periodic policy checkpoints exist.
- Pong checkpoint scoreboard now exists. The first smoke used the same
  checkpoint as `latest` and `previous`, so it proves learned-vs-learned row
  plumbing but not policy progress.
- Pong distinct-checkpoint scoreboard compared the three existing learned policy
  attempts. All beat random more often than not, but none beat `track_ball`.
- Pong periodic policy checkpoints now exist for imitation training. The smoke
  wrote reloadable epoch checkpoints and a follow-up scoreboard run loaded
  epoch 1 and epoch 3. Epoch 3 did slightly better against random, but both
  checkpoints won 0/4 against `track_ball`, and the learned-vs-learned row
  tied. This proves old-vs-new eval plumbing, not quality.
- A longer Pong imitation run wrote epoch 250/500/750/1000 checkpoints. Epoch
  1000 was selected on the selection split because it beat random 42/64 and
  beat most earlier checkpoints, but it still won 0/64 against `track_ball`.
- Heldout for selected epoch 1000 is mixed: it beat previous 50/64 and lost
  less often to `track_ball`, but still won 0/64 against `track_ball`.
- Pong now has a CPU Modal scoreboard wrapper. A remote smoke read two Pong
  checkpoints from `curvyzero-runs`, wrote eval artifacts back to the Volume,
  and returned refs/hashes. This proves remote eval plumbing, not policy
  quality.
- Larger one-step angle-tie lookahead did not improve the scoreboard: it made
  442 non-`track_ball` labels, but all lookahead checkpoints won 0/64 against
  `track_ball`, and selection stayed with imitation epoch 1000.
- Loss-delay lookahead is implemented as a training-label-only option. The
  first alpha `0.05` smoke produced 251 rows but 0 labels different from
  `track_ball`; the trained checkpoint still won 0/32 against `track_ball`.
  Keep the option, but do not scale that exact setting.
- Depth-2 ego-sequence lookahead is implemented. The first strict smoke found
  10 avoided-loss rows, but all target first actions still matched `track_ball`,
  so it was not trained.
- Pong self-play replay and training now exist. First smoke wrote 498
  random-vs-random replay rows, trained epoch 25 and 50 policy/value
  checkpoints, and scored them. It proves the correct loop shape; quality is
  still weak, with 0 wins against `track_ball`.
- Pong self-play generation 2 ran from `selfplay50` with epsilon exploration.
  It produced more data and one random-baseline bump, but every child
  checkpoint lost to the parent and still won 0 games against `track_ball`.
- Current Pong focus is the critique decision: repair the crude self-play
  trainer or switch to a known simple baseline/curriculum. Do not add more
  generations, leagues, or Modal/GPU scale until a simple learner improves for
  an inspectable reason.
- Eval rules v0 require seed-set labels, planner-only baselines, both selected
  and latest checkpoints in the summary, and a separate check before making a
  learning claim.
- Tiny Line Duel eval summaries now expose `pair_groups`, split metadata, and
  seat deltas. Use paired groups when judging multiplayer results.
- Modal run helper functions now exist, and a dummy-survival train-attempt
  wrapper uses the run/attempt/checkpoint Volume layout. It has not proven
  resume. Modal is still the serious-run target; local remains the tiny debug
  path.
- Dummy survival degradation diagnosis found the failure mode: crash-heavy
  replay learns non-positive Q/dynamics, making lower-clearance or unknown
  next states look less negative than safer known routes. Later checkpoints
  increasingly override the planner-only safety action.
- Modal Volume storage works for dummy survival. The `curvyzero-runs` quick run
  writes summary/checkpoint/metrics files under a fixed path and returns file
  hashes/refs. It cannot resume training yet.
- Modal is the real execution target for serious train/eval jobs. Local runs
  are only for tiny debug and fast artifact checks.
- Serious MuZero is not implemented for project-owned Pong or CurvyTron.
- Current library plan: LightZero-first for the next real dummy Pong MuZero
  attempt. The next proof should run LightZero's own MuZero trainer on a
  CurvyZero-owned dummy Pong env as one Modal job.
- Mctx remains the fallback/search-library lane if LightZero cannot run the
  custom env cleanly, loses required Pong telemetry, or makes artifact
  ownership too awkward.
- Do not describe CEM, imitation, supervised MLP, NumPy self-play, or Mctx
  benchmarks as MuZero progress. They are baselines or plumbing only.

## Main Goal

Build the smallest complete training loop:

```text
self-play games -> replay/output files -> shaped targets -> trainer update
-> checkpoint -> evaluator -> summary
```

The first versions can be dumb. The point is to make the loop real, observable,
resumable, and portable to Modal.

## Working Hypotheses

- H1: A single-player dummy task should come before multiplayer because it
  lets us debug replay, training targets, model updates, checkpoints, and Modal
  files separately.
- H2: A two-player game should follow quickly because CurvyTron is ultimately
  simultaneous multiplayer and self-play bugs do not show up in single-agent
  tasks.
- H3: LightZero should be tried first because it is the only complete MuZero
  trainer already proven on Modal in this repo, but the proof must use our
  dummy Pong env, not stock CartPole.
- H4: Project-owned MuZero plus Mctx may fit CurvyTron better later, especially
  for simultaneous multiplayer. Do not pay that cost until LightZero fails a
  small custom-env smoke for a clear reason.
- H5: The first useful Modal job should run a whole toy train/eval job and
  return a small JSON summary plus optional output files.
- H6: For multiplayer, store one training row per player view. Store the full
  joint action separately. Opponents can be scripts or checkpoints. Avoid
  searching over all players' actions together until we need it.
- H7: Pong is now the best near-term toy for visual learning. Tiny Line Duel
  remains useful later because it is closer to CurvyTron movement and
  simultaneous collisions.
- H8: The current tabular dummy survival learner is not improved by simply
  increasing episodes. Keep it as a diagnostic for trainer/eval behavior, not
  as the main path.
- H9: Safety-aware planning is enough to create positive dummy survival monitor
  scores, but planner-only now matches that score and continued updates can
  degrade the final checkpoint. Best/latest visibility and heldout confirmation
  matter before scaling runs.
- H10: Dummy survival gets worse because the learned values are poorly scaled.
  All learned values are zero or negative, so unknown moves can look better
  than safer known moves. Candidate fixes: make unknown states pessimistic,
  adjust planner scores, or collect better training episodes.
- H11: Pong eval must include survival/loss-delay telemetry. A `0/N wins`
  result is not enough. Use wins, losses, truncation rate, mean/median/p90/std
  survival steps, score return stats, and shaped loss-delay return stats.

## Historical Candidate Tasks

This table is background only. It is not the active next-action list.

| Task | Why It Helps | Risk | Current Lean |
| --- | --- | --- | --- |
| Synthetic one-state bandit | Exercises policy/value/reward loss and checkpoints with almost no env complexity. | Too trivial for search/replay bugs. | Use only as model-update smoke. |
| Solo turning survival | Very small; exposes replay/checkpoint/eval problems quickly. | Not visual, not multiplayer, and the planner already solves the tiny eval. | Keep as diagnostic only. |
| Tiny deterministic gridworld | Exercises sequential decisions, replay, value targets, and evaluation. | Less CurvyTron-shaped. | Backup if survival toy is awkward. |
| Pong-like 1v1 toy | Two-player, familiar, has dense feedback, and now has a raster observation. | Less CurvyTron-like than line duel. | Make this the near-term visual training toy. |
| Tiny Tron/line survival 1v1 | Closest to CurvyTron and can reuse env concepts. | Needs two-phase collision/tie handling. | Keep as later multiplayer/CurvyTron-like check. |
| 3+ player toy arena | Tests multiplayer formulation. | Too much too early. | Later canary, not first loop. |

## Historical Experiment Queue

This queue is background only. The active next action is the Pong
repair-vs-baseline decision.

| ID | Experiment | Output | Status |
| --- | --- | --- | --- |
| E0 | Local toy baseline command | JSON summary and optional local files | Exists; needs interpretation |
| E1 | Modal toy runner | Remote summary dict, optional file refs | Exists for dummy survival and Tiny Line Duel; survival Volume quick run exists |
| E2 | Dummy single-player training loop | Checkpoint, replay-like records, eval summary | Exists; keep as diagnostic |
| E3 | Synthetic Mctx benchmark | compile time, steady-state search throughput | Checklist exists in `docs/working/mctx_spike_checklist.md` |
| E4 | LightZero ego-wrapper spike | install/import and wrapper-overhead notes | Pending |
| E5 | Tiny Line Duel two-player toy | one replay row per player view and shared dummy self-play | Exists; code path only |
| E6 | 3+ player formulation canary | rank payoff and opponent-policy metadata | Later |
| E7 | Staged training evals | fixed-seed baseline/checkpoint tables | EVAL1/EVAL2 plus learned-checkpoint eval exist |
| E8 | Pong visual learner decision | Self-play replay, shaped score-plus-longevity target, tiny policy/value update, generation-2 manual loop, and checkpoint scoreboard exist locally. Gen2 failed: it lost to the parent and won 0 games against `track_ball`. Active focus is deciding repair vs a simpler known baseline/curriculum, with fixed-baseline evals first. | Active focus |

## MuZero Questions

- What minimal model shapes do we use for the dummy loop: MLP first, CNN later?
- Do we implement a tiny project-owned MuZero-shaped training loop before
  installing Mctx, or install Mctx first and make search real immediately?
- Which target-construction tests can be validated with hand-authored replay
  records without running the whole env?
- How do we represent multiplayer in replay: one row per ego player, joint
  action stored separately, opponent policy ids attached?
- What exactly is the rejection gate for LightZero?
- What is the smallest checkpoint resume API for continuing training, distinct
  from the now-working checkpoint load path for evaluation?

Working answer:

- First local loop can use dummy policy targets or shallow one-step planning
  before Mctx. It should still use MuZero-shaped functions:
  `representation`, `dynamics`, `prediction`, replay rows, target builder,
  checkpoint, evaluator.
- Mctx enters when we need real batched search. Do not make the first training
  loop wait on JAX/CUDA installation.
- LightZero is a later comparison once our own loop shape is clear.

## Environment Interface

Training should depend on a small game API, not on the reconstruction internals.
The current environment is close to a multi-agent step API, but the training
code still needs one clear boundary.

Working recommendation: create a project-owned game wrapper, tentatively
`CurvyZeroTrainingEnv`. Gymnasium, PettingZoo, LightZero, and Mctx should adapt
to this wrapper; none of them should define the core game API.

Candidate API:

```text
reset(seed, config) -> obs_by_agent, env_state_ref/meta
step(joint_action_by_agent) -> obs_by_agent, reward_by_agent,
                               terminated_by_agent, truncated_by_agent,
                               info_by_agent
observe(env_state, ego_agent, schema_id) -> observation, action_mask, metadata
replay metadata -> rules_hash, observation_schema_hash, reward_schema_hash,
                   action_schema_hash, player_count, ego_agent, joint_action
```

Decision-step meaning:

- `step` means one trainer decision step. It may hold the same action for
  several elapsed-ms source frames and stop early if the game ends.
- Require complete joint actions for live agents. Avoid silent missing-action
  defaults in training.
- Store the whole joint action even when search controls only one player's
  action.
- Dead-player no-ops are wrapper details and should not produce policy targets.
- Keep `terminated` separate from `truncated`; a timeout is not automatically a
  draw.

Required step metadata:

- `tick_before`, `tick_after`, `physics_ticks_elapsed`
- `terminal_reason`, `truncation_reason`
- `death_tick_by_player`, `death_cause_by_player`
- `collision_partner_by_player` or equivalent when known
- `tie_group_by_player`
- `rank_score_by_player` or `outcome_by_player`
- `terminal_observation_by_agent` for autoreset wrappers

Required version fields before replay enters training:

- `rules_hash`
- `action_schema_id/hash`
- `observation_schema_id/hash`
- `reward_schema_id/hash`

Wrapper decision cadence assumption:

- CurvyTron source movement is continuous over elapsed time, but the training
  API can still expose fixed decision steps as a wrapper abstraction.
- A decision step applies a complete wrapper joint action/control snapshot on a
  chosen cadence, usually by holding controls across one or more source frames.
- All live-player wrapper choices are fixed at the same decision boundary;
  movement, collision, death, and scoring are then resolved by the source-derived
  update order.
- If source reconstruction later proves a serious mismatch, adapt the wrapper
  or ruleset. Do not let this block the current dummy training work.

Open interface questions:

- Does `step` represent one physics tick, one decision with action repeat, or
  both as separate APIs?
- How are simultaneous actions, same-tick deaths, ties, and truncations
  represented without seat-order leakage?
- Where do action masks live: env core, observation adapter, or wrapper?
- Can the same API support source-fidelity scenarios, toy training
  rulesets, and future vectorized batches?
- What schema hashes must be mandatory before any replay enters training?
- Does reconstruction expose enough terminal metadata for value/reward targets:
  death tick, death cause, rank score, tie group, timeout?

Todo:

- Move this API recommendation into a design doc once E2 starts using
  a real wrapper.
- Move the accepted API into `docs/design/training_architecture.md` or
  a new design doc.
- Keep training loop code behind this wrapper so environment reconstruction can
  improve without rewriting trainer/replay/search.

## Active Next Actions

- Keep this agenda current.
- Keep dummy survival, Tiny Line Duel, and Pong labeled as small test tasks, not
  real MuZero quality claims.
- Finish the Pong critique decision: repair the crude self-play trainer or
  switch to a simpler known baseline/curriculum.
- Treat the Pong self-play plan as a hypothesis under critique. Treat old Pong
  imitation replay as completed raster plumbing, not the current objective.
- Keep local runs tiny. Put serious train/eval attempts on Modal as whole jobs
  once the chosen learner path has a reason to run.
- Use survival only when it helps debug trainer/eval behavior.
- Keep Tiny Line Duel interpreted correctly: it proves simultaneous-step/replay
  code works, while the first all-draw eval is not learning progress.
- Use small recorded eval waves as the immediate checks: survival learned
  checkpoints must beat random and approach one-step-safe; Line Duel learned
  checkpoints must beat random/sticky without seat bias before any league
  or Elo work. Sample a fresh pseudo-random eval seed list per wave and record
  the generator seed/list.
- Treat the best periodic checkpoint, not the final checkpoint, as the current
  local dummy survival checkpoint to inspect.
- Keep Modal wrappers as whole-job runners that write files. Add Volume storage
  only with explicit run ids, attempt ids, and resume behavior beyond the
  current quick run.
- Keep the environment API notes tracked, but do not open new design branches
  until code needs them.
- If repairing self-play, fix the likely `policy_grad = probs` aliasing bug,
  then add action-diversity/entropy visibility and normalize advantages more
  carefully.
- Use the Pong checkpoint scoreboard as the main eval surface: `random_uniform`
  and `track_ball` first, latest-vs-old second.
- Use the CPU Modal Pong scoreboard wrapper for remote eval of already-written
  checkpoints. Do not make local paths the durable path for serious eval.
- Require run-health metrics for new Pong runs: iteration metrics, action
  histograms by seat, entropy/collapse metrics, terminal causes, and failure
  examples.
- Treat angle-control and contact-outcome probes as observability. They explain
  failures; they are not the scoreboard.
- Keep eval score-delta-only. Use the new shaped episode return as a training
  target only, and watch for timeout farming.
- If contact choices all return zero against `track_ball`, adjust geometry only
  when it helps the scoreboard question.
- Keep the value-target smoke as plumbing for later policy/search work. Do not
  treat its weak value fit as policy progress.
- Keep expert-only rows for policy copying; do not copy random action rows as
  if they were expert behavior.
- Refine output-file layout only when the current summary/checkpoint/JSONL
  layout blocks a Modal/resume/eval need.
- Add checkpoint resume/continue-training path for dummy survival and Tiny Line
  Duel only if it helps produce better learned checkpoints or real Modal
  persistence beyond the current Volume quick run.
- Inspect the dummy learner/planner before scaling runs: saved final
  checkpoints are not showing monotonic progress.
- Use `scripts/analyze_dummy_survival_checkpoints.py` only when survival changes
  are directly helping the trainer plan.
- Add a best-checkpoint convention before larger local or Modal dummy survival
  runs.
- Keep `docs/design/training_architecture.md` aligned as the training code
  becomes less toy.
- Run only training quick-run commands, not broad pytest, unless redirected.
- Improve the dummy loop only when it helps the next API: checkpoint
  resume, model-load path, or cleaner target naming.

## Historical Later Ideas

These are not active next actions.

- Run Mctx synthetic benchmark on Modal GPU.
- Compare LightZero only after the project-owned dummy loop is understandable.
- Add staged eval checks: random/scripted baselines first, then fixed old
  checkpoints, then small policy-pool ratings only if win rates become noisy.
- Harden Tiny Line Duel two-player self-play after Pong has a basic training
  path.
- Add 3+ player tiny checks for rank payoff and replay metadata.
- Add checkpoint-pool opponents and searched-ego-only MCTS.

## Organization Rules

- Working state lives here and in `docs/working/training_coach_packet.md`.
- Research notes live under `docs/research/`.
- Stable decisions live under `docs/decisions/`.
- Completed runs get dated logs under `docs/experiments/`.
- Avoid bloated docs. If a section stops helping the next training move, delete
  or promote it.
