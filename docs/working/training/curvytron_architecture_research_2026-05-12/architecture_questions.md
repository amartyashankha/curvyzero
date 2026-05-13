# Architecture Questions

Purpose: keep broad questions visible while subagents and future runs fill in
details.

## Stock LightZero Loop

- What exact object creates `GameSegment` rows?
- What fields are stored per transition: action, reward, observation,
  `to_play`, root value, visit distribution, done?
- Where does `MuZeroGameBuffer.sample(...)` build value/reward/policy targets?
- Which parts run on GPU: initial/recurrent model inference, learner forward,
  search tensors?
- Which parts remain CPU-side: env step, rendering, tree orchestration,
  replay sampling, checkpoint/eval plumbing?
- GPU frozen-opponent caution: if learner CUDA also sets
  `opponent_use_cuda=true`, then a subprocess env manager can make env workers
  touch CUDA in forked processes. The clean next GPU canary should use
  `env_manager_type=base` with one env, or add a flag that keeps the frozen
  opponent on CPU while the learner uses GPU.

## CurvyTron Compatibility

- For a stock env row, should CurvyTron use `to_play=-1` unless intentionally
  using board-game semantics?
- Can `source_state_joint_action` use a reward that tests real learning instead
  of only "both alive"?
- Can fixed/frozen opponents be used as a practical curriculum or league
  control without pretending they are live same-policy self-play?
- What would a native-compatible two-seat `GameSegment` look like for each seat
  perspective?

## Failure Analysis

- Did the custom two-seat target builder match LightZero target semantics?
- Did pure same-policy play create too little terminal signal because both
  seats were symmetric?
- Did reward scale, replay age, or `to_play` semantics make learning harder?
- Did the custom two-seat adapter pass public player ids `0/1` as `to_play`
  where non-board-game LightZero expects a neutral `-1`?
- Which stale docs or defaults caused us to scale the wrong path?

## Observability

- Every run should report whether it called `train_muzero`.
- Every run should report whether it used native GameBuffer targets.
- Every run should report survival curve, terminal outcome rate, reward
  components, action distribution, reset randomness, and opponent source.
