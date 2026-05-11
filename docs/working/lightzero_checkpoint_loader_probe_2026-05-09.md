# LightZero Checkpoint Loader Probe - 2026-05-09

Worker lane: implementation prep for the LightZero dummy Pong checkpoint
loading boundary. This note does not edit Modal artifact inspection output and
does not claim an independent scoreboard is ready.

## Current Status Update

The strict/MCTS loader boundary is no longer blocked for the 512/8
`iteration_8` checkpoint. The full MCTS/eval-mode scorecard now runs, so this
note is loader history plus the remaining investigation items.

Passing MCTS loader smoke:

```text
training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/attempts/attempt-20260509T144635Z-ece79bad80d0/probe/lightzero_mcts_loader_smoke_20260509T145607Z.json
```

Recorded details:

```text
ok: true
mcts_eval_status: ok
strict_full_model_load_ok: true
strict_full_model_load_variant: res_connection_in_dynamics_true
call_shape.data: [1,10]
call_shape.action_mask: [[1,1,1]]
call_shape.to_play: [-1]
call_shape.ready_env_id: [0]
action: 0
visit_count_distributions: [2,1,1]
predicted_policy_logits: [0.0170983, 0.00644484, 0.0132326]
predicted_value: about 0.0000259
searched_value: about 0.000114
```

Correction: earlier docs saying the MCTS loader failed on missing
`cfg.policy.device` are stale. The device/action-mask issues were fixed. The
full MCTS/eval-mode scorecard has now run across episodes/opponents, while the
direct policy-head greedy scoreboard remains a non-MCTS control path.

Full MCTS/eval-mode scorecard:

```text
modal_url: https://modal.com/apps/modal-labs/shankha-dev/ap-Ou59sqrdljB295FFBpyIUP
summary: eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-20260509T150000Z-20260509T150243Z/summary.json
episodes: eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-20260509T150000Z-20260509T150243Z/episodes.jsonl
checkpoint: 512/8 iteration_8.pth.tar
episodes_per_seating: 16
num_simulations: 8
strict_full_model_load_variant: res_connection_in_dynamics_true
```

Rows:

```text
lightzero_iter8_vs_lagged_track_ball_1:
  LZ wins 13, opponent wins 15, mean survival 25.09,
  LZ shaped -0.0397, LZ reward -0.0625, LZ actions [801,2,0]
lightzero_iter8_vs_random_uniform:
  LZ wins 17, opponent wins 15, mean survival 13.84,
  LZ shaped 0.0953, LZ reward 0.0625, LZ actions [443,0,0]
lightzero_iter8_vs_track_ball:
  LZ wins 0, opponent wins 30, mean survival 25.66,
  LZ shaped -0.8618, LZ reward -0.9375, LZ actions [816,5,0]
```

Read: MCTS eval-mode is no longer just a loader smoke. Full episode eval works.
But checkpoint behavior is still effectively up-only: combined LightZero action
hist `[2060,7,0]`; it never chose down in this scorecard. The next blocker is
policy quality/training signal, not checkpoint loading.

The direct policy-head path was rerun with strict config after the same fix:

```text
eval_id: policy-head-scoreboard-512x8-strictcfg
modal_url: https://modal.com/apps/modal-labs/shankha-dev/ap-Q7sPmscebJQWisowuweBxV
summary: eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-strictcfg-20260509T145841Z/summary.json
episodes: eval/lightzero-dummy-pong/policy-head-scoreboard-512x8-strictcfg-20260509T145841Z/episodes.jsonl
```

`load_state_dict` is strict true through the direct policy-head path, confirming
the split residual dynamics config fix there too. The behavior remains
constant-up: lagged `[590,0,0]`, random `[388,0,0]`, track `[968,0,0]`.
This is useful loader evidence, not policy-quality or MCTS evidence.

## Implemented

Added:

```text
src/curvyzero/training/lightzero_dummy_pong_checkpoint_probe.py
```

The script is intentionally narrow:

- loads one LightZero `.pth.tar` with `torch.load(..., map_location="cpu")`;
- records checkpoint path, byte size, and sha256;
- summarizes top-level checkpoint keys and tensor-like values;
- finds a likely model state dict under common keys such as `model`,
  `state_dict`, `model_state_dict`, `network`, `target_model`, or
  `policy.model`;
- reconstructs the trainer's tabular dummy Pong model config via
  `patched_dummy_pong_configs(...)`;
- tries direct network inference with
  `lzero.model.muzero_model_mlp.MuZeroModelMLP(**model_cfg)`;
- selects one action by argmax over
  `model.initial_inference(obs_tensor).policy_logits`.

The probe only supports the first trained surface:

```text
feature_mode=tabular_ego
observation_shape=10
action_space_size=3
model_type=mlp
actions: 0=up, 1=stay, 2=down
```

## Exact API Boundary

The smallest direct network path is:

```python
from lzero.model.muzero_model_mlp import MuZeroModelMLP

model = MuZeroModelMLP(**model_cfg_without_model_type)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

obs_tensor = torch.tensor([tabular_ego_row], dtype=torch.float32)
with torch.no_grad():
    output = model.initial_inference(obs_tensor)
action_id = int(torch.argmax(output.policy_logits, dim=-1).item())
```

This bypasses LightZero MCTS and `MuZeroPolicy`; it is useful only to prove
that checkpoint tensors can be loaded into the expected model and produce a
valid policy-logit action. The future scoreboard should either:

- use the direct-network action if policy-head argmax is accepted as the eval
  behavior for the first checkpoint read; or
- use `lzero.policy.muzero.MuZeroPolicy(...).eval_mode.forward(...)` with the
  compiled config and MCTS settings if the scorecard must match LightZero eval
  behavior.

The policy/MCTS path now has a concrete one-call smoke for 512/8
`iteration_8`; what remains is scorecard-scale evaluation.

## Command To Run With A Real Checkpoint

Run inside the LightZero image or any environment with LightZero 0.2.0,
DI-engine, torch, and this repo on `PYTHONPATH`. The module form is preferred
because the existing Modal LightZero image copies `src/` into `/repo`:

```bash
python -m curvyzero.training.lightzero_dummy_pong_checkpoint_probe \
  /runs/training/lightzero-dummy-pong/lz-dpong-20260509T141607Z-3696aa333028/checkpoints/lightzero/ckpt_best.pth.tar \
  --feature-mode tabular_ego \
  --env dummy_pong_lag1 \
  --opponent-policy random_uniform \
  --seed 0 \
  --max-env-step 64
```

Expected success signal:

```text
load.ok=true
state_dict.ok=true
action_probe.ok=true
action_probe.action_id in {0,1,2}
```

If `action_probe.ok=false` but `state_dict.ok=true`, the next fix is to inspect
`state_dict.path`, `keys_sample`, and the `load_state_dict` error to adjust the
checkpoint key prefix or model config.

## Not Implemented

- No local checkpoint was present to execute the action probe in this lane.
- No independent dummy Pong scoreboard integration was added.
- No LightZero `.pth.tar` to CurvyZero-owned exported policy artifact was
  written.
- No full `MuZeroPolicy.eval_mode.forward(...)` / MCTS episode scorecard has
  been run across opponents.
- No pytest was run.

## Next Step

Fix the tiny env/helper footguns, then rerun the MCTS/eval-mode dummy Pong
scorecard with explicit matching `max_env_step` before deciding whether longer
training is worth running. Keep the existing direct-action policy-head
scoreboard as a sibling negative/control path, not as proof of LightZero MCTS
behavior.

Bug-sweep items to carry forward, not proven root causes:

- Direct policy-head argmax can collapse weak/tied logits to action `0`, but
  MCTS also mostly chooses up, so this only explains the control path.
- Horizon/config mismatch risk remains: training used `max_env_step=512`,
  checkpoint scoring can default to `64`, and independent eval currently uses
  `PongConfig()` default `max_steps=120`.
- `DummyPongLightZeroEnv.random_action()` reseeds every call; repeated helper
  calls in one episode can return the same action if this path is used.
- `timestep` compatibility is wrapper-local, not in the base env observation.

## Worker Q API Recommendation

For the first independent scoreboard, use direct
`MuZeroModelMLP.initial_inference(...)` plus greedy argmax over
`policy_logits`, and label it explicitly as:

```text
LightZero checkpoint policy-head greedy scoreboard, no MCTS
```

That path is simpler and honest because it tests only the boundary the repo has
already isolated: checkpoint bytes -> exact MLP config -> strict tensor load ->
one deterministic action. It does not pretend to match LightZero evaluator
behavior, MCTS visit counts, root value search, or LightZero's policy wrappers.

The exact direct-network inputs are:

```python
patched = patched_dummy_pong_configs(
    env="dummy_pong_lag1",
    feature_mode="tabular_ego",
    opponent_policy="random_uniform",
    seed=0,
    max_env_step=64,
    collector_env_num=1,
    evaluator_env_num=1,
    n_evaluator_episode=1,
    num_simulations=2,
    batch_size=8,
    update_per_collect=1,
)
model_cfg = dict(patched["main_config"]["policy"]["model"])
assert model_cfg.pop("model_type") == "mlp"

model = MuZeroModelMLP(**model_cfg)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

obs = torch.tensor([tabular_ego_row], dtype=torch.float32)
with torch.no_grad():
    output = model.initial_inference(obs)
action = int(torch.argmax(output.policy_logits, dim=-1).item())
```

The LightZero-faithful eval path should be a separate follow-up smoke. Upstream
`MuZeroPolicy._forward_eval` runs `initial_inference`, derives legal actions
from `action_mask`, prepares MCTS roots without root noise, runs search, and
chooses the deterministic best visit-count action. The expected standalone
shape is:

```python
from ding.config import compile_config
from ding.policy import create_policy

cfg = patched["main_config"]
create_cfg = patched["create_config"]
cfg["policy"]["cuda"] = False
cfg["policy"]["device"] = "cpu"
cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create_cfg, save_cfg=False)

policy = create_policy(cfg.policy, model=None, enable_field=["learn", "collect", "eval"])
policy.learn_mode.load_state_dict(torch.load(checkpoint_path, map_location=cfg.policy.device))

obs = torch.tensor([tabular_ego_row], dtype=torch.float32)
output = policy.eval_mode.forward(
    obs,
    action_mask=[np.ones(3, dtype=np.int8)],
    to_play=[-1],
    ready_env_id=np.array([0]),
)
action = int(output[0]["action"])
```

Use the policy/MCTS path when the scoreboard claim is "matches LightZero eval
action selection." Use the direct path first when the claim is "independently
loads and runs a mirrored checkpoint in CurvyZero-owned scoring code."
