# 2026-05-09 LightZero Dummy Pong Checkpoint Probe

Worker P added and ran the smallest Modal wrapper around
`probe_lightzero_dummy_pong_checkpoint(...)` against the real mirrored
LightZero checkpoint.

## Code Added

```text
src/curvyzero/infra/modal/lightzero_dummy_pong_checkpoint_probe.py
```

The wrapper uses the LightZero 0.2.0 Modal image, copies local `src/` to
`/repo/src`, mounts Modal Volume `curvyzero-runs` at `/runs`, calls the existing
training probe, returns JSON, and writes the JSON artifact beside the source
training attempt.

## Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_checkpoint_probe
```

Modal app:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-wqQDxGCrSBVmN9ZtI3nXvC
```

Checkpoint ref:

```text
training/lightzero-dummy-pong/lz-dpong-20260509T141607Z-3696aa333028/checkpoints/lightzero/ckpt_best.pth.tar
```

Output artifact:

```text
training/lightzero-dummy-pong/lz-dpong-20260509T141607Z-3696aa333028/attempts/attempt-20260509T141607Z-98662e4917b4/probe/lightzero_checkpoint_probe_20260509T143137Z.json
```

Artifact sha256:

```text
c22b90462743128944cd93991f64c5e7f5a4bbaebc73c595aa24932b3866dd27
```

## Result

Top-level probe status:

```text
ok: true
load_ok: true
state_dict_ok: true
action_probe_ok: true
strict_full_model_load_ok: false
direct_policy_head_possible: true
```

Checkpoint summary:

```text
bytes: 27829507
sha256: 1797dd1b496b60062d39241df4247068b68de321ad03e2456f448d62a071b520
state_dict.path: model
state_dict.tensor_count: 108
```

Direct policy-head action probe:

```text
model_class: lzero.model.muzero_model_mlp.MuZeroModelMLP
observation_shape: 10
action_labels: [up, stay, down]
policy_logits: [0.0, 0.0, 0.0]
action_id: 0
action_label: up
```

The action is a tie-broken argmax over all-zero logits, so this is a loader/API
signal rather than an evaluation-quality signal.

## Strict Load Mismatch

`model.load_state_dict(..., strict=True)` did not pass. The probe fell back to
`strict=False` with the original `model` state dict. Missing keys:

```text
dynamics_network.fc_dynamics.0.weight
dynamics_network.fc_dynamics.0.bias
dynamics_network.fc_dynamics.1.weight
dynamics_network.fc_dynamics.1.bias
dynamics_network.fc_dynamics.1.running_mean
dynamics_network.fc_dynamics.1.running_var
dynamics_network.fc_dynamics.3.weight
dynamics_network.fc_dynamics.3.bias
dynamics_network.fc_dynamics.4.weight
dynamics_network.fc_dynamics.4.bias
dynamics_network.fc_dynamics.4.running_mean
dynamics_network.fc_dynamics.4.running_var
```

Unexpected keys:

```text
dynamics_network.fc_dynamics_1.0.weight
dynamics_network.fc_dynamics_1.0.bias
dynamics_network.fc_dynamics_1.1.weight
dynamics_network.fc_dynamics_1.1.bias
dynamics_network.fc_dynamics_1.1.running_mean
dynamics_network.fc_dynamics_1.1.running_var
dynamics_network.fc_dynamics_1.1.num_batches_tracked
dynamics_network.fc_dynamics_1.3.weight
dynamics_network.fc_dynamics_1.3.bias
dynamics_network.fc_dynamics_1.4.weight
dynamics_network.fc_dynamics_1.4.bias
dynamics_network.fc_dynamics_1.4.running_mean
dynamics_network.fc_dynamics_1.4.running_var
dynamics_network.fc_dynamics_1.4.num_batches_tracked
dynamics_network.fc_dynamics_2.0.weight
dynamics_network.fc_dynamics_2.0.bias
dynamics_network.fc_dynamics_2.1.weight
dynamics_network.fc_dynamics_2.1.bias
dynamics_network.fc_dynamics_2.1.running_mean
dynamics_network.fc_dynamics_2.1.running_var
dynamics_network.fc_dynamics_2.1.num_batches_tracked
dynamics_network.fc_dynamics_2.3.weight
dynamics_network.fc_dynamics_2.3.bias
dynamics_network.fc_dynamics_2.4.weight
dynamics_network.fc_dynamics_2.4.bias
dynamics_network.fc_dynamics_2.4.running_mean
dynamics_network.fc_dynamics_2.4.running_var
dynamics_network.fc_dynamics_2.4.num_batches_tracked
```

## Eval Boundary

Direct policy-head evaluation is technically possible for this checkpoint
because `initial_inference` uses the representation and prediction networks,
and the observed mismatch is dynamics-network-only. The exact implementation is
to load `checkpoint["model"]` into `MuZeroModelMLP(**model_cfg)` with
`strict=False`, assert all missing/unexpected keys start with
`dynamics_network.`, then run:

```python
with torch.no_grad():
    output = model.initial_inference(torch.tensor([tabular_ego_row], dtype=torch.float32))
action_id = int(torch.argmax(output.policy_logits, dim=-1).item())
```

That should be wired as a direct-policy checkpoint loader for the first
CurvyZero dummy Pong scoreboard integration only if policy-head argmax is an
accepted eval behavior.

Full LightZero-style MuZero/MCTS eval is not ready from this probe. The next
full-eval implementation is to resolve the dynamics-network config/key shape
so strict load passes, then instantiate `lzero.policy.muzero.MuZeroPolicy` from
the patched config and call its `eval_mode.forward(...)` path with the same
tabular observation and action mask.
