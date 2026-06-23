"""Profile-only LightZero PyTorch to JAX shadow-model parity helpers.

This module is a bridge proof for the optimizer lane.  It does not run
LightZero training, does not call MCTX, and does not change trainer defaults.
Its job is narrower: take the current CurvyTron LightZero MuZero model weights
and check whether a JAX implementation can reproduce raw model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LIGHTZERO_JAX_SHADOW_PARITY_IMPL = (
    "curvyzero_lightzero_jax_shadow_model_parity/v0"
)
LIGHTZERO_JAX_SHADOW_PARITY_SEMANTICS = (
    "profile_only_lightzero_pytorch_to_jax_raw_model_parity_not_mctx_not_train"
)
MUTABLE_CHECKPOINT_BASENAMES = frozenset({"latest.pth.tar", "ckpt_best.pth.tar"})
IGNORED_INFERENCE_PREFIXES = ("projection.", "prediction_head.")

PROFILE_ONLY_LABELS: dict[str, bool | str] = {
    "profile_only": True,
    "not_train_muzero": True,
    "not_mctx": True,
    "touches_live_runs": False,
    "trainer_defaults_changed": False,
    "impl": LIGHTZERO_JAX_SHADOW_PARITY_IMPL,
    "semantics": LIGHTZERO_JAX_SHADOW_PARITY_SEMANTICS,
}


@dataclass(frozen=True)
class ComparisonResult:
    """Small numeric comparison record for one output tensor."""

    name: str
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    max_abs: float | None
    max_rel: float | None
    allclose: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape_a": list(self.shape_a),
            "shape_b": list(self.shape_b),
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "allclose": self.allclose,
        }


def is_mutable_checkpoint_ref(ref: str | Path) -> bool:
    """Return true for checkpoint refs that can silently change under us."""

    name = Path(str(ref)).name
    return name in MUTABLE_CHECKPOINT_BASENAMES


def require_immutable_checkpoint_ref(ref: str | Path) -> str:
    """Reject mutable checkpoint refs for reproducible parity reports."""

    text = str(ref)
    if is_mutable_checkpoint_ref(text):
        raise ValueError(
            f"checkpoint ref {text!r} is mutable; use an immutable iteration_N "
            "checkpoint for parity"
        )
    return text


def compare_arrays(
    name: str,
    a: Any,
    b: Any,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> ComparisonResult:
    """Compare two array-like values and report the largest observed error."""

    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if arr_a.shape != arr_b.shape:
        return ComparisonResult(
            name=name,
            shape_a=tuple(int(x) for x in arr_a.shape),
            shape_b=tuple(int(x) for x in arr_b.shape),
            max_abs=None,
            max_rel=None,
            allclose=False,
        )
    diff = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))
    denom = np.maximum(np.abs(arr_b.astype(np.float64)), 1e-12)
    rel = diff / denom
    return ComparisonResult(
        name=name,
        shape_a=tuple(int(x) for x in arr_a.shape),
        shape_b=tuple(int(x) for x in arr_b.shape),
        max_abs=float(diff.max(initial=0.0)),
        max_rel=float(rel.max(initial=0.0)),
        allclose=bool(np.allclose(arr_a, arr_b, atol=atol, rtol=rtol)),
    )


def inverse_scalar_transform_logits(
    logits: Any,
    *,
    support_scale: int,
    epsilon: float = 0.001,
) -> np.ndarray:
    """Convert LightZero categorical value/reward logits to scalar values."""

    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("categorical logits must have shape [B, support_size]")
    scale = int(support_scale)
    expected_size = scale * 2 + 1
    if arr.shape[1] != expected_size:
        raise ValueError(
            f"support size mismatch: logits width {arr.shape[1]} != {expected_size}"
        )
    shifted = arr - arr.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=1, keepdims=True)
    support = np.arange(-scale, scale + 1, dtype=np.float64).reshape(1, -1)
    value = (support * probs).sum(axis=1, keepdims=True)
    tmp = (
        np.sqrt(1.0 + 4.0 * epsilon * (np.abs(value) + 1.0 + epsilon)) - 1.0
    ) / (2.0 * epsilon)
    return np.sign(value) * (tmp * tmp - 1.0)


def state_dict_to_numpy(state_dict: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Copy tensor-like state dict values to NumPy arrays."""

    out: dict[str, np.ndarray] = {}
    for key, value in state_dict.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        out[str(key)] = np.asarray(value)
    return out


def inference_required_state_keys(state_dict: Mapping[str, Any]) -> list[str]:
    """State keys needed for initial/recurrent MuZero inference."""

    keys: list[str] = []
    for key in state_dict:
        key_text = str(key)
        if key_text.endswith(".num_batches_tracked"):
            continue
        if key_text.startswith(IGNORED_INFERENCE_PREFIXES):
            continue
        keys.append(key_text)
    return sorted(keys)


def summarize_state_dict_coverage(
    state_dict: Mapping[str, Any],
    consumed_keys: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Describe whether the JAX shadow consumed all inference weights."""

    required = set(inference_required_state_keys(state_dict))
    consumed = set(consumed_keys)
    ignored = sorted(
        str(key)
        for key in state_dict
        if str(key).endswith(".num_batches_tracked")
        or str(key).startswith(IGNORED_INFERENCE_PREFIXES)
    )
    missing = sorted(required - consumed)
    extra = sorted(consumed - set(str(key) for key in state_dict))
    return {
        "required_key_count": len(required),
        "consumed_key_count": len(consumed & required),
        "ignored_key_count": len(ignored),
        "missing_required_keys": missing,
        "extra_consumed_keys": extra,
        "ignored_keys": ignored,
        "ok": not missing and not extra,
    }


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a local checkpoint for an immutable parity report."""

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


class JaxMuZeroShadowModel:
    """JAX copy of the current CurvyTron LightZero MuZero conv model.

    The implementation intentionally supports the model surface we actually use:
    `[4,64,64]` observations, downsampled conv representation, BatchNorm in eval
    mode, one residual block per section, one-hot actions, and one hidden layer
    in each MLP head.  It still discovers block counts from the state dict so
    support-size changes do not break the harness.
    """

    def __init__(self, params: Mapping[str, Any], *, jax: Any, jnp: Any):
        self.params = state_dict_to_numpy(params)
        self.jax = jax
        self.jnp = jnp
        self.consumed_keys: set[str] = set()
        self.action_space_size = int(self._p("prediction_network.fc_policy.3.weight").shape[0])
        self.reward_support_size = int(
            self._p("dynamics_network.fc_reward_head.3.weight").shape[0]
        )
        self.value_support_size = int(
            self._p("prediction_network.fc_value.3.weight").shape[0]
        )

    def initial_inference(self, obs: Any) -> dict[str, Any]:
        x = self.representation(obs)
        policy_logits, value = self.prediction(x)
        reward = self.jnp.zeros((x.shape[0],), dtype=self.jnp.float32)
        return {
            "value": value,
            "reward": reward,
            "policy_logits": policy_logits,
            "latent_state": x,
        }

    def recurrent_inference(self, latent_state: Any, action: Any) -> dict[str, Any]:
        next_latent_state, reward = self.dynamics(latent_state, action)
        policy_logits, value = self.prediction(next_latent_state)
        return {
            "value": value,
            "reward": reward,
            "policy_logits": policy_logits,
            "latent_state": next_latent_state,
        }

    def representation(self, obs: Any) -> Any:
        x = self._as_jnp(obs)
        x = self._conv2d(x, "representation_network.downsample_net.conv1", stride=2, padding=1)
        x = self._batch_norm_2d(x, "representation_network.downsample_net.norm1")
        x = self.jax.nn.relu(x)
        for prefix in self._indexed_prefixes("representation_network.downsample_net.resblocks1"):
            x = self._resblock_basic(x, prefix)
        x = self._resblock_downsample(
            x, "representation_network.downsample_net.downsample_block"
        )
        for prefix in self._indexed_prefixes("representation_network.downsample_net.resblocks2"):
            x = self._resblock_basic(x, prefix)
        x = self._avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        for prefix in self._indexed_prefixes("representation_network.downsample_net.resblocks3"):
            x = self._resblock_basic(x, prefix)
        for prefix in self._indexed_prefixes("representation_network.resblocks"):
            x = self._resblock_basic(x, prefix)
        return x

    def dynamics(self, latent_state: Any, action: Any) -> tuple[Any, Any]:
        latent = self._as_jnp(latent_state)
        action_arr = self._as_jnp(action).astype(self.jnp.int32).reshape((-1,))
        one_hot = self.jax.nn.one_hot(action_arr, self.action_space_size)
        action_encoding = self.jnp.broadcast_to(
            one_hot[:, :, None, None],
            (
                latent.shape[0],
                self.action_space_size,
                latent.shape[2],
                latent.shape[3],
            ),
        )
        state_action = self.jnp.concatenate([latent, action_encoding], axis=1)
        x = self._conv2d(state_action, "dynamics_network.conv", stride=1, padding=1)
        x = self._batch_norm_2d(x, "dynamics_network.norm_common")
        x = self.jax.nn.relu(x + latent)
        for prefix in self._indexed_prefixes("dynamics_network.resblocks"):
            x = self._resblock_basic(x, prefix)
        next_latent = x
        reward = self._conv2d(x, "dynamics_network.conv1x1_reward", stride=1, padding=0)
        reward = self._batch_norm_2d(reward, "dynamics_network.norm_reward")
        reward = self.jax.nn.relu(reward)
        reward = reward.reshape((reward.shape[0], -1))
        reward = self._mlp_v2(reward, "dynamics_network.fc_reward_head")
        return next_latent, reward

    def prediction(self, latent_state: Any) -> tuple[Any, Any]:
        x = self._as_jnp(latent_state)
        for prefix in self._indexed_prefixes("prediction_network.resblocks"):
            x = self._resblock_basic(x, prefix)
        value = self._conv2d(x, "prediction_network.conv1x1_value", stride=1, padding=0)
        value = self._batch_norm_2d(value, "prediction_network.norm_value")
        value = self.jax.nn.relu(value)
        policy = self._conv2d(x, "prediction_network.conv1x1_policy", stride=1, padding=0)
        policy = self._batch_norm_2d(policy, "prediction_network.norm_policy")
        policy = self.jax.nn.relu(policy)
        value = value.reshape((value.shape[0], -1))
        policy = policy.reshape((policy.shape[0], -1))
        value = self._mlp_v2(value, "prediction_network.fc_value")
        policy = self._mlp_v2(policy, "prediction_network.fc_policy")
        return policy, value

    def coverage_summary(self) -> dict[str, Any]:
        return summarize_state_dict_coverage(self.params, self.consumed_keys)

    def _p(self, key: str) -> np.ndarray:
        if key not in self.params:
            raise KeyError(f"JAX shadow model is missing state_dict key {key!r}")
        self.consumed_keys.add(key)
        return self.params[key]

    def _as_jnp(self, value: Any) -> Any:
        return self.jnp.asarray(value, dtype=self.jnp.float32)

    def _conv2d(self, x: Any, prefix: str, *, stride: int, padding: int) -> Any:
        weight = self._as_jnp(self._p(f"{prefix}.weight"))
        y = self.jax.lax.conv_general_dilated(
            x,
            weight,
            window_strides=(stride, stride),
            padding=((padding, padding), (padding, padding)),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            precision=self.jax.lax.Precision.HIGHEST,
        )
        bias_key = f"{prefix}.bias"
        if bias_key in self.params:
            bias = self._as_jnp(self._p(bias_key)).reshape((1, -1, 1, 1))
            y = y + bias
        return y

    def _linear(self, x: Any, prefix: str) -> Any:
        weight = self._as_jnp(self._p(f"{prefix}.weight"))
        bias = self._as_jnp(self._p(f"{prefix}.bias"))
        return self.jnp.matmul(
            x,
            weight.T,
            precision=self.jax.lax.Precision.HIGHEST,
        ) + bias

    def _batch_norm_2d(self, x: Any, prefix: str) -> Any:
        weight = self._as_jnp(self._p(f"{prefix}.weight")).reshape((1, -1, 1, 1))
        bias = self._as_jnp(self._p(f"{prefix}.bias")).reshape((1, -1, 1, 1))
        mean = self._as_jnp(self._p(f"{prefix}.running_mean")).reshape((1, -1, 1, 1))
        var = self._as_jnp(self._p(f"{prefix}.running_var")).reshape((1, -1, 1, 1))
        return (x - mean) * self.jax.lax.rsqrt(var + 1e-5) * weight + bias

    def _batch_norm_1d(self, x: Any, prefix: str) -> Any:
        weight = self._as_jnp(self._p(f"{prefix}.weight")).reshape((1, -1))
        bias = self._as_jnp(self._p(f"{prefix}.bias")).reshape((1, -1))
        mean = self._as_jnp(self._p(f"{prefix}.running_mean")).reshape((1, -1))
        var = self._as_jnp(self._p(f"{prefix}.running_var")).reshape((1, -1))
        return (x - mean) * self.jax.lax.rsqrt(var + 1e-5) * weight + bias

    def _avg_pool2d(self, x: Any, *, kernel_size: int, stride: int, padding: int) -> Any:
        total = self.jax.lax.reduce_window(
            x,
            self.jnp.array(0.0, dtype=x.dtype),
            self.jax.lax.add,
            window_dimensions=(1, 1, kernel_size, kernel_size),
            window_strides=(1, 1, stride, stride),
            padding=((0, 0), (0, 0), (padding, padding), (padding, padding)),
        )
        return total / float(kernel_size * kernel_size)

    def _resblock_basic(self, x: Any, prefix: str) -> Any:
        identity = x
        x = self._conv2d(x, f"{prefix}.conv1.0", stride=1, padding=1)
        x = self._batch_norm_2d(x, f"{prefix}.conv1.1")
        x = self.jax.nn.relu(x)
        x = self._conv2d(x, f"{prefix}.conv2.0", stride=1, padding=1)
        x = self._batch_norm_2d(x, f"{prefix}.conv2.1")
        return self.jax.nn.relu(x + identity)

    def _resblock_downsample(self, x: Any, prefix: str) -> Any:
        identity = self._conv2d(x, f"{prefix}.conv3.0", stride=2, padding=1)
        x = self._conv2d(x, f"{prefix}.conv1.0", stride=2, padding=1)
        x = self._batch_norm_2d(x, f"{prefix}.conv1.1")
        x = self.jax.nn.relu(x)
        x = self._conv2d(x, f"{prefix}.conv2.0", stride=1, padding=1)
        x = self._batch_norm_2d(x, f"{prefix}.conv2.1")
        return self.jax.nn.relu(x + identity)

    def _mlp_v2(self, x: Any, prefix: str) -> Any:
        x = self._linear(x, f"{prefix}.0")
        x = self._batch_norm_1d(x, f"{prefix}.1")
        x = self.jax.nn.relu(x)
        return self._linear(x, f"{prefix}.3")

    def _indexed_prefixes(self, prefix: str) -> list[str]:
        found: set[int] = set()
        start = f"{prefix}."
        for key in self.params:
            if not key.startswith(start):
                continue
            rest = key.removeprefix(start)
            index_text = rest.split(".", 1)[0]
            if index_text.isdigit():
                found.add(int(index_text))
        return [f"{prefix}.{index}" for index in sorted(found)]


def import_jax_modules(*, platform: str | None = None) -> tuple[Any, Any]:
    """Lazy import JAX so normal local tests do not require it."""

    import jax
    import jax.numpy as jnp

    if platform:
        jax.config.update("jax_platform_name", platform)
    return jax, jnp


def jax_shadow_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    platform: str | None = None,
) -> JaxMuZeroShadowModel:
    """Build the JAX shadow model from a PyTorch-style state dict."""

    jax, jnp = import_jax_modules(platform=platform)
    return JaxMuZeroShadowModel(state_dict, jax=jax, jnp=jnp)


def deterministic_observation_batch(
    *,
    batch_size: int,
    shape: tuple[int, int, int] = (4, 64, 64),
    kind: str,
    seed: int,
) -> np.ndarray:
    """Build deterministic inputs for parity checks."""

    if kind == "zeros":
        return np.zeros((batch_size, *shape), dtype=np.float32)
    if kind == "ramp":
        base = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32)
        return np.broadcast_to(base.reshape((1, *shape)), (batch_size, *shape)).copy()
    if kind == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 1.0, size=(batch_size, *shape)).astype(np.float32)
    raise ValueError(f"unknown deterministic observation kind {kind!r}")


def profile_only_report_base() -> dict[str, Any]:
    """Return immutable safety labels for reports and tests."""

    return dict(PROFILE_ONLY_LABELS)


__all__ = [
    "ComparisonResult",
    "IGNORED_INFERENCE_PREFIXES",
    "JaxMuZeroShadowModel",
    "LIGHTZERO_JAX_SHADOW_PARITY_IMPL",
    "LIGHTZERO_JAX_SHADOW_PARITY_SEMANTICS",
    "MUTABLE_CHECKPOINT_BASENAMES",
    "PROFILE_ONLY_LABELS",
    "checkpoint_sha256",
    "compare_arrays",
    "deterministic_observation_batch",
    "inference_required_state_keys",
    "inverse_scalar_transform_logits",
    "is_mutable_checkpoint_ref",
    "jax_shadow_from_state_dict",
    "profile_only_report_base",
    "require_immutable_checkpoint_ref",
    "state_dict_to_numpy",
    "summarize_state_dict_coverage",
]
