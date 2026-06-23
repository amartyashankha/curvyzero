"""Profile-only MCTX compact search service.

This is a search-backend probe, not a trainer path. It lets the compact slab
run the same action-feedback and replay-index checks with a JAX/MCTX search
body instead of the current LightZero CTree/list control loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import time
from typing import Any

import numpy as np

from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import CompactSearchResultV1
from curvyzero.training.compact_policy_row_bridge import validate_compact_search_result_v1
from curvyzero.training.multiplayer_source_state_target_rows import ACTION_COUNT
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


MCTX_COMPACT_SEARCH_SERVICE_IMPL = "mctx_compact_search_service_profile_only_v0"
MCTX_COMPACT_SEARCH_SERVICE_SEMANTICS = (
    "profile_only_jax_mctx_gumbel_muzero_search_not_lightzero_ctree"
)


@dataclass(frozen=True, slots=True)
class MctxCompactSearchConfig:
    """Fixed-shape options for the profile-only MCTX compact service."""

    hidden_dim: int = 64
    visual_channels: int = 8
    max_depth: int | None = None
    gumbel_scale: float = 1.0
    require_all_roots_active: bool = True
    require_gpu_backend: bool = False
    normalize_uint8_observation: bool = True


class MctxCompactSearchServiceV1:
    """JAX/MCTX search backend for ``CompactSearchServiceV1`` profile probes."""

    search_impl = MCTX_COMPACT_SEARCH_SERVICE_IMPL
    profile_only = True
    calls_train_muzero = False
    trainer_defaults_changed = False
    touches_live_runs = False

    def __init__(
        self,
        *,
        num_simulations: int,
        seed: int = 0,
        config: MctxCompactSearchConfig | None = None,
        shadow_model: Any | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        simulations = int(num_simulations)
        if simulations <= 0:
            raise ValueError("num_simulations must be positive")
        self.num_simulations = simulations
        self._seed = int(seed)
        self._config = config or MctxCompactSearchConfig()
        self._shadow_model = shadow_model
        self._model_metadata = dict(model_metadata or {})
        self._call_index = 0
        self._backend_signature: tuple[Any, ...] | None = None
        self._params: dict[str, Any] | None = None
        self._run_search: Any | None = None

    def run(self, root_batch: CompactRootBatchV1) -> CompactSearchResultV1:
        """Run MCTX over active compact roots and return a checked result."""

        observation = np.asarray(root_batch.observation)
        legal_mask = _legal_mask(root_batch.legal_mask)
        active_mask = np.asarray(root_batch.active_root_mask, dtype=np.bool_).reshape(-1)
        if legal_mask.shape != (active_mask.size, ACTION_COUNT):
            raise ReplayCompatibilityError("MCTX compact service root mask shape mismatch")
        active_indices = np.flatnonzero(active_mask).astype(np.int32, copy=False)
        active_count = int(active_indices.size)
        if active_count == 0:
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=np.zeros((0,), dtype=np.int16),
                visit_policy=np.zeros((0, ACTION_COUNT), dtype=np.float32),
                root_value=np.zeros((0,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
                metadata=_metadata(profile_telemetry={"mctx_compact_search_service_zero": True}),
            )
        if self._config.require_all_roots_active and not bool(active_mask.all()):
            raise ReplayCompatibilityError(
                "MCTX compact service requires all roots active for fixed-shape profiling"
            )

        if not bool(legal_mask[active_indices].any(axis=1).all()):
            raise ReplayCompatibilityError("MCTX compact service got an active root with no legal action")

        import jax

        backend = jax.default_backend()
        if self._config.require_gpu_backend and backend not in {"gpu", "cuda"}:
            raise ReplayCompatibilityError(f"MCTX compact service expected GPU JAX backend, got {backend!r}")

        jnp, _mctx, params, run_search = self._backend(
            jax=jax,
            observation_shape=tuple(int(dim) for dim in observation.shape[1:]),
            observation_dtype=str(observation.dtype),
        )

        h2d_started = time.perf_counter()
        obs_host = observation[active_indices]
        invalid_host = ~legal_mask[active_indices]
        obs_device = jax.device_put(obs_host)
        invalid_device = jax.device_put(invalid_host)
        obs_device.block_until_ready()
        invalid_device.block_until_ready()
        h2d_sec = time.perf_counter() - h2d_started

        search_started = time.perf_counter()
        search_output = run_search(
            params,
            jax.random.PRNGKey(self._seed + self._call_index),
            obs_device,
            invalid_device,
            num_simulations=self.num_simulations,
            max_depth=self._max_depth(),
        )
        output, predicted_value_device, predicted_policy_logits_device = search_output
        output.action_weights.block_until_ready()
        search_sec = time.perf_counter() - search_started

        d2h_started = time.perf_counter()
        selected = np.asarray(output.action).astype(np.int16, copy=False)
        visit_policy = np.asarray(output.action_weights).astype(np.float32, copy=False)
        predicted_values = np.asarray(predicted_value_device).astype(np.float32, copy=False)
        predicted_policy_logits = np.asarray(predicted_policy_logits_device).astype(
            np.float32,
            copy=False,
        )
        root_values, root_value_source = _extract_mctx_root_values(output)
        if root_values is None:
            raise ReplayCompatibilityError("MCTX compact service could not extract root values")
        root_values = root_values.astype(np.float32, copy=False)
        d2h_sec = time.perf_counter() - d2h_started

        self._call_index += 1
        total_sec = h2d_sec + search_sec + d2h_sec
        profile_telemetry = {
            "mctx_compact_search_service_profile_only": True,
            "mctx_compact_search_service_not_lightzero_ctree": True,
            "mctx_compact_search_service_not_train_muzero": True,
            "mctx_compact_search_service_total_sec": float(total_sec),
            "mctx_compact_search_service_h2d_sec": float(h2d_sec),
            "mctx_compact_search_service_search_sec": float(search_sec),
            "mctx_compact_search_service_d2h_sec": float(d2h_sec),
            "mctx_compact_search_service_obs_h2d_bytes": float(obs_host.nbytes),
            "mctx_compact_search_service_mask_h2d_bytes": float(invalid_host.nbytes),
            "mctx_compact_search_service_action_d2h_bytes": float(selected.nbytes),
            "mctx_compact_search_service_replay_payload_d2h_bytes": float(
                visit_policy.nbytes + root_values.nbytes
            ),
            "mctx_compact_search_service_predicted_value_shape": list(
                predicted_values.shape
            ),
            "mctx_compact_search_service_predicted_policy_logits_shape": list(
                predicted_policy_logits.shape
            ),
            "mctx_compact_search_service_active_roots": float(active_count),
            "mctx_compact_search_service_requested_simulations": float(self.num_simulations),
            "mctx_compact_search_service_actual_search_simulations": float(
                self.num_simulations
            ),
            "mctx_compact_search_service_backend": str(backend),
            "mctx_compact_search_service_model_backend": self._model_backend_name(),
            "mctx_compact_search_service_model_metadata": dict(self._model_metadata),
            "mctx_compact_search_service_shadow_coverage": self._shadow_coverage_summary(),
            "mctx_compact_search_service_hidden_dim": float(self._config.hidden_dim),
            "mctx_compact_search_service_visual_channels": float(
                self._config.visual_channels
            ),
            "mctx_compact_search_service_root_value_source": root_value_source,
        }
        del jnp
        return validate_compact_search_result_v1(
            root_batch,
            selected_action=selected,
            visit_policy=visit_policy,
            root_value=root_values,
            predicted_value=predicted_values,
            predicted_policy_logits=predicted_policy_logits,
            search_impl=self.search_impl,
            num_simulations=self.num_simulations,
            metadata=_metadata(profile_telemetry=profile_telemetry),
        )

    def _backend(
        self,
        *,
        jax: Any,
        observation_shape: tuple[int, ...],
        observation_dtype: str,
    ) -> tuple[Any, Any, dict[str, Any], Any]:
        import jax.numpy as jnp
        import mctx

        signature = (
            observation_shape,
            observation_dtype,
            int(self._config.hidden_dim),
            int(self._config.visual_channels),
            bool(_is_visual_stack(observation_shape)),
            self._model_backend_name(),
            id(self._shadow_model) if self._shadow_model is not None else 0,
        )
        if self._backend_signature == signature and self._params is not None:
            if self._run_search is None:
                raise AssertionError("MCTX compact search backend cache lost run_search")
            return jnp, mctx, self._params, self._run_search

        hidden_dim = int(self._config.hidden_dim)
        visual_channels = int(self._config.visual_channels)
        if self._shadow_model is not None:
            return self._real_shadow_backend(
                jax=jax,
                jnp=jnp,
                mctx=mctx,
                signature=signature,
                observation_dtype=observation_dtype,
            )

        params = {
            "action_embed": _linspace_matrix(jnp, ACTION_COUNT, hidden_dim, 0.10),
            "dynamics_w": _linspace_matrix(jnp, hidden_dim, hidden_dim, 0.025),
            "dynamics_b": jnp.linspace(0.01, -0.01, hidden_dim, dtype=jnp.float32),
            "policy_w": _linspace_matrix(jnp, hidden_dim, ACTION_COUNT, 0.07),
            "policy_b": jnp.array([0.0, 0.03, -0.01], dtype=jnp.float32),
            "value_w": jnp.linspace(-0.04, 0.04, hidden_dim, dtype=jnp.float32),
            "reward_w": jnp.linspace(0.03, -0.03, hidden_dim, dtype=jnp.float32),
        }
        visual_stack = _is_visual_stack(observation_shape)
        if visual_stack:
            params.update(
                {
                    "visual_conv1": _linspace_kernel(jnp, 3, 3, 4, visual_channels, 0.035),
                    "visual_conv1_b": jnp.linspace(
                        -0.01,
                        0.01,
                        visual_channels,
                        dtype=jnp.float32,
                    ),
                    "visual_conv2": _linspace_kernel(
                        jnp,
                        3,
                        3,
                        visual_channels,
                        visual_channels,
                        0.025,
                    ),
                    "visual_conv2_b": jnp.linspace(
                        0.01,
                        -0.01,
                        visual_channels,
                        dtype=jnp.float32,
                    ),
                    "visual_projection_w": _linspace_matrix(
                        jnp,
                        visual_channels,
                        hidden_dim,
                        0.08,
                    ),
                    "visual_projection_b": jnp.linspace(
                        -0.02,
                        0.02,
                        hidden_dim,
                        dtype=jnp.float32,
                    ),
                }
            )
        else:
            obs_dim = int(np.prod(observation_shape, dtype=np.int64))
            params.update(
                {
                    "representation_w": _linspace_matrix(jnp, obs_dim, hidden_dim, 0.08),
                    "representation_b": jnp.linspace(
                        -0.02,
                        0.02,
                        hidden_dim,
                        dtype=jnp.float32,
                    ),
                }
            )

        def representation(params: dict[str, Any], obs_batch: Any) -> Any:
            x = obs_batch.astype(jnp.float32)
            if self._config.normalize_uint8_observation and observation_dtype == "uint8":
                x = x * jnp.float32(1.0 / 255.0)
            if visual_stack:
                x = jnp.transpose(x, (0, 2, 3, 1))
                x = jax.lax.conv_general_dilated(
                    x,
                    params["visual_conv1"],
                    window_strides=(2, 2),
                    padding="SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                )
                x = jax.nn.relu(x + params["visual_conv1_b"])
                x = jax.lax.conv_general_dilated(
                    x,
                    params["visual_conv2"],
                    window_strides=(2, 2),
                    padding="SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                )
                x = jax.nn.relu(x + params["visual_conv2_b"])
                x = jnp.mean(x, axis=(1, 2))
                return jnp.tanh(
                    x @ params["visual_projection_w"] + params["visual_projection_b"]
                )
            return jnp.tanh(
                x.reshape((x.shape[0], -1)) @ params["representation_w"]
                + params["representation_b"]
            )

        def prediction(params: dict[str, Any], hidden: Any) -> tuple[Any, Any]:
            prior_logits = hidden @ params["policy_w"] + params["policy_b"]
            value = jnp.tanh(hidden @ params["value_w"])
            return prior_logits, value

        def recurrent_fn(
            params: dict[str, Any],
            rng_key: Any,
            action: Any,
            hidden: Any,
        ) -> tuple[Any, Any]:
            del rng_key
            action_features = jax.nn.one_hot(action, ACTION_COUNT, dtype=jnp.float32)
            action_delta = action_features @ params["action_embed"]
            next_hidden = jnp.tanh(
                hidden + action_delta + hidden @ params["dynamics_w"] + params["dynamics_b"]
            )
            prior_logits, value = prediction(params, next_hidden)
            reward = 0.05 * jnp.tanh(next_hidden @ params["reward_w"])
            discount = jnp.full_like(value, 0.99)
            return (
                mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=discount,
                    prior_logits=prior_logits,
                    value=value,
                ),
                next_hidden,
            )

        @functools.partial(jax.jit, static_argnames=("num_simulations", "max_depth"))
        def run_search(
            params: dict[str, Any],
            rng_key: Any,
            obs: Any,
            invalid_actions: Any,
            *,
            num_simulations: int,
            max_depth: int,
        ) -> Any:
            hidden = representation(params, obs)
            prior_logits, value = prediction(params, hidden)
            root = mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=value,
                embedding=hidden,
            )
            output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=invalid_actions,
                max_depth=max_depth,
                max_num_considered_actions=ACTION_COUNT,
                gumbel_scale=float(self._config.gumbel_scale),
            )
            return output, value, prior_logits

        self._backend_signature = signature
        self._params = params
        self._run_search = run_search
        return jnp, mctx, params, run_search

    def _real_shadow_backend(
        self,
        *,
        jax: Any,
        jnp: Any,
        mctx: Any,
        signature: tuple[Any, ...],
        observation_dtype: str,
    ) -> tuple[Any, Any, dict[str, Any], Any]:
        shadow_model = self._shadow_model
        if shadow_model is None:
            raise AssertionError("real shadow backend requires a shadow_model")
        value_support_scale = _support_scale_from_size(
            int(shadow_model.value_support_size),
            "value_support_size",
        )
        reward_support_scale = _support_scale_from_size(
            int(shadow_model.reward_support_size),
            "reward_support_size",
        )
        params = {"shadow_backend_marker": jnp.asarray(1, dtype=jnp.int32)}

        def prepare_observation(obs_batch: Any) -> Any:
            x = obs_batch.astype(jnp.float32)
            if self._config.normalize_uint8_observation and observation_dtype == "uint8":
                x = x * jnp.float32(1.0 / 255.0)
            return x

        def prediction(hidden: Any) -> tuple[Any, Any, Any]:
            prior_logits, value_logits = shadow_model.prediction(hidden)
            value = _inverse_scalar_transform_logits_jax(
                jax,
                jnp,
                value_logits,
                support_scale=value_support_scale,
            )
            return prior_logits, value, value_logits

        def recurrent_fn(
            params: dict[str, Any],
            rng_key: Any,
            action: Any,
            hidden: Any,
        ) -> tuple[Any, Any]:
            del params, rng_key
            recurrent = shadow_model.recurrent_inference(hidden, action)
            reward = _inverse_scalar_transform_logits_jax(
                jax,
                jnp,
                recurrent["reward"],
                support_scale=reward_support_scale,
            )
            prior_logits, value, _value_logits = prediction(recurrent["latent_state"])
            discount = jnp.full_like(value, 0.99)
            return (
                mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=discount,
                    prior_logits=prior_logits,
                    value=value,
                ),
                recurrent["latent_state"],
            )

        @functools.partial(jax.jit, static_argnames=("num_simulations", "max_depth"))
        def run_search(
            params: dict[str, Any],
            rng_key: Any,
            obs: Any,
            invalid_actions: Any,
            *,
            num_simulations: int,
            max_depth: int,
        ) -> Any:
            del params
            hidden = shadow_model.representation(prepare_observation(obs))
            prior_logits, value, _value_logits = prediction(hidden)
            root = mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=value,
                embedding=hidden,
            )
            output = mctx.gumbel_muzero_policy(
                params={"shadow_backend_marker": jnp.asarray(1, dtype=jnp.int32)},
                rng_key=rng_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=invalid_actions,
                max_depth=max_depth,
                max_num_considered_actions=ACTION_COUNT,
                gumbel_scale=float(self._config.gumbel_scale),
            )
            return output, value, prior_logits

        self._backend_signature = signature
        self._params = params
        self._run_search = run_search
        return jnp, mctx, params, run_search

    def _max_depth(self) -> int:
        if self._config.max_depth is not None:
            return int(self._config.max_depth)
        return max(1, int(self.num_simulations))

    def _model_backend_name(self) -> str:
        if self._shadow_model is None:
            return "toy_jax_model"
        return "lightzero_jax_shadow_model"

    def _shadow_coverage_summary(self) -> dict[str, Any] | None:
        if self._shadow_model is None:
            return None
        coverage_summary = getattr(self._shadow_model, "coverage_summary", None)
        if coverage_summary is None:
            return None
        return compact_shadow_coverage_summary(coverage_summary())


def compact_shadow_coverage_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep checkpoint coverage telemetry useful without dumping every key."""

    missing = list(summary.get("missing_required_keys", []))
    extra = list(summary.get("extra_consumed_keys", []))
    return {
        "ok": bool(summary.get("ok", False)),
        "required_key_count": int(summary.get("required_key_count", 0)),
        "consumed_key_count": int(summary.get("consumed_key_count", 0)),
        "ignored_key_count": int(summary.get("ignored_key_count", 0)),
        "missing_required_key_count": len(missing),
        "extra_consumed_key_count": len(extra),
        "missing_required_keys_head": [str(key) for key in missing[:5]],
        "extra_consumed_keys_head": [str(key) for key in extra[:5]],
    }


def _metadata(*, profile_telemetry: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_backend": MCTX_COMPACT_SEARCH_SERVICE_IMPL,
        "profile_semantics": MCTX_COMPACT_SEARCH_SERVICE_SEMANTICS,
        "compact_search_service_adapter": True,
        "profile_only": True,
        "not_lightzero_ctree": True,
        "not_train_muzero": True,
        "profile_telemetry": profile_telemetry,
    }


def _legal_mask(value: Any) -> np.ndarray:
    mask = np.asarray(value, dtype=np.bool_)
    if mask.ndim != 2 or mask.shape[1] != ACTION_COUNT:
        raise ReplayCompatibilityError(
            f"MCTX compact service legal_mask must have shape [R,{ACTION_COUNT}]"
        )
    return mask


def _is_visual_stack(observation_shape: tuple[int, ...]) -> bool:
    return len(observation_shape) == 3 and observation_shape == (4, 64, 64)


def _linspace_matrix(jnp: Any, rows: int, cols: int, scale: float) -> Any:
    return jnp.linspace(-scale, scale, rows * cols, dtype=jnp.float32).reshape(rows, cols)


def _linspace_kernel(
    jnp: Any,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    scale: float,
) -> Any:
    return jnp.linspace(
        -scale,
        scale,
        height * width * in_channels * out_channels,
        dtype=jnp.float32,
    ).reshape(height, width, in_channels, out_channels)


def _support_scale_from_size(width: int, name: str) -> int:
    if width <= 0 or width % 2 != 1:
        raise ReplayCompatibilityError(f"{name} must be a positive odd support width")
    return (width - 1) // 2


def _inverse_scalar_transform_logits_jax(
    jax: Any,
    jnp: Any,
    logits: Any,
    *,
    support_scale: int,
    epsilon: float = 0.001,
) -> Any:
    scale = int(support_scale)
    if scale < 0:
        raise ReplayCompatibilityError("support_scale must be nonnegative")
    values = jax.nn.softmax(logits.astype(jnp.float32), axis=1)
    support = jnp.arange(-scale, scale + 1, dtype=jnp.float32).reshape((1, -1))
    value = jnp.sum(values * support, axis=1)
    eps = jnp.float32(epsilon)
    tmp = (jnp.sqrt(1.0 + 4.0 * eps * (jnp.abs(value) + 1.0 + eps)) - 1.0) / (
        2.0 * eps
    )
    return jnp.sign(value) * (tmp * tmp - 1.0)


def _extract_mctx_root_values(output: Any) -> tuple[np.ndarray | None, str]:
    search_tree = getattr(output, "search_tree", None)
    if search_tree is None:
        return None, "missing_search_tree"
    for name in ("node_values", "values", "raw_values"):
        value = getattr(search_tree, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim >= 2 and array.shape[0] > 0:
            return array[:, 0].astype(np.float32, copy=False), f"search_tree.{name}[:,0]"
        if array.ndim == 1:
            return array.astype(np.float32, copy=False), f"search_tree.{name}"
    try:
        summary = search_tree.summary()
    except Exception:
        summary = None
    if summary is not None:
        for name in ("value", "values", "root_value", "root_values"):
            value = getattr(summary, name, None)
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim >= 2:
                return array[..., 0].astype(np.float32, copy=False), f"summary.{name}[...,0]"
            if array.ndim == 1:
                return array.astype(np.float32, copy=False), f"summary.{name}"
    return None, "unavailable"


__all__ = [
    "MCTX_COMPACT_SEARCH_SERVICE_IMPL",
    "MCTX_COMPACT_SEARCH_SERVICE_SEMANTICS",
    "MctxCompactSearchConfig",
    "MctxCompactSearchServiceV1",
    "compact_shadow_coverage_summary",
]
