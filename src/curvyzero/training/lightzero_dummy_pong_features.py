"""Feature encoders shared by LightZero dummy Pong adapters."""

from __future__ import annotations

import numpy as np

from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongObservation

TABULAR_FEATURE_SCHEMA_ID = "dummy_pong_lightzero_tabular_ego_v0"
RASTER_FLAT_FEATURE_SCHEMA_ID = "dummy_pong_lightzero_raster_flat_v0"
FEATURE_SCHEMA_IDS = {
    "tabular_ego": TABULAR_FEATURE_SCHEMA_ID,
    "raster_flat": RASTER_FLAT_FEATURE_SCHEMA_ID,
}


def encode_tabular_ego_observation(
    observation: PongObservation,
    config: PongConfig,
) -> np.ndarray:
    """Encode the 10-float tabular_ego row used by DummyPongLightZeroEnv."""

    return np.asarray(
        [
            _scale_unit(observation.ego_paddle_y, config.height - config.paddle_height),
            _scale_unit(observation.opponent_paddle_y, config.height - config.paddle_height),
            _scale_unit(observation.ego_paddle_x, config.width - 1),
            _scale_unit(observation.opponent_paddle_x, config.width - 1),
            _scale_signed(observation.ball_dx_forward, config.width - 1),
            _scale_signed(observation.ball_dy_from_ego_center, config.height - 1),
            float(observation.ball_vx_forward),
            float(observation.ball_vy),
            _scale_unit(observation.ball_y, config.height - 1),
            _scale_unit(observation.step, config.max_steps),
        ],
        dtype=np.float32,
    )


def lightzero_observation_shape(feature_mode: str, config: PongConfig) -> int:
    if feature_mode == "tabular_ego":
        return 10
    if feature_mode == "raster_flat":
        return int(config.width * config.height)
    raise ValueError(
        f"unknown feature_mode {feature_mode!r}; expected tabular_ego or raster_flat"
    )


def lightzero_feature_schema_id(feature_mode: str) -> str:
    try:
        return FEATURE_SCHEMA_IDS[feature_mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown feature_mode {feature_mode!r}; expected tabular_ego or raster_flat"
        ) from exc


def encode_lightzero_observation(
    *,
    feature_mode: str,
    observation: PongObservation,
    config: PongConfig,
    raster_grid: np.ndarray | None = None,
) -> np.ndarray:
    if feature_mode == "tabular_ego":
        return encode_tabular_ego_observation(observation, config)
    if feature_mode == "raster_flat":
        if raster_grid is None:
            raise ValueError("raster_flat encoding requires raster_grid")
        return np.asarray(raster_grid, dtype=np.float32).reshape(-1)
    raise ValueError(
        f"unknown feature_mode {feature_mode!r}; expected tabular_ego or raster_flat"
    )


def _scale_unit(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return float(value) / float(maximum)


def _scale_signed(value: float, maximum_abs: float) -> float:
    if maximum_abs <= 0:
        return 0.0
    return float(np.clip(float(value) / float(maximum_abs), -1.0, 1.0))
