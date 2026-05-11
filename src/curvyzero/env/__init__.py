"""CurvyZero environment exports.

`VectorMultiplayerEnv` is the shared fast multiplayer runtime under hardening.
`CurvyTronSourceEnv` is an oracle/proof tool, and `CurvyTronEnv` is a legacy
toy environment kept for old tests.
"""

from curvyzero.env.config import CurvyTronConfig
from curvyzero.env.core import CurvyTronEnv, StepResult
from curvyzero.env.source_env import CurvyTronSourceEnv
from curvyzero.env.vector_multiplayer_env import (
    VectorMultiplayerBatch,
    VectorMultiplayerEnv,
)
from curvyzero.env.vector_trainer_env import (
    VectorTrainerBatch1v1NoBonus,
    VectorTrainerEnv1v1NoBonus,
)

__all__ = [
    "CurvyTronConfig",
    "CurvyTronEnv",
    "CurvyTronSourceEnv",
    "StepResult",
    "VectorMultiplayerBatch",
    "VectorMultiplayerEnv",
    "VectorTrainerBatch1v1NoBonus",
    "VectorTrainerEnv1v1NoBonus",
]
