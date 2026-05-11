import pytest

from curvyzero.infra.modal.lightzero_pong_exact_reproduction import (
    _install_lightzero_phase_profile,
)
from curvyzero.infra.modal.lightzero_pong_exact_reproduction import _LightZeroPhaseProfiler
from curvyzero.infra.modal.lightzero_pong_exact_reproduction import _LightZeroProfileStop
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    _install_lightzero_phase_profile as _install_curvytron_lightzero_phase_profile,
)
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    _LightZeroPhaseProfiler as _CurvyTronLightZeroPhaseProfiler,
)


class BaseLearner:
    def __init__(self) -> None:
        self.train_iter = 0

    def train(self) -> str:
        self.train_iter += 1
        return "trained"

    def save_checkpoint(self) -> None:
        return None

    def call_hook(self, name: str) -> None:
        return None


def fake_train_muzero() -> None:
    return None


def test_lightzero_phase_profiler_can_stop_after_learner_train_calls():
    profiler = _LightZeroPhaseProfiler(enabled=True, gpu_sample_interval_sec=0.0)
    restore = _install_lightzero_phase_profile(
        train_muzero=fake_train_muzero,
        profiler=profiler,
        stop_after_learner_train_calls=2,
    )

    learner = BaseLearner()
    try:
        assert learner.train() == "trained"
        with pytest.raises(_LightZeroProfileStop):
            learner.train()
    finally:
        restore()

    assert profiler.counts["learner_train_calls"] == 2
    assert profiler.counts["learner_train_iter_delta"] == 2
    assert "learner_train_sec" in profiler.timers
    assert BaseLearner.train(learner) == "trained"


class FakeParameter:
    device = "cuda:0"


class FakeTensor:
    def __init__(self, batch_size: int) -> None:
        self.shape = (batch_size, 2)

    def size(self, dim: int) -> int:
        return self.shape[dim]


class FakeModel:
    def parameters(self):
        return iter([FakeParameter()])

    def recurrent_inference(self, latent_state, action):
        return {"value": FakeTensor(latent_state.shape[0])}


class FakeRoots:
    num = 3


class FakeMctsCfg:
    num_simulations = 7
    device = "cuda:0"


class MuZeroMCTSCtree:
    def __init__(self) -> None:
        self._cfg = FakeMctsCfg()

    def search(self, roots, model, latent_state_roots, to_play_batch):
        return model.recurrent_inference(FakeTensor(roots.num), FakeTensor(roots.num))


def test_curvytron_phase_profiler_records_mcts_model_device_and_batch():
    profiler = _CurvyTronLightZeroPhaseProfiler(enabled=True)
    restore = _install_curvytron_lightzero_phase_profile(
        train_muzero=fake_train_muzero,
        profiler=profiler,
        stop_after_learner_train_calls=10,
    )

    try:
        result = MuZeroMCTSCtree().search(FakeRoots(), FakeModel(), [], [])
    finally:
        restore()

    assert sorted(result.keys()) == ["value"]
    assert profiler.counts["mcts_search_calls"] == 1
    assert profiler.counts["mcts_search_root_sum"] == 3
    assert profiler.counts["mcts_search_node_budget_sum"] == 21
    assert profiler.counts["model_recurrent_inference_calls"] == 1
    assert profiler.counts["model_recurrent_inference_batch_sum"] == 3
    assert (
        profiler.summary()["derived_stats"]["model_recurrent_inference_batch_mean"]
        == 3.0
    )
    assert "model_recurrent_inference_sec" in profiler.timers
    assert profiler.samples["mcts_cfg_device"] == ["cuda:0"]
    assert profiler.samples["mcts_search_return_keys"] == [["value"]]
    assert any(
        sample.endswith(":cuda:0")
        for sample in profiler.samples["model_parameter_device"]
    )
