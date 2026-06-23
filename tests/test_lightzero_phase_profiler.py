import importlib
from types import SimpleNamespace

import numpy as np
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
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT,
)
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    COLLECT_SEARCH_BACKEND_STOCK,
)
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    COLLECT_SEARCH_CTREE_BACKEND_FLAT_A3,
)
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    COLLECT_SEARCH_CTREE_BACKEND_LIGHTZERO,
)
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    _should_install_lightzero_phase_profile,
)
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    _compact_train_result_for_output,
)
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    _install_lightzero_collect_search_backend_hook,
)
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    _validate_collect_search_backend,
)
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    _validate_collect_search_ctree_backend,
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


class Evaluator:
    def __init__(self) -> None:
        self._default_n_episode = 3

    def eval(self):
        return True, {"eval_episode_return": [1.0, 1.0, 1.0]}


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


def test_curvytron_train_call_cap_installs_phase_profile_outside_profile_mode():
    assert _should_install_lightzero_phase_profile(
        mode="train",
        stop_after_learner_train_calls=1,
    )
    assert not _should_install_lightzero_phase_profile(
        mode="train",
        stop_after_learner_train_calls=0,
    )
    assert _should_install_lightzero_phase_profile(
        mode="profile",
        stop_after_learner_train_calls=0,
    )


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


def test_curvytron_phase_profiler_skip_evaluator_keeps_lightzero_return_shape():
    profiler = _CurvyTronLightZeroPhaseProfiler(enabled=True)
    restore = _install_curvytron_lightzero_phase_profile(
        train_muzero=fake_train_muzero,
        profiler=profiler,
        stop_after_learner_train_calls=10,
        skip_evaluator_eval=True,
    )

    try:
        stop, info = Evaluator().eval()
    finally:
        restore()

    assert stop is False
    assert info["skipped"] is True
    assert info["eval_episode_return"] == [0.0, 0.0, 0.0]
    assert info["eval_episode_return_mean"] == 0.0
    assert profiler.counts["evaluator_eval_skipped_calls"] == 1


def test_curvytron_phase_profiler_falls_back_to_lzero_worker_collector_for_rnd_entrypoint(
    monkeypatch,
):
    worker_module = importlib.import_module("lzero.worker")

    class RndEntrypointCollector:
        def __init__(self) -> None:
            self.envstep = 10

        def collect(self):
            self.envstep += 7
            return ["segment-a", "segment-b"]

    monkeypatch.setattr(worker_module, "MuZeroCollector", RndEntrypointCollector)
    profiler = _CurvyTronLightZeroPhaseProfiler(enabled=True)
    restore = _install_curvytron_lightzero_phase_profile(
        train_muzero=fake_train_muzero,
        profiler=profiler,
        stop_after_learner_train_calls=10,
    )

    try:
        result = worker_module.MuZeroCollector().collect()
    finally:
        restore()

    assert result == ["segment-a", "segment-b"]
    assert profiler.counts["collector_collect_calls"] == 1
    assert profiler.counts["env_steps_collected"] == 7
    assert profiler.counts["game_segments_collected"] == 2
    assert any(
        hook.endswith("RndEntrypointCollector.collect")
        for hook in profiler.installed_hooks
    )


def test_collect_search_backend_validation_and_stock_noop():
    assert _validate_collect_search_backend(COLLECT_SEARCH_BACKEND_STOCK) == "stock"
    with pytest.raises(ValueError, match="collect_search_backend"):
        _validate_collect_search_backend("dense_but_imaginary")
    assert (
        _validate_collect_search_ctree_backend(COLLECT_SEARCH_CTREE_BACKEND_LIGHTZERO)
        == "lightzero"
    )
    assert (
        _validate_collect_search_ctree_backend(COLLECT_SEARCH_CTREE_BACKEND_FLAT_A3)
        == "flat_a3"
    )
    with pytest.raises(ValueError, match="collect_search_ctree_backend"):
        _validate_collect_search_ctree_backend("wide_open")

    profiler = _CurvyTronLightZeroPhaseProfiler(enabled=True)
    restore = _install_lightzero_collect_search_backend_hook(
        train_muzero=fake_train_muzero,
        backend=COLLECT_SEARCH_BACKEND_STOCK,
        profiler=profiler,
    )
    assert restore is None


def test_direct_ctree_collect_search_hook_restores_and_falls_back(monkeypatch):
    from lzero.policy.muzero import MuZeroPolicy

    def original_forward_collect(self, data, action_mask=None, *args, **kwargs):
        return {
            "fallback": True,
            "action_mask": action_mask,
            "ready_env_id": kwargs.get("ready_env_id"),
        }

    monkeypatch.setattr(MuZeroPolicy, "_forward_collect", original_forward_collect)
    profiler = _CurvyTronLightZeroPhaseProfiler(enabled=True)
    restore = _install_lightzero_collect_search_backend_hook(
        train_muzero=fake_train_muzero,
        backend=COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT,
        profiler=profiler,
    )

    try:
        policy = SimpleNamespace(
            _cfg=SimpleNamespace(
                model=SimpleNamespace(model_type="conv_context"),
                collect_with_pure_policy=False,
                mcts_ctree=True,
            )
        )
        result = MuZeroPolicy._forward_collect(
            policy,
            FakeTensor(2),
            action_mask=[[1, 1, 1], [1, 1, 1]],
            ready_env_id=[10, 11],
        )
    finally:
        restore()

    assert result["fallback"] is True
    assert result["ready_env_id"] == [10, 11]
    assert profiler.counts["collect_search_backend_fallback_calls"] == 1
    assert profiler.samples["collect_search_backend_fallback_reason"] == [
        "unsupported_model_type:conv_context"
    ]
    assert MuZeroPolicy._forward_collect is original_forward_collect


def test_direct_ctree_collect_search_hook_can_fail_closed_on_fallback(monkeypatch):
    from lzero.policy.muzero import MuZeroPolicy

    def original_forward_collect(self, data, action_mask=None, *args, **kwargs):
        del self, data, action_mask, args, kwargs
        return {"fallback": True}

    monkeypatch.setattr(MuZeroPolicy, "_forward_collect", original_forward_collect)
    profiler = _CurvyTronLightZeroPhaseProfiler(enabled=True)
    restore = _install_lightzero_collect_search_backend_hook(
        train_muzero=fake_train_muzero,
        backend=COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT,
        profiler=profiler,
        allow_fallback=False,
    )

    try:
        policy = SimpleNamespace(
            _cfg=SimpleNamespace(
                model=SimpleNamespace(model_type="conv_context"),
                collect_with_pure_policy=False,
                mcts_ctree=True,
            )
        )
        with pytest.raises(RuntimeError, match="refused hidden fallback"):
            MuZeroPolicy._forward_collect(
                policy,
                FakeTensor(2),
                action_mask=[[1, 1, 1], [1, 1, 1]],
                ready_env_id=[10, 11],
            )
    finally:
        restore()

    assert profiler.counts["collect_search_backend_fallback_calls"] == 1
    assert MuZeroPolicy._forward_collect is original_forward_collect


def test_direct_ctree_collect_search_hook_matches_stock_collect_output_schema(
    monkeypatch,
):
    pytest.importorskip("lzero")
    torch = pytest.importorskip("torch")
    from lzero.policy import select_action
    from lzero.policy.muzero import MuZeroPolicy

    import curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod

    distributions = [[0.0, 5.0, 1.0], [3.0, 0.0, 1.0]]
    root_values = [0.75, -0.25]
    predicted_values = torch.tensor([0.25, -0.5], dtype=torch.float32)
    policy_logits = torch.tensor(
        [[0.0, 2.0, -1.0], [1.5, -2.0, 0.0]],
        dtype=torch.float32,
    )

    class FakeCudaLatent:
        device = SimpleNamespace(type="cuda")

    class FakeCollectModel:
        def eval(self):
            return None

        def initial_inference(self, data):
            return SimpleNamespace(
                latent_state=FakeCudaLatent(),
                reward=torch.zeros(int(data.shape[0]), dtype=torch.float32),
                value=predicted_values,
                policy_logits=policy_logits,
            )

    class FakeRoots:
        def __init__(self, num, legal_actions):
            self.num = num
            self.legal_actions = legal_actions
            self.prepared = False

        def prepare(self, *_args):
            self.prepared = True

        def get_distributions(self):
            return distributions

        def get_values(self):
            return root_values

    class FakeMcts:
        _cfg = SimpleNamespace(num_simulations=4)

        @staticmethod
        def roots(num, legal_actions):
            return FakeRoots(num, legal_actions)

    fake_policy = SimpleNamespace(
        _cfg=SimpleNamespace(
            model=SimpleNamespace(model_type="conv"),
            collect_with_pure_policy=False,
            mcts_ctree=True,
            sampled_algo=False,
            gumbel_algo=False,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.0,
            eps=SimpleNamespace(eps_greedy_exploration_in_collect=True),
        ),
        _collect_model=FakeCollectModel(),
        _mcts_collect=FakeMcts(),
        inverse_scalar_transform_handle=lambda value: value,
    )
    data = torch.zeros((2, 4, 64, 64), dtype=torch.float32)
    action_mask = [[1, 1, 1], [1, 1, 1]]
    ready_env_id = [10, 11]

    def stock_forward_collect(
        self,
        data,
        action_mask=None,
        temperature=1,
        to_play=None,
        epsilon=0.25,
        ready_env_id=None,
        **_kwargs,
    ):
        del self, data, to_play, epsilon
        output = {}
        for row, env_id in enumerate(ready_env_id):
            action_index, entropy = select_action(
                np.asarray(distributions[row], dtype=np.float64),
                temperature=temperature,
                deterministic=True,
            )
            legal_actions = np.flatnonzero(np.asarray(action_mask[row]) == 1)
            output[env_id] = {
                "action": legal_actions[action_index],
                "visit_count_distributions": distributions[row],
                "visit_count_distribution_entropy": entropy,
                "searched_value": root_values[row],
                "predicted_value": predicted_values.numpy()[row],
                "predicted_policy_logits": policy_logits.numpy().tolist()[row],
            }
        return output

    def fake_direct_search(**_kwargs):
        return None

    monkeypatch.setattr(MuZeroPolicy, "_forward_collect", stock_forward_collect)
    monkeypatch.setattr(
        train_mod,
        "_direct_ctree_gpu_latent_search_for_collect",
        fake_direct_search,
    )
    stock_output = MuZeroPolicy._forward_collect(
        fake_policy,
        data,
        action_mask=action_mask,
        temperature=1.0,
        epsilon=0.0,
        ready_env_id=ready_env_id,
    )

    profiler = _CurvyTronLightZeroPhaseProfiler(enabled=True)
    restore = _install_lightzero_collect_search_backend_hook(
        train_muzero=fake_train_muzero,
        backend=COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT,
        profiler=profiler,
    )
    try:
        direct_output = MuZeroPolicy._forward_collect(
            fake_policy,
            data,
            action_mask=action_mask,
            temperature=1.0,
            epsilon=0.0,
            ready_env_id=ready_env_id,
        )
    finally:
        restore()

    assert direct_output.keys() == stock_output.keys()
    for env_id in ready_env_id:
        assert set(direct_output[env_id]) == set(stock_output[env_id])
        assert direct_output[env_id]["action"] == stock_output[env_id]["action"]
        assert direct_output[env_id]["visit_count_distributions"] == (
            stock_output[env_id]["visit_count_distributions"]
        )
        assert direct_output[env_id]["searched_value"] == pytest.approx(
            stock_output[env_id]["searched_value"]
        )
        assert direct_output[env_id]["predicted_value"] == pytest.approx(
            stock_output[env_id]["predicted_value"]
        )
        assert direct_output[env_id]["predicted_policy_logits"] == pytest.approx(
            stock_output[env_id]["predicted_policy_logits"]
        )
    assert profiler.counts["collect_search_backend_direct_ctree_gpu_latent_calls"] == 1
    assert profiler.counts["collect_search_backend_fallback_calls"] == 0
    assert profiler.counts["collect_search_backend_output_fast_path_calls"] == 1
    assert profiler.counts["collect_search_backend_output_rows"] == 2


@pytest.mark.parametrize(
    ("action_mask", "distributions", "expected_actions"),
    [
        (
            [[1, 0, 1], [0, 1, 0], [1, 1, 0]],
            [[2.0, 7.0], [9.0], [5.0, 1.0]],
            [2, 1, 0],
        ),
        (
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[11.0], [12.0], [13.0]],
            [0, 1, 2],
        ),
    ],
)
def test_direct_ctree_collect_search_hook_preserves_masked_raw_visit_contract(
    monkeypatch,
    action_mask,
    distributions,
    expected_actions,
):
    pytest.importorskip("lzero")
    torch = pytest.importorskip("torch")
    from lzero.policy import select_action
    from lzero.policy.muzero import MuZeroPolicy

    import curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod

    batch_size = len(action_mask)
    ready_env_id = [100 + row for row in range(batch_size)]
    root_values = [float(row) + 0.25 for row in range(batch_size)]
    predicted_values = torch.arange(batch_size, dtype=torch.float32) / 10.0
    policy_logits = torch.arange(batch_size * 3, dtype=torch.float32).reshape(
        batch_size,
        3,
    )
    captured_roots = []

    class FakeCudaLatent:
        device = SimpleNamespace(type="cuda")

    class FakeCollectModel:
        def eval(self):
            return None

        def initial_inference(self, data):
            return SimpleNamespace(
                latent_state=FakeCudaLatent(),
                reward=torch.zeros(int(data.shape[0]), dtype=torch.float32),
                value=predicted_values,
                policy_logits=policy_logits,
            )

    class FakeRoots:
        def __init__(self, num, legal_actions):
            self.num = num
            self.legal_actions = legal_actions
            self.prepared = False

        def prepare(self, *_args):
            self.prepared = True

        def get_distributions(self):
            return distributions

        def get_values(self):
            return root_values

    class FakeMcts:
        _cfg = SimpleNamespace(num_simulations=4)

        @staticmethod
        def roots(num, legal_actions):
            roots = FakeRoots(num, legal_actions)
            captured_roots.append(roots)
            return roots

    fake_policy = SimpleNamespace(
        _cfg=SimpleNamespace(
            model=SimpleNamespace(model_type="conv"),
            collect_with_pure_policy=False,
            mcts_ctree=True,
            sampled_algo=False,
            gumbel_algo=False,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.0,
            eps=SimpleNamespace(eps_greedy_exploration_in_collect=True),
        ),
        _collect_model=FakeCollectModel(),
        _mcts_collect=FakeMcts(),
        inverse_scalar_transform_handle=lambda value: value,
    )
    data = torch.zeros((batch_size, 4, 64, 64), dtype=torch.float32)

    def stock_forward_collect(
        self,
        data,
        action_mask=None,
        temperature=1,
        to_play=None,
        epsilon=0.25,
        ready_env_id=None,
        **_kwargs,
    ):
        del self, data, to_play, epsilon
        output = {}
        for row, env_id in enumerate(ready_env_id):
            action_index, entropy = select_action(
                np.asarray(distributions[row], dtype=np.float64),
                temperature=temperature,
                deterministic=True,
            )
            legal_actions = np.flatnonzero(np.asarray(action_mask[row]) == 1)
            output[env_id] = {
                "action": legal_actions[action_index],
                "visit_count_distributions": distributions[row],
                "visit_count_distribution_entropy": entropy,
                "searched_value": root_values[row],
                "predicted_value": predicted_values.numpy()[row],
                "predicted_policy_logits": policy_logits.numpy().tolist()[row],
            }
        return output

    def fake_direct_search(**_kwargs):
        return None

    monkeypatch.setattr(MuZeroPolicy, "_forward_collect", stock_forward_collect)
    monkeypatch.setattr(
        train_mod,
        "_direct_ctree_gpu_latent_search_for_collect",
        fake_direct_search,
    )
    stock_output = MuZeroPolicy._forward_collect(
        fake_policy,
        data,
        action_mask=action_mask,
        temperature=1.0,
        epsilon=0.0,
        ready_env_id=ready_env_id,
    )

    profiler = _CurvyTronLightZeroPhaseProfiler(enabled=True)
    restore = _install_lightzero_collect_search_backend_hook(
        train_muzero=fake_train_muzero,
        backend=COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT,
        profiler=profiler,
    )
    try:
        direct_output = MuZeroPolicy._forward_collect(
            fake_policy,
            data,
            action_mask=action_mask,
            temperature=1.0,
            epsilon=0.0,
            ready_env_id=ready_env_id,
        )
    finally:
        restore()

    expected_legal_actions = [
        list(np.flatnonzero(np.asarray(mask) == 1)) for mask in action_mask
    ]
    assert captured_roots[-1].legal_actions == expected_legal_actions
    assert direct_output.keys() == stock_output.keys()
    for row, env_id in enumerate(ready_env_id):
        assert direct_output[env_id]["action"] == expected_actions[row]
        assert direct_output[env_id]["action"] == stock_output[env_id]["action"]
        assert direct_output[env_id]["action"] in expected_legal_actions[row]
        assert direct_output[env_id]["visit_count_distributions"] == distributions[row]
        assert direct_output[env_id]["visit_count_distributions"] == (
            stock_output[env_id]["visit_count_distributions"]
        )
        assert len(direct_output[env_id]["visit_count_distributions"]) == len(
            expected_legal_actions[row]
        )
        assert direct_output[env_id]["predicted_value"] == pytest.approx(
            stock_output[env_id]["predicted_value"]
        )
        assert direct_output[env_id]["predicted_policy_logits"] == pytest.approx(
            stock_output[env_id]["predicted_policy_logits"]
        )
    assert profiler.counts["collect_search_backend_direct_ctree_gpu_latent_calls"] == 1
    assert profiler.counts["collect_search_backend_fallback_calls"] == 0
    assert profiler.counts.get("collect_search_backend_output_fast_path_calls", 0) == 0
    assert profiler.counts["collect_search_backend_output_rows"] == batch_size


@pytest.mark.parametrize(
    ("action_mask", "ready_env_id", "match"),
    [
        ([[1.0, 0.5, 0.0]], [10], "binary action masks"),
        ([[0.0, 0.0, 0.0]], [10], "zero-action mask"),
        ([[1.0, 1.0, 1.0]], [10, 11], "ready_env_id length"),
    ],
)
def test_direct_ctree_collect_search_hook_rejects_bad_masks(
    monkeypatch,
    action_mask,
    ready_env_id,
    match,
):
    pytest.importorskip("lzero")
    torch = pytest.importorskip("torch")
    from lzero.policy.muzero import MuZeroPolicy

    def stock_forward_collect(self, data, action_mask=None, *args, **kwargs):
        del self, data, action_mask, args, kwargs
        return {}

    monkeypatch.setattr(MuZeroPolicy, "_forward_collect", stock_forward_collect)
    fake_policy = SimpleNamespace(
        _cfg=SimpleNamespace(
            model=SimpleNamespace(model_type="conv"),
            collect_with_pure_policy=False,
            mcts_ctree=True,
            sampled_algo=False,
            gumbel_algo=False,
        )
    )
    data = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
    profiler = _CurvyTronLightZeroPhaseProfiler(enabled=True)
    restore = _install_lightzero_collect_search_backend_hook(
        train_muzero=fake_train_muzero,
        backend=COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT,
        profiler=profiler,
    )
    try:
        with pytest.raises(ValueError, match=match):
            MuZeroPolicy._forward_collect(
                fake_policy,
                data,
                action_mask=action_mask,
                ready_env_id=ready_env_id,
            )
    finally:
        restore()
    assert profiler.counts.get("collect_search_backend_fallback_calls", 0) == 0


def test_direct_ctree_collect_search_hook_fast_path_stochastic_distribution(
    monkeypatch,
):
    pytest.importorskip("lzero")
    torch = pytest.importorskip("torch")
    from lzero.policy import select_action
    from lzero.policy.muzero import MuZeroPolicy

    import curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod

    distribution = [1.0, 3.0, 6.0]
    expected_prob = np.asarray(distribution, dtype=np.float64)
    expected_prob = expected_prob / expected_prob.sum()
    sample_count = 1200

    class FakeCudaLatent:
        device = SimpleNamespace(type="cuda")

    class FakeCollectModel:
        def eval(self):
            return None

        def initial_inference(self, data):
            return SimpleNamespace(
                latent_state=FakeCudaLatent(),
                reward=torch.zeros(int(data.shape[0]), dtype=torch.float32),
                value=torch.zeros(int(data.shape[0]), dtype=torch.float32),
                policy_logits=torch.zeros((int(data.shape[0]), 3), dtype=torch.float32),
            )

    class FakeRoots:
        num = 1

        def prepare(self, *_args):
            return None

        def get_distributions(self):
            return [distribution]

        def get_values(self):
            return [0.0]

    class FakeMcts:
        _cfg = SimpleNamespace(num_simulations=4)

        @staticmethod
        def roots(num, legal_actions):
            del num, legal_actions
            return FakeRoots()

    fake_policy = SimpleNamespace(
        _cfg=SimpleNamespace(
            model=SimpleNamespace(model_type="conv"),
            collect_with_pure_policy=False,
            mcts_ctree=True,
            sampled_algo=False,
            gumbel_algo=False,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.0,
            eps=SimpleNamespace(eps_greedy_exploration_in_collect=False),
        ),
        _collect_model=FakeCollectModel(),
        _mcts_collect=FakeMcts(),
        inverse_scalar_transform_handle=lambda value: value,
    )
    data = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
    action_mask = [[1, 1, 1]]
    ready_env_id = [10]

    def stock_forward_collect(
        self,
        data,
        action_mask=None,
        temperature=1,
        to_play=None,
        epsilon=0.25,
        ready_env_id=None,
        **_kwargs,
    ):
        del self, data, action_mask, to_play, epsilon
        action_index, entropy = select_action(
            np.asarray(distribution, dtype=np.float64),
            temperature=temperature,
            deterministic=False,
        )
        return {
            ready_env_id[0]: {
                "action": action_index,
                "visit_count_distributions": distribution,
                "visit_count_distribution_entropy": entropy,
                "searched_value": 0.0,
                "predicted_value": 0.0,
                "predicted_policy_logits": [0.0, 0.0, 0.0],
            }
        }

    def fake_direct_search(**_kwargs):
        return None

    monkeypatch.setattr(MuZeroPolicy, "_forward_collect", stock_forward_collect)
    monkeypatch.setattr(
        train_mod,
        "_direct_ctree_gpu_latent_search_for_collect",
        fake_direct_search,
    )

    stock_counts = np.zeros(3, dtype=np.int64)
    for seed in range(sample_count):
        np.random.seed(seed)
        action = MuZeroPolicy._forward_collect(
            fake_policy,
            data,
            action_mask=action_mask,
            temperature=1.0,
            epsilon=0.0,
            ready_env_id=ready_env_id,
        )[10]["action"]
        stock_counts[int(action)] += 1

    profiler = _CurvyTronLightZeroPhaseProfiler(enabled=True)
    restore = _install_lightzero_collect_search_backend_hook(
        train_muzero=fake_train_muzero,
        backend=COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT,
        profiler=profiler,
    )
    direct_counts = np.zeros(3, dtype=np.int64)
    try:
        for seed in range(sample_count):
            np.random.seed(seed)
            action = MuZeroPolicy._forward_collect(
                fake_policy,
                data,
                action_mask=action_mask,
                temperature=1.0,
                epsilon=0.0,
                ready_env_id=ready_env_id,
            )[10]["action"]
            direct_counts[int(action)] += 1
    finally:
        restore()

    stock_freq = stock_counts / sample_count
    direct_freq = direct_counts / sample_count
    assert np.max(np.abs(stock_freq - expected_prob)) < 0.06
    assert np.max(np.abs(direct_freq - expected_prob)) < 0.06
    assert np.max(np.abs(stock_freq - direct_freq)) < 0.08
    assert profiler.counts["collect_search_backend_output_fast_path_calls"] == sample_count
    assert profiler.counts["collect_search_backend_fallback_calls"] == 0


def test_curvytron_compact_output_uses_mcts_root_fallback_for_profile_steps():
    compact = _compact_train_result_for_output(
        {
            "ok": True,
            "status": "profile_stopped",
            "mode": "profile",
            "called_train_muzero": True,
            "trainer_entrypoint": "lzero.entry.train_muzero",
            "command": {
                "env_variant": "source_state_fixed_opponent",
                "env_manager_type": "curvyzero_batched_profile",
                "policy_observation_backend": "cpu_oracle",
                "collect_search_backend": "direct_ctree_gpu_latent",
                "collect_search_ctree_backend": "flat_a3",
                "policy_observation_contract_id": "curvyzero_policy_observation_surface/v1",
                "observation_contract": {
                    "contract_id": "curvyzero_policy_observation_surface/v1",
                    "surface_label": "browser_lines+simple_symbols",
                    "stack_shape": [4, 64, 64],
                    "single_frame_shape": [1, 64, 64],
                    "raw_dtype": "uint8",
                    "model_dtype": "float32",
                },
                "source_state_trail_render_mode": "browser_lines",
                "source_state_bonus_render_mode": "simple_symbols",
                "policy_trail_render_mode": "browser_lines",
                "policy_bonus_render_mode": "simple_symbols",
                "disable_death_for_profile": True,
                "opponent_death_mode": "normal",
                "exploration_bonus": {"mode": "none"},
                "skip_lightzero_eval_in_profile": True,
                "lightzero_eval_freq": 0,
            },
            "phase_profile": {
                "timers_sec": {
                    "train_muzero_wall_sec": 10.0,
                    "collect_search_backend_recurrent_inference_sec": 1.25,
                    "collect_search_backend_model_output_d2h_sec": 0.5,
                    "collect_search_backend_model_output_listify_sec": 0.25,
                    "collect_search_backend_flat_payload_sec": 0.125,
                },
                "counts": {
                    "env_steps_collected": 0,
                    "mcts_search_root_sum": 317,
                    "mcts_search_calls": 4,
                    "mcts_search_node_budget_sum": 1268,
                    "collect_search_backend_direct_ctree_gpu_latent_calls": 4,
                    "collect_search_backend_fallback_calls": 0,
                    "collect_search_backend_output_fast_path_calls": 4,
                    "collect_search_backend_recurrent_inference_calls": 16,
                    "collect_search_backend_model_output_d2h_bytes": 4096,
                    "evaluator_eval_calls": 0,
                    "evaluator_eval_skipped_calls": 0,
                },
                "samples": {
                    "collect_search_backend": ["direct_ctree_gpu_latent"],
                    "collect_search_ctree_backend": ["flat_a3"],
                },
            },
            "collect_search_backend_proof": {
                "schema_id": "curvyzero_collect_search_backend_proof/v0",
                "requested_backend": "direct_ctree_gpu_latent",
                "requested_ctree_backend": "flat_a3",
                "fallback_policy": "allow_profile_fallback",
                "observed_collect_search_backends": ["direct_ctree_gpu_latent"],
                "observed_collect_search_ctree_backends": ["flat_a3"],
                "direct_ctree_gpu_latent_calls": 4,
                "fallback_calls": 0,
                "output_rows": 317,
                "recurrent_inference_calls": 16,
                "model_output_d2h_bytes": 4096,
            },
        }
    )

    identity = compact["semantic_identity"]
    assert identity["observation_raw_dtype"] == "uint8"
    assert identity["observation_model_dtype"] == "float32"
    assert identity["lightzero_to_play_mode"] == "fixed_opponent_minus_one"
    assert identity["scalar_materialization_semantics"] == (
        "batched_profile_manager_materializes_lightzero_scalar_timesteps"
    )
    assert identity["collect_search_backend"] == "direct_ctree_gpu_latent"
    assert identity["collect_search_ctree_backend"] == "flat_a3"
    assert compact["command"]["collect_search_backend"] == "direct_ctree_gpu_latent"
    assert compact["command"]["collect_search_ctree_backend"] == "flat_a3"
    assert compact["counts"]["env_steps_collected"] == 317
    assert compact["counts"]["env_steps_collected_raw"] == 0
    assert compact["counts"]["collect_search_backend_direct_ctree_gpu_latent_calls"] == 4
    assert compact["counts"]["collect_search_backend_fallback_calls"] == 0
    assert compact["counts"]["collect_search_backend_output_fast_path_calls"] == 4
    assert compact["counts"]["collect_search_backend_model_output_d2h_bytes"] == 4096
    assert compact["timers_sec"]["collect_search_backend_recurrent_inference"] == 1.25
    assert compact["timers_sec"]["collect_search_backend_model_output_d2h"] == 0.5
    assert compact["timers_sec"]["collect_search_backend_model_output_listify"] == 0.25
    assert compact["timers_sec"]["collect_search_backend_flat_payload"] == 0.125
    assert compact["search_backend_proof"] == {
        "schema_id": "curvyzero_collect_search_backend_proof/v0",
        "requested_backend": "direct_ctree_gpu_latent",
        "requested_ctree_backend": "flat_a3",
        "fallback_policy": "allow_profile_fallback",
        "observed_collect_search_backends": ["direct_ctree_gpu_latent"],
        "observed_collect_search_ctree_backends": ["flat_a3"],
        "direct_ctree_gpu_latent_calls": 4,
        "fallback_calls": 0,
        "output_rows": 317,
        "flat_payload_timer_present": True,
    }
    assert compact["derived"]["collect_search_backend_recurrent_batch_mean"] == 79.25
    assert (
        compact["counts"]["env_steps_collected_source"]
        == "mcts_search_root_sum_profile_fallback"
    )
    assert compact["counts"]["env_steps_collected_effective"] == 317
    assert compact["counts"]["env_steps_collected_uses_fallback"] is True
    assert compact["derived"]["steps_per_sec"] == 31.7
    assert compact["derived"]["steps_per_sec_source"] == (
        "mcts_search_root_sum_profile_fallback"
    )
    assert compact["derived"]["steps_per_sec_currency"] == (
        "stock_train_muzero_profile_mcts_roots_per_sec_fallback"
    )
    assert compact["derived"]["steps_per_sec_uses_fallback_denominator"] is True
    assert compact["semantic_identity"]["speed_currency"] == (
        "stock_train_muzero_profile_mcts_roots_per_sec_fallback"
    )


def test_curvytron_compact_output_prefers_raw_collector_envstep_delta_over_mcts_roots():
    compact = _compact_train_result_for_output(
        {
            "ok": True,
            "status": "profile_stopped",
            "command": {
                "skip_lightzero_eval_in_profile": True,
                "lightzero_eval_freq": 0,
            },
            "phase_profile": {
                "timers_sec": {"train_muzero_wall_sec": 10.0},
                "counts": {
                    "env_steps_collected": 123,
                    "mcts_search_root_sum": 317,
                    "mcts_search_calls": 4,
                    "evaluator_eval_calls": 0,
                    "evaluator_eval_skipped_calls": 1,
                },
            },
        }
    )

    assert compact["counts"]["env_steps_collected"] == 123
    assert compact["counts"]["env_steps_collected_raw"] == 123
    assert compact["counts"]["env_steps_collected_effective"] == 123
    assert compact["counts"]["env_steps_collected_source"] == "collector_envstep_delta"
    assert compact["counts"]["env_steps_collected_uses_fallback"] is False
    assert compact["derived"]["steps_per_sec"] == 12.3
    assert compact["derived"]["steps_per_sec_source"] == "collector_envstep_delta"
    assert compact["derived"]["steps_per_sec_currency"] == (
        "stock_train_muzero_profile_env_steps_per_sec"
    )
    assert compact["derived"]["steps_per_sec_uses_fallback_denominator"] is False


def test_curvytron_compact_output_does_not_use_mcts_root_fallback_when_eval_ran():
    compact = _compact_train_result_for_output(
        {
            "ok": True,
            "status": "profile_stopped",
            "command": {
                "skip_lightzero_eval_in_profile": False,
                "lightzero_eval_freq": 0,
            },
            "phase_profile": {
                "timers_sec": {"train_muzero_wall_sec": 10.0},
                "counts": {
                    "env_steps_collected": 0,
                    "mcts_search_root_sum": 317,
                    "mcts_search_calls": 4,
                    "evaluator_eval_calls": 1,
                    "evaluator_eval_skipped_calls": 0,
                },
            },
        }
    )

    assert compact["counts"]["env_steps_collected"] == 0
    assert compact["counts"]["env_steps_collected_raw"] == 0
    assert compact["counts"]["env_steps_collected_source"] == "collector_envstep_delta"
    assert compact["counts"]["env_steps_collected_uses_fallback"] is False
    assert compact["derived"]["steps_per_sec"] == 0.0
    assert compact["derived"]["steps_per_sec_currency"] == (
        "stock_train_muzero_profile_env_steps_per_sec"
    )
    assert compact["derived"]["steps_per_sec_uses_fallback_denominator"] is False
