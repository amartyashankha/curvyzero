import json
import os
from pathlib import Path

from curvyzero.infra.modal import lightzero_curvytron_run_status as status_mod
from curvyzero.infra.modal import (
    lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod,
)
from curvyzero.training import lightzero_checkpoints as lz_checkpoints


def _attempt_train_root(tmp_path: Path, run_id: str, attempt_id: str) -> Path:
    return train_mod.runs.volume_path(
        tmp_path,
        train_mod.runs.attempt_train_ref(train_mod.TASK_ID, run_id, attempt_id),
    )


def _write_iteration_checkpoint(train_root: Path, exp_dir_name: str, iteration: int) -> Path:
    checkpoint = train_root / exp_dir_name / "ckpt" / f"iteration_{iteration}.pth.tar"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint-{iteration}".encode("utf-8"))
    return checkpoint


def _write_resume_state(train_root: Path, exp_dir_name: str, iteration: int) -> Path:
    state = (
        train_root
        / exp_dir_name
        / train_mod.LIGHTZERO_RESUME_STATE_DIRNAME
        / f"iteration_{iteration}.resume_state.pkl"
    )
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_bytes(f"resume-state-{iteration}".encode("utf-8"))
    return state


def test_lightzero_checkpoint_helpers_list_timestamped_siblings(tmp_path):
    train_root = tmp_path / "train"
    (train_root / "lightzero_exp").mkdir(parents=True)
    (train_root / "lightzero_exp_260513_123802").mkdir()
    (train_root / "not_lightzero_exp").mkdir()

    roots = lz_checkpoints.lightzero_exp_sibling_roots(train_root / "lightzero_exp")

    assert [root.name for root in roots] == [
        "lightzero_exp",
        "lightzero_exp_260513_123802",
    ]
    assert [path.name for path in lz_checkpoints.lightzero_exp_checkpoint_dirs(roots[0])] == [
        "ckpt",
        "ckpt",
    ]


def test_lightzero_checkpoint_name_parsers_reject_invalid_names():
    assert lz_checkpoints.lightzero_iteration_from_checkpoint_name(
        "iteration_180000.pth.tar"
    ) == 180000
    assert (
        lz_checkpoints.lightzero_iteration_from_resume_state_name(
            "iteration_180000.resume_state.pkl"
        )
        == 180000
    )
    assert lz_checkpoints.lightzero_iteration_from_checkpoint_name("iteration_x.pth.tar") is None
    assert lz_checkpoints.lightzero_iteration_from_checkpoint_name("ckpt_1.pth.tar") is None
    assert (
        lz_checkpoints.lightzero_iteration_from_resume_state_name(
            "iteration_1.resume.pkl"
        )
        is None
    )


def test_lightzero_checkpoint_candidate_helpers_select_latest_by_iteration_and_mtime(
    tmp_path,
):
    train_root = tmp_path / "train"
    older = _write_iteration_checkpoint(train_root, "lightzero_exp", 10)
    newer_same_iteration = _write_iteration_checkpoint(
        train_root,
        "lightzero_exp_260513_123802",
        10,
    )
    lower_iteration = _write_iteration_checkpoint(
        train_root,
        "lightzero_exp_260513_123803",
        9,
    )
    os.utime(older, (1, 1))
    os.utime(newer_same_iteration, (2, 2))
    os.utime(lower_iteration, (3, 3))

    latest = lz_checkpoints.latest_lightzero_iteration_checkpoint_from_dirs(
        lz_checkpoints.lightzero_exp_checkpoint_dirs(train_root / "lightzero_exp")
    )

    assert latest is not None
    assert latest.iteration == 10
    assert latest.path == newer_same_iteration
    assert latest.exp_dir_name == "lightzero_exp_260513_123802"


def test_lightzero_checkpoint_candidate_helpers_can_skip_empty_files(tmp_path):
    train_root = tmp_path / "train"
    empty = _write_iteration_checkpoint(train_root, "lightzero_exp", 20)
    empty.write_bytes(b"")
    non_empty = _write_iteration_checkpoint(
        train_root,
        "lightzero_exp_260513_123802",
        19,
    )

    candidates = lz_checkpoints.collect_lightzero_iteration_checkpoints(
        lz_checkpoints.lightzero_exp_checkpoint_dirs(train_root / "lightzero_exp"),
        require_non_empty=True,
    )

    assert [candidate.path for candidate in candidates] == [non_empty]


def test_lightzero_resume_state_candidates_select_exact_iteration_by_mtime_and_size(
    tmp_path,
):
    train_root = tmp_path / "train"
    older = _write_resume_state(train_root, "lightzero_exp", 180000)
    latest = _write_resume_state(train_root, "lightzero_exp_260513_123802", 180000)
    wrong_iteration = _write_resume_state(
        train_root,
        "lightzero_exp_260513_123803",
        179999,
    )
    os.utime(older, (1, 1))
    os.utime(latest, (2, 2))
    os.utime(wrong_iteration, (3, 3))

    candidates = lz_checkpoints.collect_lightzero_resume_state_candidates(
        lz_checkpoints.lightzero_exp_resume_state_dirs(train_root / "lightzero_exp"),
        iteration=180000,
    )
    selected = lz_checkpoints.latest_lightzero_resume_state_candidate(candidates)

    assert [candidate.path for candidate in candidates] == [older, latest]
    assert selected is not None
    assert selected.path == latest
    assert selected.iteration == 180000
    assert selected.state_name == "iteration_180000.resume_state.pkl"
    assert selected.exp_dir_name == "lightzero_exp_260513_123802"


def test_lightzero_resume_state_candidates_skip_empty_and_missing_dirs(tmp_path):
    train_root = tmp_path / "train"
    empty = _write_resume_state(train_root, "lightzero_exp", 10)
    empty.write_bytes(b"")
    non_empty = _write_resume_state(train_root, "lightzero_exp_260513_123802", 10)
    missing_dir = train_root / "missing" / train_mod.LIGHTZERO_RESUME_STATE_DIRNAME

    candidates = lz_checkpoints.collect_lightzero_resume_state_candidates(
        [
            train_root / "lightzero_exp" / train_mod.LIGHTZERO_RESUME_STATE_DIRNAME,
            missing_dir,
            train_root
            / "lightzero_exp_260513_123802"
            / train_mod.LIGHTZERO_RESUME_STATE_DIRNAME,
        ],
        iteration=10,
    )

    assert [candidate.path for candidate in candidates] == [non_empty]


def test_progress_latest_uses_timestamped_lightzero_exp_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    train_root = _attempt_train_root(tmp_path, "run-progress-timestamped", "attempt-a")
    exp_name = train_root / "lightzero_exp"
    _write_iteration_checkpoint(train_root, "lightzero_exp", 0)
    _write_iteration_checkpoint(train_root, "lightzero_exp_260513_123802", 180000)

    class FakeLearner:
        train_iter = 180000

    progress = train_mod._write_checkpoint_progress_latest(
        run_id="run-progress-timestamped",
        attempt_id="attempt-a",
        attempt_train_root=train_root,
        exp_name=exp_name,
        learner=FakeLearner(),
        started_monotonic=0.0,
    )

    payload = json.loads(Path(progress["path"]).read_text(encoding="utf-8"))
    assert payload["iteration"] == 180000
    assert payload["checkpoint_name"] == "iteration_180000.pth.tar"
    assert "lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar" in payload[
        "checkpoint_ref"
    ]


def test_checkpoint_progress_payload_prefers_checkpoint_iteration():
    payload = train_mod._build_checkpoint_progress_latest_payload(
        run_id="run-progress-payload",
        attempt_id="attempt-a",
        checkpoint={
            "iteration": 180000,
            "checkpoint_name": "iteration_180000.pth.tar",
            "checkpoint_ref": "training/run/attempt/train/lightzero_exp/ckpt/iteration_180000.pth.tar",
        },
        learner_iteration=17,
        elapsed_sec=12.3456789,
        timestamp="2026-05-13T00:00:00Z",
        source="BaseLearner.save_checkpoint",
    )

    assert payload["iteration"] == 180000
    assert payload["learner_train_iter"] == 17
    assert payload["elapsed_sec"] == 12.345679
    assert payload["checkpoint_name"] == "iteration_180000.pth.tar"
    assert payload["checkpoint_ref"].endswith("iteration_180000.pth.tar")
    assert payload["timestamp"] == "2026-05-13T00:00:00Z"
    assert payload["updated_at"] == "2026-05-13T00:00:00Z"
    assert payload["source"] == "BaseLearner.save_checkpoint"


def test_checkpoint_progress_payload_falls_back_to_learner_iteration():
    payload = train_mod._build_checkpoint_progress_latest_payload(
        run_id="run-progress-payload",
        attempt_id="attempt-a",
        checkpoint=None,
        learner_iteration=23,
        elapsed_sec=-1.0,
        timestamp="2026-05-13T00:01:00Z",
        source="SaveCkptHook.__call__",
    )

    assert payload["iteration"] == 23
    assert payload["learner_train_iter"] == 23
    assert payload["elapsed_sec"] == 0.0
    assert payload["source"] == "SaveCkptHook.__call__"
    assert "checkpoint_ref" not in payload
    assert "checkpoint_name" not in payload


def test_auto_resume_selects_timestamped_lightzero_exp_checkpoint(monkeypatch, tmp_path):
    class FakeVolume:
        def reload(self) -> None:
            return None

    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", FakeVolume())
    run_id = "run-resume-timestamped"
    attempt_id = "attempt-a"
    train_root = _attempt_train_root(tmp_path, run_id, attempt_id)
    _write_iteration_checkpoint(train_root, "lightzero_exp", 0)
    _write_iteration_checkpoint(train_root, "lightzero_exp_260513_123802", 180000)

    exp_name_ref = (
        train_mod.runs.attempt_train_ref(train_mod.TASK_ID, run_id, attempt_id)
        / "lightzero_exp"
    )
    resume = train_mod._prepare_lightzero_auto_resume(
        run_id=run_id,
        attempt_id=attempt_id,
        exp_name_ref=exp_name_ref,
    )

    assert resume["found"] is True
    assert resume["iteration"] == 180000
    assert "lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar" in resume[
        "checkpoint_ref"
    ]


def test_auto_resume_considers_prior_attempts_and_run_mirror(monkeypatch, tmp_path):
    class FakeVolume:
        def reload(self) -> None:
            return None

    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", FakeVolume())
    run_id = "run-resume-prior-and-mirror"
    attempt_id = "attempt-current"
    current_train_root = _attempt_train_root(tmp_path, run_id, attempt_id)
    prior_train_root = _attempt_train_root(tmp_path, run_id, "attempt-prior")
    _write_iteration_checkpoint(current_train_root, "lightzero_exp", 10)
    _write_iteration_checkpoint(prior_train_root, "lightzero_exp_260513_123802", 50)
    mirror_checkpoint = (
        train_mod.runs.volume_path(
            tmp_path,
            train_mod.runs.checkpoints_root_ref(train_mod.TASK_ID, run_id),
        )
        / "lightzero"
        / "iteration_75.pth.tar"
    )
    mirror_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    mirror_checkpoint.write_bytes(b"mirror-checkpoint")

    mirror_sidecar = (
        train_mod.runs.volume_path(
            tmp_path,
            train_mod.runs.checkpoints_root_ref(train_mod.TASK_ID, run_id),
        )
        / train_mod.LIGHTZERO_RESUME_STATE_DIRNAME
        / "iteration_75.resume_state.pkl"
    )
    mirror_sidecar.parent.mkdir(parents=True, exist_ok=True)
    mirror_sidecar.write_bytes(b"mirror-sidecar")

    exp_name_ref = (
        train_mod.runs.attempt_train_ref(train_mod.TASK_ID, run_id, attempt_id)
        / "lightzero_exp"
    )
    resume = train_mod._prepare_lightzero_auto_resume(
        run_id=run_id,
        attempt_id=attempt_id,
        exp_name_ref=exp_name_ref,
    )

    assert resume["found"] is True
    assert resume["iteration"] == 75
    assert resume["source_kind"] == "run_checkpoint_mirror"
    assert resume["resume_state_found"] is True
    assert resume["resume_state_source_kind"] == "run_resume_state_mirror"
    assert resume["candidate_iterations"] == [10, 50, 75]


def test_auto_resume_ignores_empty_checkpoint_and_selects_matching_sidecar(
    monkeypatch,
    tmp_path,
):
    class FakeVolume:
        def reload(self) -> None:
            return None

    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", FakeVolume())
    run_id = "run-resume-empty-checkpoint"
    attempt_id = "attempt-a"
    train_root = _attempt_train_root(tmp_path, run_id, attempt_id)
    empty = _write_iteration_checkpoint(train_root, "lightzero_exp_260513_123802", 200)
    empty.write_bytes(b"")
    _write_iteration_checkpoint(train_root, "lightzero_exp", 100)
    _write_resume_state(train_root, "lightzero_exp_260513_123802", 200)
    _write_resume_state(train_root, "lightzero_exp", 100)

    exp_name_ref = (
        train_mod.runs.attempt_train_ref(train_mod.TASK_ID, run_id, attempt_id)
        / "lightzero_exp"
    )
    resume = train_mod._prepare_lightzero_auto_resume(
        run_id=run_id,
        attempt_id=attempt_id,
        exp_name_ref=exp_name_ref,
    )

    assert resume["found"] is True
    assert resume["iteration"] == 100
    assert resume["resume_state_found"] is True
    assert resume["resume_state_iteration"] == 100
    assert resume["candidate_iterations"] == [100]


def test_resume_sidecar_scans_timestamped_lightzero_exp_state_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    run_id = "run-sidecar-timestamped"
    attempt_id = "attempt-a"
    train_root = _attempt_train_root(tmp_path, run_id, attempt_id)
    state_path = (
        train_root
        / "lightzero_exp_260513_123802"
        / train_mod.LIGHTZERO_RESUME_STATE_DIRNAME
        / "iteration_180000.resume_state.pkl"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(b"resume-state")

    exp_name_ref = (
        train_mod.runs.attempt_train_ref(train_mod.TASK_ID, run_id, attempt_id)
        / "lightzero_exp"
    )
    sidecar = train_mod._find_lightzero_resume_sidecar(
        run_id=run_id,
        attempt_id=attempt_id,
        exp_name_ref=exp_name_ref,
        iteration=180000,
    )

    assert sidecar["resume_state_found"] is True
    assert sidecar["resume_state_iteration"] == 180000
    assert "lightzero_exp_260513_123802" in sidecar["resume_state_ref"]


def test_resume_sidecar_save_uses_timestamped_lightzero_exp_checkpoint(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "_lightzero_rng_state", lambda: {})
    saved_payload = {}

    def fake_save_resume_state(path: Path, payload: dict) -> None:
        saved_payload.update(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"resume-state")

    monkeypatch.setattr(train_mod, "_save_resume_state", fake_save_resume_state)
    run_id = "run-sidecar-save-timestamped"
    attempt_id = "attempt-a"
    train_root = _attempt_train_root(tmp_path, run_id, attempt_id)
    checkpoint = _write_iteration_checkpoint(
        train_root,
        "lightzero_exp_260513_123802",
        180000,
    )

    class FakeLearner:
        train_iter = 180000
        collector_envstep = 0

    result = train_mod._save_lightzero_resume_sidecar_state(
        run_id=run_id,
        attempt_id=attempt_id,
        exp_name=train_root / "lightzero_exp",
        holder={},
        learner=FakeLearner(),
    )

    sidecar_path = (
        train_root
        / "lightzero_exp_260513_123802"
        / train_mod.LIGHTZERO_RESUME_STATE_DIRNAME
        / "iteration_180000.resume_state.pkl"
    )
    mirror_path = (
        train_mod.runs.volume_path(
            tmp_path,
            train_mod.runs.checkpoints_root_ref(train_mod.TASK_ID, run_id),
        )
        / train_mod.LIGHTZERO_RESUME_STATE_DIRNAME
        / "iteration_180000.resume_state.pkl"
    )

    assert result["saved"] is True
    assert "lightzero_exp_260513_123802" in result["ref"]
    assert sidecar_path.is_file()
    assert mirror_path.is_file()
    assert saved_payload["checkpoint_path"] == str(checkpoint)


def test_checkpoint_eval_poller_scans_timestamped_lightzero_exp_dirs(
    tmp_path, monkeypatch
):
    class FakeVolume:
        def reload(self) -> None:
            return None

    class FakeCall:
        object_id = "fc-poller-timestamped"

        def get(self):
            return {"ok": True}

    class FakeFunction:
        def __init__(self) -> None:
            self.calls = []

        def spawn(self, **kwargs):
            self.calls.append(kwargs)
            return FakeCall()

    fake_eval_function = FakeFunction()
    fake_gif_function = FakeFunction()
    monkeypatch.setattr(train_mod, "RUNS_MOUNT", tmp_path)
    monkeypatch.setattr(train_mod, "runs_volume", FakeVolume())
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect",
        fake_eval_function,
    )
    monkeypatch.setattr(
        train_mod,
        "lightzero_curvytron_visual_survival_checkpoint_selfplay_gif",
        fake_gif_function,
    )

    run_id = "run-poller-timestamped"
    attempt_id = "attempt-a"
    train_root = _attempt_train_root(tmp_path, run_id, attempt_id)
    _write_iteration_checkpoint(train_root, "lightzero_exp", 0)
    _write_iteration_checkpoint(train_root, "lightzero_exp_260513_123802", 180000)
    (train_root / "summary.json").write_text('{"ok": true}', encoding="utf-8")

    command = train_mod._checkpoint_eval_poller_command(
        seed=3,
        source_max_steps=32,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        natural_bonus_spawn=False,
        background_eval_enabled=True,
        background_eval_compute="cpu",
        background_eval_id_prefix="live_checkpoint",
        background_eval_seed_count=1,
        background_eval_seed_rng_seed=0,
        background_eval_max_steps=32,
        background_eval_step_detail_limit=2,
        background_eval_num_simulations=4,
        background_eval_batch_size=8,
        background_gif_enabled=True,
        background_gif_seed_offset=10_000,
        background_gif_max_steps=32,
        background_gif_frame_stride=2,
        background_gif_fps=12.0,
        background_gif_scale=3,
    )

    exp_name_ref = (
        train_mod.runs.attempt_train_ref(train_mod.TASK_ID, run_id, attempt_id)
        / "lightzero_exp"
    ).as_posix()
    result = train_mod._run_checkpoint_eval_poller(
        run_id=run_id,
        attempt_id=attempt_id,
        exp_name_ref=exp_name_ref,
        poll_interval_sec=0.01,
        stable_polls=0,
        max_runtime_sec=1.0,
        idle_after_train_done_sec=0.0,
        command=command,
    )

    assert result["scheduled_count"] >= 1
    assert any(
        call["checkpoint_ref"].endswith(
            "lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar"
        )
        for call in fake_eval_function.calls
    )
    assert any(
        call["checkpoint_ref"].endswith(
            "lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar"
        )
        for call in fake_gif_function.calls
    )
    assert result["gif_scheduled_count"] >= 1
    assert result["gif_completed_count"] >= 1


def test_run_status_checkpoint_summary_scans_timestamped_lightzero_exp_dirs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(status_mod, "RUNS_MOUNT", tmp_path)
    run_id = "run-status-timestamped"
    attempt_id = "attempt-a"
    train_root = status_mod.runs.volume_path(
        tmp_path,
        status_mod.runs.attempt_train_ref(status_mod.TASK_ID, run_id, attempt_id),
    )
    _write_iteration_checkpoint(train_root, "lightzero_exp", 0)
    _write_iteration_checkpoint(train_root, "lightzero_exp_260513_123802", 180000)

    summary = status_mod._checkpoint_summary(run_id, attempt_id)

    assert summary["latest_checkpoint"] == "iteration_180000"
    assert summary["checkpoint_count"] == 2
