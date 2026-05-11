"""Exact installed LightZero Atari Pong MuZero reproduction wrapper.

Dry validation:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction --mode dry

This wrapper is intentionally narrow. It uses the installed ``LightZero==0.2.0``
package surface, imports ``zoo.atari.config.atari_muzero_config``, and calls
``lzero.entry.train_muzero`` in train mode with the package's stock
``max_env_step=200000``. The default exact path mutates only
``main_config.exp_name`` so artifacts land under the Modal Volume. Train mode
sets ``exp_name`` to a relative Volume ref and runs from ``/runs`` because
LightZero/DI-engine may prepend ``./`` to ``exp_name`` when saving checkpoints.
Passing ``max_env_step_override`` switches the run to a clearly labeled
faithful-short rehearsal: same installed config, same exp_name patch, shorter
trainer ``max_env_step`` argument.
Passing ``save_ckpt_after_iter_override`` additionally changes only the
checkpoint cadence for future short rehearsals.

Passing a positive ``survival_reward_per_step`` is an explicit shaped-objective
ablation. It uses a separate registered env type,
``atari_lightzero_survival_shaped``, and requires run/attempt ids containing
``survival-shaped`` so it cannot be reported as stock/control Pong.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import inspect
import json
import os
import subprocess
import threading
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_atari_rom_image import (
    ATARI_ROM_LICENSE_NOTICE,
    build_lightzero_atari_rom_image,
)
from curvyzero.infra.modal.lightzero_pong_dry_config_smoke import LIGHTZERO_VERSION
from curvyzero.training.lightzero_atari_survival_env import (
    LIGHTZERO_ATARI_SURVIVAL_ENV_TYPE,
    REWARD_SHAPING_SCHEMA_ID,
)

APP_NAME = "curvyzero-lightzero-pong-exact-reproduction"
TASK_ID = "lightzero-official-visual-pong"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_MODE = "dry"
DEFAULT_COMPUTE = "cpu"
DEFAULT_SEED = 0
DEFAULT_RUN_ID = "lz-visual-pong-exact-installed-0.2.0-s0"
DEFAULT_ATTEMPT_ID = "dry-exact-installed-0.2.0-stock-surface"
DEFAULT_PROGRESS_INTERVAL_SEC = 300
DEFAULT_PROFILE_PHASES = False
DEFAULT_GPU_SAMPLE_INTERVAL_SEC = 15.0
DEFAULT_SURVIVAL_REWARD_PER_STEP = 0.0
EXACT_MAX_ENV_STEP = 200000
CHEAP_GPU_RESOURCE = ["L4", "T4"]
H100_GPU_RESOURCE = "H100"
H100X4_GPU_RESOURCE = "H100:4"
GPU_RESOURCE_BY_COMPUTE = {
    "gpu-l4-t4": CHEAP_GPU_RESOURCE,
    "gpu-l4-t4-cpu16": CHEAP_GPU_RESOURCE,
    "gpu-l4-t4-cpu40": CHEAP_GPU_RESOURCE,
    "gpu-h100": H100_GPU_RESOURCE,
    "gpu-h100-cpu16": H100_GPU_RESOURCE,
    "gpu-h100-cpu40": H100_GPU_RESOURCE,
    "gpu-h100x4": H100X4_GPU_RESOURCE,
}
CPU_COUNT_BY_COMPUTE = {
    "cpu": 1.0,
    "gpu-l4-t4": 8.0,
    "gpu-l4-t4-cpu16": 16.0,
    "gpu-l4-t4-cpu40": 40.0,
    "gpu-h100": 8.0,
    "gpu-h100-cpu16": 16.0,
    "gpu-h100-cpu40": 40.0,
    "gpu-h100x4": 8.0,
}
COMPUTE_CHOICES = ("cpu", *GPU_RESOURCE_BY_COMPUTE)

image = (
    build_lightzero_atari_rom_image(lightzero_version=LIGHTZERO_VERSION)
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name("curvyzero-runs", create_if_missing=True)

app = modal.App(APP_NAME)


def _version_or_missing(*packages: str) -> str:
    for package in packages:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return "missing"


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


class _PhaseTimer:
    def __init__(self, profiler: "_LightZeroPhaseProfiler", name: str):
        self.profiler = profiler
        self.name = name
        self.started = 0.0

    def __enter__(self) -> "_PhaseTimer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.profiler.add_time(self.name, time.perf_counter() - self.started)


class _LightZeroPhaseProfiler:
    """Tiny opt-in profiler for LightZero's single-container training phases."""

    def __init__(self, *, enabled: bool, gpu_sample_interval_sec: float):
        self.enabled = enabled
        self.gpu_sample_interval_sec = gpu_sample_interval_sec
        self.started = time.perf_counter()
        self.timers: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.gpu_samples: list[dict[str, Any]] = []
        self.gpu_sample_errors: list[str] = []
        self.notes: list[str] = []
        self.installed_hooks: list[str] = []
        self.candidate_hooks: list[dict[str, Any]] = []
        self.deep_hook_discovery_notes: list[str] = []

    def timer(self, name: str) -> _PhaseTimer:
        return _PhaseTimer(self, name)

    def add_time(self, name: str, elapsed_sec: float) -> None:
        self.timers[name] = self.timers.get(name, 0.0) + elapsed_sec

    def add_count(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + int(amount)

    def add_note(self, note: str) -> None:
        if len(self.notes) < 20:
            self.notes.append(note)

    def add_installed_hook(self, hook: str) -> None:
        if hook not in self.installed_hooks:
            self.installed_hooks.append(hook)

    def add_candidate_hook(self, candidate: dict[str, Any]) -> None:
        key = (
            candidate.get("module"),
            candidate.get("owner"),
            candidate.get("method"),
        )
        if any(
            (
                existing.get("module"),
                existing.get("owner"),
                existing.get("method"),
            )
            == key
            for existing in self.candidate_hooks
        ):
            return
        if len(self.candidate_hooks) < 80:
            self.candidate_hooks.append(candidate)

    def add_deep_hook_discovery_note(self, note: str) -> None:
        if len(self.deep_hook_discovery_notes) < 40:
            self.deep_hook_discovery_notes.append(note)

    def sample_gpu(self) -> None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,"
                    "memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            if len(self.gpu_sample_errors) < 10:
                self.gpu_sample_errors.append(f"{type(exc).__name__}: {exc}")
            return
        if proc.returncode != 0:
            if len(self.gpu_sample_errors) < 10:
                self.gpu_sample_errors.append(proc.stderr.strip() or f"nvidia-smi rc={proc.returncode}")
            return
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                continue
            sample = {
                "timestamp": parts[0],
                "name": parts[1],
                "gpu_util_percent": _parse_float_or_none(parts[2]),
                "memory_util_percent": _parse_float_or_none(parts[3]),
                "memory_used_mib": _parse_float_or_none(parts[4]),
                "memory_total_mib": _parse_float_or_none(parts[5]),
                "power_draw_w": _parse_float_or_none(parts[6]),
            }
            self.gpu_samples.append(sample)

    def summary(self) -> dict[str, Any]:
        gpu_util = [
            float(sample["gpu_util_percent"])
            for sample in self.gpu_samples
            if sample.get("gpu_util_percent") is not None
        ]
        mem_used = [
            float(sample["memory_used_mib"])
            for sample in self.gpu_samples
            if sample.get("memory_used_mib") is not None
        ]
        return {
            "enabled": self.enabled,
            "elapsed_sec": round(time.perf_counter() - self.started, 6),
            "timers_sec": {key: round(value, 6) for key, value in sorted(self.timers.items())},
            "counts": dict(sorted(self.counts.items())),
            "gpu_sampling": {
                "interval_sec": self.gpu_sample_interval_sec,
                "sample_count": len(self.gpu_samples),
                "first_sample": self.gpu_samples[0] if self.gpu_samples else None,
                "last_sample": self.gpu_samples[-1] if self.gpu_samples else None,
                "max_gpu_util_percent": max(gpu_util) if gpu_util else None,
                "max_memory_used_mib": max(mem_used) if mem_used else None,
                "errors": self.gpu_sample_errors,
            },
            "installed_hooks": self.installed_hooks,
            "candidate_hooks": self.candidate_hooks,
            "deep_hook_discovery_notes": self.deep_hook_discovery_notes,
            "notes": self.notes,
            "caveat": (
                "Phase timers are inclusive wrappers around LightZero internals. "
                "They split collect/eval/replay/learner/checkpoint wall time, "
                "but do not yet split MCTS tree bookkeeping from model inference."
            ),
        }


class _LightZeroProfileStop(RuntimeError):
    """Intentional profiler-only stop after enough LightZero work was timed."""


def _parse_float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _compute_choices_label() -> str:
    return ", ".join(COMPUTE_CHOICES)


def _start_gpu_sampler(
    *,
    profiler: _LightZeroPhaseProfiler,
    interval_sec: float,
) -> tuple[threading.Event, threading.Thread] | None:
    if interval_sec <= 0:
        return None

    stop_event = threading.Event()

    def _sample_loop() -> None:
        profiler.sample_gpu()
        while not stop_event.wait(interval_sec):
            profiler.sample_gpu()

    thread = threading.Thread(target=_sample_loop, name="exact-lightzero-gpu-sampler", daemon=True)
    thread.start()
    return stop_event, thread


def _discover_lightzero_deep_hook_candidates(
    *,
    train_muzero: Any,
    profiler: _LightZeroPhaseProfiler,
) -> None:
    """Record deeper LightZero hook candidates without patching or instantiating."""

    def add_note(note: str) -> None:
        profiler.add_deep_hook_discovery_note(note)

    def import_module(module_name: str) -> Any | None:
        try:
            return importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            add_note(f"could not import {module_name}: {type(exc).__name__}: {exc}")
            return None

    def raw_method(owner: Any, method_name: str) -> tuple[Any, Any] | tuple[None, None]:
        try:
            owner_mro = inspect.getmro(owner)
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            add_note(
                f"could not inspect MRO for {owner!r}.{method_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None, None
        for base in owner_mro:
            namespace = getattr(base, "__dict__", {})
            if method_name in namespace:
                return base, namespace[method_name]
        return None, None

    def callable_for_signature(raw: Any) -> Any:
        if isinstance(raw, (classmethod, staticmethod)):
            return raw.__func__
        return raw

    def add_method(owner: Any, method_name: str, *, source: str) -> None:
        defining_owner, raw = raw_method(owner, method_name)
        if defining_owner is None:
            add_note(
                f"{getattr(owner, '__module__', type(owner).__module__)}."
                f"{getattr(owner, '__qualname__', getattr(owner, '__name__', type(owner).__name__))}."
                f"{method_name} missing"
            )
            return
        target = callable_for_signature(raw)
        try:
            signature = str(inspect.signature(target))
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            signature = f"<signature unavailable: {type(exc).__name__}: {exc}>"
        try:
            file_path = inspect.getsourcefile(target) or inspect.getfile(target)
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            file_path = f"<file unavailable: {type(exc).__name__}: {exc}>"
        module = getattr(defining_owner, "__module__", type(defining_owner).__module__)
        owner_name = getattr(
            defining_owner,
            "__qualname__",
            getattr(defining_owner, "__name__", type(defining_owner).__name__),
        )
        profiler.add_candidate_hook(
            {
                "label": f"{module}.{owner_name}.{method_name}",
                "module": module,
                "owner": owner_name,
                "method": method_name,
                "signature": signature,
                "file": file_path,
                "source": source,
            }
        )

    def add_class_methods(
        module: Any,
        class_name: str,
        method_names: tuple[str, ...],
        *,
        source: str,
    ) -> None:
        cls = getattr(module, class_name, None)
        if not inspect.isclass(cls):
            add_note(f"{getattr(module, '__name__', module)!s}.{class_name} missing")
            return
        for method_name in method_names:
            add_method(cls, method_name, source=source)

    policy_module = import_module("lzero.policy.muzero")
    if policy_module is not None:
        add_class_methods(
            policy_module,
            "MuZeroPolicy",
            ("_forward_collect", "_forward_eval"),
            source="static_import:lzero.policy.muzero",
        )

    mcts_module_names = (
        "lzero.mcts",
        "lzero.mcts.ctree_muzero",
        "lzero.mcts.ptree_muzero",
        "lzero.mcts.muzero_mcts",
    )
    for module_name in mcts_module_names:
        module = import_module(module_name)
        if module is None:
            continue
        for class_name in ("MuZeroMCTSCtree", "MuZeroMCTSPtree"):
            cls = getattr(module, class_name, None)
            if inspect.isclass(cls):
                add_method(cls, "search", source=f"static_import:{module_name}")

    buffer_names = (
        "MuZeroGameBuffer",
        "EfficientZeroGameBuffer",
        "SampledEfficientZeroGameBuffer",
        "SampledMuZeroGameBuffer",
        "GumbelMuZeroGameBuffer",
        "StochasticMuZeroGameBuffer",
    )
    helper_name_fragments = ("sample", "target", "priority")
    helper_names = {
        "push_game_segments",
        "remove_oldest_data_to_fit",
        "update_priority",
    }
    buffer_candidates: list[tuple[Any, str]] = []
    globals_map = getattr(train_muzero, "__globals__", {})
    if isinstance(globals_map, dict):
        for buffer_name in buffer_names:
            candidate = globals_map.get(buffer_name)
            if inspect.isclass(candidate):
                buffer_candidates.append((candidate, "train_muzero_globals"))
    mcts_module = import_module("lzero.mcts")
    if mcts_module is not None:
        for buffer_name in buffer_names:
            candidate = getattr(mcts_module, buffer_name, None)
            if inspect.isclass(candidate):
                buffer_candidates.append((candidate, "static_import:lzero.mcts"))

    seen_buffers: set[int] = set()
    for buffer_cls, source in buffer_candidates:
        if id(buffer_cls) in seen_buffers:
            continue
        seen_buffers.add(id(buffer_cls))
        method_names = set()
        try:
            buffer_mro = inspect.getmro(buffer_cls)
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            add_note(
                f"could not inspect MRO for {buffer_cls!r}: {type(exc).__name__}: {exc}"
            )
            buffer_mro = (buffer_cls,)
        for base in buffer_mro:
            for name, value in getattr(base, "__dict__", {}).items():
                if not callable(callable_for_signature(value)):
                    continue
                lower_name = name.lower()
                if name in helper_names or any(
                    fragment in lower_name for fragment in helper_name_fragments
                ):
                    method_names.add(name)
        for method_name in sorted(method_names):
            add_method(buffer_cls, method_name, source=source)
    if not seen_buffers:
        add_note("no static MuZeroGameBuffer classes found for deep hook discovery")


def _install_lightzero_phase_profile(
    *,
    train_muzero: Any,
    profiler: _LightZeroPhaseProfiler,
    stop_after_learner_train_calls: int | None = None,
) -> Any:
    """Patch selected LightZero methods in place, returning a restore function."""

    originals: list[tuple[Any, str, Any]] = []
    patched_methods: set[tuple[int, str]] = set()
    installed_hooks: list[str] = []

    def hook_label(obj: Any, method_name: str) -> str:
        module = getattr(obj, "__module__", type(obj).__module__)
        qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", type(obj).__name__))
        return f"{module}.{qualname}.{method_name}"

    def patch_method(cls: Any, method_name: str, wrapped: Any) -> None:
        owner = next(
            (base for base in inspect.getmro(cls) if method_name in getattr(base, "__dict__", {})),
            None,
        )
        if owner is None:
            profiler.add_note(f"{hook_label(cls, method_name)} missing")
            return
        key = (id(owner), method_name)
        if key in patched_methods:
            return
        original = owner.__dict__[method_name]
        originals.append((owner, method_name, original))
        setattr(owner, method_name, wrapped(original))
        patched_methods.add(key)
        installed_hooks.append(hook_label(owner, method_name))

    def safe_int_attr(obj: Any, name: str, default: int) -> int:
        try:
            return int(getattr(obj, name, default) or default)
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(f"could not read integer attr {name}: {type(exc).__name__}: {exc}")
            return default

    def safe_len(value: Any, label: str) -> int | None:
        try:
            return len(value)
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(f"could not read length for {label}: {type(exc).__name__}: {exc}")
            return None

    def patch_init(cls: Any, timer_name: str) -> None:
        def make_wrapped(original: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                with profiler.timer(timer_name):
                    return original(self, *args, **kwargs)

            return wrapped

        patch_method(cls, "__init__", make_wrapped)

    def restore() -> None:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)

    globals_map = train_muzero.__globals__

    try:
        if "Collector" in globals_map and inspect.isclass(globals_map["Collector"]):
            collector_cls = globals_map["Collector"]
            patch_init(collector_cls, "collector_init_sec")

            def make_collect(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    before_envstep = safe_int_attr(self, "envstep", 0)
                    with profiler.timer("collector_collect_sec"):
                        result = original(self, *args, **kwargs)
                    try:
                        after_envstep = safe_int_attr(self, "envstep", before_envstep)
                        profiler.add_count("collector_collect_calls")
                        profiler.add_count("env_steps_collected", max(0, after_envstep - before_envstep))
                        game_segments = safe_len(result, "collector result")
                        if game_segments is not None:
                            profiler.add_count("game_segments_collected", game_segments)
                    except Exception as exc:  # pragma: no cover - remote diagnosis only.
                        profiler.add_note(f"collector profile count failed: {type(exc).__name__}: {exc}")
                    return result

                return wrapped

            patch_method(collector_cls, "collect", make_collect)
        else:
            profiler.add_note("train_muzero globals did not expose Collector class")

        if "Evaluator" in globals_map and inspect.isclass(globals_map["Evaluator"]):
            evaluator_cls = globals_map["Evaluator"]
            patch_init(evaluator_cls, "evaluator_init_sec")

            def make_eval(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    with profiler.timer("evaluator_eval_sec"):
                        result = original(self, *args, **kwargs)
                    try:
                        profiler.add_count("evaluator_eval_calls")
                    except Exception as exc:  # pragma: no cover - remote diagnosis only.
                        profiler.add_note(f"evaluator profile count failed: {type(exc).__name__}: {exc}")
                    return result

                return wrapped

            patch_method(evaluator_cls, "eval", make_eval)
        else:
            profiler.add_note("train_muzero globals did not expose Evaluator class")

        if "BaseLearner" in globals_map and inspect.isclass(globals_map["BaseLearner"]):
            learner_cls = globals_map["BaseLearner"]
            patch_init(learner_cls, "learner_init_sec")

            def make_train(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    before_iter = safe_int_attr(self, "train_iter", 0)
                    with profiler.timer("learner_train_sec"):
                        result = original(self, *args, **kwargs)
                    learner_train_calls = None
                    try:
                        after_iter = safe_int_attr(self, "train_iter", before_iter)
                        profiler.add_count("learner_train_calls")
                        profiler.add_count("learner_train_iter_delta", max(0, after_iter - before_iter))
                        learner_train_calls = profiler.counts.get("learner_train_calls")
                    except Exception as exc:  # pragma: no cover - remote diagnosis only.
                        profiler.add_note(f"learner train profile count failed: {type(exc).__name__}: {exc}")
                    if (
                        stop_after_learner_train_calls is not None
                        and learner_train_calls is not None
                        and learner_train_calls >= stop_after_learner_train_calls
                    ):
                        raise _LightZeroProfileStop(
                            "stopped after "
                            f"{learner_train_calls} BaseLearner.train calls "
                            "by optimizer profile cap"
                        )
                    return result

                return wrapped

            def make_save_checkpoint(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    with profiler.timer("learner_save_checkpoint_sec"):
                        result = original(self, *args, **kwargs)
                    try:
                        profiler.add_count("learner_save_checkpoint_calls")
                    except Exception as exc:  # pragma: no cover - remote diagnosis only.
                        profiler.add_note(
                            f"learner checkpoint profile count failed: {type(exc).__name__}: {exc}"
                        )
                    return result

                return wrapped

            def make_call_hook(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    hook_name = str(args[0]) if args else str(kwargs.get("name", "unknown"))
                    with profiler.timer(f"learner_hook_{hook_name}_sec"):
                        return original(self, *args, **kwargs)

                return wrapped

            patch_method(learner_cls, "train", make_train)
            patch_method(learner_cls, "save_checkpoint", make_save_checkpoint)
            patch_method(learner_cls, "call_hook", make_call_hook)
        else:
            profiler.add_note("train_muzero globals did not expose BaseLearner class")

        buffer_classes: list[Any] = []
        buffer_names = (
            "MuZeroGameBuffer",
            "EfficientZeroGameBuffer",
            "SampledEfficientZeroGameBuffer",
            "SampledMuZeroGameBuffer",
            "GumbelMuZeroGameBuffer",
            "StochasticMuZeroGameBuffer",
        )
        for buffer_name in buffer_names:
            global_candidate = globals_map.get(buffer_name)
            if inspect.isclass(global_candidate):
                buffer_classes.append(global_candidate)

        try:
            mcts_module = importlib.import_module("lzero.mcts")
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(f"could not import lzero.mcts for GameBuffer patch: {type(exc).__name__}: {exc}")
        else:
            for buffer_name in buffer_names:
                candidate = getattr(mcts_module, buffer_name, None)
                if inspect.isclass(candidate):
                    buffer_classes.append(candidate)

        seen_buffer_classes: set[int] = set()
        for buffer_cls in buffer_classes:
            if id(buffer_cls) in seen_buffer_classes:
                continue
            seen_buffer_classes.add(id(buffer_cls))
            patch_init(buffer_cls, "replay_buffer_init_sec")

            def make_push(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    with profiler.timer("replay_push_game_segments_sec"):
                        result = original(self, *args, **kwargs)
                    try:
                        profiler.add_count("replay_push_calls")
                        if args:
                            game_segments = safe_len(args[0], "pushed game segments")
                            if game_segments is not None:
                                profiler.add_count("game_segments_pushed", game_segments)
                    except Exception as exc:  # pragma: no cover - remote diagnosis only.
                        profiler.add_note(f"replay push profile count failed: {type(exc).__name__}: {exc}")
                    return result

                return wrapped

            def make_remove(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    with profiler.timer("replay_remove_oldest_sec"):
                        result = original(self, *args, **kwargs)
                    try:
                        profiler.add_count("replay_remove_oldest_calls")
                    except Exception as exc:  # pragma: no cover - remote diagnosis only.
                        profiler.add_note(f"replay remove profile count failed: {type(exc).__name__}: {exc}")
                    return result

                return wrapped

            def make_sample(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    with profiler.timer("replay_sample_target_sec"):
                        result = original(self, *args, **kwargs)
                    try:
                        profiler.add_count("replay_sample_calls")
                    except Exception as exc:  # pragma: no cover - remote diagnosis only.
                        profiler.add_note(f"replay sample profile count failed: {type(exc).__name__}: {exc}")
                    return result

                return wrapped

            def make_update_priority(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    with profiler.timer("replay_update_priority_sec"):
                        result = original(self, *args, **kwargs)
                    try:
                        profiler.add_count("replay_update_priority_calls")
                    except Exception as exc:  # pragma: no cover - remote diagnosis only.
                        profiler.add_note(f"replay priority profile count failed: {type(exc).__name__}: {exc}")
                    return result

                return wrapped

            patch_method(buffer_cls, "push_game_segments", make_push)
            patch_method(buffer_cls, "remove_oldest_data_to_fit", make_remove)
            patch_method(buffer_cls, "sample", make_sample)
            patch_method(buffer_cls, "update_priority", make_update_priority)

        if not seen_buffer_classes:
            profiler.add_note("no LightZero GameBuffer classes found for replay hooks")
    except Exception:
        restore()
        raise
    for hook in installed_hooks:
        profiler.add_installed_hook(hook)

    return restore


def _get_path(mapping: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = mapping
    try:
        for part in path:
            current = current[part]
    except KeyError:
        return default
    return current


def _set_exp_name(main_config: Any, exp_name: Path) -> dict[str, Any]:
    old_value = main_config["exp_name"]
    main_config["exp_name"] = str(exp_name)
    return {
        "path": "exp_name",
        "old": _to_plain(old_value),
        "new": str(exp_name),
        "reason": "Modal Volume artifact root only; no training/evaluator semantics changed.",
    }


def _set_save_ckpt_after_iter(main_config: Any, value: int) -> dict[str, Any]:
    current = main_config["policy"]
    for part in ("learn", "learner", "hook"):
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    hook = current
    old_value = hook.get("save_ckpt_after_iter")
    hook["save_ckpt_after_iter"] = value
    return {
        "path": "policy.learn.learner.hook.save_ckpt_after_iter",
        "old": _to_plain(old_value),
        "new": value,
        "reason": "Future short-run observability only; trainer/evaluator semantics unchanged.",
    }


def _set_survival_reward_shaping(
    main_config: Any,
    create_config: Any,
    *,
    survival_reward_per_step: float,
) -> list[dict[str, Any]]:
    old_env_type = create_config["env"]["type"]
    old_import_names = create_config["env"].get("import_names")
    old_reward = main_config["env"].get("survival_reward_per_step")
    old_schema = main_config["env"].get("reward_shaping_schema_id")
    create_config["env"]["type"] = LIGHTZERO_ATARI_SURVIVAL_ENV_TYPE
    create_config["env"]["import_names"] = ["curvyzero.training.lightzero_atari_survival_env"]
    main_config["env"]["survival_reward_per_step"] = float(survival_reward_per_step)
    main_config["env"]["survival_reward_apply_on_done"] = False
    main_config["env"]["reward_shaping_schema_id"] = REWARD_SHAPING_SCHEMA_ID
    return [
        {
            "path": "create_config.env.type",
            "old": _to_plain(old_env_type),
            "new": LIGHTZERO_ATARI_SURVIVAL_ENV_TYPE,
            "reason": "Opt-in shaped-objective ablation; stock/control env_type is untouched.",
        },
        {
            "path": "create_config.env.import_names",
            "old": _to_plain(old_import_names),
            "new": ["curvyzero.training.lightzero_atari_survival_env"],
            "reason": "Imports a separate registered env wrapper for the shaped run only.",
        },
        {
            "path": "env.survival_reward_per_step",
            "old": _to_plain(old_reward),
            "new": float(survival_reward_per_step),
            "reason": "Adds a small reward bonus on non-terminal Atari env steps.",
        },
        {
            "path": "env.reward_shaping_schema_id",
            "old": _to_plain(old_schema),
            "new": REWARD_SHAPING_SCHEMA_ID,
            "reason": "Makes the changed reward contract explicit in artifacts.",
        },
    ]


def _validate_survival_shaped_ids(*, run_id: str, attempt_id: str) -> list[str]:
    problems: list[str] = []
    marker = "survival-shaped"
    if marker not in run_id:
        problems.append(
            f"survival-shaped run_id must contain {marker!r}; got {run_id!r}"
        )
    if marker not in attempt_id:
        problems.append(
            f"survival-shaped attempt_id must contain {marker!r}; got {attempt_id!r}"
        )
    return problems


def _runtime_compute_summary(*, requested_compute: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "requested_compute": requested_compute,
        "cheap_gpu_resource": CHEAP_GPU_RESOURCE,
        "gpu_resource_by_compute": GPU_RESOURCE_BY_COMPUTE,
        "modal_cpu_count_by_compute": CPU_COUNT_BY_COMPUTE,
        "requested_modal_cpu_count": CPU_COUNT_BY_COMPUTE.get(requested_compute),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        summary.update(
            {
                "torch_cuda_available": cuda_available,
                "torch_cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            }
        )
        if cuda_available:
            device = int(torch.cuda.current_device())
            summary.update(
                {
                    "torch_cuda_current_device": device,
                    "torch_cuda_device_name": torch.cuda.get_device_name(device),
                    "torch_cuda_capability": list(torch.cuda.get_device_capability(device)),
                }
            )
    except Exception as exc:  # pragma: no cover - remote runtime diagnosis only.
        summary["torch_cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _extract_surface(main_config: Any, create_config: Any, *, max_env_step: int) -> dict[str, Any]:
    policy = main_config["policy"]
    env = main_config["env"]
    model = policy["model"]
    return {
        "exp_name": str(main_config["exp_name"]),
        "env_id": env["env_id"],
        "env_type": create_config["env"]["type"],
        "env_import_names": _to_plain(create_config["env"].get("import_names")),
        "env_manager_type": create_config["env_manager"]["type"],
        "policy_type": create_config["policy"]["type"],
        "policy_import_names": _to_plain(create_config["policy"].get("import_names")),
        "model_type": model["model_type"],
        "observation_shape": _to_plain(model["observation_shape"]),
        "env_observation_shape": _to_plain(env.get("observation_shape")),
        "action_space_size": model["action_space_size"],
        "collector_env_num": env["collector_env_num"],
        "policy_collector_env_num": policy.get("collector_env_num"),
        "n_episode": policy.get("n_episode"),
        "evaluator_env_num": env["evaluator_env_num"],
        "policy_evaluator_env_num": policy.get("evaluator_env_num"),
        "n_evaluator_episode": env.get("n_evaluator_episode"),
        "num_simulations": policy["num_simulations"],
        "batch_size": policy["batch_size"],
        "update_per_collect": policy.get("update_per_collect"),
        "replay_ratio": policy.get("replay_ratio"),
        "game_segment_length": policy.get("game_segment_length"),
        "eval_freq": policy.get("eval_freq"),
        "cuda": policy["cuda"],
        "learning_rate": policy.get("learning_rate"),
        "target_update_freq": policy.get("target_update_freq"),
        "replay_buffer_size": policy.get("replay_buffer_size"),
        "save_ckpt_after_iter": _get_path(
            policy, ("learn", "learner", "hook", "save_ckpt_after_iter")
        ),
        "collect_max_episode_steps": env.get("collect_max_episode_steps"),
        "eval_max_episode_steps": env.get("eval_max_episode_steps"),
        "frame_stack_num": env.get("frame_stack_num"),
        "gray_scale": env.get("gray_scale"),
        "image_channel": env.get("image_channel"),
        "survival_reward_per_step": env.get("survival_reward_per_step"),
        "survival_reward_apply_on_done": env.get("survival_reward_apply_on_done"),
        "reward_shaping_schema_id": env.get("reward_shaping_schema_id"),
        "max_env_step": max_env_step,
    }


def _validate_exact_surface(surface: dict[str, Any], *, expected_max_env_step: int) -> list[str]:
    problems: list[str] = []
    expected = {
        "env_id": "PongNoFrameskip-v4",
        "env_type": "atari_lightzero",
        "env_import_names": ["zoo.atari.envs.atari_lightzero_env"],
        "env_manager_type": "subprocess",
        "policy_type": "muzero",
        "policy_import_names": ["lzero.policy.muzero"],
        "model_type": "conv",
        "observation_shape": [4, 64, 64],
        "env_observation_shape": [4, 64, 64],
        "action_space_size": 6,
        "collector_env_num": 8,
        "policy_collector_env_num": 8,
        "n_episode": 8,
        "evaluator_env_num": 3,
        "policy_evaluator_env_num": 3,
        "n_evaluator_episode": 3,
        "num_simulations": 50,
        "batch_size": 256,
        "update_per_collect": None,
        "replay_ratio": 0.25,
        "game_segment_length": 400,
        "eval_freq": 2000,
        "cuda": True,
        "learning_rate": 0.2,
        "target_update_freq": 100,
        "replay_buffer_size": 1000000,
        "frame_stack_num": 4,
        "gray_scale": True,
        "image_channel": None,
        "max_env_step": expected_max_env_step,
    }
    for key, value in expected.items():
        if surface.get(key) != value:
            problems.append(f"exact installed surface {key}={surface.get(key)!r}, expected {value!r}")
    if surface.get("collect_max_episode_steps") is not None:
        problems.append("exact installed surface unexpectedly sets collect_max_episode_steps")
    if surface.get("eval_max_episode_steps") is not None:
        problems.append("exact installed surface unexpectedly sets eval_max_episode_steps")
    if surface.get("survival_reward_per_step") is not None:
        problems.append("exact installed surface unexpectedly sets survival_reward_per_step")
    if surface.get("reward_shaping_schema_id") is not None:
        problems.append("exact installed surface unexpectedly sets reward_shaping_schema_id")
    return problems


def _validate_survival_shaped_surface(
    surface: dict[str, Any],
    *,
    expected_max_env_step: int,
    survival_reward_per_step: float,
) -> list[str]:
    problems: list[str] = []
    stock_like = dict(surface)
    stock_like["env_type"] = "atari_lightzero"
    stock_like["env_import_names"] = ["zoo.atari.envs.atari_lightzero_env"]
    stock_like["survival_reward_per_step"] = None
    stock_like["survival_reward_apply_on_done"] = None
    stock_like["reward_shaping_schema_id"] = None
    problems.extend(_validate_exact_surface(stock_like, expected_max_env_step=expected_max_env_step))
    if surface.get("env_type") != LIGHTZERO_ATARI_SURVIVAL_ENV_TYPE:
        problems.append(
            f"survival-shaped env_type={surface.get('env_type')!r}, "
            f"expected {LIGHTZERO_ATARI_SURVIVAL_ENV_TYPE!r}"
        )
    if surface.get("env_import_names") != ["curvyzero.training.lightzero_atari_survival_env"]:
        problems.append(
            "survival-shaped env_import_names="
            f"{surface.get('env_import_names')!r}, expected custom survival env import"
        )
    if surface.get("survival_reward_per_step") != float(survival_reward_per_step):
        problems.append(
            "survival-shaped survival_reward_per_step="
            f"{surface.get('survival_reward_per_step')!r}, expected {survival_reward_per_step!r}"
        )
    if surface.get("survival_reward_apply_on_done") is not False:
        problems.append(
            "survival-shaped survival_reward_apply_on_done="
            f"{surface.get('survival_reward_apply_on_done')!r}, expected False"
        )
    if surface.get("reward_shaping_schema_id") != REWARD_SHAPING_SCHEMA_ID:
        problems.append(
            "survival-shaped reward_shaping_schema_id="
            f"{surface.get('reward_shaping_schema_id')!r}, expected {REWARD_SHAPING_SCHEMA_ID!r}"
        )
    return problems


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _write_json_artifact(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(path, _to_plain(payload))
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _exp_name_scan_roots(exp_name: Path) -> list[dict[str, Any]]:
    roots = [
        {
            "kind": "configured_exp_name",
            "path": exp_name,
            "is_alternate": False,
            "reason": "main_config.exp_name value patched by wrapper",
        }
    ]
    if exp_name.is_absolute():
        roots.append(
            {
                "kind": "cwd_relative_exp_name",
                "path": Path.cwd() / str(exp_name).lstrip("/"),
                "is_alternate": True,
                "reason": "guards against LightZero or DI-engine prepending './' to exp_name",
            }
        )

    unique_roots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        key = os.path.abspath(os.fspath(root["path"]))
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(root)
    return unique_roots


def _scan_one_exp_dir(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    file_count = 0
    total_bytes = 0
    checkpoint_count = 0
    checkpoint_bytes = 0
    newest_checkpoints: list[dict[str, Any]] = []
    largest_files: list[dict[str, Any]] = []
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            relative_path = path.relative_to(root).as_posix()
            item = {
                "path": relative_path,
                "bytes": int(stat.st_size),
                "mtime": float(stat.st_mtime),
            }
            file_count += 1
            total_bytes += int(stat.st_size)
            largest_files.append(item)
            if path.name.endswith((".pth.tar", ".pth", ".pt")):
                checkpoint_count += 1
                checkpoint_bytes += int(stat.st_size)
                newest_checkpoints.append(item)
    newest_checkpoints = sorted(
        newest_checkpoints,
        key=lambda item: item["mtime"],
        reverse=True,
    )[:12]
    largest_files = sorted(
        largest_files,
        key=lambda item: item["bytes"],
        reverse=True,
    )[:12]
    return {
        "path": str(root),
        "exists": root.exists(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "checkpoint_count": checkpoint_count,
        "checkpoint_bytes": checkpoint_bytes,
        "newest_checkpoints": newest_checkpoints,
        "largest_files": largest_files,
        "scan_elapsed_sec": round(time.perf_counter() - started, 6),
    }


def _scan_exact_exp_dir(exp_name: Path) -> dict[str, Any]:
    started = time.perf_counter()
    root_scans: list[dict[str, Any]] = []
    file_count = 0
    total_bytes = 0
    checkpoint_count = 0
    checkpoint_bytes = 0
    newest_checkpoints: list[dict[str, Any]] = []
    largest_files: list[dict[str, Any]] = []

    for root in _exp_name_scan_roots(exp_name):
        scan = _scan_one_exp_dir(root["path"])
        scan.update(
            {
                "kind": root["kind"],
                "is_alternate": root["is_alternate"],
                "reason": root["reason"],
            }
        )
        root_scans.append(scan)
        file_count += int(scan["file_count"])
        total_bytes += int(scan["total_bytes"])
        checkpoint_count += int(scan["checkpoint_count"])
        checkpoint_bytes += int(scan["checkpoint_bytes"])
        for item in scan["newest_checkpoints"]:
            newest_checkpoints.append({**item, "root_kind": scan["kind"], "root": scan["path"]})
        for item in scan["largest_files"]:
            largest_files.append({**item, "root_kind": scan["kind"], "root": scan["path"]})

    newest_checkpoints = sorted(
        newest_checkpoints,
        key=lambda item: item["mtime"],
        reverse=True,
    )[:12]
    largest_files = sorted(
        largest_files,
        key=lambda item: item["bytes"],
        reverse=True,
    )[:12]
    alternate_roots = [
        {
            "kind": scan["kind"],
            "path": scan["path"],
            "exists": scan["exists"],
            "file_count": scan["file_count"],
            "checkpoint_count": scan["checkpoint_count"],
        }
        for scan in root_scans
        if scan["is_alternate"]
    ]
    return {
        "exp_name": str(exp_name),
        "cwd": str(Path.cwd()),
        "exp_name_exists": exp_name.exists(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "checkpoint_count": checkpoint_count,
        "checkpoint_bytes": checkpoint_bytes,
        "newest_checkpoints": newest_checkpoints,
        "largest_files": largest_files,
        "root_scans": root_scans,
        "alternate_artifact_roots": alternate_roots,
        "scan_elapsed_sec": round(time.perf_counter() - started, 6),
    }


def _artifact_root_agreement(scan: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(scan, dict):
        return None

    root_scans = scan.get("root_scans")
    if not isinstance(root_scans, list):
        root_scans = []
    configured = [
        root for root in root_scans if isinstance(root, dict) and root.get("kind") == "configured_exp_name"
    ]
    alternates = [
        root for root in root_scans if isinstance(root, dict) and bool(root.get("is_alternate"))
    ]
    alternate_file_roots = [
        {
            "kind": root.get("kind"),
            "path": root.get("path"),
            "file_count": root.get("file_count"),
            "checkpoint_count": root.get("checkpoint_count"),
            "checkpoint_bytes": root.get("checkpoint_bytes"),
        }
        for root in alternates
        if int(root.get("file_count") or 0) > 0 or int(root.get("checkpoint_count") or 0) > 0
    ]
    configured_has_files = any(int(root.get("file_count") or 0) > 0 for root in configured)
    configured_has_checkpoints = any(int(root.get("checkpoint_count") or 0) > 0 for root in configured)
    root_mismatch_suspected = bool(alternate_file_roots) or (
        bool(alternates) and not configured_has_files and int(scan.get("file_count") or 0) > 0
    )
    return {
        "configured_root_count": len(configured),
        "alternate_root_count": len(alternates),
        "configured_has_files": configured_has_files,
        "configured_has_checkpoints": configured_has_checkpoints,
        "alternate_file_roots": alternate_file_roots,
        "root_mismatch_suspected": root_mismatch_suspected,
    }


def _observability_summary(
    *,
    train_result: dict[str, Any] | None,
    artifact_scan: dict[str, Any] | None,
) -> dict[str, Any]:
    checkpoint_bytes = (
        int(artifact_scan.get("checkpoint_bytes") or 0) if isinstance(artifact_scan, dict) else 0
    )
    total_bytes = int(artifact_scan.get("total_bytes") or 0) if isinstance(artifact_scan, dict) else 0
    return {
        "train_elapsed_sec": (
            train_result.get("elapsed_sec") if isinstance(train_result, dict) else None
        ),
        "artifact_total_bytes": total_bytes,
        "artifact_total_gib": round(total_bytes / (1024**3), 6),
        "checkpoint_count": (
            int(artifact_scan.get("checkpoint_count") or 0) if isinstance(artifact_scan, dict) else 0
        ),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_gib": round(checkpoint_bytes / (1024**3), 6),
        "artifact_root_agreement": _artifact_root_agreement(artifact_scan),
    }


def _write_progress_snapshot(
    *,
    exp_name: Path,
    run_id: str,
    attempt_id: str,
    phase: str,
    progress_interval_sec: int,
    save_ckpt_after_iter_override: int | None,
    actual_save_ckpt_after_iter: int | None,
    train_started_at: str | None,
    train_elapsed_sec: float | None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    progress_root = (
        runs.volume_path(RUNS_MOUNT, runs.attempt_train_ref(TASK_ID, run_id, attempt_id))
        / "progress"
    )
    timestamp = runs.utc_timestamp()
    try:
        scan = _scan_exact_exp_dir(exp_name)
        payload = {
            "schema": "curvyzero_lightzero_exact_reproduction_progress/v0",
            "ok": error is None,
            "phase": phase,
            "timestamp": timestamp,
            "task_id": TASK_ID,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "progress_interval_sec": progress_interval_sec,
            "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
            "actual_save_ckpt_after_iter": actual_save_ckpt_after_iter,
            "train_started_at": train_started_at,
            "train_elapsed_sec": train_elapsed_sec,
            "scan": scan,
        }
        if error is not None:
            payload["error"] = error
    except Exception as exc:  # pragma: no cover - remote progress guard only.
        payload = {
            "schema": "curvyzero_lightzero_exact_reproduction_progress/v0",
            "ok": False,
            "phase": phase,
            "timestamp": timestamp,
            "task_id": TASK_ID,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "progress_interval_sec": progress_interval_sec,
            "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
            "actual_save_ckpt_after_iter": actual_save_ckpt_after_iter,
            "train_started_at": train_started_at,
            "train_elapsed_sec": train_elapsed_sec,
            "scanner_error": _exception_result(exc),
        }

    progress_root.mkdir(parents=True, exist_ok=True)
    latest_path = progress_root / "latest.json"
    history_path = progress_root / f"progress_{phase}_{runs.utc_stamp()}.json"
    runs.write_json(latest_path, payload)
    runs.write_json(history_path, payload)
    return {
        "latest": runs.file_summary(latest_path, mount=RUNS_MOUNT),
        "history": runs.file_summary(history_path, mount=RUNS_MOUNT),
        "payload": _to_plain(payload),
    }


def _start_progress_watcher(
    *,
    exp_name: Path,
    run_id: str,
    attempt_id: str,
    progress_interval_sec: int,
    save_ckpt_after_iter_override: int | None,
    actual_save_ckpt_after_iter: int | None,
    train_started_at: str,
    train_started_perf: float,
    snapshots: list[dict[str, Any]],
) -> tuple[threading.Event, threading.Thread] | None:
    if progress_interval_sec <= 0:
        return None

    stop_event = threading.Event()

    def _watch() -> None:
        while not stop_event.wait(progress_interval_sec):
            try:
                snapshot = _write_progress_snapshot(
                    exp_name=exp_name,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    phase="running",
                    progress_interval_sec=progress_interval_sec,
                    save_ckpt_after_iter_override=save_ckpt_after_iter_override,
                    actual_save_ckpt_after_iter=actual_save_ckpt_after_iter,
                    train_started_at=train_started_at,
                    train_elapsed_sec=time.perf_counter() - train_started_perf,
                )
                snapshots.append(snapshot)
                runs_volume.commit()
            except Exception:
                # Progress reporting must never kill exact training.
                pass

    thread = threading.Thread(target=_watch, name="exact-lightzero-progress", daemon=True)
    thread.start()
    return stop_event, thread


def _run_exact_reproduction(
    *,
    mode: str,
    compute: str,
    seed: int,
    run_id: str,
    attempt_id: str,
    progress_interval_sec: int,
    max_env_step_override: int | None,
    max_train_iter_override: int | None,
    save_ckpt_after_iter_override: int | None,
    survival_reward_per_step: float,
    profile_phases: bool,
    gpu_sample_interval_sec: float,
    profile_stop_after_learner_train_calls: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if mode not in {"dry", "train"}:
        raise ValueError(f"unknown mode: {mode!r}; expected 'dry' or 'train'")
    if compute not in COMPUTE_CHOICES:
        raise ValueError(
            f"unknown compute: {compute!r}; expected one of: {_compute_choices_label()}"
        )
    if max_env_step_override is not None and max_env_step_override <= 0:
        raise ValueError("max_env_step_override must be a positive integer when provided")
    if max_train_iter_override is not None and max_train_iter_override <= 0:
        raise ValueError("max_train_iter_override must be a positive integer when provided")
    if save_ckpt_after_iter_override is not None and save_ckpt_after_iter_override <= 0:
        raise ValueError(
            "save_ckpt_after_iter_override must be a positive integer when provided"
        )
    if gpu_sample_interval_sec < 0:
        raise ValueError("gpu_sample_interval_sec must be non-negative")
    if (
        profile_stop_after_learner_train_calls is not None
        and profile_stop_after_learner_train_calls <= 0
    ):
        raise ValueError(
            "profile_stop_after_learner_train_calls must be a positive integer when provided"
        )
    if profile_stop_after_learner_train_calls is not None and not profile_phases:
        raise ValueError("profile_stop_after_learner_train_calls requires profile_phases=True")
    if survival_reward_per_step < 0.0:
        raise ValueError("survival_reward_per_step must be non-negative")

    profiler = _LightZeroPhaseProfiler(
        enabled=profile_phases,
        gpu_sample_interval_sec=gpu_sample_interval_sec,
    )

    with profiler.timer("package_version_probe_sec") if profile_phases else contextlib.nullcontext():
        packages = {
            "LightZero": _version_or_missing("LightZero", "lightzero"),
            "DI-engine": _version_or_missing("DI-engine", "ding"),
            "torch": _version_or_missing("torch"),
            "easydict": _version_or_missing("easydict"),
            "gym": _version_or_missing("gym"),
            "gymnasium": _version_or_missing("gymnasium"),
            "ale-py": _version_or_missing("ale-py", "ale_py"),
            "opencv-python-headless": _version_or_missing("opencv-python-headless"),
            "AutoROM": _version_or_missing("AutoROM"),
        }
    problems: list[str] = []
    if packages["LightZero"] != LIGHTZERO_VERSION:
        problems.append(
            f"installed LightZero version is {packages['LightZero']!r}, expected {LIGHTZERO_VERSION!r}"
        )

    module_name = "zoo.atari.config.atari_muzero_config"
    with profiler.timer("lightzero_import_sec") if profile_phases else contextlib.nullcontext():
        module = importlib.import_module(module_name)
        entry_module = importlib.import_module("lzero.entry")
        train_muzero = entry_module.train_muzero
    train_muzero_signature = inspect.signature(train_muzero)
    if profile_phases:
        with profiler.timer("deep_hook_discovery_sec"):
            try:
                _discover_lightzero_deep_hook_candidates(
                    train_muzero=train_muzero,
                    profiler=profiler,
                )
            except Exception as exc:  # pragma: no cover - remote diagnosis only.
                profiler.add_deep_hook_discovery_note(
                    f"deep hook discovery failed: {type(exc).__name__}: {exc}"
                )
    stock_max_env_step = int(getattr(module, "max_env_step"))
    if max_train_iter_override is not None and "max_train_iter" not in train_muzero_signature.parameters:
        problems.append(
            "max_train_iter_override was requested, but installed train_muzero "
            "does not expose a max_train_iter parameter."
        )
    if max_env_step_override is not None and max_env_step_override >= stock_max_env_step:
        raise ValueError(
            "max_env_step_override must be less than the installed stock max_env_step "
            f"({stock_max_env_step}) for faithful-short rehearsal"
        )
    survival_shaping_enabled = survival_reward_per_step > 0.0
    run_kind = "exact" if max_env_step_override is None else "faithful-short"
    if survival_shaping_enabled:
        run_kind = f"survival-shaped-{run_kind}"
    iteration_cap_kind = "stock" if max_train_iter_override is None else "override"
    checkpoint_cadence_kind = "stock" if save_ckpt_after_iter_override is None else "override"
    actual_max_env_step = stock_max_env_step if max_env_step_override is None else max_env_step_override
    max_env_step_patch = (
        None
        if max_env_step_override is None
        else {
            "path": "train_muzero.max_env_step",
            "old": stock_max_env_step,
            "new": actual_max_env_step,
            "reason": (
                "Faithful-short rehearsal only; installed LightZero config stays stock, "
                "but trainer max_env_step is shortened."
            ),
        }
    )
    max_train_iter_patch = (
        None
        if max_train_iter_override is None
        else {
            "path": "train_muzero.max_train_iter",
            "old": None,
            "new": max_train_iter_override,
            "reason": (
                "Optimizer profile/control only; caps learner iterations so phase "
                "profiling does not become a training run."
            ),
        }
    )
    original_main_config = module.main_config
    original_create_config = module.create_config
    original_surface = _extract_surface(
        original_main_config, original_create_config, max_env_step=stock_max_env_step
    )
    stock_save_ckpt_after_iter = original_surface.get("save_ckpt_after_iter")
    main_config = copy.deepcopy(original_main_config)
    create_config = copy.deepcopy(original_create_config)
    original_cwd = Path.cwd()
    train_workdir = RUNS_MOUNT
    exp_name_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "lightzero_exp"
    exp_name = Path(exp_name_ref.as_posix())
    patches = [_set_exp_name(main_config, exp_name)]
    if survival_shaping_enabled:
        patches.extend(
            _set_survival_reward_shaping(
                main_config,
                create_config,
                survival_reward_per_step=survival_reward_per_step,
            )
        )
    if max_env_step_patch is not None:
        patches.append(max_env_step_patch)
    if max_train_iter_patch is not None:
        patches.append(max_train_iter_patch)
    save_ckpt_after_iter_patch = None
    if save_ckpt_after_iter_override is not None:
        save_ckpt_after_iter_patch = _set_save_ckpt_after_iter(
            main_config, save_ckpt_after_iter_override
        )
        patches.append(save_ckpt_after_iter_patch)
    patched_surface = _extract_surface(main_config, create_config, max_env_step=actual_max_env_step)
    actual_save_ckpt_after_iter = patched_surface.get("save_ckpt_after_iter")
    problems.extend(_validate_exact_surface(original_surface, expected_max_env_step=EXACT_MAX_ENV_STEP))
    if survival_shaping_enabled:
        problems.extend(_validate_survival_shaped_ids(run_id=run_id, attempt_id=attempt_id))
        problems.extend(
            _validate_survival_shaped_surface(
                patched_surface,
                expected_max_env_step=actual_max_env_step,
                survival_reward_per_step=survival_reward_per_step,
            )
        )
    else:
        problems.extend(_validate_exact_surface(patched_surface, expected_max_env_step=actual_max_env_step))

    cpu_train_blocked = mode == "train" and compute == "cpu"
    if cpu_train_blocked:
        problems.append(
            "CPU training is blocked by wrapper guard; use a GPU compute for mode=train. "
            "train_muzero was not called."
        )

    train_result: dict[str, Any] | None = None
    progress_snapshots: list[dict[str, Any]] = []
    if cpu_train_blocked:
        train_result = {
            "ok": False,
            "blocked_before_train_muzero": True,
            "reason": "mode=train with compute=cpu is not allowed",
        }
    elif mode == "train" and not problems:
        os.chdir(train_workdir)
        train_started = time.perf_counter()
        train_started_at = runs.utc_timestamp()
        try:
            initial_snapshot = _write_progress_snapshot(
                exp_name=exp_name,
                run_id=run_id,
                attempt_id=attempt_id,
                phase="starting",
                progress_interval_sec=progress_interval_sec,
                save_ckpt_after_iter_override=save_ckpt_after_iter_override,
                actual_save_ckpt_after_iter=actual_save_ckpt_after_iter,
                train_started_at=train_started_at,
                train_elapsed_sec=0.0,
            )
            progress_snapshots.append(initial_snapshot)
            runs_volume.commit()
        except Exception:
            pass
        watcher = _start_progress_watcher(
            exp_name=exp_name,
            run_id=run_id,
            attempt_id=attempt_id,
            progress_interval_sec=progress_interval_sec,
            save_ckpt_after_iter_override=save_ckpt_after_iter_override,
            actual_save_ckpt_after_iter=actual_save_ckpt_after_iter,
            train_started_at=train_started_at,
            train_started_perf=train_started,
            snapshots=progress_snapshots,
        )
        restore_profile = None
        gpu_sampler = None
        try:
            if profile_phases:
                try:
                    restore_profile = _install_lightzero_phase_profile(
                        train_muzero=train_muzero,
                        profiler=profiler,
                        stop_after_learner_train_calls=profile_stop_after_learner_train_calls,
                    )
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(
                        f"phase profile install failed; training continued unpatched: "
                        f"{type(exc).__name__}: {exc}"
                    )
                gpu_sampler = _start_gpu_sampler(
                    profiler=profiler,
                    interval_sec=gpu_sample_interval_sec,
                )
            with profiler.timer("train_muzero_wall_sec") if profile_phases else contextlib.nullcontext():
                train_kwargs = {"seed": seed, "max_env_step": actual_max_env_step}
                if max_train_iter_override is not None:
                    train_kwargs["max_train_iter"] = max_train_iter_override
                output = train_muzero([main_config, create_config], **train_kwargs)
            train_result = {
                "ok": True,
                "return_type": type(output).__name__,
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
            }
        except _LightZeroProfileStop as exc:
            train_result = {
                "ok": True,
                "stopped_by_optimizer_profile_cap": True,
                "reason": str(exc),
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
            }
        except Exception as exc:  # pragma: no cover - remote trainer diagnosis only.
            problems.append(f"exact train_muzero failed: {type(exc).__name__}: {exc}")
            train_result = {"ok": False, "elapsed_sec": round(time.perf_counter() - train_started, 6)}
            train_result.update(_exception_result(exc))
        finally:
            if gpu_sampler is not None:
                gpu_stop_event, gpu_thread = gpu_sampler
                gpu_stop_event.set()
                gpu_thread.join(timeout=5)
            if restore_profile is not None:
                try:
                    restore_profile()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(f"phase profile restore failed: {type(exc).__name__}: {exc}")
            if watcher is not None:
                stop_event, thread = watcher
                stop_event.set()
                thread.join(timeout=5)
            try:
                final_snapshot = _write_progress_snapshot(
                    exp_name=exp_name,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    phase="completed" if train_result and train_result.get("ok") else "failed",
                    progress_interval_sec=progress_interval_sec,
                    save_ckpt_after_iter_override=save_ckpt_after_iter_override,
                    actual_save_ckpt_after_iter=actual_save_ckpt_after_iter,
                    train_started_at=train_started_at,
                    train_elapsed_sec=time.perf_counter() - train_started,
                    error=None if train_result and train_result.get("ok") else train_result,
                )
                progress_snapshots.append(final_snapshot)
                runs_volume.commit()
            except Exception:
                pass

    final_artifact_scan: dict[str, Any] | None = None
    artifact_scan_error: dict[str, Any] | None = None
    try:
        with profiler.timer("artifact_scan_final_sec") if profile_phases else contextlib.nullcontext():
            final_artifact_scan = _scan_exact_exp_dir(exp_name)
    except Exception as exc:  # pragma: no cover - remote summary guard only.
        artifact_scan_error = _exception_result(exc)

    result = {
        "ok": not problems and (mode == "dry" or bool(train_result and train_result.get("ok"))),
        "label": (
            (
                "survival-shaped installed LightZero 0.2.0 Atari Pong MuZero ablation"
                if survival_shaping_enabled
                else "exact installed LightZero 0.2.0 Atari Pong MuZero reproduction"
            )
            if max_env_step_override is None
            else (
                "survival-shaped faithful-short installed LightZero 0.2.0 Atari Pong MuZero ablation"
                if survival_shaping_enabled
                else "faithful-short installed LightZero 0.2.0 Atari Pong MuZero rehearsal"
            )
        ),
        "run_kind": run_kind,
        "reward_shaping": {
            "enabled": survival_shaping_enabled,
            "schema_id": REWARD_SHAPING_SCHEMA_ID if survival_shaping_enabled else None,
            "mode": "per_step_survival" if survival_shaping_enabled else "none",
            "survival_reward_per_step": float(survival_reward_per_step),
            "requires_id_marker": "survival-shaped",
            "stock_control_path_unchanged": not survival_shaping_enabled,
        },
        "is_exact_reproduction": (
            run_kind == "exact"
            and iteration_cap_kind == "stock"
            and checkpoint_cadence_kind == "stock"
            and not survival_shaping_enabled
        ),
        "stock_max_env_step": stock_max_env_step,
        "actual_max_env_step": actual_max_env_step,
        "max_env_step_override": max_env_step_override,
        "max_train_iter_override": max_train_iter_override,
        "iteration_cap_kind": iteration_cap_kind,
        "profile_stop_after_learner_train_calls": profile_stop_after_learner_train_calls,
        "stock_save_ckpt_after_iter": stock_save_ckpt_after_iter,
        "actual_save_ckpt_after_iter": actual_save_ckpt_after_iter,
        "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
        "checkpoint_cadence_kind": checkpoint_cadence_kind,
        "extra_patch": max_env_step_patch,
        "iteration_cap_patch": max_train_iter_patch,
        "checkpoint_cadence_patch": save_ckpt_after_iter_patch,
        "working_directory": {
            "original_cwd": str(original_cwd),
            "current_cwd": str(Path.cwd()),
            "train_workdir": str(train_workdir),
            "exp_name_ref": exp_name_ref.as_posix(),
            "exp_name_config_value": str(exp_name),
            "reason": (
                "Train mode runs from /runs with a relative exp_name so DI-engine './' "
                "checkpoint paths stay inside the Modal Volume."
            ),
        },
        "mode": mode,
        "compute": compute,
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "packages": packages,
        "runtime_compute": _runtime_compute_summary(requested_compute=compute),
        "problems": problems,
        "call_policy": (
            f"dry_import_config_patch_exp_name_only_no_env_no_train_{run_kind}_"
            f"{iteration_cap_kind}_iteration_cap_{checkpoint_cadence_kind}_ckpt_cadence"
            if mode == "dry"
            else (
                "blocked_cpu_train_before_train_muzero"
                if cpu_train_blocked
                else (
                    (
                        "calls_stock_lzero.entry.train_muzero_with_survival_shaped_env_and_stock_max_env_step"
                        if survival_shaping_enabled and max_env_step_override is None
                        else "calls_stock_lzero.entry.train_muzero_with_survival_shaped_env_and_max_env_step_override"
                        if survival_shaping_enabled
                        else "calls_stock_lzero.entry.train_muzero_with_installed_config_and_stock_max_env_step"
                        if max_env_step_override is None
                        else "calls_stock_lzero.entry.train_muzero_with_installed_config_and_max_env_step_override"
                    )
                )
                + f"_{iteration_cap_kind}_iteration_cap_{checkpoint_cadence_kind}_ckpt_cadence"
            )
        ),
        "stock_example": {
            "module": module_name,
            "trainer_entrypoint": "lzero.entry.train_muzero",
            "trainer_signature": str(train_muzero_signature),
            "trainer_args": {
                "seed": seed,
                "max_env_step": actual_max_env_step,
                **(
                    {"max_train_iter": max_train_iter_override}
                    if max_train_iter_override is not None
                    else {}
                ),
            },
            "original_surface": original_surface,
            "patched_surface": patched_surface,
            "patches": patches,
        },
        "train_result": train_result,
        "phase_profile": profiler.summary(),
        "progress": {
            "interval_sec": progress_interval_sec,
            "enabled": bool(mode == "train" and not cpu_train_blocked and progress_interval_sec > 0),
            "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
            "actual_save_ckpt_after_iter": actual_save_ckpt_after_iter,
            "snapshot_count": len(progress_snapshots),
            "latest": (
                progress_snapshots[-1].get("latest")
                if progress_snapshots and isinstance(progress_snapshots[-1], dict)
                else None
            ),
        },
        "artifact_scan": final_artifact_scan,
        "artifact_scan_error": artifact_scan_error,
        "observability_summary": _observability_summary(
            train_result=train_result,
            artifact_scan=final_artifact_scan,
        ),
        "rom_unblocker": {
            "license_acceptance": ATARI_ROM_LICENSE_NOTICE,
            "modal_image_step": [
                'uv_pip_install("AutoROM[accept-rom-license]==0.6.1")',
                'run_commands("AutoROM --accept-license")',
            ],
        },
        "note": (
            "Exact installed-package mode mutates only exp_name and uses max_env_step=200000. "
            "Train mode uses a relative exp_name from /runs so checkpoints persist in the "
            "Modal Volume even if DI-engine prepends './'. Faithful-short mode is not an "
            "exact reproduction: it keeps stock installed config but additionally overrides "
            "train_muzero max_env_step. The optional save_ckpt_after_iter override changes "
            "only checkpoint cadence for future short runs. When unset, both modes keep the "
            "stock checkpoint cadence. Neither mode caps episodes, reduces "
            "collector/evaluator counts, changes update_per_collect, "
            "or passes max_train_iter unless the explicit optimizer-only "
            "max_train_iter_override is set. Runs with that override are "
            "profile/control runs, not exact reproductions. A positive "
            "survival_reward_per_step is a separate shaped-objective ablation: "
            "it switches create_config.env.type to atari_lightzero_survival_shaped, "
            "requires run and attempt ids containing 'survival-shaped', and must be "
            "evaluated with unshaped control evals."
        ),
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    summary_ref = None
    if mode == "dry":
        summary_path = (
            runs.volume_path(RUNS_MOUNT, runs.attempt_root_ref(TASK_ID, run_id, attempt_id))
            / "dry_exact_summary.json"
        )
        result["artifact_refs"] = {"dry_summary": _write_json_artifact(summary_path, result)}
        summary_ref = result["artifact_refs"]["dry_summary"]["ref"]
    else:
        summary_path = (
            runs.volume_path(RUNS_MOUNT, runs.attempt_train_ref(TASK_ID, run_id, attempt_id))
            / "summary.json"
        )
        result["artifact_refs"] = {"summary": _write_json_artifact(summary_path, result)}
        summary_ref = result["artifact_refs"]["summary"]["ref"]

    attempt_path = runs.volume_path(RUNS_MOUNT, runs.attempt_manifest_ref(TASK_ID, run_id, attempt_id))
    latest_path = runs.volume_path(RUNS_MOUNT, runs.latest_attempt_ref(TASK_ID, run_id))
    status = "completed" if result["ok"] else "failed"
    _write_json_artifact(
        attempt_path,
        runs.attempt_manifest(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status=status,
            started_at=runs.utc_timestamp(),
            ended_at=runs.utc_timestamp(),
            modal_task_id=os.environ.get("MODAL_TASK_ID"),
            summary_ref=summary_ref,
            config={
                "mode": mode,
                "compute": compute,
                "seed": seed,
                "lightzero_version": LIGHTZERO_VERSION,
                "module": module_name,
                "run_kind": run_kind,
                "stock_max_env_step": stock_max_env_step,
                "actual_max_env_step": actual_max_env_step,
                "max_env_step_override": max_env_step_override,
                "max_train_iter_override": max_train_iter_override,
                "iteration_cap_kind": iteration_cap_kind,
                "reward_shaping": {
                    "enabled": survival_shaping_enabled,
                    "schema_id": REWARD_SHAPING_SCHEMA_ID if survival_shaping_enabled else None,
                    "mode": "per_step_survival" if survival_shaping_enabled else "none",
                    "survival_reward_per_step": float(survival_reward_per_step),
                },
                "stock_save_ckpt_after_iter": stock_save_ckpt_after_iter,
                "actual_save_ckpt_after_iter": actual_save_ckpt_after_iter,
                "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
                "checkpoint_cadence_kind": checkpoint_cadence_kind,
                "profile_phases": profile_phases,
                "gpu_sample_interval_sec": gpu_sample_interval_sec,
                "profile_stop_after_learner_train_calls": profile_stop_after_learner_train_calls,
            },
        ),
    )
    _write_json_artifact(
        latest_path,
        runs.latest_attempt_pointer(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status=status,
            started_at=runs.utc_timestamp(),
            ended_at=runs.utc_timestamp(),
            modal_task_id=os.environ.get("MODAL_TASK_ID"),
            summary_ref=summary_ref,
        ),
    )
    runs_volume.commit()
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=8 * 60, cpu=1.0)
def lightzero_pong_exact_reproduction_cpu(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    max_env_step_override: int | None = None,
    max_train_iter_override: int | None = None,
    save_ckpt_after_iter_override: int | None = None,
    survival_reward_per_step: float = DEFAULT_SURVIVAL_REWARD_PER_STEP,
    profile_phases: bool = DEFAULT_PROFILE_PHASES,
    gpu_sample_interval_sec: float = DEFAULT_GPU_SAMPLE_INTERVAL_SEC,
    profile_stop_after_learner_train_calls: int | None = None,
) -> dict[str, Any]:
    return _run_exact_reproduction(
        mode=mode,
        compute="cpu",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_interval_sec=progress_interval_sec,
        max_env_step_override=max_env_step_override,
        max_train_iter_override=max_train_iter_override,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
        survival_reward_per_step=survival_reward_per_step,
        profile_phases=profile_phases,
        gpu_sample_interval_sec=gpu_sample_interval_sec,
        profile_stop_after_learner_train_calls=profile_stop_after_learner_train_calls,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=18 * 60 * 60,
    cpu=8.0,
    memory=32768,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_pong_exact_reproduction_gpu(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    max_env_step_override: int | None = None,
    max_train_iter_override: int | None = None,
    save_ckpt_after_iter_override: int | None = None,
    survival_reward_per_step: float = DEFAULT_SURVIVAL_REWARD_PER_STEP,
    profile_phases: bool = DEFAULT_PROFILE_PHASES,
    gpu_sample_interval_sec: float = DEFAULT_GPU_SAMPLE_INTERVAL_SEC,
    profile_stop_after_learner_train_calls: int | None = None,
) -> dict[str, Any]:
    return _run_exact_reproduction(
        mode=mode,
        compute="gpu-l4-t4",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_interval_sec=progress_interval_sec,
        max_env_step_override=max_env_step_override,
        max_train_iter_override=max_train_iter_override,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
        survival_reward_per_step=survival_reward_per_step,
        profile_phases=profile_phases,
        gpu_sample_interval_sec=gpu_sample_interval_sec,
        profile_stop_after_learner_train_calls=profile_stop_after_learner_train_calls,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=18 * 60 * 60,
    cpu=16.0,
    memory=32768,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_pong_exact_reproduction_gpu_cpu16(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    max_env_step_override: int | None = None,
    max_train_iter_override: int | None = None,
    save_ckpt_after_iter_override: int | None = None,
    survival_reward_per_step: float = DEFAULT_SURVIVAL_REWARD_PER_STEP,
    profile_phases: bool = DEFAULT_PROFILE_PHASES,
    gpu_sample_interval_sec: float = DEFAULT_GPU_SAMPLE_INTERVAL_SEC,
    profile_stop_after_learner_train_calls: int | None = None,
) -> dict[str, Any]:
    return _run_exact_reproduction(
        mode=mode,
        compute="gpu-l4-t4-cpu16",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_interval_sec=progress_interval_sec,
        max_env_step_override=max_env_step_override,
        max_train_iter_override=max_train_iter_override,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
        survival_reward_per_step=survival_reward_per_step,
        profile_phases=profile_phases,
        gpu_sample_interval_sec=gpu_sample_interval_sec,
        profile_stop_after_learner_train_calls=profile_stop_after_learner_train_calls,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=18 * 60 * 60,
    cpu=40.0,
    memory=32768,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_pong_exact_reproduction_gpu_cpu40(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    max_env_step_override: int | None = None,
    max_train_iter_override: int | None = None,
    save_ckpt_after_iter_override: int | None = None,
    survival_reward_per_step: float = DEFAULT_SURVIVAL_REWARD_PER_STEP,
    profile_phases: bool = DEFAULT_PROFILE_PHASES,
    gpu_sample_interval_sec: float = DEFAULT_GPU_SAMPLE_INTERVAL_SEC,
    profile_stop_after_learner_train_calls: int | None = None,
) -> dict[str, Any]:
    return _run_exact_reproduction(
        mode=mode,
        compute="gpu-l4-t4-cpu40",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_interval_sec=progress_interval_sec,
        max_env_step_override=max_env_step_override,
        max_train_iter_override=max_train_iter_override,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
        survival_reward_per_step=survival_reward_per_step,
        profile_phases=profile_phases,
        gpu_sample_interval_sec=gpu_sample_interval_sec,
        profile_stop_after_learner_train_calls=profile_stop_after_learner_train_calls,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=18 * 60 * 60,
    cpu=8.0,
    memory=32768,
    gpu=H100_GPU_RESOURCE,
)
def lightzero_pong_exact_reproduction_gpu_h100(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    max_env_step_override: int | None = None,
    max_train_iter_override: int | None = None,
    save_ckpt_after_iter_override: int | None = None,
    survival_reward_per_step: float = DEFAULT_SURVIVAL_REWARD_PER_STEP,
    profile_phases: bool = DEFAULT_PROFILE_PHASES,
    gpu_sample_interval_sec: float = DEFAULT_GPU_SAMPLE_INTERVAL_SEC,
    profile_stop_after_learner_train_calls: int | None = None,
) -> dict[str, Any]:
    return _run_exact_reproduction(
        mode=mode,
        compute="gpu-h100",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_interval_sec=progress_interval_sec,
        max_env_step_override=max_env_step_override,
        max_train_iter_override=max_train_iter_override,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
        survival_reward_per_step=survival_reward_per_step,
        profile_phases=profile_phases,
        gpu_sample_interval_sec=gpu_sample_interval_sec,
        profile_stop_after_learner_train_calls=profile_stop_after_learner_train_calls,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=18 * 60 * 60,
    cpu=16.0,
    memory=32768,
    gpu=H100_GPU_RESOURCE,
)
def lightzero_pong_exact_reproduction_gpu_h100_cpu16(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    max_env_step_override: int | None = None,
    max_train_iter_override: int | None = None,
    save_ckpt_after_iter_override: int | None = None,
    survival_reward_per_step: float = DEFAULT_SURVIVAL_REWARD_PER_STEP,
    profile_phases: bool = DEFAULT_PROFILE_PHASES,
    gpu_sample_interval_sec: float = DEFAULT_GPU_SAMPLE_INTERVAL_SEC,
    profile_stop_after_learner_train_calls: int | None = None,
) -> dict[str, Any]:
    return _run_exact_reproduction(
        mode=mode,
        compute="gpu-h100-cpu16",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_interval_sec=progress_interval_sec,
        max_env_step_override=max_env_step_override,
        max_train_iter_override=max_train_iter_override,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
        survival_reward_per_step=survival_reward_per_step,
        profile_phases=profile_phases,
        gpu_sample_interval_sec=gpu_sample_interval_sec,
        profile_stop_after_learner_train_calls=profile_stop_after_learner_train_calls,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=18 * 60 * 60,
    cpu=40.0,
    memory=32768,
    gpu=H100_GPU_RESOURCE,
)
def lightzero_pong_exact_reproduction_gpu_h100_cpu40(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    max_env_step_override: int | None = None,
    max_train_iter_override: int | None = None,
    save_ckpt_after_iter_override: int | None = None,
    survival_reward_per_step: float = DEFAULT_SURVIVAL_REWARD_PER_STEP,
    profile_phases: bool = DEFAULT_PROFILE_PHASES,
    gpu_sample_interval_sec: float = DEFAULT_GPU_SAMPLE_INTERVAL_SEC,
    profile_stop_after_learner_train_calls: int | None = None,
) -> dict[str, Any]:
    return _run_exact_reproduction(
        mode=mode,
        compute="gpu-h100-cpu40",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_interval_sec=progress_interval_sec,
        max_env_step_override=max_env_step_override,
        max_train_iter_override=max_train_iter_override,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
        survival_reward_per_step=survival_reward_per_step,
        profile_phases=profile_phases,
        gpu_sample_interval_sec=gpu_sample_interval_sec,
        profile_stop_after_learner_train_calls=profile_stop_after_learner_train_calls,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=18 * 60 * 60,
    cpu=8.0,
    memory=32768,
    gpu=H100X4_GPU_RESOURCE,
)
def lightzero_pong_exact_reproduction_gpu_h100x4(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    max_env_step_override: int | None = None,
    max_train_iter_override: int | None = None,
    save_ckpt_after_iter_override: int | None = None,
    survival_reward_per_step: float = DEFAULT_SURVIVAL_REWARD_PER_STEP,
    profile_phases: bool = DEFAULT_PROFILE_PHASES,
    gpu_sample_interval_sec: float = DEFAULT_GPU_SAMPLE_INTERVAL_SEC,
    profile_stop_after_learner_train_calls: int | None = None,
) -> dict[str, Any]:
    return _run_exact_reproduction(
        mode=mode,
        compute="gpu-h100x4",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_interval_sec=progress_interval_sec,
        max_env_step_override=max_env_step_override,
        max_train_iter_override=max_train_iter_override,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
        survival_reward_per_step=survival_reward_per_step,
        profile_phases=profile_phases,
        gpu_sample_interval_sec=gpu_sample_interval_sec,
        profile_stop_after_learner_train_calls=profile_stop_after_learner_train_calls,
    )


@app.local_entrypoint()
def main(
    mode: str = DEFAULT_MODE,
    compute: str = DEFAULT_COMPUTE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    max_env_step_override: int | None = None,
    max_train_iter_override: int | None = None,
    save_ckpt_after_iter_override: int | None = None,
    survival_reward_per_step: float = DEFAULT_SURVIVAL_REWARD_PER_STEP,
    profile_phases: bool = DEFAULT_PROFILE_PHASES,
    gpu_sample_interval_sec: float = DEFAULT_GPU_SAMPLE_INTERVAL_SEC,
    profile_stop_after_learner_train_calls: int | None = None,
    wait_for_train: bool = False,
) -> None:
    if compute == "cpu":
        train_fn = lightzero_pong_exact_reproduction_cpu
    elif compute == "gpu-l4-t4":
        train_fn = lightzero_pong_exact_reproduction_gpu
    elif compute == "gpu-l4-t4-cpu16":
        train_fn = lightzero_pong_exact_reproduction_gpu_cpu16
    elif compute == "gpu-l4-t4-cpu40":
        train_fn = lightzero_pong_exact_reproduction_gpu_cpu40
    elif compute == "gpu-h100":
        train_fn = lightzero_pong_exact_reproduction_gpu_h100
    elif compute == "gpu-h100-cpu16":
        train_fn = lightzero_pong_exact_reproduction_gpu_h100_cpu16
    elif compute == "gpu-h100-cpu40":
        train_fn = lightzero_pong_exact_reproduction_gpu_h100_cpu40
    elif compute == "gpu-h100x4":
        train_fn = lightzero_pong_exact_reproduction_gpu_h100x4
    else:
        raise ValueError(
            f"unknown compute: {compute!r}; expected one of: {_compute_choices_label()}"
        )
    call_kwargs = {
        "mode": mode,
        "seed": seed,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "progress_interval_sec": progress_interval_sec,
        "max_env_step_override": max_env_step_override,
        "max_train_iter_override": max_train_iter_override,
        "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
        "survival_reward_per_step": survival_reward_per_step,
        "profile_phases": profile_phases,
        "gpu_sample_interval_sec": gpu_sample_interval_sec,
        "profile_stop_after_learner_train_calls": profile_stop_after_learner_train_calls,
    }
    if mode == "train" and not wait_for_train:
        function_call = train_fn.spawn(**call_kwargs)
        call_id = getattr(function_call, "object_id", None) or getattr(function_call, "id", None)
        print(
            json.dumps(
                {
                    "schema": "curvyzero_lightzero_pong_background_launch/v1",
                    "status": "spawned",
                    "mode": mode,
                    "compute": compute,
                    "seed": seed,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "survival_reward_per_step": survival_reward_per_step,
                    "reward_shaping_enabled": survival_reward_per_step > 0.0,
                    "function_call_id": call_id,
                    "progress_ref": (
                        runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
                        / "progress"
                        / "latest.json"
                    ).as_posix(),
                    "note": (
                        "Training was launched with Modal Function.spawn so the "
                        "remote function call is not waited on by this local entrypoint. "
                        "Use --wait-for-train for short profiling runs that should return "
                        "a complete summary."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = train_fn.remote(**call_kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
