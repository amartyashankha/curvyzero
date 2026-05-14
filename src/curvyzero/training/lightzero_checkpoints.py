"""Pure helpers for LightZero checkpoint and resume-state paths."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

LIGHTZERO_EXP_DIR_PREFIX = "lightzero_exp"
LIGHTZERO_CHECKPOINT_DIR_NAME = "ckpt"
LIGHTZERO_RESUME_STATE_DIRNAME = "lightzero_resume_state"


@dataclass(frozen=True)
class LightZeroCheckpointCandidate:
    iteration: int
    path: Path
    mtime: float
    mtime_ns: int
    size_bytes: int

    @property
    def checkpoint_name(self) -> str:
        return self.path.name

    @property
    def exp_dir_name(self) -> str:
        return self.path.parent.parent.name

    def latest_sort_key(self) -> tuple[int, int, int, str]:
        return (self.iteration, self.mtime_ns, self.size_bytes, str(self.path))


@dataclass(frozen=True)
class LightZeroResumeStateCandidate:
    iteration: int
    path: Path
    mtime: float
    mtime_ns: int
    size_bytes: int

    @property
    def state_name(self) -> str:
        return self.path.name

    @property
    def exp_dir_name(self) -> str:
        return self.path.parent.parent.name

    def latest_sort_key(self) -> tuple[int, int, str]:
        return (self.mtime_ns, self.size_bytes, str(self.path))


def lightzero_exp_sibling_roots(exp_name: Path) -> list[Path]:
    """Return the configured LightZero exp dir plus timestamped sibling dirs."""

    roots: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    add(exp_name)
    if exp_name.name.startswith(LIGHTZERO_EXP_DIR_PREFIX) and exp_name.parent.is_dir():
        for sibling in sorted(
            exp_name.parent.glob(f"{LIGHTZERO_EXP_DIR_PREFIX}*"),
            key=lambda path: path.name,
        ):
            if sibling.is_dir():
                add(sibling)
    return roots


def lightzero_exp_checkpoint_dirs(exp_name: Path) -> list[Path]:
    return [
        root / LIGHTZERO_CHECKPOINT_DIR_NAME
        for root in lightzero_exp_sibling_roots(exp_name)
    ]


def lightzero_exp_resume_state_dirs(exp_name: Path) -> list[Path]:
    return [
        root / LIGHTZERO_RESUME_STATE_DIRNAME
        for root in lightzero_exp_sibling_roots(exp_name)
    ]


def collect_lightzero_iteration_checkpoints(
    checkpoint_dirs: Iterable[Path],
    *,
    require_non_empty: bool = False,
) -> list[LightZeroCheckpointCandidate]:
    candidates: list[LightZeroCheckpointCandidate] = []
    seen_dirs: set[str] = set()
    for checkpoint_dir in checkpoint_dirs:
        directory_key = str(checkpoint_dir)
        if directory_key in seen_dirs or not checkpoint_dir.is_dir():
            continue
        seen_dirs.add(directory_key)
        for path in checkpoint_dir.glob("iteration_*.pth.tar"):
            iteration = lightzero_iteration_from_checkpoint_name(path.name)
            if iteration is None or not path.is_file():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if require_non_empty and stat.st_size <= 0:
                continue
            candidates.append(
                LightZeroCheckpointCandidate(
                    iteration=iteration,
                    path=path,
                    mtime=stat.st_mtime,
                    mtime_ns=stat.st_mtime_ns,
                    size_bytes=stat.st_size,
                )
            )
    return candidates


def latest_lightzero_iteration_checkpoint(
    candidates: Iterable[LightZeroCheckpointCandidate],
) -> LightZeroCheckpointCandidate | None:
    candidates_list = list(candidates)
    if not candidates_list:
        return None
    return max(candidates_list, key=lambda candidate: candidate.latest_sort_key())


def latest_lightzero_iteration_checkpoint_from_dirs(
    checkpoint_dirs: Iterable[Path],
    *,
    require_non_empty: bool = False,
) -> LightZeroCheckpointCandidate | None:
    return latest_lightzero_iteration_checkpoint(
        collect_lightzero_iteration_checkpoints(
            checkpoint_dirs,
            require_non_empty=require_non_empty,
        )
    )


def collect_lightzero_resume_state_candidates(
    resume_state_dirs: Iterable[Path],
    *,
    iteration: int,
    require_non_empty: bool = True,
) -> list[LightZeroResumeStateCandidate]:
    candidates: list[LightZeroResumeStateCandidate] = []
    seen_dirs: set[str] = set()
    state_name = f"iteration_{int(iteration)}.resume_state.pkl"
    for state_dir in resume_state_dirs:
        directory_key = str(state_dir)
        if directory_key in seen_dirs or not state_dir.is_dir():
            continue
        seen_dirs.add(directory_key)
        path = state_dir / state_name
        if not path.is_file():
            continue
        parsed_iteration = lightzero_iteration_from_resume_state_name(path.name)
        if parsed_iteration != int(iteration):
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if require_non_empty and stat.st_size <= 0:
            continue
        candidates.append(
            LightZeroResumeStateCandidate(
                iteration=int(iteration),
                path=path,
                mtime=stat.st_mtime,
                mtime_ns=stat.st_mtime_ns,
                size_bytes=stat.st_size,
            )
        )
    return candidates


def latest_lightzero_resume_state_candidate(
    candidates: Iterable[LightZeroResumeStateCandidate],
) -> LightZeroResumeStateCandidate | None:
    candidates_list = list(candidates)
    if not candidates_list:
        return None
    return max(candidates_list, key=lambda candidate: candidate.latest_sort_key())


def lightzero_iteration_from_checkpoint_name(name: str) -> int | None:
    return _iteration_from_name(name, prefix="iteration_", suffix=".pth.tar")


def lightzero_iteration_from_resume_state_name(name: str) -> int | None:
    return _iteration_from_name(name, prefix="iteration_", suffix=".resume_state.pkl")


def _iteration_from_name(name: str, *, prefix: str, suffix: str) -> int | None:
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    middle = name[len(prefix) : -len(suffix)]
    if not middle.isdigit():
        return None
    return int(middle)
