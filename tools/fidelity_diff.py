"""Tiny JSON diff for trace plumbing.

This reports the first mismatch only. It has no numeric tolerance or CurvyTron
schema knowledge yet, but treats equal JSON numbers as equal even when Python
loaded one as ``int`` and the other as ``float``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

try:
    from curvyzero.env.trace_compare import TraceCompareError, normalize_trace_payload
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from curvyzero.env.trace_compare import TraceCompareError, normalize_trace_payload


class DiffPayload(dict[str, object]):
    """Diff payload with compatibility for older exact dict comparisons."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict) and "status" not in other and "status" in self:
            legacy = dict(self)
            legacy.pop("status", None)
            return legacy == other
        return super().__eq__(other)


@dataclass(frozen=True, slots=True)
class FirstMismatch:
    path: str
    left: Any
    right: Any
    reason: str

    def to_payload(self) -> dict[str, object]:
        return _status_payload(
            status="fail",
            match=False,
            message=self.describe(),
            path=self.path,
            left=self.left,
            right=self.right,
            reason=self.reason,
        )

    def describe(self) -> str:
        return (
            f"First mismatch at {self.path}: "
            f"left is {_short_value(self.left)}, right is {_short_value(self.right)} "
            f"({self.reason})."
        )


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_mismatch(left: Any, right: Any) -> FirstMismatch | None:
    return _first_mismatch(left, right, "$")


def diff_payload(left: Any, right: Any, *, common_trace: bool = False) -> dict[str, object]:
    if common_trace:
        try:
            left = normalize_trace_payload(left)
            right = normalize_trace_payload(right)
        except TraceCompareError as error:
            return _blocked_payload(
                "trace normalization error",
                f"Trace normalization error: {error}",
            )

    mismatch = first_mismatch(left, right)
    if mismatch is None:
        message = "Common trace payloads match." if common_trace else "JSON payloads match exactly."
        return _status_payload(status="pass", match=True, message=message)
    return mismatch.to_payload()


def _status_payload(
    *,
    status: str,
    match: bool,
    message: str,
    **extra: object,
) -> DiffPayload:
    payload = DiffPayload({"match": match, "status": status, "message": message})
    payload.update(extra)
    return payload


def _blocked_payload(reason: str, message: str) -> DiffPayload:
    return _status_payload(status="blocked", match=False, reason=reason, message=message)


def _first_mismatch(left: Any, right: Any, path: str) -> FirstMismatch | None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys | right_keys, key=str):
            child_path = f"{path}.{key}"
            if key not in left:
                return FirstMismatch(child_path, None, right[key], "key missing on left")
            if key not in right:
                return FirstMismatch(child_path, left[key], None, "key missing on right")
            mismatch = _first_mismatch(left[key], right[key], child_path)
            if mismatch is not None:
                return mismatch
        return None

    if isinstance(left, list) and isinstance(right, list):
        shared = min(len(left), len(right))
        for index in range(shared):
            mismatch = _first_mismatch(left[index], right[index], f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        if len(left) != len(right):
            index = shared
            left_value = left[index] if index < len(left) else None
            right_value = right[index] if index < len(right) else None
            return FirstMismatch(
                f"{path}[{index}]",
                left_value,
                right_value,
                "list lengths differ",
            )
        return None

    if _both_json_numbers(left, right):
        if left != right:
            return FirstMismatch(path, left, right, "values differ")
        return None

    if type(left) is not type(right):
        return FirstMismatch(path, left, right, "types differ")

    if left != right:
        return FirstMismatch(path, left, right, "values differ")

    return None


def _both_json_numbers(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    return isinstance(left, int | float) and isinstance(right, int | float)


def _short_value(value: Any) -> str:
    text = json.dumps(value, sort_keys=True)
    if len(text) <= 120:
        return text
    return text[:117] + "..."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report the first exact mismatch between two JSON files."
    )
    parser.add_argument("left", help="Left JSON file")
    parser.add_argument("right", help="Right JSON file")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--common-trace",
        action="store_true",
        help="Normalize JS/Python runner output to the minimal common trace view before diffing",
    )
    args = parser.parse_args(argv)

    try:
        left = load_json(args.left)
        right = load_json(args.right)
    except (OSError, json.JSONDecodeError) as error:
        result = _blocked_payload("invalid input", f"Invalid input: {error}")
        exit_code = 2
    else:
        result = diff_payload(left, right, common_trace=args.common_trace)
        exit_code = 2 if result.get("status") == "blocked" else 0 if result["match"] else 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result["message"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
