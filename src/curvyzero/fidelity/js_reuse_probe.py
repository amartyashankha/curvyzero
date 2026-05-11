"""Subprocess helper for the original CurvyTron JavaScript reuse probe."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
JS_REUSE_ENV_CLI = _REPO_ROOT / "tools" / "js_reuse_probe" / "curvytron_env_cli.js"
JS_REUSE_ENV_WORKER = _REPO_ROOT / "tools" / "js_reuse_probe" / "curvytron_env_worker.js"


class JsReuseProbeError(RuntimeError):
    """Raised when the JavaScript reuse probe exits unsuccessfully."""


class CurvytronJsEnvWorker:
    """Line-json subprocess wrapper around the persistent original-JS env worker."""

    def __init__(
        self,
        *,
        node: str = "node",
        cwd: str | Path | None = None,
    ) -> None:
        self._cwd = Path(cwd) if cwd is not None else _REPO_ROOT
        self._process = subprocess.Popen(
            [node, str(JS_REUSE_ENV_WORKER)],
            cwd=self._cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._next_id = 1
        self.last_response: dict[str, Any] | None = None
        self.ready = self._read_startup()

    def __enter__(self) -> "CurvytronJsEnvWorker":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def source_load_count(self) -> int:
        if self.last_response is not None:
            return int(self.last_response["sourceLoadCount"])
        return int(self.ready["sourceLoadCount"])

    def reset(self, scenario: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(scenario, dict):
            return self._request({"cmd": "reset", "scenario": scenario})["result"]
        return self._request({"cmd": "reset", "scenarioPath": str(Path(scenario))})["result"]

    def step(self, step: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
        command: dict[str, Any] = {"cmd": "step"}
        if step is not None:
            command["step"] = step
        command.update(fields)
        return self._request(command)["result"]

    def snapshot(self) -> dict[str, Any]:
        return self._request({"cmd": "snapshot"})["result"]

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self._request({"cmd": "close"})
        finally:
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)

    def _read_startup(self) -> dict[str, Any]:
        line = self._readline()
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise JsReuseProbeError(f"worker emitted invalid startup JSON: {line}") from error
        if not payload.get("ok"):
            raise JsReuseProbeError(str(payload))
        return payload

    def _request(self, command: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        command = dict(command)
        command["id"] = request_id

        if self._process.stdin is None:
            raise JsReuseProbeError("worker stdin is closed")
        self._process.stdin.write(json.dumps(command) + "\n")
        self._process.stdin.flush()

        line = self._readline()
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise JsReuseProbeError(f"worker emitted invalid JSON: {line}") from error

        if payload.get("id") != request_id:
            raise JsReuseProbeError(
                f"worker response id mismatch: expected {request_id}, got {payload.get('id')}"
            )
        if not payload.get("ok"):
            error = payload.get("error") or {}
            message = error.get("message") or str(payload)
            raise JsReuseProbeError(f"{error.get('name', 'Error')}: {message}")
        self.last_response = payload
        return payload

    def _readline(self) -> str:
        if self._process.stdout is None:
            raise JsReuseProbeError("worker stdout is closed")
        line = self._process.stdout.readline()
        if line:
            return line

        stderr = ""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read().strip()
        raise JsReuseProbeError(stderr or "worker exited without a response")


def run_js_reuse_env_probe(
    scenario: str | Path,
    *,
    node: str = "node",
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Run one source scenario through the original-JS env-shaped probe."""

    scenario_path = Path(scenario)
    run_cwd = Path(cwd) if cwd is not None else _REPO_ROOT
    result = subprocess.run(
        [node, str(JS_REUSE_ENV_CLI), str(scenario_path)],
        cwd=run_cwd,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        try:
            error = json.loads(message)["error"]
            message = f"{error.get('name', 'Error')}: {error.get('message', message)}"
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        raise JsReuseProbeError(message)

    return json.loads(result.stdout)
