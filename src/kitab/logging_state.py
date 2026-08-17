"""Structured run logging and state for kitAb pipelines."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    """Human-readable run.log + machine-readable events.jsonl + state/summary."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.state_dir = self.output_dir / "state"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.run_log = self.logs_dir / "run.log"
        self.events_path = self.logs_dir / "events.jsonl"
        self.failures_path = self.logs_dir / "failures.tsv"
        self.summary_path = self.state_dir / "summary.json"
        self.state_path = self.state_dir / "run_state.json"
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._stage_status: dict[str, str] = {}
        self._failures: list[dict[str, str]] = []
        if not self.failures_path.exists():
            self.failures_path.write_text(
                "stage\tdataset\titem\tattempt\treason\n", encoding="utf-8"
            )

    def info(self, message: str) -> None:
        self._write_line(f"[kitab] {message}")

    def error(self, message: str) -> None:
        self._write_line(f"[kitab] ERROR: {message}")

    def event(
        self,
        *,
        stage: str,
        status: str,
        dataset: str = "",
        item: str = "",
        attempt: int | None = None,
        command: str = "",
        message: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "ts": _utc_now(),
            "stage": stage,
            "status": status,
            "dataset": dataset,
            "item": item,
            "attempt": attempt,
            "command": command,
            "message": message,
        }
        if extra:
            payload["extra"] = extra
        with self._lock:
            self._events.append(payload)
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._stage_status[stage] = status
            self._write_line(
                f"[{stage}] {status}"
                + (f" dataset={dataset}" if dataset else "")
                + (f" item={item}" if item else "")
                + (f" attempt={attempt}" if attempt is not None else "")
                + (f" — {message}" if message else "")
            )
            self._flush_state()

    def failure(
        self,
        *,
        stage: str,
        dataset: str,
        item: str,
        reason: str,
        attempt: int | None = None,
    ) -> None:
        row = {
            "stage": stage,
            "dataset": dataset,
            "item": item,
            "attempt": "" if attempt is None else str(attempt),
            "reason": reason.replace("\t", " ").replace("\n", " "),
        }
        with self._lock:
            self._failures.append(row)
            with open(self.failures_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{row['stage']}\t{row['dataset']}\t{row['item']}\t"
                    f"{row['attempt']}\t{row['reason']}\n"
                )
            self._write_line(
                f"[kitab] FAILURE stage={stage} dataset={dataset} item={item}: {reason}"
            )
            self._flush_state()

    def write_summary(self, summary: dict[str, Any]) -> Path:
        payload = dict(summary)
        payload["updated_at"] = _utc_now()
        payload["n_failures"] = len(self._failures)
        payload["stage_status"] = dict(self._stage_status)
        with self._lock:
            self.summary_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._flush_state()
        return self.summary_path

    def _flush_state(self) -> None:
        state = {
            "updated_at": _utc_now(),
            "stage_status": dict(self._stage_status),
            "n_events": len(self._events),
            "n_failures": len(self._failures),
            "failures": list(self._failures),
        }
        self.state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _write_line(self, line: str) -> None:
        with open(self.run_log, "a", encoding="utf-8") as f:
            f.write(f"{_utc_now()} {line}\n")
        print(line, flush=True)
