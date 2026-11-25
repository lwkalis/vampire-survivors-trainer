"""Run a before-and-after checkpoint comparison from the dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess
import uuid


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHECKPOINT = DATA / "visual_dqn_inlaid_library_8way.pt"
CHECKS_DIR = DATA / "learning-checks"
PHASES = {"baseline": 10, "training": 20, "followup": 10}


def latest_check() -> dict | None:
    try:
        state = json.loads((CHECKS_DIR / "latest.json").read_text(encoding="utf-8"))
        if isinstance(state, dict) and re.fullmatch(r"[0-9a-f]{12}", str(state.get("id"))):
            return state
    except (OSError, ValueError):
        pass
    return None


def phase_report(state: dict) -> Path | None:
    phase = state.get("phase")
    if phase in PHASES:
        return CHECKS_DIR / state["id"] / f"{phase}.jsonl"
    return None


def summarize(report: Path, target: int) -> dict:
    runs = {}
    if report.is_file():
        with report.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if (isinstance(record, dict) and record.get("event") == "run_completed"
                        and isinstance(record.get("run_id"), str)):
                    runs[record["run_id"]] = record
    timers = [record["game_seconds"] for record in runs.values()
              if type(record.get("game_seconds")) is int]
    kills = [record["kills"] for record in runs.values() if type(record.get("kills")) is int]
    return {
        "completed": len(runs),
        "target": target,
        "timed_runs": len(timers),
        "median_seconds": median(timers) if timers else None,
        "shortest_seconds": min(timers) if timers else None,
        "longest_seconds": max(timers) if timers else None,
        "median_kills": median(kills) if kills else None,
    }


def run_phase(report: Path, checkpoint: Path, policy_dir: Path, target: int, *, evaluate: bool) -> int:
    command = [
        "bash", str(ROOT / "run-local.sh"),
        "--evaluate" if evaluate else "--full-run",
        "--deep-learning", "--max-runs", str(target),
        "--report-path", str(report),
        "--evidence-dir", str(report.with_suffix("")), "--evidence-interval", "60",
        "--checkpoint", str(checkpoint), "--policy-dir", str(policy_dir),
    ]
    with subprocess.Popen(command, cwd=ROOT) as trainer:
        try:
            return trainer.wait()
        except KeyboardInterrupt:
            # The dashboard interrupts the whole process group. Wait for the
            # child to release keys and finish saving before leaving the phase.
            trainer.wait()
            raise


def main() -> int:
    check_id = uuid.uuid4().hex[:12]
    folder = CHECKS_DIR / check_id
    folder.mkdir(parents=True, exist_ok=False)
    baseline = folder / "baseline.pt"
    trained = folder / "trained.pt"
    frozen_policy = folder / "evaluation-policy"
    training_policy = folder / "training-policy"
    state = {"id": check_id, "pid": os.getpid(), "status": "running", "phase": "baseline",
             "targets": PHASES, "results": {}}

    def publish() -> None:
        temporary = CHECKS_DIR / f".{check_id}.tmp"
        temporary.write_text(json.dumps(state, allow_nan=False), encoding="utf-8")
        temporary.replace(CHECKS_DIR / "latest.json")

    publish()
    try:
        shutil.copyfile(CHECKPOINT, baseline)
        shutil.copyfile(baseline, trained)
        for policy_dir in (frozen_policy, training_policy):
            policy_dir.mkdir()
            for filename in ("vision_policy.json", "upgrade_policy.json"):
                source = DATA / filename
                if source.is_file():
                    shutil.copyfile(source, policy_dir / filename)
        for phase, target in PHASES.items():
            state["phase"] = phase
            publish()
            report = folder / f"{phase}.jsonl"
            evaluate = phase != "training"
            checkpoint = baseline if phase == "baseline" else trained
            print(f"learning-check: {phase}, {target} runs, checkpoint={checkpoint}", flush=True)
            try:
                status = run_phase(report, checkpoint, frozen_policy if evaluate else training_policy,
                                   target, evaluate=evaluate)
            finally:
                state["results"][phase] = summarize(report, target)
            if status or state["results"][phase]["completed"] != target:
                raise RuntimeError(f"The {phase} phase ended before completing {target} runs.")
        state["status"] = "complete"
        return 0
    except KeyboardInterrupt:
        state["status"] = "stopped"
        return 130
    except (OSError, RuntimeError) as exc:
        state["status"] = "failed"
        state["message"] = str(exc)
        print(f"learning-check: {exc}", flush=True)
        return 1
    finally:
        publish()


if __name__ == "__main__":
    raise SystemExit(main())
