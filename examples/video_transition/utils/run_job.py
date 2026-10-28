"""Resume a frozen job's training, generation, and evaluation stages under a lock."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hermit.io import atomic_json
from examples.video_transition.utils.artifacts import completed_case, file_sha256


def verify_snapshot(run: Path, plan: dict) -> None:
    for relative, digest in plan["sha256"].items():
        if file_sha256(run / relative) != digest:
            raise ValueError(f"frozen experiment file changed: {relative}")


def stage_complete(stage: dict) -> bool:
    try:
        value = json.loads(Path(stage["completion"]).read_text())
        kind = stage["kind"]
        if kind == "train":
            return value.get("stop_reason") == "completed all epochs" and value.get("steps") == stage["expected_steps"]
        if kind == "eval":
            return value.get("errors") == 0 and value.get("tasks") == stage["expected_tasks"]
        if kind == "generation":
            run = Path(stage["completion"]).parent
            if value.get("status") != "complete" or value.get("tasks") != [t["id"] for t in stage["tasks"]]:
                return False
            return all(completed_case(run / task["id"], task, stage["mode"], save_latents=True)
                       for task in stage["tasks"])
        return value.get("status") in stage.get("success_status", ["complete"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False


def recover_git_locks(run: Path) -> None:
    """Under the experiment lock, retain locks left by a killed Git process elsewhere."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for metadata in (run / "hermit").rglob(".git"):
        if not metadata.is_dir():
            continue
        for stale in metadata.rglob("*.lock"):
            target = run / "recovered-git-locks" / stamp / stale.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stale), target)


def run_job(run: Path) -> None:
    run = run.resolve()
    plan = json.loads((run / "plan.json").read_text())
    verify_snapshot(run, plan)
    with (run / ".job.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another process is already executing this experiment") from None
        if plan["mode"] == "evolve":
            recover_git_locks(run)
            for name in ("git", "tar"):
                if not shutil.which(name):
                    raise RuntimeError(f"{name} is required on the compute node; configure generator.environment.PATH")
            subprocess.run(["git", "--version"], check=True)
        progress = {"status": "running", "job_id": os.environ.get("SLURM_JOB_ID"),
                    "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"), "stages": []}
        for stage in plan["stages"]:
            record = {"id": stage["id"], "cached": stage_complete(stage)}
            progress["stages"].append(record)
            if record["cached"]:
                record["status"] = "complete"
            else:
                record.update(status="running", started=time.time())
                atomic_json(run / "progress.json", progress)
                result = subprocess.run(stage["command"], cwd=run)
                record.update(returncode=result.returncode, finished=time.time())
                if result.returncode or not stage_complete(stage):
                    record["status"] = "failed"
                    progress["status"] = "failed"
                    atomic_json(run / "progress.json", progress)
                    raise RuntimeError(f"stage {stage['id']} failed; resume this run after resolving its error")
                record["status"] = "complete"
            atomic_json(run / "progress.json", progress)
        progress["status"] = "complete"
        atomic_json(run / "progress.json", progress)
        atomic_json(run / "job-complete.json", progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_job(args.run_dir)


if __name__ == "__main__":
    main()
