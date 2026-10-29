"""Extend a completed or interrupted experiment from its last training checkpoint."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from hermit.io import atomic_json, atomic_text
from hermit.tree import Tree
from examples.video_transition.utils.artifacts import file_sha256
from examples.video_transition.utils.run_job import verify_snapshot


def epochs_for_steps(config: dict, base: Path, target: int) -> int:
    if target < 1:
        raise ValueError("--steps must be positive")
    train = Path(config["data"]["train"])
    train = train if train.is_absolute() else base / train
    count = len(train.read_text().split())
    per_epoch = -(-count // int(config["data"]["step_size"]))
    if not per_epoch or target % per_epoch:
        raise ValueError(f"--steps must be a multiple of {per_epoch} steps per epoch")
    return target // per_epoch


def extend_run(run: Path, target: int, always_propose: bool = False) -> dict:
    run = run.resolve()
    with ExitStack() as stack:
        for name in (".submit.lock", ".job.lock"):
            lock = stack.enter_context((run / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((run / "plan.json").read_text())
        verify_snapshot(run, plan)
        if plan["mode"] != "evolve":
            raise ValueError("only an evolution experiment has training steps")
        train_stage = next(stage for stage in plan["stages"] if stage["kind"] == "train")
        config_path = run / "run.yaml"
        config = yaml.safe_load(config_path.read_text())
        epochs = epochs_for_steps(config, run, target)
        previous_target = train_stage["expected_steps"]
        if target < previous_target:
            raise ValueError("extension cannot reduce the existing step target")
        if target == previous_target:
            return plan
        job_path = run / "job.json"
        if job_path.exists():
            job = json.loads(job_path.read_text())
            active = subprocess.run(["squeue", "--noheader", "--user", str(os.getuid()), "--format=%i"],
                                    capture_output=True, text=True, check=True)
            if str(job["job_id"]) in active.stdout.split():
                raise ValueError("the job is pending or active; extend after it has stopped")
        training = Path(train_stage["completion"]).parent
        journal = training / "steps.jsonl"
        records = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        completed = records[-1]["step"] if records else 0
        if completed and not (training / f"step_{completed:04d}/state").is_dir():
            raise ValueError("the last training state checkpoint is missing")
        if always_propose and config["evolver"]["name"] != "gepa":
            raise ValueError("--always-propose applies to GEPA")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = run / "extensions" / f"steps-{completed:04d}-to-{target:04d}-{stamp}"
        archive.mkdir(parents=True)
        shutil.copy2(config_path, archive / "original-run.yaml")
        shutil.copy2(run / "plan.json", archive / "original-plan.json")
        summary_path = training / "summary.json"
        historical_ref = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            tree = Tree(Path(summary["tree"]))
            historical_ref = f"{summary['run']}/history/steps-{completed}-{stamp}/final"
            tree.tag(historical_ref, summary["final_tag"])
            shutil.copytree(training / "workspace/adapter", archive / "selected-adapter", symlinks=True)
        for source, name in ((summary_path, "training-summary.json"),
                             (training / "config.yaml", "resolved-config.yaml"),
                             (training / "eval/final", "final-evaluation"),
                             (run / "job-complete.json", "job-complete.json"),
                             (run / "progress.json", "progress.json")):
            if source.exists():
                shutil.move(str(source), archive / name)
        config["trainer"]["epochs"] = epochs
        if always_propose:
            config["evolver"].setdefault("params", {})["skip_perfect_score"] = False
        atomic_text(config_path, yaml.safe_dump(config, sort_keys=False))
        train_stage["expected_steps"] = target
        plan["sha256"]["run.yaml"] = file_sha256(config_path)
        record = {"completed_steps": completed, "previous_target": previous_target, "target_steps": target,
                  "epochs": epochs, "always_propose": always_propose, "historical_final_ref": historical_ref,
                  "archive": str(archive), "resume_from_training_checkpoint": completed,
                  "data_models_evaluator_and_video_caches_unchanged": True}
        atomic_json(archive / "extension.json", record)
        plan.setdefault("extensions", []).append(record)
        atomic_json(run / "plan.json", plan)
        verify_snapshot(run, plan)
        return plan
