"""Freeze or resume a video experiment with Slurm time-limit requeue."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hermit.io import atomic_json
from examples.video_transition.utils.artifacts import file_sha256
from examples.video_transition.utils.run_job import verify_snapshot


def wall_seconds(value: str) -> int:
    try:
        days, clock = value.split("-", 1) if "-" in value else ("0", value)
        hours, minutes, seconds = map(int, clock.split(":"))
        days = int(days)
        if min(days, hours, minutes, seconds) < 0 or minutes >= 60 or seconds >= 60:
            raise ValueError
        total = days * 86400 + hours * 3600 + minutes * 60 + seconds
        if not 0 < total <= 86400:
            raise ValueError
        return total
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("wall time must be HH:MM:SS or D-HH:MM:SS, at most 24 hours") from None


def script_text(run: Path, settings: dict, mode: str, resources: dict, auto_resume: bool) -> str:
    seconds = wall_seconds(resources["time"])
    lines = ["#!/bin/bash", f"#SBATCH --partition={resources['partition']}",
             f"#SBATCH --mem={resources['mem']}", f"#SBATCH --cpus-per-task={resources['cpus']}",
             f"#SBATCH --time={resources['time']}", f"#SBATCH --job-name=hermit-video-{mode}",
             f"#SBATCH --output={run}/slurm-%j.log", "#SBATCH --open-mode=append"]
    for field in ("gres", "account"):
        if resources.get(field):
            lines.append(f"#SBATCH --{field}={resources[field]}")
    if auto_resume:
        lines += ["#SBATCH --requeue", f"#SBATCH --signal=B:USR1@{min(120, max(1, seconds // 4))}"]
    lines += ["set -euo pipefail", "unset LD_PRELOAD", "export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1",
              "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled",
              f"export OMP_NUM_THREADS={resources['cpus']} TORCHINDUCTOR_COMPILE_THREADS=2",
              "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
    lines.append(f"export PYTHONPATH={shlex.quote(str(run / 'source'))}${{PYTHONPATH:+:$PYTHONPATH}}")
    for name, value in settings["generator"].get("environment", {}).items():
        if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
            raise ValueError("invalid generator environment variable name")
        lines.append(f"export {name}={shlex.quote(str(value))}")
    if resources.get("gres"):
        lines.append("nvidia-smi --query-gpu=name,memory.total --format=csv")
    if auto_resume:
        lines += ["on_limit() {", "  trap '' USR1",
                  "  echo 'Slurm time warning: requeueing the same experiment and checkpoints.'",
                  '  if ! scontrol requeue "${SLURM_JOB_ID:?}"; then',
                  '    kill -TERM "${driver_pid:?}" 2>/dev/null || true', "    exit 75", "  fi", "  exit 0", "}",
                  "trap on_limit USR1"]
    command = [settings["orchestrator_python"], "-u", str(run / "source/examples/video_transition/utils/run_job.py"),
               "--run-dir", str(run)]
    lines += [shlex.join(command) + " &", "driver_pid=$!", "while true; do", "  code=0",
              '  wait "$driver_pid" || code=$?', '  if ! kill -0 "$driver_pid" 2>/dev/null; then',
              '    exit "$code"', "  fi", "done"]
    return "\n".join(lines) + "\n"


def freeze(args, runtime: Path, settings: dict) -> tuple[Path, dict]:
    mode = args.mode
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = Path(settings["run_root"]) / f"{mode}-{tag}"
    run.mkdir(parents=True)
    utils = run / "source/examples/video_transition/utils"
    utils.mkdir(parents=True)
    for path in (TASK_ROOT / "utils").glob("*.py"):
        shutil.copy2(path, utils / path.name)
    shutil.copytree(ROOT / "hermit", run / "source/hermit",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    adapter = run / "source/adapter"
    shutil.copytree(TASK_ROOT / "gepa/adapter", adapter)
    skill = adapter / "skills/transition/SKILL.md"
    skill.write_text(args.skill.read_text())
    if settings.get("secrets_file"):
        secret = Path(settings["secrets_file"]).expanduser()
        settings["secrets_file"] = str(secret if secret.is_absolute() else (runtime.parent / secret).resolve())
    if settings.get("rubrics_path"):
        rubric = run / "source/rubrics.json"
        shutil.copy2(settings["rubrics_path"], rubric)
        settings["rubrics_path"] = str(rubric)
    if mode == "baseline":
        settings["baseline_root"] = str(run)
    if mode == "evolve":
        settings["run_root"] = str(run / "videos")
    frozen_runtime = run / "runtime.json"
    atomic_json(frozen_runtime, settings)
    generator = settings["generator"]
    stages = []
    if mode == "evolve":
        config_path = args.gepa_config.resolve()
        config = yaml.safe_load(config_path.read_text())
        data = run / "source/data"
        data.mkdir()
        for key in ("tasks", "train", "val", "test"):
            source = (config_path.parent / config["data"][key]).resolve()
            target = data / ("tasks.jsonl" if key == "tasks" else f"{key}.txt")
            shutil.copy2(source, target)
            config["data"][key] = str(target)
        info = run / "source/task-info"
        shutil.copytree((config_path.parent / config["task_info"]).resolve(), info)
        config["task_info"] = str(info)
        config["template"] = str(adapter)
        config["agent"]["name"] = str(utils / "pipeline.py") + ":run"
        config["agent"]["params"]["runtime_config"] = str(frozen_runtime)
        config["judge"]["path"] = str(utils / "judge.py")
        config["trainer"].update(output_dir=str(run / "hermit"), tree_dir=str(run / "hermit/trees"), resume_mode="auto")
        frozen_config = run / "run.yaml"
        frozen_config.write_text(yaml.safe_dump(config, sort_keys=False))
        driver = [settings["orchestrator_python"], "-u", str(utils / "with_runtime.py"), "--runtime", str(frozen_runtime)]
        training = run / "hermit" / config["trainer"]["run_name"]
        count = len((data / "train.txt").read_text().split())
        steps = -(-count // int(config["data"]["step_size"])) * int(config["trainer"]["epochs"])
        stages.append({"id": "train", "kind": "train", "command": [*driver, "train", "--config", str(frozen_config)],
                       "completion": str(training / "summary.json"), "expected_steps": steps})
        for label in ("root", "final"):
            stages.append({"id": f"eval-{label}", "kind": "eval",
                           "command": [*driver, "eval", "--config", str(frozen_config), "--adapter", label],
                           "completion": str(training / "eval" / label / "summary.json"),
                           "expected_tasks": len((data / "test.txt").read_text().split())})
    elif mode == "score":
        for index, generated in enumerate(args.score_runs or []):
            stages.append({"id": f"score-{index}", "kind": "json",
                           "command": [settings["orchestrator_python"], "-u", str(utils / "score_run.py"),
                                       "--runtime", str(frozen_runtime), "--run-dir", str(generated.resolve())],
                           "completion": str(generated.resolve() / "evaluation-summary.json")})
        if args.identity_run:
            stages.append({"id": "verify-pilot", "kind": "json", "success_status": ["passed"],
                           "command": [settings["orchestrator_python"], str(utils / "verify_pilot.py"),
                                       "--identity-run", str(args.identity_run.resolve()),
                                       "--controller-run", str(args.controller_run.resolve()), "--output", str(run / "gate.json")],
                           "completion": str(run / "gate.json")})
    else:
        tasks = run / "tasks.jsonl"
        shutil.copy2(args.tasks, tasks)
        selected = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
        command = [generator["python"], "-u", str(utils / "longlive_worker.py"), "--tasks", str(tasks),
                   "--source", generator["source"], "--models", generator["models"],
                   "--base-checkpoint", generator["base_checkpoint"], "--lora-checkpoint", generator["lora_checkpoint"],
                   "--output-dir", str(run), "--mode", mode, "--save-latents", "--resume"]
        if args.ids_file:
            ids = run / "ids.txt"
            shutil.copy2(args.ids_file, ids)
            selected = [task for task in selected if task["id"] in set(ids.read_text().split())]
            command += ["--ids-file", str(ids)]
        if mode == "controller":
            command += ["--runtime-config", str(frozen_runtime), "--skill", str(skill)]
        stages.append({"id": "generate", "kind": "generation", "mode": mode, "command": command,
                       "completion": str(run / "complete.json"), "tasks": selected})
        if args.compare_baseline:
            stages.append({"id": "parity", "kind": "json", "success_status": ["passed"],
                           "command": [generator["python"], "-u", str(utils / "compare_latents.py"),
                                       "--reference", str(args.compare_baseline.resolve()), "--candidate", str(run)],
                           "completion": str(run / "parity.json")})
    if not stages:
        raise ValueError("no stages selected")
    slurm = settings.get("slurm", {})
    resources = {"gres": None if mode == "score" else (args.gres or slurm.get("gres")),
                 "account": args.account or slurm.get("account"), "partition": args.partition,
                 "time": args.time or ("24:00:00" if mode == "evolve" else "00:20:00"),
                 "mem": args.mem or ("8G" if mode == "score" else "60G"),
                 "cpus": args.cpus or (2 if mode == "score" else 8)}
    if mode != "score" and not resources["gres"]:
        raise ValueError("provide --gres or slurm.gres for the required GPU type")
    for value in resources.values():
        if value is not None and any(char.isspace() for char in str(value)):
            raise ValueError("Slurm resource fields cannot contain whitespace")
    script = run / "run.sbatch"
    script.write_text(script_text(run, settings, mode, resources, not args.no_auto_resume))
    files = [p for p in (run / "source").rglob("*") if p.is_file()]
    files += [p for p in run.iterdir() if p.is_file()]
    hashes = {str(path.relative_to(run)): file_sha256(path) for path in files}
    plan = {"version": 1, "mode": mode, "stages": stages, "resources": resources,
            "auto_resume": not args.no_auto_resume, "sha256": hashes}
    atomic_json(run / "plan.json", plan)
    return run, plan


def submit_run(run: Path, plan: dict, dependency: str | None, metadata_root: Path) -> dict:
    with (run / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prior = json.loads((run / "job.json").read_text()) if (run / "job.json").exists() else None
        if prior:
            result = subprocess.run(["squeue", "--noheader", "--user", str(os.getuid()), "--format=%i"],
                                    capture_output=True, text=True, check=True)
            if str(prior["job_id"]) in result.stdout.split():
                raise ValueError(f"job {prior['job_id']} is already pending or active; do not submit a duplicate")
        settings = json.loads((run / "runtime.json").read_text())
        env = os.environ.copy()
        for role in ("learner", "controller", "evaluator"):
            name = settings.get(role, {}).get("api_key_env")
            if name:
                env.pop(name, None)
        command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
        if dependency:
            command.append(f"--dependency={dependency}")
        result = subprocess.run([*command, str(run / "run.sbatch")], text=True, capture_output=True, env=env, check=True)
        record = {"job_id": result.stdout.strip().split(";")[0], "mode": plan["mode"], "run_dir": str(run),
                  "script": str(run / "run.sbatch"), "status": "submitted", "metadata_root": str(metadata_root),
                  "resources": plan["resources"], "auto_resume": plan["auto_resume"],
                  "previous_job_id": prior["job_id"] if prior else None}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        atomic_json(run / "submissions" / f"{stamp}.json", record)
        atomic_json(run / "job.json", record)
        atomic_json(metadata_root / f"latest-{plan['mode']}.json", record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--mode", choices=("baseline", "identity", "controller", "score", "evolve"))
    choice.add_argument("--resume", type=Path, help="resubmit the same frozen run directory and checkpoints")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--tasks", type=Path, default=ROOT / ".demo/video-transition/data/pilot_tasks.jsonl")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--gepa-config", type=Path, default=TASK_ROOT / "gepa/smoke.yaml")
    parser.add_argument("--skill", type=Path, default=TASK_ROOT / "gepa/adapter/skills/transition/SKILL.md")
    parser.add_argument("--gres")
    parser.add_argument("--mem")
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--time")
    parser.add_argument("--account")
    parser.add_argument("--partition", default="commons")
    parser.add_argument("--dependency")
    parser.add_argument("--compare-baseline", type=Path)
    parser.add_argument("--score-runs", type=Path, nargs="+")
    parser.add_argument("--identity-run", type=Path)
    parser.add_argument("--controller-run", type=Path)
    parser.add_argument("--no-auto-resume", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and any((args.runtime, args.time, args.gres, args.mem, args.cpus, args.account, args.no_auto_resume)):
        parser.error("--resume uses frozen settings and resources; pass only its run directory and optional --dependency")
    if args.time:
        wall_seconds(args.time)
    if args.resume:
        run = args.resume.expanduser().resolve()
        plan = json.loads((run / "plan.json").read_text())
        verify_snapshot(run, plan)
        previous = json.loads((run / "job.json").read_text())
        metadata_root = Path(previous["metadata_root"])
    else:
        if not args.runtime:
            parser.error("--runtime is required for a new experiment")
        if args.compare_baseline and args.mode != "identity":
            parser.error("--compare-baseline applies only to identity mode")
        if bool(args.identity_run) != bool(args.controller_run):
            parser.error("provide --identity-run and --controller-run together")
        runtime = args.runtime.expanduser().resolve()
        settings = json.loads(runtime.read_text())
        metadata_root = Path(settings["run_root"]).parent
        run, plan = freeze(args, runtime, settings)
    record = submit_run(run, plan, args.dependency, metadata_root)
    if not args.resume and args.mode == "baseline":
        settings = json.loads(runtime.read_text())
        settings["baseline_root"] = str(run)
        atomic_json(runtime, settings)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
