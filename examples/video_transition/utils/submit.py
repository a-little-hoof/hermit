"""Submit a reproducible code snapshot for baseline, controller, or GEPA pilots."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "identity", "controller", "evolve"), required=True)
    parser.add_argument("--tasks", type=Path, default=ROOT / ".demo/video-transition/data/pilot_tasks.jsonl")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--gepa-config", type=Path, default=TASK_ROOT / "gepa/smoke.yaml")
    parser.add_argument("--skill", type=Path, default=TASK_ROOT / "gepa/adapter/skills/transition/SKILL.md")
    parser.add_argument("--gres")
    parser.add_argument("--mem", default="60G")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--time", default="00:20:00")
    parser.add_argument("--account")
    parser.add_argument("--partition", default="commons")
    parser.add_argument("--dependency")
    parser.add_argument("--compare-baseline", type=Path)
    args = parser.parse_args()
    if args.compare_baseline and args.mode != "identity":
        parser.error("--compare-baseline applies only to identity mode")
    runtime = args.runtime.expanduser().resolve()
    settings = json.loads(runtime.read_text())
    slurm = settings.get("slurm", {})
    gres = args.gres or slurm.get("gres")
    if not gres:
        parser.error("provide --gres or slurm.gres in runtime settings for the required GPU type")
    account = args.account or slurm.get("account")
    for value in (gres, args.mem, args.time, args.partition, account):
        if value is not None and any(char.isspace() for char in value):
            parser.error("Slurm resource fields cannot contain whitespace")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = Path(settings["run_root"]) / f"{args.mode}-{tag}"
    run.mkdir(parents=True)
    utils = run / "source/examples/video_transition/utils"
    utils.mkdir(parents=True)
    hashes = {}
    for path in (TASK_ROOT / "utils").glob("*.py"):
        shutil.copy2(path, utils / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen_adapter = run / "source/adapter"
    shutil.copytree(TASK_ROOT / "gepa/adapter", frozen_adapter)
    frozen_skill = frozen_adapter / "skills/transition/SKILL.md"
    frozen_skill.write_text(args.skill.read_text())
    secrets = settings.get("secrets_file")
    if secrets:
        secret_path = Path(secrets).expanduser()
        settings["secrets_file"] = str(secret_path if secret_path.is_absolute() else runtime.parent / secret_path)
    frozen_runtime = run / "runtime.json"
    frozen_runtime.write_text(json.dumps(settings, indent=2) + "\n")
    generator = settings["generator"]
    if args.mode == "baseline":
        settings["baseline_root"] = str(run)
        frozen_runtime.write_text(json.dumps(settings, indent=2) + "\n")
    if args.mode == "evolve":
        config_path = args.gepa_config.resolve()
        config = yaml.safe_load(config_path.read_text())
        data_root = run / "source/data"
        data_root.mkdir()
        for key in ("tasks", "train", "val", "test"):
            source = (config_path.parent / config["data"][key]).resolve()
            target = data_root / ("tasks.jsonl" if key == "tasks" else f"{key}.txt")
            shutil.copy2(source, target)
            config["data"][key] = str(target)
        info = run / "source/task-info"
        shutil.copytree(TASK_ROOT / "utils/task-info", info)
        config["task_info"] = str(info)
        config["template"] = str(frozen_adapter)
        config["agent"]["name"] = str(utils / "pipeline.py") + ":run"
        config["agent"]["params"]["runtime_config"] = str(frozen_runtime)
        config["judge"]["path"] = str(utils / "judge.py")
        config["trainer"]["output_dir"] = str(run / "hermit")
        config["trainer"]["tree_dir"] = str(run / "hermit/trees")
        frozen_config = run / "run.yaml"
        frozen_config.write_text(yaml.safe_dump(config, sort_keys=False))
        commands = [[settings["orchestrator_python"], "-u", str(utils / "with_runtime.py"),
                     "--runtime", str(frozen_runtime), verb, "--config", str(frozen_config), *tail]
                    for verb, tail in (("train", []), ("eval", ["--adapter", "root"]), ("eval", ["--adapter", "final"]))]
    else:
        task_file = run / "tasks.jsonl"
        shutil.copy2(args.tasks, task_file)
        command = [generator["python"], "-u", str(utils / "longlive_worker.py"), "--tasks", str(task_file),
                   "--source", generator["source"], "--models", generator["models"],
                   "--base-checkpoint", generator["base_checkpoint"], "--lora-checkpoint", generator["lora_checkpoint"],
                   "--output-dir", str(run), "--mode", args.mode, "--save-latents"]
        if args.ids_file:
            target = run / "ids.txt"
            shutil.copy2(args.ids_file, target)
            command += ["--ids-file", str(target)]
        if args.mode == "controller":
            command += ["--runtime-config", str(frozen_runtime), "--skill", str(frozen_skill)]
        commands = [command]
        if args.compare_baseline:
            commands.append([generator["python"], "-u", str(utils / "compare_latents.py"),
                             "--reference", str(args.compare_baseline.resolve()), "--candidate", str(run)])
    lines = ["#!/bin/bash", f"#SBATCH --partition={args.partition}", f"#SBATCH --gres={gres}",
             f"#SBATCH --mem={args.mem}", f"#SBATCH --cpus-per-task={args.cpus}", f"#SBATCH --time={args.time}",
             f"#SBATCH --job-name=hermit-video-{args.mode}", f"#SBATCH --output={run}/slurm-%j.log"]
    if account:
        lines.append(f"#SBATCH --account={account}")
    if args.dependency:
        lines.append(f"#SBATCH --dependency={args.dependency}")
    lines += ["set -euo pipefail", "unset LD_PRELOAD", "export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1",
              "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled",
              f"export OMP_NUM_THREADS={args.cpus} TORCHINDUCTOR_COMPILE_THREADS=2",
              "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
    for name, value in generator.get("environment", {}).items():
        if not name.replace("_", "").isalnum() or name[0].isdigit():
            raise ValueError("invalid generator environment variable name")
        lines.append(f"export {name}={shlex.quote(str(value))}")
    lines.append("nvidia-smi --query-gpu=name,memory.total --format=csv")
    lines.extend(shlex.join(command) for command in commands)
    script = run / "run.sbatch"
    script.write_text("\n".join(lines) + "\n")
    env = os.environ.copy()
    # The submission carries credential-file references rather than provider keys.
    for role in ("learner", "controller", "evaluator"):
        key_name = settings.get(role, {}).get("api_key_env")
        if key_name:
            env.pop(key_name, None)
    result = subprocess.run(["sbatch", "--parsable", str(script)], text=True, capture_output=True, env=env, check=True)
    record = {"job_id": result.stdout.strip().split(";")[0], "mode": args.mode, "run_dir": str(run),
              "script": str(script), "code_sha256": hashes, "status": "submitted"}
    (run / "job.json").write_text(json.dumps(record, indent=2) + "\n")
    (Path(settings["run_root"]).parent / f"latest-{args.mode}.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.mode == "baseline":
        original = json.loads(runtime.read_text())
        original["baseline_root"] = str(run)
        runtime.write_text(json.dumps(original, indent=2) + "\n")
    print(json.dumps({key: value for key,value in record.items() if key != "code_sha256"}, indent=2))


if __name__ == "__main__":
    main()
