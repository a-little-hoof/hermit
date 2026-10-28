"""Hermit acting pipeline: one frozen-generator run per candidate and task."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hermit.task import RunOutput
from examples.video_transition.utils.protocol import validate_task
from examples.video_transition.utils.remote import load_settings
from examples.video_transition.utils.artifacts import completed_case


def run(task: dict, adapter: Path | None, *, params: dict, base: Path) -> RunOutput:
    validate_task(task)
    runtime = Path(os.environ.get("VIDEO_TRANSITION_CONFIG") or params.get("runtime_config", "../runtime.local.json")).expanduser()
    if not runtime.is_absolute():
        runtime = (base / runtime).resolve()
    os.environ["VIDEO_TRANSITION_CONFIG"] = str(runtime)
    settings = load_settings(runtime)
    mode = params.get("mode") or ("baseline" if adapter is None else "controller")
    if mode not in ("baseline", "identity", "controller"):
        raise ValueError(f"unknown video pipeline mode: {mode}")
    skill_path = None
    skill = ""
    if mode == "controller":
        if adapter is None:
            raise ValueError("controller mode requires an adapter")
        skill_path = Path(adapter) / "skills/transition/SKILL.md"
        skill = skill_path.read_text()
    identity = {"task": task, "mode": mode, "skill": skill,
                "planning_mode": settings.get("planning_mode", "async"),
                "generator": settings["generator"], "controller": settings.get("controller"),
                "controller_sampling": settings.get("controller_sampling"),
                "budget_chunks": settings.get("budget_chunks", 4),
                "code_sha256": hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes() for name in
                    ("pipeline.py", "protocol.py", "controller.py", "remote.py", "streaming.py", "longlive_worker.py", "artifacts.py"))).hexdigest()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    root = Path(settings["run_root"]) / f"{mode}-{task['id']}-{digest}"
    manifest_path = root / task["id"] / "manifest.json"
    if mode == "baseline":
        baseline = Path(settings["baseline_root"]) / task["id"] / "manifest.json"
        if baseline.is_file():
            saved = json.loads(baseline.read_text())
            if saved.get("task") == task and saved.get("status") == "complete":
                manifest_path = baseline
    started = time.perf_counter()
    cached = completed_case(manifest_path.parent, task, mode)
    if not cached:
        if not os.environ.get("SLURM_JOB_ID") and not settings.get("allow_local_gpu", False):
            raise RuntimeError("Uncached video generation needs a Slurm GPU allocation; run the Hermit driver inside it")
        root.mkdir(parents=True, exist_ok=True)
        task_file = root / "task.jsonl"
        task_file.write_text(json.dumps(task, ensure_ascii=False) + "\n")
        generator = settings["generator"]
        command = [generator["python"], "-u", str(Path(__file__).with_name("longlive_worker.py")),
                   "--tasks", str(task_file), "--source", generator["source"], "--models", generator["models"],
                   "--base-checkpoint", generator["base_checkpoint"], "--lora-checkpoint", generator["lora_checkpoint"],
                   "--output-dir", str(root), "--mode", mode, "--resume"]
        if mode == "controller":
            # Freeze this candidate while the separate CUDA process reads it.
            frozen = root / "SKILL.md"
            frozen.write_text(skill)
            command += ["--runtime-config", str(runtime), "--skill", str(frozen)]
        env = os.environ.copy()
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled"})
        env.update({str(k): str(v) for k,v in generator.get("environment", {}).items()})
        env.pop("LD_PRELOAD", None)
        with (root / "generation.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    timeout=float(params.get("generation_timeout_s", 1800)))
        if result.returncode:
            raise RuntimeError(f"video generator exited {result.returncode}; inspect {root / 'generation.log'}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("control_errors"):
        raise RuntimeError(f"video execution failed; inspect {manifest_path}")
    video = Path(manifest["video_path"])
    if not video.is_file():
        raise FileNotFoundError(video)
    trace = manifest.get("controller_trace", [])
    trajectory = manifest_path.parent / "trajectory.jsonl"
    records = trace or [{"event": "baseline", "switches": manifest["switches"]}]
    trajectory.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in records))
    tokens = 0
    for event in trace:
        usage = event.get("response", {}).get("usage", {})
        tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return RunOutput(
        output={"video_path": str(video), "manifest_path": str(manifest_path)}, log=trajectory,
        wall_s=time.perf_counter()-started, cost_usd=None, tokens=0 if cached else tokens,
        messages=[{"role": "system", "content": skill}, {"role": "user", "content": task["id"]},
                  {"role": "assistant", "content": json.dumps(records, ensure_ascii=False)}],
        extra={"mode": mode, "cache_hit": cached, "skill_sha256": hashlib.sha256(skill.encode()).hexdigest(),
               "generator_elapsed_s": manifest["total_s"], "run_dir": str(manifest_path.parent)},
    )
