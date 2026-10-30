"""Freeze a full-batch Astra experiment, reuse unchanged videos, then submit it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from hermit.io import atomic_json
from examples.video_transition.utils.run_job import verify_snapshot
from examples.video_transition.utils.submit import freeze, submit_run
from examples.video_transition.utils.reuse_video_cache import link_video_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True, help="completed eight-metric Astra video preflight summary")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--time-min", default="00:30:00")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    settings = json.loads(runtime.read_text())
    evaluator = settings["evaluator"]
    if evaluator.get("model") != "gpt-6-astra" or evaluator.get("reasoning_effort") != "max" or evaluator.get("transport") != "codex-exec":
        raise ValueError("this experiment requires the requested Astra max judge")
    ready = json.loads(args.preflight.read_text())
    if ready.get("status") != "complete" or ready.get("model") != evaluator["model"] or len(ready.get("metrics", {})) != 8:
        raise ValueError("complete the real eight-metric Astra preflight before allocating a GPU")
    protocol = json.loads((args.preflight.parent / "protocol.json").read_text())
    if protocol.get("client_config", {}).get("reasoning_effort") != "max" or \
            protocol.get("client_config", {}).get("binary_sha256") != evaluator.get("binary_sha256"):
        raise ValueError("preflight used a different judge reasoning level or binary")
    metadata = Path(settings["run_root"]).parent
    metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        latest = metadata / "latest-evolve.json"
        if latest.exists():
            raise ValueError("this experiment already has a run; use submit.py --resume rather than creating a duplicate")
        frozen_args = SimpleNamespace(mode="evolve", steps=args.steps, always_propose=True,
                                     gepa_config=ROOT / "examples/video_transition/gepa/ten_prompt_astra.yaml",
                                     skill=args.source_run.resolve() / "source/adapter/skills/transition/SKILL.md",
                                     gres=None, account=None, partition="commons", time="24:00:00",
                                     time_min=args.time_min, mem=None, cpus=None, no_auto_resume=False)
        run, plan = freeze(frozen_args, runtime, settings)
        reuse = link_video_cache(args.source_run, run)
        atomic_json(run / "astra-preflight.json", {"summary": ready, "protocol": protocol,
                                                  "source": str(args.preflight.resolve())})
        verify_snapshot(run, plan)
        record = submit_run(run, plan, None, metadata)
        atomic_json(metadata / "session.json", {"updated_at": datetime.now(timezone.utc).isoformat(),
                    "status": "submitted", "jobs": {"evolve": record}, "completed_training_steps": 0,
                    "expected_training_steps": args.steps, "batch_size": 10, "epochs": args.steps,
                    "evaluator": {"model": evaluator["model"], "reasoning_effort": "max", "transport": "codex-exec"},
                    "counts": {"train_pairs": 10, "validation_pairs": 4, "validation_tasks": 12, "test_tasks": 6},
                    "source_run": str(args.source_run.resolve()), "video_cache_reused_cases": reuse["linked_cases"],
                    "judge_scores_reused": False, "baseline": "same initial shared skill, scored fresh by Astra"})
        print(json.dumps({**record, "batch_size": 10, "steps": args.steps,
                          "evaluator": "gpt-6-astra", "reasoning_effort": "max", "reused_videos": reuse["linked_cases"]}, indent=2))


if __name__ == "__main__":
    main()
