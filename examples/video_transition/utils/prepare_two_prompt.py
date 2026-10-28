"""Prepare one official prompt pair with separate seeds for offline debugging."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from hermit.io import atomic_json
from examples.video_transition.utils.artifacts import completed_case
from examples.video_transition.utils.protocol import validate_task


def prepare_pair(task: dict, output: Path) -> dict:
    if len(task["events"]) != 2:
        raise ValueError("this experiment requires exactly two user prompts")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    splits = {}
    for seed, split in enumerate(("train", "val", "test")):
        row = deepcopy(task)
        row["seed"] = seed
        if seed:
            row["id"] += f"-seed{seed}"
        validate_task(row)
        rows.append(row)
        splits[split] = row["id"]
        (output / f"{split}.txt").write_text(row["id"] + "\n")
    (output / "tasks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    metadata = {"scope": "one prompt pair; seed holdouts only, not generalization to unseen prompts",
                "planning_mode": "offline", "source": task.get("source"), "splits": splits,
                "seeds": {"train": 0, "val": 1, "test": 2}, "prompts": [e["prompt"] for e in task["events"]]}
    atomic_json(output / "source.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "examples/video_transition/runtime.local.json")
    parser.add_argument("--source-tasks", type=Path, default=ROOT / ".demo/video-transition/data/pilot_tasks.jsonl")
    parser.add_argument("--source-id", default="longlive-02-pilot")
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / ".demo/video-transition/two-prompt-offline")
    parser.add_argument("--output-runtime", type=Path, default=ROOT / "examples/video_transition/runtime.two_prompt.local.json")
    args = parser.parse_args()
    task = next(row for row in map(json.loads, args.source_tasks.read_text().splitlines()) if row["id"] == args.source_id)
    experiment = args.experiment_dir.resolve()
    metadata = prepare_pair(task, experiment / "data")
    settings = json.loads(args.runtime.read_text())
    original = Path(settings["baseline_root"]) / task["id"]
    if not completed_case(original, task, "baseline", save_latents=True):
        raise ValueError("generate and verify the raw baseline for this pair before reusing it")
    reference = experiment / "reference/baseline"
    case = reference / task["id"]
    case.mkdir(parents=True, exist_ok=True)
    for name in ("video.mp4", "latents.pt"):
        if not (case / name).exists():
            os.link(original / name, case / name)
    manifest = json.loads((original / "manifest.json").read_text())
    manifest.update(video_path=str(case / "video.mp4"), reused_from=str(original / "manifest.json"))
    atomic_json(case / "manifest.json", manifest)
    atomic_json(reference / "complete.json", {"status": "complete", "tasks": [task["id"]]})
    settings.update(planning_mode="offline", run_root=str(experiment / "runs"), baseline_root=str(reference),
                    evaluator_protocol="segue-table12-glm5p3-flash-json-schema-offline-pair-v2")
    settings["evaluator_sampling"]["response_format"] = {"type": "json_schema", "json_schema": {
        "name": "TransitionJudgments", "schema": {"type": "object", "additionalProperties": False,
        "properties": {"judgments": {"type": "array", "minItems": 10, "maxItems": 10,
                                      "items": {"type": "integer", "enum": [0, 1]}},
                       "sum": {"type": "integer", "enum": list(range(11))}}, "required": ["judgments", "sum"]}}}
    atomic_json(args.output_runtime, settings)
    print(json.dumps({"runtime": str(args.output_runtime), "experiment_dir": str(experiment), **metadata}, indent=2))


if __name__ == "__main__":
    main()
