"""Link verified acting artifacts into a new run without copying old judge scores."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from hermit.io import atomic_json
from examples.video_transition.utils.artifacts import completed_case, file_sha256

ACTING_FILES = ("pipeline.py", "protocol.py", "controller.py", "remote.py", "streaming.py", "longlive_worker.py", "artifacts.py")
ACTING_SETTINGS = ("generator", "controller", "controller_sampling", "planning_mode", "budget_chunks")


def link_video_cache(source: Path, target: Path) -> dict:
    source, target = source.resolve(), target.resolve()
    before = json.loads((source / "runtime.json").read_text())
    after = json.loads((target / "runtime.json").read_text())
    for key in ACTING_SETTINGS:
        if before.get(key) != after.get(key):
            raise ValueError(f"cannot reuse videos with changed acting settings: {key}")
    relative = Path("source/examples/video_transition/utils")
    for name in ACTING_FILES:
        if file_sha256(source / relative / name) != file_sha256(target / relative / name):
            raise ValueError(f"cannot reuse videos with changed acting code: {name}")
    tasks = {row["id"]: row for row in map(json.loads, (target / "source/data/tasks.jsonl").read_text().splitlines())}
    linked, skipped = [], []
    for original in sorted((source / "videos").glob("*/*/manifest.json")):
        saved = json.loads(original.read_text())
        task = saved.get("task", {})
        if tasks.get(task.get("id")) != task or not completed_case(original.parent, task, saved.get("mode")):
            skipped.append(str(original))
            continue
        source_root = original.parent.parent
        destination_root = target / "videos" / source_root.name
        if (destination_root / task["id"] / "manifest.json").exists():
            raise ValueError("destination video case already exists")
        if not destination_root.exists():
            # Read-only evidence is linked; mutable manifests and old evaluations are handled separately.
            shutil.copytree(source_root, destination_root, copy_function=os.link,
                            ignore=shutil.ignore_patterns("manifest.json", "evaluation", "attempts", "*.partial*"))
        destination = destination_root / task["id"]
        rewritten = {**saved, "video_path": str(destination / "video.mp4"),
                     "reused_from": str(original), "previous_scores_reused": False}
        atomic_json(destination / "manifest.json", rewritten)
        if not completed_case(destination, task, saved["mode"]):
            raise RuntimeError("linked video failed checksum verification")
        linked.append({"task": task["id"], "source_manifest": str(original),
                       "destination_manifest": str(destination / "manifest.json"), "video_sha256": saved["video_sha256"]})
    report = {"source_run": str(source), "target_run": str(target), "linked_cases": len(linked),
              "cases": linked, "skipped_incomplete_or_different_cases": skipped,
              "acting_settings_and_code_verified": True, "previous_judgments_copied": False}
    atomic_json(target / "video-cache-reuse.json", report)
    return report
