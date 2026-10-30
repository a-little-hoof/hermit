"""Configure a separate Astra max evaluator experiment from the existing offline run."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from hermit.io import atomic_json
from examples.video_transition.utils.artifacts import file_sha256
from examples.video_transition.utils.codex_evaluator import JUDGMENT_SCHEMA


def prepare(source: Path, experiment: Path) -> dict:
    settings = deepcopy(json.loads((source / "runtime.json").read_text()))
    binary = shutil.which("codex")
    if not binary:
        raise RuntimeError("install and sign into Codex before preparing Astra scoring")
    binary = Path(binary).resolve()
    settings["evaluator"] = {"transport": "codex-exec", "model": "gpt-6-astra", "reasoning_effort": "max",
                             "binary": str(binary), "binary_sha256": file_sha256(binary), "timeout": 1200}
    settings["evaluator_sampling"] = {"response_format": {"type": "json_schema", "json_schema": {
        "name": "TransitionJudgments", "schema": JUDGMENT_SCHEMA}}}
    settings.update(run_root=str(experiment / "runs"), evaluator_protocol="segue-table12-astra-max-codex-v1", evaluator_workers=4)
    path = experiment / "runtime.local.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("refusing to replace a different prepared Astra runtime")
    atomic_json(path, settings)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / ".demo/video-transition/ten-prompt-bs10-astra-max")
    args = parser.parse_args()
    settings = prepare(args.source_run.resolve(), args.experiment_dir.resolve())
    print(json.dumps({"runtime": str(args.experiment_dir.resolve() / "runtime.local.json"),
                      "evaluator": settings["evaluator"]}, indent=2))


if __name__ == "__main__":
    main()
