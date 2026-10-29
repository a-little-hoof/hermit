"""Hermit judge for a frozen generator's actual video and original user targets."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hermit.task import JudgeResult, RunOutput
from examples.video_transition.utils.repair_arithmetic import evaluate_with_recovery
from examples.video_transition.utils.remote import load_settings


def score(task: dict, output: RunOutput) -> JudgeResult:
    settings = load_settings()
    manifest = Path(output.output["manifest_path"])
    result = evaluate_with_recovery(task, Path(output.output["video_path"]), settings, manifest.parent / "evaluation")
    return JudgeResult(score=result["score"], feedback=result["feedback"],
                       extra={key: value for key,value in result.items() if key not in ("score", "feedback")})
