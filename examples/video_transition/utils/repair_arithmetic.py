"""Recover redundant sum errors without changing a judge's binary decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hermit.io import atomic_json
from examples.video_transition.utils.artifacts import file_sha256
from examples.video_transition.utils.evaluate_video import evaluate, extract_frames, validate_judgments
from examples.video_transition.utils.protocol import METRICS, metric_times
from examples.video_transition.utils.remote import load_settings, parse_json_reply


def repair_sums(task: dict, video: Path, settings: dict, destination: Path) -> list[dict]:
    """Materialize a metric cache from an unchanged, evidence-bound response vector."""
    protocol_path = destination / "protocol.json"
    if not protocol_path.exists():
        return []
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("task") != task:
        return []
    rubrics = json.loads(Path(settings["rubrics_path"]).read_text())
    rubric_sha = hashlib.sha256(json.dumps(rubrics["metrics"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if rubric_sha != rubrics["rubric_sha256"] or rubric_sha != protocol["rubric_sha256"]:
        raise ValueError("cannot repair a response against a different rubric")
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    repairs = []
    for raw_path in sorted(destination.glob("switch-*.response.json")):
        match = re.fullmatch(r"switch-(\d+)-([A-Za-z]+)\.response\.json", raw_path.name)
        if not match:
            continue
        index, metric = int(match[1]), match[2]
        target = raw_path.with_name(raw_path.name.replace(".response.json", ".json"))
        if target.exists() or metric not in METRICS or not 1 <= index < len(task["events"]):
            continue
        raw = json.loads(raw_path.read_text())
        try:
            reply = parse_json_reply(raw.get("content"))
            values = reply.get("judgments")
            if not isinstance(values, list) or len(values) != 10 or any(type(v) is not int or v not in (0, 1) for v in values):
                continue
            total = sum(values)
            validate_judgments({"judgments": values, "sum": total})
        except ValueError:
            continue
        if type(reply.get("sum")) is int and reply["sum"] == total:
            continue
        identity = raw.get("identity") or {}
        if identity.get("protocol_sha256") != fingerprint:
            continue
        times = metric_times(metric, float(task["events"][index]["time_s"]))
        with tempfile.TemporaryDirectory(prefix=".verify-frames-", dir=destination) as temporary:
            extracted = extract_frames(video, times, Path(temporary))
            frames = [extracted[t] for t in times]
            evidence = hashlib.sha256(b"".join(Path(frame["path"]).read_bytes() for frame in frames)).hexdigest()
            if identity.get("evidence_sha256") != evidence:
                raise ValueError("saved response frames differ; refusing arithmetic repair")
            stable_frames = destination / "frames"
            stable_frames.mkdir(exist_ok=True)
            for frame in frames:
                path = Path(frame["path"])
                target_frame = stable_frames / path.name
                path.replace(target_frame)
                frame["path"] = str(target_frame.resolve())
        note = {"source_response": str(raw_path.resolve()), "response_sha256": file_sha256(raw_path),
                "reported_sum": reply.get("sum"), "computed_sum": total, "judgments_unchanged": True}
        result = {"metric": metric, "judgments": values, "sum": total, "score": total / 10,
                  "frames": frames, "usage": raw["usage"], "switch_index": index,
                  "requested_switch_s": task["events"][index]["time_s"],
                  "protocol_sha256": fingerprint, "evidence_sha256": evidence, "cache_hit": True,
                  "failed_criteria": [{"number": i + 1, "criterion": rubrics["metrics"][metric][i]}
                                      for i, value in enumerate(values) if not value],
                  "aggregation_repair": note}
        atomic_json(target, result)
        repairs.append(note)
    return repairs


def evaluate_with_recovery(task: dict, video: Path, settings: dict, destination: Path, **kwargs) -> dict:
    # Every retry must materialize at least one missing metric; work is bounded by the metric count.
    before = {path.name: (path.stat().st_mtime_ns, path.stat().st_size)
              for path in destination.glob("switch-*.response.json")}
    repair_sums(task, video, settings, destination)
    for _ in range((len(task["events"]) - 1) * len(METRICS) + 1):
        try:
            result = evaluate(task, video, settings, destination, **kwargs)
            notes = [json.loads(path.read_text())["aggregation_repair"] for path in destination.glob("switch-*.json")
                     if ".response." not in path.name and "aggregation_repair" in json.loads(path.read_text())]
            result["aggregation_repairs"] = notes
            changed = [json.loads(path.read_text()) for path in destination.glob("switch-*.response.json")
                       if before.get(path.name) != (path.stat().st_mtime_ns, path.stat().st_size)]
            result["new_queries"] = len(changed)
            result["new_input_tokens"] = sum(raw["usage"].get("input_tokens", 0) for raw in changed)
            result["new_output_tokens"] = sum(raw["usage"].get("output_tokens", 0) for raw in changed)
            atomic_json(destination / "summary.json", result)
            return result
        except ValueError as error:
            if "sum disagrees" not in str(error) or not repair_sums(task, video, settings, destination):
                raise
    raise RuntimeError("arithmetic recovery did not finish within the number of metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = evaluate_with_recovery(manifest["task"], Path(manifest["video_path"]), load_settings(args.runtime),
                                    args.manifest.parent / "evaluation")
    print(json.dumps({k: v for k, v in result.items() if k != "feedback"}, indent=2))


if __name__ == "__main__":
    main()
