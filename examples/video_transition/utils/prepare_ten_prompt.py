"""Select ten official transitions and four validation transitions without scene leakage."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from hermit.io import atomic_json, atomic_text
from examples.video_transition.utils.prepare import REVISION, SOURCE_PATH
from examples.video_transition.utils.protocol import validate_task

SOURCE_SHA256 = "8ba7c815adb84cfc849831d03ed2a144c29a15f3dedc98ddb38f1ad9b88bf0cb"
# Freeze selection before generating or scoring any video for this experiment.
SELECTION = {
    "train": ((2, 0), (3, 0), (5, 0), (6, 0), (7, 0), (8, 0),
              (2, 2), (3, 2), (7, 2), (8, 2)),
    "val": ((0, 0), (4, 0), (9, 0), (11, 0)),
    "test": ((1, 0), (10, 0)),
}
SEEDS = {"train": (0,), "val": (0, 1, 2), "test": (0, 1, 2)}


def prepare_transitions(output: Path, raw: bytes, revision: str = REVISION) -> dict:
    digest = hashlib.sha256(raw).hexdigest()
    if revision == REVISION and digest != SOURCE_SHA256:
        raise ValueError("official prompt bytes differ from the pinned source checksum")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    groups = {name: {row for row, _ in pairs} for name, pairs in SELECTION.items()}
    if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("source scenes overlap between splits")
    source = {"repository": "https://github.com/NVlabs/LongLive", "revision": revision,
              "path": SOURCE_PATH, "sha256": digest,
              "url": f"https://raw.githubusercontent.com/NVlabs/LongLive/{revision}/{SOURCE_PATH}"}
    tasks, splits, pairs = [], {}, {}
    for split, selection in SELECTION.items():
        splits[split], pairs[split] = [], []
        for row, start in selection:
            prompts = rows[row]["prompts"][start:start + 2]
            if len(prompts) != 2:
                raise ValueError(f"source row {row} lacks prompt pair {start}, {start + 1}")
            pair_id = f"longlive-{row:02d}-p{start:02d}-{start + 1:02d}"
            pairs[split].append({"id": pair_id, "row": row, "prompt_indices": [start, start + 1]})
            for seed in SEEDS[split]:
                task = {"id": f"{pair_id}-seed{seed}", "source_group": f"longlive-row-{row:02d}",
                        "source": {**source, "row": row, "prompt_indices": [start, start + 1]},
                        "events": [{"time_s": 10.0 * i, "prompt": prompt} for i, prompt in enumerate(prompts)],
                        "duration_s": 20.0, "seed": seed}
                # Only the logo lacks a human/animal subject. Row 6 explicitly contains a man.
                if revision == REVISION and row == 10:
                    task["subject_present"] = False
                validate_task(task)
                tasks.append(task)
                splits[split].append(task["id"])
    fingerprints = {}
    for task in tasks:
        key = tuple(event["prompt"] for event in task["events"])
        previous = fingerprints.setdefault(key, task["source_group"])
        if previous != task["source_group"]:
            raise ValueError("duplicate prompt pair appears in different source scenes")
    manifest = {"scope": "ten training transitions from six scenes; four validation transitions from four scenes; two test transitions from two scenes",
                "source": source, "planning_mode": "offline", "duration_s": 20.0,
                "selection_rule": "fixed source rows and adjacent prompt indices, selected before video scoring",
                "pairs": pairs, "source_groups": {name: sorted(group) for name, group in groups.items()},
                "seeds": SEEDS, "splits": splits,
                "counts": {name: {"prompt_pairs": len(pairs[name]), "source_scenes": len(groups[name]),
                                  "tasks": len(splits[name])} for name in SELECTION}}
    output.mkdir(parents=True, exist_ok=True)
    contents = {"tasks.jsonl": "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks),
                **{f"{name}.txt": "".join(item + "\n" for item in ids) for name, ids in splits.items()},
                "source.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"}
    # Reruns may verify an identical prepared dataset, but cannot replace a changed one.
    for name, text in contents.items():
        path = output / name
        if path.exists() and path.read_text() != text:
            raise ValueError(f"refusing to replace existing different dataset file: {path}")
    for name, text in contents.items():
        atomic_text(output / name, text)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, default=ROOT / ".demo/video-transition/vendor/LongLive" / SOURCE_PATH)
    parser.add_argument("--runtime", type=Path, default=ROOT / "examples/video_transition/runtime.two_prompt.local.json")
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / ".demo/video-transition/ten-prompt-offline")
    parser.add_argument("--output-runtime", type=Path, default=ROOT / "examples/video_transition/runtime.ten_prompt.local.json")
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    manifest = prepare_transitions(experiment / "data", args.source_file.read_bytes())
    settings = deepcopy(json.loads(args.runtime.read_text()))
    settings.update(planning_mode="offline", run_root=str(experiment / "runs"),
                    baseline_root=str(experiment / "reference/baseline"))
    if args.output_runtime.exists():
        if json.loads(args.output_runtime.read_text()) != settings:
            raise ValueError("output runtime already exists with different settings")
    else:
        atomic_json(args.output_runtime, settings)
    print(json.dumps({"experiment_dir": str(experiment), "runtime": str(args.output_runtime),
                      "counts": manifest["counts"], "source_groups": manifest["source_groups"]}, indent=2))


if __name__ == "__main__":
    main()
