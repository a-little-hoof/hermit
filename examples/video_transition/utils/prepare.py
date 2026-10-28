"""Fetch pinned official LongLive streams and prepare source-grouped pilot splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.video_transition.utils.protocol import validate_task

REVISION = "cfa4e91eec59429d583326500bf5535c9ce6eec7"
SOURCE_PATH = "LongLive1.0/example/interactive_example.jsonl"
ROOT = Path(__file__).resolve().parents[3]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def prepare(output: Path, raw: bytes, revision: str, split_seed: int = 42) -> dict:
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) < 6:
        raise ValueError("need at least six source streams for disjoint pilot and held-out groups")
    output.mkdir(parents=True, exist_ok=True)
    source = {
        "repository": "https://github.com/NVlabs/LongLive",
        "revision": revision,
        "path": SOURCE_PATH,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "url": f"https://raw.githubusercontent.com/NVlabs/LongLive/{revision}/{SOURCE_PATH}",
    }
    tasks = []
    seen = set()
    for index, row in enumerate(rows):
        prompts = row["prompts"]
        if not isinstance(prompts, list) or len(prompts) < 2:
            raise ValueError(f"source row {index} has fewer than two prompts")
        digest = hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()
        if digest in seen:
            raise ValueError("duplicate source streams must be grouped before splitting")
        seen.add(digest)
        task = {
            "id": f"longlive-{index:02d}",
            "source_group": f"longlive-row-{index:02d}",
            "source": {**source, "row": index},
            "events": [{"time_s": 10.0 * j, "prompt": prompt} for j, prompt in enumerate(prompts)],
            "duration_s": float(10 * len(prompts)),
            "seed": 0,
        }
        # Manual annotations for this pinned official example revision: river scenery and a logo.
        if revision == REVISION and index in (6, 10):
            task["subject_present"] = False
        validate_task(task)
        tasks.append(task)
    shuffled = list(tasks)
    random.Random(split_seed).shuffle(shuffled)
    holdout = max(1, len(tasks) // 6)
    split = {"train": shuffled[:-2 * holdout], "val": shuffled[-2 * holdout:-holdout], "test": shuffled[-holdout:]}
    if len(split["train"]) < 4:
        raise ValueError("need four training source groups for the integration pilot")
    (output / "splits").mkdir(exist_ok=True)
    for name, group in split.items():
        (output / "splits" / f"{name}.txt").write_text("".join(row["id"] + "\n" for row in group))
    write_rows(output / "tasks.jsonl", tasks)
    # Hermit eval.repeats does not change task.seed. Represent independent seeds as explicit tasks.
    extra_tests = [{**task, "id": f"{task['id']}-seed{seed}", "seed": seed}
                   for task in split["test"] for seed in (1, 2)]
    write_rows(output / "tasks_multiseed.jsonl", tasks + extra_tests)
    test_ids = [task["id"] for task in split["test"] + extra_tests]
    (output / "splits/test_multiseed.txt").write_text("".join(name + "\n" for name in test_ids))
    # All four short pilot cases come from training source groups. Formal test groups stay unused.
    pilot = []
    for task in split["train"][:4]:
        short = {**task, "id": task["id"] + "-pilot", "events": task["events"][:2], "duration_s": 20.0}
        validate_task(short)
        pilot.append(short)
    write_rows(output / "pilot_tasks.jsonl", pilot)
    for name, group in {"train": pilot[:2], "val": pilot[2:3], "test": pilot[3:]}.items():
        (output / "splits" / f"pilot_{name}.txt").write_text("".join(row["id"] + "\n" for row in group))
    manifest = {
        "source": source,
        "split_seed": split_seed,
        "streams": len(tasks),
        "splits": {name: [task["id"] for task in group] for name, group in split.items()},
        "pilot_ids": [task["id"] for task in pilot],
        "test_seeds": [0, 1, 2],
        "test_seed_ids": test_ids,
        "pilot_scope": "integration only; four 20-second prefixes from formal training source groups",
    }
    (output / "source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".demo/video-transition/data")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--source-file", type=Path)
    args = parser.parse_args()
    if args.source_file:
        raw = args.source_file.read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/NVlabs/LongLive/{args.revision}/{SOURCE_PATH}"
        request = urllib.request.Request(url, headers={"User-Agent": "hermit-video-transition"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
    print(json.dumps(prepare(args.output_dir, raw, args.revision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
