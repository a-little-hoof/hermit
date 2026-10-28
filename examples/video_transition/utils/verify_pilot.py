"""Require generator parity and a controller that acted before video completion."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hermit.io import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-run", type=Path, required=True)
    parser.add_argument("--controller-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parity = json.loads((args.identity_run / "parity.json").read_text())
    if parity["status"] != "passed":
        raise ValueError("official and adapted inference loops disagree")
    complete = json.loads((args.controller_run / "complete.json").read_text())
    records = []
    for identifier in complete["tasks"]:
        manifest = json.loads((args.controller_run / identifier / "manifest.json").read_text())
        applied = [event for event in manifest["controller_trace"] if event["event"] == "prompt_applied"]
        if not applied:
            raise ValueError(f"{identifier}: controller made no change before video completion")
        records.append({"id": identifier, "prompt_changes": len(applied)})
    atomic_json(args.output, {"status": "passed", "cases": records})


if __name__ == "__main__":
    main()
