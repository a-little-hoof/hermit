"""Load private model credentials, then execute a Hermit CLI command."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.video_transition.utils.remote import load_settings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.command or args.command[0] not in ("check","train","eval"):
        parser.error("pass a Hermit check, train, or eval command after --runtime")
    settings=load_settings(args.runtime)
    os.environ["VIDEO_TRANSITION_CONFIG"]=settings["_config_path"]
    learner={key:value for key,value in settings["learner"].items()
             if key in ("base_url","model","api_key_env","key_required","price","extra_body","timeout")}
    if args.command[0] == "check" and settings.get("slurm", {}).get("max_hours") is not None:
        from hermit.config import Config
        from hermit.settings_check import run_check
        check = argparse.ArgumentParser()
        check.add_argument("--config", type=Path, required=True)
        check.add_argument("--no-probe", action="store_true")
        check.add_argument("overrides", nargs="*")
        options = check.parse_args(args.command[1:])
        config = Config.load(options.config, [*options.overrides, "llm=" + json.dumps(learner)])
        raise SystemExit(0 if run_check(config, probe=not options.no_probe,
                                      allocation_hours=float(settings["slurm"]["max_hours"])) else 1)
    if args.command[0] == "train":
        command=[sys.executable,"-u",str(Path(__file__).with_name("resume_training.py")),
                 *args.command[1:],"llm="+json.dumps(learner)]
    else:
        command=[sys.executable,"-m","hermit.cli",*args.command,"llm="+json.dumps(learner)]
    raise SystemExit(subprocess.run(command).returncode)


if __name__=="__main__":
    main()
