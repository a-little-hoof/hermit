"""Train with Hermit's step checkpoints and replay saved reflection responses."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hermit.config import Config
from hermit.evolver.base import Usage
from hermit.io import atomic_json
from hermit.trainer.trainer import Trainer


class ReflectionCache:
    """Scope paid responses to a training step; a retried step uses the same proposal."""
    def __init__(self, client, directory: Path):
        self.client = client
        self.directory = directory
        self.scope = "setup"

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, messages, **sampling):
        identity = {"scope": self.scope, "messages": messages, "sampling": sampling,
                    "model": self.client.model, "base_url": self.client.base_url,
                    "extra_body": self.client.extra_body}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.directory / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            return saved["message"], Usage(model=self.client.model, cost_usd=0.0)
        message, usage = self.client.chat(messages, **sampling)
        # The reflection consumes final text; do not retain private reasoning content.
        final = {key: value for key, value in message.items() if key != "reasoning_content"}
        atomic_json(path, {"message": final, "usage": asdict(usage), "scope": self.scope, "request_sha256": key})
        return message, usage


class VideoTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.llm is not None:
            self.llm = ReflectionCache(self.llm, self.run_dir / "reflection-cache")

    def _evolve(self, what, progress, *args, **kwargs):
        # _evolve receives the training rollouts before any learner call or checkpoint.
        results = args[2] if len(args) >= 3 else kwargs.get("results", [])
        if any(record.get("status") != "ok" for record in results):
            raise RuntimeError("video rollout or judge failed; resolve the error and resume this step")
        if isinstance(self.llm, ReflectionCache):
            self.llm.scope = f"{what}-{progress.step}"
        return super()._evolve(what, progress, *args, **kwargs)

    def _commit_step(self, progress, result, *args, **kwargs):
        if result.status in ("failed", "out_of_budget"):
            raise RuntimeError(f"evolution step failed: {result.summary}; resume after resolving the error")
        return super()._commit_step(progress, result, *args, **kwargs)

    def _run_agent_tool(self):
        tool = super()._run_agent_tool()
        original = tool.fn

        def checked(**arguments):
            result = original(**arguments)
            if any(record.get("status") != "ok" for record in result["results"]):
                raise RuntimeError("candidate rollout or judge failed; resume after resolving the error")
            return result
        tool.fn = checked
        return tool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    summary = VideoTrainer(Config.load(args.config, args.overrides)).train()
    print(json.dumps(summary, indent=2))
    if summary["stop_reason"] != "completed all epochs":
        raise SystemExit("training stopped before all requested epochs completed")


if __name__ == "__main__":
    main()
