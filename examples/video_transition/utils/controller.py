"""Compile the latest arrived instruction and visible state into a bounded plan."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.video_transition.utils.remote import image_part, load_settings, model_client, parse_json_reply


def validate_plan(reply: dict, budget: int, max_steps: int = 4) -> list[dict]:
    steps = reply.get("steps")
    if not isinstance(steps, list) or len(steps) > max_steps:
        raise ValueError("steps must be an array within the configured step limit")
    used = 0
    result = []
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"prompt", "duration_chunks"}:
            raise ValueError("each step must contain only prompt and duration_chunks")
        prompt, duration = step["prompt"], step["duration_chunks"]
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
            raise ValueError("a step needs a nonempty prompt no longer than 2000 characters")
        if type(duration) is not int or duration <= 0:
            raise ValueError("duration_chunks must be a positive integer")
        used += duration
        result.append({"prompt": prompt.strip(), "duration_chunks": duration})
    if used > budget:
        raise ValueError("plan exceeds the transition chunk budget")
    return result


def plan(request: dict, skill: str, settings: dict, client=None) -> dict:
    required = {"request_id", "video_time_s", "previous_user_prompt", "latest_user_prompt", "frames", "budget_chunks"}
    if required - request.keys():
        raise ValueError(f"controller request missing fields: {sorted(required - request.keys())}")
    if type(request["budget_chunks"]) is not int or request["budget_chunks"] < 1:
        raise ValueError("budget_chunks must be a positive integer")
    now = float(request["video_time_s"])
    if not request["frames"] or any(float(frame["time_s"]) > now for frame in request["frames"]):
        raise ValueError("controller must receive existing frames, with no future observations")
    if not isinstance(request["latest_user_prompt"], str) or not request["latest_user_prompt"].strip():
        raise ValueError("latest_user_prompt must be nonempty")
    # Whitelist the visible fields; complete offline task schedules are never sent to this model.
    visible = {name: request[name] for name in (
        "request_id", "video_time_s", "previous_user_prompt", "latest_user_prompt", "budget_chunks"
    )}
    if request.get("active_prompt"):
        visible["active_prompt"] = request["active_prompt"]
    content = [{"type": "text", "text": json.dumps(visible, ensure_ascii=False)}]
    for frame in request["frames"]:
        content.append({"type": "text", "text": f"Observed video frame at {frame['time_s']:.6f} seconds:"})
        content.append(image_part(frame["path"]))
    messages = [
        {"role": "system", "content": (
            "Plan a short transition for a frozen streaming video generator. Follow the skill below. "
            "Return a JSON object containing only steps, an array of {prompt, duration_chunks}. "
            "Durations are positive integer generation chunks. An empty array means switch directly to the latest user prompt. "
            "After the plan, the runner always restores the unchanged latest user prompt.\n\n" + skill
        )},
        {"role": "user", "content": content},
    ]
    client = client or model_client(settings, "controller")
    before = time.perf_counter()
    sampling = settings.get("controller_sampling") or {"temperature": 0.0, "max_tokens": 4096}
    message, usage = client.chat(messages, **sampling)
    reply = parse_json_reply(message.get("content"))
    steps = validate_plan(reply, request["budget_chunks"])
    return {
        "request_id": request["request_id"], "steps": steps,
        "skill_sha256": hashlib.sha256(skill.encode()).hexdigest(),
        "planning_s": time.perf_counter() - before, "usage": asdict(usage),
        "model": client.model, "observation_time_s": now,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    request = json.loads(args.request.read_text())
    result = plan(request, args.skill.read_text(), settings)
    args.response.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
