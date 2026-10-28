# The denoising loop is adapted from NVIDIA LongLive 1.0,
# pipeline/interactive_causal_inference.py at cfa4e91eec59429d583326500bf5535c9ce6eec7.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Causal prompt scheduling around LongLive's unchanged model and cache helpers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

from examples.video_transition.utils.protocol import FPS, latent_boundary, pixel_time


class PromptSchedule:
    """A newer instruction invalidates the unexecuted plan and stale replies."""
    def __init__(self, initial: str):
        self.latest = initial
        self.active = initial
        self.version = 0
        self.pending = False
        self.remaining = 0
        self.steps = []

    def request(self, prompt: str) -> int:
        self.latest = prompt
        self.version += 1
        self.pending = True
        self.remaining = 0
        self.steps.clear()
        return self.version

    def install(self, version: int, steps: list[dict]) -> bool:
        if version != self.version:
            return False
        self.steps = list(steps)
        self.remaining = 0
        self.pending = False
        return True

    def next_prompt(self) -> str:
        if self.pending:
            return self.active
        if self.remaining == 0:
            if self.steps:
                step = self.steps.pop(0)
                self.active = step["prompt"]
                self.remaining = step["duration_chunks"]
            else:
                self.active = self.latest
        if self.remaining:
            self.remaining -= 1
        return self.active


def offline_plan(schedule, version, query, root):
    """Finish planning before the caller advances to the next generation block."""
    response = query(root)
    return response, schedule.install(version, response["steps"])


def _observations(pipeline, latents, current_start: int, case: Path, version: int):
    import torch
    from PIL import Image

    preview = pipeline.vae.decode_to_pixel(latents[:, :current_start], use_cache=False)
    preview = (preview * 0.5 + 0.5).clamp(0, 1)
    last = preview.shape[1] - 1
    first = max(0, last - 2 * FPS)
    indices = sorted(set(round(first + (last - first) * j / 3) for j in range(4)))
    root = case / "controller" / f"request-{version:02d}"
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for index in indices:
        array = (preview[0, index].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
        path = root / f"frame-{index:05d}.jpg"
        Image.fromarray(array).save(path, quality=95)
        result.append({"time_s": index / FPS, "path": str(path)})
    return root, result


def controlled_inference(pipeline, noise, task, case, settings=None, skill_path=None):
    """With settings=None, execute raw switches for comparison to official inference."""
    import torch

    batch_size, length, channels, height, width = noise.shape
    block = pipeline.num_frame_per_block
    planning_mode = settings.get("planning_mode", "async") if settings else "identity"
    if planning_mode not in ("async", "offline", "identity"):
        raise ValueError(f"unknown planning_mode: {planning_mode}")
    output = torch.zeros_like(noise)
    pipeline._initialize_kv_cache(batch_size, dtype=noise.dtype, device=noise.device,
                                  kv_cache_size_override=pipeline.local_attn_size * pipeline.frame_seq_length)
    pipeline._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
    pipeline.generator.model.local_attn_size = pipeline.local_attn_size
    pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)
    schedule = PromptSchedule(task["events"][0]["prompt"])
    active = schedule.active
    conditional = pipeline.text_encoder(text_prompts=[active])
    event_index = 1
    pending = []
    trace, errors = [], []
    executor = ThreadPoolExecutor(max_workers=2)
    started = time.perf_counter()

    def query(root):
        command = [settings["orchestrator_python"], "-u", str(Path(__file__).with_name("controller.py")),
                   "--config", settings["_config_path"], "--skill", str(skill_path),
                   "--request", str(root / "request.json"), "--response", str(root / "response.json")]
        with (root / "process.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=float(settings.get("controller_process_timeout_s", 300)))
        if result.returncode:
            raise RuntimeError(f"controller exited {result.returncode}; inspect {root / 'process.log'}")
        return json.loads((root / "response.json").read_text())

    try:
        for current_start in range(0, length, block):
            while event_index < len(task["events"]) and current_start >= latent_boundary(task["events"][event_index]["time_s"]):
                event = task["events"][event_index]
                previous = schedule.latest
                version = schedule.request(event["prompt"])
                trace.append({"event": "instruction_arrived", "request_id": version,
                              "requested_video_time_s": event["time_s"], "observed_video_time_s": pixel_time(current_start),
                              "wall_s": time.perf_counter() - started})
                if settings is None:
                    schedule.install(version, [])
                else:
                    root, frames = _observations(pipeline, output, current_start, Path(case), version)
                    request = {"request_id": version, "video_time_s": pixel_time(current_start),
                               "previous_user_prompt": previous, "latest_user_prompt": event["prompt"],
                               "active_prompt": active, "frames": frames,
                               "budget_chunks": min(int(settings.get("budget_chunks", 4)), (length-current_start)//block)}
                    (root / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
                    if planning_mode == "offline":
                        response, applied = offline_plan(schedule, version, query, root)
                        trace.append({"event": "plan_returned", "request_id": version, "accepted": applied,
                                      "planning_mode": "offline", "video_time_s": pixel_time(current_start),
                                      "response": response, "wall_s": time.perf_counter() - started})
                    else:
                        pending.append((version, executor.submit(query, root)))
                event_index += 1
            for version, future in list(pending):
                if not future.done():
                    continue
                pending.remove((version, future))
                try:
                    response = future.result()
                    applied = schedule.install(version, response["steps"])
                    trace.append({"event": "plan_returned", "request_id": version, "accepted": applied,
                                  "video_time_s": pixel_time(current_start), "response": response,
                                  "wall_s": time.perf_counter() - started})
                except Exception as error:
                    errors.append(str(error))
                    schedule.install(version, [])
            prompt = schedule.next_prompt()
            if prompt != active:
                conditional = pipeline.text_encoder(text_prompts=[prompt])
                pipeline._recache_after_switch(output, current_start, conditional)
                active = prompt
                trace.append({"event": "prompt_applied", "prompt": prompt, "request_id": schedule.version,
                              "video_time_s": pixel_time(current_start), "wall_s": time.perf_counter() - started})
            noisy = noise[:, current_start:current_start + block]
            for index, timestep_value in enumerate(pipeline.denoising_step_list):
                timestep = torch.ones([batch_size, block], device=noise.device, dtype=torch.int64) * timestep_value
                _, denoised = pipeline.generator(noisy_image_or_video=noisy, conditional_dict=conditional, timestep=timestep,
                    kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start * pipeline.frame_seq_length)
                if index < len(pipeline.denoising_step_list)-1:
                    next_timestep = pipeline.denoising_step_list[index+1]
                    noisy = pipeline.scheduler.add_noise(denoised.flatten(0,1), torch.randn_like(denoised.flatten(0,1)),
                        next_timestep * torch.ones([batch_size*block], device=noise.device, dtype=torch.long)).unflatten(0, denoised.shape[:2])
            output[:, current_start:current_start + block] = denoised
            pipeline.generator(noisy_image_or_video=denoised, conditional_dict=conditional,
                timestep=torch.ones_like(timestep)*pipeline.args.context_noise,
                kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start * pipeline.frame_seq_length)
        # A late response is recorded but cannot rewrite frames already generated.
        for version, future in pending:
            try:
                response = future.result()
                trace.append({"event": "plan_after_video_end", "request_id": version, "accepted": False, "response": response})
            except Exception as error:
                errors.append(str(error))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    video = pipeline.vae.decode_to_pixel(output, use_cache=False)
    video = (video*0.5+0.5).clamp(0,1)
    return video, output, trace, errors
