"""Run frozen official LongLive inference in its separate CUDA environment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.video_transition.utils.protocol import FPS, latent_boundary, pixel_time, validate_task
from examples.video_transition.utils.artifacts import file_sha256, prepare_case
from hermit.io import atomic_json


def source_revision(source: Path) -> str:
    """Read a checkout's commit without requiring Git inside the CUDA job image."""
    for root in (source, *source.parents):
        git = root / ".git"
        if not git.exists():
            continue
        if git.is_file():
            git = (root / git.read_text().strip().removeprefix("gitdir: ")).resolve()
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        reference = head[5:]
        common = (git / (git / "commondir").read_text().strip()).resolve() if (git / "commondir").is_file() else git
        for candidate in (git / reference, common / reference):
            if candidate.is_file():
                return candidate.read_text().strip()
        for line in (common / "packed-refs").read_text().splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[1] == reference:
                return fields[0]
        raise ValueError(f"cannot resolve official source ref {reference}")
    raise ValueError("official source must be inside a versioned checkout")


def encode_video(path, video):
    import av
    import torch

    frames = (video[0].permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16", "preset": "medium"}
        for array in frames:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return list(frames.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--mode", choices=("baseline", "identity", "controller"), default="baseline")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse verified complete cases and retry interrupted cases")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.runtime_config:
        args.runtime_config = args.runtime_config.resolve()
    if args.skill:
        args.skill = args.skill.resolve()
    settings = json.loads(args.runtime_config.read_text()) if args.runtime_config else None
    if settings is not None:
        settings["_config_path"] = str(args.runtime_config)
    if args.mode == "controller" and (settings is None or args.skill is None):
        raise ValueError("controller mode needs --runtime-config and --skill")
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if args.ids_file:
        ids = set(args.ids_file.read_text().split())
        tasks = [task for task in tasks if task["id"] in ids]
        if {task["id"] for task in tasks} != ids:
            raise ValueError("ids-file names tasks that are not present")
    if not tasks:
        raise ValueError("no tasks selected")
    for task in tasks:
        validate_task(task)
    identity = {"mode": args.mode, "source_revision": source_revision(source),
                "assets": [{"path": str(path.resolve()), "bytes": path.stat().st_size,
                            "mtime_ns": path.stat().st_mtime_ns}
                           for path in (args.base_checkpoint, args.lora_checkpoint, *sorted(args.models.glob("*.pth")),
                                        args.models / "diffusion_pytorch_model.safetensors")],
                "code": {name: file_sha256(Path(__file__).with_name(name)) for name in
                         ("longlive_worker.py", "streaming.py", "controller.py", "remote.py", "protocol.py", "artifacts.py")}}
    if args.mode == "controller":
        identity.update(skill=args.skill.read_text(), controller=settings["controller"],
                        sampling=settings.get("controller_sampling"), budget_chunks=settings.get("budget_chunks", 4),
                        planning_mode=settings.get("planning_mode", "async"))
    generation_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    pending = [task for task in tasks if not prepare_case(destination / task["id"], task, args.mode,
                                                          generation_id, args.save_latents, args.resume)]
    if not pending:
        atomic_json(destination / "complete.json", {"tasks": [task["id"] for task in tasks], "status": "complete"})
        print("All requested videos are verified cache hits.", flush=True)
        return
    model_link = destination / "runtime/wan_models/Wan2.1-T2V-1.3B"
    model_link.parent.mkdir(parents=True, exist_ok=True)
    if model_link.is_symlink():
        if model_link.resolve() != args.models.resolve():
            raise ValueError("output runtime already points at different Wan assets")
    elif model_link.exists():
        raise ValueError("refusing to replace an existing runtime model directory")
    else:
        model_link.symlink_to(args.models.resolve(), target_is_directory=True)
    for name in ("config.json", "diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
        if not (model_link / name).is_file():
            raise FileNotFoundError(f"missing Wan asset: {name}")
    for p in (args.base_checkpoint, args.lora_checkpoint):
        if not p.is_file():
            raise FileNotFoundError(p)
    sys.path.insert(0, str(source))
    os.chdir(destination / "runtime")

    import torch
    import peft
    from omegaconf import OmegaConf
    from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
    from utils.lora_utils import configure_lora_for_model
    from utils.misc import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("LongLive generation requires an allocated CUDA GPU")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    started = time.perf_counter()
    config = OmegaConf.create({
        "denoising_step_list": [1000, 750, 500, 250], "warp_denoising_step": True,
        "num_frame_per_block": 3, "model_name": "Wan2.1-T2V-1.3B",
        "model_kwargs": {"local_attn_size": 12, "timestep_shift": 5.0, "sink_size": 3},
        "global_sink": True, "context_noise": 0,
        "adapter": {"type": "lora", "rank": 256, "alpha": 256, "dropout": 0.0, "verbose": False},
    })
    pipeline = InteractiveCausalInferencePipeline(config, device=device)
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    pipeline.generator.load_state_dict(checkpoint["generator"])
    del checkpoint
    pipeline.generator.model = configure_lora_for_model(pipeline.generator.model, "generator", config.adapter)
    lora = torch.load(args.lora_checkpoint, map_location="cpu", weights_only=False)
    lora_state = lora.get("generator_lora", lora)
    loaded = peft.set_peft_model_state_dict(pipeline.generator.model, lora_state)
    missing = [key for key in loaded.missing_keys if "lora_" in key]
    if missing or loaded.unexpected_keys:
        raise RuntimeError(f"LoRA mismatch: {missing[:5]}, {loaded.unexpected_keys[:5]}")
    lora_count = len(lora_state)
    del lora, lora_state
    pipeline.is_lora_enabled = True
    pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    pipeline.eval()
    torch.cuda.synchronize()
    provenance = {
        "official_source": str(source),
        "official_commit": source_revision(source),
        "inference": ("official InteractiveCausalInferencePipeline.inference" if args.mode == "baseline"
                      else "causal controller loop adapted from official inference"),
        "mode": args.mode,
        "planning_mode": settings.get("planning_mode", "async") if args.mode == "controller" else "raw",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "peft": peft.__version__,
        "base_checkpoint": str(args.base_checkpoint), "lora_checkpoint": str(args.lora_checkpoint),
        "lora_tensor_count": lora_count, "config": OmegaConf.to_container(config),
        "startup_s": time.perf_counter() - started,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    provenance_path = destination / "provenance" / f"{stamp}.json"
    atomic_json(provenance_path, provenance)
    atomic_json(destination / "provenance.json", provenance)
    original_recache = pipeline._recache_after_switch
    for task in pending:
        case = destination / task["id"]
        case.mkdir(exist_ok=True)
        task_start = time.perf_counter()
        set_seed(int(task["seed"]))
        torch.cuda.reset_peak_memory_stats()
        switches = []

        def recache(self, output, current_start_frame, conditional):
            torch.cuda.synchronize()
            before = time.perf_counter()
            result = original_recache(output, current_start_frame, conditional)
            torch.cuda.synchronize()
            switches.append({"latent_index": current_start_frame, "effective_video_time_s": pixel_time(current_start_frame),
                             "recache_s": time.perf_counter() - before})
            return result

        pipeline._recache_after_switch = types.MethodType(recache, pipeline)
        latent_count = latent_boundary(float(task["duration_s"]))
        noise = torch.randn([1, latent_count, 16, 60, 104], device=device, dtype=torch.bfloat16)
        if args.mode == "baseline":
            video, latents = pipeline.inference(
                noise=noise,
                text_prompts_list=[[event["prompt"]] for event in task["events"]],
                switch_frame_indices=[latent_boundary(float(event["time_s"])) for event in task["events"][1:]],
                return_latents=True,
            )
            controller_trace, control_errors = [], []
        else:
            from examples.video_transition.utils.streaming import controlled_inference
            video, latents, controller_trace, control_errors = controlled_inference(
                pipeline, noise, task, case, settings if args.mode == "controller" else None, args.skill)
        torch.cuda.synchronize()
        generation_s = time.perf_counter() - task_start
        video_path = case / "video.mp4"
        temporary_video = case / ".video.partial.mp4"
        shape = encode_video(temporary_video, video)
        temporary_video.replace(video_path)
        manifest = {
            "status": "error" if control_errors else "complete", "mode": args.mode, "task": task,
            "planning_mode": provenance["planning_mode"],
            "video_path": str(video_path), "frames_shape": shape, "fps": FPS,
            "duration_s": shape[0] / FPS, "latent_count": latent_count, "switches": switches,
            "generation_and_decode_s": generation_s, "total_s": time.perf_counter() - task_start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "provenance_path": str(provenance_path), "generation_id": generation_id,
            "video_sha256": file_sha256(video_path),
            "controller_trace": controller_trace, "control_errors": control_errors,
        }
        if args.save_latents:
            temporary_latents = case / ".latents.partial.pt"
            torch.save(latents.detach().cpu(), temporary_latents)
            temporary_latents.replace(case / "latents.pt")
            manifest["latents_sha256"] = file_sha256(case / "latents.pt")
        atomic_json(case / "manifest.json", manifest)
        print("CASE_COMPLETE", json.dumps({"id": task["id"], "generation_s": generation_s, "shape": shape}), flush=True)
        if control_errors:
            raise RuntimeError(f"controller failed; inspect {case / 'manifest.json'}")
        del noise, video, latents
    atomic_json(destination / "complete.json", {"tasks": [task["id"] for task in tasks], "status": "complete"})


if __name__ == "__main__":
    main()
