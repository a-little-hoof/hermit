"""Validate completed videos and preserve interrupted attempts before retrying."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_case(case: Path, task: dict, mode: str, generation_id: str | None = None,
                   save_latents: bool = False) -> bool:
    try:
        manifest = json.loads((case / "manifest.json").read_text())
        if manifest.get("task") != task or manifest.get("mode") != mode:
            raise ValueError(f"resume input differs from the saved task or mode: {case}")
        if generation_id is not None and manifest.get("generation_id") != generation_id:
            raise ValueError(f"resume generator, controller or code differs: {case}")
        if manifest.get("status") != "complete" or manifest.get("control_errors"):
            return False
        video = case / "video.mp4"
        if Path(manifest["video_path"]).resolve() != video.resolve():
            raise ValueError(f"saved video path does not belong to this case: {case}")
        if not video.is_file() or video.stat().st_size == 0:
            return False
        if file_sha256(video) != manifest.get("video_sha256"):
            return False
        if save_latents:
            latent = case / "latents.pt"
            if not latent.is_file() or file_sha256(latent) != manifest.get("latents_sha256"):
                return False
        return True
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False


def prepare_case(case: Path, task: dict, mode: str, generation_id: str,
                 save_latents: bool, resume: bool) -> bool:
    """Return True for a verified cache hit; otherwise move old attempt files aside."""
    case.mkdir(parents=True, exist_ok=True)
    previous = [path for path in case.iterdir() if path.name != "attempts"]
    if not previous:
        return False
    if not resume:
        raise FileExistsError(f"case already contains artifacts; use --resume: {case}")
    if completed_case(case, task, mode, generation_id, save_latents):
        return True
    target = case / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target.mkdir(parents=True)
    for path in previous:
        shutil.move(str(path), target / path.name)
    return False
