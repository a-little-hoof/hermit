"""Task-specific model settings; credentials stay in environment variables."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import shlex
import stat

from hermit.llm import ChatClient


def load_settings(path: Path | str | None = None) -> dict:
    name = path or os.environ.get("VIDEO_TRANSITION_CONFIG")
    if not name:
        raise RuntimeError("Set VIDEO_TRANSITION_CONFIG to your runtime.local.json (see runtime.example.json)")
    source = Path(name).expanduser().resolve()
    settings = json.loads(source.read_text())
    settings["_config_path"] = str(source)
    if settings.get("secrets_file"):
        secret_file = Path(settings["secrets_file"]).expanduser()
        if not secret_file.is_absolute():
            secret_file = source.parent / secret_file
        if stat.S_IMODE(secret_file.stat().st_mode) & 0o077:
            raise ValueError(f"credential file needs mode 600: {secret_file}")
        for line in secret_file.read_text().splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("export "):
                text = text[7:]
            key, separator, value = text.partition("=")
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key):
                raise ValueError("credential file must contain literal KEY=value assignments")
            parts = shlex.split(value, comments=True)
            if len(parts) != 1:
                raise ValueError("credential values must be one literal, optionally quoted")
            os.environ.setdefault(key, parts[0])
    return settings


def model_client(settings: dict, role: str) -> ChatClient:
    spec = settings.get(role) or {}
    for field in ("base_url", "model"):
        value = spec.get(field)
        if not isinstance(value, str) or not value or "<" in value:
            raise RuntimeError(f"Configure {role}.{field} in runtime.local.json")
    return ChatClient.from_spec(spec, max_retries=int(spec.get("max_retries", 1)))


def image_part(path: Path | str) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    suffix = Path(path).suffix.lower()
    media = "image/png" if suffix == ".png" else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}


def parse_json_reply(text: str | None) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("model returned no JSON content")
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("unterminated JSON code fence")
        text = "\n".join(lines[1:-1])
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("model reply must be a JSON object")
    return result
