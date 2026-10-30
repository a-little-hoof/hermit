"""Score ordered video frames with a signed-in Codex model and native JSON Schema output."""
from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import uuid

from hermit.evolver.base import Usage
from hermit.io import atomic_json, atomic_text
from examples.video_transition.utils.artifacts import file_sha256

JUDGMENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "judgments": {"type": "array", "minItems": 10, "maxItems": 10,
                      "items": {"type": "integer", "enum": [0, 1]}},
        "sum": {"type": "integer", "enum": list(range(11))},
    }, "required": ["judgments", "sum"],
}
ATTACHMENT_RULE = ("\nAttached images are numbered in attachment order and correspond to the image markers "
                   "in the supplied text. Evaluate only these inputs. Do not use tools or external sources.\n")


class CodexEvaluatorClient:
    transport = "codex-exec"
    base_url = "codex://chatgpt"

    def __init__(self, spec: dict, directory: Path):
        self.model = spec.get("model")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("Configure evaluator.model for Codex vision scoring")
        self.effort = spec.get("reasoning_effort", "max")
        self.timeout = float(spec.get("timeout", 1200))
        if self.timeout <= 0:
            raise ValueError("evaluator.timeout must be positive")
        binary = shutil.which(str(spec.get("binary", "codex")))
        if not binary:
            raise RuntimeError("Codex CLI is unavailable on the evaluator host")
        self.binary = str(Path(binary).resolve())
        digest = file_sha256(Path(self.binary))
        if spec.get("binary_sha256") and digest != spec["binary_sha256"]:
            raise RuntimeError("Codex binary differs from evaluator.binary_sha256")
        self.extra_body = {"transport": self.transport, "reasoning_effort": self.effort,
                           "binary_sha256": digest,
                           "attachment_rule_sha256": hashlib.sha256(ATTACHMENT_RULE.encode()).hexdigest()}
        self.directory = Path(directory).resolve()
        self.total = Usage(model=self.model)
        self._lock = threading.Lock()

    @staticmethod
    def environment() -> dict:
        removed = {"OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "OPENAI_BASE_URL",
                   "OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "CODEX_THREAD_ID", "CODEX_TURN_ID"}
        return {k: v for k, v in os.environ.items() if k not in removed
                and not k.startswith("CODEX_INTERNAL_") and not k.endswith("API_KEY")}

    def check_auth(self) -> str:
        result = subprocess.run([self.binary, "login", "status"], capture_output=True, text=True,
                                env=self.environment(), timeout=30)
        status = (result.stdout + result.stderr).strip()
        if result.returncode or "Logged in using ChatGPT" not in status:
            raise RuntimeError("Codex evaluator requires a ChatGPT login; run `codex login` on this host")
        return status

    def chat(self, messages: list[dict], **sampling) -> tuple[dict, Usage]:
        supported = {"response_format", "reasoning_effort"}
        if any(value is not None for key, value in sampling.items() if key not in supported):
            raise ValueError("Codex evaluator uses native output limits; remove API sampling parameters")
        if sampling.get("reasoning_effort", self.effort) != self.effort:
            raise ValueError("sampling reasoning effort differs from evaluator.reasoning_effort")
        formatting = sampling.get("response_format")
        if formatting and (formatting.get("type") != "json_schema" or
                           formatting.get("json_schema", {}).get("schema") != JUDGMENT_SCHEMA):
            raise ValueError("Codex evaluator requires the fixed ten-binary-judgment schema")
        if len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user" \
                or not isinstance(messages[0].get("content"), str) or not isinstance(messages[1].get("content"), list):
            raise ValueError("Codex evaluator expects one system instruction and one multimodal user request")
        self.check_auth()
        attempt = self.directory / uuid.uuid4().hex
        workspace = attempt / "workspace"
        workspace.mkdir(parents=True)
        text, images, evidence = [], [], []
        for part in messages[1]["content"]:
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                text.append(part["text"])
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                header, separator, encoded = url.partition(",")
                if not separator or header not in ("data:image/jpeg;base64", "data:image/png;base64"):
                    raise ValueError("Codex evaluator requires local JPEG/PNG image data")
                data = base64.b64decode(encoded, validate=True)
                if not data:
                    raise ValueError("empty evaluator image")
                image = workspace / f"image-{len(images) + 1:02d}.{'jpg' if 'jpeg' in header else 'png'}"
                image.write_bytes(data)
                images.append(image)
                evidence.append({"attachment": len(images), "path": str(image),
                                 "sha256": hashlib.sha256(data).hexdigest()})
                text.append(f"[Attached image {len(images)}: the frame described immediately above.]")
            else:
                raise ValueError("unsupported evaluator content part")
        if not 1 <= len(images) <= 8:
            raise ValueError("each metric needs between one and eight ordered images")
        instructions = attempt / "instructions.txt"
        atomic_text(instructions, messages[0]["content"] + ATTACHMENT_RULE)
        schema = attempt / "schema.json"
        atomic_json(schema, JUDGMENT_SCHEMA)
        prompt = "\n\n".join(text)
        atomic_text(attempt / "prompt.txt", prompt)
        answer = attempt / "answer.json"
        command = [self.binary, "exec", "--json", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                   "--skip-git-repo-check", "--sandbox", "read-only", "-C", str(workspace),
                   "--output-last-message", str(answer), "--output-schema", str(schema), "--model", self.model,
                   "-c", 'model_provider="openai"', "-c", f"model_reasoning_effort={json.dumps(self.effort)}",
                   "-c", 'model_reasoning_summary="none"', "-c", "project_doc_max_bytes=0",
                   "-c", f"model_instructions_file={json.dumps(str(instructions))}", "-c", 'web_search="disabled"']
        for feature in ("shell_tool", "shell_snapshot", "multi_agent", "apps", "plugins", "hooks", "memories",
                        "skill_search", "browser_use", "computer_use", "image_generation", "view_image", "sleep_tool"):
            command += ["-c", f"features.{feature}=false"]
        command += ["-c", "features.skip_host_skill_discovery=true"]
        for image in images:
            command += ["--image", str(image)]
        command += ["-"]
        atomic_json(attempt / "request.json", {"model": self.model, **self.extra_body, "command": command,
                    "evidence": evidence, "status": "running", "timeout": self.timeout})
        started = time.monotonic()
        with (attempt / "stderr.log").open("w") as error_log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error_log,
                                       text=True, env=self.environment(), cwd=workspace, start_new_session=True)
            try:
                output, _ = process.communicate(prompt, timeout=self.timeout)
            except BaseException as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                atomic_json(attempt / "result.json", {"status": "failed", "error": type(exc).__name__,
                                                       "elapsed_s": time.monotonic() - started})
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise RuntimeError(f"Codex evaluator exceeded {self.timeout:g}s; see {attempt}") from None
                raise
        events = []
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if (event.get("item") or {}).get("type") != "reasoning":
                events.append(event)
        atomic_text(attempt / "events.jsonl", "".join(json.dumps(e) + "\n" for e in events))
        completed = [e for e in events if e.get("type") == "turn.completed"]
        failed = any(e.get("type") in ("error", "turn.failed") for e in events)
        tools = [e for e in events if e.get("type") in ("item.started", "item.completed")
                 and (e.get("item") or {}).get("type") not in ("agent_message", "reasoning", "error")]
        content = answer.read_text().strip() if answer.exists() else ""
        if process.returncode or failed or len(completed) != 1 or not completed[0].get("usage") or not content or tools:
            atomic_json(attempt / "result.json", {"status": "failed", "returncode": process.returncode,
                                                   "elapsed_s": time.monotonic() - started})
            raise RuntimeError(f"Codex evaluator did not complete an image scoring turn; see {attempt}")
        reply = json.loads(content)
        values = reply.get("judgments")
        if set(reply) != {"judgments", "sum"} or not isinstance(values, list) or len(values) != 10 or \
                any(type(v) is not int or v not in (0, 1) for v in values) or type(reply.get("sum")) is not int:
            raise ValueError("Codex evaluator returned invalid binary judgments")
        # The shared evaluator checks redundant arithmetic and preserves the original response for recovery.
        raw = completed[0]["usage"]
        usage = Usage(model=self.model, input_tokens=int(raw.get("input_tokens", 0)),
                      cached_input_tokens=int(raw.get("cached_input_tokens", 0)),
                      output_tokens=int(raw.get("output_tokens", 0)), calls=1)
        atomic_json(attempt / "result.json", {"status": "complete", "usage": asdict(usage),
                                               "elapsed_s": time.monotonic() - started})
        with self._lock:
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "calls"):
                setattr(self.total, key, getattr(self.total, key) + getattr(usage, key))
        return {"role": "assistant", "content": content, "finish_reason": "stop"}, usage


def evaluator_client(settings: dict, destination: Path):
    spec = settings.get("evaluator") or {}
    transport = spec.get("transport", "chat")
    if transport == "codex-exec":
        return CodexEvaluatorClient(spec, Path(destination) / "codex-calls")
    if transport != "chat":
        raise ValueError(f"unknown evaluator.transport: {transport!r}")
    from examples.video_transition.utils.remote import model_client
    return model_client(settings, "evaluator")
