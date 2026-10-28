import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from hermit.evolver.base import Usage
from hermit.io import atomic_json, atomic_text
from examples.video_transition.utils.artifacts import completed_case, file_sha256, prepare_case
from examples.video_transition.utils.resume_training import ReflectionCache
from examples.video_transition.utils.run_job import recover_git_locks, run_job, verify_snapshot
from examples.video_transition.utils.submit import script_text, submit_run, wall_seconds


def test_atomic_publish_leaves_previous_checkpoint_after_failed_rename(tmp_path, monkeypatch):
    import hermit.io as io
    path = tmp_path / "checkpoint.json"
    atomic_json(path, {"step": 1})

    def crash(*args):
        raise OSError("simulated interruption")
    monkeypatch.setattr(io.os, "replace", crash)
    with pytest.raises(OSError):
        atomic_json(path, {"step": 2})
    assert json.loads(path.read_text()) == {"step": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_resume_verifies_video_and_archives_incomplete_attempt(tmp_path):
    task = {"id": "example", "seed": 0}
    case = tmp_path / "example"
    assert not prepare_case(case, task, "baseline", "generation", False, True)
    video = case / "video.mp4"
    video.write_bytes(b"a completed encoded video")
    manifest = {"task": task, "mode": "baseline", "generation_id": "generation", "status": "complete",
                "video_path": str(video), "video_sha256": file_sha256(video)}
    atomic_json(case / "manifest.json", manifest)
    assert prepare_case(case, task, "baseline", "generation", False, True)
    with pytest.raises(ValueError, match="generator"):
        prepare_case(case, task, "baseline", "other-generation", False, True)
    with pytest.raises(ValueError, match="task"):
        prepare_case(case, {**task, "seed": 1}, "baseline", "generation", False, True)
    video.write_bytes(b"truncated")
    assert not completed_case(case, task, "baseline")
    assert not prepare_case(case, task, "baseline", "generation", False, True)
    assert not video.exists()
    attempts = list((case / "attempts").iterdir())
    assert len(attempts) == 1 and (attempts[0] / "video.mp4").read_bytes() == b"truncated"


def test_reflection_retry_reuses_response_only_within_same_step(tmp_path):
    class Client:
        model, base_url, extra_body = "fake", "http://example.test", {}
        calls = 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            return {"content": "a proposal", "reasoning_content": "do not persist"}, Usage(model=self.model, input_tokens=10, calls=1)
    client = Client()
    first = ReflectionCache(client, tmp_path)
    first.scope = "step-1"
    first.chat([{"role": "user", "content": "same feedback"}], max_tokens=100)
    resumed = ReflectionCache(client, tmp_path)
    resumed.scope = "step-1"
    reply, usage = resumed.chat([{"role": "user", "content": "same feedback"}], max_tokens=100)
    assert reply["content"] == "a proposal" and client.calls == 1
    assert usage.input_tokens == 0 and usage.calls == 0 and usage.cost_usd == 0
    assert "reasoning_content" not in json.loads(next(tmp_path.glob("*.json")).read_text())["message"]
    resumed.scope = "step-2"
    resumed.chat([{"role": "user", "content": "same feedback"}], max_tokens=100)
    assert client.calls == 2


def test_job_resumes_after_failed_stage_without_repeating_finished_stage(tmp_path):
    stages = []
    for name in ("first", "second"):
        destination = tmp_path / f"{name}.json"
        count = tmp_path / f"{name}.calls"
        code = f"from pathlib import Path; p=Path({str(count)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
        if name == "second":
            code += f"import sys; sys.exit(1) if not Path({str(tmp_path / 'allow')!r}).exists() else None; "
        code += f"Path({str(destination)!r}).write_text('{{\"status\":\"complete\"}}')"
        stages.append({"id": name, "kind": "json", "completion": str(destination), "command": [sys.executable, "-c", code]})
    atomic_json(tmp_path / "plan.json", {"mode": "score", "sha256": {}, "stages": stages})
    with pytest.raises(RuntimeError, match="second"):
        run_job(tmp_path)
    (tmp_path / "allow").touch()
    run_job(tmp_path)
    assert (tmp_path / "first.calls").read_text() == "x"
    assert (tmp_path / "second.calls").read_text() == "xx"
    assert json.loads((tmp_path / "job-complete.json").read_text())["status"] == "complete"


def test_resume_refuses_changed_frozen_source(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("original")
    plan = {"sha256": {"source.py": file_sha256(path)}}
    verify_snapshot(tmp_path, plan)
    path.write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(tmp_path, plan)


def test_resume_preserves_stale_git_locks_outside_metadata(tmp_path):
    stale=tmp_path/'hermit/trees/.git/worktrees/run/index.lock'
    stale.parent.mkdir(parents=True)
    stale.write_text('interrupted git write')
    unrelated=tmp_path/'other.lock'
    unrelated.touch()
    recover_git_locks(tmp_path)
    assert not stale.exists() and unrelated.exists()
    saved=list((tmp_path/'recovered-git-locks').rglob('index.lock'))
    assert len(saved)==1 and saved[0].read_text()=='interrupted git write'


def test_manual_resume_refuses_an_active_job(tmp_path, monkeypatch):
    atomic_json(tmp_path/'job.json',{'job_id':'123'})
    calls=[]

    def active(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command,0,stdout='123\n',stderr='')
    monkeypatch.setattr(subprocess,'run',active)
    with pytest.raises(ValueError,match='already pending or active'):
        submit_run(tmp_path,{},None,tmp_path)
    assert len(calls)==1 and calls[0][0]=='squeue'


def test_24_hour_boundary_and_slurm_warning_requeue(tmp_path):
    assert wall_seconds("24:00:00") == wall_seconds("1-00:00:00") == 86400
    for invalid in ("24:00:01", "2-00:00:00", "00:00:00", "01:99:00", "bad"):
        with pytest.raises(Exception):
            wall_seconds(invalid)
    utils = tmp_path / "source/examples/video_transition/utils"
    utils.mkdir(parents=True)
    ready = tmp_path / "ready"
    (utils / "run_job.py").write_text(f"from pathlib import Path\nimport time\nPath({str(ready)!r}).touch()\ntime.sleep(30)\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    called = tmp_path / "requeue"
    scontrol = bin_dir / "scontrol"
    scontrol.write_text(f"#!/bin/bash\nprintf '%s ' \"$@\" > {str(called)!r}\n")
    scontrol.chmod(0o755)
    resources = {"time": "24:00:00", "partition": "commons", "mem": "1G", "cpus": 1}
    settings = {"orchestrator_python": sys.executable, "generator": {"environment": {"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}}}
    script = tmp_path / "run.sbatch"
    script.write_text(script_text(tmp_path, settings, "score", resources, True))
    assert "#SBATCH --signal=B:USR1@120" in script.read_text()
    process = subprocess.Popen(["bash", str(script)], env={**os.environ, "SLURM_JOB_ID": "123"}, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        process.send_signal(signal.SIGUSR1)
        assert process.wait(timeout=5) == 0
        assert called.read_text() == "requeue 123 "
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
