from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from examples.video_transition.utils.prepare import prepare
from examples.video_transition.utils.protocol import latent_boundary, metric_times, pixel_time, validate_task
from examples.video_transition.utils.longlive_worker import source_revision


def test_causal_decode_timing_and_chunk_quantization():
    assert latent_boundary(0) == 0
    assert latent_boundary(10) == 42
    assert pixel_time(42) == 10.3125
    assert pixel_time(39) < 10
    assert latent_boundary(20) == 81
    with pytest.raises(ValueError):
        latent_boundary(float("nan"))


def test_metric_frames_match_paper_sampling():
    assert metric_times("OSE", 20) == pytest.approx([19.5, 20.3, 20.6, 20.9, 21.2])
    assert metric_times("VBS", 20) == metric_times("MBS", 20)
    assert len(metric_times("TP", 20)) == 8
    assert metric_times("SUF", 20)[2:] == [21, 22, 23, 24]
    with pytest.raises(ValueError):
        metric_times("unknown", 20)


def test_preparation_keeps_pilot_groups_out_of_formal_holdout(tmp_path):
    raw="\n".join(json.dumps({"prompts": [f"initial {i}", f"target {i}"]}) for i in range(12)).encode()
    result=prepare(tmp_path, raw, "test-revision")
    train, val, test = (set(result["splits"][name]) for name in ("train", "val", "test"))
    assert [len(train),len(val),len(test)] == [8,2,2]
    assert not (train & val or train & test or val & test)
    assert all(t.removesuffix("-pilot") in train for t in result["pilot_ids"])
    multi=[json.loads(line) for line in (tmp_path/'tasks_multiseed.jsonl').read_text().splitlines()]
    test_rows=[row for row in multi if row['id'] in result['test_seed_ids']]
    assert len(test_rows)==6
    assert set(row['seed'] for row in test_rows)=={0,1,2}
    assert set(row['source_group'] for row in test_rows)=={f"longlive-row-{int(name.rsplit('-',1)[1]):02d}" for name in test}
    assert result == prepare(tmp_path, raw, "test-revision")


def test_short_final_window_and_colliding_events_are_rejected():
    task={"id":"x", "events":[{"time_s":0,"prompt":"old"},{"time_s":10,"prompt":"new"}],"duration_s":14}
    with pytest.raises(ValueError, match="five seconds"):
        validate_task(task)
    task["duration_s"]=20
    task["events"].append({"time_s":10.01,"prompt":"latest"})
    with pytest.raises(ValueError, match="distinct"):
        validate_task(task)


def test_source_revision_does_not_require_git_binary(tmp_path):
    repo=tmp_path/'official'
    source=repo/'LongLive1.0'
    source.mkdir(parents=True)
    git=repo/'.git'
    (git/'refs/heads').mkdir(parents=True)
    (git/'HEAD').write_text('ref: refs/heads/main\n')
    sha='a'*40
    (git/'refs/heads/main').write_text(sha+'\n')
    assert source_revision(source)==sha
    (git/'refs/heads/main').unlink()
    (git/'packed-refs').write_text(sha+' refs/heads/main\n')
    assert source_revision(source)==sha
