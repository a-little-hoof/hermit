from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from examples.video_transition.utils.prepare import prepare
from examples.video_transition.utils.protocol import latent_boundary, metric_times, pixel_time, validate_task
from examples.video_transition.utils.longlive_worker import source_revision
from examples.video_transition.utils.prepare_two_prompt import prepare_pair
from examples.video_transition.utils.prepare_ten_prompt import prepare_transitions


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


def test_two_prompt_debugging_split_preserves_pair_and_separates_seeds(tmp_path):
    task={"id":"example-pair","events":[{"time_s":0,"prompt":"holds cards"},
          {"time_s":10,"prompt":"celebrates"}],"duration_s":20,"seed":0}
    result=prepare_pair(task,tmp_path)
    rows=[json.loads(line) for line in (tmp_path/'tasks.jsonl').read_text().splitlines()]
    assert [row['seed'] for row in rows]==[0,1,2]
    assert all(row['events']==task['events'] for row in rows)
    assert len(set(result['splits'].values()))==3
    assert task['seed']==0 and task['id']=='example-pair'


def test_ten_training_and_four_validation_transitions_have_disjoint_source_scenes(tmp_path):
    original=[{'prompts':[f'scene {i} event {j}' for j in range(6)]} for i in range(12)]
    raw='\n'.join(json.dumps(row) for row in original).encode()
    result=prepare_transitions(tmp_path,raw,'test-revision')
    rows=[json.loads(line) for line in (tmp_path/'tasks.jsonl').read_text().splitlines()]
    by_id={row['id']:row for row in rows}
    assert len(by_id)==28
    assert [result['counts'][name]['prompt_pairs'] for name in ('train','val','test')]==[10,4,2]
    assert [len(result['splits'][name]) for name in ('train','val','test')]==[10,12,6]
    groups={name:{by_id[i]['source_group'] for i in result['splits'][name]} for name in result['splits']}
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    for name in ('val','test'):
        for scene in groups[name]:
            assert {by_id[i]['seed'] for i in result['splits'][name] if by_id[i]['source_group']==scene}=={0,1,2}
    for row in rows:
        source=row['source']
        assert [e['prompt'] for e in row['events']]==[original[source['row']]['prompts'][j] for j in source['prompt_indices']]
        assert [e['time_s'] for e in row['events']]==[0,10] and row['duration_s']==20
    assert result==prepare_transitions(tmp_path,raw,'test-revision')
    changed=json.loads(json.dumps(original));changed[2]['prompts'][0]='changed training prompt'
    with pytest.raises(ValueError,match='refusing to replace'):
        prepare_transitions(tmp_path,'\n'.join(json.dumps(row) for row in changed).encode(),'test-revision')


def test_ten_prompt_preparation_refuses_unpinned_official_bytes(tmp_path):
    with pytest.raises(ValueError,match='pinned source checksum'):
        prepare_transitions(tmp_path,b'{}')
