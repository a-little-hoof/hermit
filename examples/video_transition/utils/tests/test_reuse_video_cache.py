import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from hermit.io import atomic_json
from examples.video_transition.utils.artifacts import completed_case, file_sha256
from examples.video_transition.utils.reuse_video_cache import ACTING_FILES, link_video_cache


def cached_runs(tmp_path):
    source,target=(tmp_path/name for name in ('old','new'))
    task={'id':'example','seed':0,'events':[{'time_s':0,'prompt':'before'},{'time_s':10,'prompt':'after'}],'duration_s':20}
    for run in (source,target):
        atomic_json(run/'runtime.json',{'generator':{'model':'frozen'},'controller':{'model':'same'},'planning_mode':'offline'})
        code=run/'source/examples/video_transition/utils';code.mkdir(parents=True)
        for name in ACTING_FILES:(code/name).write_text(name)
    data=target/'source/data';data.mkdir(parents=True)
    (data/'tasks.jsonl').write_text(json.dumps(task)+'\n')
    case=source/'videos/controller-example-123/example';case.mkdir(parents=True)
    (case/'video.mp4').write_bytes(b'verified video')
    atomic_json(case/'evaluation/summary.json',{'score':1,'model':'old-judge'})
    atomic_json(case/'manifest.json',{'task':task,'mode':'controller','status':'complete',
                'video_path':str(case/'video.mp4'),'video_sha256':file_sha256(case/'video.mp4')})
    return source,target,task,case


def test_linked_videos_keep_provenance_but_never_copy_old_judgments(tmp_path):
    source,target,task,case=cached_runs(tmp_path)
    report=link_video_cache(source,target)
    destination=target/'videos/controller-example-123/example'
    assert report['linked_cases']==1 and not report['previous_judgments_copied']
    assert completed_case(destination,task,'controller')
    assert (destination/'video.mp4').stat().st_ino==(case/'video.mp4').stat().st_ino
    assert not (destination/'evaluation').exists()
    assert json.loads((destination/'manifest.json').read_text())['reused_from']==str(case/'manifest.json')
    assert json.loads((case/'manifest.json').read_text())['video_path']==str(case/'video.mp4')


def test_changed_acting_code_cannot_reuse_a_video(tmp_path):
    source,target,_,_=cached_runs(tmp_path)
    (target/'source/examples/video_transition/utils/controller.py').write_text('changed behavior')
    with pytest.raises(ValueError,match='changed acting code'):
        link_video_cache(source,target)


def test_incomplete_video_is_skipped(tmp_path):
    source,target,_,case=cached_runs(tmp_path)
    (case/'video.mp4').write_bytes(b'truncated')
    report=link_video_cache(source,target)
    assert report['linked_cases']==0 and len(report['skipped_incomplete_or_different_cases'])==1
