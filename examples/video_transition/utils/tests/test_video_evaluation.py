from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import av
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from examples.video_transition.utils.evaluate_video import evaluate, extract_frames
from examples.video_transition.utils.protocol import METRICS


@pytest.fixture
def video(tmp_path):
    path=tmp_path/'synthetic.mp4'
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=4)
        stream.width=stream.height=32
        stream.pix_fmt='yuv420p'
        for i in range(80):
            frame=av.VideoFrame.from_image(Image.new('RGB',(32,32),(i*3,40,100)))
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    return path


@dataclass
class Usage:
    input_tokens:int=12
    output_tokens:int=20


class Client:
    model='fake-evaluator'
    base_url='http://not-a-real-endpoint.test/v1'

    def __init__(self): self.calls=[]

    def chat(self,messages,**kwargs):
        self.calls.append(messages)
        return {'content':json.dumps({'judgments':[1]*10,'sum':10})},Usage()


def settings(tmp_path):
    metrics={name:[f'Example criterion {i}' for i in range(10)] for name in METRICS}
    digest=hashlib.sha256(json.dumps(metrics,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    rubric=tmp_path/'rubrics.json'
    rubric.write_text(json.dumps({'metrics':metrics,'rubric_sha256':digest}))
    return {'rubrics_path':str(rubric),'evaluator_protocol':'fake-protocol-for-unit-test'}


def test_frame_extraction_uses_nearest_native_presentation_times(video,tmp_path):
    frames=extract_frames(video,[0.1,0.2,19.9],tmp_path/'frames')
    assert frames[0.1]['actual_time_s']==0
    assert frames[0.2]['actual_time_s']==0.25
    assert frames[19.9]['actual_time_s']==19.75
    with pytest.raises(ValueError,match='ends before'):
        extract_frames(video,[20.1],tmp_path/'too-late')


def test_evaluation_cache_is_bound_to_original_targets(video,tmp_path):
    task={'id':'fixture','events':[{'time_s':0,'prompt':'old'},{'time_s':10,'prompt':'new'}], 'duration_s':20}
    config=settings(tmp_path)
    client=Client()
    first=evaluate(task,video,config,tmp_path/'scores',client=client)
    assert first['score']==1 and len(client.calls)==8
    second=evaluate(task,video,config,tmp_path/'scores',client=client)
    assert second['metrics']==first['metrics'] and len(client.calls)==8
    task['events'][1]['prompt']='another user target'
    evaluate(task,video,config,tmp_path/'scores',client=client)
    assert len(client.calls)==16


def test_interrupted_target_window_is_not_silently_scored_as_standard(video,tmp_path):
    task={'id':'fixture','events':[{'time_s':0,'prompt':'old'},{'time_s':10,'prompt':'new'},
                                  {'time_s':12,'prompt':'latest'}], 'duration_s':20}
    client=Client()
    with pytest.raises(ValueError,match='interruption protocol'):
        evaluate(task,video,settings(tmp_path),tmp_path/'scores',client=client)
    assert not client.calls
