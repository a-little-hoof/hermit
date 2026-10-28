from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from examples.video_transition.utils.controller import plan, validate_plan
from examples.video_transition.utils.evaluate_video import evaluate_metric, validate_judgments
from examples.video_transition.utils.streaming import PromptSchedule, offline_plan


@dataclass
class Usage:
    input_tokens: int = 12
    output_tokens: int = 5


class Client:
    model = "fake-vision-model"

    def __init__(self, response):
        self.response = response
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return {"content": self.response}, Usage()


def test_controller_sends_only_arrived_information(tmp_path):
    image=tmp_path/'frame.jpg'
    image.write_bytes(b'fake image for transport test')
    client=Client('{"steps":[{"prompt":"stands up","duration_chunks":1}]}')
    request={"request_id":1,"video_time_s":10,"previous_user_prompt":"sits","latest_user_prompt":"walks",
             "frames":[{"time_s":9,"path":str(image)}],"budget_chunks":4,"future_prompt":"secret future instruction"}
    result=plan(request,"read the visible state",{},client=client)
    assert result['request_id']==1
    assert 'secret future instruction' not in str(client.messages)
    request['frames'][0]['time_s']=11
    with pytest.raises(ValueError, match='future'):
        plan(request,'skill',{},client=client)


def test_invalid_durations_and_over_budget_plans_are_rejected():
    with pytest.raises(ValueError):
        validate_plan({'steps':[{'prompt':'x','duration_chunks':True}]},4)
    with pytest.raises(ValueError,match='budget'):
        validate_plan({'steps':[{'prompt':'x','duration_chunks':5}]},4)
    assert validate_plan({'steps':[]},4)==[]


def test_binary_judgments_require_consistent_sum():
    assert validate_judgments({'judgments':[1]*8+[0]*2,'sum':8})==[1]*8+[0]*2
    with pytest.raises(ValueError,match='sum'):
        validate_judgments({'judgments':[1]*10,'sum':9})
    with pytest.raises(ValueError,match='integer'):
        validate_judgments({'judgments':[True]*10,'sum':10})


def test_visual_metrics_omit_user_prompts_but_semantic_metrics_keep_originals(tmp_path):
    image=tmp_path/'frame.jpg'
    image.write_bytes(b'fake image for transport test')
    frames=[{'path':str(image),'actual_time_s':9.0}]
    client=Client('{"judgments":[1,1,1,1,1,1,1,1,0,0],"sum":8}')
    criteria=[f'check {i}' for i in range(10)]
    result=evaluate_metric('VBS',frames,'old user target','new user target',criteria,client,{})
    assert result['score']==0.8
    assert 'old user target' not in str(client.messages)
    result=evaluate_metric('NSE',frames,'old user target','new user target',criteria,client,{})
    assert 'old user target' in str(client.messages)
    assert 'new user target' in str(client.messages)


def test_new_instruction_cancels_old_suffix_and_rejects_stale_plan():
    schedule=PromptSchedule('old')
    version=schedule.request('target one')
    assert schedule.next_prompt()=='old'
    assert schedule.install(version,[{'prompt':'bridge one','duration_chunks':2},{'prompt':'obsolete suffix','duration_chunks':1}])
    assert schedule.next_prompt()=='bridge one'
    newer=schedule.request('target two')
    assert not schedule.install(version,[{'prompt':'stale','duration_chunks':1}])
    assert schedule.install(newer,[{'prompt':'bridge two','duration_chunks':1}])
    assert schedule.next_prompt()=='bridge two'
    assert schedule.next_prompt()=='target two'
    assert schedule.next_prompt()=='target two'


def test_offline_plan_finishes_before_generation_can_advance():
    schedule=PromptSchedule('sitting')
    version=schedule.request('celebrating')
    order=[]

    def query(root):
        assert schedule.pending and schedule.active=='sitting'
        order.append('planning finished')
        return {'steps':[{'prompt':'puts cards down','duration_chunks':1}]}
    response,accepted=offline_plan(schedule,version,query,None)
    assert accepted and response['steps']
    order.append(schedule.next_prompt())
    order.append(schedule.next_prompt())
    assert order==['planning finished','puts cards down','celebrating']
