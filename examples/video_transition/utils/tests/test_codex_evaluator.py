import base64
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from examples.video_transition.utils.codex_evaluator import CodexEvaluatorClient, JUDGMENT_SCHEMA
from examples.video_transition.utils.remote import image_part
from examples.video_transition.utils.evaluate_video import evaluate
from examples.video_transition.utils.tests.test_video_evaluation import video, settings


@pytest.fixture
def executable(tmp_path):
    mode = tmp_path / 'mode'
    mode.write_text('success')
    capture = tmp_path / 'sent.json'
    program = tmp_path / 'codex'
    program.write_text(f'''#!{sys.executable}
import json,os,sys,time
from pathlib import Path
mode=Path({str(mode)!r}).read_text()
if sys.argv[1:]==['login','status']:
    print('Logged in using API key' if mode=='api-key' else 'Logged in using ChatGPT')
    sys.exit(0)
prompt=sys.stdin.read()
images=[sys.argv[i+1] for i,v in enumerate(sys.argv) if v=='--image']
Path({str(capture)!r}).write_text(json.dumps({{'images':images,'prompt':prompt,
    'api_keys':[k for k in os.environ if k.endswith('API_KEY')],
    'schema':json.loads(Path(sys.argv[sys.argv.index('--output-schema')+1]).read_text()),
    'arguments':sys.argv}}))
if mode=='timeout':time.sleep(20)
if mode=='failed':
    print(json.dumps({{'type':'turn.failed','error':{{'message':'quota'}}}}));sys.exit(0)
if mode=='tool':
    print(json.dumps({{'type':'item.completed','item':{{'type':'command_execution'}}}}))
print(json.dumps({{'type':'item.completed','item':{{'type':'reasoning','text':'private reasoning'}}}}))
reply={{'judgments':[1]*10,'sum':10}}
if mode=='invalid':reply['judgments']=[True]*10
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps(reply))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':120,'cached_input_tokens':20,'output_tokens':25}}}}))
''')
    program.chmod(0o755)
    return {'binary':str(program),'model':'fixture-vision','reasoning_effort':'max','timeout':3},mode,capture


def messages(tmp_path):
    frames=[]
    for i,color in enumerate(('red','blue')):
        path=tmp_path/f'frame-{i}.png'
        Image.new('RGB',(16,16),color).save(path)
        frames.extend([{'type':'text','text':f'Frame {i+1}, video time {9+i}s'},image_part(path)])
    return [{'role':'system','content':'Judge the supplied frames.'},{'role':'user','content':frames}]


def test_native_images_keep_order_and_schema_without_api_credentials(tmp_path,executable,monkeypatch):
    spec,_,capture=executable
    monkeypatch.setenv('OPENAI_API_KEY','must-not-use')
    monkeypatch.setenv('FIREWORKS_API_KEY','must-not-use')
    client=CodexEvaluatorClient(spec,tmp_path/'calls')
    reply,usage=client.chat(messages(tmp_path))
    assert json.loads(reply['content'])=={'judgments':[1]*10,'sum':10}
    assert usage.calls==1 and usage.input_tokens==120 and usage.cost_usd is None
    sent=json.loads(capture.read_text())
    assert not sent['api_keys'] and sent['schema']==JUDGMENT_SCHEMA
    assert [Image.open(p).getpixel((0,0)) for p in sent['images']]==[(255,0,0),(0,0,255)]
    assert sent['prompt'].index('Frame 1')<sent['prompt'].index('Attached image 1')<sent['prompt'].index('Frame 2')
    assert 'model_reasoning_effort="max"' in sent['arguments']
    assert 'private reasoning' not in next((tmp_path/'calls').glob('*/events.jsonl')).read_text()


@pytest.mark.parametrize('mode',['failed','tool','invalid'])
def test_invalid_or_tool_using_turn_is_never_a_valid_score(tmp_path,executable,mode):
    spec,control,_=executable;control.write_text(mode)
    client=CodexEvaluatorClient(spec,tmp_path/'calls')
    with pytest.raises((ValueError,RuntimeError)):
        client.chat(messages(tmp_path))
    assert client.total.calls==0


def test_codex_evaluator_cache_changes_with_reasoning_effort(video,tmp_path,executable):
    spec,_,_=executable
    config=settings(tmp_path)
    config['evaluator']={'transport':'codex-exec',**spec}
    config['evaluator_sampling']={'response_format':{'type':'json_schema','json_schema':{'schema':JUDGMENT_SCHEMA}}}
    task={'id':'fixture','events':[{'time_s':0,'prompt':'old'},{'time_s':10,'prompt':'new'}],'duration_s':20}
    first=evaluate(task,video,config,tmp_path/'scores')
    assert first['new_queries']==8
    second=evaluate(task,video,config,tmp_path/'scores')
    assert second['new_queries']==0
    config['evaluator']['reasoning_effort']='high'
    third=evaluate(task,video,config,tmp_path/'scores')
    assert third['new_queries']==8 and third['protocol_sha256']!=first['protocol_sha256']


def test_timeout_and_api_key_login_cannot_fall_back(tmp_path,executable):
    spec,control,_=executable
    control.write_text('api-key')
    with pytest.raises(RuntimeError,match='ChatGPT login'):
        CodexEvaluatorClient(spec,tmp_path/'calls').chat(messages(tmp_path))
    control.write_text('timeout')
    with pytest.raises(RuntimeError,match='exceeded'):
        CodexEvaluatorClient({**spec,'timeout':0.1},tmp_path/'calls').chat(messages(tmp_path))


def test_api_sampling_limits_are_not_silently_ignored(tmp_path,executable):
    spec,_,_=executable
    with pytest.raises(ValueError,match='native output limits'):
        CodexEvaluatorClient(spec,tmp_path/'calls').chat(messages(tmp_path),max_tokens=4096)
