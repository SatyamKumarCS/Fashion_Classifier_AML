"""Verify Gradio callbacks, HTTP upload/prediction and dataset-free startup.

Start python app.py first, then run: python scripts/verify_demo.py
Uses exported training samples only, never training/test dataset loaders.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import httpx
import numpy as np
from PIL import Image
from gradio_client import Client, handle_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import app
from src.predict import FashionPredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:7860')
    args=parser.parse_args()
    checks=[]
    examples,names=app.sample_inputs()
    assert len(examples)==len(names)==11
    image_path=ROOT/'assets/examples/00.png'
    app.get_predictor.cache_clear()
    with patch.object(app,'FashionPredictor',wraps=FashionPredictor) as constructor:
        demo=app.create_demo()
        for path,background in examples:
            heading,scores,preview=app.handle_prediction(path,background)
            expected=app.get_predictor().predict(path,background=background)
            assert expected.label in heading
            assert list(scores)==list(expected.scores)
            assert np.allclose(list(scores.values()),list(expected.scores.values()),atol=1e-7)
            assert np.array_equal(np.asarray(preview),np.asarray(expected.preview))
        assert constructor.call_count==1
        assert len([c for c in demo.config['components'] if c['type']=='label'])==1
    checks.append('11 training samples: callback scores/preview match reusable predictor; one cached model')
    heading,scores,preview=app.handle_prediction(None,'dark')
    assert 'Add an image' in heading and scores is None and preview is None
    heading,scores,preview=app.handle_prediction(Image.new('L',(28,28),0),'dark')
    assert 'no grayscale contrast' in heading and scores is None and preview is None
    heading,scores,preview=app.handle_prediction(str(image_path),'auto')
    assert 'background' in heading and scores is None and preview is None
    assert app.clear_inputs()==(None,'dark',app.READY,None,None)
    assert app.reset_result()==(app.READY,None,None)
    checks.append('missing/blank input and invalid background clear results; reset/clear restore ready state')
    with httpx.Client(trust_env=False,timeout=30) as http:
        response=http.get(args.url)
        assert response.status_code==200 and 'Clothing Classifier' in response.text
        config=http.get(args.url+'/config').json()
        image_component=next(c for c in config['components'] if c['type']=='image' and c['props'].get('label')=='Your image')
        assert image_component['props'].get('image_mode') is None and image_component['props']['type']=='filepath'
        assert config['enable_queue']
    checks.append('live localhost page/config, queued prediction and original-format upload preprocessing')
    with TemporaryDirectory() as temp:
        temp=Path(temp)
        client=Client(args.url,verbose=False,analytics_enabled=False,
                      httpx_kwargs={'trust_env':False},download_files=temp/'downloads')
        def verify_http(path,background):
            actual=client.predict(handle_file(str(path)),background,api_name='/predict')
            heading,scored,preview_path=actual
            expected=app.get_predictor().predict(path,background=background)
            assert expected.label in heading and scored['label']==expected.label
            api_scores={r['label']:r['confidence'] for r in scored['confidences']}
            assert set(api_scores)==set(expected.scores)
            assert all(np.isclose(api_scores[k],v,atol=1e-6) for k,v in expected.scores.items())
            with Image.open(preview_path) as preview:
                assert preview.size==(28,28)
                assert np.array_equal(np.asarray(preview.convert('L')),expected.prepared.pixels)
            return api_scores
        native=verify_http(image_path,'dark')
        inverted=verify_http(ROOT/'assets/examples/light-background.png','light')
        assert all(np.isclose(native[k],inverted[k],atol=1e-6) for k in native)
        with Image.open(image_path) as original:
            raw=np.array(original.convert('L'))
        rgba=np.full((28,28,4),255,dtype=np.uint8)
        rgba[...,3]=raw
        alpha_path=temp/'transparent.png'
        Image.fromarray(rgba).save(alpha_path)
        alpha=verify_http(alpha_path,'dark')
        assert all(np.isclose(native[k],alpha[k],atol=1e-6) for k in native)
        oriented=temp/'oriented.jpg'
        picture=Image.fromarray(np.repeat(raw,2,axis=1)).convert('RGB')
        exif=picture.getexif()
        exif[274]=6
        picture.save(oriented,exif=exif)
        verify_http(oriented,'dark')
        blank=temp/'blank.png'
        Image.new('L',(28,28),0).save(blank)
        bad=client.predict(handle_file(str(blank)),'dark',api_name='/predict')
        assert 'no grayscale contrast' in bad[0] and bad[1] == {} and bad[2] is None
        missing=client.predict(None,'dark',api_name='/predict')
        assert 'Add an image' in missing[0] and missing[1] == {} and missing[2] is None
        cleared=client.predict(api_name='/clear')
        assert cleared[0] is None and cleared[1]=='dark' and cleared[2]==app.READY
        assert cleared[3] == {} and cleared[4] is None
    checks.append('HTTP uploads: native PNG, light background, RGBA alpha and EXIF JPEG match direct inference')
    checks.append('HTTP missing/blank input and clear endpoint return empty results without stale prediction')
    with TemporaryDirectory() as temp:
        isolated=Path(temp)
        (isolated/'src').mkdir()
        (isolated/'models/cnn').mkdir(parents=True)
        shutil.copy2(ROOT/'app.py',isolated/'app.py')
        for name in ['__init__.py','model.py','predict.py']:
            shutil.copy2(ROOT/'src'/name,isolated/'src'/name)
        shutil.copy2(ROOT/'models/cnn/best.pt',isolated/'models/cnn/best.pt')
        shutil.copytree(ROOT/'assets',isolated/'assets')
        runner="""from unittest.mock import patch
import sys
with patch('urllib.request.urlopen',side_effect=AssertionError('network')), patch('socket.create_connection',side_effect=AssertionError('network')):
    import app
    demo=app.create_demo()
    heading,scores,preview=app.handle_prediction('assets/examples/00.png','dark')
    assert len(scores)==10 and preview.size==(28,28)
    assert not {'src.data','src.train','src.baselines','src.evaluate'} & set(sys.modules)
print('Isolated app construction and callback passed')
"""
        env=dict(os.environ)
        env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,'-c',runner],cwd=isolated,env=env,
            capture_output=True,text=True,check=True,timeout=60)
        assert 'Isolated app construction and callback passed' in result.stdout
        assert not (isolated/'data').exists() and not (isolated/'reports').exists()
    checks.append('isolated app startup/callback with network blocked and no dataset/training/evaluation modules')
    protected=json.loads((ROOT/'reports/demo/protected-sha256.json').read_text())
    for relative,expected in protected.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    checks.append('all frozen weights, split files, inference contracts and prior evaluation artifacts unchanged')
    print(json.dumps({'status':'passed','url':args.url,'sample_count':len(examples),
                      'check_groups':checks,'official_test_loaded':False,
                      'gradio_version':app.gr.__version__},indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
