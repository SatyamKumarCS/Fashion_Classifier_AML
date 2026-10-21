"""Behavioral checks for reusable preprocessing and offline inference.

Run: python scripts/verify_prediction.py
Uses saved training indices only; no official test images or results are opened.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import FashionPredictor, prepare_image
from src.data import CLASS_NAMES, load_training_data, scale_pixels


def main():
    torch.set_num_threads(2)
    images, labels = load_training_data(ROOT, download=False)
    subset = np.load(ROOT / 'data/processed/baseline_train_indices.npy', allow_pickle=False)
    indices = np.array([subset[np.flatnonzero(labels[subset] == label)[0]] for label in range(10)])
    source = images[indices[0]]
    passed = []
    checkpoint = ROOT / 'models/cnn/best.pt'
    initial_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Serving itself must neither contact the network nor load any dataset.
    with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected network access')), \
         patch('socket.create_connection', side_effect=AssertionError('Unexpected network access')), \
         patch('src.data.load_training_data', side_effect=AssertionError('Unexpected dataset load')):
        predictor = FashionPredictor()
        assert predictor.class_names == CLASS_NAMES and predictor.best_epoch == 9
        assert not predictor.model.training
        state_before = {k:v.clone() for k,v in predictor.model.state_dict().items()}
        for index in indices:
            prepared = prepare_image(images[index])
            assert np.array_equal(prepared.pixels, images[index])
            assert np.array_equal(prepared.tensor.numpy()[0,0], scale_pixels(images[index]))
            assert prepared.tensor.dtype == torch.float32
            assert list(prepared.tensor.shape) == [1,1,28,28]
            result = predictor.predict(images[index])
            with torch.inference_mode():
                direct = predictor.model(torch.from_numpy(scale_pixels(images[index])[None,None])).softmax(1)[0].numpy()
            assert np.allclose(list(result.scores.values()), direct, atol=1e-7)
            assert list(result.scores) == list(CLASS_NAMES)
            assert np.isclose(sum(result.scores.values()), 1, atol=1e-6)
            assert result.class_index == int(np.argmax(direct))
            assert result.label == CLASS_NAMES[result.class_index]
            assert len(result.top_predictions) == 3
            assert result.top_predictions[0]['label'] == result.label
            assert all(result.top_predictions[j]['score'] >= result.top_predictions[j+1]['score'] for j in range(2))
            json.dumps(result.as_dict())
        passed.append('native grayscale pixels, exact training scaling, direct-model score equivalence, all ten labels')
        rgb = np.repeat(source[...,None], 3, axis=2)
        rgba_dark = np.full((28,28,4),255,dtype=np.uint8)
        rgba_dark[...,3] = source
        rgba_light = np.zeros((28,28,4),dtype=np.uint8)
        rgba_light[...,3] = source
        variants = [prepare_image(rgb), prepare_image(source[...,None]),
                    prepare_image(source.astype(np.float32)/255),
                    prepare_image(rgba_dark), prepare_image(rgba_light,background='light'),
                    prepare_image(255-source,background='light')]
        for prepared in variants:
            assert np.array_equal(prepared.pixels,source)
        passed.append('RGB, single-channel array, normalized float, alpha compositing and explicit light-background equivalence')
        rectangle = np.repeat(source, 2, axis=1)
        prepared = prepare_image(rectangle)
        assert prepared.metadata['resized_size'] == [28,14]
        assert prepared.metadata['padding_left_top_right_bottom'] == [0,7,0,7]
        assert np.all(prepared.pixels[:7] == 0) and np.all(prepared.pixels[-7:] == 0)
        passed.append('aspect-preserving resize and centered black letterbox')
        oriented = Image.fromarray(rectangle)
        oriented.getexif()[274] = 6
        expected = prepare_image(np.rot90(rectangle,k=-1))
        corrected = prepare_image(oriented)
        assert corrected.metadata['exif_orientation_applied']
        assert corrected.metadata['oriented_size'] == [28,56]
        assert np.array_equal(corrected.pixels,expected.pixels)
        assert oriented.getexif()[274] == 6
        passed.append('EXIF orientation correction without mutating original image')
        extreme = np.tile(np.array([[0],[255]],dtype=np.uint8),(500,1))
        assert prepare_image(extreme).metadata['resized_size'] == [1,28]
        passed.append('extreme aspect ratio remains a valid nonzero image size')
        invalid = [None, np.zeros((0,28),dtype=np.uint8), np.zeros((28,28,2),dtype=np.uint8),
                   source.astype(np.int64), np.full((28,28),np.nan),
                   np.full((28,28),1.1), source.astype(np.float32),
                   np.zeros((28,28),dtype=np.uint8), np.full((28,28),255,dtype=np.uint8),
                   np.zeros((28,28,4),dtype=np.uint8)]
        for value in invalid:
            try:
                prepare_image(value)
            except (TypeError,ValueError):
                pass
            else:
                raise AssertionError(f'Invalid input accepted: {type(value)}')
        for top_k in (0,11,True,1.5):
            try:
                predictor.predict(source,top_k=top_k)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid top_k accepted')
        for background in ('auto',None):
            try:
                prepare_image(source,background=background)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid background accepted')
        passed.append('reject blank, transparent-empty, nonfinite, invalid range/shape/dtype and invalid options')
        with TemporaryDirectory() as temp:
            temp=Path(temp)
            png=temp/'sample.png'
            Image.fromarray(source).save(png)
            assert np.array_equal(prepare_image(png).pixels,source)
            jpeg=temp/'color.jpg'
            Image.fromarray(rgb).resize((56,56)).save(jpeg)
            assert predictor.predict(jpeg).preview.size == (28,28)
            corrupt=temp/'bad.png'
            corrupt.write_bytes(b'not an image')
            try:
                prepare_image(corrupt)
            except OSError:
                pass
            else:
                raise AssertionError('Corrupt image accepted')
            try:
                prepare_image(temp/'missing.png')
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('Missing image accepted')
        passed.append('PNG lossless identity, RGB JPEG upload, corrupt/missing file errors')
        assert all(torch.equal(v,state_before[k]) for k,v in predictor.model.state_dict().items())
        assert all(p.grad is None for p in predictor.model.parameters())
        passed.append('prediction leaves weights and BatchNorm buffers unchanged and creates no gradients')
    # Simulate a deployable package without data, reports, sklearn-dependent loaders or training modules.
    with TemporaryDirectory() as temp:
        isolated=Path(temp)
        (isolated/'src').mkdir()
        (isolated/'models/cnn').mkdir(parents=True)
        for name in ('__init__.py','model.py','predict.py'):
            shutil.copy2(ROOT/'src'/name,isolated/'src'/name)
        shutil.copy2(checkpoint,isolated/'models/cnn/best.pt')
        Image.fromarray(source).save(isolated/'input.png')
        # Block network before importing/initializing the prediction path.
        runner = """from unittest.mock import patch
with patch('urllib.request.urlopen', side_effect=AssertionError('network')), patch('socket.create_connection', side_effect=AssertionError('network')):
    from src.predict import main
    main()
"""
        env=dict(os.environ)
        env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,'-c',runner,'input.png','--preview','preview.png'],
                              cwd=isolated,env=env,capture_output=True,text=True,check=True,timeout=60)
        output=json.loads(result.stdout)
        assert len(output['scores']) == 10 and output['label'] in CLASS_NAMES
        assert Image.open(isolated/'preview.png').size == (28,28)
        assert not (isolated/'data').exists() and not (isolated/'reports').exists()
    passed.append('isolated CLI serving with no datasets/reports/training modules and blocked network')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==initial_hash
    print(json.dumps({'status':'passed','training_examples_used':indices.tolist(),
                      'check_groups':passed,'checkpoint_sha256':initial_hash,
                      'official_test_loaded':False},indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
