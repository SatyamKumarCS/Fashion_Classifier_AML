"""Upload preprocessing contract and, when weights are installed, prediction."""
import json

import numpy as np
import pytest
from PIL import Image

from src.data import CLASS_NAMES
from src.predict import FashionPredictor, prepare_image


def test_native_input_is_pixel_identical():
    pixels = np.random.default_rng(0).integers(0, 256, (28, 28), dtype=np.uint8)
    prepared = prepare_image(Image.fromarray(pixels))
    assert np.array_equal(prepared.pixels, pixels)
    assert prepared.tensor.shape == (1, 1, 28, 28)
    assert np.allclose(prepared.tensor.numpy()[0, 0], pixels / 255)


def test_light_background_is_inverted_and_letterboxed():
    image = np.full((28, 56), 255, dtype=np.uint8)
    image[10:18, 20:36] = 0
    prepared = prepare_image(image, background='light')
    assert prepared.metadata['inverted'] and prepared.metadata['resized_size'] == [28, 14]
    assert prepared.pixels[:7].max() == 0 and prepared.pixels[-7:].max() == 0  # black padding
    assert prepared.pixels.max() == 255


def test_blank_and_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match='contrast'):
        prepare_image(np.zeros((28, 28), dtype=np.uint8))
    with pytest.raises(ValueError):
        prepare_image(np.full((28, 28), 1.5))
    with pytest.raises(ValueError):
        prepare_image(np.zeros((28, 28), dtype=np.uint8) + 1, background='grey')


@pytest.fixture(scope='module')
def predictor(root):
    checkpoint = root / 'models/cnn/best.pt'
    if not checkpoint.exists():
        pytest.skip('No installed CNN weights; run the pipeline first.')
    return FashionPredictor(checkpoint, device='cpu')


def test_predictions_are_valid_and_deterministic(predictor, root):
    for record in json.loads((root / 'assets/examples/manifest.json').read_text()):
        first = predictor.predict(root / record['path'], background=record['background'], top_k=3)
        second = predictor.predict(root / record['path'], background=record['background'], top_k=3)
        assert first.label in CLASS_NAMES and list(first.scores) == list(CLASS_NAMES)
        assert abs(sum(first.scores.values()) - 1) < 1e-5
        assert first.scores == second.scores
