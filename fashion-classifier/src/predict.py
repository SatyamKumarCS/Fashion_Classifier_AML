"""Dataset-free CNN prediction for PIL images, paths, and NumPy arrays.

Example: python -m src.predict image.png --background dark --top-k 3
Float arrays must already be in [0,1]; uint8 arrays are interpreted as [0,255].
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image, ImageOps
import torch

from .model import SmallCNN, MODEL_CONFIG

INFERENCE_CONFIG = {
    'version': 1, 'output_shape': [1, 1, 28, 28], 'output_dtype': 'float32',
    'scaling': 'uint8 grayscale / 255.0',
    'array_inputs': 'uint8 [0,255] or finite float [0,1]; normalized floats rounded to uint8 before conversion',
    'orientation': 'apply EXIF orientation; no additional transpose, rotation or flip',
    'resize': 'bilinear contain to 28x28, preserve aspect ratio within integer rounding, minimum side 1 pixel',
    'padding': 'centered black padding; no cropping or extra benchmark margin',
    'background': 'explicit dark (default) or light; light is inverted after grayscale conversion',
    'alpha': 'composite on explicit background before grayscale conversion',
    'native_contract': '28x28 uint8 grayscale on dark background is pixel-identical',
    'blank_input': 'reject images with no grayscale contrast',
    'scores': 'softmax scores in checkpoint class order; not calibrated confidence',
    'upload_scope': 'Adaptation for uploads is not part of the frozen benchmark evaluation; real photo accuracy is unmeasured.',
}


@dataclass
class PreparedImage:
    pixels: np.ndarray
    tensor: torch.Tensor
    metadata: dict

    @property
    def preview(self):
        return Image.fromarray(self.pixels.copy())


@dataclass
class Prediction:
    label: str
    class_index: int
    scores: dict
    top_predictions: list
    prepared: PreparedImage

    @property
    def preview(self):
        return self.prepared.preview

    def as_dict(self):
        """JSON-ready result; preview pixels are available separately as .preview."""
        return {'label': self.label, 'class_index': self.class_index, 'scores': self.scores,
                'top_predictions': self.top_predictions, 'preprocessing': self.prepared.metadata}


def _orient(source):
    original_size = list(source.size)
    orientation = source.getexif().get(274, 1)
    oriented = ImageOps.exif_transpose(source)
    oriented.load()
    return oriented, {'original_size': original_size, 'oriented_size': list(oriented.size),
                      'exif_orientation_applied': orientation in range(2, 9)}


def _read_image(image):
    metadata = {}
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            converted, metadata = _orient(opened)
        metadata['source_type'] = 'file'
    elif isinstance(image, Image.Image):
        converted, metadata = _orient(image)
        metadata['source_type'] = 'PIL image'
    elif isinstance(image, np.ndarray):
        values = np.asarray(image)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]
        if values.ndim not in (2, 3) or (values.ndim == 3 and values.shape[-1] not in (3, 4)):
            raise ValueError('Expected HxW grayscale, HxWx1, HxWx3 RGB, or HxWx4 RGBA array.')
        if values.size == 0 or min(values.shape[:2]) == 0:
            raise ValueError('Image must contain pixels.')
        if values.dtype == np.uint8:
            metadata['array_range'] = 'uint8 [0,255]'
        elif np.issubdtype(values.dtype, np.floating):
            if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                raise ValueError('Float image arrays must be finite and scaled to [0,1]; use uint8 for [0,255].')
            values = np.rint(values.astype(np.float64) * 255).astype(np.uint8)
            metadata['array_range'] = 'float [0,1] rounded to uint8'
        else:
            raise TypeError('Image arrays must have uint8 or floating dtype; convert integer arrays explicitly.')
        converted, oriented = _orient(Image.fromarray(np.ascontiguousarray(values)))
        metadata.update(oriented)
        metadata['source_type'] = 'NumPy array'
    else:
        raise TypeError('Provide a PIL image, image-file path, or NumPy array.')
    if min(converted.size) <= 0:
        raise ValueError('Image dimensions must be positive.')
    if converted.mode not in ('1', 'L', 'LA', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr'):
        raise ValueError(f'Unsupported image mode {converted.mode}; convert to 8-bit L, RGB or RGBA first.')
    metadata['original_mode'] = converted.mode
    return converted, metadata


def prepare_image(image, *, background='dark'):
    """Prepare a single upright image without learned or test-tuned preprocessing.

    Native 28x28 grayscale images are unchanged with background='dark'.
    Other uploads use explicit background polarity, EXIF correction, grayscale,
    aspect-preserving resize and centered padding. No background removal is attempted.
    """
    if background not in ('dark', 'light'):
        raise ValueError("background must be 'dark' or 'light'.")
    source, metadata = _read_image(image)
    alpha = source.mode in ('RGBA', 'LA') or (source.mode == 'P' and 'transparency' in source.info)
    if alpha:
        fill = 0 if background == 'dark' else 255
        canvas = Image.new('RGBA', source.size, (fill, fill, fill, 255))
        source = Image.alpha_composite(canvas, source.convert('RGBA')).convert('RGB')
    grayscale = source.convert('L')
    if background == 'light':
        grayscale = ImageOps.invert(grayscale)
    low, high = grayscale.getextrema()
    if low == high:
        raise ValueError('Image has no grayscale contrast; upload a visible clothing item.')
    width, height = grayscale.size
    ratio = min(28 / width, 28 / height)
    resized_size = (max(1, min(28, round(width * ratio))), max(1, min(28, round(height * ratio))))
    resized = grayscale if grayscale.size == resized_size else grayscale.resize(resized_size, Image.Resampling.BILINEAR)
    left, top = (28 - resized_size[0]) // 2, (28 - resized_size[1]) // 2
    preview = Image.new('L', (28, 28), 0)
    preview.paste(resized, (left, top))
    pixels = np.array(preview, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy((pixels.astype(np.float32) / np.float32(255))[None, None].copy())
    metadata.update({'background': background, 'inverted': background == 'light',
                     'alpha_composited': alpha, 'resized_size': list(resized_size),
                     'padding_left_top_right_bottom': [left, top, 28 - resized_size[0] - left, 28 - resized_size[1] - top],
                     'output_shape': list(tensor.shape), 'pixel_range': [float(tensor.min()), float(tensor.max())],
                     'upload_adaptation': source.size != (28, 28) or background != 'dark' or metadata['original_mode'] != 'L' or metadata['exif_orientation_applied']})
    return PreparedImage(pixels, tensor, metadata)


class FashionPredictor:
    """Load once and reuse. This class imports no dataset loader or training module."""
    def __init__(self, checkpoint_path=None, *, device='cpu'):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else Path(__file__).resolve().parents[1] / 'models/cnn/best.pt'
        if device not in ('cpu', 'mps'):
            raise ValueError("device must be 'cpu' or 'mps'.")
        if device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS is unavailable; choose device=cpu.')
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint.get('model_config') != MODEL_CONFIG:
            raise ValueError('Checkpoint architecture does not match SmallCNN.')
        if checkpoint.get('preprocessing') != 'uint8 -> float32 / 255; (N,1,28,28); original orientation':
            raise ValueError('Checkpoint preprocessing contract is incompatible.')
        classes = checkpoint.get('class_names')
        if not isinstance(classes, list) or len(classes) != 10 or not all(isinstance(c, str) for c in classes) or len(set(classes)) != 10:
            raise ValueError('Checkpoint must contain ten unique class names in output order.')
        self.class_names = tuple(classes)
        self.device = torch.device(device)
        self.best_epoch = int(checkpoint['epoch'])
        self.model = SmallCNN().to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval()

    def predict(self, image, *, background='dark', top_k=3):
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= 10:
            raise ValueError('top_k must be an integer from 1 to 10.')
        prepared = prepare_image(image, background=background)
        with torch.inference_mode():
            probabilities = self.model(prepared.tensor.to(self.device)).softmax(1)[0].cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError('Model produced non-finite scores.')
        scores = {name: float(probabilities[i]) for i, name in enumerate(self.class_names)}
        order = np.argsort(-probabilities, kind='stable')
        top = [{'label': self.class_names[i], 'class_index': int(i), 'score': float(probabilities[i])}
               for i in order[:top_k]]
        index = int(order[0])
        return Prediction(self.class_names[index], index, scores, top, prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--background', choices=('dark', 'light'), default='dark')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--preview', type=Path, help='Optional path for the processed 28x28 PNG.')
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        result = FashionPredictor(args.checkpoint, device=args.device).predict(args.image, background=args.background, top_k=args.top_k)
        if args.preview:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            result.preview.save(args.preview, format='PNG')
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result.as_dict(), indent=2))

if __name__ == '__main__':
    main()
