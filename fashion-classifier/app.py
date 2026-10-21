"""Local Gradio upload-and-predict demo. Run: python app.py

Startup loads saved weights only. No dataset or training module is imported.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import json
import os

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import gradio as gr
import torch

from src.predict import FashionPredictor

ROOT = Path(__file__).resolve().parent
READY = '### Ready when you are\nUpload an image or choose a sample, then click **Predict category**.'
CSS = '''
.gradio-container {max-width: 1100px !important; margin: auto;}
#model-preview img, #upload-image img {image-rendering: pixelated; object-fit: contain; width: 100%; height: 100%;}
#intro {padding-bottom: 0.4rem;}
'''


@lru_cache(maxsize=1)
def get_predictor():
    """One CPU model per server process; no mutable inference state."""
    torch.set_num_threads(2)
    return FashionPredictor(ROOT / 'models/cnn/best.pt', device='cpu')


def handle_prediction(image, background):
    if image is None:
        return '### Add an image first\nUpload a clothing image or choose one of the samples below.', None, None
    try:
        result = get_predictor().predict(image, background=background, top_k=3)
    except (OSError, ValueError, TypeError) as exc:
        return f'### Check your image\n{exc}', None, None
    return f'### {result.label}\nPrediction from the processed image shown below.', result.scores, result.preview


def reset_result():
    return READY, None, None


def clear_inputs():
    return None, 'dark', READY, None, None


def sample_inputs():
    """Use pre-exported assets only; never download or open a dataset at startup."""
    manifest = ROOT / 'assets/examples/manifest.json'
    if not manifest.exists():
        return [], []
    rows, names = [], []
    for record in json.loads(manifest.read_text()):
        path = ROOT / record['path']
        if path.exists():
            rows.append([str(path), record['background']])
            names.append(record['display_name'])
    return rows, names


def create_demo():
    get_predictor()  # Fail clearly at startup if the saved checkpoint is missing/incompatible.
    with gr.Blocks(title='Clothing Classifier · AML Project', analytics_enabled=False,
                   delete_cache=(3600, 3600)) as demo:
        gr.Markdown('''# Clothing Classifier
Upload one clothing image to see its category.

Best with a single, upright item on a plain background. This model learned from tiny grayscale clothing images; everyday photos may be less reliable.''', elem_id='intro')
        with gr.Row():
            with gr.Column(scale=1):
                # filepath + image_mode=None preserve the uploaded alpha/EXIF and native pixels.
                image = gr.Image(type='filepath', image_mode=None, sources=['upload'], format='png',
                                 label='Your image', height=260, elem_id='upload-image')
                background = gr.Radio(choices=[('Dark background', 'dark'), ('Light background', 'light')],
                                      value='dark', label='Image background',
                                      info='Choose the background in your image. Transparent areas use this color.')
                with gr.Row():
                    predict_button = gr.Button('Predict category', variant='primary')
                    clear_button = gr.Button('Clear')
            with gr.Column(scale=1):
                result = gr.Markdown(READY)
                scores = gr.Label(num_top_classes=10, label='Scores across 10 categories')
                preview = gr.Image(type='pil', image_mode='L', format='png', label='Processed image · 28 × 28',
                                   height=190, interactive=False, elem_id='model-preview', buttons=['download'])
        gr.Markdown('Scores compare the model’s categories; they are not a guarantee of correctness. The preview shows the grayscale image the model receives. Backgrounds are not automatically removed.')
        predict_button.click(handle_prediction, inputs=[image, background], outputs=[result, scores, preview],
                             api_name='predict', concurrency_limit=1)
        clear_button.click(clear_inputs, outputs=[image, background, result, scores, preview],
                           api_name='clear', queue=False)
        image.change(reset_result, outputs=[result, scores, preview], queue=False, api_visibility='private')
        background.change(reset_result, outputs=[result, scores, preview], queue=False, api_visibility='private')
        examples, names = sample_inputs()
        if examples:
            gr.Examples(examples=examples, example_labels=names, inputs=[image, background],
                        examples_per_page=12, visible_columns=[0], cache_examples=False,
                        run_on_click=False, label='Try a training sample')
    demo.queue(default_concurrency_limit=1, max_size=16)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    demo = create_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=False,
                inbrowser=args.open_browser, show_error=False, ssr_mode=False,
                theme=gr.themes.Soft(primary_hue='teal', font=['system-ui', 'sans-serif']),
                css=CSS, footer_links=[])

if __name__ == '__main__':
    main()
