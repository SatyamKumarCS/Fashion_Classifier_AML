"""Local Gradio upload-and-predict demo for Clothing Classifier.

Run: python app.py
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

CATEGORY_ICONS = {
    'T-shirt/top': '👕',
    'Trouser': '👖',
    'Pullover': '🧥',
    'Dress': '👗',
    'Coat': '🧥',
    'Sandal': '👡',
    'Shirt': '👔',
    'Sneaker': '👟',
    'Bag': '👜',
    'Ankle boot': '👢',
}

def make_hero_card(title="Ready when you are", subtitle="Upload an image or pick a sample below",
                   confidence="—", icon="✨", status="ready"):
    return f"""<div class="hero-pred-banner {status}">
  <div class="banner-left">
    <div class="banner-icon">{icon}</div>
    <div class="banner-labels">
      <span class="banner-sub">Predicted Category</span>
      <span class="banner-title">{title}</span>
    </div>
  </div>
  <div class="banner-right">
    <span class="banner-conf-label">Confidence (model score)</span>
    <div class="banner-conf-score">{confidence}</div>
  </div>
</div>"""

READY = make_hero_card(
    title="Ready when you are",
    subtitle="Upload a clothing image and click Predict category",
    confidence="—",
    icon="✨",
    status="empty"
)

# Load external CSS if present, with inline fallback
CUSTOM_CSS_PATH = ROOT / 'assets/custom.css'
if CUSTOM_CSS_PATH.exists():
    CSS = CUSTOM_CSS_PATH.read_text()
else:
    CSS = '''
    .gradio-container {max-width: 1200px !important; margin: auto;}
    #model-preview img, #upload-image img {image-rendering: pixelated; object-fit: contain; width: 100%; height: 100%;}
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

    icon = CATEGORY_ICONS.get(result.label, '🧥')
    top_score = result.scores[result.label]
    hero_card = make_hero_card(
        title=result.label,
        subtitle="Prediction from the processed 28×28 image",
        confidence=f"{top_score:.2f}",
        icon=icon,
        status="active"
    )
    return hero_card, result.scores, result.preview


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

        # Top Navigation Bar
        gr.HTML("""
        <header class="top-navbar">
          <a class="nav-brand" href="#demo">
            <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"></path>
            </svg>
            Clothing Classifier
          </a>
          <nav class="nav-links">
            <a class="nav-link active" href="#demo">Home</a>
            <a class="nav-link" href="#about">About</a>
            <a class="nav-link" href="#dataset">Dataset</a>
            <a class="nav-link" href="#model">Model</a>
            <a class="nav-link" href="#demo">Try Demo</a>
          </nav>
          <div class="nav-actions">
            <span style="font-size: 1.1rem; cursor: pointer;" title="Toggle theme">🌙</span>
            <a class="github-btn" href="https://github.com/krishnaverma09/Fashion_Classifier_AML" target="_blank" rel="noopener">
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
              GitHub
            </a>
          </div>
        </header>
        """)

        # Main Two-Column Layout
        with gr.Row():
            # LEFT COLUMN: Hero & Input Controls
            with gr.Column(scale=5):
                # Hero Introduction
                gr.HTML("""
                <div class="hero-section">
                  <div class="hero-badge">• AI • Computer Vision</div>
                  <h1 class="hero-title">Clothing <span>Classifier</span></h1>
                  <p class="hero-subtitle">Upload an image of a clothing item and let our model predict its category using deep learning.</p>
                  <div class="feature-pills">
                    <div class="feature-pill">
                      <div class="pill-icon blue">✦</div>
                      <div class="pill-text">
                        <span class="pill-title">10 Categories</span>
                        <span class="pill-desc">Fashion-MNIST classes</span>
                      </div>
                    </div>
                    <div class="feature-pill">
                      <div class="pill-icon purple">⚡</div>
                      <div class="pill-text">
                        <span class="pill-title">Fast & Simple</span>
                        <span class="pill-desc">Instant prediction</span>
                      </div>
                    </div>
                    <div class="feature-pill">
                      <div class="pill-icon green">🛡️</div>
                      <div class="pill-text">
                        <span class="pill-title">Powered by CNN</span>
                        <span class="pill-desc">Trained on 60,000+ images</span>
                      </div>
                    </div>
                  </div>
                </div>
                """)

                # Card: Upload Image
                with gr.Group(elem_classes=["glass-panel"]):
                    gr.HTML("""
                    <h3 class="section-title">Upload an image</h3>
                    <p class="section-desc">Choose a clothing image (preferably a single, upright item).</p>
                    """)
                    # Preserves type='filepath', image_mode=None, label='Your image' for verify_demo
                    image = gr.Image(type='filepath', image_mode=None, sources=['upload'], format='png',
                                     label='Your image', height=240, elem_id='upload-image')

                # Card: Background Style Toggle
                with gr.Group(elem_classes=["glass-panel"]):
                    gr.HTML("""
                    <h3 class="section-title">Image background</h3>
                    <p class="section-desc">Choose the background style in your image. Transparent areas use this color.</p>
                    """)
                    background = gr.Radio(choices=[('🌙 Dark background', 'dark'), ('☀️ Light background', 'light')],
                                          value='dark', label='Background Mode', show_label=False,
                                          elem_id='bg-radio')

                # Action Buttons Row
                with gr.Row():
                    predict_button = gr.Button('✨ Predict Category →', variant='primary', elem_classes=["predict-btn"], scale=3)
                    clear_button = gr.Button('🔄 Clear', elem_classes=["clear-btn"], scale=1)

            # RIGHT COLUMN: Results & Performance Visualizations
            with gr.Column(scale=6):
                with gr.Group(elem_classes=["glass-panel"]):
                    # Section Header with Status Pill
                    gr.HTML("""
                    <div class="results-header-row">
                      <h3 class="section-title" style="margin-bottom: 0;">Prediction Results</h3>
                      <span class="status-pill"><span class="status-dot"></span> Ready</span>
                    </div>
                    <p class="section-desc">Scores show how close the image is to each category. Higher means more likely.</p>
                    """)

                    # Dual Image Previews Row (side-by-side)
                    with gr.Row(elem_classes=["dual-preview-grid"]):
                        preview = gr.Image(type='pil', image_mode='L', format='png', label='Processed image · 28 × 28',
                                           height=180, interactive=False, elem_id='model-preview', buttons=['download'])

                    # Prominent Hero Prediction Banner
                    result = gr.HTML(READY, elem_id="hero-result-card")

                    # Scores Across 10 Categories
                    gr.HTML('<h4 style="font-weight: 700; margin: 16px 0 8px; font-size: 0.95rem; color: #1e293b;">Scores across 10 categories</h4>')
                    scores = gr.Label(num_top_classes=10, show_label=False, elem_id='scores-label')

                    # Interactive Sample Picker
                    gr.HTML("""
                    <div style="margin-top: 18px; border-top: 1px solid #e2e8f0; padding-top: 14px;">
                      <h4 style="font-weight: 700; font-size: 0.9rem; margin-bottom: 4px; color: #1e293b;">Try a sample image</h4>
                      <p style="font-size: 0.8rem; color: #64748b; margin-bottom: 8px;">Click on any sample to test the model.</p>
                    </div>
                    """)

                    examples, names = sample_inputs()
                    if examples:
                        gr.Examples(examples=examples, example_labels=names, inputs=[image, background],
                                    examples_per_page=11, visible_columns=[0], cache_examples=False,
                                    run_on_click=False, label='')

        # Interactivity Wire-up
        predict_button.click(handle_prediction, inputs=[image, background], outputs=[result, scores, preview],
                             api_name='predict', concurrency_limit=1)
        clear_button.click(clear_inputs, outputs=[image, background, result, scores, preview],
                           api_name='clear', queue=False)
        image.change(reset_result, outputs=[result, scores, preview], queue=False, api_visibility='private')
        background.change(reset_result, outputs=[result, scores, preview], queue=False, api_visibility='private')

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
                theme=gr.themes.Soft(primary_hue='indigo', font=['system-ui', 'sans-serif']),
                css=CSS, footer_links=[])

if __name__ == '__main__':
    main()
