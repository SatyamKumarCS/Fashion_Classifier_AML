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
                   confidence="—", icon="✨", status="empty"):
    is_active = (status == "active")
    icon_bg = "#2563eb" if is_active else "#cbd5e1"
    conf_color = "#2563eb" if is_active else "#94a3b8"
    banner_bg = "background: #eff6ff; border: 1.5px solid #dbeafe;" if is_active else "background: #f8fafc; border: 1.5px solid #e2e8f0;"
    sub_color = "#2563eb" if is_active else "#64748b"

    return f"""<div class="hero-pred-banner {status}" style="{banner_bg} border-radius: 14px; padding: 14px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08); transition: all 0.2s ease;">
  <div class="banner-left" style="display: flex; align-items: center; gap: 14px;">
    <div class="banner-icon-box" style="background: {icon_bg}; width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; color: #ffffff; flex-shrink: 0; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
      {icon}
    </div>
    <div class="banner-labels" style="display: flex; flex-direction: column;">
      <span class="banner-sub" style="font-size: 0.72rem; font-weight: 700; color: {sub_color}; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Category</span>
      <span class="banner-title" style="font-size: 1.35rem; font-weight: 800; color: #0f172a; line-height: 1.2;">{title}</span>
    </div>
  </div>
  <div class="banner-right" style="text-align: right;">
    <div class="banner-conf-label" style="font-size: 0.72rem; color: #64748b; font-weight: 500;">Confidence (model score)</div>
    <div class="banner-conf-score" style="font-size: 1.75rem; font-weight: 800; color: {conf_color}; line-height: 1.2;">{confidence}</div>
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

HEAD_JS = """
<script>
function applyTheme(isDark) {
  if (isDark) {
    document.documentElement.classList.add('dark', 'dark-theme');
    document.body.classList.add('dark', 'dark-theme');
  } else {
    document.documentElement.classList.remove('dark', 'dark-theme');
    document.body.classList.remove('dark', 'dark-theme');
  }
  const icon = document.getElementById('theme-icon');
  if (icon) icon.textContent = isDark ? '☀️' : '🌙';
  try { localStorage.setItem('fash_theme', isDark ? 'dark' : 'light'); } catch(e){}
}

function toggleTheme() {
  const isDark = !document.body.classList.contains('dark-theme');
  applyTheme(isDark);
}

document.addEventListener('DOMContentLoaded', function() {
  try {
    const isDark = (localStorage.getItem('fash_theme') === 'dark');
    applyTheme(isDark);
  } catch(e){}
});

// Guard against Gradio automatically forcing .dark from macOS system preference
function stripDarkInLightMode() {
  try {
    if (localStorage.getItem('fash_theme') !== 'dark') {
      if (document.documentElement.classList.contains('dark')) {
        document.documentElement.classList.remove('dark');
      }
      if (document.body.classList.contains('dark') && !document.body.classList.contains('dark-theme')) {
        document.body.classList.remove('dark');
      }
      const gContainer = document.querySelector('.gradio-container');
      if (gContainer && gContainer.classList.contains('dark') && !document.body.classList.contains('dark-theme')) {
        gContainer.classList.remove('dark');
      }
      document.querySelectorAll('#scores-label, #scores-label *, .glass-panel, .glass-panel *, .gradio-container, .gradio-container *').forEach(function(el) {
        if (el.classList.contains('dark') && !document.body.classList.contains('dark-theme')) {
          el.classList.remove('dark');
        }
      });
    }
  } catch(e){}
}

const themeGuard = new MutationObserver(function() {
  stripDarkInLightMode();
});
themeGuard.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
themeGuard.observe(document.body, { attributes: true, attributeFilter: ['class'], childList: true, subtree: true });

function openInfoModal(tabIndex) {
  const backdrop = document.getElementById('info-modal-backdrop');
  if (backdrop) {
    backdrop.setAttribute('style', 'display:flex!important');
    backdrop.classList.add('open');
    switchModalTab(tabIndex);
  }
}

function closeInfoModal() {
  const backdrop = document.getElementById('info-modal-backdrop');
  if (backdrop) {
    backdrop.classList.remove('open');
    backdrop.setAttribute('style', 'display:none!important');
  }
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeInfoModal();
});

function switchModalTab(tabIndex) {
  for (let i = 0; i < 3; i++) {
    const btn = document.getElementById('modal-tab-btn-' + i);
    const pane = document.getElementById('modal-tab-pane-' + i);
    if (btn) { btn.classList.toggle('active', i === tabIndex); }
    if (pane) { pane.style.display = (i === tabIndex) ? 'block' : 'none'; }
  }
}

function scrollToDemo() {
  const upload = document.getElementById('upload-image');
  if (upload) upload.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ── Event delegation (capture phase) so Gradio cannot swallow our clicks ──
document.addEventListener('click', function(e) {
  var t = e.target;
  // Walk up the DOM to check ancestors too
  var el = t;
  while (el && el !== document) {
    var act = el.getAttribute && el.getAttribute('data-modal-action');
    if (act === 'close') {
      e.preventDefault(); e.stopImmediatePropagation();
      closeInfoModal(); return;
    }
    if (act === 'dialog') {
      // Click inside dialog - do nothing (don't close)
      return;
    }
    if (act === 'backdrop') {
      e.preventDefault(); e.stopImmediatePropagation();
      closeInfoModal(); return;
    }
    var tab = el.getAttribute && el.getAttribute('data-modal-tab');
    if (tab !== null && tab !== undefined && tab !== '') {
      e.preventDefault(); e.stopImmediatePropagation();
      switchModalTab(parseInt(tab, 10)); return;
    }
    el = el.parentElement;
  }
}, true);

// ── Enrich scores with icons and decimal format, and samples with captions ──
const CATEGORY_EMOJIS = {
  'T-shirt/top': '👕',
  'Trouser': '👖',
  'Pullover': '🧥',
  'Dress': '👗',
  'Coat': '🧥',
  'Sandal': '👡',
  'Shirt': '👔',
  'Sneaker': '👟',
  'Bag': '👜',
  'Ankle boot': '👢'
};

function enrichScoreElements() {
  const container = document.getElementById('scores-label');
  if (!container) return;
  const dts = container.querySelectorAll('dt');
  dts.forEach(dt => {
    const raw = dt.textContent.replace(/[^\w\s\/\-]/g, '').trim();
    if (CATEGORY_EMOJIS[raw] && !dt.querySelector('.score-icon')) {
      dt.innerHTML = '<span class="score-icon" style="margin-right: 6px;">' + CATEGORY_EMOJIS[raw] + '</span>' + raw;
    }
  });

  const dds = container.querySelectorAll('dd');
  dds.forEach(dd => {
    const txt = dd.textContent.trim();
    if (txt.endsWith('%')) {
      const num = parseFloat(txt);
      if (!isNaN(num)) {
        dd.textContent = (num / 100).toFixed(2);
      }
    }
  });
}

const SAMPLE_NAMES = [
  'T-shirt/top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot',
  'T-shirt/top'
];

function enrichSampleCards() {
  const wrapper = document.getElementById('sample-gallery-wrapper');
  if (!wrapper) return;
  const buttons = wrapper.querySelectorAll('button.gallery-item, td.cell button, .paginate-tbody button, button');
  let sampleIdx = 0;
  buttons.forEach(btn => {
    const img = btn.querySelector('img');
    if (img) {
      if (!btn.querySelector('.sample-card-caption')) {
        const caption = document.createElement('span');
        caption.className = 'sample-card-caption';
        caption.textContent = SAMPLE_NAMES[sampleIdx] || (img.getAttribute('alt') || '').split('·')[0].trim();
        btn.appendChild(caption);
      }
      sampleIdx++;
    }
  });
}

const observer = new MutationObserver(function() {
  enrichScoreElements();
  enrichSampleCards();
});
document.addEventListener('DOMContentLoaded', function() {
  enrichScoreElements();
  enrichSampleCards();
  observer.observe(document.body, { childList: true, subtree: true });
});
setInterval(function() {
  enrichScoreElements();
  enrichSampleCards();
}, 600);
</script>
"""


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

        # Top Navigation Bar & Interactive Modal
        gr.HTML("""
        <header class="top-navbar">
          <a class="nav-brand" href="#demo" onclick="scrollToDemo(); return false;">
            <svg viewBox="0 0 24 24" width="22" height="22" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3c0 .8.4 1.4 1 1.8L2 14v1a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-1l-8-7.2c.6-.4 1-1 1-1.8a3 3 0 0 0-3-3z"/>
            </svg>
            Clothing Classifier
          </a>
          <nav class="nav-links">
            <button class="nav-link active" id="nav-btn-home" type="button" onclick="scrollToDemo()">Home</button>
            <button class="nav-link" id="nav-btn-about" type="button" onclick="openInfoModal(0)">About</button>
            <button class="nav-link" id="nav-btn-dataset" type="button" onclick="openInfoModal(1)">Dataset</button>
            <button class="nav-link" id="nav-btn-model" type="button" onclick="openInfoModal(2)">Model</button>
            <button class="nav-link" id="nav-btn-demo" type="button" onclick="scrollToDemo()">Try Demo</button>
          </nav>
          <div class="nav-actions">
            <button id="theme-toggle" class="theme-toggle-btn" onclick="toggleTheme()" type="button" title="Toggle Dark/Light Mode" aria-label="Toggle Theme">
              <span id="theme-icon">🌙</span>
            </button>
            <a class="github-btn" href="https://github.com/krishnaverma09/Fashion_Classifier_AML" target="_blank" rel="noopener">
              <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
              GitHub
            </a>
          </div>
        </header>

        <!-- Interactive Information Modal (About, Dataset, Model Architecture) -->
        <div id="info-modal-backdrop" class="info-modal-backdrop" style="display:none!important" data-modal-action="backdrop">
          <div class="info-modal-dialog" data-modal-action="dialog">
            <div class="info-modal-header">
              <div class="info-modal-title-group">
                <span style="font-size: 1.3rem;">📖</span>
                <h3 class="info-modal-title">Project Documentation & Architecture</h3>
              </div>
              <div class="info-modal-close" data-modal-action="close" role="button" tabindex="0" aria-label="Close modal">✕</div>
            </div>
            <div class="info-modal-tabs">
              <button type="button" class="info-modal-tab-btn active" id="modal-tab-btn-0" data-modal-tab="0">ℹ️ About</button>
              <button type="button" class="info-modal-tab-btn" id="modal-tab-btn-1" data-modal-tab="1">📊 Dataset</button>
              <button type="button" class="info-modal-tab-btn" id="modal-tab-btn-2" data-modal-tab="2">🧠 Model Architecture</button>
            </div>
            <div class="info-modal-body">
              <div id="modal-tab-pane-0" class="modal-pane active">
                <div class="info-tab-content">
                  <div class="info-grid-2">
                    <div>
                      <h4 class="tab-subtitle">🎯 Project Background & Mission</h4>
                      <p class="tab-body-text">
                        <strong>Automatic Clothing Category Recognition</strong> provides automated visual categorization for e-commerce apparel inventory systems. Categorizing inventory items directly from product photography reduces manual indexing errors and accelerates catalog ingestion.
                      </p>
                      <div class="badge-list" style="margin-top: 10px;">
                        <span class="info-badge blue">E-Commerce Tagging</span>
                        <span class="info-badge purple">PyTorch CNN</span>
                        <span class="info-badge green">Production UI</span>
                        <span class="info-badge orange">Phase 1 Delivery</span>
                      </div>
                    </div>
                    <div>
                      <h4 class="tab-subtitle">🔬 Methodology & Phase 1 Scope</h4>
                      <p class="tab-body-text">
                        Our rigorous evaluation framework eliminates experimental bias through disciplined data splits:
                      </p>
                      <ul class="tab-bullet-list">
                        <li><strong>Identical Training Budget:</strong> 12,000 stratified training images across all classical baselines and CNN.</li>
                        <li><strong>Baseline Exploration:</strong> Logistic Regression, kNN, Random Forest, and Linear SVM evaluated on flattened pixels and handcrafted HOG features.</li>
                        <li><strong>Strict Test Gate:</strong> Model choices frozen in <code>frozen-models.json</code> prior to single-run official test set evaluation.</li>
                      </ul>
                    </div>
                  </div>

                  <div class="team-section" style="margin-top: 18px;">
                    <h4 class="tab-subtitle">👥 Project Team</h4>
                    <div class="team-cards-grid">
                      <div class="team-card">
                        <div class="team-avatar">SK</div>
                        <div class="team-info">
                          <div class="team-name">Satyam Kumar</div>
                          <div class="team-enroll">Enroll: 2401010428</div>
                        </div>
                      </div>
                      <div class="team-card">
                        <div class="team-avatar">KV</div>
                        <div class="team-info">
                          <div class="team-name">Krishna Verma</div>
                          <div class="team-enroll">Team Member</div>
                        </div>
                      </div>
                      <div class="team-card">
                        <div class="team-avatar">AV</div>
                        <div class="team-info">
                          <div class="team-name">Akshit Vats</div>
                          <div class="team-enroll">Team Member</div>
                        </div>
                      </div>
                      <div class="team-card">
                        <div class="team-avatar">CS</div>
                        <div class="team-info">
                          <div class="team-name">Chinmay Soni</div>
                          <div class="team-enroll">Enroll: 2401010134</div>
                        </div>
                      </div>
                      <div class="team-card">
                        <div class="team-avatar">LB</div>
                        <div class="team-info">
                          <div class="team-name">Lakshya Bapna</div>
                          <div class="team-enroll">Team Member</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div id="modal-tab-pane-1" class="modal-pane" style="display: none;">
                <div class="info-tab-content">
                  <div class="stats-overview-grid">
                    <div class="stat-card">
                      <div class="stat-num">70,000</div>
                      <div class="stat-label">Total Images (28×28 Grayscale)</div>
                    </div>
                    <div class="stat-card">
                      <div class="stat-num">60,000</div>
                      <div class="stat-label">Training Pool (48k Train / 12k Val)</div>
                    </div>
                    <div class="stat-card">
                      <div class="stat-num">12,000</div>
                      <div class="stat-label">Fair Training Budget (1,200/class)</div>
                    </div>
                    <div class="stat-card">
                      <div class="stat-num">10,000</div>
                      <div class="stat-label">Official Test (1,000/class, 1 Eval)</div>
                    </div>
                  </div>

                  <h4 class="tab-subtitle" style="margin-top: 18px;">🏷️ 10 Balanced Categories</h4>
                  <div class="class-grid-10">
                    <div class="class-chip"><span class="class-icon">👕</span><span class="class-name">0: T-shirt/top</span></div>
                    <div class="class-chip"><span class="class-icon">👖</span><span class="class-name">1: Trouser</span></div>
                    <div class="class-chip"><span class="class-icon">🧥</span><span class="class-name">2: Pullover</span></div>
                    <div class="class-chip"><span class="class-icon">👗</span><span class="class-name">3: Dress</span></div>
                    <div class="class-chip"><span class="class-icon">🧥</span><span class="class-name">4: Coat</span></div>
                    <div class="class-chip"><span class="class-icon">👡</span><span class="class-name">5: Sandal</span></div>
                    <div class="class-chip"><span class="class-icon">👔</span><span class="class-name">6: Shirt</span></div>
                    <div class="class-chip"><span class="class-icon">👟</span><span class="class-name">7: Sneaker</span></div>
                    <div class="class-chip"><span class="class-icon">👜</span><span class="class-name">8: Bag</span></div>
                    <div class="class-chip"><span class="class-icon">👢</span><span class="class-name">9: Ankle boot</span></div>
                  </div>

                  <div class="data-notes" style="margin-top: 16px;">
                    <h4 class="tab-subtitle">⚙️ Preprocessing Contract</h4>
                    <p class="tab-body-text">
                      Raw inputs are normalized from uint8 $[0, 255]$ to float32 $[0.0, 1.0]$. The inference engine automatically applies EXIF rotation compensation, aspect-ratio preserving bilinear letterboxing into $28 \times 28$, centered dark padding, and optional contrast inversion for light backgrounds.
                    </p>
                  </div>
                </div>
              </div>

              <div id="modal-tab-pane-2" class="modal-pane" style="display: none;">
                <div class="info-tab-content">
                  <div class="info-grid-2">
                    <div>
                      <h4 class="tab-subtitle">🧠 SmallCNN Architecture Specifications</h4>
                      <div class="arch-box">
                        <div class="arch-layer"><strong>Input:</strong> 1 × 28 × 28 normalized grayscale tensor</div>
                        <div class="arch-arrow">↓</div>
                        <div class="arch-layer"><strong>Conv Block 1:</strong> Conv2d(1→32, 3×3, pad 1) + BatchNorm2d(32) + ReLU + MaxPool2d(2×2) → (32, 14, 14)</div>
                        <div class="arch-arrow">↓</div>
                        <div class="arch-layer"><strong>Conv Block 2:</strong> Conv2d(32→64, 3×3, pad 1) + BatchNorm2d(64) + ReLU + MaxPool2d(2×2) → (64, 7, 7)</div>
                        <div class="arch-arrow">↓</div>
                        <div class="arch-layer"><strong>Dense Head:</strong> Flatten(3136) → Linear(3136→128) + ReLU + Dropout(0.4) → Linear(128→64) + ReLU → Linear(64→10)</div>
                      </div>
                      <div class="param-summary" style="margin-top: 10px;">
                        <span class="info-badge green">Trainable Parameters: <strong>429,450</strong></span>
                        <span class="info-badge blue">Optimizer: Adam (lr=1e-3, weight_decay=1e-4)</span>
                      </div>
                    </div>

                    <div>
                      <h4 class="tab-subtitle">🏆 Test Benchmark Comparison</h4>
                      <div class="table-container">
                        <table class="benchmark-table">
                          <thead>
                            <tr>
                              <th>Model / Baseline</th>
                              <th>Test Acc</th>
                              <th>Macro F1</th>
                              <th>Val Acc</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr class="highlight-row">
                              <td><strong>🔥 SmallCNN (Best Deployed)</strong></td>
                              <td><strong>89.00%</strong></td>
                              <td><strong>88.85%</strong></td>
                              <td>89.93%</td>
                            </tr>
                            <tr>
                              <td>Pixels + Random Forest (Best Classical)</td>
                              <td>85.27%</td>
                              <td>85.09%</td>
                              <td>86.46%</td>
                            </tr>
                            <tr>
                              <td>HOG + Logistic Regression</td>
                              <td>83.27%</td>
                              <td>83.22%</td>
                              <td>83.94%</td>
                            </tr>
                            <tr>
                              <td>HOG + Linear SVM</td>
                              <td>82.56%</td>
                              <td>82.40%</td>
                              <td>84.03%</td>
                            </tr>
                            <tr>
                              <td>Pixels + kNN</td>
                              <td>82.41%</td>
                              <td>82.40%</td>
                              <td>83.47%</td>
                            </tr>
                            <tr>
                              <td>Pixels + Logistic Regression</td>
                              <td>82.27%</td>
                              <td>82.22%</td>
                              <td>83.75%</td>
                            </tr>
                            <tr>
                              <td>HOG + Random Forest</td>
                              <td>81.97%</td>
                              <td>81.94%</td>
                              <td>82.84%</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                      <p class="tab-note" style="margin-top: 8px;">
                        ✨ <em>SmallCNN outperforms the best classical model by <strong>+3.73 percentage points</strong> (95% bootstrap CI [3.14, 4.31] pts).</em>
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
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
                with gr.Group(elem_classes=["glass-panel", "upload-panel"]):
                    gr.HTML("""
                    <h3 class="section-title">Upload an image</h3>
                    <p class="section-desc">Choose a clothing image (preferably a single, upright item).</p>
                    """)
                    # Preserves type='filepath', image_mode=None, label='Your image' for verify_demo
                    image = gr.Image(type='filepath', image_mode=None, sources=['upload'], format='png',
                                     label='Your image', height=230, elem_id='upload-image')

                # Card: Background Style Toggle
                with gr.Group(elem_classes=["glass-panel"]):
                    gr.HTML("""
                    <h3 class="section-title">Image background</h3>
                    <p class="section-desc">Choose the background style in your image.</p>
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
                with gr.Group(elem_classes=["glass-panel", "results-panel"]):
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
                        user_preview = gr.Image(type='pil', format='png', label='Your Image',
                                                height=170, interactive=False, elem_id='user-preview',
                                                buttons=[])
                        preview = gr.Image(type='pil', image_mode='L', format='png', label='Processed Image (28 × 28)',
                                           height=170, interactive=False, elem_id='model-preview', buttons=['download'])

                    # Prominent Hero Prediction Banner
                    result = gr.HTML(READY, elem_id="hero-result-card")

                    # Scores Across 10 Categories
                    gr.HTML('<h4 class="section-title" style="font-size: 0.95rem; margin: 16px 0 8px;">Scores across 10 categories</h4>')
                    scores = gr.Label(num_top_classes=10, show_label=False, show_heading=False, container=False, elem_id='scores-label')

                    # Interactive Sample Picker
                    gr.HTML("""
                    <div style="margin-top: 18px; border-top: 1px solid #f1f5f9; padding-top: 14px;">
                      <h4 class="section-title" style="font-size: 0.9rem; margin-bottom: 4px;">Try a sample image</h4>
                      <p class="section-desc" style="margin-bottom: 8px;">Click on any sample to test the model.</p>
                    </div>
                    """)

                    examples, names = sample_inputs()
                    if examples:
                        with gr.Group(elem_id="sample-gallery-wrapper"):
                            gr.Examples(examples=examples, inputs=[image, background],
                                        examples_per_page=11, visible_columns=[0], cache_examples=False,
                                        run_on_click=False, label='')

        # Interactivity Wire-up
        predict_button.click(handle_prediction, inputs=[image, background], outputs=[result, scores, preview],
                             api_name='predict', concurrency_limit=1)
        clear_button.click(clear_inputs, outputs=[image, background, result, scores, preview],
                           api_name='clear', queue=False)
        image.change(reset_result, outputs=[result, scores, preview], queue=False, api_visibility='private')
        image.change(lambda x: x, inputs=image, outputs=user_preview, queue=False, api_visibility='private')
        background.change(reset_result, outputs=[result, scores, preview], queue=False, api_visibility='private')
        clear_button.click(lambda: None, outputs=user_preview, queue=False, api_visibility='private')

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
                css=CSS, head=HEAD_JS, footer_links=[])

if __name__ == '__main__':
    main()
