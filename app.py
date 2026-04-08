"""
Indian Language Classifier - Streamlit GUI
Supports: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, Urdu
Models: XGBoost, Random Forest, SVM
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Language Classifier",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102,126,234,0.35);
    }
    .main-header h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 0.4rem 0; }
    .main-header p  { font-size: 1.05rem; opacity: 0.9; margin: 0; }

    .metric-card {
        background: #1e1e2e;
        border: 1px solid #2e2e3e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform .2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .label { font-size: .8rem; color: #888; text-transform: uppercase; letter-spacing:.08em; }
    .metric-card .value { font-size: 1.9rem; font-weight: 700; margin: .3rem 0; }
    .metric-card .sub   { font-size: .8rem; color: #aaa; }

    .result-card {
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: .6rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .result-card.top    { background: linear-gradient(135deg,#4CAF50,#2E7D32); color:white; }
    .result-card.second { background: linear-gradient(135deg,#2196F3,#1565C0); color:white; }
    .result-card.third  { background: linear-gradient(135deg,#FF9800,#E65100); color:white; }
    .result-card .lang  { font-size: 1.5rem; font-weight: 700; }
    .result-card .conf  { font-size: 1.2rem; font-weight: 600; }
    .result-card .algo  { font-size: .85rem; opacity: .85; }

    .algo-info-card {
        background: #1a1a2e;
        border: 1px solid #16213e;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
    }
    .algo-info-card h3 { color: #667eea; font-size: 1.1rem; margin-bottom: .8rem; }
    .algo-info-card p  { font-size: .88rem; line-height: 1.6; color: #ccc; }
    .algo-tag {
        display: inline-block;
        padding: .2rem .6rem;
        border-radius: 20px;
        font-size: .75rem;
        font-weight: 600;
        margin: .2rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: .95rem;
        font-weight: 600;
        padding: .6rem 1.4rem;
    }
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        background: rgba(102,126,234,.06);
    }
    .stButton > button {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: .7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: .88; color: white; }

    .lang-flag { font-size: 1.4rem; }
    div[data-testid="stSidebar"] { background: #0f0f1a; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
LANGUAGES = ['Bengali','Gujarati','Hindi','Kannada','Malayalam','Marathi','Punjabi','Tamil','Telugu','Urdu']
LANG_FLAGS = {
    'Bengali':'🇧🇩','Gujarati':'🇮🇳','Hindi':'🇮🇳','Kannada':'🇮🇳',
    'Malayalam':'🇮🇳','Marathi':'🇮🇳','Punjabi':'🇮🇳','Tamil':'🇮🇳','Telugu':'🇮🇳','Urdu':'🇵🇰'
}
LANG_COLORS = {
    'Bengali':'#FF6B6B','Gujarati':'#4ECDC4','Hindi':'#45B7D1','Kannada':'#96CEB4',
    'Malayalam':'#FFEAA7','Marathi':'#DDA0DD','Punjabi':'#98D8C8','Tamil':'#F0A500','Telugu':'#C8A2C8','Urdu':'#7EC8E3'
}
ALGO_INFO = {
    "XGBoost": {
        "full": "eXtreme Gradient Boosting",
        "color": "#FF6B35",
        "icon": "⚡",
        "tags": [("Ensemble","#FF6B35"),("Boosting","#FF4500"),("Tree-based","#FF8C00")],
        "description": (
            "XGBoost builds an ensemble of decision trees sequentially. "
            "Each new tree corrects errors made by previous trees using gradient descent on a loss function. "
            "It is highly efficient, handles missing values natively, and supports L1/L2 regularization to prevent overfitting. "
            "Excellent for tabular feature data like MFCCs."
        ),
        "pros": ["Very fast training","Handles missing values","Built-in regularization","Top competition winner"],
        "cons": ["Many hyperparameters","Can overfit on small data"],
        "complexity": "O(n·d·k) per tree",
        "feature": "Feature importance via gain"
    },
    "RandomForest": {
        "full": "Random Forest Classifier",
        "color": "#4CAF50",
        "icon": "🌲",
        "tags": [("Ensemble","#4CAF50"),("Bagging","#388E3C"),("Tree-based","#1B5E20")],
        "description": (
            "Random Forest trains many independent decision trees on random subsets of data (bagging) "
            "and random subsets of features. Predictions are made by majority vote. "
            "It is robust to overfitting, requires minimal tuning, and provides reliable feature importance scores. "
            "Works well with high-dimensional audio features."
        ),
        "pros": ["Robust to overfitting","Parallelizable","Good out-of-box performance","Feature importance"],
        "cons": ["Slower inference","Large memory for many trees"],
        "complexity": "O(n·√d·k) per tree",
        "feature": "Mean decrease impurity"
    },
    "SVM": {
        "full": "Support Vector Machine",
        "color": "#2196F3",
        "icon": "🔷",
        "tags": [("Kernel-based","#2196F3"),("RBF Kernel","#1565C0"),("Max-margin","#0D47A1")],
        "description": (
            "SVM finds the optimal hyperplane that maximally separates classes. "
            "With the RBF (Radial Basis Function) kernel, it maps data into a high-dimensional space, "
            "making it effective for non-linearly separable patterns in audio feature space. "
            "Probability calibration via Platt scaling is used for confidence scores."
        ),
        "pros": ["Effective in high dimensions","Memory efficient","Versatile kernels","Strong theoretical basis"],
        "cons": ["Slow on large datasets","Sensitive to feature scaling","Hard to interpret"],
        "complexity": "O(n²·d) to O(n³·d)",
        "feature": "Support vectors"
    }
}

# ─────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner=False)
def load_models():
    # ── sklearn cross-version compatibility patch ─────────────────
    # Fixes: 'DecisionTreeClassifier' object has no attribute 'monotonic_cst'
    # This happens when .pkl files were saved with sklearn < 1.2
    # but the current environment has sklearn >= 1.2
    import sklearn.tree
    for _cls in (
        sklearn.tree.DecisionTreeClassifier,
        sklearn.tree.DecisionTreeRegressor,
        sklearn.tree.ExtraTreeClassifier,
        sklearn.tree.ExtraTreeRegressor,
    ):
        if not hasattr(_cls, 'monotonic_cst'):
            _cls.monotonic_cst = None
    # ─────────────────────────────────────────────────────────────

    errors = []
    models, scaler, encoder = {}, None, None
    paths = {
        "scaler":  os.path.join(MODEL_DIR, "feature_scaler.pkl"),
        "encoder": os.path.join(MODEL_DIR, "label_encoder.pkl"),
        "XGBoost":       os.path.join(MODEL_DIR, "XGBoost_language_model.pkl"),
        "RandomForest":  os.path.join(MODEL_DIR, "RandomForest_language_model.pkl"),
        "SVM":           os.path.join(MODEL_DIR, "SVM_language_model.pkl"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            errors.append(f"Missing: {os.path.basename(path)}")
            continue
        try:
            obj = joblib.load(path)
            if name == "scaler":  scaler  = obj
            elif name == "encoder": encoder = obj
            else: models[name] = obj
        except Exception as e:
            errors.append(f"Load error {name}: {e}")
    return models, scaler, encoder, errors

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y, _ = librosa.effects.trim(y)
    mfcc     = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma   = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, contrast]).reshape(1, -1), y, sr

# ─────────────────────────────────────────────
# AUDIO VISUALIZATION HELPERS
# ─────────────────────────────────────────────
def plot_spectrogram(y, sr, title="Mel Spectrogram"):
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                                   ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title, color='white', fontsize=13, pad=10)
    ax.tick_params(colors='white'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
    for spine in ax.spines.values(): spine.set_edgecolor('#444')
    plt.tight_layout()
    return fig

def plot_mfcc(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    img  = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax, cmap='coolwarm')
    fig.colorbar(img, ax=ax)
    ax.set_title("MFCC (40 coefficients)", color='white', fontsize=13, pad=10)
    ax.tick_params(colors='white'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
    for spine in ax.spines.values(): spine.set_edgecolor('#444')
    plt.tight_layout()
    return fig

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2.5), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    times = np.linspace(0, len(y)/sr, len(y))
    ax.plot(times, y, color='#667eea', linewidth=.6, alpha=.9)
    ax.fill_between(times, y, alpha=.25, color='#764ba2')
    ax.set_xlabel("Time (s)", color='white'); ax.set_ylabel("Amplitude", color='white')
    ax.set_title("Waveform", color='white', fontsize=13, pad=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('#444')
    plt.tight_layout()
    return fig

def plot_chroma(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    img = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=ax, cmap='viridis')
    fig.colorbar(img, ax=ax)
    ax.set_title("Chromagram", color='white', fontsize=13, pad=10)
    ax.tick_params(colors='white'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
    for spine in ax.spines.values(): spine.set_edgecolor('#444')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_all(features_scaled, models, encoder):
    results = {}

    for name, model in models.items():
        # Predict class index
        pred_idx = model.predict(features_scaled)[0]

        # Convert to label
        language = encoder.inverse_transform([pred_idx])[0]

        # Get probabilities
        probs_raw = model.predict_proba(features_scaled)[0]

        # 🔥 FIX 1: Normalize probabilities (safety)
        probs_raw = probs_raw / np.sum(probs_raw)

        # 🔥 FIX 2: Get confidence for predicted class ONLY
        confidence = probs_raw[pred_idx] * 100

        # 🔥 FIX 3: Clamp to valid range (extra safety)
        confidence = float(min(max(confidence, 0), 100))

        # Convert all probs to percentage
        classes = encoder.inverse_transform(range(len(probs_raw)))
        probs_percent = (probs_raw * 100).tolist()

        results[name] = {
            "language": language,
            "confidence": confidence,
            "probs": dict(zip(classes, probs_percent))
        }

    return results

# ─────────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────────
def confidence_bar_chart(results):
    algos = list(results.keys())
    confs = [results[a]["confidence"] for a in algos]
    colors = [ALGO_INFO[a]["color"] for a in algos]
    fig = go.Figure(go.Bar(
        x=algos, y=confs,
        marker_color=colors,
        text=[f"{c:.1f}%" for c in confs],
        textposition='outside',
        width=.45,
    ))
    fig.update_layout(
        title="Confidence Score by Algorithm",
        yaxis=dict(range=[0,115], title="Confidence (%)", gridcolor='#2a2a3e'),
        xaxis_title="Algorithm",
        plot_bgcolor='#0f0f1a', paper_bgcolor='#0f0f1a',
        font=dict(color='white'),
        title_font_size=15,
        margin=dict(t=50,b=40,l=50,r=20),
    )
    return fig

def probability_distribution_chart(results):
    fig = make_subplots(rows=1, cols=len(results),
                        subplot_titles=list(results.keys()),
                        horizontal_spacing=0.08)
    for i, (algo, res) in enumerate(results.items(), 1):
        probs = res["probs"]
        langs = list(probs.keys()); vals = list(probs.values())
        bar_colors = [LANG_COLORS.get(l, '#888') for l in langs]
        fig.add_trace(go.Bar(
            x=vals, y=langs, orientation='h',
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in vals],
            textposition='outside',
            showlegend=False,
            name=algo
        ), row=1, col=i)
        fig.update_xaxes(title_text="Probability (%)", gridcolor='#2a2a3e', row=1, col=i)
    fig.update_layout(
        title="Language Probability Distribution (All Models)",
        plot_bgcolor='#0f0f1a', paper_bgcolor='#0f0f1a',
        font=dict(color='white', size=11),
        height=420, margin=dict(t=60,b=40,l=120,r=30),
        title_font_size=15,
    )
    return fig

def radar_chart(results):
    langs_all = LANGUAGES
    fig = go.Figure()
    for algo, res in results.items():
        vals = [res["probs"].get(l, 0) for l in langs_all]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=langs_all + [langs_all[0]],
            fill='toself',
            name=algo,
            line_color=ALGO_INFO[algo]["color"],
            opacity=0.65,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], gridcolor='#2a2a3e', color='#aaa'),
            angularaxis=dict(gridcolor='#2a2a3e', color='white'),
            bgcolor='#0f0f1a'
        ),
        showlegend=True,
        plot_bgcolor='#0f0f1a', paper_bgcolor='#0f0f1a',
        font=dict(color='white'),
        title="Radar: Language Probability Across Models",
        title_font_size=15,
        margin=dict(t=60,b=20,l=20,r=20),
        height=420,
    )
    return fig

def agreement_gauge(results):
    preds = [r["language"] for r in results.values()]
    agree = len(set(preds)) == 1
    votes = pd.Series(preds).value_counts()
    top   = votes.index[0]; count = votes.iloc[0]
    pct   = count / len(preds) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        title={"text": f"Model Agreement<br><span style='font-size:.85em;color:#aaa'>Top: {top}</span>",
               "font": {"color":"white","size":16}},
        number={"suffix":"%","font":{"color":"white","size":36}},
        gauge={
            "axis": {"range":[0,100],"tickcolor":"white"},
            "bar":  {"color": "#4CAF50" if agree else "#FF9800"},
            "steps": [
                {"range":[0,50],  "color":"#2a1a1a"},
                {"range":[50,80], "color":"#1a2a1a"},
                {"range":[80,100],"color":"#0a1a0a"},
            ],
            "threshold": {"line":{"color":"white","width":3},"thickness":.75,"value":66},
            "bgcolor": "#0f0f1a", "bordercolor": "#2a2a3e",
        }
    ))
    fig.update_layout(
        paper_bgcolor='#0f0f1a', font=dict(color='white'),
        height=280, margin=dict(t=30,b=10,l=20,r=20)
    )
    return fig

def heatmap_probs(results):
    algos = list(results.keys())
    langs = LANGUAGES
    matrix = [[results[a]["probs"].get(l, 0) for l in langs] for a in algos]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=langs, y=algos,
        colorscale='Plasma',
        text=[[f"{v:.1f}%" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size":10},
        hoverongaps=False,
        colorbar=dict(title="Prob %", tickfont=dict(color='white'), titlefont=dict(color='white'))
    ))
    fig.update_layout(
        title="Probability Heatmap: Algorithm × Language",
        plot_bgcolor='#0f0f1a', paper_bgcolor='#0f0f1a',
        font=dict(color='white'),
        height=250,
        margin=dict(t=50,b=80,l=100,r=30),
        title_font_size=15,
        xaxis=dict(tickangle=-30),
    )
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ Language Classifier")
    st.markdown("---")
    st.markdown("### 📁 Model Directory")
    model_dir_input = st.text_input("Path", value=MODEL_DIR, key="model_dir",
                                    help="Folder containing .pkl files")
    if model_dir_input != MODEL_DIR:
        MODEL_DIR = model_dir_input
        st.cache_resource.clear()

    st.markdown("---")
    st.markdown("### 🌐 Supported Languages")
    cols = st.columns(2)
    for i, lang in enumerate(LANGUAGES):
        cols[i%2].markdown(f"{LANG_FLAGS.get(lang,'🌐')} {lang}")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_waveform   = st.toggle("Show Waveform",    value=True)
    show_spectrogram= st.toggle("Show Spectrogram", value=True)
    show_mfcc       = st.toggle("Show MFCC",        value=True)
    show_chroma     = st.toggle("Show Chromagram",  value=False)
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + Librosa")

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
with st.spinner("Loading models..."):
    models, scaler, encoder, load_errors = load_models()

if load_errors:
    st.error("⚠️ Model Loading Issues:\n" + "\n".join(f"- {e}" for e in load_errors))
    st.info("📂 Make sure all `.pkl` files are in the same folder as `app.py`.")
    st.stop()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎙️ Indian Language Classifier</h1>
    <p>Upload an MP3 audio file to identify the spoken language using XGBoost, Random Forest, and SVM</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predict", "📊 Analysis & Charts", "🔬 Audio Features", "📚 About Algorithms"
])

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Upload MP3 Audio")
    
    col_upload, col_info = st.columns([3, 2])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Drop your MP3 file here",
            type=["mp3", "wav", "ogg", "flac"],
            help="Supported: MP3, WAV, OGG, FLAC"
        )

    with col_info:
        if uploaded_file:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">File</div>
                <div class="value" style="font-size:1rem;">{uploaded_file.name}</div>
                <div class="sub">{uploaded_file.size/1024:.1f} KB</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🎵 Upload an MP3 to classify the spoken language")

    if uploaded_file:
        st.audio(uploaded_file, format="audio/mp3")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if st.button("🚀 Classify Language", key="predict_btn"):
            with st.spinner("Extracting features and running models..."):
                try:
                    features, y_audio, sr = extract_features(tmp_path)
                    features_scaled = scaler.transform(features)
                    results = predict_all(features_scaled, models, encoder)
                    st.session_state["results"]  = results
                    st.session_state["y_audio"]  = y_audio
                    st.session_state["sr"]       = sr
                    st.session_state["filename"] = uploaded_file.name
                    st.success("✅ Classification complete!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                finally:
                    os.unlink(tmp_path)

    # ── Show results ──
    if "results" in st.session_state:
        results  = st.session_state["results"]
        y_audio  = st.session_state["y_audio"]
        sr       = st.session_state["sr"]

        st.markdown("---")
        st.markdown("### 🏆 Prediction Results")

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        preds = [r["language"] for r in results.values()]
        confs = [r["confidence"] for r in results.values()]
        agree = "✅ Yes" if len(set(preds)) == 1 else "⚠️ Partial"
        top_pred = pd.Series(preds).value_counts().index[0]
        top_conf = np.mean([r["confidence"] for r in results.values() if r["language"] == top_pred])

        for col, label, val, sub in [
            (m1, "Consensus Language", top_pred, "Top voted"),
            (m2, "Avg Confidence", f"{top_conf:.1f}%", "Across models"),
            (m3, "Model Agreement", agree, f"{len(set(preds))} unique pred(s)"),
            (m4, "Audio Duration", f"{len(y_audio)/sr:.1f}s", f"SR: {sr} Hz"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Per-algorithm result cards
        rank_styles = ["top","second","third"]
        sorted_res = sorted(results.items(), key=lambda x: x[1]["confidence"], reverse=True)
        for rank, (algo, res) in enumerate(sorted_res):
            style = rank_styles[rank] if rank < 3 else "third"
            flag  = LANG_FLAGS.get(res["language"], "🌐")
            st.markdown(f"""
            <div class="result-card {style}">
                <div>
                    <div class="algo">#{rank+1} · {ALGO_INFO[algo]['icon']} {algo}</div>
                    <div class="lang">{flag} {res['language']}</div>
                </div>
                <div class="conf">{res['confidence']:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # Confidence bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(confidence_bar_chart(results), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — ANALYSIS & CHARTS
# ══════════════════════════════════════════════
with tab2:
    if "results" not in st.session_state:
        st.info("🎯 Run a prediction first in the **Predict** tab.")
    else:
        results = st.session_state["results"]
        st.markdown("### 📊 Deep Analysis")

        st.plotly_chart(probability_distribution_chart(results), use_container_width=True)
        
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(heatmap_probs(results), use_container_width=True)
        with c2:
            st.plotly_chart(agreement_gauge(results), use_container_width=True)

        st.plotly_chart(radar_chart(results), use_container_width=True)

        # Detailed probability table
        st.markdown("### 📋 Detailed Probability Table")
        df_rows = []
        for algo, res in results.items():
            for lang, prob in res["probs"].items():
                df_rows.append({"Algorithm": algo, "Language": lang, "Probability (%)": round(prob, 2),
                                 "Predicted": "✅" if lang == res["language"] else ""})
        df = pd.DataFrame(df_rows).pivot_table(index="Language", columns="Algorithm",
                                                 values="Probability (%)", aggfunc="first")
        df = df.round(2)
        st.dataframe(df.style.background_gradient(cmap='YlOrRd', axis=None).format("{:.2f}%"),
                     use_container_width=True, height=380)

# ══════════════════════════════════════════════
# TAB 3 — AUDIO FEATURES
# ══════════════════════════════════════════════
with tab3:
    if "y_audio" not in st.session_state:
        st.info("🎯 Run a prediction first in the **Predict** tab.")
    else:
        y_audio = st.session_state["y_audio"]
        sr      = st.session_state["sr"]
        fname   = st.session_state.get("filename", "audio")

        st.markdown(f"### 🔬 Audio Feature Visualizations — `{fname}`")

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        dur = len(y_audio) / sr
        rms = float(np.sqrt(np.mean(y_audio**2)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y_audio)[0]))
        ste = float(np.mean(librosa.feature.spectral_centroid(y=y_audio, sr=sr)[0]))
        for col, label, val in [
            (col_stat1, "Duration", f"{dur:.2f}s"),
            (col_stat2, "RMS Energy", f"{rms:.4f}"),
            (col_stat3, "Zero Crossing Rate", f"{zcr:.4f}"),
            (col_stat4, "Spectral Centroid", f"{ste:.0f} Hz"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="font-size:1.4rem;">{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if show_waveform:
            st.markdown("#### 〰️ Waveform")
            st.pyplot(plot_waveform(y_audio, sr))

        if show_spectrogram:
            st.markdown("#### 🌈 Mel Spectrogram")
            st.pyplot(plot_spectrogram(y_audio, sr))

        if show_mfcc:
            st.markdown("#### 🎚️ MFCC (40 Coefficients)")
            st.pyplot(plot_mfcc(y_audio, sr))

        if show_chroma:
            st.markdown("#### 🎵 Chromagram")
            st.pyplot(plot_chroma(y_audio, sr))

        # MFCC mean bar chart
        st.markdown("#### 📊 Mean MFCC Values")
        mfcc_vals = np.mean(librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40).T, axis=0)
        fig_mfcc = px.bar(
            x=[f"MFCC {i+1}" for i in range(40)],
            y=mfcc_vals,
            color=mfcc_vals,
            color_continuous_scale='RdYlGn',
            labels={"x": "MFCC Coefficient", "y": "Mean Value"},
            title="Mean MFCC Coefficients",
        )
        fig_mfcc.update_layout(
            plot_bgcolor='#0f0f1a', paper_bgcolor='#0f0f1a',
            font=dict(color='white'), showlegend=False,
            coloraxis_showscale=False,
            margin=dict(t=50,b=80,l=50,r=20),
        )
        st.plotly_chart(fig_mfcc, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — ABOUT ALGORITHMS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 📚 Algorithm Details")

    for algo, info in ALGO_INFO.items():
        with st.expander(f"{info['icon']} {algo} — {info['full']}", expanded=(algo == "XGBoost")):
            c_left, c_right = st.columns([3, 2])
            with c_left:
                st.markdown(f"<div class='algo-info-card'><h3>{info['icon']} {info['full']}</h3>", unsafe_allow_html=True)
                tags_html = " ".join(
                    f"<span class='algo-tag' style='background:{col}22;color:{col};border:1px solid {col}'>{t}</span>"
                    for t, col in info["tags"]
                )
                st.markdown(tags_html, unsafe_allow_html=True)
                st.markdown(f"<p style='margin-top:.8rem'>{info['description']}</p></div>", unsafe_allow_html=True)

            with c_right:
                st.markdown("**✅ Pros**")
                for p in info["pros"]: st.markdown(f"- {p}")
                st.markdown("**❌ Cons**")
                for p in info["cons"]: st.markdown(f"- {p}")
                st.markdown(f"**⏱ Complexity:** `{info['complexity']}`")
                st.markdown(f"**🔍 Key Feature:** {info['feature']}")

    st.markdown("---")
    st.markdown("### 🔄 Feature Engineering Pipeline")
    
    pipeline_steps = [
        ("🎵 MP3 Input", "Raw audio file", "#667eea"),
        ("📥 Load @ 16kHz", "librosa.load(sr=16000)", "#764ba2"),
        ("✂️ Trim Silence", "librosa.effects.trim()", "#9c4dcc"),
        ("🎚️ MFCC ×40", "40 mel-frequency cepstral coefficients", "#c084fc"),
        ("🎹 Chroma ×12", "12 chroma features from STFT", "#e879f9"),
        ("📈 Contrast ×7", "7 spectral contrast bands", "#f0abfc"),
        ("🔗 Concatenate", "59-dim feature vector", "#4CAF50"),
        ("⚖️ Scale", "StandardScaler normalize", "#45B7D1"),
        ("🤖 Classify", "XGBoost / RF / SVM predict", "#FF6B35"),
    ]
    cols = st.columns(len(pipeline_steps))
    for col, (title, desc, color) in zip(cols, pipeline_steps):
        col.markdown(f"""
        <div style="background:{color}18;border:1px solid {color};border-radius:10px;
                    padding:.8rem .5rem;text-align:center;height:100px;display:flex;
                    flex-direction:column;justify-content:center;">
            <div style="font-size:1.2rem;">{title.split()[0]}</div>
            <div style="font-size:.72rem;font-weight:700;color:{color};margin:.2rem 0">
                {' '.join(title.split()[1:])}</div>
            <div style="font-size:.68rem;color:#aaa;line-height:1.3;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Algorithm Comparison Table")
    comp_df = pd.DataFrame([
        {"Algorithm": f"{ALGO_INFO[a]['icon']} {a}", "Type": ALGO_INFO[a]["tags"][0][0],
         "Training Speed": ts, "Inference Speed": ins, "Interpretability": interp,
         "High-Dim Perf": hdp, "Requires Scaling": rs}
        for a, ts, ins, interp, hdp, rs in [
            ("XGBoost",      "⚡⚡⚡", "⚡⚡⚡", "⭐⭐",   "⭐⭐⭐⭐", "❌ No"),
            ("RandomForest", "⚡⚡",   "⚡⚡",   "⭐⭐⭐", "⭐⭐⭐",   "❌ No"),
            ("SVM",          "⚡",     "⚡⚡⚡", "⭐",    "⭐⭐⭐⭐", "✅ Yes"),
        ]
    ])
    st.dataframe(comp_df.set_index("Algorithm"), use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#555;font-size:.85rem;padding:1rem 0">
    🎙️ Indian Language Classifier &nbsp;|&nbsp; 
    Models: XGBoost · Random Forest · SVM &nbsp;|&nbsp;
    Features: MFCC · Chroma · Spectral Contrast &nbsp;|&nbsp;
    Languages: 10 Indian Languages
</div>
""", unsafe_allow_html=True)