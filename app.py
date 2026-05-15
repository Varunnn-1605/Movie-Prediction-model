import os
import sys
from pathlib import Path

# Fix 1: Force all file loading relative to app.py's location
APP_DIR = Path(__file__).parent.resolve()
os.chdir(APP_DIR)

# Fix 2: Use an absolute path for model cache — avoids Windows backslash bug
os.environ["HF_HOME"] = str(APP_DIR / "hf_cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(APP_DIR / "hf_cache")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap
from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineScore — IMDb Rating Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Page background */
.stApp { background: #0d0d0f; color: #e8e4dc; }
section[data-testid="stSidebar"] { background: #111113; border-right: 1px solid #2a2a2e; }

/* Header */
.cinescore-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.cinescore-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #f5c842;
    letter-spacing: -0.5px;
    margin: 0;
}
.cinescore-header p {
    color: #888;
    font-size: 1rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* Score card */
.score-card {
    background: linear-gradient(135deg, #1a1a1e 0%, #1f1c14 100%);
    border: 1px solid #3a3520;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.score-number {
    font-family: 'DM Serif Display', serif;
    font-size: 5rem;
    color: #f5c842;
    line-height: 1;
    margin: 0;
}
.score-label {
    color: #888;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.score-range {
    color: #555;
    font-size: 0.8rem;
    margin-top: 0.3rem;
}

/* Section headers */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e8e4dc;
    border-bottom: 1px solid #2a2a2e;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}

/* Retrieved movie cards */
.movie-card {
    background: #17171a;
    border: 1px solid #2a2a2e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.movie-card:hover { border-color: #f5c842; }
.movie-card-title { font-weight: 500; color: #e8e4dc; font-size: 0.95rem; }
.movie-card-meta  { color: #666; font-size: 0.8rem; margin-top: 0.2rem; }
.movie-card-score { color: #f5c842; font-weight: 600; font-size: 1rem; float: right; }

/* LLM explanation box */
.llm-box {
    background: #141418;
    border-left: 3px solid #f5c842;
    border-radius: 0 10px 10px 0;
    padding: 1.2rem 1.5rem;
    color: #c8c4bc;
    font-size: 0.95rem;
    line-height: 1.7;
    margin: 1rem 0;
}

/* Confidence meter */
.confidence-bar-wrap {
    background: #1e1e22;
    border-radius: 6px;
    height: 8px;
    margin: 0.5rem 0;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #c8a020, #f5c842);
    transition: width 0.6s ease;
}

/* Sidebar labels */
.stSidebar label { color: #aaa !important; font-size: 0.85rem !important; }

/* Buttons */
.stButton > button {
    background: #f5c842 !important;
    color: #0d0d0f !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    width: 100%;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover { background: #e6b830 !important; }

/* Info pills */
.pill {
    display: inline-block;
    background: #1e1e22;
    border: 1px solid #2a2a2e;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    color: #888;
    margin: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Artifact loading ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    xgb         = joblib.load("xgb_model.pkl")
    scaler      = joblib.load("scaler.pkl")
    features    = joblib.load("features.pkl")
    mlb_genre   = joblib.load("mlb_genre.pkl")
    pca         = joblib.load("pca_model.pkl")
    raw_embs    = joblib.load("overview_embeddings_raw.pkl")
    titles      = joblib.load("movie_titles.pkl")
    return xgb, scaler, features, mlb_genre, pca, raw_embs, titles

@st.cache_resource(show_spinner="Building vector index...")
def build_faiss_index(raw_embs):
    try:
        import faiss
        dim   = raw_embs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(raw_embs.astype("float32"))
        return index, "faiss"
    except ImportError:
        # Fallback: cosine similarity with numpy
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        normed = raw_embs / (norms + 1e-9)
        return normed, "numpy"

@st.cache_resource(show_spinner="Loading sentence encoder...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── RAG retrieval ─────────────────────────────────────────────────────────────
def retrieve_similar(query_text, raw_embs, index_obj, index_type, titles, pca, top_k=5):
    embedder = load_embedder()
    q_emb = embedder.encode([query_text]).astype("float32")

    if index_type == "faiss":
        _, idx = index_obj.search(q_emb, top_k + 1)
        idx = idx[0]
    else:
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        sims   = index_obj @ q_norm.T
        idx    = np.argsort(sims.ravel())[::-1][:top_k + 1]

    return [titles[i] for i in idx if i < len(titles)][:top_k]


# ── Feature builder ───────────────────────────────────────────────────────────
def build_feature_row(form, mlb_genre, pca, embedder, features):
    row = {}

    # Log transforms
    row["log_budget"]        = np.log1p(form["budget"])
    row["log_popularity"]    = np.log1p(form["popularity"])
    row["log_vote_count"]    = np.log1p(form["vote_count"])
    row["budget_per_minute"] = form["budget"] / (form["runtime"] + 1)
    row["budget_missing"]    = 1 if form["budget"] == 0 else 0
    row["runtime"]           = form["runtime"]
    row["release_year"]      = form["release_year"]
    row["release_month"]     = form["release_month"]
    row["genre_count"]       = len(form["genres"])
    row["keyword_count"]     = 0
    row["is_english"]        = 1 if form["language"] == "en" else 0

    # Genre multi-label
    genre_encoded = mlb_genre.transform([form["genres"]])[0]
    for g, v in zip(mlb_genre.classes_, genre_encoded):
        row[f"genre_{g}"] = int(v)

    # NLP embedding → PCA 10-dim
    text = f"{form['overview']} {form.get('tagline', '')}".strip()
    raw_emb = embedder.encode([text])
    emb_reduced = pca.transform(raw_emb)[0]
    for i, val in enumerate(emb_reduced):
        row[f"embed_{i}"] = val

    # Decade dummies
    decade = (form["release_year"] // 10) * 10
    for d in range(1910, 2030, 10):
        row[f"decade_{d}"] = int(decade == d)

    # Top-30 keyword flags — all zero for new input (no keyword list available)
    for feat in features:
        if feat.startswith("kw_") and feat not in row:
            row[feat] = 0

    # Align to exact training feature order
    X = pd.DataFrame([row])
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]

    # Drop any object columns
    X = X.select_dtypes(include=["number"])
    X = X[features] if all(f in X.columns for f in features) else X

    return X


# ── LLM explanation via Claude API ───────────────────────────────────────────
def get_llm_explanation(movie_name, predicted_rating, similar_movies, form, api_key):
    if not api_key:
        return None

    similar_str = "\n".join(
        [f"- {t} (a comparable film retrieved from the dataset)" for t in similar_movies]
    )

    prompt = f"""You are a film analyst AI. A machine learning model has predicted an IMDb rating for a new movie.

Movie being evaluated:
- Title: {movie_name}
- Overview: {form['overview']}
- Genres: {', '.join(form['genres'])}
- Budget: ${form['budget']:,.0f}
- Runtime: {form['runtime']} minutes
- Language: {form['language']}
- Release Year: {form['release_year']}
- Popularity score: {form['popularity']}
- Vote count estimate: {form['vote_count']}

Most similar films retrieved from the dataset (by plot/theme similarity):
{similar_str}

The model predicts an IMDb rating of {predicted_rating:.1f}/10.

In 3–4 sentences, explain WHY this rating makes sense given the movie's characteristics and the comparable films. Be specific — reference the genres, budget, and similar films. Keep the tone analytical but accessible."""

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=20
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["content"][0]["text"]
    except Exception:
        pass
    return None


# ── SHAP waterfall ────────────────────────────────────────────────────────────
def plot_shap(xgb_model, X_row):
    try:
        X_row_clean = X_row.astype("float64")
        
        # XGBoost 3.x+ and SHAP parsing workaround for base_score
        import builtins
        original_float = builtins.float
        def mock_float(val):
            if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
                val = val.strip('[]')
            return original_float(val)
        
        builtins.float = mock_float
        try:
            explainer   = shap.TreeExplainer(xgb_model)
        finally:
            builtins.float = original_float
            
        shap_values = explainer.shap_values(X_row_clean)

        shap_exp = shap.Explanation(
            values        = shap_values[0],
            base_values   = float(explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value),
            data          = X_row_clean.iloc[0].values,
            feature_names = X_row_clean.columns.tolist()
        )

        # Capture whatever figure SHAP internally creates
        plt.clf()
        shap.plots.waterfall(shap_exp, max_display=12, show=False)
        fig = plt.gcf()                          # ← grab SHAP's own figure
        fig.patch.set_facecolor("#141418")
        fig.set_size_inches(9, 5)
        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"SHAP error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="cinescore-header">
  <h1>🎬 CineScore</h1>
  <p>Predict IMDb ratings with ML + semantic retrieval</p>
</div>
""", unsafe_allow_html=True)

# Load artifacts
try:
    xgb, scaler, features, mlb_genre, pca, raw_embs, titles = load_artifacts()
    index_obj, index_type = build_faiss_index(raw_embs)
    embedder = load_embedder()
    artifacts_ok = True
except FileNotFoundError as e:
    st.error(f"Could not find model artifacts. Make sure all .pkl files are in the same folder as app.py.\n\nMissing: {e}")
    artifacts_ok = False

if artifacts_ok:

    # ── Sidebar: inputs ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎥 Movie Details")
        movie_name = st.text_input("Movie title", placeholder="e.g. Interstellar")
        overview   = st.text_area(
            "Overview / Plot",
            placeholder="Describe the plot in 2–4 sentences...",
            height=120
        )
        tagline = st.text_input("Tagline (optional)", placeholder="e.g. The end is only the beginning")

        st.markdown("---")
        st.markdown("### 🎭 Metadata")

        all_genres = list(mlb_genre.classes_)
        genres_sel = st.multiselect("Genres", all_genres, default=["Drama"])

        col1, col2 = st.columns(2)
        with col1:
            release_year  = st.number_input("Release year",  min_value=1920, max_value=2030, value=2024)
            runtime       = st.number_input("Runtime (min)", min_value=30,   max_value=300,  value=110)
        with col2:
            release_month = st.number_input("Release month", min_value=1,    max_value=12,   value=6)
            language      = st.selectbox("Language", ["en", "fr", "es", "de", "ja", "ko", "hi", "other"])

        st.markdown("---")
        st.markdown("### 💰 Production")
        budget     = st.number_input("Budget ($)", min_value=0, max_value=500_000_000, value=30_000_000, step=1_000_000, format="%d")
        popularity = st.slider("Popularity score", 0.0, 200.0, 25.0, 0.5)
        vote_count = st.number_input("Expected vote count", min_value=0, max_value=500_000, value=5000, step=500)

        st.markdown("---")
        st.markdown("### 🤖 AI Explanation")
        api_key = st.text_input(
            "Anthropic API key (optional)",
            type="password",
            placeholder="sk-ant-...",
            help="Provide your key to get an LLM-generated explanation grounded in the retrieved similar films."
        )

        predict_btn = st.button("⚡ Predict Rating", use_container_width=True)

    # ── Main panel ─────────────────────────────────────────────────────────────
    if not predict_btn:
        # Landing state
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #444;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎞</div>
            <p style="font-size: 1.1rem; color: #555; max-width: 480px; margin: 0 auto; line-height: 1.6;">
                Fill in the movie details in the sidebar and click <strong style="color:#f5c842">Predict Rating</strong> to get an IMDb score prediction powered by XGBoost + semantic search.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️  How this works"):
            st.markdown("""
            **CineScore** is a hybrid ML + RAG system:

            1. **Feature extraction** — your input is transformed into the same features used in training: multi-label genre encoding, log-transformed budget/popularity/votes, decade dummies, and a 10-dim PCA embedding of your overview text.
            2. **RAG retrieval** — your overview is embedded with `all-MiniLM-L6-v2` and searched against 4800+ TMDB movie embeddings using FAISS to find the most thematically similar films.
            3. **Prediction** — XGBoost predicts the IMDb rating from the feature vector.
            4. **SHAP explanation** — a waterfall plot shows which features drove the prediction up or down.
            5. **LLM explanation** — if you provide an API key, Claude generates a human-readable explanation grounded in the retrieved comparable films.
            """)

    else:
        # ── Validate ───────────────────────────────────────────────────────────
        if not overview.strip():
            st.warning("Please enter a plot overview — it's needed for both the embedding features and RAG retrieval.")
            st.stop()

        if not genres_sel:
            st.warning("Please select at least one genre.")
            st.stop()

        form = dict(
            overview=overview,
            tagline=tagline,
            genres=genres_sel,
            budget=budget,
            popularity=popularity,
            vote_count=vote_count,
            runtime=runtime,
            release_year=release_year,
            release_month=release_month,
            language=language,
        )

        # ── Step 1: RAG retrieval ──────────────────────────────────────────────
        with st.spinner("🔍 Retrieving similar films..."):
            query_text   = f"{overview} {tagline}".strip()
            similar_movies = retrieve_similar(
                query_text, raw_embs, index_obj, index_type, titles, pca, top_k=5
            )

        # ── Step 2: Build features + predict ─────────────────────────────────
        with st.spinner("⚙️  Building features & predicting..."):
            X_row = build_feature_row(form, mlb_genre, pca, embedder, features)
            pred  = float(xgb.predict(X_row)[0])
            pred  = round(np.clip(pred, 1.0, 10.0), 1)

        # ── Step 3: LLM explanation ───────────────────────────────────────────
        llm_text = None
        if api_key:
            with st.spinner("✨ Generating AI explanation..."):
                llm_text = get_llm_explanation(
                    movie_name or "Untitled", pred, similar_movies, form, api_key
                )

        # ══════════════════════════════════════════════════════════════════════
        # Layout: 3 columns
        # ══════════════════════════════════════════════════════════════════════
        col_score, col_rag, col_explain = st.columns([1.1, 1.4, 1.5])

        # ── Column 1: Score ────────────────────────────────────────────────────
        with col_score:
            # Colour the score
            if pred >= 7.5:   score_color = "#4caf7d"
            elif pred >= 6.0: score_color = "#f5c842"
            else:             score_color = "#e05c4b"

            st.markdown(f"""
            <div class="score-card">
                <div class="score-label">Predicted IMDb Rating</div>
                <div class="score-number" style="color:{score_color}">{pred:.1f}</div>
                <div class="score-range">out of 10</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence proxy: how close pred is to the safe middle
            conf = 1 - abs(pred - 6.5) / 5.0
            conf = round(np.clip(conf, 0.3, 0.95), 2)
            conf_pct = int(conf * 100)

            st.markdown(f"""
            <p style="color:#666; font-size:0.8rem; margin:0.5rem 0 0.2rem">Model confidence</p>
            <div class="confidence-bar-wrap">
              <div class="confidence-bar-fill" style="width:{conf_pct}%"></div>
            </div>
            <p style="color:#f5c842; font-size:0.8rem; text-align:right; margin:0">{conf_pct}%</p>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

            # Input pills
            st.markdown(f"""
            <p style="color:#555; font-size:0.75rem; margin-bottom:0.4rem">Input snapshot</p>
            {"".join(f'<span class="pill">{g}</span>' for g in genres_sel)}
            <span class="pill">{release_year}</span>
            <span class="pill">${budget/1e6:.0f}M</span>
            <span class="pill">{runtime}min</span>
            <span class="pill">{language.upper()}</span>
            """, unsafe_allow_html=True)

        # ── Column 2: RAG similar films ────────────────────────────────────────
        with col_rag:
            st.markdown('<div class="section-title">📽 Similar Films Retrieved</div>', unsafe_allow_html=True)
            st.markdown(
                '<p style="color:#555; font-size:0.8rem; margin-bottom:0.8rem">'
                'Found via semantic search over 4,800+ TMDB plot embeddings</p>',
                unsafe_allow_html=True
            )
            for i, title in enumerate(similar_movies):
                rank_icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                st.markdown(f"""
                <div class="movie-card">
                    <span class="movie-card-title">{rank_icons[i]} {title}</span><br>
                    <span class="movie-card-meta">Retrieved match #{i+1} · semantic similarity</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Column 3: Explanation ─────────────────────────────────────────────
        with col_explain:
            st.markdown('<div class="section-title">🧠 Why This Rating?</div>', unsafe_allow_html=True)

            if llm_text:
                st.markdown(f'<div class="llm-box">{llm_text}</div>', unsafe_allow_html=True)
                st.markdown(
                    '<p style="color:#333; font-size:0.72rem; margin-top:0.3rem">'
                    '✦ Generated by Claude · grounded in retrieved comparable films</p>',
                    unsafe_allow_html=True
                )
            else:
                # Rule-based fallback
                genre_str = ", ".join(genres_sel)
                budget_tier = "high-budget" if budget > 80_000_000 else "mid-budget" if budget > 15_000_000 else "low-budget"
                if pred >= 7.5:
                    verdict = "strong performance"
                    reason  = f"The combination of {genre_str} genres, a {budget_tier} production, and strong popularity signals aligns with well-received films in the dataset."
                elif pred >= 6.0:
                    verdict = "solid, watchable rating"
                    reason  = f"The {genre_str} genre mix and {budget_tier} profile place it in the range of competent but unremarkable releases."
                else:
                    verdict = "below-average rating"
                    reason  = f"The model detects risk factors: {budget_tier} budget relative to genre expectations, or low audience engagement signals."

                st.markdown(f"""
                <div class="llm-box">
                    The model predicts a <strong>{verdict}</strong> of {pred:.1f}/10. {reason}
                    Similar films retrieved — {", ".join(similar_movies[:3])} — provide the nearest
                    thematic context from the training data.<br><br>
                    <em style="color:#555">Add an Anthropic API key in the sidebar for a richer, Claude-generated explanation.</em>
                </div>
                """, unsafe_allow_html=True)

        # ── SHAP waterfall ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">📊 Feature Attribution (SHAP)</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#555; font-size:0.85rem">Which features pushed the prediction above or below the dataset average?</p>',
            unsafe_allow_html=True
        )

        shap_fig = plot_shap(xgb, X_row)
        if shap_fig:
            st.pyplot(shap_fig, use_container_width=True)
        else:
            st.info("SHAP plot unavailable for this prediction.")

        # ── Top feature table ──────────────────────────────────────────────────
        with st.expander("🔢 Full feature values for this prediction"):
            display_df = X_row.T.rename(columns={X_row.index[0]: "value"})
            display_df = display_df[display_df["value"] != 0].sort_values("value", ascending=False)
            st.dataframe(display_df.style.format("{:.4f}"), height=300)
