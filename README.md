# 🎬 CineScore — IMDb Rating Predictor

A hybrid ML + RAG Streamlit app that predicts IMDb ratings using XGBoost and semantic search.

---

## How It Works

```
User Input (movie metadata + overview)
        │
        ▼
Sentence Transformer (all-MiniLM-L6-v2)
        │
   384-dim embedding
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
FAISS Vector Store                        PCA (384 → 10 dims)
(4800+ TMDB movies)                               │
        │                                  Feature row builder
   Top-5 similar films                            │
        │                                   XGBoost model
        │                                         │
        └──────────────────────────────────────────┤
                                                   ▼
                                         Predicted IMDb Rating
                                                   │
                                    ┌──────────────┼──────────────┐
                                    ▼              ▼              ▼
                              SHAP waterfall   LLM explanation  Similar films
                              (feature attr.)  (Claude API)     (RAG context)
```

---

## Setup

### 1. Copy pkl files from Colab

After running the notebook, download all 7 pkl files and place them in this folder:

```
streamlit_app/
├── app.py
├── requirements.txt
├── README.md
├── xgb_model.pkl               ← trained XGBoost model
├── scaler.pkl                  ← StandardScaler (for linear models)
├── features.pkl                ← ordered feature column names
├── mlb_genre.pkl               ← fitted MultiLabelBinarizer
├── pca_model.pkl               ← fitted PCA (384 → 10 dims)
├── overview_embeddings_raw.pkl ← 384-dim embeddings for all 4800 movies
└── movie_titles.pkl            ← movie titles aligned to embeddings
```

**To download from Colab:**
```python
from google.colab import files
for f in ['xgb_model.pkl','scaler.pkl','features.pkl','mlb_genre.pkl',
          'pca_model.pkl','overview_embeddings_raw.pkl','movie_titles.pkl']:
    files.download(f)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> `faiss-cpu` is optional but recommended for faster retrieval.
> The app falls back to numpy cosine similarity if FAISS is not installed.

### 3. Run the app

```bash
streamlit run app.py
```

---

## RAG Pipeline Details

The RAG (Retrieval-Augmented Generation) component works as follows:

1. **Index building** — at startup, `overview_embeddings_raw.pkl` (shape: 4803 × 384) is loaded into a FAISS `IndexFlatL2`. This runs once and is cached.

2. **Query encoding** — the user's overview + tagline is encoded with the same `all-MiniLM-L6-v2` model used in training.

3. **Retrieval** — the top-5 nearest neighbors in embedding space are returned. These are the most thematically similar films to the user's input.

4. **Grounded explanation** — if an Anthropic API key is provided, Claude is prompted with the predicted rating, the movie's metadata, and the 5 retrieved films to generate a human-readable explanation.

---

## LLM Explanation (optional)

The app calls the Anthropic Messages API with a structured prompt that includes:
- The movie's metadata (genres, budget, runtime, language)
- The predicted rating
- The 5 most similar retrieved films (grounding context)

This produces a 3–4 sentence explanation grounded in real comparable films — not hallucinated.

Get an API key at: https://console.anthropic.com

---

## Deploying to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to share.streamlit.io → New app → select your repo
3. Set `app.py` as the main file
4. Add `ANTHROPIC_API_KEY` as a secret (optional)

Note: `overview_embeddings_raw.pkl` is ~7MB — fine for Streamlit Cloud's 1GB limit.
