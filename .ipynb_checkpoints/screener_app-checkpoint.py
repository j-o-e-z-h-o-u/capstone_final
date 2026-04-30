"""
Film Investment Screener — Gradio web UI
========================================

A browser-based interface for the trained capstone models.

Setup
-----
1. Make sure the `artifacts/` directory (created by capstone_full.ipynb)
   sits next to this file. It must contain:
       gbm_fused.pkl
       profit_clf.pkl
       feature_pipeline.pkl
       profit_feature_pipeline.pkl
       residuals.npy
       reference_films.parquet

2. Install Gradio:
       pip install gradio

3. Run:
       python screener_app.py

   The app opens at http://127.0.0.1:7860 in your browser.

   To get a temporary public link (72 hours, useful for sharing demos),
   change `demo.launch()` at the bottom to `demo.launch(share=True)`.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# Load all artifacts once at startup
# ============================================================================
ARTIFACTS = Path("artifacts")
gbm_fused        = joblib.load(ARTIFACTS / "gbm_fused.pkl")
profit_clf       = joblib.load(ARTIFACTS / "profit_clf.pkl")
vec, scaler, ohe = joblib.load(ARTIFACTS / "feature_pipeline.pkl")
vec_p, scaler_p, ohe_p = joblib.load(ARTIFACTS / "profit_feature_pipeline.pkl")
residuals        = np.load(ARTIFACTS / "residuals.npy")
reference        = pd.read_parquet(ARTIFACTS / "reference_films.parquet")

ref_text_vec = vec.transform(reference["text_features"].fillna(""))

print(f"Loaded {len(reference):,} reference films")
print(f"GBM expects {gbm_fused.n_features_in_} features")
print(f"Profitability classifier expects {profit_clf.n_features_in_} features")


NUMERIC_COLS = ["budget_log", "runtime", "vote_count_log", "vote_average",
                "popularity", "director_mean_log_revenue_prior",
                "director_n_prior_films", "release_month",
                "is_summer", "is_holiday_window"]
CAT_COLS = ["primary_genre", "original_language", "release_quarter"]

DEFAULTS = {
    "vote_count": 1500,
    "vote_average": 6.5,
    "popularity": 25.0,
    "runtime": 105,
    "original_language": "en",
    "primary_genre": "drama",
    "release_month": 7,
    "director_mean_log_revenue_prior": float(np.median(reference["log_box_office"])),
    "director_n_prior_films": 0,
}


# ============================================================================
# Feature builders
# ============================================================================
def _resolve(film: dict) -> tuple:
    """Apply defaults and derive computed fields. Returns (resolved_film, text)."""
    f = {**DEFAULTS, **film}
    budget = float(f.get("budget", 0) or 0)
    f["budget_log"] = (
        np.log1p(budget) if budget > 0
        else np.log1p(reference["budget"].dropna().median())
    )
    f["vote_count_log"] = np.log1p(f["vote_count"])
    rm = int(f["release_month"])
    f["release_quarter"] = (rm - 1) // 3 + 1
    f["is_summer"] = int(rm in (5, 6, 7, 8))
    f["is_holiday_window"] = int(rm in (11, 12))

    text = " ".join(filter(None, [
        f.get("genres", ""), f.get("keywords", ""),
        f.get("overview", ""), f.get("tagline", ""),
    ])).lower().strip()
    return f, text


def build_features(film: dict, profit: bool = False):
    """Build a sparse feature row using the appropriate pipeline."""
    f, text = _resolve(film)
    v, s, o = (vec_p, scaler_p, ohe_p) if profit else (vec, scaler, ohe)

    X_text = v.transform([text])
    X_num  = csr_matrix(s.transform(
        pd.DataFrame([[f[c] for c in NUMERIC_COLS]], columns=NUMERIC_COLS)
    ))
    cat = pd.DataFrame([[f[c] for c in CAT_COLS]], columns=CAT_COLS)\
            .fillna("unknown").astype(str)
    X_cat = o.transform(cat)
    return hstack([X_text, X_num, X_cat]).tocsr(), f, text


# ============================================================================
# Core predictions
# ============================================================================
def predict_revenue(film: dict, ci: float = 0.80) -> dict:
    X, info, _ = build_features(film, profit=False)
    pred_log = float(gbm_fused.predict(X.toarray())[0])
    alpha = (1 - ci) / 2
    low_off, high_off = np.quantile(residuals, [alpha, 1 - alpha])
    return {
        "log_pred":     pred_log,
        "revenue_pred": float(np.expm1(pred_log)),
        "revenue_low":  float(np.expm1(pred_log + low_off)),
        "revenue_high": float(np.expm1(pred_log + high_off)),
        "info":         info,
    }


def predict_profitability(film: dict) -> dict:
    if not film.get("budget"):
        return {"p_profitable": None, "flag": "NO BUDGET"}
    X, _, _ = build_features(film, profit=True)
    p = float(profit_clf.predict_proba(X)[0, 1])
    if p > 0.65:
        flag = "🟢 GREEN — likely greenlight"
    elif p > 0.45:
        flag = "🟡 YELLOW — borderline, review carefully"
    else:
        flag = "🔴 RED — likely unprofitable"
    return {"p_profitable": p, "flag": flag,
            "break_even_target": 2.5 * film["budget"]}


def find_similar_films(film: dict, k: int = 5,
                       min_similarity: float = 0.05) -> pd.DataFrame:
    _, _, text = build_features(film, profit=False)
    query_vec = vec.transform([text])
    sims = cosine_similarity(query_vec, ref_text_vec).ravel()
    top_idx = np.argsort(-sims)[:k * 3]
    top_idx = [i for i in top_idx if sims[i] >= min_similarity][:k]
    if not top_idx:
        return pd.DataFrame()
    out = reference.iloc[top_idx].copy()
    out["Similarity"] = sims[top_idx].round(2)
    out["Budget ($M)"] = (out["budget"] / 1e6).round(1)
    out["Revenue ($M)"] = (out["total_box_office"] / 1e6).round(1)
    out["ROI"] = (out["total_box_office"] / out["budget"]).round(2)
    out["Year"] = out["release_year"].astype(int)
    out = out.rename(columns={"clean_title": "Title", "primary_genre": "Genre"})
    return out[["Title", "Year", "Genre", "Budget ($M)",
                "Revenue ($M)", "ROI", "Similarity"]].reset_index(drop=True)


def generate_risk_flags(film: dict, prediction: dict) -> list:
    flags = []
    info = prediction["info"]
    budget = info["budget"] if "budget" in info else film.get("budget", 0)
    budget = float(budget or 0)

    if budget == 0:
        flags.append(("HIGH",
            "No budget specified — model uses imputed median budget. Review manually."))
    elif budget < 5e6:
        flags.append(("MEDIUM",
            f"Low budget (${budget/1e6:.1f}M) is below the median; predictions are noisy."))
    elif budget > 250e6:
        flags.append(("MEDIUM",
            f"Very high budget (${budget/1e6:.0f}M) — model tends to undershoot the largest blockbusters."))

    lang = info["original_language"]
    if lang and lang != "en":
        flags.append(("HIGH",
            f"Non-English film (lang='{lang}'). Model over-predicts US box office for these."))

    if info["release_month"] in (1, 2, 9, 10):
        flags.append(("MEDIUM",
            f"Weak release window (month {info['release_month']}). Summer/holiday would predict ~20-40% higher."))

    genre = (info["primary_genre"] or "").lower()
    if genre in ("romance", "drama") and budget > 50e6:
        flags.append(("MEDIUM",
            f"High-budget {genre} films historically struggle to recoup."))

    track_films = info.get("director_n_prior_films", 0)
    if track_films == 0:
        flags.append(("LOW",
            "First-time or unknown director — director-history feature uses median fallback."))

    rev_pred = prediction["revenue_pred"]
    if rev_pred > 0:
        spread = (prediction["revenue_high"] - prediction["revenue_low"]) / rev_pred
        if spread > 5:
            flags.append(("HIGH",
                f"Very wide CI ({spread:.1f}x point estimate). High uncertainty."))

    if not flags:
        flags.append(("OK", "No significant risk flags. Prediction is in a well-covered regime."))
    return flags


# ============================================================================
# Gradio handler — turns form inputs into a markdown report + a similar-films table
# ============================================================================
def run_screening(title, keywords, overview, tagline, budget_M, runtime,
                  release_month, primary_genre, original_language,
                  director_track_record, director_prior_films,
                  vote_count, vote_average, popularity):
    """Build the film dict from form inputs, run all predictions, return UI outputs."""

    # Helper: convert UI input to int/float safely; returns default if None or empty
    def safe_int(v, default=0):
        try:
            return int(v) if v not in (None, "") else default
        except (TypeError, ValueError):
            return default

    def safe_float(v, default=0.0):
        try:
            return float(v) if v not in (None, "") else default
        except (TypeError, ValueError):
            return default

    budget_val = safe_float(budget_M, 0)
    film = {
        "title": title or "Untitled",
        "keywords": keywords or "",
        "overview": overview or "",
        "tagline": tagline or "",
        "budget": budget_val * 1e6 if budget_val > 0 else None,
        "runtime": safe_int(runtime, 105),
        "release_month": safe_int(release_month, 7),
        "primary_genre": (primary_genre or "drama").lower(),
        "original_language": original_language or "en",
        "director_mean_log_revenue_prior": safe_float(director_track_record, 18.5),
        "director_n_prior_f

    # ---- Build the markdown report ----
    rev_pt   = revenue["revenue_pred"] / 1e6
    rev_low  = revenue["revenue_low"]  / 1e6
    rev_high = revenue["revenue_high"] / 1e6
    spread   = (rev_high - rev_low) / max(rev_pt, 0.01)

    if spread < 2:
        spread_label = "tight ✅"
    elif spread < 5:
        spread_label = "normal"
    else:
        spread_label = "wide ⚠️"

    report = f"""## 🎬 {film["title"]}

### 💰 Revenue prediction (80% confidence interval)

| | |
|---|---|
| **Point estimate** | **${rev_pt:,.1f}M** |
| 80% CI low  | ${rev_low:,.1f}M |
| 80% CI high | ${rev_high:,.1f}M |
| Range / point | {spread:.1f}× — {spread_label} |

### 📊 Profitability

"""
    if profit["p_profitable"] is None:
        report += "_No budget specified — profitability cannot be computed._\n\n"
    else:
        p = profit["p_profitable"]
        report += f"""| | |
|---|---|
| **P(revenue > 2.5× budget)** | **{p:.0%}** |
| Verdict | {profit["flag"]} |
| Break-even target | ${profit["break_even_target"]/1e6:,.1f}M |

"""

    report += "### ⚠️ Risk flags\n\n"
    for sev, msg in flags:
        marker = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "OK": "✅"}.get(sev, "•")
        report += f"- {marker} **[{sev}]** {msg}\n"

    return report, similar


# ============================================================================
# Build the Gradio interface
# ============================================================================
GENRES = ["drama", "comedy", "action", "thriller", "horror", "romance",
          "science", "animation", "adventure", "fantasy", "crime", "war",
          "history", "mystery", "documentary", "family", "music", "western"]

LANGUAGES = ["en", "zh", "es", "fr", "ja", "ko", "de", "it", "ru", "hi", "pt"]

EXAMPLES = [
    [
        "Project Stellaris",
        "space alien future battle spaceship astronaut planet science fiction adventure",
        "An astronaut crew discovers an ancient alien artifact on a distant planet that threatens human civilization.",
        "The future will not survive its past.",
        150, 130, 7, "science", "en", 19.5, 4, 8000, 7.0, 60.0,
    ],
    [
        "Last Summer in Paris",
        "love romance relationship couple wedding paris woman heart",
        "Two strangers meet on a train in Paris and spend one weekend deciding whether to upend their lives.",
        "One weekend. One choice.",
        25, 105, 2, "romance", "en", 17.0, 2, 800, 7.2, 12.0,
    ],
    [
        "Sky Pirates",
        "family animation adventure fantasy magic children friendship pirate sky",
        "A young girl discovers a hidden city in the clouds and joins a band of sky pirates to save her village.",
        "Adventure is in the air.",
        80, 95, 11, "animation", "en", 18.5, 3, 2500, 7.5, 35.0,
    ],
]


with gr.Blocks(title="Film Investment Screener", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# 🎬 Film Investment Screener
        
        A multimodal box-office prediction tool. Enter a hypothetical film concept and the model returns:
        a revenue forecast with 80% confidence interval, the probability the film is profitable, the five most
        similar historical films, and concrete risk flags.

        Built on the capstone GBM regressor (R² = 0.33 on time-aware test set, 73% top-10% precision)
        and logistic regression profitability classifier (AUC = 0.88 on test).

        **Tip:** click a row at the bottom to load a worked example.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Concept")
            title    = gr.Textbox(label="Title", placeholder="e.g. Project Stellaris")
            keywords = gr.Textbox(
                label="Keywords (space-separated)",
                placeholder="e.g. space alien future adventure",
                lines=2,
                info="Use words that match genre/theme vocabulary, e.g. 'science fiction alien space' for sci-fi.",
            )
            overview = gr.Textbox(
                label="Overview / synopsis (optional)", lines=3,
                placeholder="One or two sentences describing the plot.",
            )
            tagline  = gr.Textbox(label="Tagline (optional)",
                                  placeholder="A short marketing line.")

        with gr.Column(scale=1):
            gr.Markdown("### Production")
            budget_M = gr.Number(label="Budget ($M)", value=100,
                                 info="Production budget in millions of USD.")
            runtime  = gr.Number(label="Runtime (minutes)", value=110)
            release_month = gr.Dropdown(
                choices=[(f"{m} — {n}", m) for m, n in [
                    (1, "Jan"), (2, "Feb"), (3, "Mar"), (4, "Apr"),
                    (5, "May (summer)"), (6, "Jun (summer)"),
                    (7, "Jul (summer)"), (8, "Aug (summer)"),
                    (9, "Sep"), (10, "Oct"),
                    (11, "Nov (holiday)"), (12, "Dec (holiday)")
                ]],
                value=7, label="Release month",
            )
            primary_genre = gr.Dropdown(
                choices=GENRES, value="drama", label="Primary genre",
            )
            original_language = gr.Dropdown(
                choices=LANGUAGES, value="en", label="Original language",
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Director track record")
            director_track_record = gr.Slider(
                minimum=14, maximum=21, value=18.5, step=0.1,
                label="Director's avg log-revenue (prior films)",
                info="14 ≈ flop director, 18.5 ≈ median, 21+ ≈ blockbuster director. "
                     "Use 18.5 if unknown.",
            )
            director_prior_films = gr.Number(
                label="Director's prior film count", value=2, precision=0,
            )

        with gr.Column():
            gr.Markdown("### Pre-release buzz (optional)")
            vote_count = gr.Number(
                label="Vote count (TMDB-style audience engagement)", value=1500,
                info="Higher = stronger pre-release attention. "
                     "Set to 0 if film hasn't opened anywhere yet.",
            )
            vote_average = gr.Slider(
                minimum=1, maximum=10, value=6.5, step=0.1,
                label="Quality rating expectation (1-10)",
            )
            popularity = gr.Slider(
                minimum=0, maximum=200, value=25, step=1,
                label="TMDB popularity score (proxy for current trend)",
            )

    submit = gr.Button("🔍 Run screening", variant="primary", size="lg")

    with gr.Row():
        report_output  = gr.Markdown(label="Screening report")
    with gr.Row():
        similar_output = gr.Dataframe(
            label="5 most similar historical films",
            interactive=False, wrap=True,
        )

    submit.click(
        fn=run_screening,
        inputs=[title, keywords, overview, tagline, budget_M, runtime,
                release_month, primary_genre, original_language,
                director_track_record, director_prior_films,
                vote_count, vote_average, popularity],
        outputs=[report_output, similar_output],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[title, keywords, overview, tagline, budget_M, runtime,
                release_month, primary_genre, original_language,
                director_track_record, director_prior_films,
                vote_count, vote_average, popularity],
        label="Click to load an example concept",
    )

    gr.Markdown(
        """---
        **Caveats** — this is a research prototype, not a production investment tool.
        Predictions are correlational, not causal. The model has limited coverage for non-English
        films, festival/limited releases, very low budget films (<$5M), and theatrical re-releases.
        Final greenlight decisions should incorporate marketing strategy, talent assessment, and
        cultural-timing judgment that are outside this model's scope.
        """
    )


if __name__ == "__main__":
    # demo.launch(share=True) gets you a 72-hour public URL — use for sharing demos.
    # demo.launch() runs locally only at http://127.0.0.1:7860
    demo.launch()
