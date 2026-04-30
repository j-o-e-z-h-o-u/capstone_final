"""
Film Investment Screener — Gradio web UI (redesigned)
=====================================================

Setup
-----
1. Place this file next to your `artifacts/` folder.
2. pip install gradio joblib scikit-learn pandas numpy scipy pyarrow
3. python3.11 screener_app.py
4. Open http://127.0.0.1:7860 in a browser.
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
# Load artifacts
# ============================================================================
ARTIFACTS = Path("artifacts")
gbm_fused        = joblib.load(ARTIFACTS / "gbm_fused.pkl")
profit_clf       = joblib.load(ARTIFACTS / "profit_clf.pkl")
vec, scaler, ohe = joblib.load(ARTIFACTS / "feature_pipeline.pkl")
vec_p, scaler_p, ohe_p = joblib.load(ARTIFACTS / "profit_feature_pipeline.pkl")
residuals        = np.load(ARTIFACTS / "residuals.npy")
reference        = pd.read_parquet(ARTIFACTS / "reference_films.parquet")
ref_text_vec     = vec.transform(reference["text_features"].fillna(""))

print(f"Loaded {len(reference):,} reference films")
print(f"GBM expects {gbm_fused.n_features_in_} features")
print(f"Profitability classifier expects {profit_clf.n_features_in_} features")


NUMERIC_COLS = ["budget_log", "runtime", "vote_count_log", "vote_average",
                "popularity", "director_mean_log_revenue_prior",
                "director_n_prior_films", "release_month",
                "is_summer", "is_holiday_window"]
CAT_COLS = ["primary_genre", "original_language", "release_quarter"]

DEFAULTS = {
    "vote_count": 1500, "vote_average": 6.5, "popularity": 25.0,
    "runtime": 105, "original_language": "en", "primary_genre": "drama",
    "release_month": 7,
    "director_mean_log_revenue_prior": float(np.median(reference["log_box_office"])),
    "director_n_prior_films": 0,
}


# ============================================================================
# Safe input coercion (handles None / "" / bad input from UI)
# ============================================================================
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


# ============================================================================
# Feature builders
# ============================================================================
def _resolve(film: dict):
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
# Predictions
# ============================================================================
def predict_revenue(film: dict, ci: float = 0.80):
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


def predict_profitability(film: dict):
    if not film.get("budget"):
        return {"p_profitable": None, "flag": "NO BUDGET",
                "color": "gray", "label": "Need budget"}
    X, _, _ = build_features(film, profit=True)
    p = float(profit_clf.predict_proba(X)[0, 1])
    if p > 0.65:
        return {"p_profitable": p, "flag": "GREEN — likely greenlight",
                "color": "#10b981", "label": "GREENLIGHT",
                "break_even_target": 2.5 * film["budget"]}
    elif p > 0.45:
        return {"p_profitable": p, "flag": "YELLOW — borderline",
                "color": "#f59e0b", "label": "BORDERLINE",
                "break_even_target": 2.5 * film["budget"]}
    else:
        return {"p_profitable": p, "flag": "RED — likely unprofitable",
                "color": "#ef4444", "label": "REJECT",
                "break_even_target": 2.5 * film["budget"]}


def find_similar_films(film: dict, k: int = 5, min_similarity: float = 0.05):
    _, _, text = build_features(film, profit=False)
    query_vec = vec.transform([text])
    sims = cosine_similarity(query_vec, ref_text_vec).ravel()
    top_idx = np.argsort(-sims)[:k * 3]
    top_idx = [i for i in top_idx if sims[i] >= min_similarity][:k]
    if not top_idx:
        return pd.DataFrame()
    out = reference.iloc[top_idx].copy()

    # Visual similarity strength — helps the user see which comps to trust.
    # Cosine similarity on TF-IDF text overlap, so 'high' means the textual
    # description is very close to the input regardless of primary_genre label.
    def strength(s):
        if s >= 0.30: return f"🟢 {s:.2f} (high)"
        if s >= 0.15: return f"🟡 {s:.2f} (medium)"
        return f"🔴 {s:.2f} (low)"

    out["Match Strength"] = [strength(s) for s in sims[top_idx]]
    out["Budget ($M)"] = (out["budget"] / 1e6).round(1)
    out["Revenue ($M)"] = (out["total_box_office"] / 1e6).round(1)
    out["ROI"] = (out["total_box_office"] / out["budget"]).round(2)
    out["Year"] = out["release_year"].astype(int)
    out = out.rename(columns={"clean_title": "Title", "primary_genre": "Genre"})
    return out[["Title", "Year", "Genre", "Budget ($M)",
                "Revenue ($M)", "ROI", "Match Strength"]].reset_index(drop=True)


def generate_risk_flags(film: dict, prediction: dict):
    flags = []
    info = prediction["info"]
    budget = float(info.get("budget", 0) or film.get("budget", 0) or 0)

    if budget == 0:
        flags.append(("HIGH",
            "No budget specified — model uses imputed median budget. Review manually."))
    elif budget < 5e6:
        flags.append(("MEDIUM",
            f"Low budget (${budget/1e6:.1f}M); predictions are noisy."))
    elif budget > 250e6:
        flags.append(("MEDIUM",
            f"Very high budget (${budget/1e6:.0f}M) — model tends to undershoot largest blockbusters."))

    lang = info["original_language"]
    if lang and lang != "en":
        flags.append(("HIGH",
            f"Non-English film (lang='{lang}'); may over-predict US box office."))

    if info["release_month"] in (1, 2, 9, 10):
        flags.append(("MEDIUM",
            f"Weak release month ({info['release_month']}). Summer/holiday could lift ~20-40%."))

    genre = (info["primary_genre"] or "").lower()
    if genre in ("romance", "drama") and budget > 50e6:
        flags.append(("MEDIUM",
            f"High-budget {genre} films historically struggle to recoup."))

    if info.get("director_n_prior_films", 0) == 0:
        flags.append(("LOW",
            "First-time/unknown director — uses median fallback."))

    rev_pred = prediction["revenue_pred"]
    if rev_pred > 0:
        spread = (prediction["revenue_high"] - prediction["revenue_low"]) / rev_pred
        if spread > 5:
            flags.append(("HIGH",
                f"Very wide CI ({spread:.1f}x point estimate); high uncertainty."))

    if not flags:
        flags.append(("OK", "No significant risk flags — well-covered regime."))
    return flags


# ============================================================================
# Output formatter — builds a clean HTML report with cards
# ============================================================================
def format_report_html(film: dict, revenue: dict, profit: dict, flags: list) -> str:
    rev_pt   = revenue["revenue_pred"] / 1e6
    rev_low  = revenue["revenue_low"]  / 1e6
    rev_high = revenue["revenue_high"] / 1e6
    spread   = (rev_high - rev_low) / max(rev_pt, 0.01)

    if spread < 2:
        spread_color = "#10b981"; spread_text = "tight"
    elif spread < 5:
        spread_color = "#3b82f6"; spread_text = "normal"
    else:
        spread_color = "#f59e0b"; spread_text = "wide"

    p = profit["p_profitable"]
    profit_label = profit["label"]
    profit_color = profit["color"]
    profit_pct   = f"{p:.0%}" if p is not None else "—"
    break_even   = (f"${profit['break_even_target']/1e6:,.1f}M"
                    if "break_even_target" in profit else "—")

    severity_colors = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b",
                       "LOW": "#3b82f6", "OK": "#10b981"}
    flag_html = ""
    for sev, msg in flags:
        c = severity_colors.get(sev, "#6b7280")
        flag_html += f"""
        <div style="display:flex; align-items:flex-start; gap:10px;
                    padding:10px 14px; margin-bottom:8px;
                    background:#f9fafb; border-left:4px solid {c};
                    border-radius:6px; font-size:14px;">
            <span style="background:{c}; color:white; padding:2px 8px;
                         border-radius:4px; font-size:11px;
                         font-weight:600; flex-shrink:0;">{sev}</span>
            <span style="color:#374151;">{msg}</span>
        </div>
        """

    title = film.get("title") or "Untitled"

    return f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif;">

  <h2 style="margin:0 0 24px 0; padding-bottom:14px;
             border-bottom:2px solid #e5e7eb; color:#111827;">
    🎬 {title}
  </h2>

  <!-- Top row: 2 big primary cards -->
  <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;
              margin-bottom:24px;">

    <!-- Revenue card -->
    <div style="background:linear-gradient(135deg, #eff6ff, #dbeafe);
                padding:24px; border-radius:12px; border:1px solid #bfdbfe;">
      <div style="font-size:12px; color:#1e40af; font-weight:600;
                  text-transform:uppercase; letter-spacing:0.5px;
                  margin-bottom:8px;">
        💰 Predicted Revenue
      </div>
      <div style="font-size:42px; font-weight:700; color:#1e3a8a;
                  line-height:1; margin-bottom:8px;">
        ${rev_pt:,.0f}<span style="font-size:24px; opacity:0.7;">M</span>
      </div>
      <div style="font-size:13px; color:#475569; margin-bottom:4px;">
        80% confidence interval
      </div>
      <div style="font-size:14px; color:#1e3a8a; font-weight:500;">
        ${rev_low:,.1f}M&nbsp;–&nbsp;${rev_high:,.1f}M
      </div>
      <div style="margin-top:10px; font-size:12px; color:{spread_color};
                  font-weight:600;">
        ● {spread_text} range ({spread:.1f}× point estimate)
      </div>
    </div>

    <!-- Profitability card -->
    <div style="background:linear-gradient(135deg,
                {profit_color}15, {profit_color}25);
                padding:24px; border-radius:12px;
                border:1px solid {profit_color}50;">
      <div style="font-size:12px; color:{profit_color}; font-weight:600;
                  text-transform:uppercase; letter-spacing:0.5px;
                  margin-bottom:8px;">
        📊 Profitability
      </div>
      <div style="font-size:42px; font-weight:700; color:{profit_color};
                  line-height:1; margin-bottom:8px;">
        {profit_pct}
      </div>
      <div style="font-size:13px; color:#475569; margin-bottom:4px;">
        {profit_label}
      </div>
      <div style="font-size:14px; color:#374151; font-weight:500;">
        Break-even target: {break_even}
      </div>
      <div style="margin-top:10px; font-size:12px; color:{profit_color};
                  font-weight:600;">
        ● P(revenue &gt; 2.5× budget)
      </div>
    </div>
  </div>

  <!-- Risk flags -->
  <div style="margin-top:8px;">
    <div style="font-size:13px; font-weight:600; color:#374151;
                text-transform:uppercase; letter-spacing:0.5px;
                margin-bottom:12px;">
      ⚠️&nbsp; Risk assessment
    </div>
    {flag_html}
  </div>

</div>
"""


# ============================================================================
# Gradio handler
# ============================================================================
def run_screening(title, keywords, budget_M, release_month, primary_genre,
                  logline, tagline, runtime, original_language,
                  director_track_record, director_prior_films,
                  vote_count, vote_average, popularity):

    budget_val = safe_float(budget_M, 0)
    film = {
        "title": title or "Untitled",
        "keywords": keywords or "",
        "overview": logline or "",
        "tagline": tagline or "",
        "budget": budget_val * 1e6 if budget_val > 0 else None,
        "runtime": safe_int(runtime, 105),
        "release_month": safe_int(release_month, 7),
        "primary_genre": (primary_genre or "drama").lower(),
        "original_language": original_language or "en",
        "director_mean_log_revenue_prior": safe_float(director_track_record, 18.5),
        "director_n_prior_films": safe_int(director_prior_films, 0),
        "vote_count": safe_int(vote_count, 1500),
        "vote_average": safe_float(vote_average, 6.5),
        "popularity": safe_float(popularity, 25.0),
    }

    revenue = predict_revenue(film)
    profit  = predict_profitability(film)
    similar = find_similar_films(film, k=5)
    flags   = generate_risk_flags(film, revenue)

    report_html = format_report_html(film, revenue, profit, flags)
    return report_html, similar


# ============================================================================
# Gradio interface
# ============================================================================
GENRES = ["drama", "comedy", "action", "thriller", "horror", "romance",
          "science", "animation", "adventure", "fantasy", "crime", "war",
          "history", "mystery", "documentary", "family", "music", "western"]

LANGUAGES = ["en", "zh", "es", "fr", "ja", "ko", "de", "it", "ru", "hi", "pt"]

MONTHS = [
    ("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4),
    ("May ☀️", 5), ("Jun ☀️", 6), ("Jul ☀️", 7), ("Aug ☀️", 8),
    ("Sep", 9), ("Oct", 10), ("Nov 🎄", 11), ("Dec 🎄", 12),
]

EXAMPLES = [
    ["Project Stellaris",
     "space alien future battle spaceship astronaut planet science fiction adventure",
     150, 7, "science",
     "An astronaut crew discovers an ancient alien artifact on a distant planet that threatens human civilization.",
     "The future will not survive its past.",
     130, "en", 19.5, 4, 8000, 7.0, 60.0],
    ["Last Summer in Paris",
     "love romance relationship couple wedding paris woman heart",
     25, 2, "romance",
     "Two strangers meet on a train in Paris and spend one weekend deciding whether to upend their lives.",
     "One weekend. One choice.",
     105, "en", 17.0, 2, 800, 7.2, 12.0],
    ["Sky Pirates",
     "family animation adventure fantasy magic children friendship pirate sky",
     80, 11, "animation",
     "A young girl discovers a hidden city in the clouds and joins a band of sky pirates.",
     "Adventure is in the air.",
     95, "en", 18.5, 3, 2500, 7.5, 35.0],
]

CUSTOM_CSS = """
.required-section {
    background: linear-gradient(135deg, #fef3c7, #fde68a) !important;
    border: 2px solid #f59e0b !important;
    border-radius: 14px !important;
    padding: 18px !important;
    margin-bottom: 12px !important;
}
.required-section label span {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #78350f !important;
}
.optional-label {
    color: #6b7280 !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}
.optional-section label span {
    font-size: 15px !important;
    font-weight: 500 !important;
    color: #374151 !important;
}
.optional-section .gradio-textbox textarea,
.optional-section .gradio-textbox input,
.optional-section .gradio-number input {
    font-size: 15px !important;
}
#run-btn {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    padding: 16px !important;
    margin: 16px 0 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
}
#clear-all-btn {
    background: #f3f4f6 !important;
    color: #6b7280 !important;
    font-size: 13px !important;
    border: 1px solid #d1d5db !important;
    padding: 6px 14px !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
}
.clear-row { gap: 4px !important; align-items: flex-end !important; }
.tiny-clear-btn {
    min-width: 32px !important;
    max-width: 32px !important;
    padding: 0 !important;
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    color: #9ca3af !important;
    font-size: 14px !important;
    margin-bottom: 6px !important;
}
.tiny-clear-btn:hover {
    background: #fee2e2 !important;
    color: #ef4444 !important;
    border-color: #fca5a5 !important;
}
.section-header {
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    color: #6b7280 !important;
    margin: 4px 0 !important;
}
"""

with gr.Blocks(
    title="Film Investment Screener",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown(
        """# 🎬 Film Investment Screener
        
        Multimodal box-office forecasting. Enter a hypothetical film concept; get a revenue estimate,
        profitability probability, comparable historical films, and risk flags.
        """
    )

    # ------------------------------------------------------------------
    # Top-level controls
    # ------------------------------------------------------------------
    with gr.Row():
        clear_all_btn = gr.Button("🗑️ Clear all fields",
                                  elem_id="clear-all-btn", scale=0)

    # ------------------------------------------------------------------
    # ⭐ Required inputs — visually highlighted
    # Each field is paired with a tiny ❌ button that clears just that field.
    # ------------------------------------------------------------------
    gr.Markdown('<div class="section-header">⭐ Required — these drive the prediction</div>')

    with gr.Group(elem_classes="required-section"):

        with gr.Row(elem_classes="clear-row"):
            title = gr.Textbox(
                label="🎬 Title",
                placeholder="e.g. Project Stellaris",
                value="", scale=20,
            )
            clr_title = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)

        with gr.Row(elem_classes="clear-row"):
            keywords = gr.Textbox(
                label="🔑 Keywords (English, space-separated)",
                placeholder="e.g. space alien future adventure science fiction",
                lines=2, scale=20,
                info="The single most important input. These drive the model's understanding of genre/theme.",
            )
            clr_keywords = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row(elem_classes="clear-row"):
                    budget_M = gr.Number(
                        label="💰 Budget ($M)", value=100, minimum=0, scale=10,
                        info="Production budget in millions USD.",
                    )
                    clr_budget = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)

            with gr.Column(scale=1):
                with gr.Row(elem_classes="clear-row"):
                    release_month = gr.Dropdown(
                        choices=[(name, val) for name, val in MONTHS],
                        value=7, label="📅 Release Month", scale=10,
                    )
                    clr_month = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)

            with gr.Column(scale=1):
                with gr.Row(elem_classes="clear-row"):
                    primary_genre = gr.Dropdown(
                        choices=GENRES, value="drama",
                        label="🎭 Primary Genre", scale=10,
                    )
                    clr_genre = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)

    # ------------------------------------------------------------------
    # ⚙️ Optional inputs — collapsed by default, larger fonts
    # ------------------------------------------------------------------
    with gr.Accordion("⚙️ Optional details (improves prediction accuracy)",
                      open=False, elem_classes="optional-section"):

        gr.Markdown(
            '<div class="optional-label" style="margin-top:8px;">'
            '📖 Story details — used as additional text features.'
            '</div>'
        )
        with gr.Row(elem_classes="clear-row"):
            logline = gr.Textbox(
                label="Logline / Synopsis",
                placeholder="One or two sentences about the plot.",
                lines=2, scale=20,
            )
            clr_logline = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)

        with gr.Row(elem_classes="clear-row"):
            tagline = gr.Textbox(
                label="Tagline",
                placeholder="Marketing one-liner.", scale=20,
            )
            clr_tagline = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)

        gr.Markdown(
            '<div class="optional-label" style="margin-top:14px;">'
            '🎥 Production details.</div>'
        )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row(elem_classes="clear-row"):
                    runtime = gr.Number(label="Runtime (min)", value=110, scale=10)
                    clr_runtime = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)
            with gr.Column(scale=1):
                with gr.Row(elem_classes="clear-row"):
                    original_language = gr.Dropdown(
                        choices=LANGUAGES, value="en",
                        label="Original language", scale=10,
                    )
                    clr_lang = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)

        gr.Markdown(
            '<div class="optional-label" style="margin-top:14px;">'
            '🎬 Director track record. Use defaults if unknown.</div>'
        )
        with gr.Row(elem_classes="clear-row"):
            director_track_record = gr.Slider(
                minimum=14, maximum=21, value=18.5, step=0.1,
                label="Director's avg log-revenue (prior films)",
                info="14 ≈ flop director • 18.5 ≈ median • 21+ ≈ blockbuster director",
                scale=20,
            )
            clr_dir_tr = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)
        with gr.Row(elem_classes="clear-row"):
            director_prior_films = gr.Number(
                label="Number of prior films directed", value=2,
                precision=0, scale=20,
            )
            clr_dir_pf = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)

        gr.Markdown(
            '<div class="optional-label" style="margin-top:14px;">'
            '📈 Pre-release buzz (leave defaults if unknown).</div>'
        )
        with gr.Row(elem_classes="clear-row"):
            vote_count = gr.Number(
                label="Pre-release vote count (TMDB-style audience engagement)",
                value=1500, scale=20,
            )
            clr_vc = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row(elem_classes="clear-row"):
                    vote_average = gr.Slider(
                        minimum=1, maximum=10, value=6.5, step=0.1,
                        label="Expected quality rating (1-10)", scale=10,
                    )
                    clr_va = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)
            with gr.Column(scale=1):
                with gr.Row(elem_classes="clear-row"):
                    popularity = gr.Slider(
                        minimum=0, maximum=200, value=25, step=1,
                        label="TMDB popularity", scale=10,
                    )
                    clr_pop = gr.Button("✕", elem_classes="tiny-clear-btn", scale=1)

    # ------------------------------------------------------------------
    # Wire up clear buttons
    # ------------------------------------------------------------------
    # Per-field clear: each tiny ✕ button resets just one field to its default.
    clr_title.click(   lambda: "",   outputs=title)
    clr_keywords.click(lambda: "",   outputs=keywords)
    clr_budget.click(  lambda: 100,  outputs=budget_M)
    clr_month.click(   lambda: 7,    outputs=release_month)
    clr_genre.click(   lambda: "drama", outputs=primary_genre)
    clr_logline.click( lambda: "",   outputs=logline)
    clr_tagline.click( lambda: "",   outputs=tagline)
    clr_runtime.click( lambda: 110,  outputs=runtime)
    clr_lang.click(    lambda: "en", outputs=original_language)
    clr_dir_tr.click(  lambda: 18.5, outputs=director_track_record)
    clr_dir_pf.click(  lambda: 2,    outputs=director_prior_films)
    clr_vc.click(      lambda: 1500, outputs=vote_count)
    clr_va.click(      lambda: 6.5,  outputs=vote_average)
    clr_pop.click(     lambda: 25,   outputs=popularity)

    # Top-level "Clear all" — resets every field to its default in one click.
    def _clear_all():
        return ("", "", 100, 7, "drama",
                "", "", 110, "en", 18.5, 2, 1500, 6.5, 25)

    clear_all_btn.click(
        _clear_all,
        outputs=[title, keywords, budget_M, release_month, primary_genre,
                 logline, tagline, runtime, original_language,
                 director_track_record, director_prior_films,
                 vote_count, vote_average, popularity],
    )

    # ------------------------------------------------------------------
    # Run button
    # ------------------------------------------------------------------
    submit = gr.Button("🔍  Run Screening", elem_id="run-btn", size="lg")

    # ------------------------------------------------------------------
    # Outputs — vertical, with breathing room
    # ------------------------------------------------------------------
    gr.Markdown(
        '<div class="section-header" style="margin-top:24px;">'
        '📋 Screening Report</div>'
    )
    report_output = gr.HTML()

    gr.Markdown(
        '<div class="section-header">🎯 Top 5 historically comparable films</div>'
        '<div style="font-size:13px; color:#6b7280; margin-bottom:8px;">'
        'Matched by text similarity (genres + keywords + plot), so the listed '
        '<i>Genre</i> may differ from your input — that is expected. '
        'Trust 🟢 high-strength matches; treat 🔴 low-strength rows as weak references.'
        '</div>'
    )
    similar_output = gr.Dataframe(
        interactive=False, wrap=True, headers=None,
    )

    submit.click(
        fn=run_screening,
        inputs=[title, keywords, budget_M, release_month, primary_genre,
                logline, tagline, runtime, original_language,
                director_track_record, director_prior_films,
                vote_count, vote_average, popularity],
        outputs=[report_output, similar_output],
    )

    # ------------------------------------------------------------------
    # Examples
    # ------------------------------------------------------------------
    gr.Markdown(
        '<div class="section-header" style="margin-top:24px;">'
        '💡 Click an example to try</div>'
    )
    gr.Examples(
        examples=EXAMPLES,
        inputs=[title, keywords, budget_M, release_month, primary_genre,
                logline, tagline, runtime, original_language,
                director_track_record, director_prior_films,
                vote_count, vote_average, popularity],
        label="",
    )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    gr.Markdown(
        """
        ---
        **Caveats** — research prototype. Predictions are correlational, not causal.
        Limited coverage for non-English films, festival/limited releases, very low budget films,
        and theatrical re-releases. Final greenlight decisions should incorporate marketing
        strategy, talent, and cultural-timing judgment outside this model's scope.
        """
    )


if __name__ == "__main__":
    # Use demo.launch(share=True) for a 72-hour public URL.
    demo.launch()
