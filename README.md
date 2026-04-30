---
title: Film Investment Screener
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: Multimodal box-office forecasting tool
---

# 🎬 Film Investment Screener

A research prototype that predicts box-office revenue and profitability for hypothetical film projects. Built as a **University of Chicago capstone project** extending an earlier coursework version (TF-IDF on keywords + genres, R² ≈ 0.46 on a random train/test split).

The capstone version addresses three concrete limitations of the original:

1. **Structured metadata is now used.** The original course project prepared budget, runtime, and release-timing features but never integrated them into the model — the final pipeline ran on TF-IDF text only. This version fuses text + numeric + categorical features.
2. **The evaluation is time-aware.** Train ≤ 2018, validation 2019–2021 (deliberately includes the pandemic shock), test ≥ 2022. Random splits leak future information; this protocol does not.
3. **Investment-specific metrics are reported.** Beyond RMSE/R², the model is graded on top-10% precision, decile lift, and hypothetical top-25 portfolio ROI — the metrics an investor actually cares about.

---

## 📊 Performance (time-aware test set, 2022+ films)

| Model | Test R² | Top-10% precision | Decile lift | Top-25 portfolio ROI |
|---|---|---|---|---|
| Course-project baseline (TF-IDF only) | **−0.11** | 0.57 | 4.6× | 2.12× |
| Capstone fused multimodal (Ridge) | 0.39 | 0.70 | 5.8× | 3.04× |
| Capstone fused multimodal (RF, tuned) | 0.32 | 0.72 | 5.9× | 3.13× |
| **Capstone fused multimodal (GBM)** | **0.33** | **0.73** | **5.9×** | **3.57×** |
| Profitability classifier (logistic) | — | — | — | **AUC = 0.88** |

The negative R² of the course-project baseline under a time-aware split was the empirical reversal that motivated this capstone — the original 0.46 R² on a random split was an artifact of the split, not real predictive power.

---

## 🛠️ Methodology

### Data
- **Box Office Mojo** annual CSVs (2000s, 2010s, 2024) — audited revenue ground truth
- **MovieLens** `movies.csv` — initial genre tagging
- **TMDB API** — expanded enrichment in a single combined call per film:
  `details + credits + keywords` returns budget, release_date, director, top-5 cast, overview, tagline, production companies/countries, plus the original genre/keyword/vote signals
- Final dataset: **4,999 films** with full text features; **3,471 films** with known budget for profitability training

### Features (5,045 total fused into one sparse matrix)
- ~5,000 TF-IDF text features over `genres + keywords + overview + tagline`
- 10 numeric: `budget_log`, `runtime`, `vote_count_log`, `vote_average`, `popularity`, `director_mean_log_revenue_prior`, `director_n_prior_films`, `release_month`, `is_summer`, `is_holiday_window`
- ~35 one-hot categorical: `primary_genre`, `original_language`, `release_quarter`

### Leakage-safe director history
For each film, director track-record features use **only films released strictly before** that film's release date. Iteration is chronological; the director's history is updated *after* writing out each row's features. This is the right way to encode talent history in a forecasting setting.

### Models
- **Regressor**: `GradientBoostingRegressor` on the fused feature matrix (best on investment metrics)
- **Profitability classifier**: `LogisticRegression` with class_weight="balanced" on the budget-known subset, target `revenue > 2.5 × budget` (industry theatrical break-even rule)
- Hyperparameters tuned via `TimeSeriesSplit` cross-validation, not random k-fold

### Confidence intervals
80% prediction intervals are constructed from the **empirical distribution of training residuals** — non-parametric, no Gaussian assumption.

### Genre patterns (unsupervised cross-validation)
KMeans clustering on TF-IDF text and LDA topic modeling, both run *without* using revenue as a feature, independently identify the same commercial-genre ranking:

| Rank | Cluster (top terms) | Mean revenue | Median |
|---|---|---|---|
| 🥇 | sci-fi / space / alien / future | **\$284M** | \$150M |
| 🥈 | family / animation / fantasy / magic | **\$210M** | \$93M |
| 🥉 | crime / thriller / police / detective | \$100M | \$50M |
| 4 | war / true story / biography | \$92M | \$45M |
| 5 | broad horror/comedy/drama | \$90M | \$40M |
| 6 | romance / love / relationship | \$70M | \$39M |
| 7 | school / teen / coming-of-age | \$70M | \$37M |

---

## ⚠️ Honest limitations

This is a research prototype, **not a production investment tool**. The model:

1. **Has no marketing-spend data.** P&A budgets are proprietary and absent from any public API. This is the largest missing signal in the entire academic box-office literature and limits the ceiling of any public-data model.
2. **Discounts non-English films heavily.** Training data is dominated by English-language releases. The risk-flag system warns when this fires.
3. **Underpredicts true blockbusters.** \$1B+ films are systematically undershot by 30–50% because what separates a \$500M film from a \$1.5B film is cultural-moment factors not in the feature set.
4. **Cannot distinguish theatrical re-releases** from new releases. Films like the 2024 LOTR re-release are over-predicted because TMDB carries the original release's high vote_count and popularity.
5. **Identifies correlations, not causation.** High-importance features like `mcu`, `marvel`, `sequel`, and `budget_log` are *proxies* for the real drivers (franchise strength, distribution scale, marketing intensity), not the drivers themselves.
6. **Does not handle streaming-first releases** distinctly from theatrical exclusives.
7. **Revenue is not inflation-adjusted**, which inflates the apparent difficulty of older-film prediction.

The recommended workflow is: use the model to narrow a candidate pool to a shortlist, apply manual filters for the issues above, and leave the final greenlight decision to human reviewers who can incorporate marketing strategy, talent assessment, and cultural-timing judgment outside the model's scope.

---

## 🎯 What the UI returns

For each hypothetical film the screener returns:

- **Predicted revenue with 80% CI** — non-parametric interval from training residuals
- **Profitability probability** — `P(revenue > 2.5 × budget)` with 🟢 / 🟡 / 🔴 verdict
- **Top 5 most similar historical films** — TF-IDF cosine, color-coded by match strength so weak comparisons are flagged
- **Risk flags** — concrete warnings (non-English film, weak release window, missing budget, very wide CI, etc.) derived from the capstone's error analysis

---

## 🔬 Reproducibility

Full pipeline (data collection, cleaning, feature engineering, modeling, evaluation, error analysis, unsupervised analysis, investment metrics, and limitations discussion) is documented in the capstone notebook. Trained artifacts (`gbm_fused.pkl`, `profit_clf.pkl`, `feature_pipeline.pkl`, `profit_feature_pipeline.pkl`, `residuals.npy`, `reference_films.parquet`) are loaded at startup; the Space rebuilds from these artifacts on each cold start.

---

**License:** MIT  
**Author:** UChicago capstone project, 2026
