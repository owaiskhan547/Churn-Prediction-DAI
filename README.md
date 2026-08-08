# Bank Customer Churn — Predictive Modeling and Risk Scoring

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#setup)

Predictive churn intelligence system for a European retail bank: risk scoring, driver explainability, and a scenario-simulation dashboard, built against the "Predictive Modeling and Risk Scoring for Bank Customer Churn" project brief.

## Objectives

**Primary**
- Predict customer churn with high accuracy
- Generate churn probability scores
- Identify key churn drivers

**Secondary**
- Reduce false positives in churn detection
- Improve interpretability of ML models
- Enable scenario-based churn risk analysis

## Project structure

```
churn-prediction/
├── app.py                          Streamlit dashboard (4 modules)
├── data/                            European_Bank.csv (not tracked in git)
├── models/                          model, scaler, and column metadata (not tracked in git)
├── outputs/                         scored_customers.csv, feature_importance.csv (not tracked in git)
├── src/
│   ├── data_prep.py                 loading, encoding, scaling, train/test split
│   ├── feature_engineering.py       derived interaction features
│   ├── train.py                     4-model comparison + k-fold cross-validation
│   ├── threshold.py                 precision-targeted threshold tuning
│   ├── scoring.py                   churn probability + risk tier assignment
│   ├── explain.py                   SHAP (global + local) and partial dependence
│   ├── scenario.py                  what-if simulation, feature-consistent
│   └── pipeline.py                  orchestrates the full flow end to end
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place `European_Bank.csv` in `data/`.

## Data

`data/`, `models/`, and `outputs/` are gitignored — this repo ships code only,
not the dataset or trained artifacts. `European_Bank.csv` (10,000 rows) is a
labeled bank-churn dataset with the columns:

`Year, CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited`

`Exited` is the churn label (1 = churned); the class split is roughly 80/20
(7,963 retained vs. 2,037 churned). `CustomerId` and `Surname` are dropped
before modeling and never used as features.

Bring your own copy of this dataset (or a compatible one with the same schema)
and place it at `data/European_Bank.csv` — see [Contributing](CONTRIBUTING.md)
for the full setup-and-run sequence. If you're publishing a fork with a
different source dataset, confirm you have the right to redistribute it, or
keep it out of the repo (as this one does) and document where to obtain it
instead.

## Running the pipeline

```
cd src
python3 pipeline.py --data ../data/European_Bank.csv \
                     --model-out ../models/churn_model.joblib \
                     --scored-out ../outputs/scored_customers.csv \
                     --target-precision 0.65 \
                     --primary-model GradientBoosting
```

## Running the dashboard

```
streamlit run app.py
```

Requires the pipeline to have been run first (the app loads the persisted model, scaler, scored customers, and feature importance from `models/` and `outputs/`).

### Dashboard modules
- **Risk Calculator** — enter a customer's features, get a live churn probability and risk tier
- **Probability Distribution** — histogram of churn probability across the scored test set, by risk tier
- **Feature Importance** — SHAP-based global driver ranking
- **What-If Simulator** — adjust product count / active-member status for the configured customer and see the probability shift

## Methodology

**Preprocessing**: drop `CustomerId`/`Surname`/`Year` (non-informative or zero-variance), one-hot encode `Geography`/`Gender`, standard-scale continuous features (fit on train, applied to test — matters most for Logistic Regression, which is scale-sensitive).

**Feature engineering** (derived, not in raw data):
- `BalanceSalaryRatio` = Balance / EstimatedSalary
- `ProductDensity` = NumOfProducts / (Tenure + 1)
- `EngagementProductInteraction` = IsActiveMember × NumOfProducts
- `AgeTenureInteraction` = Age × Tenure

**Models compared** (stratified 80/20 split, 5-fold stratified cross-validation on ROC-AUC): Logistic Regression (interpretability baseline), Decision Tree, Random Forest, Gradient Boosting. XGBoost was scoped out of this build.

**Model selection**: Gradient Boosting — highest test ROC-AUC and the tightest cross-validation spread (most consistent across folds), used as the primary model. Its default-threshold precision/recall profile (high precision, lower recall) differs meaningfully from Random Forest's (lower precision, higher recall); the threshold is re-tuned rather than left at 0.5 for exactly this reason.

**Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC — plus 5-fold CV mean/std to check generalization, not just a single test-set score.

**Explainability**: SHAP for global feature ranking and per-customer local explanations; partial dependence plots for the top driver.

**False-positive reduction**: threshold selected from the precision-recall curve to hit a target precision, rather than defaulting to 0.5.

**Scenario simulation**: `simulate_scenario` operates on raw (unscaled) feature values, recomputes any dependent engineered features (e.g. changing `NumOfProducts` also updates `ProductDensity` and `EngagementProductInteraction`), then scales and predicts — so a "what if this customer had 2 products" query reflects a fully consistent, realistic customer, not just one column changed in isolation.

## Results (this dataset, `random_state=42`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC (mean ± std) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.709 | 0.382 | 0.698 | 0.494 | 0.776 | 0.766 ± 0.021 |
| Decision Tree | 0.759 | 0.446 | 0.771 | 0.565 | 0.829 | 0.825 ± 0.012 |
| Random Forest | 0.824 | 0.552 | 0.705 | 0.619 | 0.866 | 0.856 ± 0.012 |
| **Gradient Boosting** | **0.865** | **0.762** | 0.486 | 0.594 | **0.868** | **0.861 ± 0.011** |

**Gradient Boosting, threshold tuning:**

| Metric | Default (0.5) | Tuned (~0.35) |
|---|---|---|
| Precision | 0.762 | 0.651 |
| Recall | 0.486 | 0.609 |
| F1 | 0.594 | 0.629 |
| False positives | 62 | 133 |

Note the tradeoff direction is reversed from a model like Random Forest: Gradient Boosting's default threshold already favors precision over recall, so tuning here recovers recall rather than suppressing false positives further — the right choice depends on which error (missed churner vs. wasted outreach) costs the business more.

**Top global churn drivers (mean absolute SHAP value):** NumOfProducts, Age, Geography (Germany), Gender, EngagementProductInteraction, BalanceSalaryRatio.

## Known caveats

- `NumOfProducts = 3` (266 customers) and `= 4` (60 customers) show very high churn (82.7% and 100%), but on small sample sizes — treat as a real but low-confidence signal, worth validating on more data before it drives operational decisions.
- This dataset covers a single year (2025); no seasonal or trend effects can be assessed.

## Not yet built

- Research paper (EDA writeup, insights, recommendations)
- Executive summary for stakeholders

These are deferred by design (deprioritized for this build pass) and can be produced from the artifacts already generated in `outputs/` and this README's Results section.

## Contributing

Bug reports, fixes, and small improvements are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, project conventions, and how to
open a PR.

## License

Released under the [MIT License](LICENSE). The code is MIT-licensed; the
dataset is not included in this repository and is not covered by that
license — see [Data](#data) above.
