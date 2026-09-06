# Predicting and Explaining Software Development Estimation Error

**Why developers get effort estimates wrong — and whether the error can be predicted**

A study of 80,709 tasks from three independent public datasets, each recording **both the
estimate and the actual effort** for every task.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erdankud/dev_estimations_analysis/blob/main/notebooks/Dev_Estimations_Analysis.ipynb)

📄 **[Full report → reports/final_report.docx](reports/final_report.docx)** (Word)

---

## Headline result

**What is predictable is not the task — it is the circumstances.**

A model given only the developer's estimate predicts underestimation at ROC-AUC **0.49**,
no better than a coin flip. Add context — who, which project, which phase — and AUC rises
to **0.71**.

> The estimate carries no information about whether it is wrong. A developer does not
> "somewhat know" about their error; they know nothing about it.

![Ablation](reports/figures/11_ablation.png)

## Three of seven hypotheses were refuted

| RQ | hypothesis | verdict |
|---|---|---|
| RQ1 | Developers systematically underestimate | ❌ **refuted** — the aggregate bias is toward overestimation (×0.80 / ×0.86 / ×0.95) |
| RQ2 | Larger tasks are underestimated more | ❌ **refuted, opposite sign** — regression to the mean |
| RQ3 | Accuracy depends on the person | ⚠️ supported in CESAW (R² 0.10), not in SiP (0.03) |
| RQ4 | Project explains part of the variation | ⚠️ supported, small effect (0.03) |
| RQ5 | Type of work matters | ⚠️ supported, small effect (0.04) |
| RQ6 | Interruptions relate to the error | ✅ **supported — the strongest factor found** (×2 within every size band) |
| RQ7 | Round-number estimates are less accurate | ❌ **not supported** — the raw effect turned out to be an artefact of task size |

## Other findings

| | |
|---|---|
| 🎭 | **34% of SiP's "perfect estimates" are a logging artefact.** The share of `actual == estimate` falls from 60% on sub-hour tasks to 1% on multi-day ones: this is time logged against the plan, not accuracy. Leave those rows in and every metric lies. |
| 📉 | **Regression to the mean replicates across all three datasets** — different industries, cultures and units (minutes, hours, pomodoros). Slopes −0.17 / −0.24 / −0.81, all `p < 1e−140`. |
| 🧱 | **The explainability ceiling is low.** All observed factors combined explain 17% of the variance in CESAW and 14% in SiP. **83–86% is explained by nothing.** |
| 🔀 | **The split strategy shifts conclusions more than the choice of model:** ROC-AUC 0.775 (random) / 0.738 (unseen projects) / 0.709 (temporal). Reporting the random split would have inflated everything. |
| 📊 | **ML wins on small tasks and loses on large ones** (+6.6% on 2–4h tasks, −36.6% above 16h). The headline "10% gain" without that breakdown would be misleading. |
| 📏 | **The right product is an interval.** Conformal calibration lifts coverage from 73% to 78% against a nominal 80%. It does not reach nominal — the process is non-stationary, and claiming 80% would be untrue. |

## Data

| dataset | tasks | units | period | role |
|---|---:|---|---|---|
| [**CESAW**](https://arxiv.org/abs/2106.03679) | 60,284 | minutes | 2008–2017 | primary: 247 people, 45 projects |
| [**SiP**](https://arxiv.org/abs/1901.01621) | 10,266 | hours | 2004–2014 | validation on a different organisation |
| [**Renzo Pomodoro**](https://shape-of-code.com/2019/12/15/the-renzo-pomodoro-dataset/) | 10,159 | pomodoros | 2009–2019 | validation on a single individual |

Source: [Derek-Jones/Software-estimation-datasets](https://github.com/Derek-Jones/Software-estimation-datasets).
The data is not stored in this repository — it is downloaded on run.

## Layout

```
├── PROJECT_CONTEXT.md                    ← problem statement and methodology rules
├── DATA_DICTIONARY.md                    ← variable dictionary (Phase 0)
├── reports/
│   ├── final_report.docx                 ← full report (Word)
│   └── figures/                          ← 16 figures
├── notebooks/
│   └── Dev_Estimations_Analysis.ipynb    ← Google Colab notebook
├── src/
│   └── analysis.py                       ← the whole pipeline (percent format)
├── build_notebook.py                     ← builds the notebook from analysis.py
├── results/                              ← 34 metric tables (CSV)
└── requirements.txt
```

`src/analysis.py` is written in percent format (`# %%` / `# %% [markdown]`), so it is both
an executable script and the notebook source — the code is never duplicated and the two
artifacts cannot drift apart.

## Running it

**Google Colab** — open the [notebook](notebooks/Dev_Estimations_Analysis.ipynb) via the
badge above. Dependencies and data are pulled by the first cell; a full run takes about
two minutes.

**Locally:**

```bash
git clone https://github.com/erdankud/dev_estimations_analysis
cd dev_estimations_analysis
pip install -r requirements.txt

python src/analysis.py        # downloads data, computes everything, writes reports/ and results/
python build_notebook.py      # rebuild the notebook after editing analysis.py
```

## Methodology

The rules were fixed **before** modelling (details in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)):

* **Understanding the data comes before modelling.** What one CESAW row represents was
  established by key search: `(project, wbs, plan_item, phase, person)` is the only unique
  key, and aggregating sessions by it reproduces the recorded actual for 99.3% of tasks.
* **Nothing is dropped silently** — the cleaning log lives in `results/01_cleaning_log.csv`.
* **Everything in log space.** `exp(MAE_log)` reads as "typically wrong by a factor of N".
* **Leakage control.** Features are limited to what is known at estimation time.
  Interruptions and session counts are excluded even though they are the strongest factor
  (RQ6). Historical features use a shifted expanding mean, and the absence of leakage is
  checked by an `assert`.
* **Three split strategies** instead of one; the temporal split is what gets reported.
* **A baseline is mandatory** — "actual = estimate", and it is very strong.
* **Significance ≠ effect size.** For factors with many levels (`person` has 246) an
  adjusted and an out-of-sample R² are computed as well.
* **Confounders are checked** — this is how hypothesis RQ7 fell apart.

## Stack

`pandas` · `numpy` · `scipy` · `scikit-learn` (`HistGradientBoosting`, `RandomForest`,
`LogisticRegression`, quantile regression, permutation importance) · `shap` · `matplotlib`
