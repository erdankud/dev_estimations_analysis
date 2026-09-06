# PROJECT_CONTEXT.md

The frame for this project: problem statement, research questions, methodology rules and
the limits of what the findings support.

---

## 1. Name and purpose

**Predicting and Explaining Software Development Estimation Error**

The project investigates why developers get effort estimates wrong, and whether the error
can be predicted in advance. Two complementary goals:

1. **Explanatory** — identify the factors associated with estimation error.
2. **Predictive** — build models that predict the fact of underestimation and the
   magnitude of the error.

The central question:

> **Can we predict that a task will be underestimated, and what factors explain the
> estimation error?**

The phrasing is deliberate. "Predict `actual_hours`" is an ordinary regression exercise.
"Why is `Actual ≠ Estimate`" is an investigation in which the data can refute a hypothesis
rather than only confirm it. That is what happened here: **three of seven hypotheses were
refuted**.

## 2. Definition of success

The project is judged **not by maximum R² or accuracy**. Success means demonstrating the
ability to:

- work through messy real data and establish what one row means;
- state testable hypotheses;
- run EDA and find data-collection artefacts;
- distinguish correlation from causation and find confounders;
- engineer features without leakage;
- establish an honest baseline;
- train and evaluate models;
- explain their predictions;
- communicate uncertainty correctly;
- state the limits of the conclusions.

A model that loses to the baseline but explains **why** is a result. A model with a high
R² built on leaked features is not.

## 3. Data

| dataset | tasks | role |
|---|---:|---|
| **CESAW** | 60,284 | primary: 247 people, 45 projects, has interruption data |
| **SiP** | 10,266 | validation on a different organisation |
| **Renzo Pomodoro** | 10,159 | validation at the individual level |

CESAW was chosen as primary for three reasons: the volume supports non-toy ML; it carries
a genuine estimate → actual pair at the **task** level rather than the project level
(unlike PROMISE/COCOMO, where a row is a whole project); and it has contextual variables,
including interruptions.

ISBSG (13,147 projects) was considered and rejected: full access is paid (AUD 3,000).

Variable-level documentation is in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

## 4. Target variables

Absolute error `actual − estimate` is useless on its own: two hours of error on a two-hour
task and on a 200-hour task are different events. Hence:

| | formula | purpose |
|---|---|---|
| `estimation_ratio` | `actual / estimate` | the primary quantity |
| `log_ratio` | `log(actual / estimate)` | regression target |
| `underestimated` | `1 if actual > estimate` | classification target |

**Why logarithms.** The ratio is strictly positive and heavily skewed (the CESAW maximum is
×298). In log space "twice as long" and "half as long" are symmetric, the mean of
`log_ratio` is the log of the geometric mean (a robust bias measure), and `exp(MAE_log)`
reads as "typically wrong by a factor of N".

## 5. Research questions and verdicts

| RQ | hypothesis | verdict |
|---|---|---|
| **RQ1** | There is systematic underestimation | **refuted** — the aggregate shows overestimation (×0.80 / ×0.86 / ×0.95) |
| **RQ2** | Larger tasks are underestimated more | **refuted, opposite sign** — regression to the mean |
| **RQ3** | Accuracy differs between people | **supported in CESAW** (R² 0.103), but not in SiP (0.026) |
| **RQ4** | Project explains part of the variation | supported, small effect (R² 0.030) |
| **RQ5** | Type of work matters | supported, small effect (R² 0.040) |
| **RQ6** | Interruptions relate to the error | **supported, strong effect** (×~2 with size controlled) |
| **RQ7** | Round estimates are less accurate | **not supported** — the raw effect is explained by task size |
| **RQ8** | Underestimation can be predicted | **yes, but not from the estimate** — ROC-AUC 0.49 → 0.71 once context is added |
| **RQ9** | The magnitude of the error can be predicted | partly — a 10% gain in RMSE_log over the baseline |

## 6. Methodology rules

These were fixed **before** modelling and held across every experiment.

### 6.1 Nothing is dropped silently

Every decision to exclude rows is recorded in `CLEANING_LOG` and written to
`results/01_cleaning_log.csv`.

### 6.2 Understanding precedes modelling

The order is strict:

```
understand the structure → clean → build task-level → EDA → hypotheses
    → statistics → features → baseline → ML → evaluation → explainability → conclusions
```

Jumping from raw data straight to a model is not allowed. Half the findings in this
project came from the first three steps.

### 6.3 Leakage control

A feature may only be something known **at the moment the estimate is given**.

Excluded:
- `task_actual_time_minutes`, `HoursActual` — the target;
- `task_actual_complete_date`, `CompletedOn` — known afterwards;
- `n_sessions`, `interrupt_min` — known afterwards (even though this is the strongest
  factor in RQ6);
- `DeveloperHoursActual`, `TaskPerformance`, `DeveloperPerformance` in SiP — direct
  functions of the target.

Historical features ("how wrong was this person before") use an **expanding mean shifted
by one step** in chronological order: for task #500 only tasks #1–499 are used. The
absence of leakage is checked automatically (`assert`): the first task of every person
must have `NaN` history.

### 6.4 The split strategy is part of the result

The data contains repeated observations of the same people and projects, so a random split
gives an optimistic result. All three strategies are computed:

| | what it tests | realism |
|---|---|---|
| A. Random | basic learnability | optimistic |
| B. By project | transfer to a new project | strict generalisation test |
| C. Temporal | predicting the future from the past | **matches production** |

Headline results are reported under **C**. The spread between strategies
(ROC-AUC 0.775 / 0.738 / 0.709) is itself a finding.

### 6.5 A baseline is mandatory

`actual = estimate` — "just trust the developer". Free, always available and strong. A
model that fails to beat it is declared useless rather than tuned until it wins.

### 6.6 Logging artefacts are isolated

In 8% of CESAW tasks, 34% of SiP and 44% of Renzo the actual equals the estimate
**exactly**. This is time logged against the plan, not accuracy: the share falls from ~60%
on small tasks to ~1% on large ones. Key results are computed twice — on all data and on
the subset with exact matches removed.

### 6.7 Significance ≠ effect size

At 55,000 observations almost everything is significant. So wherever a p-value appears, an
effect size appears beside it. For categorical factors with many levels (`person` has 246)
an adjusted and an **out-of-sample** R² are computed as well.

### 6.8 Confounders are checked

No group comparison is accepted without controlling for task size — it correlates with
almost everything. That is exactly how hypothesis RQ7 fell apart.

## 7. What this project does NOT claim

- **Causality.** All the data is observational. The association between interruptions and
  overrun is established; the direction is not.
- **Absence of a self-fulfilling prophecy.** A developer may have paced the work to the
  quoted deadline. Inseparable without an experiment.
- **Sample completeness.** Unfinished and cancelled tasks are absent from the datasets →
  survivorship bias, likely biasing the conclusions optimistic.
- **Transferable coefficients.** CESAW is a formal TSP process with some safety-critical
  projects. The direction of the effects replicates; their magnitude does not.

## 8. Stack

`Python` · `pandas` · `numpy` · `scipy` · `scikit-learn` (`HistGradientBoosting`,
`RandomForest`, `LogisticRegression`, quantile regression, permutation importance) ·
`shap` · `matplotlib`

## 9. Departures from the original plan

The original frame was designed as a **mentoring format**: the assistant sets tasks and
the analyst writes the code. Here the work was carried out in full at the client's
explicit request — a deliberate departure, not an oversight.

Second departure: instead of eight separate notebooks there is **one end-to-end notebook**.
The reason is the requirement for "a file that can be opened in Google Colab" — a single
notebook runs with one button and needs no intermediate artifacts passed between files.
The phases of the original plan are preserved as notebook sections.

Third: `wbs_parent.csv` (the Work Breakdown Structure hierarchy) is unused — left as a
direction for further work.
