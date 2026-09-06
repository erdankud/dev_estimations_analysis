# %% [markdown]
# # Predicting and Explaining Software Development Estimation Error
#
# **Why developers get effort estimates wrong — and whether the error can be predicted**
#
# Data: [Derek Jones — Software estimation datasets](https://github.com/Derek-Jones/Software-estimation-datasets)
#
# ---
#
# ## Central question
#
# > **Can we predict that a task will be underestimated, and what factors explain
# > the estimation error?**
#
# Note the phrasing. This is **not** "predict `actual_hours`" — that would be an
# ordinary regression exercise. The interesting question is why `Actual ≠ Estimate`,
# and whether that error carries predictable structure or is pure noise.
#
# ## Datasets
#
# | | Tasks | Units | Period | Role |
# |---|---:|---|---|---|
# | **CESAW** | 60,284 | minutes → hours | 2008–2017 | **Primary.** 247 people, 45 projects, TSP process. Has interruption data. |
# | **SiP** | 10,266 | hours | 2004–2014 | Validation on a different organisation. 22 developers, 20 projects. |
# | **Renzo Pomodoro** | 10,159 | pomodoros | 2009–2019 | Validation at the individual level. One person, 10 years. |
#
# CESAW is the primary set: 61,817 tasks across 45 projects is enough for
# non-toy ML, and unlike the PROMISE-family datasets it carries a genuine
# **estimate → actual** pair per *task* rather than per project.
#
# ## Target variables
#
# Absolute error `actual − estimate` is useless on its own: two hours of error on a
# two-hour task and on a 200-hour task are entirely different events. So we work
# with three related quantities:
#
# | | definition | meaning |
# |---|---|---|
# | `estimation_ratio` | `actual / estimate` | > 1 underestimate, < 1 overestimate |
# | `log_ratio` | `log(actual / estimate)` | same, but symmetric and additive |
# | `underestimated` | `1 if actual > estimate` | classification target |
#
# **Why logarithms.** The ratio is strictly positive and heavily skewed. In log space
# "took twice as long" (+0.69) and "took half as long" (−0.69) are symmetric, the mean
# of `log_ratio` is the log of the **geometric** mean (a robust measure of bias), and
# `exp(MAE_log)` reads directly as "typically wrong by a factor of N".
#
# ## Order of work
#
# ```
# Phase 0  Understand the data      →  DATA_DICTIONARY.md
# Phase 1  Cleaning                 →  decision log
# Phase 2  Build task-level dataset →  one row = one task
# Phase 3  EDA
# Phase 4  Hypotheses RQ1–RQ7       →  statistical tests
# Phase 5  Feature engineering      →  leak-free historical features
# Phase 6  Baseline + split strategies
# Phase 7  ML: regression + classification
# Phase 8  Explainability + error analysis
# Phase 9  Intervals + personal calibration
# ```
#
# No model is built before the data is understood. That is not a formality: half the
# findings in this project came out of Phases 0–2.

# %%
# --- Setup and data download (Google Colab) ---------------------------------
import os, sys, subprocess, pathlib

IN_COLAB = "google.colab" in sys.modules
DATA_REPO = "https://github.com/Derek-Jones/Software-estimation-datasets"
DATA_DIR = pathlib.Path("Software-estimation-datasets" if IN_COLAB else
                        os.environ.get("SED_DATA_DIR", "data/Software-estimation-datasets"))

if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "pandas", "numpy", "scikit-learn", "matplotlib", "scipy", "shap"], check=False)

if not DATA_DIR.exists():
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning datasets into {DATA_DIR} ...")
    subprocess.run(["git", "clone", "--depth", "1", DATA_REPO, str(DATA_DIR)], check=True)

print("Data:", DATA_DIR.resolve())
print(sorted(p.name for p in DATA_DIR.iterdir() if not p.name.startswith(".")))

# %%
# --- Imports and shared settings --------------------------------------------
import tarfile, warnings, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 50)

FIG_DIR = pathlib.Path("reports/figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = pathlib.Path("results"); RES_DIR.mkdir(exist_ok=True)

INK, MUTED, GRID = "#1b1b1f", "#6b6b76", "#e3e3e8"
ACCENT, ACCENT2, ACCENT3, ACCENT4 = "#2b6cb0", "#c05621", "#2f855a", "#6b46c1"
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": .7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

def save(fig, name):
    p = FIG_DIR / f"{name}.png"; fig.savefig(p); print("saved", p); return p

def dump(df, name):
    p = RES_DIR / f"{name}.csv"; df.to_csv(p); print("saved", p); return p

RNG = 0
CLEANING_LOG = []      # no row is ever dropped silently

def drop(df, mask, reason):
    """Filter rows out, recording the decision in the cleaning log."""
    n = int(mask.sum())
    CLEANING_LOG.append({"dataset": df.attrs.get("name", "?"), "decision": reason,
                         "rows affected": n, "rows left": len(df) - n})
    return df[~mask]

# %% [markdown]
# ---
# # Phase 0 — Understanding the data
#
# The first task is not a model but an answer to **"what does one row represent?"**
# In CESAW that answer is not obvious, and getting it wrong costs more than any
# hyperparameter choice.
#
# The `CESAW.tgz` archive holds two fact tables:
#
# * `CESAW_task_fact.csv` — **61,817 rows**, one per task: planned and actual minutes,
#   person, project, team, process phase, start and completion timestamps;
# * `CESAW_time_fact.csv` — **203,621 rows**, one per *work session*: start, end,
#   duration and **interruption minutes**.
#
# So the advertised "203,621 observations" are not tasks — they are individual timer
# entries. One task averages 3.3 sessions, with a maximum of 439.
#
# The link between the tables is undocumented and had to be recovered. Key search:
#
# | key | groups in `task_fact` |
# |---|---:|
# | `(project, wbs, plan_item)` | 51,582 of 61,817 ← **not unique** |
# | `(project, wbs, plan_item, phase)` | 51,582 ← still not |
# | `(project, wbs, plan_item, phase, person)` | **61,817 = every row** ✓ |
#
# So the natural unit of observation is **a task performed by one person in one
# process phase**. Aggregating `time_fact` by that same key reproduces
# `task_actual_time_minutes` to the minute for **99.3%** of tasks — which is what
# proves the key was recovered rather than guessed.
#
# ### Data lineage
#
# ```
# CESAW.tgz
#     ├── CESAW_task_fact.csv   (61,817 tasks: plan, actual, who, where, when)
#     └── CESAW_time_fact.csv   (203,621 sessions: duration, interruptions)
#                    │
#                    ▼  aggregate by (project, wbs, plan_item, phase, person)
#             sessions → per task: n sessions, interruption minutes
#                    │
#                    ▼  join back into task_fact  (99.3% actual reconciled)
#             task-level dataset
#                    │
#                    ▼  cleaning (Phase 1) + derived quantities
#             analysis dataset  ──► EDA / hypotheses
#                    │
#                    ▼  leak-free historical features (Phase 5)
#             ML dataset
# ```
#
# **Critical for Phase 5:** session count and interruption minutes are only known
# *after* a task is done. They are fair game for explanation (RQ6) but **cannot be
# features** of a model that predicts the outcome before work starts. See "Leakage".

# %%
def load_cesaw(data_dir=DATA_DIR, work_dir=pathlib.Path("data/_cesaw")):
    """CESAW: assemble a task-level dataset from the two fact tables."""
    work_dir.mkdir(parents=True, exist_ok=True)
    task_path = work_dir / "data" / "CESAW_task_fact.csv.xz"
    time_path = work_dir / "data" / "CESAW_time_fact.csv.xz"
    if not task_path.exists():
        with tarfile.open(data_dir / "CESAW.tgz") as t:
            t.extractall(work_dir)

    tasks = pd.read_csv(task_path)
    times = pd.read_csv(time_path)
    print(f"task_fact: {tasks.shape}   time_fact: {times.shape}")

    # phase_key arrives as text in task_fact (it contains the literal '\N') -> to numeric.
    tasks["phase_key"] = pd.to_numeric(tasks.phase_key, errors="coerce")
    KEY = ["project_key", "wbs_element_key", "plan_item_key", "phase_key", "person_key"]

    for k in [KEY[:3], KEY[:4], KEY]:
        print(f"  key {tuple(x.replace('_key','') for x in k)}: "
              f"{tasks.groupby(k, dropna=False).ngroups} groups out of {len(tasks)} rows")

    sessions = times.groupby(KEY).agg(
        n_sessions=("time_log_fact_key", "size"),
        interrupt_min=("time_log_interrupt_minutes", "sum"),
        logged_min=("time_log_delta_minutes", "sum"),
    ).reset_index()

    df = tasks.merge(sessions, on=KEY, how="left")
    reconciled = np.isclose(df.task_actual_time_minutes, df.logged_min.fillna(-1)).mean()
    print(f"  sessions matched for {df.n_sessions.notna().mean():.1%} of tasks; "
          f"sum(sessions) == task_actual for {reconciled:.1%} -> key correctly recovered")

    df.attrs["name"] = "CESAW"
    return df


cesaw_raw = load_cesaw()

# %% [markdown]
# ---
# # Phase 1 — Cleaning
#
# The rule: **no row is dropped silently.** Every decision is recorded in
# `CLEANING_LOG` and printed at the end of the phase.
#
# Questions checked (from the Phase 1 checklist):
#
# * can the plan be zero? — **yes, 692 CESAW tasks**. These were never estimated, so
#   `actual/estimate` is undefined for them;
# * can the actual be zero or negative? — no;
# * can the start date be missing? — yes, in isolated cases;
# * can one task have several people on it? — **yes**, and in SiP that produces
#   duplicate rows (see below);
# * can one task have several time records? — yes, that is what `time_fact` is.

# %%
def add_targets(df):
    """Derived quantities: three forms of estimation error."""
    df = df.copy()
    df["ratio"] = df.actual / df.estimate               # estimation_ratio
    df["log_ratio"] = np.log(df.ratio)
    df["log_est"] = np.log(df.estimate)
    df["log_act"] = np.log(df.actual)
    df["abs_error"] = df.actual - df.estimate           # absolute error
    df["rel_error"] = df.abs_error / df.estimate        # relative error
    df["underestimated"] = (df.actual > df.estimate).astype(int)
    df["exact"] = np.isclose(df.actual, df.estimate)
    return df


def clean_cesaw(raw):
    df = raw.copy(); df.attrs["name"] = "CESAW"
    df = drop(df, df.task_plan_time_minutes <= 0,
              "plan <= 0 (task was never estimated) — ratio undefined")
    df = drop(df, df.task_actual_time_minutes <= 0, "actual <= 0 — impossible value")
    df["date"] = pd.to_datetime(df.task_actual_start_date, errors="coerce")
    df = drop(df, df.date.isna(), "start date could not be parsed")
    df["estimate"] = df.task_plan_time_minutes / 60.0     # everything in hours
    df["actual"] = df.task_actual_time_minutes / 60.0
    df = df.rename(columns={"person_key": "person", "project_key": "project"})
    df["dataset"] = "CESAW"
    return add_targets(df).sort_values("date").reset_index(drop=True)


def load_sip(data_dir=DATA_DIR):
    """SiP: 10k tasks from a commercial company. The trap here is duplicate rows."""
    # The file contains cp1252 characters (typographic quotes) -> utf-8 fails.
    tasks = pd.read_csv(data_dir / "SiP" / "Sip-task-info.csv", encoding="latin-1")
    dates = pd.read_csv(data_dir / "SiP" / "est-act-dates.csv")
    tasks.attrs["name"] = "SiP"

    n_rows, n_tasks = len(tasks), tasks.TaskNumber.nunique()
    per_task = tasks.groupby("TaskNumber")
    print(f"SiP: {n_rows} rows, but {n_tasks} unique tasks")
    print(f"  estimate/actual differ within a task: "
          f"{int((per_task.HoursEstimate.nunique() > 1).sum())} tasks -> values are duplicated")
    print(f"  sum(DeveloperHoursActual) == HoursActual: "
          f"{np.isclose(per_task.DeveloperHoursActual.sum(), per_task.HoursActual.first()).mean():.1%} "
          f"-> HoursActual is the task total, not the person's share")

    tasks = drop(tasks, tasks.TaskNumber.duplicated(),
                 "duplicate rows: a multi-person task is stored one row per developer, "
                 "with estimate and actual repeated in each")
    dates = dates.drop_duplicates("TaskNumber")
    dates["EstimateOn"] = pd.to_datetime(dates.EstimateOn, format="%d-%b-%y", errors="coerce")

    df = tasks.merge(dates[["TaskNumber", "EstimateOn"]], on="TaskNumber", how="left")
    df.attrs["name"] = "SiP"
    df = drop(df, (df.HoursEstimate <= 0) | (df.HoursActual <= 0), "zero estimate or actual")
    df = drop(df, df.EstimateOn.isna(), "no estimation date")
    df = df.rename(columns={"HoursEstimate": "estimate", "HoursActual": "actual",
                            "EstimateOn": "date", "DeveloperID": "person",
                            "ProjectCode": "project"})
    df["dataset"] = "SiP"
    return add_targets(df).sort_values("date").reset_index(drop=True)


def load_renzo(data_dir=DATA_DIR):
    """Renzo Pomodoro: one person, 10 years, estimate and actual in pomodoros."""
    df = pd.read_csv(data_dir / "renzo-pomodoro.csv"); df.attrs["name"] = "Renzo"
    df = drop(df, df.estimate.isna() | df.actual.isna(), "estimate or actual missing")
    df = drop(df, (df.estimate <= 0) | (df.actual <= 0), "zero estimate or actual")
    df = drop(df, df.estimate > 40,
              "estimate > 40 pomodoros (raw file holds values up to 5e7) — data entry junk")
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    df = drop(df, df.date.isna(), "date could not be parsed")
    df["person"] = "renzo"; df["project"] = "renzo"; df["dataset"] = "Renzo"
    return add_targets(df).sort_values("date").reset_index(drop=True)


cesaw = clean_cesaw(cesaw_raw)
sip = load_sip()
renzo = load_renzo()

cleaning = pd.DataFrame(CLEANING_LOG)
dump(cleaning, "01_cleaning_log")
print("\nCLEANING LOG (no row removed silently):")
print(cleaning.to_string(index=False))
print("\nFinal sizes:", {k: len(v) for k, v in
                         {"CESAW": cesaw, "SiP": sip, "Renzo": renzo}.items()})

# %% [markdown]
# ---
# # Phase 3 — Exploratory analysis
#
# ## 3.1 Dataset overview

# %%
DATASETS = [("CESAW", cesaw), ("SiP", sip), ("Renzo", renzo)]

overview = pd.DataFrame({
    name: {
        "tasks": len(d),
        "period": f"{d.date.min():%Y}–{d.date.max():%Y}",
        "people": d.person.nunique(),
        "projects": d.project.nunique(),
        "median estimate": round(d.estimate.median(), 2),
        "median actual": round(d.actual.median(), 2),
        "median ratio": round(d.ratio.median(), 3),
        "geometric mean ratio": round(float(np.exp(d.log_ratio.mean())), 3),
        "SD log_ratio": round(float(d.log_ratio.std()), 3),
        "share underestimated": f"{d.underestimated.mean():.1%}",
        "share actual == estimate": f"{d.exact.mean():.1%}",
    } for name, d in DATASETS
})
dump(overview, "02_overview")
print(overview.to_string())

# %%
# Missing values in the raw CESAW table — a mandatory EDA checklist figure.
miss = cesaw_raw.isna().mean().sort_values(ascending=False)
miss = miss[miss > 0]
fig, ax = plt.subplots(figsize=(7, 3))
if len(miss):
    ax.barh(np.arange(len(miss)), miss.values, color=ACCENT2)
    ax.set_yticks(np.arange(len(miss))); ax.set_yticklabels(miss.index, fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.set_xlabel("share missing")
else:
    ax.text(.5, .5, "No missing values", ha="center", va="center", fontsize=13, color=MUTED)
    ax.set_axis_off()
ax.set_title("CESAW: missing values in the raw task table")
save(fig, "01_missing_values"); plt.close(fig)
print("Missing share by column:\n", (miss * 100).round(2).to_string() if len(miss) else "none")

# %% [markdown]
# ## 3.2 Distributions: why logarithms are not optional
#
# Estimates and actuals are approximately log-normal and span four to five orders of
# magnitude. A linear-scale mean is driven by a dozen giant tasks and describes nothing.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
for ax, (name, d) in zip(axes, DATASETS):
    lo = max(d.estimate.min(), 1e-2); hi = d[["estimate", "actual"]].max().max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 45)
    ax.hist(d.estimate, bins=bins, alpha=.65, color=ACCENT, label="estimate")
    ax.hist(d.actual, bins=bins, alpha=.55, color=ACCENT2, label="actual")
    ax.set_xscale("log")
    ax.set_title(f"{name}  (N={len(d):,})")
    ax.set_xlabel("hours" if name != "Renzo" else "pomodoros")
axes[0].set_ylabel("tasks"); axes[0].legend()
fig.suptitle("Estimates and actuals are log-normal — the whole analysis runs in log space",
             y=1.04, fontsize=13, fontweight="bold")
save(fig, "02_distributions"); plt.close(fig)

# %%
# The key chart: actual against estimate. The diagonal is a perfect estimate.
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
for ax, (name, d) in zip(axes, [("CESAW", cesaw), ("SiP", sip)]):
    sub = d.sample(min(len(d), 8000), random_state=RNG)
    ax.scatter(sub.estimate, sub.actual, s=6, alpha=.12, color=ACCENT, edgecolors="none")
    lim = [d.estimate.min() * .7, d.actual.max() * 1.3]
    ax.plot(lim, lim, color=INK, lw=1.3, ls="--", label="actual = estimate")
    q = pd.qcut(d.estimate.rank(method="first"), 8, labels=False)
    med = d.groupby(q).agg(e=("estimate", "median"), a=("actual", "median"))
    ax.plot(med.e, med.a, "o-", color=ACCENT2, lw=2.5, ms=7, label="median actual by group")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("estimate, hours"); ax.set_ylabel("actual, hours")
    ax.set_title(f"{name}: the median line is flatter than the diagonal")
    ax.legend(loc="upper left", fontsize=9)
fig.suptitle("Above the diagonal is underestimation, below is overestimation. "
             "A median slope < 1 is regression to the mean",
             y=1.01, fontsize=12.5, fontweight="bold")
save(fig, "03_estimate_vs_actual"); plt.close(fig)

# %% [markdown]
# ## 3.3 Distribution of the estimation ratio
#
# The median is more informative than the mean: the distribution is skewed enough that
# the arithmetic mean of `actual/estimate` in CESAW is 1.49 while the median is 0.94.
# The first number describes the tail, the second describes a typical task.

# %%
ratio_stats = pd.DataFrame({
    name: {
        "mean (arithmetic)": round(d.ratio.mean(), 3),
        "geometric mean": round(float(np.exp(d.log_ratio.mean())), 3),
        "median": round(d.ratio.median(), 3),
        "p05": round(d.ratio.quantile(.05), 3), "p25": round(d.ratio.quantile(.25), 3),
        "p75": round(d.ratio.quantile(.75), 3), "p90": round(d.ratio.quantile(.90), 3),
        "p95": round(d.ratio.quantile(.95), 3), "max": round(d.ratio.max(), 1),
    } for name, d in DATASETS
})
dump(ratio_stats, "03_ratio_stats")
print(ratio_stats.to_string())

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
for ax, (name, d) in zip(axes, DATASETS):
    lr = d.log_ratio.clip(-3, 3)
    ax.hist(lr[~d.exact], bins=70, color=ACCENT, alpha=.85, label="actual ≠ estimate")
    ax.hist(lr[d.exact], bins=70, color=ACCENT2, alpha=.95,
            label=f"actual == estimate ({d.exact.mean():.0%})")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xticks(np.log([1/8, 1/4, 1/2, 1, 2, 4, 8]))
    ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.set_xlabel("actual / estimate (log scale)")
    ax.set_title(name); ax.legend(fontsize=8)
axes[0].set_ylabel("tasks")
fig.suptitle("Estimation error: wide, skewed, and with an artificial spike at 1.0",
             y=1.04, fontsize=13, fontweight="bold")
save(fig, "04_ratio_distribution"); plt.close(fig)

# %% [markdown]
# ---
# # Phase 4 — Testing the hypotheses (RQ1–RQ7)
#
# ## First, the trap that corrupts every metric
#
# In **8% of CESAW tasks, 34% of SiP and 44% of Renzo**, the actual equals the estimate
# exactly. These are not perfect estimates: the share of exact matches falls sharply
# with task size — logging time "as planned" is easy on a half-hour task and impossible
# on a two-week one. This is a time-logging practice, not accuracy.
#
# Why it matters for ML: the "actual = estimate" baseline gets zero error on those rows
# for free, which makes it nearly unbeatable on median metrics. So every key result is
# computed **twice** — on all data and on the subset with exact matches removed.
# The effect is weakest in CESAW, which is part of why it makes a more honest primary
# dataset than SiP.

# %%
SIZE_BINS = [0, .5, 1, 2, 4, 8, 16, np.inf]
SIZE_LABELS = ["≤0.5h", "0.5–1h", "1–2h", "2–4h", "4–8h", "8–16h", ">16h"]
for d in (cesaw, sip):
    d["size_bin"] = pd.cut(d.estimate, SIZE_BINS, labels=SIZE_LABELS)

exact_by_size = pd.DataFrame({
    name: d.groupby("size_bin", observed=True).exact.mean().round(3)
    for name, d in [("CESAW", cesaw), ("SiP", sip)]})
dump(exact_by_size, "04_exact_by_size")
print("Share of tasks with actual == estimate, by size:\n", exact_by_size.to_string())

fig, ax = plt.subplots(figsize=(7.5, 3.8))
x = np.arange(len(exact_by_size))
ax.bar(x - .2, exact_by_size.CESAW, .4, color=ACCENT, label="CESAW")
ax.bar(x + .2, exact_by_size.SiP, .4, color=ACCENT2, label="SiP")
ax.set_xticks(x); ax.set_xticklabels(exact_by_size.index, fontsize=9)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylabel("share with actual == estimate"); ax.set_xlabel("task size by estimate")
ax.set_title("\"Perfect estimates\" are a logging artefact: they vanish on large tasks")
ax.legend()
save(fig, "05_exact_match_artifact"); plt.close(fig)

# From here on, *_ne = subsets with exact matches removed ("honest" data).
cesaw_ne, sip_ne, renzo_ne = cesaw[~cesaw.exact], sip[~sip.exact], renzo[~renzo.exact]

# %% [markdown]
# ## RQ1 — Is there a systematic bias?
#
# Hypothesis: developers systematically underestimate tasks.
#
# Tested with a sign test (share of `ratio > 1` against 50%) and a t-test of `log_ratio`
# against zero — the latter is equivalent to testing "geometric mean = 1".

# %%
rq1 = {}
for name, d in [("CESAW", cesaw_ne), ("SiP", sip_ne), ("Renzo", renzo_ne)]:
    n_under = int(d.underestimated.sum()); n = len(d)
    binom = stats.binomtest(n_under, n, .5)
    t = stats.ttest_1samp(d.log_ratio, 0)
    rq1[name] = {
        "N": n,
        "share underestimated": round(n_under / n, 3),
        "p (sign test vs 50%)": f"{binom.pvalue:.2e}",
        "geometric mean ratio": round(float(np.exp(d.log_ratio.mean())), 3),
        "p (log_ratio vs 0)": f"{t.pvalue:.2e}",
        "verdict": ("underestimation" if d.log_ratio.mean() > 0 else "OVERestimation"),
    }
rq1_tbl = pd.DataFrame(rq1)
dump(rq1_tbl, "05_rq1_systematic_bias")
print(rq1_tbl.to_string())

# %% [markdown]
# **RQ1 answer: the hypothesis fails — in aggregate the bias runs the other way.**
# In all three datasets the geometric mean of `actual/estimate` is **below one**
# (CESAW ×0.80, SiP ×0.86, Renzo ×0.95), and only 42–49% of tasks exceed their estimate.
#
# One discrepancy worth noting on Renzo: the sign test does not reject 50%
# (`p = 0.20`) while the t-test on `log_ratio` does (`p = 8e−7`). There is no
# contradiction — tasks exceed their estimate about half the time, but the overruns are
# smaller in magnitude than the underruns. One person over ten years turns out to be
# calibrated in frequency but not in magnitude.
#
# ## RQ2 — Does the error depend on task size?
#
# Hypothesis: the larger the task, the stronger the underestimation.

# %%
def size_profile(d):
    g = d.groupby("size_bin", observed=True)
    return pd.DataFrame({
        "N": g.size(),
        "geometric mean ratio": g.log_ratio.apply(lambda x: np.exp(x.mean())),
        "median ratio": g.ratio.median(),
        "share underestimated": g.underestimated.mean(),
        "SD log_ratio": g.log_ratio.std(),
    }).round(3)

rq2 = {name: size_profile(d) for name, d in [("CESAW", cesaw_ne), ("SiP", sip_ne)]}
rq2_tbl = pd.concat(rq2, names=["dataset"])
dump(rq2_tbl, "06_rq2_size_effect")
print(rq2_tbl.to_string())

# Formal test: slope of log_ratio ~ log_est. Negative = regression to the mean.
slopes = {}
for name, d in [("CESAW", cesaw_ne), ("SiP", sip_ne), ("Renzo", renzo_ne)]:
    r = stats.linregress(d.log_est, d.log_ratio)
    slopes[name] = {"slope": round(r.slope, 3), "SE": round(r.stderr, 4),
                    "p-value": f"{r.pvalue:.1e}", "R²": round(r.rvalue ** 2, 3), "N": len(d)}
slopes_tbl = pd.DataFrame(slopes).T
dump(slopes_tbl, "06_rq2_slopes")
print("\nSlope of log_ratio ~ log_est (negative = regression to the mean):")
print(slopes_tbl.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ax = axes[0]
x = np.arange(len(SIZE_LABELS))
for (name, t), c, m in zip(rq2.items(), [ACCENT, ACCENT2], ["o", "s"]):
    ax.plot(x, t["geometric mean ratio"], m + "-", color=c, lw=2.4, ms=7, label=name)
ax.axhline(1, color=INK, lw=1, ls="--")
ax.set_xticks(x); ax.set_xticklabels(SIZE_LABELS, fontsize=9)
ax.set_ylabel("geometric mean actual / estimate"); ax.set_xlabel("task size by estimate")
ax.set_title("The bias changes sign with task size")
ax.annotate("underestimated", (1.6, 1.22), color=ACCENT2, fontweight="bold", fontsize=9)
ax.annotate("overestimated", (3.4, 0.60), color=ACCENT, fontweight="bold", fontsize=9)
ax.legend()

ax = axes[1]
for (name, t), c, m in zip(rq2.items(), [ACCENT, ACCENT2], ["o", "s"]):
    ax.plot(x, t["SD log_ratio"], m + "-", color=c, lw=2.4, ms=7, label=name)
ax.set_xticks(x); ax.set_xticklabels(SIZE_LABELS, fontsize=9)
ax.set_ylabel("SD log(actual/estimate)"); ax.set_xlabel("task size by estimate")
ax.set_title("Uncertainty grows with size too")
ax.legend()
fig.suptitle("RQ2: the hypothesis \"large tasks are underestimated more\" is REFUTED — "
             "it is the exact opposite", y=1.03, fontsize=12.5, fontweight="bold")
save(fig, "06_size_effect"); plt.close(fig)

# %% [markdown]
# **RQ2 answer: the hypothesis is refuted, with the opposite sign.**
# The slope of `log_ratio ~ log_est` is negative in all three datasets
# (CESAW −0.17, SiP −0.24, Renzo −0.81; all `p < 1e−140`): the larger the task, the
# **smaller** the actual/estimate ratio.
#
# The strength differs. In SiP the bias literally flips sign — from ×1.30 on tasks
# ≤ 0.5h to ×0.51 on tasks > 16h. In CESAW the sign does not flip (there is
# overestimation across the whole range) but the trend is the same: ×0.98 down to ×0.57.
# For the single individual (Renzo) the slope is three to five times steeper than for
# organisations: without organisational averaging, regression to the mean shows most clearly.
#
# ## RQ3–RQ5 — Who is to blame: the person, the project or the type of work?
#
# Rather than eyeballing group means, ask the quantitative question:
# **how much of the variance in `log_ratio` does each factor explain?**
#
# It is easy to fool yourself here. The `person` feature in CESAW has 246 levels, i.e.
# 246 dummy variables, and plain R² grows mechanically with that count. So we compute
# three versions: plain R², an adjusted R², and — most importantly — **out-of-sample R²
# from 5-fold cross-validation**, the only one that cannot be inflated by level count.
# Plus a combined "all factors" row to see the ceiling on explainability.

# %%
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def variance_explained(d, factors, target="log_ratio"):
    """R² from fitting a single categorical factor.

    Plain R² is inflated for factors with many levels (`person` has 246 in CESAW —
    that is 246 dummies), so we also compute an adjusted R² and an out-of-sample R²
    from 5-fold cross-validation. Only the last one cannot be gamed by level count.
    """
    from sklearn.model_selection import cross_val_score, KFold
    y = d[target].values
    n = len(d)
    rows = {}
    for f in list(factors) + ["ALL FACTORS COMBINED"]:
        cols = list(factors) if f == "ALL FACTORS COMBINED" else [f]
        X = OneHotEncoder(handle_unknown="ignore").fit_transform(d[cols].astype(str))
        r2 = LinearRegression().fit(X, y).score(X, y)
        p_ = X.shape[1]
        cv = cross_val_score(Ridge(alpha=1.0), X, y,
                             cv=KFold(5, shuffle=True, random_state=RNG), scoring="r2").mean()
        rows[f] = {"levels": int(sum(d[c].nunique() for c in cols)),
                   "R²": round(r2, 4),
                   "R² adjusted": round(1 - (1 - r2) * (n - 1) / max(n - p_ - 1, 1), 4),
                   "R² out-of-sample": round(float(cv), 4)}
    out = pd.DataFrame(rows).T
    tail = out.loc[["ALL FACTORS COMBINED"]]
    return pd.concat([out.drop(index="ALL FACTORS COMBINED")
                        .sort_values("R² out-of-sample", ascending=False), tail])

ve_cesaw = variance_explained(cesaw_ne, ["size_bin", "person", "project",
                                         "phase_short_name", "team_key", "process_name"])
ve_sip = variance_explained(sip_ne, ["size_bin", "person", "project",
                                     "SubCategory", "Category", "Priority"])
dump(ve_cesaw, "07_variance_explained_cesaw")
dump(ve_sip, "07_variance_explained_sip")
print("CESAW:\n", ve_cesaw.to_string())
print("\nSiP:\n", ve_sip.to_string())

# %%
def bias_table(d, key, min_n=100):
    g = d.groupby(key, observed=True)
    t = pd.DataFrame({"N": g.size(),
                      "geometric mean": g.log_ratio.apply(lambda x: np.exp(x.mean())),
                      "median": g.ratio.median(),
                      "share underestimated": g.underestimated.mean(),
                      "SD log": g.log_ratio.std()})
    t["SE"] = t["SD log"] / np.sqrt(t.N)
    t["CI low"] = np.exp(np.log(t["geometric mean"]) - 1.96 * t.SE)
    t["CI high"] = np.exp(np.log(t["geometric mean"]) + 1.96 * t.SE)
    return t[t.N >= min_n].sort_values("geometric mean")

by_phase = bias_table(cesaw_ne, "phase_short_name", 200)
by_person = bias_table(cesaw_ne, "person", 200)
by_project = bias_table(cesaw_ne, "project", 200)
for t, n in [(by_phase, "08_rq5_by_phase"), (by_person, "08_rq3_by_person"),
             (by_project, "08_rq4_by_project")]:
    dump(t.round(3), n)
print("CESAW, bias by process phase:\n", by_phase.round(2).to_string())

# ANOVA-style check: are the group differences significant at all?
anova = {}
for label, key in [("person (RQ3)", "person"), ("project (RQ4)", "project"),
                   ("process phase (RQ5)", "phase_short_name")]:
    groups = [g.log_ratio.values for _, g in cesaw_ne.groupby(key) if len(g) >= 30]
    f, p = stats.f_oneway(*groups)
    kw = stats.kruskal(*groups)          # non-parametric backup: log_ratio is not normal
    anova[label] = {"groups": len(groups), "F": round(f, 1), "p (ANOVA)": f"{p:.1e}",
                    "p (Kruskal-Wallis)": f"{kw.pvalue:.1e}",
                    "R² out-of-sample": round(float(ve_cesaw.loc[key, "R² out-of-sample"]), 4)}
anova_tbl = pd.DataFrame(anova).T
dump(anova_tbl, "08_rq345_anova")
print("\nGroup differences are significant but explain little:")
print(anova_tbl.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
for ax, (t, title) in zip(axes, [(by_phase, "by process phase (RQ5)"),
                                 (by_person.tail(20), "by person (RQ3), top 20")]):
    y = np.arange(len(t))
    colors = [ACCENT2 if v > 1 else ACCENT for v in t["geometric mean"]]
    ax.hlines(y, t["CI low"], t["CI high"], color=colors, lw=2, alpha=.5)
    ax.scatter(t["geometric mean"], y, color=colors, s=30, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([str(i) for i in t.index], fontsize=8)
    ax.axvline(1, color=INK, lw=1, ls="--")
    ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["×0.25", "×0.5", "×1", "×2", "×4"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_title(f"Estimation bias {title}")
    ax.set_xlabel("geometric mean actual / estimate (95% CI)")
fig.suptitle("CESAW: differences are significant, but all factors combined explain only 17%",
             y=1.02, fontsize=12.5, fontweight="bold")
save(fig, "07_bias_forest"); plt.close(fig)

# %% [markdown]
# **RQ3–RQ5 answer.** All three factors produce statistically significant differences
# (`p < 1e−50` under both ANOVA and Kruskal-Wallis), but at 55,000 observations almost
# everything is significant — what matters is effect size.
#
# | factor | CESAW, R² out-of-sample | SiP, R² out-of-sample |
# |---|---:|---:|
# | person | **0.103** | 0.026 |
# | phase / type of work | 0.040 | 0.023 |
# | project | 0.030 | 0.012 |
# | task size | 0.021 | **0.080** |
# | **all factors combined** | **0.170** | **0.137** |
#
# Two conclusions, both important.
#
# **First: the dominant factor differs between organisations.** In CESAW the person
# matters most (10% of variance; the effect survives both the level-count adjustment
# and cross-validation, so it is real rather than an artefact of 246 dummies). In SiP
# it is the reverse: the person explains almost nothing while task size explains 8%.
# "What matters here is who estimates" cannot be carried from one company to another —
# it is a property of the organisation, not of software development.
#
# **Second, and more important: the ceiling is low.** All observed factors together
# explain 17% of the variance in CESAW and 14% in SiP. That is, **83–86% of the variance
# in estimation error is explained by nothing recorded in the data.**
#
# This is the sobering headline result, and Phase 9 follows directly from it: if five
# sixths of the variance is irreducible, a point prediction is doomed regardless of the
# model, and the right product is an interval.
#
# ## RQ6 — Do interruptions matter?
#
# Care is needed here. Interruptions are known only **after** a task is done, so this is
# a question of *explanation*, not prediction, and they will not enter the model's
# features (see Phase 5). Also, longer tasks get interrupted more simply because they
# are longer — so the comparison must be made **within size bands**.

# %%
c = cesaw_ne[cesaw_ne.n_sessions.notna()].copy()
c["interrupted"] = np.where(c.interrupt_min.fillna(0) > 0, "interrupted", "uninterrupted")
c["sess_bin"] = pd.cut(c.n_sessions, [0, 1, 2, 4, 8, 16, 1e9],
                       labels=["1", "2", "3–4", "5–8", "9–16", ">16"])

rq6_raw = c.groupby("interrupted").agg(
    N=("log_ratio", "size"), geo_mean=("log_ratio", lambda x: np.exp(x.mean())),
    median_estimate=("estimate", "median"), under_rate=("underestimated", "mean")).round(3)
rq6_ctrl = c.groupby(["size_bin", "interrupted"], observed=True).agg(
    N=("log_ratio", "size"), geo_mean=("log_ratio", lambda x: np.exp(x.mean()))).round(3)
rq6_sess = c.groupby("sess_bin", observed=True).agg(
    N=("log_ratio", "size"), geo_mean=("log_ratio", lambda x: np.exp(x.mean())),
    median_estimate=("estimate", "median")).round(3)
for t, n in [(rq6_raw, "09_rq6_interrupt_raw"), (rq6_ctrl, "09_rq6_interrupt_by_size"),
             (rq6_sess, "09_rq6_sessions")]:
    dump(t, n)
print("Raw comparison (careful — task size is not controlled):\n", rq6_raw.to_string())
print("\nWithin size bands:\n", rq6_ctrl.to_string())
print("\nBy number of work sessions:\n", rq6_sess.to_string())

piv = rq6_ctrl.geo_mean.unstack()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
x = np.arange(len(piv))
ax.bar(x - .2, piv["uninterrupted"], .4, color=ACCENT, label="uninterrupted")
ax.bar(x + .2, piv["interrupted"], .4, color=ACCENT2, label="interrupted")
ax.axhline(1, color=INK, lw=1, ls="--")
ax.set_xticks(x); ax.set_xticklabels(piv.index, fontsize=9)
ax.set_ylabel("geometric mean actual / estimate"); ax.set_xlabel("task size by estimate")
ax.set_title("The interruption effect holds within every size band")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(np.arange(len(rq6_sess)), rq6_sess.geo_mean, "o-", color=ACCENT4, lw=2.4, ms=7)
ax.axhline(1, color=INK, lw=1, ls="--")
ax.set_xticks(np.arange(len(rq6_sess))); ax.set_xticklabels(rq6_sess.index)
ax.set_ylabel("geometric mean actual / estimate")
ax.set_xlabel("work sessions per task")
ax.set_title("The more fragmented the work, the larger the overrun")
fig.suptitle("RQ6: interruptions are the strongest factor found — "
             "but they are only known after the fact", y=1.03, fontsize=12.5, fontweight="bold")
save(fig, "08_interruptions"); plt.close(fig)

# %% [markdown]
# **RQ6 answer.** Within **every** size band, interrupted tasks overrun roughly twice as
# much as uninterrupted ones (e.g. in the >16h band: ×0.99 against ×0.55). The effect
# does not reduce to size, and it is larger than anything RQ3–RQ5 produced.
#
# But the conclusions must stay careful. Causal direction is not established:
# interruptions may lengthen a task — or they may be a *consequence* of a task turning
# out harder than expected and stretching over many days. The data cannot separate those.
# And for prediction it is useless anyway: at estimation time we do not know how many
# interruptions there will be.
#
# ## RQ7 — Do round-number estimates matter?
#
# The original CESAW work flags a tendency toward round numbers as a factor related to
# accuracy. Let us check it — and show along the way why such effects cannot be taken
# at face value.
#
# First, what counts as "round": that cannot be defined before looking at the distribution.

# %%
top_est = cesaw.task_plan_time_minutes.value_counts().head(15)
print("Most frequent CESAW estimates (minutes):\n", top_est.to_string())
print(f"\nTop 15 values cover {top_est.sum() / len(cesaw):.1%} of all estimates")

def multiple_of(x, m, tol=1e-6):
    r = (np.asarray(x, dtype=float) / m) % 1
    return np.isclose(r, 0, atol=tol) | np.isclose(r, 1, atol=tol)

for d, col, unit in [(cesaw, "task_plan_time_minutes", "min"), (sip, "estimate", "h")]:
    d["is_round"] = ((d[col] % 30 == 0) | (d[col] % 60 == 0)).astype(int) if unit == "min" \
        else multiple_of(d[col], 1).astype(int)
cesaw_ne, sip_ne = cesaw[~cesaw.exact], sip[~sip.exact]

# Granularity: estimates are coarser than actuals — direct evidence of anchoring.
gran = pd.DataFrame({
    "estimates": {"unique values": cesaw.task_plan_time_minutes.nunique(),
                  "top 10 cover": f"{cesaw.task_plan_time_minutes.value_counts(normalize=True).head(10).sum():.1%}",
                  "multiple of 30 or 60 min": f"{cesaw.is_round.mean():.1%}"},
    "actuals": {"unique values": cesaw.task_actual_time_minutes.nunique(),
                "top 10 cover": f"{cesaw.task_actual_time_minutes.value_counts(normalize=True).head(10).sum():.1%}",
                "multiple of 30 or 60 min": f"{((cesaw.task_actual_time_minutes % 30 == 0) | (cesaw.task_actual_time_minutes % 60 == 0)).mean():.1%}"},
})
dump(gran, "10_rq7_granularity")
print("\n", gran.to_string())

rq7_raw = cesaw_ne.groupby("is_round").agg(
    N=("log_ratio", "size"), geo_mean=("log_ratio", lambda x: np.exp(x.mean())),
    median_estimate=("estimate", "median"), under_rate=("underestimated", "mean")).round(3)
rq7_ctrl = cesaw_ne.groupby(["size_bin", "is_round"], observed=True).agg(
    N=("log_ratio", "size"), geo_mean=("log_ratio", lambda x: np.exp(x.mean()))).round(3)
dump(rq7_raw, "10_rq7_round_raw"); dump(rq7_ctrl, "10_rq7_round_by_size")
print("\nRaw comparison of round vs non-round estimates:\n", rq7_raw.to_string())
print("\nWithin size bands:\n", rq7_ctrl.to_string())

piv7 = rq7_ctrl.geo_mean.unstack()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
vals = cesaw.task_plan_time_minutes.value_counts().head(15).sort_index()
rnd = ((vals.index % 30 == 0) | (vals.index % 60 == 0))
ax.bar(np.arange(len(vals)), vals.values,
       color=[ACCENT3 if r else ACCENT for r in rnd])
ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels([f"{v:g}" for v in vals.index], fontsize=8)
ax.set_xlabel("estimate, minutes"); ax.set_ylabel("tasks")
ax.set_title("Estimates land on a grid: 30, 60, 120, 180 min")
ax.bar(0, 0, color=ACCENT3, label="multiple of 30/60 min"); ax.legend(fontsize=9)

ax = axes[1]
x = np.arange(len(piv7))
ax.bar(x - .2, piv7[0], .4, color=ACCENT, label="non-round estimate")
ax.bar(x + .2, piv7[1], .4, color=ACCENT3, label="round estimate")
ax.axhline(1, color=INK, lw=1, ls="--")
ax.set_xticks(x); ax.set_xticklabels(piv7.index, fontsize=9)
ax.set_ylabel("geometric mean actual / estimate"); ax.set_xlabel("task size by estimate")
ax.set_title("...but within bands the effect is inconsistent")
ax.legend(fontsize=9)
fig.suptitle("RQ7: the raw round-number effect is mostly an artefact of task size",
             y=1.03, fontsize=12.5, fontweight="bold")
save(fig, "09_round_numbers"); plt.close(fig)

# %% [markdown]
# **RQ7 answer — and the best example in this project of why you cannot stop at the
# first comparison.**
#
# The raw comparison looks convincing: round estimates give ×0.77, non-round ×0.83.
# Effect found, apparently. But the median round estimate is 1 hour while the median
# non-round one is 0.72 hours: **round estimates are simply used more often on larger
# tasks**, and larger tasks are overestimated (RQ2). Once we compare within size bands,
# the direction stops being consistent — in some bands round estimates are more accurate,
# in others worse.
#
# Conclusion: **the observed round-number effect is explained by task size, not by
# roundness itself.** This is exactly the case the project's methodology warns about:
# association ≠ causation, and the confounder here is task size.
#
# ### Interim scoreboard

# %%
verdicts = pd.DataFrame([
    ("RQ1", "There is systematic underestimation", "REFUTED",
     "Geometric mean < 1 in all three datasets (×0.80 / ×0.86 / ×0.95)"),
    ("RQ2", "Large tasks are underestimated more", "REFUTED, opposite sign",
     "Regression to the mean: small underestimated, large overestimated; slope < 0, p<1e−140"),
    ("RQ3", "Accuracy differs between people", "SUPPORTED (CESAW)",
     "Strongest factor in CESAW: R² out-of-sample 0.103. But only 0.026 in SiP — "
     "the dominant factor depends on the organisation"),
    ("RQ4", "Project explains part of the variation", "supported, small effect",
     "p < 1e−50, R² out-of-sample 0.030"),
    ("RQ5", "Type of work (phase) matters", "supported, small effect",
     "p < 1e−50, R² out-of-sample 0.040"),
    ("RQ6", "Interruptions relate to the error", "SUPPORTED, strong effect",
     "×~2 within every size band; but only known after the fact"),
    ("RQ7", "Round estimates are less accurate", "NOT supported",
     "The raw effect disappears once task size is controlled for — confounder"),
], columns=["RQ", "hypothesis", "verdict", "evidence"])
dump(verdicts.set_index("RQ"), "11_hypotheses_verdicts")
print(verdicts.to_string(index=False))

# %% [markdown]
# ---
# # Phase 5 — Feature engineering and leakage control
#
# ## What may and may not be used
#
# The model must work at the moment the estimate is given and the work has not started.
# Anything that becomes known later is leakage.
#
# | feature | available before start? | decision |
# |---|---|---|
# | estimate, project, phase, person, team, process | yes | **use** |
# | calendar features of the estimation date | yes | use |
# | "estimate is round" | yes | use |
# | historical accuracy of person / project / phase | yes, if computed from the past only | **use (carefully)** |
# | `task_actual_time_minutes` | no | target |
# | `n_sessions`, `interrupt_min` | **no** | excluded, despite being the strongest factor (RQ6) |
# | `task_actual_complete_date` | no | excluded |
#
# A separate note on SiP: `DeveloperHoursActual`, `TaskPerformance` and
# `DeveloperPerformance` are direct functions of the target. Including them would yield
# R² ≈ 1 and a completely meaningless model.
#
# ## Historical features: where it is easy to go wrong
#
# The most valuable feature is "how wrong was this person before". But computing it
# naively (`groupby(person).log_ratio.mean()`) uses the future: for task #500 the mean
# would include tasks #501–900.
#
# The correct way is an **expanding mean shifted by one step**, in chronological order:
# for each task only strictly preceding tasks by the same person are used. The first
# task of every person gets `NaN` — and that is honest.

# %%
HIST_SPECS = [("person", "person"), ("project", "project"), ("phase_short_name", "phase")]

def add_history(df):
    """Leak-free historical features: strictly past tasks only."""
    df = df.sort_values("date").reset_index(drop=True)
    for col, name in HIST_SPECS:
        if col not in df.columns:
            continue
        g = df.groupby(col)["log_ratio"]
        df[f"{name}_hist_lr"] = g.transform(lambda s: s.shift(1).expanding().mean())
        df[f"{name}_hist_n"] = g.transform(lambda s: s.shift(1).expanding().count())
    df["person_hist_under"] = df.groupby("person")["underestimated"].transform(
        lambda s: s.shift(1).expanding().mean())
    df["person_recent_lr"] = df.groupby("person")["log_ratio"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    df["month"] = df.date.dt.month
    df["dow"] = df.date.dt.dayofweek
    return df

model_df = add_history(cesaw)

# Leakage check: the first task of every person must have no history.
first_task = model_df.groupby("person").head(1)
assert first_task.person_hist_lr.isna().all(), "leak: the first task has history!"
print(f"Check passed: all {len(first_task)} first-tasks-per-person have history = NaN")
print(f"person_hist_lr coverage: {model_df.person_hist_lr.notna().mean():.1%} of tasks")

CAT_FEATURES = ["phase_short_name", "process_name", "project", "person", "team_key"]
BASE_FEATURES = ["log_est", "is_round", "month", "dow"]
HIST_FEATURES = ["person_hist_lr", "person_hist_n", "project_hist_lr", "phase_hist_lr",
                 "person_hist_under", "person_recent_lr"]
FEATURES = CAT_FEATURES + BASE_FEATURES + HIST_FEATURES
print(f"\nFeatures: {len(FEATURES)} = {len(CAT_FEATURES)} categorical + "
      f"{len(BASE_FEATURES)} base + {len(HIST_FEATURES)} historical")

# %% [markdown]
# ---
# # Phase 6 — Baseline and split strategies
#
# ## The baseline to beat
#
# `actual = estimate`, i.e. "just trust the developer". Not a straw man: free, always
# available, and — as it turns out — very strong.
#
# ## Three split strategies, and why this is not a detail
#
# The data contains repeated observations of the same people and projects. A random
# split lets the model see *the same person on the same project* in both train and test.
# That is not target leakage, but it makes the result optimistic.
#
# | | what it tests | realism |
# |---|---|---|
# | **A. Random** | basic learnability | optimistic |
# | **B. By project** | transfer to a new project | strict generalisation test |
# | **C. Temporal** | predicting the future from the past | **matches production** |
#
# We compute all three and look at how far apart they land. Everything after this uses C.

# %%
from sklearn.ensemble import (HistGradientBoostingRegressor, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix, roc_curve,
                             precision_recall_curve)
from sklearn.inspection import permutation_importance

def prep(train, test, feats=FEATURES, cats=CAT_FEATURES):
    """HistGBM needs categories aligned between train and test."""
    a, b = train[feats].copy(), test[feats].copy()
    for c in cats:
        a[c] = a[c].astype(str).astype("category")
        b[c] = pd.Categorical(b[c].astype(str), categories=a[c].cat.categories)
    return a, b, [a.columns.get_loc(c) for c in cats]

def reg_metrics(pred_log, test, name):
    err = pred_log - test.log_act.values
    mae = float(np.abs(err).mean())
    mre = np.abs(np.exp(pred_log) - test.actual.values) / test.actual.values
    return {"model": name, "MAE_log": round(mae, 3), "typical error": f"×{np.exp(mae):.2f}",
            "RMSE_log": round(float(np.sqrt((err ** 2).mean())), 3),
            "MdMRE": round(float(np.median(mre)), 3),
            "PRED(25)": round(float((mre <= .25).mean()), 3)}

def clf_metrics(y, proba, name, thr=.5):
    yh = (proba >= thr).astype(int)
    return {"model": name, "accuracy": round(accuracy_score(y, yh), 3),
            "precision": round(precision_score(y, yh, zero_division=0), 3),
            "recall": round(recall_score(y, yh), 3), "F1": round(f1_score(y, yh), 3),
            "ROC-AUC": round(roc_auc_score(y, proba), 3),
            "PR-AUC": round(average_precision_score(y, proba), 3)}

def make_splits(df):
    n = len(df)
    idx = np.random.RandomState(RNG).permutation(n); cut = int(n * .75)
    yield "A: random", df.iloc[np.sort(idx[:cut])], df.iloc[np.sort(idx[cut:])]

    projects = df.project.value_counts().index.tolist()
    rs = np.random.RandomState(RNG); rs.shuffle(projects)
    test_projects, total = [], 0
    for p in projects:
        if total < n * .25:
            test_projects.append(p); total += int((df.project == p).sum())
    hold = df.project.isin(test_projects)
    yield "B: unseen projects", df[~hold], df[hold]

    cut_date = df.date.quantile(.75)
    yield "C: temporal", df[df.date <= cut_date], df[df.date > cut_date]

split_rows = []
for name, tr, te in make_splits(model_df):
    Xtr, Xte, ci = prep(tr, te)
    reg = HistGradientBoostingRegressor(categorical_features=ci, max_iter=300,
                                        learning_rate=.06, random_state=RNG).fit(Xtr, tr.log_ratio)
    clf = HistGradientBoostingClassifier(categorical_features=ci, max_iter=300,
                                         learning_rate=.06, random_state=RNG).fit(Xtr, tr.underestimated)
    m = reg_metrics(te.log_est.values + reg.predict(Xte), te, name)
    b0 = reg_metrics(te.log_est.values, te, "B0")
    c = clf_metrics(te.underestimated, clf.predict_proba(Xte)[:, 1], name)
    split_rows.append({"strategy": name, "N train": len(tr), "N test": len(te),
                       "B0 RMSE_log": b0["RMSE_log"], "GBM RMSE_log": m["RMSE_log"],
                       "RMSE gain": f"{100*(1-m['RMSE_log']/b0['RMSE_log']):.1f}%",
                       "ROC-AUC": c["ROC-AUC"], "PR-AUC": c["PR-AUC"],
                       "base rate": round(te.underestimated.mean(), 3)})
splits_tbl = pd.DataFrame(split_rows)
dump(splits_tbl, "12_split_strategies")
print(splits_tbl.to_string(index=False))

# %% [markdown]
# **The split strategy shifts the conclusions more than the choice of model does.**
# The random split paints a noticeably rosier picture than the temporal one; the
# by-project split lands in between. Reporting the random split would have inflated both
# ROC-AUC and the RMSE gain. Everything below uses the **temporal split (C)** — the only
# one that matches the real scenario.

# %%
CUT_DATE = model_df.date.quantile(.75)
train, test = model_df[model_df.date <= CUT_DATE], model_df[model_df.date > CUT_DATE]
Xtr, Xte, CAT_IDX = prep(train, test)
print(f"Train: {len(train)} tasks up to {CUT_DATE:%Y-%m-%d}   Test: {len(test)} tasks after")
print(f"Share underestimated: train {train.underestimated.mean():.3f}, "
      f"test {test.underestimated.mean():.3f}")

# %% [markdown]
# ---
# # Phase 7 — Machine learning
#
# ## 7.1 Task 1: regression — predict the size of the error
#
# We predict `log_ratio` rather than `log(actual)` directly: that way the model learns a
# **correction to the estimate** instead of re-deriving task duration from scratch, and
# the developer's estimate stays in the prediction as an anchor.
#
# We go up in complexity: baseline → global coefficient → log-linear calibration → Ridge
# → boosting. A more complex model is accepted only if it buys a visible gain.

# %%
reg_rows = [reg_metrics(np.full(len(test), train.log_act.mean()), test, "constant"),
            reg_metrics(test.log_est.values, test, "B0: actual = estimate")]
shift = train.log_ratio.mean()
reg_rows.append(reg_metrics(test.log_est.values + shift, test,
                            f"B1: global coefficient ×{np.exp(shift):.2f}"))
a, b = np.polyfit(train.log_est, train.log_act, 1)
reg_rows.append(reg_metrics(a * test.log_est.values + b, test,
                            f"B2: log-linear calibration (slope {a:.2f})"))

ridge_pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_FEATURES),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), BASE_FEATURES + HIST_FEATURES)])
ridge = Pipeline([("pre", ridge_pre), ("m", Ridge(alpha=1.0))]).fit(train[FEATURES], train.log_ratio)
reg_rows.append(reg_metrics(test.log_est.values + ridge.predict(test[FEATURES]), test,
                            "M1: Ridge + features"))

gbm_reg = HistGradientBoostingRegressor(categorical_features=CAT_IDX, max_iter=400,
                                        learning_rate=.06, random_state=RNG).fit(Xtr, train.log_ratio)
gbm_pred = test.log_est.values + gbm_reg.predict(Xte)
reg_rows.append(reg_metrics(gbm_pred, test, "M2: Gradient Boosting"))

reg_tbl = pd.DataFrame(reg_rows).set_index("model")
dump(reg_tbl, "13_regression_models")
print(reg_tbl.to_string())

# %% [markdown]
# ## 7.2 Task 2: classification — predict the fact of underestimation
#
# Target `underestimated = 1 if actual > estimate`. Same progression, from class share to
# boosting. Accuracy is not informative here (classes are imbalanced, ~36% positive), so
# the primary measures are **PR-AUC** against the base rate, and ROC-AUC.

# %%
clf_rows = [clf_metrics(test.underestimated,
                        np.full(len(test), train.underestimated.mean()),
                        f"Baseline: class share ({train.underestimated.mean():.2f})")]
logit = Pipeline([("pre", ridge_pre),
                  ("m", LogisticRegression(max_iter=2000))]).fit(train[FEATURES], train.underestimated)
clf_rows.append(clf_metrics(test.underestimated, logit.predict_proba(test[FEATURES])[:, 1],
                            "Logistic regression"))
tree = Pipeline([("pre", ridge_pre),
                 ("m", DecisionTreeClassifier(max_depth=6, random_state=RNG))]).fit(train[FEATURES], train.underestimated)
clf_rows.append(clf_metrics(test.underestimated, tree.predict_proba(test[FEATURES])[:, 1],
                            "Decision tree (depth=6)"))
forest = Pipeline([("pre", ridge_pre),
                   ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                random_state=RNG, n_jobs=-1))]).fit(train[FEATURES], train.underestimated)
rf_proba = forest.predict_proba(test[FEATURES])[:, 1]
clf_rows.append(clf_metrics(test.underestimated, rf_proba, "Random Forest"))
gbm_clf = HistGradientBoostingClassifier(categorical_features=CAT_IDX, max_iter=400,
                                         learning_rate=.06, random_state=RNG).fit(Xtr, train.underestimated)
gbm_proba = gbm_clf.predict_proba(Xte)[:, 1]
clf_rows.append(clf_metrics(test.underestimated, gbm_proba, "Gradient Boosting"))

clf_tbl = pd.DataFrame(clf_rows).set_index("model")
dump(clf_tbl, "14_classification_models")
print(f"Base rate on test: {test.underestimated.mean():.3f}\n")
print(clf_tbl.to_string())

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
for proba, name, c in [(rf_proba, "Random Forest", ACCENT3), (gbm_proba, "Gradient Boosting", ACCENT)]:
    fpr, tpr, _ = roc_curve(test.underestimated, proba)
    ax.plot(fpr, tpr, color=c, lw=2.2, label=f"{name} (AUC={roc_auc_score(test.underestimated, proba):.3f})")
ax.plot([0, 1], [0, 1], color=INK, ls="--", lw=1, label="random model")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC curve"); ax.legend(fontsize=8)

ax = axes[1]
base = test.underestimated.mean()
for proba, name, c in [(rf_proba, "Random Forest", ACCENT3), (gbm_proba, "Gradient Boosting", ACCENT)]:
    pr, rc, _ = precision_recall_curve(test.underestimated, proba)
    ax.plot(rc, pr, color=c, lw=2.2,
            label=f"{name} (PR-AUC={average_precision_score(test.underestimated, proba):.3f})")
ax.axhline(base, color=INK, ls="--", lw=1, label=f"base rate = {base:.2f}")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Precision-Recall (what matters under imbalance)"); ax.legend(fontsize=8)

ax = axes[2]
cm = confusion_matrix(test.underestimated, (rf_proba >= .5).astype(int))
ax.imshow(cm, cmap="Blues"); ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(["on budget", "underestimated"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["on budget", "underestimated"])
ax.set_xlabel("predicted"); ax.set_ylabel("actual")
ax.set_title("Confusion matrix, Random Forest (threshold 0.5)")
save(fig, "10_classification"); plt.close(fig)

# %% [markdown]
# ## 7.3 Ablation: where the predictive power actually comes from
#
# The most informative experiment in the project. The same model is trained on three
# nested feature sets to see what buys the gain.

# %%
ABLATION = [("estimate only", ["log_est"]),
            ("+ context (project / phase / person)", CAT_FEATURES + BASE_FEATURES),
            ("+ history (leak-free)", CAT_FEATURES + BASE_FEATURES + HIST_FEATURES)]
abl_rows = []
b0 = reg_metrics(test.log_est.values, test, "B0")
for label, feats in ABLATION:
    cats = [c for c in CAT_FEATURES if c in feats]
    A, B, ci = prep(train, test, feats, cats)
    r = HistGradientBoostingRegressor(categorical_features=ci, max_iter=400,
                                      learning_rate=.06, random_state=RNG).fit(A, train.log_ratio)
    c = HistGradientBoostingClassifier(categorical_features=ci, max_iter=400,
                                       learning_rate=.06, random_state=RNG).fit(A, train.underestimated)
    m = reg_metrics(test.log_est.values + r.predict(B), test, label)
    cm_ = clf_metrics(test.underestimated, c.predict_proba(B)[:, 1], label)
    abl_rows.append({"feature set": label, "MAE_log": m["MAE_log"],
                     "RMSE_log": m["RMSE_log"], "ROC-AUC": cm_["ROC-AUC"],
                     "PR-AUC": cm_["PR-AUC"]})
abl_tbl = pd.DataFrame(abl_rows).set_index("feature set")
dump(abl_tbl, "15_ablation")
print(f"B0 (actual = estimate): MAE_log={b0['MAE_log']}, RMSE_log={b0['RMSE_log']}\n")
print(abl_tbl.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
labels = [l for l, _ in ABLATION]
x = np.arange(len(labels))
ax = axes[0]
ax.bar(x, abl_tbl["ROC-AUC"], color=[MUTED, ACCENT, ACCENT3])
ax.axhline(.5, color=INK, ls="--", lw=1.2)
ax.annotate("random model", (1.35, .508), fontsize=9, color=INK, ha="center")
ax.set_xticks(x); ax.set_xticklabels(["estimate\nonly", "+ context", "+ history"], fontsize=9)
ax.set_ylim(.4, .8); ax.set_ylabel("ROC-AUC")
ax.set_title("The estimate does not predict its own error")
for xi, v in zip(x, abl_tbl["ROC-AUC"]):
    ax.text(xi, v + .008, f"{v:.3f}", ha="center", fontsize=9)

ax = axes[1]
ax.bar(x, abl_tbl["RMSE_log"], color=[MUTED, ACCENT, ACCENT3])
ax.axhline(b0["RMSE_log"], color=INK, ls="--", lw=1.2)
ax.set_ylim(0, b0["RMSE_log"] * 1.28)
ax.annotate("B0: actual = estimate", (1.0, b0["RMSE_log"] * 1.06), fontsize=9,
            color=INK, ha="center")
ax.set_xticks(x); ax.set_xticklabels(["estimate\nonly", "+ context", "+ history"], fontsize=9)
ax.set_ylabel("RMSE_log (lower is better)")
ax.set_title("Only context beats the baseline")
for xi, v in zip(x, abl_tbl["RMSE_log"]):
    ax.text(xi, v + .008, f"{v:.3f}", ha="center", fontsize=9)
fig.suptitle("Ablation: all the predictive power is in context and history, not the estimate",
             y=1.08, fontsize=12.5, fontweight="bold")
save(fig, "11_ablation"); plt.close(fig)

# %% [markdown]
# **This is the key result of the project.**
#
# A model that knows **only the estimate** predicts underestimation at ROC-AUC ≈ 0.49 —
# **no better than a coin flip** — and on MAE it is worse than simply trusting the
# estimate. In other words: *the estimate contains no information about whether it is
# wrong.* A developer does not "somewhat know" about their error — they know nothing
# about it.
#
# All the predictive power appears once **context** is added (who, which project, which
# phase): AUC jumps to ≈ 0.69. Leak-free historical features add a little more
# (to ≈ 0.71) and improve RMSE.
#
# The answer to RQ8 ("can underestimation be predicted?") is **yes, but not from the
# estimate.** What is predictable is not the task but the circumstances.

# %% [markdown]
# ---
# # Phase 8 — Explainability and error analysis
#
# ## 8.1 Permutation importance
#
# How much the error grows when one feature is randomly shuffled. Unlike built-in
# importances, this is computed on the test set and does not inflate features with many
# levels.

# %%
perm = permutation_importance(gbm_reg, Xte, test.log_ratio.values, n_repeats=8,
                              random_state=RNG, scoring="neg_root_mean_squared_error")
imp = pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean,
                    "std": perm.importances_std}).sort_values("importance", ascending=False)
dump(imp.round(4), "16_permutation_importance")
print(imp.round(4).to_string(index=False))

# %% [markdown]
# ## 8.2 SHAP: why the model considers a particular task risky
#
# Permutation importance answers "which feature matters in general". SHAP answers
# "why did **this** task get this prediction" — which is what you need if the
# explanation is going to be shown to a manager.

# %%
import shap

shap_sample = Xte.sample(min(2000, len(Xte)), random_state=RNG)

# TreeExplainer does not understand pandas categoricals, while HistGBM works on their
# codes internally. We pass the codes — the numbering is identical, so the SHAP values
# are correct; check_additivity verifies it (contributions must sum to the prediction).
shap_input = shap_sample.copy()
for c in CAT_FEATURES:
    shap_input[c] = shap_input[c].cat.codes.astype(float)

explainer = shap.TreeExplainer(gbm_reg)
shap_values = explainer(shap_input, check_additivity=True)
shap_values.data = shap_sample.values      # use original values for labelling
print(f"SHAP computed for {len(shap_sample)} tasks; additivity check passed")

fig = plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values.values, shap_input, max_display=12, show=False, plot_size=None)
plt.title("SHAP: feature contributions to the predicted correction", fontsize=12,
          fontweight="bold", pad=14)
plt.tight_layout()
save(plt.gcf(), "12_shap_summary"); plt.close("all")

shap_rank = pd.DataFrame({
    "feature": shap_sample.columns,
    "mean |SHAP|": np.abs(shap_values.values).mean(0)}).sort_values(
    "mean |SHAP|", ascending=False)
dump(shap_rank.round(4), "17_shap_importance")
print(shap_rank.round(4).to_string(index=False))

# Breaking down one concrete prediction — what you would actually show a manager.
risky = int(np.argmax(gbm_reg.predict(shap_sample)))
case = shap_sample.iloc[risky]
contrib = pd.Series(shap_values.values[risky], index=shap_sample.columns).sort_values(key=abs, ascending=False)
print(f"\nExample: the task with the largest predicted correction "
      f"(prediction ×{np.exp(gbm_reg.predict(shap_sample)[risky]):.2f} on the estimate)")
print("Top drivers of this prediction:")
for f, v in contrib.head(6).items():
    print(f"  {f:22s} value={str(case[f])[:18]:20s} contribution={v:+.3f}")

# %% [markdown]
# ## 8.3 Where the model fails
#
# Averaged metrics hide the fact that quality is very uneven. We break the error down by
# task size, phase, and whether the person has any history.

# %%
test_err = test.copy()
test_err["pred_log"] = gbm_pred
test_err["err"] = test_err.pred_log - test_err.log_act
test_err["abs_err"] = test_err.err.abs()
test_err["b0_abs_err"] = (test_err.log_est - test_err.log_act).abs()
test_err["has_history"] = np.where(test_err.person_hist_n.fillna(0) >= 20,
                                   "≥20 past tasks", "little history")

err_by = {}
for key, label in [("size_bin", "task size"), ("phase_short_name", "process phase"),
                   ("has_history", "person history")]:
    g = test_err.groupby(key, observed=True)
    t = pd.DataFrame({"N": g.size(), "GBM MAE_log": g.abs_err.mean(),
                      "B0 MAE_log": g.b0_abs_err.mean(),
                      "bias (mean err)": g.err.mean()})
    t["gain %"] = (100 * (1 - t["GBM MAE_log"] / t["B0 MAE_log"])).round(1)
    err_by[label] = t.round(3)
    dump(t.round(3), f"18_error_by_{key}")
    print(f"\nError by \"{label}\":\n{t.round(3).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
t = err_by["task size"]
x = np.arange(len(t))
ax.bar(x - .2, t["B0 MAE_log"], .4, color=MUTED, label="B0: actual = estimate")
ax.bar(x + .2, t["GBM MAE_log"], .4, color=ACCENT3, label="GBM")
ax.set_xticks(x); ax.set_xticklabels(t.index, fontsize=9)
ax.set_ylabel("MAE_log"); ax.set_xlabel("task size by estimate")
ax.set_title("The gain is on small and medium tasks only"); ax.legend(fontsize=9)

ax = axes[1]
sub = test_err.sample(min(6000, len(test_err)), random_state=RNG)
ax.scatter(np.exp(sub.pred_log), sub.actual, s=6, alpha=.15, color=ACCENT, edgecolors="none")
lim = [test_err.actual.min() * .7, test_err.actual.max() * 1.3]
ax.plot(lim, lim, color=INK, ls="--", lw=1.3, label="perfect prediction")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("predicted, hours"); ax.set_ylabel("actual, hours")
ax.set_title("Predicted against actual on the test set"); ax.legend(fontsize=9)
save(fig, "13_error_analysis"); plt.close(fig)

# %% [markdown]
# **The model does not win everywhere — and that matters more than the headline number.**
#
# | task size | N | B0 MAE_log | GBM MAE_log | gain |
# |---|---:|---:|---:|---:|
# | ≤ 0.5h | 5,618 | 0.660 | 0.642 | +2.7% |
# | 0.5–1h | 3,341 | 0.706 | 0.679 | +3.8% |
# | 1–2h | 2,558 | 0.761 | 0.719 | +5.5% |
# | 2–4h | 1,676 | 0.783 | 0.732 | +6.6% |
# | 4–8h | 1,064 | 0.677 | 0.671 | +0.9% |
# | 8–16h | 644 | 0.538 | 0.600 | **−11.6%** |
# | > 16h | 170 | 0.442 | 0.604 | **−36.6%** |
#
# On large tasks the model **loses to the baseline**, and by a wide margin. The N column
# shows why: large tasks are rare in training (170 in test against 5,618 small ones) and
# their spread is the widest (RQ2). The model pulls predictions toward typical behaviour
# and systematically undershoots on the rare large tasks — mean bias there is −0.24.
#
# Practical conclusion: **apply the model only to small and medium tasks**, and on large
# ones keep the developer's estimate. The headline "10% gain" without this breakdown
# would be misleading.

# %%
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.scatter(np.exp(sub.pred_log), sub.err, s=6, alpha=.15, color=ACCENT, edgecolors="none")
ax.axhline(0, color=INK, lw=1.2, ls="--")
ax.set_xscale("log")
ax.set_yticks(np.log([1/8, 1/4, 1/2, 1, 2, 4, 8]))
ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
ax.set_xlabel("predicted, hours (log scale)")
ax.set_ylabel("predicted / actual")
ax.set_title("Residual plot: no systematic tilt,\nbut the spread is huge across the range")
save(fig, "14_residuals"); plt.close(fig)

# %% [markdown]
# ---
# # Phase 9 — What can actually be handed to a team
#
# ## 9.1 Personal calibration (checking the obvious fix first)
#
# Before proposing ML, test the obvious idea: every developer has their own historical
# coefficient, so multiply the estimate by it. Computed strictly leak-free — the
# coefficient comes only from that person's past tasks.

# %%
calib = test.copy()
calib_rows = [reg_metrics(calib.log_est.values, calib, "B0: actual = estimate")]
for min_n, label in [(5, "personal coefficient (≥5 past tasks)"),
                     (20, "personal coefficient (≥20 past tasks)")]:
    adj = np.where(calib.person_hist_n.fillna(0) >= min_n, calib.person_hist_lr.fillna(0), 0)
    calib_rows.append(reg_metrics(calib.log_est.values + adj, calib, label))
calib_rows.append(reg_metrics(calib.log_est.values + shift, calib,
                              f"global coefficient ×{np.exp(shift):.2f}"))
calib_rows.append(reg_metrics(gbm_pred, calib, "M2: Gradient Boosting"))
calib_tbl = pd.DataFrame(calib_rows).set_index("model")
dump(calib_tbl, "19_personal_calibration")
print(calib_tbl.to_string())

# %% [markdown]
# The result cuts both ways, and this is a case where a single metric would have lied.
#
# The personal coefficient **improves `RMSE_log`** (1.053 → 0.997) but **worsens
# `MAE_log`** (0.694 → 0.716). It helps on outliers and hurts on typical tasks: by
# shifting every estimate, it damages the ones that were already close. Raising the
# threshold from "≥5 past tasks" to "≥20" changes nothing — the issue is not the volume
# of history but that personal bias explains too little (RQ3).
#
# The full model beats both calibration variants on both metrics, precisely because it
# accounts for not only *who* but also *what* and *where*.
#
# ## 9.2 An interval instead of a number
#
# If 83–86% of the variance is irreducible (Phase 4), a point prediction is meaningless
# no matter who produced it. The only honest product is an interval.
#
# Method: quantile regression (GBM with pinball loss on the 10% and 90% quantiles of
# `log_ratio`), then **conformalized quantile regression (CQR)**. The calendar-last 20%
# of the training set is held out as a calibration set, the conformity score
# `s = max(lo − y, y − hi)` is computed on it, and the interval is widened by its
# empirical quantile. This gives coverage close to nominal **with no distributional
# assumptions**.
#
# Exchangeability — the condition behind the CQR guarantee — only holds approximately
# here, because the process is non-stationary. So coverage is verified empirically on
# each window rather than taken on faith. And as the numbers below show, the result is
# honest but not perfect: calibration lifts coverage from ~73% to ~78% against a nominal
# 80%, so a residual overconfidence remains. That is expected — non-stationarity breaks
# the method's premise, and claiming exactly 80% would be untrue.

# %%
def conformal_intervals(df, alpha=.2, n_folds=5, start=.55, stop=.90, horizon=.08):
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    rows, example = [], None
    for k, q in enumerate(np.linspace(start, stop, n_folds)):
        c1, c2 = df.date.quantile(q), df.date.quantile(min(q + horizon, .999))
        tr_all, te = df[df.date <= c1], df[(df.date > c1) & (df.date <= c2)]
        if len(te) < 200:
            continue
        n_fit = int(len(tr_all) * .8)
        fit, cal = tr_all.iloc[:n_fit], tr_all.iloc[n_fit:]
        Xf, Xt, ci = prep(fit, te); _, Xc, _ = prep(fit, cal)
        models = {p: HistGradientBoostingRegressor(loss="quantile", quantile=p,
                                                   categorical_features=ci, max_iter=250,
                                                   learning_rate=.06, random_state=RNG).fit(Xf, fit.log_ratio)
                  for p in (lo_q, hi_q)}
        lo = te.log_est.values + models[lo_q].predict(Xt)
        hi = te.log_est.values + models[hi_q].predict(Xt)
        y = te.log_act.values
        lo_c = cal.log_est.values + models[lo_q].predict(Xc)
        hi_c = cal.log_est.values + models[hi_q].predict(Xc)
        score = np.maximum(lo_c - cal.log_act.values, cal.log_act.values - hi_c)
        rank = int(np.ceil((len(score) + 1) * (1 - alpha))) - 1
        Q = np.sort(score)[min(max(rank, 0), len(score) - 1)]
        rows.append({"fold": k + 1, "N test": len(te),
                     "coverage (raw)": round(float(((y >= lo) & (y <= hi)).mean()), 3),
                     "width (raw)": round(float(np.median(np.exp(hi - lo))), 2),
                     "coverage (CQR)": round(float(((y >= lo - Q) & (y <= hi + Q)).mean()), 3),
                     "width (CQR)": round(float(np.median(np.exp((hi + Q) - (lo - Q)))), 2),
                     "correction Q": round(float(Q), 3)})
        if example is None:
            example = te.assign(lo=np.exp(lo - Q), hi=np.exp(hi + Q))
    return pd.DataFrame(rows), example

intervals, iv_example = conformal_intervals(model_df)
dump(intervals, "20_conformal_intervals")
print(intervals.to_string(index=False))
print(f"\nNominal coverage 80%. Raw quantiles: {intervals['coverage (raw)'].mean():.1%}, "
      f"after CQR: {intervals['coverage (CQR)'].mean():.1%}")
print(f"Median interval width: raw ×{intervals['width (raw)'].mean():.1f}, "
      f"after calibration ×{intervals['width (CQR)'].mean():.1f}")

iv_example["size_bin"] = pd.cut(iv_example.estimate, SIZE_BINS, labels=SIZE_LABELS)
width_by_size = iv_example.groupby("size_bin", observed=True).apply(
    lambda g: pd.Series({
        "N": len(g),
        "coverage": round(float(((g.actual >= g.lo) & (g.actual <= g.hi)).mean()), 2),
        "lower bound / estimate": round(float(np.median(g.lo / g.estimate)), 2),
        "upper bound / estimate": round(float(np.median(g.hi / g.estimate)), 2),
    }), include_groups=False)
dump(width_by_size, "20_interval_width_by_size")
print("\nWhat a developer's estimate actually means (80% interval):")
print(width_by_size.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ax = axes[0]
x = np.arange(len(intervals))
ax.plot(x, intervals["coverage (raw)"], "o-", color=ACCENT2, lw=2, label="raw quantile regression")
ax.plot(x, intervals["coverage (CQR)"], "s-", color=ACCENT3, lw=2.4, label="after CQR calibration")
ax.axhline(.8, color=INK, ls="--", lw=1.2)
ax.annotate("nominal 80%", (0, .807), fontsize=9, color=INK)
ax.set_xticks(x); ax.set_xticklabels([f"fold {i}" for i in intervals["fold"]])
ax.set_ylim(.5, 1.0)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylabel("share of actuals inside the interval")
ax.set_title("Raw quantiles are overconfident; CQR fixes most of it")
ax.legend(loc="lower right", fontsize=9)

ax = axes[1]
ex = iv_example.sort_values("estimate").iloc[::max(1, len(iv_example) // 60)]
xs = np.arange(len(ex))
ax.vlines(xs, ex.lo, ex.hi, color=ACCENT, alpha=.45, lw=3)
inside = ((ex.actual >= ex.lo) & (ex.actual <= ex.hi)).values
ax.scatter(xs[inside], ex.actual.values[inside], s=14, color=INK, zorder=3, label="actual inside")
ax.scatter(xs[~inside], ex.actual.values[~inside], s=24, color=ACCENT2, zorder=3,
           marker="x", label="actual outside")
ax.plot(xs, ex.estimate, color=ACCENT3, lw=1.6, ls="--", label="original estimate")
ax.set_yscale("log")
ax.set_xlabel("test-window tasks, sorted by estimate")
ax.set_ylabel("hours (log scale)")
ax.set_title("A calibrated 80% interval"); ax.legend(fontsize=8)
save(fig, "15_intervals"); plt.close(fig)

# %% [markdown]
# ---
# # Validation on other organisations
#
# CESAW is one development culture (TSP, partly safety-critical projects). Let us check
# the headline finding — regression to the mean — on SiP (a commercial company) and
# Renzo (one person, personal tracking). To make the datasets comparable we split each
# into quantile bands by estimate size, which removes the difference in units.

# %%
def quantile_profile(d, n_bins=7):
    d = d.copy()
    d["qbin"] = pd.qcut(d.estimate.rank(method="first"), n_bins, labels=False)
    g = d.groupby("qbin")
    return pd.DataFrame({"N": g.size(), "median estimate": g.estimate.median(),
                         "geometric mean": g.log_ratio.apply(lambda x: np.exp(x.mean()))})

repl = {name: quantile_profile(d) for name, d in
        [("CESAW", cesaw_ne), ("SiP", sip_ne), ("Renzo", renzo_ne)]}
repl_tbl = pd.concat(repl, names=["dataset"]).round(3)
dump(repl_tbl, "21_replication")
print(repl_tbl.to_string())

fig, ax = plt.subplots(figsize=(8.5, 4.6))
for (name, t), c, m in zip(repl.items(), [ACCENT, ACCENT2, ACCENT3], ["o", "s", "^"]):
    ax.plot(t["median estimate"], t["geometric mean"], m + "-", color=c, lw=2.2, ms=7,
            label=f"{name} (N={int(t.N.sum()):,})")
ax.axhline(1, color=INK, lw=1, ls="--")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_yticks([0.5, 0.75, 1, 1.5, 2, 3])
ax.set_yticklabels(["×0.5", "×0.75", "×1", "×1.5", "×2", "×3"])
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
ax.set_xlabel("median estimate in band (hours / pomodoros, log scale)")
ax.set_ylabel("geometric mean actual / estimate")
ax.set_title("Regression to the mean replicates across three independent sources:\n"
             "different industries, cultures and units of measurement")
ax.legend()
save(fig, "16_replication"); plt.close(fig)

# %%
# --- Summary of the key numbers (the basis for the report) ------------------
summary = {
    "CESAW: tasks after cleaning": len(cesaw),
    "CESAW: share actual == estimate": f"{cesaw.exact.mean():.1%}",
    "SiP: share actual == estimate": f"{sip.exact.mean():.1%}",
    "Renzo: share actual == estimate": f"{renzo.exact.mean():.1%}",
    "RQ1 CESAW: geometric mean ratio": round(float(np.exp(cesaw_ne.log_ratio.mean())), 3),
    "RQ1 CESAW: share underestimated": f"{cesaw_ne.underestimated.mean():.1%}",
    "RQ2 slope log_ratio~log_est (CESAW)": float(slopes_tbl.loc["CESAW", "slope"]),
    "RQ2 slope (SiP)": float(slopes_tbl.loc["SiP", "slope"]),
    "RQ2 slope (Renzo)": float(slopes_tbl.loc["Renzo", "slope"]),
    "RQ3 R² person": float(ve_cesaw.loc["person", "R² out-of-sample"]),
    "RQ4 R² project": float(ve_cesaw.loc["project", "R² out-of-sample"]),
    "RQ5 R² phase": float(ve_cesaw.loc["phase_short_name", "R² out-of-sample"]),
    "R² task size": float(ve_cesaw.loc["size_bin", "R² out-of-sample"]),
    "R² all factors combined": float(ve_cesaw.loc["ALL FACTORS COMBINED", "R² out-of-sample"]),
    "B0 RMSE_log": b0["RMSE_log"],
    "GBM RMSE_log": float(reg_tbl.loc["M2: Gradient Boosting", "RMSE_log"]),
    "Ablation: AUC estimate only": float(abl_tbl.iloc[0]["ROC-AUC"]),
    "Ablation: AUC + context": float(abl_tbl.iloc[1]["ROC-AUC"]),
    "Ablation: AUC + history": float(abl_tbl.iloc[2]["ROC-AUC"]),
    "Best classifier PR-AUC": float(clf_tbl["PR-AUC"].max()),
    "Underestimation base rate (test)": round(float(test.underestimated.mean()), 3),
    "Interval coverage before CQR": f"{intervals['coverage (raw)'].mean():.1%}",
    "Interval coverage after CQR": f"{intervals['coverage (CQR)'].mean():.1%}",
}
summary_tbl = pd.DataFrame.from_dict(summary, orient="index", columns=["value"])
dump(summary_tbl, "22_summary")
print(summary_tbl.to_string())

# %% [markdown]
# ---
# # Conclusions
#
# ## What we learned about estimates
#
# **1. "Perfect estimates" are the first thing to check.** In 8% of CESAW tasks, 34% of
# SiP and 44% of Renzo the actual equals the estimate exactly, and the share falls from
# ~60% on small tasks to ~1% on large ones. That is logging time against the plan, not
# accuracy. A dashboard built on top of this artefact praises the team for something
# that is not there.
#
# **2. There is no systematic underestimation — there is regression to the mean**
# (RQ1, RQ2). In aggregate teams even come in under their estimates (×0.80 in CESAW),
# but the aggregate is deceptive: small tasks are underestimated, large ones
# overestimated, and the two biases cancel. The effect replicates across three
# independent sources with different industries, cultures and units.
#
# **3. Significant ≠ important, and the dominant factor depends on the organisation**
# (RQ3–RQ5). In CESAW the person matters most (R² out-of-sample 0.103), in SiP task size
# (0.080), and in each dataset the other's factor barely works. "What matters here is who
# estimates" does not transfer between companies.
#
# But the ceiling is low in both cases: **all observed factors combined explain 17% of
# the variance in CESAW and 14% in SiP — so 83–86% is explained by nothing.** At 55,000
# observations almost everything is statistically significant; the practical conclusion
# comes from effect size, not p-values.
#
# **4. Work fragmentation is the strongest factor found** (RQ6). Interrupted tasks
# overrun roughly twice as much within **every** size band. But causality is not
# established (interruptions may be a consequence, not a cause), and for prediction it is
# useless: at estimation time they have not happened yet.
#
# **5. The round-number effect did not hold up** (RQ7). The raw comparison shows a
# difference, but it is entirely explained by round estimates being used on larger tasks.
# Control for size and the effect falls apart. A good example of why confounders must be
# checked.
#
# ## What we learned about predictability
#
# **6. The estimate contains no information about its own error.** A model given only the
# estimate predicts underestimation at ROC-AUC 0.49 — coin-flip level. What is
# predictable is the **circumstances**: adding context lifts AUC to ~0.69, adding history
# to ~0.71.
#
# **7. The split strategy shifts conclusions more than the choice of model.** A random
# split gives noticeably rosier numbers than a temporal one. Reporting a random split on
# data with repeated observations is misleading.
#
# **8. ML wins on small and medium tasks and loses on large ones.** The gain shows up
# mainly in `RMSE_log` — on tasks that blow up by multiples. But above 8 hours the model
# is worse than the baseline, because large tasks are rare and their spread is widest.
#
# **9. Personal coefficients work halfway.** "Multiply the estimate by the developer's
# historical coefficient" improves `RMSE_log` (1.053 → 0.997) but worsens `MAE_log`
# (0.694 → 0.716): it helps on outliers and hurts on typical tasks. Requiring more
# history changes nothing. The full model beats both variants on both metrics.
#
# **10. Hand over an interval — an honest one.** Without calibration quantile regression
# is overconfident (coverage ~73% against a nominal 80%); the conformal correction lifts
# it to ~78%. It does not reach nominal: exchangeability is broken by the
# non-stationarity of the process, and claiming exactly 80% would be untrue. A wide
# interval is not a defect of the model but a **measured property of the domain**.
#
# ## What a team should do with this
#
# * **Check your own logging first.** The share of exact matches by task size is a free
#   health test for the data. If it falls from 60% to 1%, you are measuring logging
#   discipline, not estimation accuracy.
# * **Drop the single safety coefficient.** It errs in opposite directions on different
#   tasks. Calibrate by size: multiply small, divide large.
# * **Measure what dominates in your organisation.** In one company it was the person,
#   in another task size. It is one R² table to compute, and it decides where effort is
#   worth spending at all.
# * **Protect focus.** Fragmentation is the only strong controllable factor found.
# * **Plan with intervals.** On this data a "4 hours" estimate means an 80% range of
#   roughly one to eight hours. Better to know that up front.
#
# ## Limitations
#
# * **Observational data.** We do not know whether the estimate influenced the actual:
#   a developer may have paced the work to the quoted deadline (self-fulfilling
#   prophecy). Inseparable without an experiment.
# * **Survivorship bias.** Unfinished and cancelled tasks are absent from the datasets;
#   the conclusions are likely biased optimistic.
# * **RQ6 is not causal.** The association between interruptions and overrun is
#   established; the direction is not.
# * **Exchangeability for CQR holds only approximately** — the process is
#   non-stationary, so coverage was checked empirically on each window (0.76–0.81).
# * **CESAW is a specific environment** (formal TSP process, some safety-critical
#   projects). SiP and Renzo confirm the direction of the effects but not their size.
# * **The model is unfit for large tasks** — see Phase 8.
#
# ## Next steps
#
# * A hierarchical Bayesian model: partial pooling across person and phase would give
#   stable estimates for rare categories instead of noisy group means — and would
#   probably fix the failure on large tasks.
# * A censored-observation model for unfinished tasks, to remove the survivorship bias.
# * Aggregation to sprint level: do individual task errors cancel out or compound? For
#   planning this is the central question, and the CESAW data can answer it.
# * The WBS hierarchy (`wbs_parent.csv`) is still unused.

# %%
print("\n" + "=" * 72)
print(f"DONE. Figures: {len(list(FIG_DIR.glob('*.png')))} | "
      f"tables: {len(list(RES_DIR.glob('*.csv')))}")
print("=" * 72)
