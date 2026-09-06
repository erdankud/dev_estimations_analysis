# %% [markdown]
# # Predicting and Explaining Software Development Estimation Error
#
# **Почему разработчики ошибаются в оценках — и можно ли предсказать ошибку заранее**
#
# Данные: [Derek Jones — Software estimation datasets](https://github.com/Derek-Jones/Software-estimation-datasets)
#
# ---
#
# ## Центральный вопрос
#
# > **Можно ли предсказать, что задача будет недооценена, и какие факторы объясняют
# > ошибку оценки?**
#
# Заметьте формулировку. Это **не** «предскажи `actual_hours`» — это был бы обычный
# regression exercise. Интересный вопрос другой: почему `Actual ≠ Estimate`, и содержит
# ли эта ошибка предсказуемую структуру или это чистый шум.
#
# ## Датасеты
#
# | | Задач | Единицы | Период | Роль в проекте |
# |---|---:|---|---|---|
# | **CESAW** | 60 284 | минуты → часы | 2008–2017 | **Основной.** 247 человек, 45 проектов, TSP-процесс. Есть данные о прерываниях. |
# | **SiP** | 10 266 | часы | 2004–2014 | Валидация на другой организации. 22 разработчика, 20 проектов. |
# | **Renzo Pomodoro** | 10 159 | помидоры | 2009–2019 | Валидация на индивидуальном уровне. Один человек, 10 лет. |
#
# CESAW выбран основным: 61 817 задач в 45 проектах — достаточно, чтобы делать
# не игрушечный ML, и, в отличие от PROMISE-датасетов, здесь есть настоящая пара
# **estimate → actual** на уровне отдельной задачи, а не проекта.
#
# ## Целевые переменные
#
# Абсолютная ошибка `actual − estimate` бесполезна сама по себе: ошибка в 2 часа
# на задаче в 2 часа и на задаче в 200 часов — совершенно разные события. Поэтому
# работаем с тремя связанными величинами:
#
# | | определение | смысл |
# |---|---|---|
# | `estimation_ratio` | `actual / estimate` | > 1 недооценка, < 1 переоценка |
# | `log_ratio` | `log(actual / estimate)` | то же, но симметрично и аддитивно |
# | `underestimated` | `1 if actual > estimate` | таргет классификации |
#
# **Почему логарифм.** Отношение положительно и сильно скошено. В логах ошибка
# «в 2 раза дольше» (+0.69) и «в 2 раза быстрее» (−0.69) симметричны, среднее
# лог-отношения = логарифм геометрического среднего (устойчивая мера смещения),
# а `exp(MAE_log)` читается напрямую как «типичная ошибка в N раз».
#
# ## Порядок работы
#
# ```
# Phase 0  Понимание данных      →  DATA_DICTIONARY.md
# Phase 1  Очистка               →  журнал решений
# Phase 2  Сборка task-level     →  одна строка = одна задача
# Phase 3  EDA
# Phase 4  Гипотезы RQ1–RQ7      →  статистические тесты
# Phase 5  Feature engineering   →  leak-free исторические признаки
# Phase 6  Baseline + стратегии разбиения
# Phase 7  ML: регрессия + классификация
# Phase 8  Объяснимость + error analysis
# Phase 9  Интервалы + персональная калибровка
# ```
#
# Модели не строятся, пока данные не поняты. Это не формальность: половина выводов
# этого проекта появилась на Phase 0–2.

# %%
# --- Установка и загрузка данных (Google Colab) -----------------------------
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
    print(f"Клонирую датасеты в {DATA_DIR} ...")
    subprocess.run(["git", "clone", "--depth", "1", DATA_REPO, str(DATA_DIR)], check=True)

print("Данные:", DATA_DIR.resolve())
print(sorted(p.name for p in DATA_DIR.iterdir() if not p.name.startswith(".")))

# %%
# --- Импорты и общие настройки ---------------------------------------------
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
CLEANING_LOG = []      # ни одна строка не выбрасывается молча

def drop(df, mask, reason):
    """Отфильтровать строки, записав решение в журнал очистки."""
    n = int(mask.sum())
    CLEANING_LOG.append({"датасет": df.attrs.get("name", "?"), "решение": reason,
                         "строк затронуто": n, "осталось": len(df) - n})
    return df[~mask]

# %% [markdown]
# ---
# # Phase 0 — Понимание данных
#
# Первая задача — не модель, а ответ на вопрос **«что представляет собой одна строка?»**.
# В CESAW ответ неочевиден, и цена ошибки здесь выше, чем цена неудачного выбора
# гиперпараметров.
#
# Архив `CESAW.tgz` содержит две таблицы фактов:
#
# * `CESAW_task_fact.csv` — **61 817 строк**, по одной на задачу: план и факт в минутах,
#   человек, проект, команда, фаза процесса, даты начала и завершения;
# * `CESAW_time_fact.csv` — **203 621 строка**, по одной на *сеанс работы*: начало, конец,
#   длительность и **минуты прерываний**.
#
# То есть заявленные «203 621 наблюдение» — это не задачи, а отдельные включения таймера.
# Одна задача — это в среднем 3.3 сеанса, максимум 439.
#
# Связь между таблицами не документирована, её пришлось восстанавливать. Проверка ключей:
#
# | ключ | групп в `task_fact` |
# |---|---:|
# | `(project, wbs, plan_item)` | 51 582 из 61 817 ← **не уникален** |
# | `(project, wbs, plan_item, phase)` | 51 582 ← тоже нет |
# | `(project, wbs, plan_item, phase, person)` | **61 817 = все строки** ✓ |
#
# Значит, естественная единица наблюдения — **задача, выполняемая конкретным человеком
# в конкретной фазе процесса**. Агрегация `time_fact` по этому же ключу воспроизводит
# `task_actual_time_minutes` с точностью до копейки для **99.3 %** задач — это и есть
# доказательство, что ключ восстановлен верно, а не подобран.
#
# ### Диаграмма происхождения данных
#
# ```
# CESAW.tgz
#     ├── CESAW_task_fact.csv   (61 817 задач: план, факт, кто, где, когда)
#     └── CESAW_time_fact.csv   (203 621 сеанс: длительность, прерывания)
#                    │
#                    ▼  агрегация по (project, wbs, plan_item, phase, person)
#             сеансы → на задачу: n сеансов, минуты прерываний
#                    │
#                    ▼  join обратно в task_fact  (99.3 % совпадение факта)
#             task-level dataset
#                    │
#                    ▼  очистка (Phase 1) + производные величины
#             analysis dataset  ──► EDA / гипотезы
#                    │
#                    ▼  leak-free исторические признаки (Phase 5)
#             ML dataset
# ```
#
# **Критично для Phase 5:** число сеансов и минуты прерываний становятся известны
# только *после* выполнения задачи. Они годятся для объяснения (RQ6), но **не могут
# быть признаками** модели, предсказывающей исход до начала работы. Подробнее — в §
# «Утечки».

# %%
def load_cesaw(data_dir=DATA_DIR, work_dir=pathlib.Path("data/_cesaw")):
    """CESAW: собираем task-level датасет из двух таблиц фактов."""
    work_dir.mkdir(parents=True, exist_ok=True)
    task_path = work_dir / "data" / "CESAW_task_fact.csv.xz"
    time_path = work_dir / "data" / "CESAW_time_fact.csv.xz"
    if not task_path.exists():
        with tarfile.open(data_dir / "CESAW.tgz") as t:
            t.extractall(work_dir)

    tasks = pd.read_csv(task_path)
    times = pd.read_csv(time_path)
    print(f"task_fact: {tasks.shape}   time_fact: {times.shape}")

    # phase_key в task_fact прочитан как строка (есть литерал '\N') — приводим к числу.
    tasks["phase_key"] = pd.to_numeric(tasks.phase_key, errors="coerce")
    KEY = ["project_key", "wbs_element_key", "plan_item_key", "phase_key", "person_key"]

    for k in [KEY[:3], KEY[:4], KEY]:
        print(f"  ключ {tuple(x.replace('_key','') for x in k)}: "
              f"{tasks.groupby(k, dropna=False).ngroups} групп из {len(tasks)} строк")

    sessions = times.groupby(KEY).agg(
        n_sessions=("time_log_fact_key", "size"),
        interrupt_min=("time_log_interrupt_minutes", "sum"),
        logged_min=("time_log_delta_minutes", "sum"),
    ).reset_index()

    df = tasks.merge(sessions, on=KEY, how="left")
    reconciled = np.isclose(df.task_actual_time_minutes, df.logged_min.fillna(-1)).mean()
    print(f"  сеансы найдены для {df.n_sessions.notna().mean():.1%} задач; "
          f"sum(сеансы) == task_actual для {reconciled:.1%} -> ключ восстановлен верно")

    df.attrs["name"] = "CESAW"
    return df


cesaw_raw = load_cesaw()

# %% [markdown]
# ---
# # Phase 1 — Очистка
#
# Правило: **ни одна строка не выбрасывается молча.** Каждое решение записывается
# в журнал `CLEANING_LOG` и печатается в конце фазы.
#
# Проверяемые вопросы (из чеклиста Phase 1):
#
# * может ли план быть нулевым? — **да, 692 задачи в CESAW**. Это задачи, которые
#   вообще не оценивали; отношение `actual/estimate` для них не определено;
# * может ли факт быть нулевым или отрицательным? — нет;
# * может ли дата начала отсутствовать? — да, единичные случаи;
# * может ли у одной задачи быть несколько исполнителей? — **да**, и в SiP это
#   создаёт дубли строк (см. ниже);
# * может ли у одной задачи быть несколько записей времени? — да, это и есть `time_fact`.

# %%
def add_targets(df):
    """Производные величины: три формы ошибки оценки."""
    df = df.copy()
    df["ratio"] = df.actual / df.estimate               # estimation_ratio
    df["log_ratio"] = np.log(df.ratio)
    df["log_est"] = np.log(df.estimate)
    df["log_act"] = np.log(df.actual)
    df["abs_error"] = df.actual - df.estimate           # абсолютная ошибка
    df["rel_error"] = df.abs_error / df.estimate        # относительная ошибка
    df["underestimated"] = (df.actual > df.estimate).astype(int)
    df["exact"] = np.isclose(df.actual, df.estimate)
    return df


def clean_cesaw(raw):
    df = raw.copy(); df.attrs["name"] = "CESAW"
    df = drop(df, df.task_plan_time_minutes <= 0,
              "план ≤ 0 (задачу не оценивали) — отношение не определено")
    df = drop(df, df.task_actual_time_minutes <= 0, "факт ≤ 0 — невозможное значение")
    df["date"] = pd.to_datetime(df.task_actual_start_date, errors="coerce")
    df = drop(df, df.date.isna(), "не распознана дата начала")
    df["estimate"] = df.task_plan_time_minutes / 60.0     # всё в часах
    df["actual"] = df.task_actual_time_minutes / 60.0
    df = df.rename(columns={"person_key": "person", "project_key": "project"})
    df["dataset"] = "CESAW"
    return add_targets(df).sort_values("date").reset_index(drop=True)


def load_sip(data_dir=DATA_DIR):
    """SiP: 10k задач коммерческой компании. Главная ловушка — дубли строк."""
    # В файле есть символы cp1252 (типографские кавычки) -> utf-8 падает.
    tasks = pd.read_csv(data_dir / "SiP" / "Sip-task-info.csv", encoding="latin-1")
    dates = pd.read_csv(data_dir / "SiP" / "est-act-dates.csv")
    tasks.attrs["name"] = "SiP"

    n_rows, n_tasks = len(tasks), tasks.TaskNumber.nunique()
    per_task = tasks.groupby("TaskNumber")
    print(f"SiP: {n_rows} строк, но {n_tasks} уникальных задач")
    print(f"  оценка/факт различаются внутри задачи: "
          f"{int((per_task.HoursEstimate.nunique() > 1).sum())} задач -> значения продублированы")
    print(f"  sum(DeveloperHoursActual) == HoursActual: "
          f"{np.isclose(per_task.DeveloperHoursActual.sum(), per_task.HoursActual.first()).mean():.1%} "
          f"-> HoursActual это тотал по задаче, а не по человеку")

    tasks = drop(tasks, tasks.TaskNumber.duplicated(),
                 "дубли строк: многолюдная задача записана по строке на исполнителя, "
                 "оценка и факт в них продублированы")
    dates = dates.drop_duplicates("TaskNumber")
    dates["EstimateOn"] = pd.to_datetime(dates.EstimateOn, format="%d-%b-%y", errors="coerce")

    df = tasks.merge(dates[["TaskNumber", "EstimateOn"]], on="TaskNumber", how="left")
    df.attrs["name"] = "SiP"
    df = drop(df, (df.HoursEstimate <= 0) | (df.HoursActual <= 0), "нулевая оценка или факт")
    df = drop(df, df.EstimateOn.isna(), "нет даты выдачи оценки")
    df = df.rename(columns={"HoursEstimate": "estimate", "HoursActual": "actual",
                            "EstimateOn": "date", "DeveloperID": "person",
                            "ProjectCode": "project"})
    df["dataset"] = "SiP"
    return add_targets(df).sort_values("date").reset_index(drop=True)


def load_renzo(data_dir=DATA_DIR):
    """Renzo Pomodoro: один человек, 10 лет, оценка и факт в помидорах."""
    df = pd.read_csv(data_dir / "renzo-pomodoro.csv"); df.attrs["name"] = "Renzo"
    df = drop(df, df.estimate.isna() | df.actual.isna(), "нет оценки или факта")
    df = drop(df, (df.estimate <= 0) | (df.actual <= 0), "нулевая оценка или факт")
    df = drop(df, df.estimate > 40,
              "оценка > 40 помидоров (в сырых данных есть значения до 5e7) — мусор ввода")
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    df = drop(df, df.date.isna(), "не распознана дата")
    df["person"] = "renzo"; df["project"] = "renzo"; df["dataset"] = "Renzo"
    return add_targets(df).sort_values("date").reset_index(drop=True)


cesaw = clean_cesaw(cesaw_raw)
sip = load_sip()
renzo = load_renzo()

cleaning = pd.DataFrame(CLEANING_LOG)
dump(cleaning, "01_cleaning_log")
print("\nЖУРНАЛ ОЧИСТКИ (ни одна строка не удалена молча):")
print(cleaning.to_string(index=False))
print("\nИтог:", {k: len(v) for k, v in {"CESAW": cesaw, "SiP": sip, "Renzo": renzo}.items()})

# %% [markdown]
# ---
# # Phase 3 — Разведочный анализ
#
# ## 3.1 Обзор датасетов

# %%
DATASETS = [("CESAW", cesaw), ("SiP", sip), ("Renzo", renzo)]

overview = pd.DataFrame({
    name: {
        "задач": len(d),
        "период": f"{d.date.min():%Y}–{d.date.max():%Y}",
        "людей": d.person.nunique(),
        "проектов": d.project.nunique(),
        "медиана оценки": round(d.estimate.median(), 2),
        "медиана факта": round(d.actual.median(), 2),
        "медиана ratio": round(d.ratio.median(), 3),
        "геом. среднее ratio": round(float(np.exp(d.log_ratio.mean())), 3),
        "SD log_ratio": round(float(d.log_ratio.std()), 3),
        "доля недооценённых": f"{d.underestimated.mean():.1%}",
        "доля факт == оценка": f"{d.exact.mean():.1%}",
    } for name, d in DATASETS
})
dump(overview, "02_overview")
print(overview.to_string())

# %%
# Пропуски в исходной таблице CESAW — обязательная картинка чеклиста EDA.
miss = cesaw_raw.isna().mean().sort_values(ascending=False)
miss = miss[miss > 0]
fig, ax = plt.subplots(figsize=(7, 3))
if len(miss):
    ax.barh(np.arange(len(miss)), miss.values, color=ACCENT2)
    ax.set_yticks(np.arange(len(miss))); ax.set_yticklabels(miss.index, fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.set_xlabel("доля пропусков")
else:
    ax.text(.5, .5, "Пропусков нет", ha="center", va="center", fontsize=13, color=MUTED)
    ax.set_axis_off()
ax.set_title("CESAW: пропущенные значения в сырой таблице задач")
save(fig, "01_missing_values"); plt.close(fig)
print("Доля пропусков по колонкам:\n", (miss * 100).round(2).to_string() if len(miss) else "нет")

# %% [markdown]
# ## 3.2 Распределения: почему без логарифмов работать нельзя
#
# Оценка и факт распределены приблизительно логнормально и охватывают 4–5 порядков.
# Среднее в линейной шкале определяется десятком гигантских задач и ничего не описывает.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
for ax, (name, d) in zip(axes, DATASETS):
    lo = max(d.estimate.min(), 1e-2); hi = d[["estimate", "actual"]].max().max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 45)
    ax.hist(d.estimate, bins=bins, alpha=.65, color=ACCENT, label="оценка")
    ax.hist(d.actual, bins=bins, alpha=.55, color=ACCENT2, label="факт")
    ax.set_xscale("log")
    ax.set_title(f"{name}  (N={len(d):,})".replace(",", " "))
    ax.set_xlabel("часы" if name != "Renzo" else "помидоры")
axes[0].set_ylabel("число задач"); axes[0].legend()
fig.suptitle("Оценки и факты логнормальны — весь анализ ведётся в логарифмах",
             y=1.04, fontsize=13, fontweight="bold")
save(fig, "02_distributions"); plt.close(fig)

# %%
# Ключевая диаграмма: факт против оценки. Диагональ = идеальная оценка.
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
for ax, (name, d) in zip(axes, [("CESAW", cesaw), ("SiP", sip)]):
    sub = d.sample(min(len(d), 8000), random_state=RNG)
    ax.scatter(sub.estimate, sub.actual, s=6, alpha=.12, color=ACCENT, edgecolors="none")
    lim = [d.estimate.min() * .7, d.actual.max() * 1.3]
    ax.plot(lim, lim, color=INK, lw=1.3, ls="--", label="факт = оценка")
    q = pd.qcut(d.estimate.rank(method="first"), 8, labels=False)
    med = d.groupby(q).agg(e=("estimate", "median"), a=("actual", "median"))
    ax.plot(med.e, med.a, "o-", color=ACCENT2, lw=2.5, ms=7, label="медиана факта по группам")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("оценка, часы"); ax.set_ylabel("факт, часы")
    ax.set_title(f"{name}: медианная линия положе диагонали")
    ax.legend(loc="upper left", fontsize=9)
fig.suptitle("Выше диагонали — недооценка, ниже — переоценка. "
             "Наклон медианы < 1 — это регрессия к среднему",
             y=1.01, fontsize=12.5, fontweight="bold")
save(fig, "03_estimate_vs_actual"); plt.close(fig)

# %% [markdown]
# ## 3.3 Распределение estimation ratio
#
# Медиана информативнее среднего: распределение скошено настолько, что среднее
# арифметическое `actual/estimate` в CESAW равно 1.49, тогда как медиана — 0.94.
# Первое число описывает хвост, второе — типичную задачу.

# %%
ratio_stats = pd.DataFrame({
    name: {
        "среднее (арифм.)": round(d.ratio.mean(), 3),
        "геом. среднее": round(float(np.exp(d.log_ratio.mean())), 3),
        "медиана": round(d.ratio.median(), 3),
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
    ax.hist(lr[~d.exact], bins=70, color=ACCENT, alpha=.85, label="факт ≠ оценка")
    ax.hist(lr[d.exact], bins=70, color=ACCENT2, alpha=.95,
            label=f"факт == оценка ({d.exact.mean():.0%})")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xticks(np.log([1/8, 1/4, 1/2, 1, 2, 4, 8]))
    ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.set_xlabel("факт / оценка (лог-шкала)")
    ax.set_title(name); ax.legend(fontsize=8)
axes[0].set_ylabel("число задач")
fig.suptitle("Распределение ошибки оценки: широкое, скошенное, с искусственным пиком на 1.0",
             y=1.04, fontsize=13, fontweight="bold")
save(fig, "04_ratio_distribution"); plt.close(fig)

# %% [markdown]
# ---
# # Phase 4 — Проверка гипотез (RQ1–RQ7)
#
# ## Сначала — ловушка, которая портит все метрики
#
# У **8 % задач CESAW, 34 % SiP и 44 % Renzo** факт в точности равен оценке.
# Это не идеальные оценки: доля точных совпадений резко падает с ростом задачи —
# списать время «по плану» легко на получасовой задаче и невозможно на двухнедельной.
# Это учётная практика, а не точность.
#
# Почему это важно именно для ML: baseline «факт = оценка» получает на этих строках
# нулевую ошибку бесплатно, и по медианным метрикам его становится почти невозможно
# побить. Поэтому ключевые результаты считаются **дважды** — на всех данных
# и на подвыборке без точных совпадений. В CESAW эффект слабее, что и делает его
# более честным основным датасетом, чем SiP.

# %%
SIZE_BINS = [0, .5, 1, 2, 4, 8, 16, np.inf]
SIZE_LABELS = ["≤0.5ч", "0.5–1ч", "1–2ч", "2–4ч", "4–8ч", "8–16ч", ">16ч"]
for d in (cesaw, sip):
    d["size_bin"] = pd.cut(d.estimate, SIZE_BINS, labels=SIZE_LABELS)

exact_by_size = pd.DataFrame({
    name: d.groupby("size_bin", observed=True).exact.mean().round(3)
    for name, d in [("CESAW", cesaw), ("SiP", sip)]})
dump(exact_by_size, "04_exact_by_size")
print("Доля задач с факт == оценка, по размеру:\n", exact_by_size.to_string())

fig, ax = plt.subplots(figsize=(7.5, 3.8))
x = np.arange(len(exact_by_size))
ax.bar(x - .2, exact_by_size.CESAW, .4, color=ACCENT, label="CESAW")
ax.bar(x + .2, exact_by_size.SiP, .4, color=ACCENT2, label="SiP")
ax.set_xticks(x); ax.set_xticklabels(exact_by_size.index, fontsize=9)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylabel("доля задач с факт == оценка"); ax.set_xlabel("размер задачи по оценке")
ax.set_title("«Идеальные оценки» — артефакт учёта: они исчезают на больших задачах")
ax.legend()
save(fig, "05_exact_match_artifact"); plt.close(fig)

# С этого момента ne_* = подвыборки без точных совпадений ("честные" данные).
cesaw_ne, sip_ne, renzo_ne = cesaw[~cesaw.exact], sip[~sip.exact], renzo[~renzo.exact]

# %% [markdown]
# ## RQ1 — Есть ли систематическое смещение?
#
# Гипотеза: разработчики систематически недооценивают задачи.
#
# Проверяем знаковым тестом (доля `ratio > 1` против 50 %) и t-тестом на `log_ratio`
# против нуля — второй эквивалентен проверке «геометрическое среднее = 1».

# %%
rq1 = {}
for name, d in [("CESAW", cesaw_ne), ("SiP", sip_ne), ("Renzo", renzo_ne)]:
    n_under = int(d.underestimated.sum()); n = len(d)
    binom = stats.binomtest(n_under, n, .5)
    t = stats.ttest_1samp(d.log_ratio, 0)
    rq1[name] = {
        "N": n,
        "доля недооценённых": round(n_under / n, 3),
        "p (знаковый тест vs 50%)": f"{binom.pvalue:.2e}",
        "геом. среднее ratio": round(float(np.exp(d.log_ratio.mean())), 3),
        "p (log_ratio vs 0)": f"{t.pvalue:.2e}",
        "вывод": ("недооценка" if d.log_ratio.mean() > 0 else "ПЕРЕоценка"),
    }
rq1_tbl = pd.DataFrame(rq1)
dump(rq1_tbl, "05_rq1_systematic_bias")
print(rq1_tbl.to_string())

# %% [markdown]
# **Ответ на RQ1: гипотеза не подтвердилась — в агрегате смещение направлено
# в противоположную сторону.** Во всех трёх датасетах геометрическое среднее
# `actual/estimate` **меньше единицы** (CESAW ×0.80, SiP ×0.86, Renzo ×0.95),
# и лишь 42–49 % задач превышают оценку.
#
# Стоит отметить расхождение двух тестов на Renzo: знаковый тест не отвергает
# гипотезу о 50 % (`p = 0.20`), а t-тест на `log_ratio` — отвергает (`p = 8e−7`).
# Противоречия здесь нет: задачи *чаще* превышают оценку примерно в половине
# случаев, но по величине превышения меньше занижений. Один человек за 10 лет
# оказался откалиброван по частоте, но не по величине.
#
# Это не значит, что «разработчики оценивают с запасом». Как показывает RQ2,
# агрегат скрывает смену знака и потому вводит в заблуждение — классический
# случай, когда средним пользоваться нельзя.
#
# ## RQ2 — Зависит ли ошибка от размера задачи?
#
# Гипотеза: чем крупнее задача, тем сильнее недооценка.

# %%
def size_profile(d):
    g = d.groupby("size_bin", observed=True)
    return pd.DataFrame({
        "N": g.size(),
        "геом. среднее ratio": g.log_ratio.apply(lambda x: np.exp(x.mean())),
        "медиана ratio": g.ratio.median(),
        "доля недооценённых": g.underestimated.mean(),
        "SD log_ratio": g.log_ratio.std(),
    }).round(3)

rq2 = {name: size_profile(d) for name, d in [("CESAW", cesaw_ne), ("SiP", sip_ne)]}
rq2_tbl = pd.concat(rq2, names=["датасет"])
dump(rq2_tbl, "06_rq2_size_effect")
print(rq2_tbl.to_string())

# Формальный тест: наклон log_ratio ~ log_est. Отрицательный = регрессия к среднему.
slopes = {}
for name, d in [("CESAW", cesaw_ne), ("SiP", sip_ne), ("Renzo", renzo_ne)]:
    r = stats.linregress(d.log_est, d.log_ratio)
    slopes[name] = {"наклон": round(r.slope, 3), "SE": round(r.stderr, 4),
                    "p-value": f"{r.pvalue:.1e}", "R²": round(r.rvalue ** 2, 3), "N": len(d)}
slopes_tbl = pd.DataFrame(slopes).T
dump(slopes_tbl, "06_rq2_slopes")
print("\nНаклон log_ratio ~ log_est (отрицательный = регрессия к среднему):")
print(slopes_tbl.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ax = axes[0]
x = np.arange(len(SIZE_LABELS))
for (name, t), c, m in zip(rq2.items(), [ACCENT, ACCENT2], ["o", "s"]):
    ax.plot(x, t["геом. среднее ratio"], m + "-", color=c, lw=2.4, ms=7, label=name)
ax.axhline(1, color=INK, lw=1, ls="--")
ax.set_xticks(x); ax.set_xticklabels(SIZE_LABELS, fontsize=9)
ax.set_ylabel("геом. среднее факт / оценка"); ax.set_xlabel("размер задачи по оценке")
ax.set_title("Смещение меняет знак с размером задачи")
ax.annotate("недооценка", (0.05, 1.35), color=ACCENT2, fontweight="bold", fontsize=9)
ax.annotate("переоценка", (4.4, 0.62), color=ACCENT, fontweight="bold", fontsize=9)
ax.legend()

ax = axes[1]
for (name, t), c, m in zip(rq2.items(), [ACCENT, ACCENT2], ["o", "s"]):
    ax.plot(x, t["SD log_ratio"], m + "-", color=c, lw=2.4, ms=7, label=name)
ax.set_xticks(x); ax.set_xticklabels(SIZE_LABELS, fontsize=9)
ax.set_ylabel("SD log(факт/оценка)"); ax.set_xlabel("размер задачи по оценке")
ax.set_title("Неопределённость растёт вместе с размером")
ax.legend()
fig.suptitle("RQ2: гипотеза «крупные задачи недооценивают сильнее» ОПРОВЕРГНУТА — "
             "всё ровно наоборот", y=1.03, fontsize=12.5, fontweight="bold")
save(fig, "06_size_effect"); plt.close(fig)

# %% [markdown]
# **Ответ на RQ2: гипотеза опровергнута, и знак противоположен ожидаемому.**
# Наклон `log_ratio ~ log_est` отрицателен во всех трёх датасетах
# (CESAW −0.17, SiP −0.24, Renzo −0.81; все `p < 1e−140`): чем крупнее задача,
# тем **меньше** отношение факт/оценка.
#
# Сила эффекта различается. В SiP смещение буквально меняет знак — от ×1.30
# на задачах ≤ 0.5 ч до ×0.51 на задачах > 16 ч. В CESAW знак не меняется
# (там переоценка на всём диапазоне), но тренд тот же: от ×0.98 до ×0.57.
# У одного человека (Renzo) наклон в 3–5 раз круче, чем у организаций: без
# организационного усреднения регрессия к среднему видна ярче всего.
#
# Механизм — **регрессия к среднему**: оценка это шумный сигнал, и экстремальные
# оценки в обе стороны систематически «стягиваются» к типичной длительности задачи.
# Практический вывод обратен фольклору «умножай на два»: **умножать надо мелкие
# задачи, а крупные — делить**, и единый коэффициент запаса на всю команду вреден,
# потому что ошибается в разные стороны на разных задачах.
#
# ## RQ3–RQ5 — Кто виноват: исполнитель, проект или тип работы?
#
# Вместо того чтобы разглядывать групповые средние, спросим количественно:
# **сколько дисперсии `log_ratio` объясняет каждый фактор?**
#
# Здесь легко себя обмануть. У признака `person` в CESAW 246 уровней — это 246
# дамми-переменных, и обычный R² механически растёт с их числом. Поэтому считаем
# три версии: обычный R², скорректированный на число параметров и, главное,
# **out-of-sample R² по 5-фолдовой кросс-валидации** — единственный, который
# нельзя накрутить количеством уровней. Плюс строку «все факторы вместе»,
# чтобы понять потолок объяснимости.

# %%
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def variance_explained(d, factors, target="log_ratio"):
    """R² при подгонке одним категориальным признаком.

    Обычный R² завышен для факторов с большим числом уровней (у `person` в CESAW
    их 246 — это 246 дамми-переменных), поэтому считаем ещё две честные версии:
    скорректированный R² и out-of-sample R² по 5-фолдовой кросс-валидации.
    Последний — единственный, который нельзя накрутить числом уровней.
    """
    from sklearn.model_selection import cross_val_score, KFold
    y = d[target].values
    n = len(d)
    rows = {}
    for f in list(factors) + ["ВСЕ ФАКТОРЫ ВМЕСТЕ"]:
        cols = list(factors) if f == "ВСЕ ФАКТОРЫ ВМЕСТЕ" else [f]
        X = OneHotEncoder(handle_unknown="ignore").fit_transform(d[cols].astype(str))
        r2 = LinearRegression().fit(X, y).score(X, y)
        p_ = X.shape[1]
        cv = cross_val_score(Ridge(alpha=1.0), X, y,
                             cv=KFold(5, shuffle=True, random_state=RNG), scoring="r2").mean()
        rows[f] = {"уровней": int(sum(d[c].nunique() for c in cols)),
                   "R²": round(r2, 4),
                   "R² скорр.": round(1 - (1 - r2) * (n - 1) / max(n - p_ - 1, 1), 4),
                   "R² out-of-sample": round(float(cv), 4)}
    out = pd.DataFrame(rows).T
    tail = out.loc[["ВСЕ ФАКТОРЫ ВМЕСТЕ"]]
    return pd.concat([out.drop(index="ВСЕ ФАКТОРЫ ВМЕСТЕ")
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
                      "геом. среднее": g.log_ratio.apply(lambda x: np.exp(x.mean())),
                      "медиана": g.ratio.median(),
                      "доля недооценённых": g.underestimated.mean(),
                      "SD log": g.log_ratio.std()})
    t["SE"] = t["SD log"] / np.sqrt(t.N)
    t["CI низ"] = np.exp(np.log(t["геом. среднее"]) - 1.96 * t.SE)
    t["CI верх"] = np.exp(np.log(t["геом. среднее"]) + 1.96 * t.SE)
    return t[t.N >= min_n].sort_values("геом. среднее")

by_phase = bias_table(cesaw_ne, "phase_short_name", 200)
by_person = bias_table(cesaw_ne, "person", 200)
by_project = bias_table(cesaw_ne, "project", 200)
for t, n in [(by_phase, "08_rq5_by_phase"), (by_person, "08_rq3_by_person"),
             (by_project, "08_rq4_by_project")]:
    dump(t.round(3), n)
print("CESAW, смещение по фазе процесса:\n", by_phase.round(2).to_string())

# ANOVA-подобная проверка: значимы ли различия между группами вообще?
anova = {}
for label, key in [("исполнитель (RQ3)", "person"), ("проект (RQ4)", "project"),
                   ("фаза процесса (RQ5)", "phase_short_name")]:
    groups = [g.log_ratio.values for _, g in cesaw_ne.groupby(key) if len(g) >= 30]
    f, p = stats.f_oneway(*groups)
    kw = stats.kruskal(*groups)          # непараметрический дубль: log_ratio не нормален
    anova[label] = {"групп": len(groups), "F": round(f, 1), "p (ANOVA)": f"{p:.1e}",
                    "p (Kruskal-Wallis)": f"{kw.pvalue:.1e}",
                    "R² out-of-sample": round(float(ve_cesaw.loc[key, "R² out-of-sample"]), 4)}
anova_tbl = pd.DataFrame(anova).T
dump(anova_tbl, "08_rq345_anova")
print("\nРазличия между группами значимы, но объясняют мало:")
print(anova_tbl.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
for ax, (t, title) in zip(axes, [(by_phase, "по фазе процесса (RQ5)"),
                                 (by_person.tail(20), "по исполнителю (RQ3), топ-20")]):
    y = np.arange(len(t))
    colors = [ACCENT2 if v > 1 else ACCENT for v in t["геом. среднее"]]
    ax.hlines(y, t["CI низ"], t["CI верх"], color=colors, lw=2, alpha=.5)
    ax.scatter(t["геом. среднее"], y, color=colors, s=30, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([str(i) for i in t.index], fontsize=8)
    ax.axvline(1, color=INK, lw=1, ls="--")
    ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["×0.25", "×0.5", "×1", "×2", "×4"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_title(f"Смещение оценки {title}")
    ax.set_xlabel("геом. среднее факт / оценка (95 % ДИ)")
fig.suptitle("CESAW: различия значимы, но все факторы вместе объясняют лишь 17 % дисперсии",
             y=1.02, fontsize=12.5, fontweight="bold")
save(fig, "07_bias_forest"); plt.close(fig)

# %% [markdown]
# **Ответ на RQ3–RQ5.** Все три фактора дают статистически значимые различия
# (`p < 1e−50` и по ANOVA, и по Краскелу–Уоллису), но при 55 тысячах наблюдений
# значимо почти всё — смотреть надо на величину эффекта.
#
# | фактор | CESAW, R² out-of-sample | SiP, R² out-of-sample |
# |---|---:|---:|
# | исполнитель | **0.103** | 0.026 |
# | фаза / тип работы | 0.040 | 0.023 |
# | проект | 0.030 | 0.012 |
# | размер задачи | 0.021 | **0.080** |
# | **все факторы вместе** | **0.170** | **0.137** |
#
# Два вывода, и оба важные.
#
# **Первое: доминирующий фактор различается между организациями.** В CESAW сильнее
# всего исполнитель (10 % дисперсии, эффект переживает и корректировку на число
# уровней, и кросс-валидацию — значит, он настоящий, а не артефакт 246 дамми).
# В SiP наоборот: исполнитель почти ничего не даёт, а размер задачи объясняет 8 %.
# Переносить «у нас главное — кто оценивает» с одной компании на другую нельзя;
# это свойство организации, а не разработки как таковой.
#
# **Второе, более важное: потолок низкий.** Все наблюдаемые факторы вместе
# объясняют 17 % дисперсии в CESAW и 14 % в SiP. То есть **83–86 % дисперсии
# ошибки оценки не объясняется ничем из того, что записано в данных.**
#
# Это главный отрезвляющий результат проекта, и из него прямо следует Phase 9:
# если пять шестых дисперсии нередуцируемы, точечный прогноз обречён независимо
# от модели, и правильный продукт — интервал.
#
# ## RQ6 — Влияют ли прерывания?
#
# Здесь нужна осторожность. Прерывания известны только **после** выполнения задачи,
# поэтому это вопрос *объяснения*, а не прогноза, и в признаки модели они не пойдут
# (см. Phase 5). Кроме того, длинные задачи прерываются чаще просто потому, что они
# длиннее — поэтому сравнение обязательно **внутри групп по размеру**.

# %%
c = cesaw_ne[cesaw_ne.n_sessions.notna()].copy()
c["interrupted"] = np.where(c.interrupt_min.fillna(0) > 0, "были прерывания", "без прерываний")
c["sess_bin"] = pd.cut(c.n_sessions, [0, 1, 2, 4, 8, 16, 1e9],
                       labels=["1", "2", "3–4", "5–8", "9–16", ">16"])

rq6_raw = c.groupby("interrupted").agg(
    N=("log_ratio", "size"), geo=("log_ratio", lambda x: np.exp(x.mean())),
    med_est=("estimate", "median"), under=("underestimated", "mean")).round(3)
rq6_ctrl = c.groupby(["size_bin", "interrupted"], observed=True).agg(
    N=("log_ratio", "size"), geo=("log_ratio", lambda x: np.exp(x.mean()))).round(3)
rq6_sess = c.groupby("sess_bin", observed=True).agg(
    N=("log_ratio", "size"), geo=("log_ratio", lambda x: np.exp(x.mean())),
    med_est=("estimate", "median")).round(3)
for t, n in [(rq6_raw, "09_rq6_interrupt_raw"), (rq6_ctrl, "09_rq6_interrupt_by_size"),
             (rq6_sess, "09_rq6_sessions")]:
    dump(t, n)
print("Сырое сравнение (осторожно — размер задачи не контролируется):\n", rq6_raw.to_string())
print("\nВнутри групп по размеру:\n", rq6_ctrl.to_string())
print("\nПо числу рабочих сеансов:\n", rq6_sess.to_string())

piv = rq6_ctrl.geo.unstack()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
x = np.arange(len(piv))
ax.bar(x - .2, piv["без прерываний"], .4, color=ACCENT, label="без прерываний")
ax.bar(x + .2, piv["были прерывания"], .4, color=ACCENT2, label="были прерывания")
ax.axhline(1, color=INK, lw=1, ls="--")
ax.set_xticks(x); ax.set_xticklabels(piv.index, fontsize=9)
ax.set_ylabel("геом. среднее факт / оценка"); ax.set_xlabel("размер задачи по оценке")
ax.set_title("Эффект прерываний сохраняется внутри каждой группы по размеру")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(np.arange(len(rq6_sess)), rq6_sess.geo, "o-", color=ACCENT4, lw=2.4, ms=7)
ax.axhline(1, color=INK, lw=1, ls="--")
ax.set_xticks(np.arange(len(rq6_sess))); ax.set_xticklabels(rq6_sess.index)
ax.set_ylabel("геом. среднее факт / оценка")
ax.set_xlabel("число рабочих сеансов на задачу")
ax.set_title("Чем сильнее фрагментирована работа, тем больше перерасход")
fig.suptitle("RQ6: прерывания — сильнейший из найденных факторов, "
             "но он известен только постфактум", y=1.03, fontsize=12.5, fontweight="bold")
save(fig, "08_interruptions"); plt.close(fig)

# %% [markdown]
# **Ответ на RQ6.** Внутри **каждой** группы по размеру задачи прерванные задачи
# перерасходуют примерно вдвое сильнее непрерванных (например, в группе >16 ч:
# ×0.99 против ×0.55). Эффект не сводится к размеру и по величине превосходит
# всё, что дали RQ3–RQ5.
#
# Но выводы из этого делать надо аккуратно. Направление причинности не установлено:
# прерывания могут удлинять задачу — а могут быть *следствием* того, что задача
# оказалась тяжелее ожидаемого и растянулась на много дней. Данные не позволяют
# различить эти объяснения. И для прогноза это всё равно бесполезно: в момент
# оценки мы не знаем, сколько будет прерываний.
#
# ## RQ7 — Влияют ли «круглые» оценки?
#
# Исходная работа по CESAW отмечает склонность к круглым числам как фактор,
# связанный с точностью. Проверим — и заодно покажем, почему такие эффекты нельзя
# принимать на веру.
#
# Сначала посмотрим, что вообще считать «круглым»: определять это до того, как
# увидел распределение, нельзя.

# %%
top_est = cesaw.task_plan_time_minutes.value_counts().head(15)
print("Самые частые оценки CESAW (минуты):\n", top_est.to_string())
print(f"\nТоп-15 значений покрывают {top_est.sum() / len(cesaw):.1%} всех оценок")

def multiple_of(x, m, tol=1e-6):
    r = (np.asarray(x, dtype=float) / m) % 1
    return np.isclose(r, 0, atol=tol) | np.isclose(r, 1, atol=tol)

for d, col, unit in [(cesaw, "task_plan_time_minutes", "мин"), (sip, "estimate", "ч")]:
    d["is_round"] = ((d[col] % 30 == 0) | (d[col] % 60 == 0)).astype(int) if unit == "мин" \
        else multiple_of(d[col], 1).astype(int)
cesaw_ne, sip_ne = cesaw[~cesaw.exact], sip[~sip.exact]

# Гранулярность: у оценок она грубее, чем у фактов — прямое свидетельство якорения.
gran = pd.DataFrame({
    "оценки": {"уникальных значений": cesaw.task_plan_time_minutes.nunique(),
               "топ-10 покрывают": f"{cesaw.task_plan_time_minutes.value_counts(normalize=True).head(10).sum():.1%}",
               "кратно 30 или 60 мин": f"{cesaw.is_round.mean():.1%}"},
    "факты": {"уникальных значений": cesaw.task_actual_time_minutes.nunique(),
              "топ-10 покрывают": f"{cesaw.task_actual_time_minutes.value_counts(normalize=True).head(10).sum():.1%}",
              "кратно 30 или 60 мин": f"{((cesaw.task_actual_time_minutes % 30 == 0) | (cesaw.task_actual_time_minutes % 60 == 0)).mean():.1%}"},
})
dump(gran, "10_rq7_granularity")
print("\n", gran.to_string())

rq7_raw = cesaw_ne.groupby("is_round").agg(
    N=("log_ratio", "size"), geo=("log_ratio", lambda x: np.exp(x.mean())),
    med_est=("estimate", "median"), under=("underestimated", "mean")).round(3)
rq7_ctrl = cesaw_ne.groupby(["size_bin", "is_round"], observed=True).agg(
    N=("log_ratio", "size"), geo=("log_ratio", lambda x: np.exp(x.mean()))).round(3)
dump(rq7_raw, "10_rq7_round_raw"); dump(rq7_ctrl, "10_rq7_round_by_size")
print("\nСырое сравнение круглых и некруглых оценок:\n", rq7_raw.to_string())
print("\nВнутри групп по размеру:\n", rq7_ctrl.to_string())

piv7 = rq7_ctrl.geo.unstack()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
vals = cesaw.task_plan_time_minutes.value_counts().head(15).sort_index()
rnd = ((vals.index % 30 == 0) | (vals.index % 60 == 0))
ax.bar(np.arange(len(vals)), vals.values,
       color=[ACCENT3 if r else ACCENT for r in rnd])
ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels([f"{v:g}" for v in vals.index], fontsize=8)
ax.set_xlabel("оценка, минут"); ax.set_ylabel("число задач")
ax.set_title("Оценки садятся на сетку: 30, 60, 120, 180 мин")
ax.bar(0, 0, color=ACCENT3, label="кратно 30/60 мин"); ax.legend(fontsize=9)

ax = axes[1]
x = np.arange(len(piv7))
ax.bar(x - .2, piv7[0], .4, color=ACCENT, label="некруглая оценка")
ax.bar(x + .2, piv7[1], .4, color=ACCENT3, label="круглая оценка")
ax.axhline(1, color=INK, lw=1, ls="--")
ax.set_xticks(x); ax.set_xticklabels(piv7.index, fontsize=9)
ax.set_ylabel("геом. среднее факт / оценка"); ax.set_xlabel("размер задачи по оценке")
ax.set_title("...но эффект круглых чисел внутри групп непоследователен")
ax.legend(fontsize=9)
fig.suptitle("RQ7: сырой эффект круглых оценок — в основном артефакт размера задачи",
             y=1.03, fontsize=12.5, fontweight="bold")
save(fig, "09_round_numbers"); plt.close(fig)

# %% [markdown]
# **Ответ на RQ7 — и лучший пример в проекте, почему нельзя останавливаться
# на первом сравнении.**
#
# Сырое сравнение выглядит убедительно: круглые оценки дают ×0.77, некруглые ×0.83.
# Казалось бы, эффект найден. Но медианная круглая оценка — 1 час, а некруглая —
# 0.72 часа: **круглые оценки просто чаще ставят на более крупные задачи**, а по RQ2
# крупные задачи переоцениваются. Как только мы сравниваем внутри групп по размеру,
# направление эффекта перестаёт быть последовательным: в одних группах круглые
# оценки точнее, в других — хуже.
#
# Вывод: **наблюдаемый эффект круглых чисел объясняется размером задачи, а не
# «округлостью» как таковой.** Это ровно тот случай, о котором предупреждает
# методологическая рамка проекта: association ≠ causation, и конфаундер здесь —
# размер задачи.
#
# ### Промежуточный итог по гипотезам

# %%
verdicts = pd.DataFrame([
    ("RQ1", "Есть систематическая недооценка", "ОПРОВЕРГНУТА",
     "В агрегате геом. среднее < 1 во всех трёх датасетах (×0.80 / ×0.86 / ×0.95)"),
    ("RQ2", "Крупные задачи недооценивают сильнее", "ОПРОВЕРГНУТА, знак обратный",
     "Регрессия к среднему: мелкие недооценены, крупные переоценены; наклон < 0, p<1e−140"),
    ("RQ3", "Точность различается между исполнителями", "ПОДТВЕРЖДЕНА (CESAW)",
     "Сильнейший фактор в CESAW: R² out-of-sample 0.103. Но в SiP лишь 0.026 — "
     "доминирующий фактор зависит от организации"),
    ("RQ4", "Проект объясняет часть вариации", "подтверждена, эффект мал",
     "p < 1e−50, R² out-of-sample 0.030"),
    ("RQ5", "Тип работы (фаза) влияет", "подтверждена, эффект мал",
     "p < 1e−50, R² out-of-sample 0.040"),
    ("RQ6", "Прерывания связаны с ошибкой", "ПОДТВЕРЖДЕНА, сильный эффект",
     "×~2 внутри каждой группы по размеру; но известно только постфактум"),
    ("RQ7", "Круглые оценки менее точны", "НЕ подтверждена",
     "Сырой эффект исчезает при контроле размера задачи — конфаундер"),
], columns=["RQ", "гипотеза", "вердикт", "основание"])
dump(verdicts.set_index("RQ"), "11_hypotheses_verdicts")
print(verdicts.to_string(index=False))

# %% [markdown]
# ---
# # Phase 5 — Feature engineering и защита от утечек
#
# ## Что можно использовать, а что нельзя
#
# Модель должна работать в момент, когда оценка уже дана, а работа ещё не начата.
# Всё, что становится известно позже, — утечка.
#
# | признак | доступен до старта? | решение |
# |---|---|---|
# | оценка, проект, фаза, исполнитель, команда, процесс | да | **используем** |
# | календарные признаки даты оценки | да | используем |
# | «оценка круглая» | да | используем |
# | историческая точность исполнителя/проекта/фазы | да, если считать только по прошлому | **используем (осторожно)** |
# | `task_actual_time_minutes` | нет | таргет |
# | `n_sessions`, `interrupt_min` | **нет** | исключены, хотя это сильнейший фактор (RQ6) |
# | `task_actual_complete_date` | нет | исключена |
#
# Отдельно про SiP: поля `DeveloperHoursActual`, `TaskPerformance` и
# `DeveloperPerformance` — прямые функции таргета. Их включение дало бы R² ≈ 1
# и полностью бессмысленную модель.
#
# ## Исторические признаки: где легко ошибиться
#
# Самый ценный признак — «насколько этот человек ошибался раньше». Но посчитать его
# наивно (`groupby(person).log_ratio.mean()`) значит использовать будущее: для задачи
# №500 в среднее попадут задачи №501–900.
#
# Правильно — **расширяющееся среднее со сдвигом на один шаг**, в хронологическом
# порядке: для каждой задачи берутся только строго предшествующие ей задачи того же
# человека. Первая задача каждого исполнителя получает `NaN` — и это честно.

# %%
HIST_SPECS = [("person", "person"), ("project", "project"), ("phase_short_name", "phase")]

def add_history(df):
    """Leak-free исторические признаки: только строго прошлые задачи."""
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

# Проверка отсутствия утечки: у первой задачи каждого исполнителя истории быть не должно.
first_task = model_df.groupby("person").head(1)
assert first_task.person_hist_lr.isna().all(), "утечка: у первой задачи есть история!"
print(f"Проверка пройдена: у всех {len(first_task)} первых задач исполнителей история = NaN")
print(f"Покрытие person_hist_lr: {model_df.person_hist_lr.notna().mean():.1%} задач")

CAT_FEATURES = ["phase_short_name", "process_name", "project", "person", "team_key"]
BASE_FEATURES = ["log_est", "is_round", "month", "dow"]
HIST_FEATURES = ["person_hist_lr", "person_hist_n", "project_hist_lr", "phase_hist_lr",
                 "person_hist_under", "person_recent_lr"]
FEATURES = CAT_FEATURES + BASE_FEATURES + HIST_FEATURES
print(f"\nПризнаков: {len(FEATURES)} = {len(CAT_FEATURES)} категориальных + "
      f"{len(BASE_FEATURES)} базовых + {len(HIST_FEATURES)} исторических")

# %% [markdown]
# ---
# # Phase 6 — Baseline и стратегии разбиения
#
# ## Baseline, который надо побить
#
# `факт = оценка`, то есть «просто поверь разработчику». Это не соломенное чучело:
# бесплатный, всегда доступный и, как выяснится, очень сильный.
#
# ## Три стратегии разбиения — и почему это не деталь
#
# Данные содержат повторные наблюдения по людям и проектам. Случайное разбиение
# позволяет модели увидеть *того же исполнителя в том же проекте* и в train, и в test.
# Это не утечка таргета, но результат становится оптимистичным.
#
# | | что проверяет | реалистичность |
# |---|---|---|
# | **A. Случайное** | базовая обучаемость | оптимистично |
# | **B. По проектам** | перенос на новый проект | строгая проверка обобщения |
# | **C. Временное** | предсказание будущего по прошлому | **соответствует проду** |
#
# Считаем все три и смотрим, насколько расходятся. Дальше по умолчанию используется C.

# %%
from sklearn.ensemble import (HistGradientBoostingRegressor, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix, roc_curve,
                             precision_recall_curve)
from sklearn.inspection import permutation_importance

def prep(train, test, feats=FEATURES, cats=CAT_FEATURES):
    """HistGBM требует согласованных категорий между train и test."""
    a, b = train[feats].copy(), test[feats].copy()
    for c in cats:
        a[c] = a[c].astype(str).astype("category")
        b[c] = pd.Categorical(b[c].astype(str), categories=a[c].cat.categories)
    return a, b, [a.columns.get_loc(c) for c in cats]

def reg_metrics(pred_log, test, name):
    err = pred_log - test.log_act.values
    mae = float(np.abs(err).mean())
    mre = np.abs(np.exp(pred_log) - test.actual.values) / test.actual.values
    return {"модель": name, "MAE_log": round(mae, 3), "типичная ошибка": f"×{np.exp(mae):.2f}",
            "RMSE_log": round(float(np.sqrt((err ** 2).mean())), 3),
            "MdMRE": round(float(np.median(mre)), 3),
            "PRED(25)": round(float((mre <= .25).mean()), 3)}

def clf_metrics(y, proba, name, thr=.5):
    yh = (proba >= thr).astype(int)
    return {"модель": name, "accuracy": round(accuracy_score(y, yh), 3),
            "precision": round(precision_score(y, yh, zero_division=0), 3),
            "recall": round(recall_score(y, yh), 3), "F1": round(f1_score(y, yh), 3),
            "ROC-AUC": round(roc_auc_score(y, proba), 3),
            "PR-AUC": round(average_precision_score(y, proba), 3)}

def make_splits(df):
    n = len(df)
    idx = np.random.RandomState(RNG).permutation(n); cut = int(n * .75)
    yield "A: случайное", df.iloc[np.sort(idx[:cut])], df.iloc[np.sort(idx[cut:])]

    projects = df.project.value_counts().index.tolist()
    rs = np.random.RandomState(RNG); rs.shuffle(projects)
    test_projects, total = [], 0
    for p in projects:
        if total < n * .25:
            test_projects.append(p); total += int((df.project == p).sum())
    hold = df.project.isin(test_projects)
    yield "B: новые проекты", df[~hold], df[hold]

    cut_date = df.date.quantile(.75)
    yield "C: временное", df[df.date <= cut_date], df[df.date > cut_date]

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
    split_rows.append({"стратегия": name, "N train": len(tr), "N test": len(te),
                       "B0 RMSE_log": b0["RMSE_log"], "GBM RMSE_log": m["RMSE_log"],
                       "выигрыш RMSE": f"{100*(1-m['RMSE_log']/b0['RMSE_log']):.1f} %",
                       "ROC-AUC": c["ROC-AUC"], "PR-AUC": c["PR-AUC"],
                       "base rate": round(te.underestimated.mean(), 3)})
splits_tbl = pd.DataFrame(split_rows)
dump(splits_tbl, "12_split_strategies")
print(splits_tbl.to_string(index=False))

# %% [markdown]
# **Стратегия разбиения меняет выводы сильнее, чем выбор модели.** Случайное
# разбиение даёт заметно более радужную картину, чем временное; разбиение по
# проектам — промежуточную. Если бы мы отчитались по случайному сплиту, мы бы
# завысили и ROC-AUC, и выигрыш по RMSE. Дальше везде используется **временное
# разбиение (C)** как единственное, соответствующее реальному сценарию.

# %%
CUT_DATE = model_df.date.quantile(.75)
train, test = model_df[model_df.date <= CUT_DATE], model_df[model_df.date > CUT_DATE]
Xtr, Xte, CAT_IDX = prep(train, test)
print(f"Train: {len(train)} задач до {CUT_DATE:%Y-%m-%d}   Test: {len(test)} задач после")
print(f"Доля недооценённых: train {train.underestimated.mean():.3f}, "
      f"test {test.underestimated.mean():.3f}")

# %% [markdown]
# ---
# # Phase 7 — Машинное обучение
#
# ## 7.1 Задача 1: регрессия — предсказать величину ошибки
#
# Предсказываем `log_ratio`, а не `log(actual)` напрямую: так модель учит **поправку
# к оценке**, а не заново выводит длительность задачи, и оценка разработчика
# остаётся в прогнозе как опорная точка.
#
# Идём по возрастанию сложности: baseline → глобальный коэффициент → лог-линейная
# калибровка → Ridge → бустинг. Более сложная модель принимается, только если
# даёт заметный прирост.

# %%
reg_rows = [reg_metrics(np.full(len(test), train.log_act.mean()), test, "константа"),
            reg_metrics(test.log_est.values, test, "B0: факт = оценка")]
shift = train.log_ratio.mean()
reg_rows.append(reg_metrics(test.log_est.values + shift, test,
                            f"B1: глобальный коэффициент ×{np.exp(shift):.2f}"))
a, b = np.polyfit(train.log_est, train.log_act, 1)
reg_rows.append(reg_metrics(a * test.log_est.values + b, test,
                            f"B2: лог-линейная калибровка (наклон {a:.2f})"))

ridge_pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_FEATURES),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), BASE_FEATURES + HIST_FEATURES)])
ridge = Pipeline([("pre", ridge_pre), ("m", Ridge(alpha=1.0))]).fit(train[FEATURES], train.log_ratio)
reg_rows.append(reg_metrics(test.log_est.values + ridge.predict(test[FEATURES]), test,
                            "M1: Ridge + признаки"))

gbm_reg = HistGradientBoostingRegressor(categorical_features=CAT_IDX, max_iter=400,
                                        learning_rate=.06, random_state=RNG).fit(Xtr, train.log_ratio)
gbm_pred = test.log_est.values + gbm_reg.predict(Xte)
reg_rows.append(reg_metrics(gbm_pred, test, "M2: Gradient Boosting"))

reg_tbl = pd.DataFrame(reg_rows).set_index("модель")
dump(reg_tbl, "13_regression_models")
print(reg_tbl.to_string())

# %% [markdown]
# ## 7.2 Задача 2: классификация — предскажем сам факт недооценки
#
# Таргет `underestimated = 1 if actual > estimate`. Прогресс моделей тот же:
# от доли класса к бустингу. Accuracy здесь малоинформативна (классы несбалансированы,
# ~36 % положительных), поэтому смотрим прежде всего на **PR-AUC** относительно
# base rate и на ROC-AUC.

# %%
clf_rows = [clf_metrics(test.underestimated,
                        np.full(len(test), train.underestimated.mean()),
                        f"Baseline: доля класса ({train.underestimated.mean():.2f})")]
logit = Pipeline([("pre", ridge_pre),
                  ("m", LogisticRegression(max_iter=2000))]).fit(train[FEATURES], train.underestimated)
clf_rows.append(clf_metrics(test.underestimated, logit.predict_proba(test[FEATURES])[:, 1],
                            "Логистическая регрессия"))
tree = Pipeline([("pre", ridge_pre),
                 ("m", DecisionTreeClassifier(max_depth=6, random_state=RNG))]).fit(train[FEATURES], train.underestimated)
clf_rows.append(clf_metrics(test.underestimated, tree.predict_proba(test[FEATURES])[:, 1],
                            "Дерево решений (depth=6)"))
forest = Pipeline([("pre", ridge_pre),
                   ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                random_state=RNG, n_jobs=-1))]).fit(train[FEATURES], train.underestimated)
rf_proba = forest.predict_proba(test[FEATURES])[:, 1]
clf_rows.append(clf_metrics(test.underestimated, rf_proba, "Random Forest"))
gbm_clf = HistGradientBoostingClassifier(categorical_features=CAT_IDX, max_iter=400,
                                         learning_rate=.06, random_state=RNG).fit(Xtr, train.underestimated)
gbm_proba = gbm_clf.predict_proba(Xte)[:, 1]
clf_rows.append(clf_metrics(test.underestimated, gbm_proba, "Gradient Boosting"))

clf_tbl = pd.DataFrame(clf_rows).set_index("модель")
dump(clf_tbl, "14_classification_models")
print(f"Base rate на тесте: {test.underestimated.mean():.3f}\n")
print(clf_tbl.to_string())

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
for proba, name, c in [(rf_proba, "Random Forest", ACCENT3), (gbm_proba, "Gradient Boosting", ACCENT)]:
    fpr, tpr, _ = roc_curve(test.underestimated, proba)
    ax.plot(fpr, tpr, color=c, lw=2.2, label=f"{name} (AUC={roc_auc_score(test.underestimated, proba):.3f})")
ax.plot([0, 1], [0, 1], color=INK, ls="--", lw=1, label="случайная модель")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC-кривая"); ax.legend(fontsize=8)

ax = axes[1]
base = test.underestimated.mean()
for proba, name, c in [(rf_proba, "Random Forest", ACCENT3), (gbm_proba, "Gradient Boosting", ACCENT)]:
    pr, rc, _ = precision_recall_curve(test.underestimated, proba)
    ax.plot(rc, pr, color=c, lw=2.2,
            label=f"{name} (PR-AUC={average_precision_score(test.underestimated, proba):.3f})")
ax.axhline(base, color=INK, ls="--", lw=1, label=f"base rate = {base:.2f}")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Precision-Recall (важнее при дисбалансе)"); ax.legend(fontsize=8)

ax = axes[2]
cm = confusion_matrix(test.underestimated, (rf_proba >= .5).astype(int))
ax.imshow(cm, cmap="Blues"); ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}".replace(",", " "), ha="center", va="center",
                fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(["уложился", "недооценил"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["уложился", "недооценил"])
ax.set_xlabel("прогноз"); ax.set_ylabel("факт")
ax.set_title("Матрица ошибок, Random Forest (порог 0.5)")
save(fig, "10_classification"); plt.close(fig)

# %% [markdown]
# ## 7.3 Абляция: откуда на самом деле берётся предсказательная сила
#
# Самый информативный эксперимент во всём проекте. Обучаем одну и ту же модель на
# трёх вложенных наборах признаков и смотрим, что даёт прирост.

# %%
ABLATION = [("только оценка", ["log_est"]),
            ("+ контекст (проект / фаза / человек)", CAT_FEATURES + BASE_FEATURES),
            ("+ история (leak-free)", CAT_FEATURES + BASE_FEATURES + HIST_FEATURES)]
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
    abl_rows.append({"набор признаков": label, "MAE_log": m["MAE_log"],
                     "RMSE_log": m["RMSE_log"], "ROC-AUC": cm_["ROC-AUC"],
                     "PR-AUC": cm_["PR-AUC"]})
abl_tbl = pd.DataFrame(abl_rows).set_index("набор признаков")
dump(abl_tbl, "15_ablation")
print(f"B0 (факт = оценка): MAE_log={b0['MAE_log']}, RMSE_log={b0['RMSE_log']}\n")
print(abl_tbl.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
labels = [l for l, _ in ABLATION]
x = np.arange(len(labels))
ax = axes[0]
ax.bar(x, abl_tbl["ROC-AUC"], color=[MUTED, ACCENT, ACCENT3])
ax.axhline(.5, color=INK, ls="--", lw=1.2)
ax.annotate("случайная модель", (-.4, .512), fontsize=9, color=INK)
ax.set_xticks(x); ax.set_xticklabels(["только\nоценка", "+ контекст", "+ история"], fontsize=9)
ax.set_ylim(.4, .8); ax.set_ylabel("ROC-AUC")
ax.set_title("Оценка не предсказывает свою ошибку")
for xi, v in zip(x, abl_tbl["ROC-AUC"]):
    ax.text(xi, v + .008, f"{v:.3f}", ha="center", fontsize=9)

ax = axes[1]
ax.bar(x, abl_tbl["RMSE_log"], color=[MUTED, ACCENT, ACCENT3])
ax.axhline(b0["RMSE_log"], color=INK, ls="--", lw=1.2)
ax.set_ylim(0, b0["RMSE_log"] * 1.28)
ax.annotate("B0: факт = оценка", (1.0, b0["RMSE_log"] * 1.06), fontsize=9,
            color=INK, ha="center")
ax.set_xticks(x); ax.set_xticklabels(["только\nоценка", "+ контекст", "+ история"], fontsize=9)
ax.set_ylabel("RMSE_log (меньше — лучше)")
ax.set_title("Побить baseline позволяет только контекст")
for xi, v in zip(x, abl_tbl["RMSE_log"]):
    ax.text(xi, v + .008, f"{v:.3f}", ha="center", fontsize=9)
fig.suptitle("Абляция: вся предсказательная сила — в контексте и истории, а не в оценке",
             y=1.08, fontsize=12.5, fontweight="bold")
save(fig, "11_ablation"); plt.close(fig)

# %% [markdown]
# **Это ключевой результат проекта.**
#
# Модель, знающая **только оценку**, предсказывает недооценку с ROC-AUC ≈ 0.49 —
# то есть **не лучше подбрасывания монеты**, и по MAE она хуже, чем просто поверить
# оценке. Иначе говоря: *в самой оценке нет информации о том, ошибочна ли она.*
# Разработчик не «немного знает» о своей ошибке — он не знает о ней ничего.
#
# Вся предсказательная сила появляется, когда добавляется **контекст**
# (кто, какой проект, какая фаза): AUC подскакивает до ≈ 0.69. Исторические
# leak-free признаки добавляют ещё немного (до ≈ 0.71) и улучшают RMSE.
#
# Ответ на RQ8 («можно ли предсказать недооценку?») — **да, но не по оценке.**
# Предсказуема не задача, а обстоятельства, в которых она делается.
#
# ---
# # Phase 8 — Объяснимость и разбор ошибок
#
# ## 8.1 Permutation importance
#
# Насколько вырастет ошибка, если случайно перемешать один признак. В отличие от
# встроенной важности, считается на тестовой выборке и не завышает вклад признаков
# с большим числом уровней.

# %%
perm = permutation_importance(gbm_reg, Xte, test.log_ratio.values, n_repeats=8,
                              random_state=RNG, scoring="neg_root_mean_squared_error")
imp = pd.DataFrame({"признак": FEATURES, "важность": perm.importances_mean,
                    "std": perm.importances_std}).sort_values("важность", ascending=False)
dump(imp.round(4), "16_permutation_importance")
print(imp.round(4).to_string(index=False))

# %% [markdown]
# ## 8.2 SHAP: почему модель считает конкретную задачу рискованной
#
# Permutation importance отвечает на вопрос «какой признак важен вообще».
# SHAP отвечает на вопрос «почему **эта** задача получила такой прогноз» —
# именно это нужно, чтобы объяснение можно было показать менеджеру.

# %%
import shap

shap_sample = Xte.sample(min(2000, len(Xte)), random_state=RNG)

# TreeExplainer не понимает pandas-категории, а HistGBM внутри работает с их кодами.
# Передаём коды — нумерация та же, поэтому значения SHAP корректны; проверяем
# это встроенной check_additivity (сумма вкладов должна давать прогноз модели).
shap_input = shap_sample.copy()
for c in CAT_FEATURES:
    shap_input[c] = shap_input[c].cat.codes.astype(float)

explainer = shap.TreeExplainer(gbm_reg)
shap_values = explainer(shap_input, check_additivity=True)
shap_values.data = shap_sample.values      # для подписей берём исходные значения
print(f"SHAP посчитан для {len(shap_sample)} задач; проверка аддитивности пройдена")

fig = plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values.values, shap_input, max_display=12, show=False, plot_size=None)
plt.title("SHAP: вклад признаков в предсказанную поправку к оценке", fontsize=12,
          fontweight="bold", pad=14)
plt.tight_layout()
save(plt.gcf(), "12_shap_summary"); plt.close("all")

shap_rank = pd.DataFrame({
    "признак": shap_sample.columns,
    "средний |SHAP|": np.abs(shap_values.values).mean(0)}).sort_values(
    "средний |SHAP|", ascending=False)
dump(shap_rank.round(4), "17_shap_importance")
print(shap_rank.round(4).to_string(index=False))

# Разбор одного конкретного прогноза — то, что реально показывают менеджеру.
risky = int(np.argmax(gbm_reg.predict(shap_sample)))
case = shap_sample.iloc[risky]
contrib = pd.Series(shap_values.values[risky], index=shap_sample.columns).sort_values(key=abs, ascending=False)
print(f"\nПример: задача с самой большой предсказанной поправкой "
      f"(прогноз ×{np.exp(gbm_reg.predict(shap_sample)[risky]):.2f} к оценке)")
print("Топ факторов этого прогноза:")
for f, v in contrib.head(6).items():
    print(f"  {f:22s} значение={str(case[f])[:18]:20s} вклад={v:+.3f}")

# %% [markdown]
# ## 8.3 Где модель ошибается
#
# Метрики «в среднем» скрывают, что качество сильно неоднородно. Смотрим ошибку
# в разрезе размера задачи, фазы и наличия истории у исполнителя.

# %%
test_err = test.copy()
test_err["pred_log"] = gbm_pred
test_err["err"] = test_err.pred_log - test_err.log_act
test_err["abs_err"] = test_err.err.abs()
test_err["b0_abs_err"] = (test_err.log_est - test_err.log_act).abs()
test_err["has_history"] = np.where(test_err.person_hist_n.fillna(0) >= 20,
                                   "≥20 прошлых задач", "мало истории")

err_by = {}
for key, label in [("size_bin", "размер задачи"), ("phase_short_name", "фаза процесса"),
                   ("has_history", "история исполнителя")]:
    g = test_err.groupby(key, observed=True)
    t = pd.DataFrame({"N": g.size(), "GBM MAE_log": g.abs_err.mean(),
                      "B0 MAE_log": g.b0_abs_err.mean(),
                      "смещение (среднее err)": g.err.mean()})
    t["выигрыш"] = (100 * (1 - t["GBM MAE_log"] / t["B0 MAE_log"])).round(1)
    err_by[label] = t.round(3)
    dump(t.round(3), f"18_error_by_{key}")
    print(f"\nОшибка в разрезе «{label}»:\n{t.round(3).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
t = err_by["размер задачи"]
x = np.arange(len(t))
ax.bar(x - .2, t["B0 MAE_log"], .4, color=MUTED, label="B0: факт = оценка")
ax.bar(x + .2, t["GBM MAE_log"], .4, color=ACCENT3, label="GBM")
ax.set_xticks(x); ax.set_xticklabels(t.index, fontsize=9)
ax.set_ylabel("MAE_log"); ax.set_xlabel("размер задачи по оценке")
ax.set_title("Выигрыш только на мелких и средних задачах"); ax.legend(fontsize=9)

ax = axes[1]
sub = test_err.sample(min(6000, len(test_err)), random_state=RNG)
ax.scatter(np.exp(sub.pred_log), sub.actual, s=6, alpha=.15, color=ACCENT, edgecolors="none")
lim = [test_err.actual.min() * .7, test_err.actual.max() * 1.3]
ax.plot(lim, lim, color=INK, ls="--", lw=1.3, label="идеальный прогноз")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("прогноз, часы"); ax.set_ylabel("факт, часы")
ax.set_title("Прогноз против факта на тесте"); ax.legend(fontsize=9)
save(fig, "13_error_analysis"); plt.close(fig)

# %% [markdown]
# **Модель выигрывает не везде — и это важнее общей цифры.**
#
# | размер задачи | N | B0 MAE_log | GBM MAE_log | выигрыш |
# |---|---:|---:|---:|---:|
# | ≤ 0.5 ч | 5 618 | 0.660 | 0.642 | +2.7 % |
# | 0.5–1 ч | 3 341 | 0.706 | 0.679 | +3.8 % |
# | 1–2 ч | 2 558 | 0.761 | 0.719 | +5.5 % |
# | 2–4 ч | 1 676 | 0.783 | 0.732 | +6.6 % |
# | 4–8 ч | 1 064 | 0.677 | 0.671 | +0.9 % |
# | 8–16 ч | 644 | 0.538 | 0.600 | **−11.6 %** |
# | > 16 ч | 170 | 0.442 | 0.604 | **−36.6 %** |
#
# На крупных задачах модель **проигрывает baseline**, и заметно. Причина видна
# по столбцу N: крупных задач в обучении мало (170 в тесте против 5 618 мелких),
# а разброс на них максимален (RQ2). Модель тянет прогноз к типичному поведению
# и на редких крупных задачах систематически промахивается — среднее смещение
# там −0.24, то есть она их занижает.
#
# Практический вывод: **применять модель осмысленно только на мелких и средних
# задачах**, а на крупных честнее оставить оценку разработчика. Общая цифра
# «выигрыш 10 %» без этой разбивки ввела бы в заблуждение.

# %%
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.scatter(np.exp(sub.pred_log), sub.err, s=6, alpha=.15, color=ACCENT, edgecolors="none")
ax.axhline(0, color=INK, lw=1.2, ls="--")
ax.set_xscale("log")
ax.set_yticks(np.log([1/8, 1/4, 1/2, 1, 2, 4, 8]))
ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
ax.set_xlabel("прогноз, часы (лог-шкала)")
ax.set_ylabel("прогноз / факт")
ax.set_title("График остатков: систематического наклона нет,\nно разброс огромен на всём диапазоне")
save(fig, "14_residuals"); plt.close(fig)

# %% [markdown]
# ---
# # Phase 9 — Что из этого можно реально отдать команде
#
# ## 9.1 Персональная калибровка (проверка простого решения)
#
# Прежде чем предлагать ML, надо проверить очевидную идею: у каждого разработчика
# свой исторический коэффициент, умножим оценку на него. Считаем строго leak-free:
# коэффициент берётся только по прошлым задачам этого человека.

# %%
calib = test.copy()
calib_rows = [reg_metrics(calib.log_est.values, calib, "B0: факт = оценка")]
for min_n, label in [(5, "персональный коэффициент (≥5 прошлых задач)"),
                     (20, "персональный коэффициент (≥20 прошлых задач)")]:
    adj = np.where(calib.person_hist_n.fillna(0) >= min_n, calib.person_hist_lr.fillna(0), 0)
    calib_rows.append(reg_metrics(calib.log_est.values + adj, calib, label))
calib_rows.append(reg_metrics(calib.log_est.values + shift, calib,
                              f"глобальный коэффициент ×{np.exp(shift):.2f}"))
calib_rows.append(reg_metrics(gbm_pred, calib, "M2: Gradient Boosting"))
calib_tbl = pd.DataFrame(calib_rows).set_index("модель")
dump(calib_tbl, "19_personal_calibration")
print(calib_tbl.to_string())

# %% [markdown]
# Результат двойственный, и это тот случай, когда одна метрика соврала бы.
#
# Персональный коэффициент **улучшает `RMSE_log`** (1.053 → 0.997), но
# **ухудшает `MAE_log`** (0.694 → 0.716). То есть он помогает на выбросах и вредит
# на типичной задаче: сдвигая каждую оценку, он портит те, что и так были близки.
# Порог «≥ 20 прошлых задач» вместо «≥ 5» ничего не меняет — дело не в объёме
# истории, а в том, что персональное смещение объясняет слишком мало (RQ3).
#
# Полноценная модель обходит обе версии калибровки по обеим метрикам, и именно
# потому, что учитывает не только «кто», но и «что» и «где».

# %% [markdown]
# ## 9.2 Интервал вместо числа
#
# Если ~85 % дисперсии нередуцируемы (Phase 4), точечный прогноз бессмысленен, кем бы
# он ни был выдан. Единственный честный продукт — интервал.
#
# Метод: квантильная регрессия (GBM с pinball loss на 10 % и 90 % квантили `log_ratio`),
# затем **conformalized quantile regression (CQR)**. Календарно последние 20 % обучающей
# выборки отрезаются как calibration set, на нём считается conformity score
# `s = max(lo − y, y − hi)`, и интервал расширяется на его эмпирический квантиль.
# Это даёт покрытие, близкое к номинальному, **без предположений о распределении**.
#
# Обменность (условие гарантии CQR) выполняется здесь лишь приблизительно — процесс
# нестационарен, — поэтому покрытие проверяется эмпирически на каждом временном
# окне, а не принимается на веру. И, как видно ниже, результат честный, но не
# идеальный: калибровка поднимает покрытие с ~73 % до ~78 % при номинальных 80 %,
# то есть остаточная недоуверенность сохраняется. Это ожидаемо — нестационарность
# нарушает предпосылку метода, и заявлять ровно 80 % было бы неправдой.

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
        rows.append({"фолд": k + 1, "N test": len(te),
                     "покрытие (сырое)": round(float(((y >= lo) & (y <= hi)).mean()), 3),
                     "ширина (сырое)": round(float(np.median(np.exp(hi - lo))), 2),
                     "покрытие (CQR)": round(float(((y >= lo - Q) & (y <= hi + Q)).mean()), 3),
                     "ширина (CQR)": round(float(np.median(np.exp((hi + Q) - (lo - Q)))), 2),
                     "поправка Q": round(float(Q), 3)})
        if example is None:
            example = te.assign(lo=np.exp(lo - Q), hi=np.exp(hi + Q))
    return pd.DataFrame(rows), example

intervals, iv_example = conformal_intervals(model_df)
dump(intervals, "20_conformal_intervals")
print(intervals.to_string(index=False))
print(f"\nНоминальное покрытие 80 %. Сырые квантили: {intervals['покрытие (сырое)'].mean():.1%}, "
      f"после CQR: {intervals['покрытие (CQR)'].mean():.1%}")

iv_example["size_bin"] = pd.cut(iv_example.estimate, SIZE_BINS, labels=SIZE_LABELS)
width_by_size = iv_example.groupby("size_bin", observed=True).apply(
    lambda g: pd.Series({
        "N": len(g),
        "покрытие": round(float(((g.actual >= g.lo) & (g.actual <= g.hi)).mean()), 2),
        "нижняя граница / оценка": round(float(np.median(g.lo / g.estimate)), 2),
        "верхняя граница / оценка": round(float(np.median(g.hi / g.estimate)), 2),
    }), include_groups=False)
dump(width_by_size, "20_interval_width_by_size")
print("\nЧто на самом деле означает оценка разработчика (80 %-й интервал):")
print(width_by_size.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ax = axes[0]
x = np.arange(len(intervals))
ax.plot(x, intervals["покрытие (сырое)"], "o-", color=ACCENT2, lw=2, label="сырая квантильная регрессия")
ax.plot(x, intervals["покрытие (CQR)"], "s-", color=ACCENT3, lw=2.4, label="после CQR-калибровки")
ax.axhline(.8, color=INK, ls="--", lw=1.2)
ax.annotate("номинальные 80 %", (0, .807), fontsize=9, color=INK)
ax.set_xticks(x); ax.set_xticklabels([f"фолд {i}" for i in intervals["фолд"]])
ax.set_ylim(.5, 1.0)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylabel("доля фактов внутри интервала")
ax.set_title("Сырые квантили самоуверенны, CQR это чинит")
ax.legend(loc="lower right", fontsize=9)

ax = axes[1]
ex = iv_example.sort_values("estimate").iloc[::max(1, len(iv_example) // 60)]
xs = np.arange(len(ex))
ax.vlines(xs, ex.lo, ex.hi, color=ACCENT, alpha=.45, lw=3)
inside = ((ex.actual >= ex.lo) & (ex.actual <= ex.hi)).values
ax.scatter(xs[inside], ex.actual.values[inside], s=14, color=INK, zorder=3, label="факт внутри")
ax.scatter(xs[~inside], ex.actual.values[~inside], s=24, color=ACCENT2, zorder=3,
           marker="x", label="факт вне интервала")
ax.plot(xs, ex.estimate, color=ACCENT3, lw=1.6, ls="--", label="исходная оценка")
ax.set_yscale("log")
ax.set_xlabel("задачи тестового окна, отсортированные по оценке")
ax.set_ylabel("часы (лог-шкала)")
ax.set_title("Калиброванный 80 %-й интервал"); ax.legend(fontsize=8)
save(fig, "15_intervals"); plt.close(fig)

# %% [markdown]
# ---
# # Валидация на других организациях
#
# CESAW — это одна культура разработки (TSP, частично safety-critical проекты).
# Проверим главный вывод — регрессию к среднему — на SiP (коммерческая компания)
# и Renzo (один человек, личный трекинг). Чтобы датасеты были сопоставимы, делим
# каждый на квантильные группы по размеру оценки: это снимает разницу в единицах
# измерения.

# %%
def quantile_profile(d, n_bins=7):
    d = d.copy()
    d["qbin"] = pd.qcut(d.estimate.rank(method="first"), n_bins, labels=False)
    g = d.groupby("qbin")
    return pd.DataFrame({"N": g.size(), "медиана оценки": g.estimate.median(),
                         "геом. среднее": g.log_ratio.apply(lambda x: np.exp(x.mean()))})

repl = {name: quantile_profile(d) for name, d in
        [("CESAW", cesaw_ne), ("SiP", sip_ne), ("Renzo", renzo_ne)]}
repl_tbl = pd.concat(repl, names=["датасет"]).round(3)
dump(repl_tbl, "21_replication")
print(repl_tbl.to_string())

fig, ax = plt.subplots(figsize=(8.5, 4.6))
for (name, t), c, m in zip(repl.items(), [ACCENT, ACCENT2, ACCENT3], ["o", "s", "^"]):
    ax.plot(t["медиана оценки"], t["геом. среднее"], m + "-", color=c, lw=2.2, ms=7,
            label=f"{name} (N={int(t.N.sum()):,})".replace(",", " "))
ax.axhline(1, color=INK, lw=1, ls="--")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_yticks([0.5, 0.75, 1, 1.5, 2, 3])
ax.set_yticklabels(["×0.5", "×0.75", "×1", "×1.5", "×2", "×3"])
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
ax.set_xlabel("медианная оценка в группе (часы / помидоры, лог-шкала)")
ax.set_ylabel("геом. среднее факт / оценка")
ax.set_title("Регрессия к среднему воспроизводится на трёх независимых источниках:\n"
             "разные индустрии, культуры и единицы измерения")
ax.legend()
save(fig, "16_replication"); plt.close(fig)

# %%
# --- Сводка ключевых чисел (основа для отчёта) ------------------------------
summary = {
    "CESAW: задач после очистки": len(cesaw),
    "CESAW: доля факт == оценка": f"{cesaw.exact.mean():.1%}",
    "SiP: доля факт == оценка": f"{sip.exact.mean():.1%}",
    "Renzo: доля факт == оценка": f"{renzo.exact.mean():.1%}",
    "RQ1 CESAW: геом. среднее ratio": round(float(np.exp(cesaw_ne.log_ratio.mean())), 3),
    "RQ1 CESAW: доля недооценённых": f"{cesaw_ne.underestimated.mean():.1%}",
    "RQ2 наклон log_ratio~log_est (CESAW)": float(slopes_tbl.loc["CESAW", "наклон"]),
    "RQ2 наклон (SiP)": float(slopes_tbl.loc["SiP", "наклон"]),
    "RQ2 наклон (Renzo)": float(slopes_tbl.loc["Renzo", "наклон"]),
    "RQ3 R² исполнителя": float(ve_cesaw.loc["person", "R² out-of-sample"]),
    "RQ4 R² проекта": float(ve_cesaw.loc["project", "R² out-of-sample"]),
    "RQ5 R² фазы": float(ve_cesaw.loc["phase_short_name", "R² out-of-sample"]),
    "R² размера задачи": float(ve_cesaw.loc["size_bin", "R² out-of-sample"]),
    "B0 RMSE_log": b0["RMSE_log"],
    "GBM RMSE_log": float(reg_tbl.loc["M2: Gradient Boosting", "RMSE_log"]),
    "Абляция: AUC только по оценке": float(abl_tbl.iloc[0]["ROC-AUC"]),
    "Абляция: AUC + контекст": float(abl_tbl.iloc[1]["ROC-AUC"]),
    "Абляция: AUC + история": float(abl_tbl.iloc[2]["ROC-AUC"]),
    "Лучший классификатор PR-AUC": float(clf_tbl["PR-AUC"].max()),
    "Base rate недооценки (тест)": round(float(test.underestimated.mean()), 3),
    "Покрытие интервала до CQR": f"{intervals['покрытие (сырое)'].mean():.1%}",
    "Покрытие интервала после CQR": f"{intervals['покрытие (CQR)'].mean():.1%}",
}
summary_tbl = pd.DataFrame.from_dict(summary, orient="index", columns=["значение"])
dump(summary_tbl, "22_summary")
print(summary_tbl.to_string())

# %% [markdown]
# ---
# # Выводы
#
# ## Что мы узнали об оценках
#
# **1. «Идеальные оценки» надо проверять первыми.** У 8 % задач CESAW, 34 % SiP
# и 44 % Renzo факт в точности равен оценке, причём доля падает от ~60 % на мелких
# задачах до ~1 % на крупных. Это списание времени по плану, а не точность.
# Дашборд, построенный поверх такого артефакта, хвалит команду за то, чего нет.
#
# **2. Систематической недооценки нет — есть регрессия к среднему** (RQ1, RQ2).
# В агрегате команды даже укладываются в оценку (×0.81 в CESAW), но агрегат
# обманчив: мелкие задачи недооценены, крупные переоценены, и эти смещения
# взаимно гасятся. Эффект воспроизводится на трёх независимых источниках
# с разными индустриями, культурами и единицами измерения.
#
# **3. Значимо ≠ важно, и доминирующий фактор зависит от организации** (RQ3–RQ5).
# В CESAW сильнее всего исполнитель (R² out-of-sample 0.103), в SiP — размер задачи
# (0.080), причём в каждом датасете «чужой» фактор почти не работает. Переносить
# «у нас главное — кто оценивает» между компаниями нельзя.
#
# Но потолок в обоих случаях низкий: **все наблюдаемые факторы вместе объясняют
# 17 % дисперсии в CESAW и 14 % в SiP — то есть 83–86 % не объясняется ничем.**
# При 55 тысячах наблюдений статистически значимо почти всё; практический вывод
# даёт величина эффекта, а не p-value.
#
# **4. Фрагментация работы — сильнейший найденный фактор** (RQ6). Прерванные задачи
# перерасходуют примерно вдвое сильнее внутри **каждой** группы по размеру. Но
# причинность не установлена (прерывания могут быть следствием, а не причиной),
# и для прогноза это бесполезно: в момент оценки их ещё нет.
#
# **5. Эффект круглых чисел не подтвердился** (RQ7). Сырое сравнение показывает
# разницу, но она полностью объясняется тем, что круглые оценки ставят на более
# крупные задачи. При контроле размера эффект распадается. Хороший пример того,
# зачем проверять конфаундеры.
#
# ## Что мы узнали о предсказуемости
#
# **6. В самой оценке нет информации о её ошибке.** Модель, знающая только оценку,
# предсказывает недооценку с ROC-AUC 0.49 — уровень монетки. Предсказуема не задача,
# а **обстоятельства**: с добавлением контекста AUC растёт до ~0.69, с историей — до ~0.71.
#
# **7. Стратегия разбиения меняет выводы сильнее выбора модели.** Случайный сплит
# даёт заметно более радужные цифры, чем временной. Отчитываться по случайному
# разбиению на данных с повторными наблюдениями — значит вводить в заблуждение.
#
# **8. ML выигрывает в хвостах, а не в медиане.** Прирост идёт прежде всего по
# `RMSE_log` — то есть на задачах, которые уезжают в разы. Это ровно тот вопрос,
# который нужен менеджеру: *какая из этих задач рискует сорваться?*
#
# **9. Персональные коэффициенты работают наполовину.** «Умножь оценку на
# исторический коэффициент разработчика» улучшает `RMSE_log` (1.053 → 0.997),
# но ухудшает `MAE_log` (0.694 → 0.716): помогает на выбросах, вредит на типичной
# задаче. Увеличение требуемой истории с 5 до 20 задач ничего не меняет.
# Полная модель обходит обе версии по обеим метрикам.
#
# **10. Отдавать надо интервал — но честный.** Без калибровки квантильная регрессия
# самоуверенна (покрытие ~73 % вместо 80 %); conformal-поправка поднимает его
# до ~78 %. До номинала не дотягивает: обменность нарушена нестационарностью
# процесса, и заявлять ровно 80 % было бы неправдой. Широкий интервал —
# не недостаток модели, а **измеренное свойство предметной области**.
#
# ## Что с этим делать команде
#
# * **Проверить свой учёт.** Доля точных совпадений в разрезе размера задачи —
#   бесплатный тест на здоровье данных. Если она падает с 60 % до 1 %, вы измеряете
#   дисциплину списания, а не точность.
# * **Отказаться от единого коэффициента запаса.** Он ошибается в разные стороны
#   на разных задачах. Калибровать надо по размеру: мелкое умножать, крупное делить.
# * **Смотреть, что доминирует именно у вас.** В одной организации главным
#   фактором оказался исполнитель, в другой — размер задачи. Это дешёвый расчёт
#   (одна таблица R²), и он определяет, куда вообще стоит прикладывать усилия.
# * **Защищать фокус.** Единственный сильный управляемый фактор из найденных —
#   фрагментация работы.
# * **Планировать интервалами.** Оценка «4 часа» на этих данных означает
#   80 %-й диапазон примерно от часа до полутора дней. Лучше знать это заранее.
#
# ## Ограничения
#
# * **Наблюдательные данные.** Неизвестно, влияла ли оценка на факт: разработчик мог
#   подгонять работу под срок (self-fulfilling prophecy). Разделить это без
#   эксперимента невозможно.
# * **Survivorship bias.** Незакрытых и отменённых задач в датасетах нет; выводы,
#   вероятно, смещены в оптимистичную сторону.
# * **RQ6 не причинный.** Связь прерываний с перерасходом установлена, направление — нет.
# * **Обменность для CQR выполняется лишь приблизительно** — процесс нестационарен,
#   поэтому покрытие проверялось эмпирически на каждом окне.
# * **CESAW — специфическая среда** (формальный TSP-процесс, часть проектов
#   safety-critical). SiP и Renzo подтверждают эффект размера, но не абсолютные
#   коэффициенты.
#
# ## Что дальше
#
# * Иерархическая байесовская модель: частичный пулинг по исполнителю и фазе даст
#   устойчивые оценки для редких категорий вместо шумных групповых средних.
# * Модель цензурированных наблюдений для незакрытых задач — снять survivorship bias.
# * Агрегация на уровень спринта: гасят ли ошибки отдельных задач друг друга или
#   складываются? Для планирования это главный вопрос, и данные CESAW позволяют его задать.
# * Проверка гипотезы о фрагментации на данных, где известен *плановый* режим работы,
#   а не только фактический.

# %%
print("\n" + "=" * 72)
print(f"ГОТОВО. Фигур: {len(list(FIG_DIR.glob('*.png')))} | "
      f"таблиц: {len(list(RES_DIR.glob('*.csv')))}")
print("=" * 72)
