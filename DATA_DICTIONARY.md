# DATA_DICTIONARY.md

A variable dictionary for the three datasets. Anything that could not be established from
documentation or verified against the data is marked `UNKNOWN — requires investigation`.
Nothing is invented: where a column's meaning is unknown, it says so.

All sources come from
[Derek-Jones/Software-estimation-datasets](https://github.com/Derek-Jones/Software-estimation-datasets).

---

## 1. CESAW — the primary dataset

Source: Derek M. Jones, William R. Nichols, "The CESAW dataset: a conversation",
[arXiv:2106.03679](https://arxiv.org/abs/2106.03679). Archive `CESAW.tgz`.

### What one row represents

The key Phase 0 question. The answer **differs between the two tables**:

* `CESAW_task_fact.csv` — **one row = one task performed by one person in one process
  phase**. 61,817 rows.
* `CESAW_time_fact.csv` — **one row = one work session** (one timer entry). 203,621 rows.
  These are the "203,621 observations" from the repository description — but they are not
  tasks.

The natural task key was established empirically:

| key tested | groups out of 61,817 rows |
|---|---:|
| `(project_key, wbs_element_key, plan_item_key)` | 51,582 — not unique |
| `+ phase_key` | 51,582 — not unique |
| `+ phase_key, person_key` | **61,817 — unique** ✔ |

Confirmation: aggregating `time_fact` by that same key reproduces
`task_actual_time_minutes` to the minute for **99.3%** of tasks.

### `CESAW_task_fact.csv` (61,817 rows)

| column | type | meaning | unit | missing | example | available before work starts? |
|---|---|---|---|---|---|---|
| `project_key` | int | project identifier (45 unique) | — | 0 | `100` | yes |
| `person_key` | int | anonymous performer id (247 unique) | — | 0 | `296` | yes |
| `team_key` | int | team identifier (10 unique) | — | 0 | `31` | yes |
| `wbs_element_key` | int | Work Breakdown Structure element | — | 0 | `11087` | yes |
| `plan_item_key` | int | plan item | — | 0 | `81765` | yes |
| `task_plan_time_minutes` | float | **the ESTIMATE** | minutes | 0 (but 692 zeros) | `90.0` | yes |
| `task_actual_time_minutes` | int | **the ACTUAL effort** | minutes | 0 | `7` | **no — this is the target** |
| `task_actual_start_date` | str | work start date and time | `M/D/YYYY H:MM` | 0 | `10/29/2012 13:12` | yes (used as the timestamp) |
| `task_actual_complete_date` | str | completion date and time | `M/D/YYYY H:MM` | 0 | `10/29/2012 13:20` | **no — leakage** |
| `phase_key` | str | process phase key; contains the literal `\N` instead of NULL | — | as `\N` | `337` | yes |
| `phase_short_name` | str | phase name (100 unique) | — | 0 | `Code Inspect` | yes |
| `process_name` | str | process: `Process-A/B/D`, `TSP(SM)` | — | 1 as `\N` | `TSP(SM)` | yes |

### `CESAW_time_fact.csv` (203,621 rows)

| column | type | meaning | unit | missing | example |
|---|---|---|---|---|---|
| `time_log_fact_key` | int | work session identifier | — | 0 | `23003` |
| `organization` | str | organisation | — | 29 | `C` |
| `project_key`, `person_key`, `team_key`, `wbs_element_key`, `plan_item_key`, `phase_key` | int | the same keys as in `task_fact` | — | 0 | — |
| `time_log_start_date` | str | session start | `M/D/YYYY H:MM` | 0 | `9/16/2011 3:03` |
| `time_log_end_date` | str | session end | `M/D/YYYY H:MM` | 0 | `9/16/2011 3:11` |
| `time_log_delta_minutes` | int | net session duration | minutes | 0 | `8` |
| `time_log_interrupt_minutes` | int | **interruption minutes within the session** | minutes | 0 | `0` |
| `phase.process_key` | int | process key | — | 0 | `1` |
| `process_name` | str | process name | — | 0 | `TSP(SM)` |

### `wbs_parent.csv` (16,778 rows)

| column | type | meaning |
|---|---|---|
| `project_key` | int | project |
| `wbs_element_key` | int | WBS element |
| `parent_wbs_element_key` | float | parent element; `NaN` at the root |

*Unused in this analysis — the WBS hierarchy was left untouched.*

### Features derived from `time_fact`

| feature | how it is computed | available before start? |
|---|---|---|
| `n_sessions` | count of `time_fact` rows per task | **no** |
| `interrupt_min` | sum of `time_log_interrupt_minutes` per task | **no** |
| `logged_min` | sum of `time_log_delta_minutes` (reconciliation check) | **no** |

These three are used **only** for the descriptive RQ6 analysis and are excluded from the
model's features: at estimation time they are unknown.

---

## 2. SiP — validation on a different organisation

Source: Derek M. Jones, Stephen Cullum, "A conversation around the analysis of the SiP
effort estimation dataset", [arXiv:1901.01621](https://arxiv.org/abs/1901.01621).

### What one row represents

**A trap.** `Sip-task-info.csv` has 12,299 rows but only 10,266 unique `TaskNumber`
values. A task worked on by several people is stored as several rows, with
`HoursEstimate` / `HoursActual` **duplicated** across them — those are task totals.
Verified: `sum(DeveloperHoursActual) == HoursActual` for 100% of tasks.

One row = **one (task, developer) pair**. Analysing estimates requires deduplicating by
`TaskNumber`, otherwise multi-person tasks get 2–5× the weight.

**File encoding is cp1252/latin-1**, not UTF-8 (typographic quotes in `Summary`).

### `SiP/Sip-task-info.csv`

| column | type | meaning | unit | missing | available before start? |
|---|---|---|---|---|---|
| `TaskNumber` | int | task identifier | — | 0 | yes |
| `Summary` | str | free-text task description | — | 0 | yes |
| `Priority` | int | priority (1–10) | — | 0 | yes |
| `RaisedByID` | int | who raised the task | — | 0 | yes |
| `AssignedToID` | int | who it is assigned to | — | 0 | yes |
| `AuthorisedByID` | float | who authorised it | — | 8,034 | yes |
| `StatusCode` | str | status: `FINISHED`, `COMPLETED`, `CANCELLED`, … | — | 0 | no (final status) |
| `ProjectCode` | str | project (20 unique) | — | 0 | yes |
| `ProjectBreakdownCode` | str | sub-project (77 unique) | — | 0 | yes |
| `Category` | str | `Development` / `Management` / `Operational` | — | 0 | yes |
| `SubCategory` | str | type of work (24 values: `Bug`, `Enhancement`, …) | — | 0 | yes |
| `HoursEstimate` | float | **the ESTIMATE** (task total) | hours | 0 | yes |
| `HoursActual` | float | **the ACTUAL** (task total) | hours | 0 | **no — the target** |
| `DeveloperID` | int | performer (22 unique) | — | 0 | yes |
| `DeveloperHoursActual` | float | that developer's share of the hours | hours | 0 | **no — a function of the target** |
| `TaskPerformance` | float | `HoursEstimate − HoursActual` | hours | 0 | **no — a direct function of the target** |
| `DeveloperPerformance` | float | the same at developer level | hours | 2,099 | **no — a direct function of the target** |

The last three columns are a classic leakage source. Including them would give R² ≈ 1 and
a meaningless model.

### `SiP/est-act-dates.csv`

| column | type | meaning | format | available before start? |
|---|---|---|---|---|
| `TaskNumber` | int | key into `Sip-task-info.csv` | — | yes |
| `EstimateOn` | date | **date the estimate was given** | `DD-Mon-YY` | yes — this is the reference point |
| `StartedOn` | date | work start date | `DD-Mon-YY` | no |
| `CompletedOn` | date | completion date | `DD-Mon-YY` | **no — leakage** |

The file carries the same `TaskNumber` duplicates and is deduplicated alongside the main
table.

---

## 3. Renzo Pomodoro — validation at the individual level

Source: Derek M. Jones, "The Renzo Pomodoro dataset",
[The Shape of Code](https://shape-of-code.com/2019/12/15/the-renzo-pomodoro-dataset/).

One person, a daily task log spanning about ten years. One row = one task planned for a day.

| column | type | meaning | unit | missing | available before start? |
|---|---|---|---|---|---|
| `X.words` | str | task tag / category (`@planning`, `@general`, …) | — | 960 | yes |
| `word_cnt` | str | comma-separated list of counters | — | 68 | `UNKNOWN — requires investigation` |
| `description` | int | numeric description identifier | — | 0 | `UNKNOWN — requires investigation` |
| `DONE` | int | whether the task was completed (0/1) | — | 0 | no |
| `date` | date | record date | `YYYY-MM-DD` | 0 | yes |
| `estimate` | float | **the ESTIMATE** | pomodoros (25 min) | 1,478 | yes |
| `actual` | float | **the ACTUAL** | pomodoros | 7,103 | **no — the target** |

**Anomaly in the raw data:** the maximum `estimate` is 5.1 × 10⁷ pomodoros (≈ 2,400 years).
Clearly data-entry junk. The analysis cuts estimates above 40 pomodoros; the decision is
recorded in the cleaning log.

---

## Target variables (built for all three datasets)

| variable | formula | meaning |
|---|---|---|
| `estimate` | converted to hours (CESAW: minutes ÷ 60) | the estimate |
| `actual` | converted to hours | the actual |
| `ratio` | `actual / estimate` | > 1 underestimate, < 1 overestimate |
| `log_ratio` | `log(ratio)` | the main analysis quantity: symmetric and additive |
| `abs_error` | `actual − estimate` | absolute error, hours |
| `rel_error` | `(actual − estimate) / estimate` | relative error |
| `underestimated` | `1 if actual > estimate else 0` | classification target |
| `exact` | `actual == estimate` | flag for the logging artefact (see Phase 4) |

## Summary after cleaning

| | CESAW | SiP | Renzo |
|---|---:|---:|---:|
| input rows | 61,817 | 12,299 | 17,764 |
| tasks after cleaning | **60,284** | **10,266** | **10,159** |
| period | 2008-09 – 2017-07 | 2004-02 – 2014-12 | 2009-04 – 2019 |
| people | 247 | 22 | 1 |
| projects | 45 | 20 | — |
| median estimate | 0.9 h | 2.5 h | 2 pomodoros |
| share `actual == estimate` | 8.1% | 34.1% | 43.9% |

The full log of row-removal decisions is in `results/01_cleaning_log.csv`; it is printed
on every run of `src/analysis.py`.
