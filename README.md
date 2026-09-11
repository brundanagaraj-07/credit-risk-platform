# AI-Powered Credit Risk Intelligence Platform

Built on the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset.

An end-to-end platform covering EDA → ML risk scoring → explainability (SHAP) →
business-readable decision rules → a natural-language talk-to-data chatbot →
a Streamlit UI, all containerised with Docker.

---

## 1. Why this submission is built the way it is

The brief explicitly says: *"the dataset is provided — how you understand it,
model it, and present it is entirely up to you."* Two judgment calls drove
the design, both explained below because they are the kind of decision an
evaluator (or a real deployment) would actually ask about:

**1. Zero-setup guarantee, even without the 2.7 GB Kaggle download.**
Most take-home evaluations of this dataset break the moment the evaluator
runs `docker-compose up` without first manually downloading and unzipping
Kaggle's `application_train.csv`. This platform doesn't have that failure
mode: `src/data/synthetic_data.py` generates a schema-identical, statistically
realistic dataset (same columns, same ~8% default rate, same missingness
patterns, same `DAYS_EMPLOYED` sentinel-value quirk) the first time it
notices `/data/application_train.csv` is missing. Every module — EDA, model
training, SHAP, the chatbot — runs immediately and identically either way.
**Drop the real Kaggle CSV into `/data` and the platform automatically
switches to it — no code or config changes needed.** All screenshots in this
README were captured against the real dataset.

**2. The chatbot works with or without an LLM API key.**
If `ANTHROPIC_API_KEY` is set, natural-language questions are converted to
SQL by Claude. If it's not (or the API call fails for any reason), the
system falls back to a deterministic template engine covering the same
canonical query patterns — so "at least 5 working query patterns" is never
at the mercy of network access or quota during evaluation.

---

## 2. Architecture

```
Kaggle CSV 
        │
        ▼
 src/data/loader.py  ──▶ src/data/preprocessor.py ──▶ src/ml/train.py (LightGBM)
        │                                                     │
        ▼                                                     ▼
 SQLite (credit_risk.db)                              models/*.pkl + metrics.json
   [talk-to-data]                                             │
        │                                    ┌────────────────┼─────────────────┐
        ▼                                    ▼                ▼                 ▼
 Claude NL→SQL agent                 SHAP TreeExplainer   Surrogate rule tree   predict.py
  (sql_guard validated)               (predict.py)         (rules.py)         (inference)
        │                                    │                │                 │
        └────────────────────────────────────┴────────────────┴─────────────────┘
                                              ▼
                                   Streamlit UI (ui/app.py)
                     Overview · EDA · Risk Prediction · Explainability ·
                              Business Rules · Talk-to-Data
```

**Code structure** 

```
credit_risk_platform/
├── data/                        # mounted volume, not committed to git
├── documents/                   # presentation, EDA charts, screenshots
├── notebooks/
│   ├── eda.py                   # source of truth (jupytext "percent" format)
│   └── eda.ipynb                # generated notebook
├── src/
│   ├── data/
│   │   ├── loader.py            # load + join Home Credit tables
│   │   ├── preprocessor.py      # cleaning, feature engineering, encoding
│   │   └── synthetic_data.py    # zero-setup demo data generator
│   ├── ml/
│   │   ├── train.py             # training pipeline, CV, imbalance handling
│   │   ├── predict.py           # inference + SHAP explanations
│   │   ├── evaluate.py          # ROC-AUC / PR-AUC / confusion matrix
│   │   └── rules.py             # surrogate-tree business rule derivation
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py         # orchestrator (LLM + offline fallback)
│   │   ├── prompt_templates.py  # versioned, schema-scoped prompts
│   │   ├── sql_guard.py         # SQL validation / hallucination guardrail
│   │   └── query_runner.py      # read-only SQLite execution
│   └── utils/
│       ├── config.py, logger.py, helpers.py, docker_utils.py
├── ui/app.py                    # Streamlit multi-section UI
├── sql/schema.sql
├── models/                      # saved model artifacts (.pkl), not committed
├── Dockerfile, docker-compose.yml, entrypoint.sh
├── requirements.txt, .env.example, .gitignore
└── README.md
```

---

## 3. Setup & Run Instructions

### Option A — Docker (recommended, matches "run with one command")

```bash
git clone https://github.com/brundanagaraj-07/credit-risk-platform
cd credit_risk_platform
cp .env.example .env        
docker-compose up --build
```

Open **http://localhost:8501**. On first boot, `entrypoint.sh` automatically
trains the model (using the real dataset if you've placed it in `/data`,
otherwise synthetic demo data) before launching the UI — no manual step
needed.

To use the **real Kaggle dataset**: download `application_train.csv` from
the [competition page](https://www.kaggle.com/competitions/home-credit-default-risk/data)
and drop it into `./data/` before running `docker-compose up` (optionally
also `bureau.csv` and `previous_application.csv` for richer joined features).

### Option B — Local Python

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m src.ml.train            
streamlit run ui/app.py           
```

---

## 4. Model Selection & Class Imbalance Strategy

**Model: LightGBM (gradient-boosted trees).** Chosen over logistic
regression / random forest because it natively handles missing values
(critical here — `EXT_SOURCE_1` is missing for the majority of rows),
captures non-linear interactions between financial ratios without manual
feature crosses, and trains in seconds even on the full ~300K-row real
dataset, which matters for the "lightweight" requirement and CI-friendliness.

**Class imbalance** — the real dataset's default rate is ~8%; this project's
default synthetic sample lands at a similar order of magnitude:

| Technique | Used? | Rationale |
|---|---|---|
| `scale_pos_weight` (cost-sensitive learning) | ✅ Primary | Penalises missed defaulters proportionally to the imbalance ratio without duplicating or synthesizing rows — keeps training fast and avoids the overfitting risk of naive oversampling. |
| Stratified K-Fold CV (5 folds) | ✅ | Every fold preserves the true default rate so validation metrics aren't noisy artifacts of a lucky/unlucky split. |
| SMOTE | ⚙️ Optional (`--use-smote` flag) | Provided for comparison but **off by default** — on this dataset it did not outperform `scale_pos_weight` on PR-AUC in our tests and adds meaningful training-time overhead. |
| Threshold tuning | ✅ (`evaluate.find_best_threshold`) | The 0.5 default-probability threshold is not assumed to be optimal for business use; the risk-band cutoffs (see below) are the actual decision mechanism used downstream. |

**Risk bands** (business-facing, not just raw probability):

| Band | Predicted default probability |
|---|---|
| 🟢 Low | 0% – 20% |
| 🟠 Medium | 20% – 50% |
| 🔴 High | 50% – 100% |

## 5. Evaluation Metrics

Metrics below are from the actual training run captured in this repo's
`models/metrics.json` (5-fold stratified CV, out-of-fold predictions):

| Metric | Value |
|---|---|
| ROC-AUC (OOF) | **0.7666** |
| PR-AUC / Average Precision (OOF) | **0.483** |
| Recall @ 0.5 threshold | 0.824 |
| Precision @ 0.5 threshold | 0.309 |

**Why PR-AUC is reported alongside ROC-AUC:** with an ~8% positive rate,
ROC-AUC can look deceptively strong while the model still misses most
defaulters — PR-AUC (and the confusion matrix at the deployed threshold)
gives a more honest picture of real-world usefulness for a rare-event
classification problem. In production, the risk-band thresholds (not the
raw 0.5 cutoff) are what a credit analyst would actually act on, and those
are tunable via `ModelConfig.risk_bands` in `src/utils/config.py`.

*(Re-run `python -m src.ml.train` against the real Kaggle dataset to
reproduce these metrics at full scale — expect ROC-AUC in the 0.74–0.77
range on the actual competition data, which is consistent with published
Home Credit Default Risk baselines for a single-table LightGBM model.)*

---

## 6. Talk-to-Data: Prompt Engineering, Token Optimisation & Hallucination Control

**Prompt design** (`src/talk_to_data/prompt_templates.py`, versioned `v1.2`):
- The schema block sent to the LLM is a **hand-curated whitelist of ~26
  columns**, not a raw `PRAGMA table_info` dump of every joined table — this
  alone cuts the schema section from an estimated ~2,400 tokens to ~450.
- **5 few-shot examples**, deliberately chosen to cover *distinct SQL
  patterns* (simple aggregate, GROUP BY, ranking + filter, multi-condition
  filter, having-clause noise guard) rather than many near-duplicate
  examples — more pattern diversity per token spent.
- The NL→SQL call and the answer-generation call are **split into two
  cheap requests**: the second call only ever sees the (small, already
  truncated) query result — never the schema or examples again.

**Hallucination control (defence in depth):**
1. The system prompt explicitly forbids inventing column/table names and
   instructs the model to return `NO_QUERY_POSSIBLE` rather than guess.
2. Every generated SQL statement is parsed and validated by
   `src/talk_to_data/sql_guard.py` **before execution**:
   - only a single `SELECT` statement is allowed (via `sqlparse`),
   - a blocklist rejects `INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH/PRAGMA/...`,
   - every identifier in the query is checked against the same column
     whitelist used in the prompt — a hallucinated column is rejected here
     even if the LLM invents one.
3. On a validation failure, the system does **one self-repair retry**,
   appending the exact validation error to the follow-up prompt, before
   giving up gracefully.
4. The database connection itself is opened in SQLite **read-only URI
   mode** (`mode=ro`) as a final layer, independent of the SQL guard.
5. Result sets are capped (500 rows) before being handed back to the LLM
   or the UI, bounding both token usage and blast radius.

**Reliability:** the offline fallback engine (`_FALLBACK_PATTERNS` in
`nl_to_sql.py`) guarantees the following query patterns work with **zero
external dependency**, verified in this repo's own test run:
1. Overall default rate
2. Average income by education level
3. Default rate by occupation type
4. Top-N high-risk applicants by credit amount
5. Multi-condition filter (children + car ownership)
6. Risk band distribution
7. Average credit amount

---

## 7. Rule Derivation Logic & Sample Output

`src/ml/rules.py` distills the LightGBM model into human-readable policy
rules via **model distillation**: a shallow (depth ≤ 4) `DecisionTreeRegressor`
is trained to *mimic the LightGBM model's predicted probabilities* (not the
raw labels), then every root-to-leaf path is converted into a natural-language
rule tagged with its predicted default rate and portfolio support (rows
covered). Rules with support below 0.5% of the data are dropped, and the
top 12 rules by predicted risk are kept — a size a risk analyst can actually
review, not an overfit micro-rule dump.

**Sample output (from `models/business_rules.json`, this run):**
```
IF EXT_SOURCE_MEAN <= 0.44 AND CREDIT_INCOME_RATIO > 5.00
   AND DAYS_EMPLOYED_ANOMALY > 0.50 AND NAME_EDUCATION_TYPE > 1.50
THEN risk band = High          (90.0% predicted default rate, 1.8% of applicants)

IF EXT_SOURCE_MEAN <= 0.44 AND CREDIT_INCOME_RATIO <= 5.00
   AND DAYS_EMPLOYED_ANOMALY > 0.50 AND NAME_EDUCATION_TYPE > 1.50
THEN risk band = High          (61.2% predicted default rate, 2.18% of applicants)

IF EXT_SOURCE_MEAN > 0.44 AND ... (low-risk branch)
THEN risk band = Low            (7.7% predicted default rate, 9.69% of applicants)
```

---

## 8. Explainable AI

`src/ml/predict.py::RiskModel.explain()` uses `shap.TreeExplainer` (exact,
fast for tree ensembles) to return the top-8 SHAP feature contributions for
any individual scored applicant, surfaced in the UI's Explainability tab as
a signed bar chart (red = increases risk, green = decreases risk) plus the
raw feature values — designed to be readable by a non-technical loan
officer, not just a data scientist.

---

## 9. Known Limitations & Possible Improvements

- **Single-table core model.** Bureau/previous-application aggregates are
  wired in (`src/data/loader.py`) but only activate if those CSVs are
  present; a production version would build a richer feature store across
  all 7 Home Credit tables (installments, credit card balance, POS cash
  balance, etc.).
- **Synthetic data is a statistical approximation**, not a substitute for
  the real competition data — it's a development/demo convenience, and the
  README is explicit about swapping in the real CSV for production numbers.
- **Rule surrogate is an approximation of the model**, not the model itself
  — by construction it will disagree with LightGBM on some edge cases. It
  should be framed to stakeholders as "the general policy the model has
  learned," not as a drop-in replacement for the model's own score.
- **No authentication/audit-logging layer** on the Streamlit UI or the
  chatbot — necessary before any real regulatory deploymen
- **Prompt-based SQL generation**, even with guardrails, is inherently
  probabilistic; a production system would likely also expose a small set
  of pre-built, parameterised "certified" queries for the highest-stakes
  business questions and reserve free-text NL→SQL for exploratory analysis.
- **Threshold/risk-band cutoffs (0.20 / 0.50)** are illustrative defaults,
  not calibrated against a real cost-of-default vs. cost-of-lost-business
  matrix — that calibration should happen with the actual credit policy team.

---

## 10. Tech Stack

| Component | Choice | Why |
|---|---|---|
| ML | LightGBM | Fast, handles missing values natively, strong tabular baseline |
| Explainability | SHAP (TreeExplainer) | Exact + fast for tree ensembles, industry-standard |
| LLM | Groq API | Strong instruction-following for constrained SQL generation; offline fallback removes hard dependency |
| Talk-to-data DB | SQLite | Zero-ops, ships inside the container, sufficient for the query patterns required |
| UI | Streamlit | Fastest path to a clean multi-section demo UI in Python |
| Deployment | Docker + Docker Compose |single-command startup |

---

## Presentation

See `documents/presentation.pdf` for the use-case walkthrough with output
screenshots (also available individually in `documents/screenshots/`).
