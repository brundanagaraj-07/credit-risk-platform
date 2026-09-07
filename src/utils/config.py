"""
Centralised configuration for the Credit Risk Intelligence Platform.
All paths, thresholds and constants live here so nothing is hard-coded
inside pipeline/model/UI code.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models"))
SQL_DB_PATH = Path(os.getenv("SQL_DB_PATH", DATA_DIR / "credit_risk.db"))
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT_DIR / "logs"))

for d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    target_col: str = "TARGET"
    id_col: str = "SK_ID_CURR"
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    # LightGBM params tuned for a tabular, imbalanced binary classification task
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 40,
        "n_estimators": 800,
        "verbosity": -1,
        "random_state": 42,
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 34,
        "max_depth": 8,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "random_state": 42,
        "n_jobs": -1,
    })
    # Risk-band cut points on predicted probability of default
    risk_bands: dict = field(default_factory=lambda: {
        "Low": (0.0, 0.20),
        "Medium": (0.20, 0.50),
        "High": (0.50, 1.01),
    })


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "800"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))


MODEL_CFG = ModelConfig()
LLM_CFG = LLMConfig()

# Whitelisted tables/columns the NL-to-SQL agent is allowed to touch.
# Keeping this explicit (rather than letting the LLM see the raw DB schema
# introspection) is a core part of the hallucination-control strategy.
ALLOWED_TABLES = {
    "applications": [
        "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "EXT_SOURCE_1",
        "EXT_SOURCE_2", "EXT_SOURCE_3", "AGE_YEARS", "EMPLOYED_YEARS",
        "RISK_SCORE", "RISK_BAND",
    ],
}
