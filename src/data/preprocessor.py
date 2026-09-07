"""
Cleaning, feature engineering, encoding and imputation.

Design choice: a single deterministic function (`build_features`) is shared
by training, batch scoring and the Streamlit "score a new applicant" form,
so there is exactly one definition of every feature -- eliminating
train/serve skew.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from src.utils.helpers import timeit
from src.utils.logger import get_logger

logger = get_logger(__name__)

NUMERIC_FEATURES = [
    # Raw numeric columns
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AGE_YEARS", "EMPLOYED_YEARS",
    # Original baseline ratios
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "GOODS_CREDIT_RATIO",
    # New domain interaction features
    "EXT_SOURCES_MEAN", "EXT_SOURCES_STD",
    "CREDIT_TO_ANNUITY_RATIO", "CREDIT_TO_GOODS_RATIO", "EMPLOYED_TO_AGE_RATIO",
    # Aggregated external bureau & loan history features (if joined)
    "BUREAU_CNT", "BUREAU_CREDIT_SUM_MEAN", "BUREAU_CREDIT_OVERDUE_SUM", "BUREAU_ACTIVE_CNT",
    "PREV_APP_CNT", "PREV_APPROVED_RATE", "PREV_AMT_CREDIT_MEAN",
]

CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
]


@timeit
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Sentinel for "not employed" (365243 days) -> treat as missing
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})

    return df


@timeit
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure and age transformations
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).round(1)

    # Standard ratios
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)
    if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:
        df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace(0, np.nan)

    # 1. External Sources composites
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCES_STD"] = df[ext_cols].std(axis=1).fillna(0)

    # 2. Key Credit and Debt Burden Ratios
    if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
        df["CREDIT_TO_ANNUITY_RATIO"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1e-5)
    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1e-5)
    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        df["EMPLOYED_TO_AGE_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-5)

    return df

@timeit
def build_features(
    df: pd.DataFrame,
    encoder: OrdinalEncoder = None,
    num_imputer: SimpleImputer = None,
    fit: bool = True,
):
    """Full preprocessing pipeline. Returns (X, y, encoder, num_imputer)."""
    df = clean_raw(df)
    df = engineer_features(df)

    y = df["TARGET"] if "TARGET" in df.columns else None

    if fit:
        num_feats = [c for c in NUMERIC_FEATURES if c in df.columns]
        cat_feats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    else:
        num_feats = list(num_imputer.feature_names_in_)
        cat_feats = list(encoder.feature_names_in_)

    # Build X_num ensuring every expected column exists
    num_data = {}
    for col in num_feats:
        if col in df.columns:
            num_data[col] = df[col]
        else:
            num_data[col] = pd.Series(np.nan, index=df.index)
    X_num = pd.DataFrame(num_data, index=df.index)[num_feats]

    # Build X_cat ensuring every expected column exists
    cat_data = {}
    for col in cat_feats:
        if col in df.columns:
            cat_data[col] = df[col].astype(str)
        else:
            cat_data[col] = pd.Series("Missing", index=df.index)
    X_cat = pd.DataFrame(cat_data, index=df.index)[cat_feats]

    if fit:
        num_imputer = SimpleImputer(strategy="median")
        X_num_imp = pd.DataFrame(num_imputer.fit_transform(X_num), columns=num_feats, index=df.index)

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat_enc = pd.DataFrame(encoder.fit_transform(X_cat), columns=cat_feats, index=df.index)
    else:
        X_num_imp = pd.DataFrame(num_imputer.transform(X_num), columns=num_feats, index=df.index)
        X_cat_enc = pd.DataFrame(encoder.transform(X_cat), columns=cat_feats, index=df.index)

    # Flags
    if "DAYS_EMPLOYED_ANOMALY" not in df.columns:
        df["DAYS_EMPLOYED_ANOMALY"] = 0
    extra_flags = ["DAYS_EMPLOYED_ANOMALY"]

    X = pd.concat([X_num_imp, X_cat_enc, df[extra_flags]], axis=1)

    logger.info(f"Feature matrix built: {X.shape[0]:,} rows x {X.shape[1]} features")
    return X, y, encoder, num_imputer