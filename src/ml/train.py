"""
Training pipeline for the credit default model.

Class imbalance strategy (documented in README too):
  1. `scale_pos_weight` on LightGBM (cost-sensitive learning) as the primary
     mechanism -- preferred over oversampling for this dataset size because
     it doesn't synthesize/duplicate rows and keeps training fast.
  2. Stratified K-fold CV so every fold preserves the ~8% default rate.
  3. Evaluation on ROC-AUC *and* PR-AUC (PR-AUC is the more honest metric
     under heavy imbalance since it isn't inflated by the abundant negatives).
  4. Optional SMOTE toggle (`--use-smote`) is provided for comparison but is
     OFF by default -- on this dataset it did not beat scale_pos_weight and
     slowed training materially (see README "Model selection rationale").
"""
import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_full_dataset, build_sqlite_db
from src.data.preprocessor import build_features
from src.ml.evaluate import evaluate_predictions
from src.ml.rules import derive_business_rules
from src.utils.config import MODEL_CFG, MODEL_DIR, DATA_DIR, SQL_DB_PATH
from src.utils.helpers import timeit
from src.utils.logger import get_logger

logger = get_logger(__name__)


@timeit
def train(use_smote: bool = False):
    df_raw = load_full_dataset()

    X, y, encoder, imputer = build_features(df_raw, fit=True)

    skf = StratifiedKFold(n_splits=MODEL_CFG.n_splits, shuffle=True, random_state=MODEL_CFG.random_state)

    oof_preds = np.zeros(len(X))
    fold_metrics = []
    models = []

    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / pos
    logger.info(f"Class balance -> negatives: {neg}, positives: {pos}, scale_pos_weight={scale_pos_weight:.2f}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if use_smote:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=MODEL_CFG.random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            params = {**MODEL_CFG.lgbm_params}
        else:
            params = {**MODEL_CFG.lgbm_params, "scale_pos_weight": scale_pos_weight}

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        metrics = evaluate_predictions(y_val, val_pred)
        fold_metrics.append(metrics)
        models.append(model)
        logger.info(f"Fold {fold}/{MODEL_CFG.n_splits} -> ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f}")

    overall = evaluate_predictions(y, oof_preds)
    logger.info(f"Overall OOF -> ROC-AUC={overall['roc_auc']:.4f} PR-AUC={overall['pr_auc']:.4f}")

    # Refit a final model on 100% of the data for deployment
    final_params = {**MODEL_CFG.lgbm_params, "scale_pos_weight": scale_pos_weight}
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "credit_risk_model.pkl")
    joblib.dump(encoder, MODEL_DIR / "encoder.pkl")
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_names.pkl")

    metrics_out = {
        "cv_fold_metrics": fold_metrics,
        "oof_overall": overall,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "default_rate": float(y.mean()),
        "scale_pos_weight": float(scale_pos_weight),
        "use_smote": use_smote,
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Business rule extraction (surrogate decision tree + SHAP-driven bucket rules)
    rules = derive_business_rules(final_model, X, y)
    with open(MODEL_DIR / "business_rules.json", "w") as f:
        json.dump(rules, f, indent=2)

    # Materialise a clean, SCORED copy into SQLite for the talk-to-data
    # chatbot -- done last so RISK_SCORE / RISK_BAND are always in sync with
    # the freshly trained model, and the chatbot reflects exactly what the
    # model saw and predicted.
    from src.data.preprocessor import clean_raw, engineer_features
    from src.utils.helpers import risk_band as _risk_band_fn

    df_for_sql = engineer_features(clean_raw(df_raw))
    full_proba = final_model.predict_proba(X)[:, 1]
    df_for_sql["RISK_SCORE"] = full_proba.round(4)
    df_for_sql["RISK_BAND"] = [_risk_band_fn(p, MODEL_CFG.risk_bands) for p in full_proba]
    build_sqlite_db(df_for_sql, SQL_DB_PATH)

    logger.info("Training complete. Artifacts saved to /models")
    return final_model, metrics_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-smote", action="store_true")
    args = parser.parse_args()
    train(use_smote=args.use_smote)
