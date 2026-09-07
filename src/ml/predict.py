"""
Load saved artifacts and score new applicant(s), returning a probability of
default, a business risk band, and (optionally) a SHAP explanation.
"""
from pathlib import Path

import joblib
import pandas as pd

from src.data.preprocessor import build_features
from src.utils.config import MODEL_CFG, MODEL_DIR
from src.utils.helpers import risk_band
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "credit_risk_model.pkl")
        self.encoder = joblib.load(self.model_dir / "encoder.pkl")
        self.imputer = joblib.load(self.model_dir / "imputer.pkl")
        self.feature_names = joblib.load(self.model_dir / "feature_names.pkl")
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def _prepare(self, df: pd.DataFrame):
        X, _, _, _ = build_features(df, encoder=self.encoder, num_imputer=self.imputer, fit=False)
        X = X.reindex(columns=self.feature_names, fill_value=0)
        return X

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with RISK_SCORE (probability of default) and RISK_BAND."""
        X = self._prepare(df)
        proba = self.model.predict_proba(X)[:, 1]
        out = df.copy()
        out["RISK_SCORE"] = proba.round(4)
        out["RISK_BAND"] = [risk_band(p, MODEL_CFG.risk_bands) for p in proba]
        return out

    def explain(self, df: pd.DataFrame, top_n: int = 8) -> list[dict]:
        """Return per-row top-N SHAP feature contributions, sorted by |impact|."""
        X = self._prepare(df)
        shap_values = self.explainer.shap_values(X)
        # LightGBM binary classifier -> shap_values may be a list [class0, class1] or single array
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        explanations = []
        for i in range(len(X)):
            row_shap = sv[i]
            contributions = sorted(
                zip(X.columns, row_shap, X.iloc[i].values),
                key=lambda t: abs(t[1]),
                reverse=True,
            )[:top_n]
            explanations.append([
                {"feature": f, "shap_value": float(v), "feature_value": float(val)}
                for f, v, val in contributions
            ])
        return explanations
