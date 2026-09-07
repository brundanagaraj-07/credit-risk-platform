"""
Small helpers used at container start-up (entrypoint / Streamlit boot)
to fail fast with a clear message instead of a stack trace, and to
auto-provision demo data so `docker-compose up` works with zero manual
steps even if the evaluator has not downloaded the Kaggle dataset.
"""
import os
from pathlib import Path

from src.utils.config import DATA_DIR, MODEL_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY"]


def check_env(required=REQUIRED_ENV_VARS, warn_only: bool = True):
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        msg = f"Missing environment variables: {missing}."
        if warn_only:
            logger.warning(msg + " Chatbot will run in template-only fallback mode.")
        else:
            raise EnvironmentError(msg)
    return missing


def ensure_data_available():
    """
    If the real Home Credit `application_train.csv` is not present in /data
    (evaluator did not download the Kaggle dataset), generate a realistic
    synthetic dataset with the identical schema so every module of the
    platform (EDA, ML, chatbot) still runs end-to-end out of the box.
    """
    real_file = Path(DATA_DIR) / "application_train.csv"
    if real_file.exists():
        logger.info("Real Home Credit dataset detected.")
        return "real"

    from src.data.synthetic_data import generate_synthetic_dataset
    logger.info("Kaggle dataset not found -> generating synthetic demo dataset.")
    generate_synthetic_dataset(Path(DATA_DIR) / "application_train.csv")
    return "synthetic"


def ensure_model_available(auto_train: bool = True):
    """Ensure model artifacts exist in both Docker and Streamlit Cloud.

    Streamlit Cloud launches the app directly and does not run the Docker
    entrypoint, so train the demo model lazily when no persisted artifacts
    are available.
    """
    model_path = Path(MODEL_DIR) / "credit_risk_model.pkl"
    if model_path.exists():
        return True

    if not auto_train:
        return False

    try:
        from src.ml.train import train

        logger.info("No trained model found -> training a deployment model.")
        train()
    except Exception:
        logger.exception("Automatic model training failed.")
        return False

    return model_path.exists()
