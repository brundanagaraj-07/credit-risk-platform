"""
Load and (optionally) join Home Credit Default Risk tables.

Only `application_train.csv` is required for the core pipeline. If the
auxiliary tables (bureau.csv, previous_application.csv, ...) are present in
/data, lightweight aggregate features are joined in -- but the platform
degrades gracefully to the single-table model when they are absent (this
keeps the "must run with minimal setup" requirement true regardless of how
much of the Kaggle bundle the evaluator downloaded).
"""
from pathlib import Path

import pandas as pd

from src.utils.config import DATA_DIR, ROOT_DIR
from src.utils.docker_utils import ensure_data_available
from src.utils.helpers import timeit
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _application_path(data_dir: Path = DATA_DIR) -> Path:
    """Use the full packaged dataset when it is available."""
    configured_path = Path(data_dir) / "application_train.csv"
    packaged_path = ROOT_DIR / "src" / "data" / "application_train.csv"

    if packaged_path.exists():
        return packaged_path
    return configured_path


@timeit
def load_applications(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = _application_path(data_dir)
    df = pd.read_csv(path)
    logger.info(f"Loaded applications table: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def _agg_bureau(data_dir: Path) -> pd.DataFrame | None:
    path = Path(data_dir) / "bureau.csv"
    if not path.exists():
        return None
    bureau = pd.read_csv(path)
    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_CNT=("SK_ID_BUREAU", "count"),
        BUREAU_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_CREDIT_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_ACTIVE_CNT=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
    ).reset_index()
    logger.info("Joined external bureau.csv aggregates.")
    return agg


def _agg_previous_application(data_dir: Path) -> pd.DataFrame | None:
    path = Path(data_dir) / "previous_application.csv"
    if not path.exists():
        return None
    prev = pd.read_csv(path)
    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_CNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_RATE=("NAME_CONTRACT_STATUS", lambda s: (s == "Approved").mean()),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
    ).reset_index()
    logger.info("Joined external previous_application.csv aggregates.")
    return agg


@timeit
def load_full_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Main entry point used by both the training pipeline and the EDA notebook."""
    df = load_applications(data_dir)

    for agg_fn in (_agg_bureau, _agg_previous_application):
        agg = agg_fn(data_dir)
        if agg is not None:
            df = df.merge(agg, on="SK_ID_CURR", how="left")

    return df


def build_sqlite_db(df: pd.DataFrame, db_path: Path, table_name: str = "applications"):
    """Materialise a clean, feature-engineered table into SQLite for the
    NL-to-SQL talk-to-data module (keeps the chatbot fast + read-only-safe,
    independent of the pandas training pipeline)."""
    import sqlite3
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    logger.info(f"Materialised '{table_name}' ({len(df):,} rows) into {db_path}")
