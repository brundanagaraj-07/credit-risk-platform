"""
Executes validated SQL against the SQLite database in strict read-only mode
(URI `mode=ro`) as a final defence layer even below the SQL guard.
"""
import sqlite3
from pathlib import Path

import pandas as pd

from src.utils.config import SQL_DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_query(sql: str, db_path: Path = SQL_DB_PATH, row_limit: int = 500) -> pd.DataFrame:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}. Run training pipeline first.")

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        df = pd.read_sql_query(sql, conn)
        if len(df) > row_limit:
            logger.warning(f"Query returned {len(df)} rows, truncating to {row_limit} for the chatbot response.")
            df = df.head(row_limit)
        return df
    finally:
        conn.close()
