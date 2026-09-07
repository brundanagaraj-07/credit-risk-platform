-- Structured schema of the SQLite database materialised by
-- src/data/loader.py::build_sqlite_db, used by the talk-to-data agent.
-- This file documents the schema; the actual table is generated at
-- training time from the cleaned/feature-engineered dataframe.

CREATE TABLE IF NOT EXISTS applications (
    SK_ID_CURR              INTEGER PRIMARY KEY,
    TARGET                  INTEGER,              -- 1 = defaulted, 0 = repaid
    NAME_CONTRACT_TYPE      TEXT,
    CODE_GENDER             TEXT,
    FLAG_OWN_CAR            TEXT,
    FLAG_OWN_REALTY         TEXT,
    CNT_CHILDREN            INTEGER,
    AMT_INCOME_TOTAL        REAL,
    AMT_CREDIT              REAL,
    AMT_ANNUITY             REAL,
    AMT_GOODS_PRICE         REAL,
    NAME_INCOME_TYPE        TEXT,
    NAME_EDUCATION_TYPE     TEXT,
    NAME_FAMILY_STATUS      TEXT,
    NAME_HOUSING_TYPE       TEXT,
    OCCUPATION_TYPE         TEXT,
    ORGANIZATION_TYPE       TEXT,
    EXT_SOURCE_1            REAL,
    EXT_SOURCE_2            REAL,
    EXT_SOURCE_3            REAL,
    AGE_YEARS               REAL,
    EMPLOYED_YEARS          REAL,
    DAYS_EMPLOYED_ANOMALY   INTEGER,
    RISK_SCORE              REAL,                 -- populated after scoring
    RISK_BAND               TEXT                  -- 'Low' | 'Medium' | 'High'
);

CREATE INDEX IF NOT EXISTS idx_applications_risk_band ON applications (RISK_BAND);
CREATE INDEX IF NOT EXISTS idx_applications_occupation ON applications (OCCUPATION_TYPE);
