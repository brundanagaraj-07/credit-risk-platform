"""
Defence-in-depth SQL validation. The LLM prompt already forbids unsafe SQL,
but we NEVER trust an LLM output blindly -- every generated statement is
re-validated here with plain string/AST checks before it ever touches
the database connection.
"""
import re
import sqlparse

from src.utils.config import ALLOWED_TABLES

BLOCKED_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "GRANT", "REVOKE", "TRUNCATE",
    "EXEC", "EXECUTE",
}


class SQLValidationError(Exception):
    pass


def _extract_identifiers(sql: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql)
    return {t.upper() for t in tokens}


def validate_sql(sql: str) -> str:
    """Raise SQLValidationError if the SQL is unsafe or out-of-schema.
    Returns the cleaned, single-statement SQL string if valid."""
    if not sql or not sql.strip():
        raise SQLValidationError("Empty SQL.")

    sql = sql.strip().strip("`")
    if sql.lower().startswith("sql:"):
        sql = sql[4:].strip()

    statements = [s for s in sqlparse.split(sql) if s.strip()]
    if len(statements) != 1:
        raise SQLValidationError("Only a single SQL statement is allowed.")
    stmt = statements[0].strip()

    parsed = sqlparse.parse(stmt)[0]
    stmt_type = parsed.get_type()
    if stmt_type != "SELECT":
        raise SQLValidationError(f"Only SELECT statements are allowed, got: {stmt_type}")

    tokens = _extract_identifiers(stmt)
    blocked_hit = tokens & BLOCKED_KEYWORDS
    if blocked_hit:
        raise SQLValidationError(f"Blocked keyword(s) detected: {blocked_hit}")

    # Table whitelist check
    allowed_tables = {t.upper() for t in ALLOWED_TABLES.keys()}
    if not any(t in tokens for t in allowed_tables):
        raise SQLValidationError("Query does not reference an allowed table.")

    # Column whitelist check (best-effort: every identifier that isn't a
    # known SQL keyword/table/function must be an allowed column or a
    # numeric/string literal component)
    allowed_cols = {c.upper() for cols in ALLOWED_TABLES.values() for c in cols}
    sql_keywords = {
        "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING", "LIMIT",
        "AS", "AND", "OR", "NOT", "NULL", "IS", "IN", "LIKE", "BETWEEN",
        "COUNT", "AVG", "SUM", "MIN", "MAX", "ROUND", "DESC", "ASC",
        "DISTINCT", "CASE", "WHEN", "THEN", "ELSE", "END", "ON", "JOIN",
        "INNER", "LEFT", "OUTER", "APPLICATIONS",
    }
    unknown = tokens - allowed_cols - sql_keywords - allowed_tables
    unknown = {u for u in unknown if not u.isdigit()}
    if unknown:
        raise SQLValidationError(f"Unrecognised/disallowed identifiers: {unknown}")

    return stmt
