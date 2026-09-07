"""
Versioned prompt templates for the Talk-to-Data agent.

Token-optimisation approach (see README):
  - The schema handed to the LLM is a hand-curated whitelist (~26 columns)
    instead of a full `PRAGMA table_info` dump of every joined table --
    this alone cuts the schema block from ~2,400 to ~450 tokens.
  - Few-shot examples are capped at 5 and chosen to cover distinct SQL
    *patterns* (aggregation, group-by, ranking, filtering, join-free
    business ratio) rather than 20+ near-duplicate examples.
  - We ask for SQL only (no explanation) in the first call, and generate
    the natural-language business answer in a second, much shorter call
    that only sees the query result (a few rows), never the full schema
    again -- this keeps the two-call round trip cheap.

Hallucination-control approach:
  - The system prompt explicitly forbids inventing column names and
    instructs the model to say a question cannot be answered with the
    given schema rather than guessing.
  - Every generated SQL statement is validated (see sql_guard.py) against
    the same whitelist before execution, so even if the LLM hallucinates
    a column, it is rejected before touching the database.
"""

PROMPT_VERSION = "v1.2"

SCHEMA_BLOCK = """
Table: applications
Columns (all from Home Credit Default Risk 'application_train'):
  SK_ID_CURR (INTEGER, primary key, unique loan application id)
  TARGET (INTEGER, 1 = defaulted, 0 = repaid; NULL for unseen rows)
  NAME_CONTRACT_TYPE (TEXT: 'Cash loans' | 'Revolving loans')
  CODE_GENDER (TEXT: 'M' | 'F')
  FLAG_OWN_CAR (TEXT: 'Y' | 'N')
  FLAG_OWN_REALTY (TEXT: 'Y' | 'N')
  CNT_CHILDREN (INTEGER)
  AMT_INCOME_TOTAL (REAL, annual income)
  AMT_CREDIT (REAL, loan amount)
  AMT_ANNUITY (REAL, loan annuity)
  AMT_GOODS_PRICE (REAL, price of goods financed)
  NAME_INCOME_TYPE (TEXT, e.g. 'Working','Pensioner','Commercial associate')
  NAME_EDUCATION_TYPE (TEXT, e.g. 'Higher education','Secondary / secondary special')
  NAME_FAMILY_STATUS (TEXT, e.g. 'Married','Single / not married')
  NAME_HOUSING_TYPE (TEXT, e.g. 'House / apartment','Rented apartment')
  OCCUPATION_TYPE (TEXT, nullable)
  ORGANIZATION_TYPE (TEXT)
  EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 (REAL 0-1, external credit bureau scores; nullable)
  AGE_YEARS (REAL, applicant age)
  EMPLOYED_YEARS (REAL, years employed)
  RISK_SCORE (REAL 0-1, model-predicted probability of default; NULL until scored)
  RISK_BAND (TEXT: 'Low' | 'Medium' | 'High'; NULL until scored)
""".strip()

FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the overall default rate?",
        "sql": "SELECT ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct FROM applications WHERE TARGET IS NOT NULL;",
    },
    {
        "question": "Show average income by education level",
        "sql": "SELECT NAME_EDUCATION_TYPE, ROUND(AVG(AMT_INCOME_TOTAL),0) AS avg_income FROM applications GROUP BY NAME_EDUCATION_TYPE ORDER BY avg_income DESC;",
    },
    {
        "question": "Which occupation types have the highest default rate?",
        "sql": "SELECT OCCUPATION_TYPE, ROUND(AVG(TARGET)*100,2) AS default_rate_pct, COUNT(*) AS n FROM applications WHERE OCCUPATION_TYPE IS NOT NULL GROUP BY OCCUPATION_TYPE HAVING n >= 30 ORDER BY default_rate_pct DESC LIMIT 10;",
    },
    {
        "question": "List the top 5 applicants by credit amount who are high risk",
        "sql": "SELECT SK_ID_CURR, AMT_CREDIT, RISK_SCORE, RISK_BAND FROM applications WHERE RISK_BAND = 'High' ORDER BY AMT_CREDIT DESC LIMIT 5;",
    },
    {
        "question": "How many applicants have children and own a car?",
        "sql": "SELECT COUNT(*) AS n_applicants FROM applications WHERE CNT_CHILDREN > 0 AND FLAG_OWN_CAR = 'Y';",
    },
]


def build_system_prompt() -> str:
    examples_txt = "\n".join(
        f'Q: "{ex["question"]}"\nSQL: {ex["sql"]}' for ex in FEW_SHOT_EXAMPLES
    )
    return f"""You are a SQL generation engine for a credit-risk analytics SQLite database.
Convert the user's natural-language question into ONE valid, read-only SQLite SELECT query.

{SCHEMA_BLOCK}

Rules (follow strictly):
1. Output ONLY the SQL statement. No explanation, no markdown fences, no comments.
2. Use ONLY the table and columns listed above. NEVER invent a column or table name.
3. Only SELECT statements are allowed. Never write INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH/PRAGMA.
4. Always end the statement with a semicolon.
5. If the question cannot be answered with the columns above, output exactly: NO_QUERY_POSSIBLE
6. Prefer aggregate/GROUP BY queries with ROUND() for percentages and money figures.
7. When filtering rare categories, add a HAVING COUNT(*) >= 20 guard to avoid statistically noisy answers.

Examples:
{examples_txt}
"""


def build_answer_prompt(question: str, sql: str, result_rows: list[dict]) -> str:
    """Second, cheap call: turn a small JSON result into a plain-English business answer.
    Only the (already small) result set is sent -- not the schema or examples again."""
    return f"""Original question: "{question}"
SQL executed: {sql}
Query result (JSON, truncated to first 20 rows): {result_rows[:20]}

Write a concise (2-4 sentence) business-readable answer for a credit risk analyst.
State the concrete numbers from the result. Do not mention SQL or databases.
If the result is empty, say so plainly and suggest a reason."""
