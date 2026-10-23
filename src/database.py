"""
src/database.py
SQLite database management module.
Handles local relational database initialization, schema migration, parameterized ingestion,
and demonstrates core SQL concepts (SELECT, WHERE, GROUP BY, HAVING, ORDER BY, JOIN).
"""

import os
import sqlite3
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

from src.data_loader import DATA_DIR, load_cleaned_data, load_raw_data

DB_PATH = os.path.join(DATA_DIR, "hackathon_insights.db")


def get_db_path() -> str:
    """Returns absolute path to SQLite database file."""
    return DB_PATH


def get_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_database(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Initializes SQLite tables:
    1. 'problem_statements': core table with primary key 'problem_id'
    2. 'problem_technologies': relational junction table mapping problem_id to discrete technology tools
    3. 'domains_dim': domain dimension table
    Inserts or replaces records from the provided or loaded DataFrame.
    """
    if df is None:
        df = load_cleaned_data()
        if df is None:
            df = load_raw_data()

    conn = get_connection()
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS problem_statements (
        problem_id TEXT PRIMARY KEY,
        hackathon_name TEXT,
        year INTEGER,
        organizer_type TEXT,
        problem_title TEXT,
        problem_description TEXT,
        domain TEXT,
        sub_domain TEXT,
        industry TEXT,
        technology_tags TEXT,
        ai_required INTEGER,
        hardware_required INTEGER,
        data_availability TEXT,
        difficulty_level TEXT,
        estimated_solution_days REAL,
        team_size INTEGER,
        innovation_score REAL,
        impact_score REAL,
        implementation_feasibility TEXT,
        problem_length_words INTEGER,
        number_of_technologies INTEGER,
        source_url TEXT,
        data_type TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS problem_technologies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id TEXT,
        technology_name TEXT,
        FOREIGN KEY (problem_id) REFERENCES problem_statements (problem_id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS domains_dim (
        domain_name TEXT PRIMARY KEY,
        sector_category TEXT,
        priority_level TEXT
    );
    """)

    # Populate domains dimension
    domains_data = [
        ("Healthcare", "Life Sciences", "High"),
        ("FinTech", "Financial Services", "High"),
        ("Agriculture", "Primary Sector", "Critical"),
        ("Smart Cities", "Civic Infrastructure", "Medium"),
        ("Cybersecurity", "National Security", "Critical"),
        ("Clean Energy", "Sustainability", "High"),
        ("Disaster Management", "Emergency Services", "Critical"),
        ("Smart Education", "Human Capital", "Medium"),
        ("Transportation", "Mobility", "High"),
        ("Space Tech", "Deep Tech", "Medium")
    ]
    cursor.executemany("""
    INSERT OR REPLACE INTO domains_dim (domain_name, sector_category, priority_level)
    VALUES (?, ?, ?);
    """, domains_data)

    # Ingest problem statements using parameterized INSERT OR REPLACE
    insert_sql = """
    INSERT OR REPLACE INTO problem_statements (
        problem_id, hackathon_name, year, organizer_type, problem_title,
        problem_description, domain, sub_domain, industry, technology_tags,
        ai_required, hardware_required, data_availability, difficulty_level,
        estimated_solution_days, team_size, innovation_score, impact_score,
        implementation_feasibility, problem_length_words, number_of_technologies,
        source_url, data_type
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    );
    """

    cursor.execute("DELETE FROM problem_technologies;")

    records_to_insert = []
    tech_records = []

    for _, row in df.iterrows():
        p_id = str(row.get("problem_id", "")).strip()
        records_to_insert.append((
            p_id,
            str(row.get("hackathon_name", "")),
            int(row.get("year", 2023)),
            str(row.get("organizer_type", "")),
            str(row.get("problem_title", "")),
            str(row.get("problem_description", "")),
            str(row.get("domain", "")),
            str(row.get("sub_domain", "")),
            str(row.get("industry", "")),
            str(row.get("technology_tags", "")),
            int(row.get("ai_required", 0)),
            int(row.get("hardware_required", 0)),
            str(row.get("data_availability", "")),
            str(row.get("difficulty_level", "")),
            float(row.get("estimated_solution_days", 14.0)),
            int(row.get("team_size", 4)),
            float(row.get("innovation_score", 7.0)),
            float(row.get("impact_score", 7.0)),
            str(row.get("implementation_feasibility", "Medium")),
            int(row.get("problem_length_words", 0)),
            int(row.get("number_of_technologies", 0)),
            str(row.get("source_url", "")),
            str(row.get("data_type", "synthetic"))
        ))

        # Extract normalized tech tags for the relational table
        raw_tags = str(row.get("technology_tags", ""))
        for t in raw_tags.split(","):
            cleaned_tag = t.strip()
            if cleaned_tag:
                tech_records.append((p_id, cleaned_tag))

    cursor.executemany(insert_sql, records_to_insert)
    cursor.executemany("""
    INSERT INTO problem_technologies (problem_id, technology_name) VALUES (?, ?);
    """, tech_records)

    conn.commit()

    # Verify counts
    cursor.execute("SELECT COUNT(*) FROM problem_statements;")
    ps_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM problem_technologies;")
    tech_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM domains_dim;")
    dim_count = cursor.fetchone()[0]

    conn.close()

    return {
        "status": "success",
        "database_path": DB_PATH,
        "problem_statements_count": ps_count,
        "technologies_count": tech_count,
        "domains_dim_count": dim_count
    }


def run_custom_query(query: str, params: Tuple = ()) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Executes a read-only SQL query against the SQLite database and returns (df, error_string).
    """
    try:
        conn = get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)


def get_sql_demonstrations() -> Dict[str, Dict[str, str]]:
    """
    Returns curated SQL queries demonstrating core relational concepts:
    SELECT, WHERE, ORDER BY, GROUP BY, HAVING, and JOIN.
    """
    return {
        "1. Filtered Search (SELECT, WHERE, ORDER BY)": {
            "concept": "Basic filtering and ordered sorting",
            "sql": """
SELECT problem_id, problem_title, domain, difficulty_level, estimated_solution_days, innovation_score
FROM problem_statements
WHERE ai_required = 1 AND estimated_solution_days <= 20.0
ORDER BY innovation_score DESC
LIMIT 10;
            """
        },
        "2. Domain Group Aggregation (GROUP BY, HAVING)": {
            "concept": "Aggregation and threshold filtering with HAVING",
            "sql": """
SELECT 
    domain, 
    COUNT(problem_id) AS total_problems,
    ROUND(AVG(estimated_solution_days), 1) AS avg_days,
    ROUND(AVG(innovation_score), 2) AS avg_innovation,
    SUM(ai_required) AS total_ai_problems
FROM problem_statements
GROUP BY domain
HAVING COUNT(problem_id) >= 20
ORDER BY total_problems DESC;
            """
        },
        "3. Relational Multi-Table Query (INNER JOIN)": {
            "concept": "Joining problem_statements with domains dimension table",
            "sql": """
SELECT 
    p.problem_id,
    p.problem_title,
    p.domain,
    d.sector_category,
    d.priority_level,
    p.difficulty_level,
    p.estimated_solution_days
FROM problem_statements p
INNER JOIN domains_dim d ON p.domain = d.domain_name
WHERE d.priority_level = 'Critical'
ORDER BY p.estimated_solution_days ASC
LIMIT 10;
            """
        },
        "4. Technology Junction Join (JOIN & GROUP BY)": {
            "concept": "Querying normalized many-to-many technology tags",
            "sql": """
SELECT 
    t.technology_name,
    COUNT(DISTINCT t.problem_id) AS usage_count,
    ROUND(AVG(p.innovation_score), 2) AS avg_innovation_score
FROM problem_technologies t
JOIN problem_statements p ON t.problem_id = p.problem_id
GROUP BY t.technology_name
HAVING COUNT(DISTINCT t.problem_id) >= 15
ORDER BY usage_count DESC;
            """
        }
    }
