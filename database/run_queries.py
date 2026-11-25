"""
Runs every query in queries.sql against the database and prints the results.

Usage
    python run_queries.py --db data/thesis.db
"""

import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def queries(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^-- name: (.+)$", text, flags=re.M)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/thesis.db")
    args = ap.parse_args()
    con = sqlite3.connect(args.db)
    pd.set_option("display.width", 160)
    for name, sql in queries(HERE / "queries.sql"):
        print(f"\n{name}\n{'-' * len(name)}")
        print(pd.read_sql(sql, con).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
