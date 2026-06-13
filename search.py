from __future__ import annotations
"""
Searchable database interface for Hindi manuscript genealogical records.

Provides researchers efficient access to extracted names, familial relationships,
caste, location, and ritual date information stored in manuscripts.db.

Usage examples:
    python search.py --name "राम कुमार"
    python search.py --caste "राजपूत"
    python search.py --place "आगरा"
    python search.py --date-range 1900 1950
    python search.py --family-id F042
    python search.py --relation "पिता"
    python search.py --flagged          (show low-confidence records for review)
    python search.py --stats            (database summary statistics)
    python search.py --name "राम" --export results.xlsx
"""

import argparse
import os
import sys
import time

from pipeline.database import DB_PATH, init_db, query

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Search functions (each benchmarked for query time reporting)
# ---------------------------------------------------------------------------

def _run_query(sql: str, params: tuple = (), db_path: str = DB_PATH) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    rows = query(sql, params, db_path)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return rows, elapsed_ms


def search_by_name(name: str, db_path: str = DB_PATH) -> tuple[list[dict], float]:
    sql = """
        SELECT p.individual_id, p.given_name, p.surname, p.gender, p.relation,
               p.caste, p.subcaste, p.place, p.family_id, p.confidence, p.flagged,
               r.ritual_date_gregorian, r.whose_ritual,
               d.file_name, d.bahi_number, d.folio_number
        FROM persons p
        LEFT JOIN rituals r ON r.person_id = p.id
        LEFT JOIN documents d ON d.id = p.document_id
        WHERE p.given_name LIKE ?
        ORDER BY p.given_name
    """
    return _run_query(sql, (f"%{name}%",), db_path)


def search_by_caste(caste: str, db_path: str = DB_PATH) -> tuple[list[dict], float]:
    sql = """
        SELECT p.individual_id, p.given_name, p.surname, p.gender, p.relation,
               p.caste, p.subcaste, p.place, p.family_id, p.confidence,
               r.ritual_date_gregorian, d.file_name
        FROM persons p
        LEFT JOIN rituals r ON r.person_id = p.id
        LEFT JOIN documents d ON d.id = p.document_id
        WHERE p.caste LIKE ?
        ORDER BY p.caste, p.given_name
    """
    return _run_query(sql, (f"%{caste}%",), db_path)


def search_by_place(place: str, db_path: str = DB_PATH) -> tuple[list[dict], float]:
    sql = """
        SELECT p.individual_id, p.given_name, p.surname, p.caste, p.relation,
               p.place, p.family_id, r.ritual_date_gregorian, d.file_name
        FROM persons p
        LEFT JOIN rituals r ON r.person_id = p.id
        LEFT JOIN documents d ON d.id = p.document_id
        WHERE p.place LIKE ?
        ORDER BY p.place, p.given_name
    """
    return _run_query(sql, (f"%{place}%",), db_path)


def search_by_date_range(year_from: int, year_to: int, db_path: str = DB_PATH) -> tuple[list[dict], float]:
    sql = """
        SELECT p.individual_id, p.given_name, p.caste, p.place,
               r.ritual_date_gregorian, r.whose_ritual, r.family_id, d.file_name
        FROM rituals r
        JOIN persons p ON p.id = r.person_id
        JOIN documents d ON d.id = r.document_id
        WHERE r.ritual_date_gregorian BETWEEN ? AND ?
        ORDER BY r.ritual_date_gregorian
    """
    return _run_query(sql, (f"{year_from}-01-01", f"{year_to}-12-31"), db_path)


def search_by_family(family_id: str, db_path: str = DB_PATH) -> tuple[list[dict], float]:
    sql = """
        SELECT p.individual_id, p.given_name, p.surname, p.gender, p.relation,
               p.caste, p.place, r.ritual_date_gregorian, d.file_name
        FROM persons p
        LEFT JOIN rituals r ON r.person_id = p.id
        LEFT JOIN documents d ON d.id = p.document_id
        WHERE p.family_id = ?
        ORDER BY p.id
    """
    return _run_query(sql, (family_id,), db_path)


def search_by_relation(relation: str, db_path: str = DB_PATH) -> tuple[list[dict], float]:
    sql = """
        SELECT p.individual_id, p.given_name, p.gender, p.relation,
               p.caste, p.place, p.family_id, d.file_name
        FROM persons p
        LEFT JOIN documents d ON d.id = p.document_id
        WHERE p.relation LIKE ?
        ORDER BY p.relation, p.given_name
    """
    return _run_query(sql, (f"%{relation}%",), db_path)


def search_flagged(db_path: str = DB_PATH) -> tuple[list[dict], float]:
    sql = """
        SELECT p.individual_id, p.given_name, p.relation, p.confidence,
               p.additional_info, d.file_name
        FROM persons p
        LEFT JOIN documents d ON d.id = p.document_id
        WHERE p.flagged = 1
        ORDER BY p.confidence
    """
    return _run_query(sql, (), db_path)


def get_stats(db_path: str = DB_PATH) -> dict:
    t0 = time.perf_counter()
    docs = query("SELECT COUNT(*) as n FROM documents", db_path=db_path)[0]["n"]
    persons = query("SELECT COUNT(*) as n FROM persons", db_path=db_path)[0]["n"]
    rituals = query("SELECT COUNT(*) as n FROM rituals WHERE ritual_date_gregorian != ''", db_path=db_path)[0]["n"]
    locations = query("SELECT COUNT(*) as n FROM locations", db_path=db_path)[0]["n"]
    flagged = query("SELECT COUNT(*) as n FROM persons WHERE flagged = 1", db_path=db_path)[0]["n"]
    avg_conf_row = query("SELECT AVG(confidence) as c FROM persons", db_path=db_path)
    avg_conf = round(avg_conf_row[0]["c"] or 0, 3)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "documents": docs, "persons": persons, "dated_rituals": rituals,
        "unique_locations": locations, "flagged_records": flagged,
        "avg_confidence": avg_conf, "query_time_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict], query_ms: float) -> None:
    if not rows:
        print("No results found.")
        print(f"Query time: {query_ms} ms")
        return

    print(f"Found {len(rows)} result(s) — {query_ms} ms\n")
    if not rows:
        return

    keys = list(rows[0].keys())
    col_widths = {k: max(len(str(k)), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    col_widths = {k: min(v, 30) for k, v in col_widths.items()}

    header = "  ".join(str(k).ljust(col_widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "  ".join(str(row.get(k, ""))[:col_widths[k]].ljust(col_widths[k]) for k in keys)
        print(line)


def _export(rows: list[dict], path: str) -> None:
    if not _PANDAS_AVAILABLE:
        print("pandas not installed — cannot export to Excel. Install with: pip install pandas openpyxl")
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, engine="openpyxl")
    print(f"Exported {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Search the Hindi manuscript database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--name",       help="Search by person name (partial match)")
    parser.add_argument("--caste",      help="Filter by caste")
    parser.add_argument("--place",      help="Filter by village / place of origin")
    parser.add_argument("--date-range", nargs=2, type=int, metavar=("FROM_YEAR", "TO_YEAR"),
                        help="Filter by Gregorian year range")
    parser.add_argument("--family-id",  help="Show all members of a family (e.g. F042)")
    parser.add_argument("--relation",   help="Filter by relation keyword (e.g. पिता, भाई)")
    parser.add_argument("--flagged",    action="store_true",
                        help="Show low-confidence records flagged for review")
    parser.add_argument("--stats",      action="store_true",
                        help="Print database summary statistics")
    parser.add_argument("--export",     metavar="FILE.xlsx",
                        help="Export results to Excel file")
    parser.add_argument("--db",         default=DB_PATH,
                        help=f"Path to SQLite database (default: {DB_PATH})")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found at {args.db}.")
        print("Run batch_processor.py first to populate the database.")
        sys.exit(1)

    rows, elapsed_ms = [], 0.0

    if args.stats:
        s = get_stats(args.db)
        print("Database Statistics")
        print("=" * 35)
        for k, v in s.items():
            print(f"  {k:<25} {v}")
        return

    if args.name:
        rows, elapsed_ms = search_by_name(args.name, args.db)
    elif args.caste:
        rows, elapsed_ms = search_by_caste(args.caste, args.db)
    elif args.place:
        rows, elapsed_ms = search_by_place(args.place, args.db)
    elif args.date_range:
        rows, elapsed_ms = search_by_date_range(args.date_range[0], args.date_range[1], args.db)
    elif args.family_id:
        rows, elapsed_ms = search_by_family(args.family_id, args.db)
    elif args.relation:
        rows, elapsed_ms = search_by_relation(args.relation, args.db)
    elif args.flagged:
        rows, elapsed_ms = search_flagged(args.db)
    else:
        parser.print_help()
        return

    _print_table(rows, elapsed_ms)

    if args.export and rows:
        _export(rows, args.export)


if __name__ == "__main__":
    main()
