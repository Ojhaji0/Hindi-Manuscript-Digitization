"""
SQLite database layer for manuscript extraction results.

Schema:
  documents  – one row per processed image file
  persons    – one row per extracted individual
  rituals    – one row per ritual date/event linked to a person
  locations  – normalized unique place names

Indexes are built after bulk insert for maximum write speed,
then queried via search.py.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Iterable

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "manuscripts.db")

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name       TEXT NOT NULL,
    bahi_number     TEXT,
    folio_number    TEXT,
    processed_at    REAL NOT NULL,
    record_count    INTEGER DEFAULT 0,
    UNIQUE(file_name)
);

CREATE TABLE IF NOT EXISTS locations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    UNIQUE(normalized_name)
);

CREATE TABLE IF NOT EXISTS persons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id),
    individual_id   TEXT,
    given_name      TEXT,
    surname         TEXT,
    gender          TEXT,
    relation        TEXT,
    caste           TEXT,
    subcaste        TEXT,
    place           TEXT,
    family_id       TEXT,
    confidence      REAL DEFAULT 0.5,
    flagged         INTEGER DEFAULT 0,
    additional_info TEXT
);

CREATE TABLE IF NOT EXISTS rituals (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id            INTEGER NOT NULL REFERENCES documents(id),
    person_id              INTEGER NOT NULL REFERENCES persons(id),
    ritual_date_text       TEXT,
    ritual_date_gregorian  TEXT,
    whose_ritual           TEXT,
    family_id              TEXT
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_persons_name    ON persons(given_name);
CREATE INDEX IF NOT EXISTS idx_persons_caste   ON persons(caste);
CREATE INDEX IF NOT EXISTS idx_persons_place   ON persons(place);
CREATE INDEX IF NOT EXISTS idx_persons_family  ON persons(family_id);
CREATE INDEX IF NOT EXISTS idx_persons_flagged ON persons(flagged);
CREATE INDEX IF NOT EXISTS idx_rituals_date    ON rituals(ritual_date_gregorian);
CREATE INDEX IF NOT EXISTS idx_rituals_family  ON rituals(family_id);
CREATE INDEX IF NOT EXISTS idx_locations_name  ON locations(normalized_name);
"""


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    conn = _connect(db_path)
    conn.executescript(_DDL)
    conn.commit()
    conn.close()


def build_indexes(db_path: str = DB_PATH) -> None:
    conn = _connect(db_path)
    conn.executescript(_INDEXES)
    conn.commit()
    conn.close()


def _upsert_location(conn: sqlite3.Connection, place: str) -> None:
    if not place:
        return
    norm = place.strip().lower()
    conn.execute(
        "INSERT OR IGNORE INTO locations (name, normalized_name) VALUES (?, ?)",
        (place.strip(), norm),
    )


def write_records(
    records: list[dict],
    file_name: str,
    bahi_number: str = "",
    folio_number: str = "",
    db_path: str = DB_PATH,
) -> int:
    """
    Insert extracted records for one image into the database.
    Returns the document_id assigned to this file.
    """
    init_db(db_path)
    conn = _connect(db_path)

    # Upsert document row
    conn.execute(
        """INSERT INTO documents (file_name, bahi_number, folio_number, processed_at, record_count)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(file_name) DO UPDATE SET
               processed_at = excluded.processed_at,
               record_count = excluded.record_count""",
        (file_name, bahi_number, folio_number, time.time(), len(records)),
    )
    doc_id = conn.execute(
        "SELECT id FROM documents WHERE file_name = ?", (file_name,)
    ).fetchone()["id"]

    for rec in records:
        place = rec.get("From Which Place", "")
        _upsert_location(conn, place)

        conn.execute(
            """INSERT INTO persons
               (document_id, individual_id, given_name, surname, gender,
                relation, caste, subcaste, place, family_id,
                confidence, flagged, additional_info)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                rec.get("Individual ID", ""),
                rec.get("Given Name", ""),
                rec.get("Surname", ""),
                rec.get("Gender", ""),
                rec.get("Relation", ""),
                rec.get("Caste", ""),
                rec.get("Subcaste", ""),
                place,
                rec.get("Family Id", ""),
                rec.get("confidence", 0.5),
                int(rec.get("flagged", False)),
                rec.get("Additional Information 1", ""),
            ),
        )
        person_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        date_text = rec.get("Date of Ritual", "")
        date_greg = rec.get("Date of Ritual (Gregorian)", "")
        whose = rec.get("Whose Ritual 1", "")
        if date_text or date_greg or whose:
            conn.execute(
                """INSERT INTO rituals
                   (document_id, person_id, ritual_date_text,
                    ritual_date_gregorian, whose_ritual, family_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (doc_id, person_id, date_text, date_greg, whose, rec.get("Family Id", "")),
            )

    conn.commit()
    conn.close()
    return doc_id


def query(
    sql: str,
    params: tuple = (),
    db_path: str = DB_PATH,
) -> list[dict]:
    conn = _connect(db_path)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
