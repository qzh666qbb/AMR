from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from .models import AMR_AUDIT_VERSION, AMRRecord


SCHEMA_VERSION = "3"


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


class AMRCache:
    """SQLite cache keyed by normalized text and parser fingerprint."""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS amr_records (
                    text_sha256 TEXT NOT NULL,
                    parser_fingerprint TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    amr TEXT NOT NULL,
                    concepts_json TEXT NOT NULL,
                    triples_json TEXT NOT NULL,
                    audit_version TEXT,
                    penman_warnings_json TEXT,
                    graph_valid INTEGER,
                    roundtrip_valid INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (text_sha256, parser_fingerprint)
                );
                """
            )
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(amr_records)"
                ).fetchall()
            }
            migrations = {
                "audit_version": "TEXT",
                "penman_warnings_json": "TEXT",
                "graph_valid": "INTEGER",
                "roundtrip_valid": "INTEGER",
            }
            for name, declaration in migrations.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE amr_records ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                """
                INSERT INTO cache_metadata(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (SCHEMA_VERSION,),
            )

    def get(self, text: str, parser_fingerprint: str) -> Optional[AMRRecord]:
        normalized = normalize_text(text)
        digest = text_sha256(normalized)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT normalized_text, amr, concepts_json, triples_json,
                       audit_version, penman_warnings_json, graph_valid,
                       roundtrip_valid
                FROM amr_records
                WHERE text_sha256=? AND parser_fingerprint=?
                      AND audit_version=?
                """,
                (digest, parser_fingerprint, AMR_AUDIT_VERSION),
            ).fetchone()
        if row is None:
            return None
        if row[0] != normalized:
            raise RuntimeError("SHA-256 collision or corrupt AMR cache row")
        return AMRRecord(
            text=row[0],
            text_sha256=digest,
            parser_fingerprint=parser_fingerprint,
            amr=row[1],
            concepts=tuple(json.loads(row[2])),
            triples=tuple(tuple(item) for item in json.loads(row[3])),
            audit_version=row[4],
            penman_warnings=tuple(json.loads(row[5])),
            graph_valid=bool(row[6]),
            roundtrip_valid=bool(row[7]),
        )

    def put(self, record: AMRRecord) -> None:
        normalized = normalize_text(record.text)
        if record.text_sha256 != text_sha256(normalized):
            raise ValueError("AMRRecord text hash does not match normalized text")
        if record.audit_version != AMR_AUDIT_VERSION:
            raise ValueError("AMRRecord audit version is not current")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO amr_records(
                    text_sha256, parser_fingerprint, normalized_text,
                    amr, concepts_json, triples_json, audit_version,
                    penman_warnings_json, graph_valid, roundtrip_valid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(text_sha256, parser_fingerprint) DO UPDATE SET
                    normalized_text=excluded.normalized_text,
                    amr=excluded.amr,
                    concepts_json=excluded.concepts_json,
                    triples_json=excluded.triples_json,
                    audit_version=excluded.audit_version,
                    penman_warnings_json=excluded.penman_warnings_json,
                    graph_valid=excluded.graph_valid,
                    roundtrip_valid=excluded.roundtrip_valid
                """,
                (
                    record.text_sha256,
                    record.parser_fingerprint,
                    normalized,
                    record.amr,
                    json.dumps(record.concepts, ensure_ascii=True),
                    json.dumps(record.triples, ensure_ascii=True),
                    record.audit_version,
                    json.dumps(record.penman_warnings, ensure_ascii=True),
                    int(record.graph_valid),
                    int(record.roundtrip_valid),
                ),
            )

    def put_many(self, records: Iterable[AMRRecord]) -> None:
        rows = []
        for record in records:
            normalized = normalize_text(record.text)
            if record.text_sha256 != text_sha256(normalized):
                raise ValueError("AMRRecord text hash does not match normalized text")
            if record.audit_version != AMR_AUDIT_VERSION:
                raise ValueError("AMRRecord audit version is not current")
            rows.append(
                (
                    record.text_sha256,
                    record.parser_fingerprint,
                    normalized,
                    record.amr,
                    json.dumps(record.concepts, ensure_ascii=True),
                    json.dumps(record.triples, ensure_ascii=True),
                    record.audit_version,
                    json.dumps(record.penman_warnings, ensure_ascii=True),
                    int(record.graph_valid),
                    int(record.roundtrip_valid),
                )
            )
        if not rows:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO amr_records(
                    text_sha256, parser_fingerprint, normalized_text,
                    amr, concepts_json, triples_json, audit_version,
                    penman_warnings_json, graph_valid, roundtrip_valid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(text_sha256, parser_fingerprint) DO UPDATE SET
                    normalized_text=excluded.normalized_text,
                    amr=excluded.amr,
                    concepts_json=excluded.concepts_json,
                    triples_json=excluded.triples_json,
                    audit_version=excluded.audit_version,
                    penman_warnings_json=excluded.penman_warnings_json,
                    graph_valid=excluded.graph_valid,
                    roundtrip_valid=excluded.roundtrip_valid
                """,
                rows,
            )

    def count(self, parser_fingerprint: Optional[str] = None) -> int:
        with self._connect() as connection:
            if parser_fingerprint is None:
                row = connection.execute(
                    "SELECT COUNT(*) FROM amr_records"
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) FROM amr_records WHERE parser_fingerprint=?",
                    (parser_fingerprint,),
                ).fetchone()
        return int(row[0])
