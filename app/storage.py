"""SQLite persistence with whole-version atomic replacement.

Every successful upload becomes a new version.  The write transaction
deletes the previous version and inserts the new one, so readers observe
either the entire old round or the entire new round -- never a mixture.
Readers open one explicit read transaction, which in WAL mode gives a
snapshot taken at the first read; concurrent writers cannot alter it.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from .pairing import Observation, PairingResult

DEFAULT_DATABASE_PATH = "/data/trips.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    version      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL,
    entry_name   TEXT    NOT NULL,
    exit_name    TEXT    NOT NULL,
    entry_sha256 TEXT    NOT NULL,
    exit_sha256  TEXT    NOT NULL,
    entry_size   INTEGER NOT NULL,
    exit_size    INTEGER NOT NULL,
    max_trips    INTEGER NOT NULL,
    min_deviation_seconds INTEGER NOT NULL,
    optimal_matchings     INTEGER NOT NULL,
    entry_csv    BLOB    NOT NULL,
    exit_csv     BLOB    NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    version        INTEGER NOT NULL REFERENCES rounds(version) ON DELETE CASCADE,
    observation_id TEXT    NOT NULL,
    radar          TEXT    NOT NULL,
    utc_minute     INTEGER NOT NULL,
    status         TEXT    NOT NULL CHECK (status IN ('confirmed', 'pending')),
    pending_reason TEXT,
    PRIMARY KEY (version, radar, observation_id)
);

CREATE TABLE IF NOT EXISTS trips (
    version           INTEGER NOT NULL REFERENCES rounds(version) ON DELETE CASCADE,
    entry_id          TEXT    NOT NULL,
    exit_id           TEXT    NOT NULL,
    entry_minute      INTEGER NOT NULL,
    exit_minute       INTEGER NOT NULL,
    duration_seconds  INTEGER NOT NULL,
    deviation_seconds INTEGER NOT NULL,
    PRIMARY KEY (version, entry_id)
);

CREATE INDEX IF NOT EXISTS idx_trips_version ON trips(version);
CREATE INDEX IF NOT EXISTS idx_observations_version
    ON observations(version, status);
"""


class StorageError(RuntimeError):
    """A persistence failure; the previous version stays untouched."""


@dataclass(frozen=True)
class RoundMeta:
    version: int
    created_at: str
    entry_name: str
    exit_name: str
    entry_sha256: str
    exit_sha256: str
    entry_size: int
    exit_size: int
    max_trips: int
    min_deviation_seconds: int
    optimal_matchings: int


@dataclass(frozen=True)
class WriteResult:
    version: int
    replaced_version: Optional[int]
    created_at: str


@dataclass
class StoredRound:
    meta: RoundMeta
    trips: List[dict]
    pending: List[dict]


def database_path() -> Path:
    return Path(os.environ.get("DATABASE_PATH", DEFAULT_DATABASE_PATH))


def utc_minute_to_iso(minute: int) -> str:
    return datetime.fromtimestamp(minute * 60, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = FULL")
    return conn


def init_db() -> None:
    """Create the schema and ensure WAL mode (needed for reader snapshots)."""

    conn = _connect()
    try:
        mode_row = conn.execute("PRAGMA journal_mode = WAL").fetchone()
        mode = (mode_row[0] if mode_row else "").lower()
        if mode != "wal":
            raise StorageError(
                f"database could not enable WAL journal mode (got {mode!r}); "
                "refusing to run because concurrent reads could mix versions"
            )
        conn.executescript(_SCHEMA)
    except sqlite3.Error as exc:
        raise StorageError(f"cannot initialise database: {exc}") from exc
    finally:
        conn.close()


@contextmanager
def _write_transaction() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.Error as exc:
        conn.close()
        raise StorageError(f"cannot start write transaction: {exc}") from exc
    try:
        yield conn
        conn.execute("COMMIT")
    except sqlite3.Error as exc:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - rollback best effort
            pass
        raise StorageError(f"cannot commit new round: {exc}") from exc
    except BaseException:
        # Any non-SQL error (e.g. a programming bug caught by a mock) must
        # still leave the previous round intact.
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - rollback best effort
            pass
        raise
    finally:
        conn.close()


@contextmanager
def _read_snapshot() -> Iterator[sqlite3.Connection]:
    """One read transaction: all queries see the same committed version."""

    conn = _connect()
    try:
        conn.execute("BEGIN")
        yield conn
    finally:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - connection already closed
            pass
        conn.close()


def replace_round(
    *,
    result: PairingResult,
    entry_observations: List[Observation],
    exit_observations: List[Observation],
    entry_name: str,
    exit_name: str,
    entry_sha256: str,
    exit_sha256: str,
    entry_bytes: bytes,
    exit_bytes: bytes,
) -> WriteResult:
    """Atomically replace all stored results with this round."""

    created_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with _write_transaction() as conn:
            previous = conn.execute(
                "SELECT version FROM rounds ORDER BY version DESC LIMIT 1"
            ).fetchone()
            replaced_version = int(previous["version"]) if previous else None

            conn.execute("DELETE FROM rounds")  # trips/observations cascade

            cursor = conn.execute(
                """
                INSERT INTO rounds (
                    created_at, entry_name, exit_name, entry_sha256, exit_sha256,
                    entry_size, exit_size, max_trips, min_deviation_seconds,
                    optimal_matchings, entry_csv, exit_csv
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    entry_name,
                    exit_name,
                    entry_sha256,
                    exit_sha256,
                    len(entry_bytes),
                    len(exit_bytes),
                    result.max_trips,
                    result.min_total_deviation_seconds,
                    result.optimal_matchings,
                    sqlite3.Binary(entry_bytes),
                    sqlite3.Binary(exit_bytes),
                ),
            )
            version = int(cursor.lastrowid)

            observation_rows = []
            for trip in result.confirmed:
                observation_rows.append(
                    (version, trip.entry.observation_id, trip.entry.radar,
                     trip.entry.utc_minute, "confirmed", None)
                )
                observation_rows.append(
                    (version, trip.exit.observation_id, trip.exit.radar,
                     trip.exit.utc_minute, "confirmed", None)
                )
            for pending in result.pending:
                observation_rows.append(
                    (
                        version,
                        pending.observation.observation_id,
                        pending.observation.radar,
                        pending.observation.utc_minute,
                        "pending",
                        pending.reason,
                    )
                )
            conn.executemany(
                """
                INSERT INTO observations
                    (version, observation_id, radar, utc_minute, status,
                     pending_reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                observation_rows,
            )
            conn.executemany(
                """
                INSERT INTO trips
                    (version, entry_id, exit_id, entry_minute, exit_minute,
                     duration_seconds, deviation_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        version,
                        trip.entry.observation_id,
                        trip.exit.observation_id,
                        trip.entry.utc_minute,
                        trip.exit.utc_minute,
                        trip.duration_seconds,
                        trip.deviation_seconds,
                    )
                    for trip in result.confirmed
                ],
            )
            return WriteResult(
                version=version,
                replaced_version=replaced_version,
                created_at=created_at,
            )
    except sqlite3.Error as exc:
        # The transaction was rolled back, so the previous round is intact.
        raise StorageError(f"failed to write new round: {exc}") from exc


def _pending_payload(row: sqlite3.Row) -> dict:
    return {
        "observation_id": row["observation_id"],
        "radar": row["radar"],
        "time": utc_minute_to_iso(row["utc_minute"]),
        "utc_minute": row["utc_minute"],
        "reason": row["pending_reason"],
    }


def load_current_round() -> Optional[StoredRound]:
    """Return the current round, read from one consistent snapshot."""
    try:
        with _read_snapshot() as conn:
            row = conn.execute(
                "SELECT * FROM rounds ORDER BY version DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            version = int(row["version"])
            trip_rows = conn.execute(
                """
                SELECT entry_id, exit_id, entry_minute, exit_minute,
                       duration_seconds, deviation_seconds
                FROM trips WHERE version = ? ORDER BY entry_minute, entry_id
                """,
                (version,),
            ).fetchall()
            pending_rows = conn.execute(
                """
                SELECT observation_id, radar, utc_minute, pending_reason
                FROM observations
                WHERE version = ? AND status = 'pending'
                ORDER BY utc_minute, radar, observation_id
                """,
                (version,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"failed to read current round: {exc}") from exc

    meta = RoundMeta(
        version=version,
        created_at=row["created_at"],
        entry_name=row["entry_name"],
        exit_name=row["exit_name"],
        entry_sha256=row["entry_sha256"],
        exit_sha256=row["exit_sha256"],
        entry_size=row["entry_size"],
        exit_size=row["exit_size"],
        max_trips=row["max_trips"],
        min_deviation_seconds=row["min_deviation_seconds"],
        optimal_matchings=row["optimal_matchings"],
    )
    trips = [
        {
            "entry_id": trip["entry_id"],
            "exit_id": trip["exit_id"],
            "entry_time": utc_minute_to_iso(trip["entry_minute"]),
            "exit_time": utc_minute_to_iso(trip["exit_minute"]),
            "duration_seconds": trip["duration_seconds"],
            "deviation_seconds": trip["deviation_seconds"],
        }
        for trip in trip_rows
    ]
    return StoredRound(
        meta=meta,
        trips=trips,
        pending=[_pending_payload(p) for p in pending_rows],
    )
