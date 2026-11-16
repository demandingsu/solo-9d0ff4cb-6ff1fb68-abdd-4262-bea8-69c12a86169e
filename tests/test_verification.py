"""Tests for the read-only verification API and its pure verification logic."""

from __future__ import annotations

import dataclasses
import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import storage
from app.main import app
from app.pairing import ENTRY, EXIT, Observation
from app.storage import (
    RoundMeta,
    StoredObservation,
    StoredTrip,
    VerificationSnapshot,
)
from app.verification import MISMATCH, UNVERIFIABLE, VERIFIED, verify_snapshot

BASE = datetime(2026, 11, 15, 22, 0, tzinfo=timezone.utc)


def iso(minute_offset: int) -> str:
    ts = BASE.timestamp() + minute_offset * 60
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def csv_bytes(rows, header="observation_id,utc_minute") -> bytes:
    body = "\n".join(f"{obs_id},{iso(offset)}" for obs_id, offset in rows)
    return (header + "\n" + body + "\n").encode()


CLEAN_ENTRY = [("E1", 0), ("E2", 2), ("E3", 10)]
CLEAN_EXIT = [("X1", 5), ("X2", 7), ("X3", 15)]

AMBIG_ENTRY = [("E1", 0), ("E2", 0)]
AMBIG_EXIT = [("X1", 5), ("X2", 5)]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as test_client:
        yield test_client


def upload(client, entry_rows, exit_rows):
    return client.post(
        "/api/rounds",
        files={
            "entry_file": ("entry.csv", csv_bytes(entry_rows), "text/csv"),
            "exit_file": ("exit.csv", csv_bytes(exit_rows), "text/csv"),
        },
    )


def verify(client):
    return client.get("/api/rounds/current/verification")


def open_db():
    return sqlite3.connect(str(Path(storage.database_path())))


# ---------------------------------------------------------------------------
# HTTP-level behaviour
# ---------------------------------------------------------------------------


def test_verification_404_when_no_round(client):
    response = verify(client)
    assert response.status_code == 404


def test_clean_round_is_verified_and_deliverable(client):
    upload_response = upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    assert upload_response.status_code == 201

    response = verify(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["version"] == 1
    assert body["status"] == VERIFIED
    assert body["deliverable"] is True
    assert body["differences"] == []
    assert body["parse_errors"] == []
    assert len(body["digest_checks"]) == 2
    assert all(check["sha256_ok"] and check["size_ok"] for check in body["digest_checks"])
    assert {check["radar"] for check in body["digest_checks"]} == {"entry", "exit"}
    assert all(len(check["actual_sha256"]) == 64 for check in body["digest_checks"])


def test_ambiguous_round_with_pending_reasons_is_verified(client):
    upload(client, AMBIG_ENTRY, AMBIG_EXIT)
    body = verify(client).json()
    assert body["status"] == VERIFIED
    assert body["deliverable"] is True
    assert body["differences"] == []


def test_round_without_any_feasible_pair_is_verified(client):
    # Nothing lies in the 3..8 minute window: zero trips, every observation
    # pending with no_optimal_pair; verification must still pass cleanly.
    upload(client, [("E1", 0)], [("X1", 30)])
    body = verify(client).json()
    assert body["status"] == VERIFIED
    assert body["parse_errors"] == []
    assert body["differences"] == []
    assert all(check["sha256_ok"] for check in body["digest_checks"])


def test_verification_is_read_only(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    before = client.get("/api/rounds/current").json()
    verify(client)
    verify(client)
    after = client.get("/api/rounds/current").json()
    assert after == before


def test_tampered_blob_is_unverifiable_without_differences(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    with open_db() as db:
        db.execute(
            "UPDATE rounds SET entry_csv = ?",
            (sqlite3.Binary(b"observation_id,utc_minute\nE99,2026-11-15T22:00Z\n"),),
        )
        db.commit()

    body = verify(client).json()
    assert body["status"] == UNVERIFIABLE
    assert body["deliverable"] is False
    entry_check = next(c for c in body["digest_checks"] if c["radar"] == "entry")
    assert entry_check["sha256_ok"] is False
    assert entry_check["actual_sha256"] != entry_check["recorded_sha256"]
    # No field comparison may be used as evidence when the originals fail.
    assert body["differences"] == []
    assert "无法核实" in body["status_reason"]


def test_tampered_recorded_digest_is_unverifiable(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    with open_db() as db:
        db.execute("UPDATE rounds SET exit_sha256 = ?", ("0" * 64,))
        db.commit()

    body = verify(client).json()
    assert body["status"] == UNVERIFIABLE
    exit_check = next(c for c in body["digest_checks"] if c["radar"] == "exit")
    assert exit_check["sha256_ok"] is False
    entry_check = next(c for c in body["digest_checks"] if c["radar"] == "entry")
    assert entry_check["sha256_ok"] is True


def test_size_mismatch_is_unverifiable(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    with open_db() as db:
        db.execute("UPDATE rounds SET entry_size = 1")
        db.commit()

    body = verify(client).json()
    assert body["status"] == UNVERIFIABLE
    entry_check = next(c for c in body["digest_checks"] if c["radar"] == "entry")
    assert entry_check["size_ok"] is False
    assert entry_check["sha256_ok"] is True


def test_unparseable_blob_with_matching_digest_is_unverifiable(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    bad_blob = b"observation_id,utc_minute\nE1,not-a-time\n"
    with open_db() as db:
        db.execute(
            "UPDATE rounds SET entry_csv = ?, entry_sha256 = ?, entry_size = ?",
            (
                sqlite3.Binary(bad_blob),
                hashlib.sha256(bad_blob).hexdigest(),
                len(bad_blob),
            ),
        )
        db.commit()

    body = verify(client).json()
    assert body["status"] == UNVERIFIABLE
    assert all(check["sha256_ok"] for check in body["digest_checks"])
    assert body["parse_errors"]
    assert any("entry.csv" in message for message in body["parse_errors"])
    assert body["differences"] == []


def test_deleted_trip_is_mismatch_and_identifies_trip(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    with open_db() as db:
        db.execute("DELETE FROM trips WHERE entry_id = 'E2'")
        db.commit()

    body = verify(client).json()
    assert body["status"] == MISMATCH
    assert body["deliverable"] is False
    assert len(body["differences"]) == 1
    diff = body["differences"][0]
    assert diff["scope"] == "trip"
    assert diff["field"] == "existence"
    assert diff["entry_id"] == "E2"
    assert diff["exit_id"] == "X2"
    assert "不可交付" in body["status_reason"]


def test_trip_field_tampering_is_mismatch(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    with open_db() as db:
        db.execute(
            "UPDATE trips SET duration_seconds = 360 WHERE entry_id = 'E1'"
        )
        db.commit()

    body = verify(client).json()
    assert body["status"] == MISMATCH
    diffs = [d for d in body["differences"] if d["scope"] == "trip"]
    assert len(diffs) == 1
    assert diffs[0]["field"] == "duration_seconds"
    assert diffs[0]["expected"] == 300
    assert diffs[0]["actual"] == 360
    assert diffs[0]["entry_id"] == "E1"
    assert diffs[0]["exit_id"] == "X1"


def test_pending_reason_tampering_is_mismatch_with_observation_id(client):
    upload(client, AMBIG_ENTRY, AMBIG_EXIT)
    with open_db() as db:
        db.execute(
            "UPDATE observations SET pending_reason = 'no_optimal_pair' "
            "WHERE radar = 'entry' AND observation_id = 'E1'"
        )
        db.commit()

    body = verify(client).json()
    assert body["status"] == MISMATCH
    assert len(body["differences"]) == 1
    diff = body["differences"][0]
    assert diff["scope"] == "observation"
    assert diff["field"] == "pending_reason"
    assert diff["radar"] == "entry"
    assert diff["observation_id"] == "E1"
    assert diff["expected"] == "ambiguous_pairing"
    assert diff["actual"] == "no_optimal_pair"


def test_observation_status_tampering_is_mismatch(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    with open_db() as db:
        db.execute(
            "UPDATE observations SET status = 'pending', "
            "pending_reason = 'no_optimal_pair' "
            "WHERE radar = 'entry' AND observation_id = 'E1'"
        )
        db.commit()

    body = verify(client).json()
    assert body["status"] == MISMATCH
    fields = {d["field"] for d in body["differences"]}
    assert "status" in fields
    flagged = next(
        d for d in body["differences"] if d["field"] == "status"
    )
    assert flagged["observation_id"] == "E1"
    assert flagged["radar"] == "entry"


def test_extra_observation_row_is_mismatch(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    with open_db() as db:
        db.execute(
            "INSERT INTO observations "
            "(version, observation_id, radar, utc_minute, status, pending_reason) "
            "SELECT version, 'E99', 'entry', 0, 'pending', 'no_optimal_pair' "
            "FROM rounds"
        )
        db.commit()

    body = verify(client).json()
    assert body["status"] == MISMATCH
    diff = next(d for d in body["differences"] if d["field"] == "existence")
    assert diff["scope"] == "observation"
    assert diff["observation_id"] == "E99"
    assert diff["expected"] == "absent"
    assert diff["actual"] == "present"


def test_summary_tampering_is_mismatch(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    with open_db() as db:
        db.execute("UPDATE rounds SET max_trips = 99")
        db.commit()

    body = verify(client).json()
    assert body["status"] == MISMATCH
    assert len(body["differences"]) == 1
    diff = body["differences"][0]
    assert diff["scope"] == "summary"
    assert diff["field"] == "max_trips"
    assert diff["expected"] == 3
    assert diff["actual"] == 99


def test_report_always_matches_one_committed_round_under_concurrent_replaces(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    other_entry = [("C1", 0), ("C2", 1)]
    other_exit = [("D1", 5), ("D2", 6)]

    stop = threading.Event()
    reports = []
    failures = []

    def reader():
        while not stop.is_set():
            response = verify(client)
            if response.status_code != 200:
                failures.append(response.status_code)
                continue
            reports.append(response.json())

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for _ in range(8):
            response = upload(client, other_entry, other_exit)
            assert response.status_code == 201
    finally:
        stop.set()
        thread.join()

    assert not failures
    assert reports
    for report in reports:
        # Every report belongs wholly to one committed, internally consistent
        # round, so digests match and the verdict is verified.
        assert report["status"] == VERIFIED, report
        assert all(
            check["sha256_ok"] and check["size_ok"]
            for check in report["digest_checks"]
        )
        assert report["differences"] == []
        # AUTOINCREMENT keeps increasing even though only one round exists;
        # the first upload is version 1 and replacements 2..9.
        assert report["version"] in range(1, 10)


# ---------------------------------------------------------------------------
# Pure-module tests (no HTTP, hand-built snapshots)
# ---------------------------------------------------------------------------


def _minute(offset: int) -> int:
    return int(BASE.timestamp()) // 60 + offset


def _good_snapshot() -> VerificationSnapshot:
    entry_blob = csv_bytes(CLEAN_ENTRY)
    exit_blob = csv_bytes(CLEAN_EXIT)
    meta = RoundMeta(
        version=7,
        created_at="2026-11-16T00:00:00Z",
        entry_name="entry.csv",
        exit_name="exit.csv",
        entry_sha256=hashlib.sha256(entry_blob).hexdigest(),
        exit_sha256=hashlib.sha256(exit_blob).hexdigest(),
        entry_size=len(entry_blob),
        exit_size=len(exit_blob),
        max_trips=3,
        min_deviation_seconds=0,
        optimal_matchings=1,
    )
    observations = [
        StoredObservation("E1", ENTRY, _minute(0), "confirmed", None),
        StoredObservation("E2", ENTRY, _minute(2), "confirmed", None),
        StoredObservation("E3", ENTRY, _minute(10), "confirmed", None),
        StoredObservation("X1", EXIT, _minute(5), "confirmed", None),
        StoredObservation("X2", EXIT, _minute(7), "confirmed", None),
        StoredObservation("X3", EXIT, _minute(15), "confirmed", None),
    ]
    trips = [
        StoredTrip("E1", "X1", _minute(0), _minute(5), 300, 0),
        StoredTrip("E2", "X2", _minute(2), _minute(7), 300, 0),
        StoredTrip("E3", "X3", _minute(10), _minute(15), 300, 0),
    ]
    return VerificationSnapshot(
        meta=meta,
        entry_csv=entry_blob,
        exit_csv=exit_blob,
        observations=observations,
        trips=trips,
    )


def test_verify_snapshot_clean():
    report = verify_snapshot(_good_snapshot())
    assert report.status == VERIFIED
    assert report.deliverable is True
    assert report.version == 7
    assert report.differences == []
    assert report.parse_errors == []
    assert all(check.ok for check in report.digest_checks)


def test_verify_snapshot_detects_wrong_utc_minute():
    snapshot = _good_snapshot()
    tampered = StoredObservation("E1", ENTRY, _minute(99), "confirmed", None)
    snapshot = dataclasses.replace(
        snapshot,
        observations=[tampered, *snapshot.observations[1:]],
    )

    report = verify_snapshot(snapshot)
    assert report.status == MISMATCH
    diff = next(d for d in report.differences if d.field == "utc_minute")
    assert diff.observation_id == "E1"
    assert diff.actual == _minute(99)
    assert diff.expected == _minute(0)


def test_verify_snapshot_detects_deviation_and_summary():
    snapshot = _good_snapshot()
    new_trips = [
        StoredTrip("E1", "X1", _minute(0), _minute(5), 300, 120),
        *snapshot.trips[1:],
    ]
    snapshot = dataclasses.replace(
        snapshot,
        trips=new_trips,
        meta=dataclasses.replace(snapshot.meta, optimal_matchings=42),
    )

    report = verify_snapshot(snapshot)
    assert report.status == MISMATCH
    fields = {d.field for d in report.differences}
    assert "deviation_seconds" in fields
    assert "optimal_pairings" in fields


def test_verify_snapshot_unparseable_is_unverifiable():
    snapshot = _good_snapshot()
    bad = b"garbage bytes, no real header\n"
    snapshot = dataclasses.replace(
        snapshot,
        entry_csv=bad,
        meta=dataclasses.replace(
            snapshot.meta,
            entry_sha256=hashlib.sha256(bad).hexdigest(),
            entry_size=len(bad),
        ),
    )

    report = verify_snapshot(snapshot)
    assert report.status == UNVERIFIABLE
    assert report.deliverable is False
    assert report.parse_errors
    assert report.differences == []


def test_verify_snapshot_digest_failure_takes_precedence_over_differences():
    # Even with an obviously wrong stored trip, digest failure must dominate:
    # the round is reported unverifiable rather than merely mismatched.
    snapshot = _good_snapshot()
    new_trips = [
        StoredTrip("E1", "X1", _minute(0), _minute(5), 301, 1),
        *snapshot.trips[1:],
    ]
    snapshot = dataclasses.replace(
        snapshot,
        trips=new_trips,
        meta=dataclasses.replace(snapshot.meta, entry_sha256="f" * 64),
    )

    report = verify_snapshot(snapshot)
    assert report.status == UNVERIFIABLE
    assert report.differences == []


def test_verify_snapshot_observation_lists_stay_immutable_inputs():
    snapshot = _good_snapshot()
    original_observations = list(snapshot.observations)
    verify_snapshot(snapshot)
    assert snapshot.observations == original_observations
    assert isinstance(snapshot.observations[0], StoredObservation)
    # Sanity: domain objects from the pairing engine are what the parser makes.
    parsed_entry_obs = Observation("E1", ENTRY, _minute(0))
    assert parsed_entry_obs.observation_id == "E1"
