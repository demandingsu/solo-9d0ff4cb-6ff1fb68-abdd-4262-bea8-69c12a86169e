"""API-level tests: upload, validation, atomic replacement, snapshots."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import storage
from app.main import app

BASE = datetime(2026, 11, 15, 22, 0, tzinfo=timezone.utc)


def iso(minute_offset: int) -> str:
    ts = BASE.timestamp() + minute_offset * 60
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def csv_bytes(rows, header="observation_id,utc_minute") -> bytes:
    body = "\n".join(f"{obs_id},{iso(offset)}" for obs_id, offset in rows)
    return (header + "\n" + body + "\n").encode()


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


CLEAN_ENTRY = [("E1", 0), ("E2", 2), ("E3", 10)]
CLEAN_EXIT = [("X1", 5), ("X2", 7), ("X3", 15)]


def test_docs_and_health(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_empty_state_returns_404(client):
    assert client.get("/api/rounds/current").status_code == 404
    assert client.get("/api/trips").status_code == 404
    assert client.get("/api/pending").status_code == 404


def test_upload_clean_round(client):
    response = upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["version"] == 1
    assert body["replaced_version"] is None
    assert body["summary"] == {
        "max_trips": 3,
        "confirmed_trips": 3,
        "pending_observations": 0,
        "min_total_deviation_seconds": 0,
        "optimal_pairings": 1,
    }
    assert [
        (t["entry_id"], t["exit_id"], t["duration_seconds"])
        for t in body["confirmed_trips"]
    ] == [("E1", "X1", 300), ("E2", "X2", 300), ("E3", "X3", 300)]
    assert body["pending_observations"] == []
    assert body["sources"]["entry"]["filename"] == "entry.csv"
    assert len(body["sources"]["entry"]["sha256"]) == 64

    current = client.get("/api/rounds/current").json()
    assert current["version"] == 1
    assert current["confirmed_trips"] == body["confirmed_trips"]
    assert client.get("/api/trips").json() == body["confirmed_trips"]
    assert client.get("/api/pending").json() == []


def test_ambiguous_round_has_no_confirmations(client):
    response = upload(client, [("E1", 0), ("E2", 0)], [("X1", 5), ("X2", 5)])
    assert response.status_code == 201
    body = response.json()
    assert body["summary"]["max_trips"] == 2
    assert body["summary"]["optimal_pairings"] == 2
    assert body["confirmed_trips"] == []
    assert {p["observation_id"] for p in body["pending_observations"]} == {
        "E1",
        "E2",
        "X1",
        "X2",
    }
    assert all(
        p["reason"] == "ambiguous_pairing" for p in body["pending_observations"]
    )


def test_partial_round_reports_reasons(client):
    response = upload(
        client,
        [("E1", 2), ("E2", 5), ("E3", 6)],
        [("X1", 10), ("X2", 11), ("X3", 12)],
    )
    body = response.json()
    assert body["summary"]["optimal_pairings"] == 2
    assert [(t["entry_id"], t["exit_id"]) for t in body["confirmed_trips"]] == [
        ("E1", "X1")
    ]
    reasons = {p["observation_id"]: p["reason"] for p in body["pending_observations"]}
    assert reasons == {
        "E2": "ambiguous_pairing",
        "E3": "ambiguous_pairing",
        "X2": "ambiguous_pairing",
        "X3": "ambiguous_pairing",
    }


def test_duration_window_inclusive_over_http(client):
    response = upload(client, [("E1", 0), ("E2", 20)], [("X1", 3), ("X2", 28)])
    body = response.json()
    assert body["summary"]["confirmed_trips"] == 2
    assert {t["duration_seconds"] for t in body["confirmed_trips"]} == {180, 480}


def test_alternate_headers_and_extra_columns(client):
    entry = (
        "观测号,观测时间,备注\n"
        f"E1,{iso(0)},入口车道A\n"
        f"E2,{iso(1)},入口车道B\n"
    ).encode()
    exit_ = (
        "time,id\n"
        f"{iso(5)},X1\n"
        f"{iso(9)},X2\n"
    ).encode()
    response = client.post(
        "/api/rounds",
        files={
            "entry_file": ("entry.csv", entry, "text/csv"),
            "exit_file": ("exit.csv", exit_, "text/csv"),
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["summary"]["confirmed_trips"] == 2
    assert {t["duration_seconds"] for t in body["confirmed_trips"]} == {300, 480}


def test_utf8_bom_and_crlf_are_accepted(client):
    text = (
        "﻿observation_id,utc_minute\r\n"
        f"E1,{iso(0)}\r\n"
        f"X1-dummy,{iso(5)}\r\n"
    ).encode("utf-8")
    exit_text = ("id,time\r\n" f"X1,{iso(5)}\r\n").encode("utf-8")
    response = client.post(
        "/api/rounds",
        files={
            "entry_file": ("entry.csv", text, "text/csv"),
            "exit_file": ("exit.csv", exit_text, "text/csv"),
        },
    )
    assert response.status_code == 201, response.text


@pytest.mark.parametrize(
    "entry_bytes,exit_bytes",
    [
        # more than 12 rows
        (
            (
                "observation_id,utc_minute\n"
                + "".join(f"E{i},{iso(i)}\n" for i in range(13))
            ).encode(),
            ("observation_id,utc_minute\nX1," + iso(5) + "\n").encode(),
        ),
        # missing time column
        (b"observation_id,note\nE1,hello\n", b"id,time\nX1," + iso(5).encode() + b"\n"),
        # seconds not zero
        (
            b"observation_id,utc_minute\nE1,2026-11-15T22:00:30Z\n",
            b"id,time\nX1," + iso(5).encode() + b"\n",
        ),
        # non-UTC offset
        (
            b"observation_id,utc_minute\nE1,2026-11-15T22:00:00+08:00\n",
            b"id,time\nX1," + iso(5).encode() + b"\n",
        ),
        # duplicate ids
        (
            ("observation_id,utc_minute\nE1," + iso(0) + "\nE1," + iso(1) + "\n").encode(),
            ("id,time\nX1," + iso(5) + "\n").encode(),
        ),
        # empty file
        (b"", b"id,time\nX1," + iso(5).encode() + b"\n"),
        # invalid date
        (b"observation_id,utc_minute\nE1,2026-02-30T22:00:00Z\n", b"id,time\nX1," + iso(5).encode() + b"\n"),
        # bad utf-8
        (b"observation_id,utc_minute\nE1,\xff\xfe\n", b"id,time\nX1," + iso(5).encode() + b"\n"),
    ],
)
def test_invalid_files_are_rejected(client, entry_bytes, exit_bytes):
    response = client.post(
        "/api/rounds",
        files={
            "entry_file": ("entry.csv", entry_bytes, "text/csv"),
            "exit_file": ("exit.csv", exit_bytes, "text/csv"),
        },
    )
    assert response.status_code == 422, response.text
    # Nothing was stored.
    assert client.get("/api/rounds/current").status_code == 404


def test_failed_upload_keeps_previous_version(client):
    first = upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    assert first.status_code == 201
    before = client.get("/api/rounds/current").json()

    bad = client.post(
        "/api/rounds",
        files={
            "entry_file": (
                "entry.csv",
                b"observation_id,utc_minute\nE1,not-a-time\n",
                "text/csv",
            ),
            "exit_file": (
                "exit.csv",
                ("id,time\nX1," + iso(5) + "\n").encode(),
                "text/csv",
            ),
        },
    )
    assert bad.status_code == 422

    after = client.get("/api/rounds/current").json()
    assert after == before
    assert after["version"] == 1


def test_successful_upload_replaces_whole_version(client):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    second = upload(client, [("E9", 0), ("E8", 1)], [("X9", 5), ("X8", 9)])
    assert second.status_code == 201
    body = second.json()
    assert body["version"] == 2
    assert body["replaced_version"] == 1

    current = client.get("/api/rounds/current").json()
    assert current["version"] == 2
    assert {t["entry_id"] for t in current["confirmed_trips"]} == {"E9", "E8"}
    # Old rows must be gone entirely.
    triples = client.get("/api/trips").json()
    assert {t["entry_id"] for t in triples} == {"E9", "E8"}
    assert client.get("/api/pending").json() == []

    db = sqlite3.connect(str(Path(storage.database_path())))
    try:
        assert db.execute("SELECT COUNT(*) FROM rounds").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 2
        version = db.execute("SELECT version FROM rounds").fetchone()[0]
        assert version == 2
        stored_versions = {
            row[0] for row in db.execute("SELECT DISTINCT version FROM trips")
        }
        assert stored_versions == {2}
    finally:
        db.close()


def test_write_failure_keeps_previous_version(client, monkeypatch):
    upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    before = client.get("/api/rounds/current").json()

    def boom(**_kwargs):
        raise storage.StorageError("simulated disk failure")

    monkeypatch.setattr(storage, "replace_round", boom)
    response = upload(client, [("E1", 0)], [("X1", 5)])
    assert response.status_code == 500
    assert "保留上一版" in response.json()["detail"]

    after = client.get("/api/rounds/current").json()
    assert after == before


def test_real_sql_failure_midwrite_rolls_back(client):
    """A failure inside the write transaction must leave the old round intact."""

    first = upload(client, CLEAN_ENTRY, CLEAN_EXIT)
    assert first.status_code == 201
    before = client.get("/api/rounds/current").json()

    db = sqlite3.connect(str(Path(storage.database_path())))
    try:
        db.execute(
            "CREATE TRIGGER fail_trips_insert BEFORE INSERT ON trips "
            "BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END"
        )
        db.commit()
    finally:
        db.close()

    response = upload(client, [("N1", 0), ("N2", 1)], [("M1", 5), ("M2", 6)])
    assert response.status_code == 500
    assert "保留上一版" in response.json()["detail"]

    after = client.get("/api/rounds/current").json()
    assert after == before

    db = sqlite3.connect(str(Path(storage.database_path())))
    try:
        db.execute("DROP TRIGGER fail_trips_insert")
        db.commit()
        assert db.execute("SELECT COUNT(*) FROM rounds").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 6
    finally:
        db.close()

    # After the underlying fault is gone, writes work again.
    recovered = upload(client, [("N1", 0), ("N2", 1)], [("M1", 5), ("M2", 6)])
    assert recovered.status_code == 201
    assert recovered.json()["summary"]["confirmed_trips"] == 2


def test_reader_never_sees_mixed_versions(client):
    """Concurrent readers during a replacement must see one full snapshot."""

    version_a_entry = [("A1", 0), ("A2", 1), ("A3", 2)]
    version_a_exit = [("B1", 5), ("B2", 6), ("B3", 7)]
    version_b_entry = [("C1", 0), ("C2", 1), ("C3", 2)]
    version_b_exit = [("D1", 5), ("D2", 6), ("D3", 7)]

    upload(client, version_a_entry, version_a_exit)

    stop = threading.Event()
    failures = []
    observations = []

    def reader():
        while not stop.is_set():
            response = client.get("/api/rounds/current")
            if response.status_code != 200:
                failures.append(response.status_code)
                continue
            body = response.json()
            entry_ids = {t["entry_id"] for t in body["confirmed_trips"]}
            exit_ids = {t["exit_id"] for t in body["confirmed_trips"]}
            observations.append((body["version"], frozenset(entry_ids), frozenset(exit_ids)))

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for _ in range(8):
            response = upload(client, version_b_entry, version_b_exit)
            assert response.status_code == 201
    finally:
        stop.set()
        thread.join()

    assert not failures
    assert observations, "reader thread made no observations"
    dataset_a = (frozenset({"A1", "A2", "A3"}), frozenset({"B1", "B2", "B3"}))
    dataset_b = (frozenset({"C1", "C2", "C3"}), frozenset({"D1", "D2", "D3"}))
    for version, entries, exits in observations:
        assert (entries, exits) in (dataset_a, dataset_b), (version, entries, exits)


def test_twelve_rows_per_file_is_accepted(client):
    # 12 rows on each side is the documented maximum and must work.  All
    # identical times make every completion optimal, so nothing is confirmed
    # and every observation is reported pending.
    entry_rows = [(f"E{i:02d}", 0) for i in range(12)]
    exit_rows = [(f"X{i:02d}", 5) for i in range(12)]
    response = upload(client, entry_rows, exit_rows)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["summary"]["max_trips"] == 12
    assert body["summary"]["optimal_pairings"] == 479001600  # 12!
    assert body["confirmed_trips"] == []
    assert len(body["pending_observations"]) == 24


def test_oversized_file_is_rejected(client):
    filler = b"x" * (1024 * 1024 + 1)
    response = client.post(
        "/api/rounds",
        files={
            "entry_file": ("entry.csv", filler, "text/csv"),
            "exit_file": ("exit.csv", csv_bytes(CLEAN_EXIT), "text/csv"),
        },
    )
    assert response.status_code == 422
    assert client.get("/api/rounds/current").status_code == 404


def test_missing_file_field_is_422(client):
    response = client.post(
        "/api/rounds",
        files={"entry_file": ("entry.csv", csv_bytes(CLEAN_ENTRY), "text/csv")},
    )
    assert response.status_code == 422
