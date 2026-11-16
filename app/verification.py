"""Read-only verification of a stored round against its original radar CSVs.

The quality-control workflow needs an independent check before the travel
time samples of a round are delivered.  Given a :class:`~app.storage.
VerificationSnapshot` read from one committed round, this module:

1. recomputes SHA-256 / size of the stored original CSV blobs and compares
   them with the digests recorded at upload time;
2. re-parses the blobs with the same strict parser used at upload;
3. re-runs the exact pairing engine on the parsed observations;
4. compares the freshly computed result with the observations, confirmed
   trips, pending reasons and summary statistics held in the database.

Verdicts
--------
``unverifiable``
    A stored digest does not match the recomputed digest, or a CSV cannot
    be parsed (or re-paired).  The database content must not be trusted on
    its own, so no field comparison is presented as evidence.
``mismatch``
    The originals are intact and parse cleanly, but at least one stored
    field differs from what the rules produce from them: the round is not
    deliverable.
``verified``
    Digests match, parsing succeeds and every compared field is equal.

The module is pure: it never opens the database and never writes anything.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .inputs import InputError, parse_csv_file
from .pairing import (
    ENTRY,
    EXIT,
    PairingError,
    PairingResult,
    pair_observations,
)
from .storage import (
    RoundMeta,
    StoredObservation,
    StoredTrip,
    VerificationSnapshot,
)

# Public verdict tokens (Chinese explanations live in the API schema).
VERIFIED = "verified"
MISMATCH = "mismatch"
UNVERIFIABLE = "unverifiable"

STATUS_MESSAGES = {
    VERIFIED: "已核实：原始文件摘要一致，重新解析与配对结果与入库内容完全相符，可交付。",
    MISMATCH: "不可交付：原始文件可核实，但重新解析/配对结果与入库内容存在差异。",
    UNVERIFIABLE: "无法核实：原始文件摘要不符或文件无法解析，不能依据库内行程判定可信。",
}


@dataclass(frozen=True)
class DigestCheck:
    """Digest comparison for one stored original CSV blob."""

    radar: str
    filename: str
    recorded_sha256: str
    actual_sha256: str
    recorded_size_bytes: int
    actual_size_bytes: int
    sha256_ok: bool
    size_ok: bool

    @property
    def ok(self) -> bool:
        return self.sha256_ok and self.size_ok


@dataclass(frozen=True)
class Difference:
    """One mismatch between the recomputed result and the stored round."""

    scope: str
    field: str
    expected: Any
    actual: Any
    radar: Optional[str] = None
    observation_id: Optional[str] = None
    entry_id: Optional[str] = None
    exit_id: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class VerificationReport:
    version: int
    created_at: str
    status: str
    status_reason: str
    digest_checks: List[DigestCheck]
    parse_errors: List[str]
    differences: List[Difference] = field(default_factory=list)

    @property
    def deliverable(self) -> bool:
        return self.status == VERIFIED


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_digests(snapshot: VerificationSnapshot) -> List[DigestCheck]:
    """Recompute digest/size of each stored blob versus the recorded values."""

    meta = snapshot.meta
    checks: List[DigestCheck] = []
    for radar, filename, blob, recorded_hash, recorded_size in (
        (
            ENTRY,
            meta.entry_name,
            snapshot.entry_csv,
            meta.entry_sha256,
            meta.entry_size,
        ),
        (
            EXIT,
            meta.exit_name,
            snapshot.exit_csv,
            meta.exit_sha256,
            meta.exit_size,
        ),
    ):
        actual_hash = _sha256(blob)
        actual_size = len(blob)
        checks.append(
            DigestCheck(
                radar=radar,
                filename=filename,
                recorded_sha256=recorded_hash,
                actual_sha256=actual_hash,
                recorded_size_bytes=recorded_size,
                actual_size_bytes=actual_size,
                sha256_ok=actual_hash == recorded_hash,
                size_ok=actual_size == recorded_size,
            )
        )
    return checks


def _observation_differences(
    result: PairingResult, stored: List[StoredObservation]
) -> List[Difference]:
    """Compare every observation's existence, status and pending reason."""

    expected: Dict[tuple, Dict[str, Any]] = {}
    for trip in result.confirmed:
        for observation in (trip.entry, trip.exit):
            expected[(observation.radar, observation.observation_id)] = {
                "utc_minute": observation.utc_minute,
                "status": "confirmed",
                "pending_reason": None,
            }
    for pending in result.pending:
        observation = pending.observation
        expected[(observation.radar, observation.observation_id)] = {
            "utc_minute": observation.utc_minute,
            "status": "pending",
            "pending_reason": pending.reason,
        }

    actual = {
        (observation.radar, observation.observation_id): observation
        for observation in stored
    }

    differences: List[Difference] = []
    for key in sorted(set(expected) | set(actual)):
        radar, observation_id = key
        if key not in actual:
            differences.append(
                Difference(
                    scope="observation",
                    field="existence",
                    expected="present",
                    actual="missing",
                    radar=radar,
                    observation_id=observation_id,
                    detail="重新配对得到的观测在入库观测表中缺失",
                )
            )
            continue
        if key not in expected:
            differences.append(
                Difference(
                    scope="observation",
                    field="existence",
                    expected="absent",
                    actual="present",
                    radar=radar,
                    observation_id=observation_id,
                    detail="入库观测表存在原始文件重新解析后得不到的观测",
                )
            )
            continue

        wanted = expected[key]
        stored_observation = actual[key]
        if stored_observation.utc_minute != wanted["utc_minute"]:
            differences.append(
                Difference(
                    scope="observation",
                    field="utc_minute",
                    expected=wanted["utc_minute"],
                    actual=stored_observation.utc_minute,
                    radar=radar,
                    observation_id=observation_id,
                )
            )
        if stored_observation.status != wanted["status"]:
            differences.append(
                Difference(
                    scope="observation",
                    field="status",
                    expected=wanted["status"],
                    actual=stored_observation.status,
                    radar=radar,
                    observation_id=observation_id,
                )
            )
        if (
            stored_observation.pending_reason != wanted["pending_reason"]
            and wanted["status"] == "pending"
        ):
            differences.append(
                Difference(
                    scope="observation",
                    field="pending_reason",
                    expected=wanted["pending_reason"],
                    actual=stored_observation.pending_reason,
                    radar=radar,
                    observation_id=observation_id,
                )
            )
    return differences


def _trip_differences(
    result: PairingResult, stored: List[StoredTrip]
) -> List[Difference]:
    """Compare confirmed trips keyed by (entry_id, exit_id)."""

    expected = {
        (trip.entry.observation_id, trip.exit.observation_id): trip
        for trip in result.confirmed
    }
    actual = {(trip.entry_id, trip.exit_id): trip for trip in stored}
    differences: List[Difference] = []

    for key in sorted(set(expected) | set(actual)):
        entry_id, exit_id = key
        if key not in actual:
            differences.append(
                Difference(
                    scope="trip",
                    field="existence",
                    expected="confirmed",
                    actual="missing",
                    entry_id=entry_id,
                    exit_id=exit_id,
                    detail="重新配对确认的行程在入库行程表中缺失",
                )
            )
            continue
        if key not in expected:
            differences.append(
                Difference(
                    scope="trip",
                    field="existence",
                    expected="not_confirmed",
                    actual="confirmed",
                    entry_id=entry_id,
                    exit_id=exit_id,
                    detail="入库行程表包含重新配对无法确认的行程",
                )
            )
            continue

        recomputed = expected[key]
        stored_trip = actual[key]
        field_specs = (
            ("entry_minute", recomputed.entry.utc_minute, stored_trip.entry_minute),
            ("exit_minute", recomputed.exit.utc_minute, stored_trip.exit_minute),
            (
                "duration_seconds",
                recomputed.duration_seconds,
                stored_trip.duration_seconds,
            ),
            (
                "deviation_seconds",
                recomputed.deviation_seconds,
                stored_trip.deviation_seconds,
            ),
        )
        for field_name, wanted, got in field_specs:
            if wanted != got:
                differences.append(
                    Difference(
                        scope="trip",
                        field=field_name,
                        expected=wanted,
                        actual=got,
                        entry_id=entry_id,
                        exit_id=exit_id,
                    )
                )
    return differences


_SUMMARY_FIELDS = (
    "max_trips",
    "min_total_deviation_seconds",
    "optimal_pairings",
)


def _summary_differences(
    result: PairingResult, meta: RoundMeta
) -> List[Difference]:
    """Compare the round-level summary statistics."""

    recomputed = {
        "max_trips": result.max_trips,
        "min_total_deviation_seconds": result.min_total_deviation_seconds,
        "optimal_pairings": result.optimal_matchings,
    }
    stored_values = {
        "max_trips": meta.max_trips,
        "min_total_deviation_seconds": meta.min_deviation_seconds,
        "optimal_pairings": meta.optimal_matchings,
    }
    return [
        Difference(
            scope="summary",
            field=field_name,
            expected=recomputed[field_name],
            actual=stored_values[field_name],
        )
        for field_name in _SUMMARY_FIELDS
        if recomputed[field_name] != stored_values[field_name]
    ]


def verify_snapshot(snapshot: VerificationSnapshot) -> VerificationReport:
    """Run the full read-only verification for one committed round."""

    meta = snapshot.meta
    digest_checks = _check_digests(snapshot)
    parse_errors: List[str] = []
    differences: List[Difference] = []

    digests_ok = all(check.ok for check in digest_checks)

    parsed_entry = None
    parsed_exit = None
    for blob, radar, filename in (
        (snapshot.entry_csv, ENTRY, meta.entry_name),
        (snapshot.exit_csv, EXIT, meta.exit_name),
    ):
        try:
            parsed = parse_csv_file(blob, radar, filename)
        except InputError as exc:
            parse_errors.extend(exc.errors)
            continue
        if radar == ENTRY:
            parsed_entry = parsed
        else:
            parsed_exit = parsed

    # Pairing only makes sense when both files parsed; a PairingError (e.g.
    # duplicate ids accepted nowhere else) is likewise unverifiable.
    result: Optional[PairingResult] = None
    if parsed_entry is not None and parsed_exit is not None:
        try:
            result = pair_observations(
                parsed_entry.observations, parsed_exit.observations
            )
        except PairingError as exc:
            parse_errors.append(str(exc))

    if not digests_ok or result is None:
        # Digests failing or unparseable originals: the database rows alone
        # cannot establish trust, so field differences are not offered as
        # evidence even when some comparisons would be possible.
        status = UNVERIFIABLE
    else:
        differences.extend(_observation_differences(result, snapshot.observations))
        differences.extend(_trip_differences(result, snapshot.trips))
        differences.extend(_summary_differences(result, meta))
        status = MISMATCH if differences else VERIFIED

    return VerificationReport(
        version=meta.version,
        created_at=meta.created_at,
        status=status,
        status_reason=STATUS_MESSAGES[status],
        digest_checks=digest_checks,
        parse_errors=parse_errors,
        differences=differences,
    )
