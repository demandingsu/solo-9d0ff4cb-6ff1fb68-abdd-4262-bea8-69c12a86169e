"""Strict but friendly parsing of the two radar CSV files."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

from .pairing import (
    ENTRY,
    EXIT,
    MAX_OBSERVATIONS_PER_FILE,
    Observation,
)

MAX_UPLOAD_BYTES = 1024 * 1024
MAX_ID_LENGTH = 128

# Accepted header spellings (matched after lower-casing and trimming).
_ID_HEADERS = {
    "observation_id",
    "observation",
    "obs_id",
    "id",
    "passage_id",
    "record_id",
    "观测号",
    "观测编号",
    "编号",
}
_TIME_HEADERS = {
    "utc_minute",
    "utc_minutes",
    "utc_time",
    "utc_timestamp",
    "time",
    "timestamp",
    "observed_at",
    "observation_time",
    "utc整分钟",
    "utc时间",
    "观测时间",
    "时间",
}
# Extra columns are tolerated on input; they are not used by the pairing.

_TIMESTAMP_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})[T ]"
    r"(?P<hour>\d{2}):(?P<minute>\d{2})"
    r"(?::(?P<second>\d{2}))?"
    r"(?P<zone>[Zz]|[+-]\d{2}:\d{2})?$"
)


class InputError(ValueError):
    """A CSV file failed validation; ``errors`` holds user-facing messages."""

    def __init__(self, errors: Sequence[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = list(errors)


@dataclass(frozen=True)
class ParsedFile:
    filename: str
    radar: str
    observations: List[Observation]


def _normalise_header(name: str) -> str:
    return name.replace("﻿", "").strip().lower()


def _parse_utc_minute(raw: str) -> int:
    value = raw.strip()
    match = _TIMESTAMP_RE.match(value)
    if not match:
        raise ValueError(
            "expected UTC time like 2026-11-15T23:05Z or "
            "2026-11-15 23:05 (whole minute)"
        )
    second = match.group("second")
    if second is not None and int(second) != 0:
        raise ValueError("time must be a whole UTC minute (seconds must be 00)")
    zone = match.group("zone")
    if zone is not None and zone.lower() != "z" and zone != "+00:00":
        raise ValueError("time must be expressed in UTC (Z or +00:00)")
    try:
        moment = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            tzinfo=timezone.utc,
        )
    except ValueError as exc:
        raise ValueError(f"invalid calendar date/time: {value!r}") from exc
    return int(moment.timestamp()) // 60


def _choose_columns(header: Sequence[str]) -> Tuple[int, int]:
    id_columns: List[int] = []
    time_columns: List[int] = []
    normalised: List[str] = []
    for index, raw in enumerate(header):
        name = _normalise_header(raw)
        if not name:
            raise InputError([f"header column {index + 1} is empty"])
        if name in normalised:
            raise InputError([f"duplicate header column: {raw.strip()!r}"])
        normalised.append(name)
        if name in _ID_HEADERS:
            id_columns.append(index)
        elif name in _TIME_HEADERS:
            time_columns.append(index)
    if not id_columns:
        raise InputError(
            [
                "missing observation id column "
                "(accepted headers include observation_id / 观测号 / id)"
            ]
        )
    if not time_columns:
        raise InputError(
            [
                "missing UTC minute column "
                "(accepted headers include utc_minute / time / 观测时间)"
            ]
        )
    if len(id_columns) > 1:
        raise InputError(["more than one observation id column in header"])
    if len(time_columns) > 1:
        raise InputError(["more than one UTC minute column in header"])
    return id_columns[0], time_columns[0]


def _detect_delimiter(text: str) -> str:
    """Pick the delimiter from the header line (comma, semicolon, tab)."""

    header_line = ""
    for line in text.splitlines():
        if line.strip():
            header_line = line
            break
    counts = {delimiter: header_line.count(delimiter) for delimiter in ",;\t"}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] > 0 else ","


def parse_csv_file(
    data: bytes, radar: str, filename: str
) -> ParsedFile:
    """Parse one radar CSV, raising ``InputError`` with all found problems."""

    if radar not in (ENTRY, EXIT):
        raise ValueError(f"unknown radar: {radar!r}")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InputError(
            [f"{filename}: file is larger than {MAX_UPLOAD_BYTES} bytes"]
        )
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError([f"{filename}: file is not valid UTF-8 ({exc.reason})"])

    reader = csv.reader(
        io.StringIO(text, newline=""), delimiter=_detect_delimiter(text)
    )
    header: Optional[List[str]] = None
    id_index = time_index = -1
    errors: List[str] = []
    observations: List[Observation] = []
    seen_ids: set[str] = set()
    data_rows = 0

    try:
        for row in reader:
            line = reader.line_num
            if not row or all(not cell.strip() for cell in row):
                continue  # ignore blank lines
            if header is None:
                header = row
                try:
                    id_index, time_index = _choose_columns(header)
                except InputError as exc:
                    raise InputError(
                        [f"{filename}: {message}" for message in exc.errors]
                    ) from exc
                continue

            data_rows += 1
            if data_rows > MAX_OBSERVATIONS_PER_FILE:
                errors.append(
                    f"{filename}: line {line}: more than "
                    f"{MAX_OBSERVATIONS_PER_FILE} observation rows"
                )
                break
            if len(row) != len(header):
                errors.append(
                    f"{filename}: line {line}: expected {len(header)} columns, "
                    f"found {len(row)}"
                )
                continue

            raw_id = row[id_index].strip()
            raw_time = row[time_index].strip()
            row_ok = True
            if not raw_id:
                errors.append(f"{filename}: line {line}: observation id is empty")
                row_ok = False
            elif len(raw_id) > MAX_ID_LENGTH:
                errors.append(
                    f"{filename}: line {line}: observation id is longer than "
                    f"{MAX_ID_LENGTH} characters"
                )
                row_ok = False
            elif raw_id in seen_ids:
                errors.append(
                    f"{filename}: line {line}: duplicate observation id "
                    f"{raw_id!r}"
                )
                row_ok = False
            else:
                seen_ids.add(raw_id)

            utc_minute = 0
            if not raw_time:
                errors.append(
                    f"{filename}: line {line}: UTC minute is empty"
                )
                row_ok = False
            else:
                try:
                    utc_minute = _parse_utc_minute(raw_time)
                except ValueError as exc:
                    errors.append(f"{filename}: line {line}: {exc}")
                    row_ok = False

            if not row_ok:
                continue
            observations.append(
                Observation(
                    observation_id=raw_id, radar=radar, utc_minute=utc_minute
                )
            )
    except csv.Error as exc:
        raise InputError(
            [f"{filename}: malformed CSV at line {reader.line_num}: {exc}"]
        )

    if header is None:
        raise InputError([f"{filename}: file is empty"])
    if data_rows == 0:
        raise InputError([f"{filename}: file contains no observation rows"])
    if errors:
        raise InputError(errors)
    return ParsedFile(filename=filename, radar=radar, observations=observations)
