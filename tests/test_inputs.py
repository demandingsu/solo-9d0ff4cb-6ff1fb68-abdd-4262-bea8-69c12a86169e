"""Tests for CSV parsing and validation (unit level, no HTTP)."""

from __future__ import annotations

import pytest

from app.inputs import InputError, parse_csv_file
from app.pairing import ENTRY, EXIT


def parse(text: str, radar: str = ENTRY):
    return parse_csv_file(text.encode("utf-8"), radar, "f.csv")


def errors_of(text: str, radar: str = ENTRY):
    with pytest.raises(InputError) as info:
        parse(text, radar)
    return info.value.errors


def test_comma_separated():
    parsed = parse("observation_id,utc_minute\nE1,2026-11-15T22:00:00Z\n")
    assert [o.observation_id for o in parsed.observations] == ["E1"]
    assert parsed.observations[0].radar == ENTRY


def test_semicolon_and_tab_delimiters():
    assert parse("observation_id;utc_minute\nE1;2026-11-15T22:00Z\n").observations
    assert parse("observation_id\tutc_minute\nE1\t2026-11-15T22:00Z\n").observations


def test_chinese_headers_and_extra_columns():
    parsed = parse(
        "观测号,观测时间,车道\nE1,2026-11-15T22:00Z,A\n"
    )
    assert parsed.observations[0].observation_id == "E1"


def test_header_order_is_irrelevant():
    parsed = parse("utc_minute,observation_id\n2026-11-15T22:00Z,E1\n")
    assert parsed.observations[0].observation_id == "E1"


def test_zz_utc_suffix_accepted():
    parsed = parse("observation_id,utc_minute\nE1,2026-11-15T22:00z\n")
    assert parsed.observations[0].utc_minute == 29913000


def test_blank_lines_are_skipped():
    parsed = parse(
        "observation_id,utc_minute\n\nE1,2026-11-15T22:00Z\n\n\n"
    )
    assert len(parsed.observations) == 1


@pytest.mark.parametrize(
    "text,needle",
    [
        ("", "empty"),
        ("observation_id,utc_minute\n", "no observation rows"),
        ("observation_id\nE1\n", "missing UTC minute"),
        ("utc_minute\n2026-11-15T22:00Z\n", "missing observation id"),
        (
            "observation_id,utc_minute\nE1,2026-11-15T22:00Z\nE1,2026-11-15T22:01Z\n",
            "duplicate observation id",
        ),
        ("observation_id,utc_minute\nE1,2026-11-15T22:00:30Z\n", "whole UTC minute"),
        ("observation_id,utc_minute\nE1,2026-02-30T22:00Z\n", "invalid calendar"),
        ("observation_id,utc_minute\n,2026-11-15T22:00Z\n", "observation id is empty"),
        ("observation_id,utc_minute\nE1,\n", "UTC minute is empty"),
        ("observation_id,utc_minute\nE1,2026-11-15T22:00+08:00\n", "UTC"),
        ("observation_id,observation,utc_minute\nE1,E2,2026-11-15T22:00Z\n", "more than one observation id"),
        ("observation_id,time,utc_minute\nE1,t,2026-11-15T22:00Z\n", "more than one UTC minute"),
        ("observation_id,utc_minute\nE1,2026-11-15T22:00Z,extra\n", "expected 2 columns"),
    ],
)
def test_invalid_content_is_reported(text, needle):
    errors = errors_of(text)
    assert any(needle in message for message in errors), errors


def test_too_many_rows_are_reported():
    rows = "".join(
        f"E{i},2026-11-15T22:00Z\n" for i in range(13)
    )
    errors = errors_of("observation_id,utc_minute\n" + rows)
    assert any("more than 12 observation rows" in message for message in errors)


def test_thirteen_rows_with_duplicates_still_reports_row_limit():
    rows = "".join(f"E,2026-11-15T22:00Z\n" for _ in range(13))
    errors = errors_of("observation_id,utc_minute\n" + rows)
    assert any("more than 12" in message for message in errors)


def test_non_utf8_is_rejected():
    with pytest.raises(InputError) as info:
        parse_csv_file(b"observation_id,utc_minute\nE1,\xff\n", ENTRY, "f.csv")
    assert "not valid UTF-8" in info.value.errors[0]


def test_oversized_file_is_rejected():
    from app.inputs import MAX_UPLOAD_BYTES

    with pytest.raises(InputError) as info:
        parse_csv_file(b"x" * (MAX_UPLOAD_BYTES + 1), ENTRY, "f.csv")
    assert "larger than" in info.value.errors[0]


def test_all_errors_in_one_pass_are_collected():
    text = (
        "observation_id,utc_minute\n"
        "E1,not-a-time\n"
        "E1,2026-11-15T22:01Z\n"
        "E3,2026-02-30T22:02Z\n"
    )
    errors = errors_of(text)
    assert len(errors) == 3, errors


def test_exit_radar_is_tagged_correctly():
    parsed = parse("id,time\nX1,2026-11-15T22:05Z\n", EXIT)
    assert parsed.observations[0].radar == EXIT
