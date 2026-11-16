"""Tests for the exact pairing engine."""

from __future__ import annotations

import itertools
import random
from datetime import datetime, timezone
from typing import Iterable, List, Sequence, Tuple

import pytest

from app.pairing import (
    ENTRY,
    EXIT,
    MAX_OBSERVATIONS_PER_FILE,
    TARGET_DURATION_SECONDS,
    Observation,
    PairingError,
    pair_observations,
)

BASE = datetime(2026, 11, 15, 22, 0, tzinfo=timezone.utc)


def minute(offset: int) -> int:
    return int(BASE.timestamp()) // 60 + offset


def obs(observation_id: str, radar: str, offset: int) -> Observation:
    return Observation(
        observation_id=observation_id, radar=radar, utc_minute=minute(offset)
    )


def ids(observations: Iterable) -> List[str]:
    return [o.observation_id for o in observations]


# ---------------------------------------------------------------------------
# Brute force reference implementation
# ---------------------------------------------------------------------------


def brute_force(
    entries: Sequence[Observation], exits: Sequence[Observation]
) -> Tuple[int, int, List[Tuple[str, str]], int]:
    """Return (max pairs, min cost, pairs in every optimum, #optima)."""

    best_count = -1
    best_cost: int | None = None
    best_matchings: List[List[Tuple[str, str]]] = []

    # index of each entry/exit
    for size in range(min(len(entries), len(exits)) + 1):
        for chosen_entries in itertools.combinations(range(len(entries)), size):
            for exit_order in itertools.permutations(
                range(len(exits)), size
            ):
                pairs = []
                cost = 0
                ok = True
                for ei, xj in zip(chosen_entries, exit_order):
                    duration = (
                        exits[xj].utc_minute - entries[ei].utc_minute
                    ) * 60
                    if not (3 * 60 <= duration <= 8 * 60):
                        ok = False
                        break
                    cost += abs(duration - TARGET_DURATION_SECONDS)
                    pairs.append(
                        (
                            entries[ei].observation_id,
                            exits[xj].observation_id,
                        )
                    )
                if not ok:
                    continue
                count = len(pairs)
                if count > best_count or (
                    count == best_count and cost < best_cost  # type: ignore[operator]
                ):
                    best_count = count
                    best_cost = cost
                    best_matchings = [pairs]
                elif count == best_count and cost == best_cost:
                    best_matchings.append(pairs)
    assert best_cost is not None
    common = set(best_matchings[0])
    for matching in best_matchings[1:]:
        common &= set(matching)
    return best_count, best_cost, sorted(common), len(best_matchings)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clean_one_to_one_round_is_fully_confirmed():
    entries = [obs("E1", ENTRY, 0), obs("E2", ENTRY, 2), obs("E3", ENTRY, 10)]
    exits = [obs("X1", EXIT, 5), obs("X2", EXIT, 7), obs("X3", EXIT, 15)]

    result = pair_observations(entries, exits)

    assert result.max_trips == 3
    assert result.min_total_deviation_seconds == 0
    assert result.optimal_matchings == 1
    assert [(t.entry.observation_id, t.exit.observation_id) for t in result.confirmed] == [
        ("E1", "X1"),
        ("E2", "X2"),
        ("E3", "X3"),
    ]
    assert result.pending == []


def test_duration_window_is_inclusive_at_three_and_eight_minutes():
    entries = [obs("E1", ENTRY, 0), obs("E2", ENTRY, 20)]
    exits = [obs("X1", EXIT, 3), obs("X2", EXIT, 28)]

    result = pair_observations(entries, exits)

    durations = {t.entry.observation_id: t.duration_seconds for t in result.confirmed}
    assert durations == {"E1": 180, "E2": 480}
    assert result.max_trips == 2


@pytest.mark.parametrize("gap", [2, 9, -5])
def test_outside_window_is_never_paired(gap):
    entries = [obs("E1", ENTRY, 0)]
    exits = [obs("X1", EXIT, gap)]

    result = pair_observations(entries, exits)

    assert result.max_trips == 0
    assert result.confirmed == []
    assert {p.observation.observation_id for p in result.pending} == {"E1", "X1"}


def test_ambiguous_identical_times_are_all_pending():
    entries = [obs("E1", ENTRY, 0), obs("E2", ENTRY, 0)]
    exits = [obs("X1", EXIT, 5), obs("X2", EXIT, 5)]

    result = pair_observations(entries, exits)

    assert result.max_trips == 2
    assert result.optimal_matchings == 2
    assert result.confirmed == []
    assert {p.observation.observation_id for p in result.pending} == {
        "E1",
        "E2",
        "X1",
        "X2",
    }
    assert all(p.reason == "ambiguous_pairing" for p in result.pending)


def test_only_forced_pairs_are_delivered():
    # Two optimal 3-trip pairings exist; E1-X1 (exactly 8 minutes) is in
    # both of them, while all other observations are interchangeable.
    entries = [obs("E1", ENTRY, 2), obs("E2", ENTRY, 5), obs("E3", ENTRY, 6)]
    exits = [obs("X1", EXIT, 10), obs("X2", EXIT, 11), obs("X3", EXIT, 12)]

    result = pair_observations(entries, exits)

    assert result.max_trips == 3
    assert result.optimal_matchings == 2
    assert [(t.entry.observation_id, t.exit.observation_id) for t in result.confirmed] == [
        ("E1", "X1")
    ]
    assert {p.observation.observation_id for p in result.pending} == {
        "E2",
        "E3",
        "X2",
        "X3",
    }
    assert all(p.reason == "ambiguous_pairing" for p in result.pending)


def test_cardinality_beats_cost():
    # Pairing E1-X1 alone is exactly 5 minutes (cost 0) but only one trip;
    # the only 2-trip matching must win even though its cost is higher.
    entries = [obs("E1", ENTRY, 0), obs("E2", ENTRY, 1)]
    exits = [obs("X1", EXIT, 5), obs("X2", EXIT, 9)]

    result = pair_observations(entries, exits)

    assert result.max_trips == 2
    # E1-X1 is 5m (cost 0), E2-X2 is 8m (cost 180); the alternative
    # E1-X2 (9m) is infeasible, so this optimum is unique.
    assert result.min_total_deviation_seconds == 180
    assert result.optimal_matchings == 1
    assert len(result.confirmed) == 2


def test_zero_trips_when_nothing_in_window():
    result = pair_observations([obs("E1", ENTRY, 0)], [obs("X1", EXIT, 30)])
    assert result.max_trips == 0
    assert result.min_total_deviation_seconds == 0
    assert result.confirmed == []
    assert len(result.pending) == 2


def test_unmatched_observation_reason_is_no_optimal_pairing():
    # X2 is 60 minutes after everything and can never pair.
    entries = [obs("E1", ENTRY, 0)]
    exits = [obs("X1", EXIT, 4), obs("X2", EXIT, 60)]

    result = pair_observations(entries, exits)

    reasons = {p.observation.observation_id: p.reason for p in result.pending}
    assert reasons == {"X2": "no_optimal_pair"}


def test_one_entry_two_identical_exits_is_ambiguous():
    entries = [obs("E1", ENTRY, 0)]
    exits = [obs("X1", EXIT, 5), obs("X2", EXIT, 5)]

    result = pair_observations(entries, exits)

    assert result.max_trips == 1
    assert result.optimal_matchings == 2
    assert result.confirmed == []
    assert {p.observation.observation_id for p in result.pending} == {"E1", "X1", "X2"}


def test_unique_pair_is_confirmed_while_extra_entry_is_pending():
    entries = [obs("E1", ENTRY, 0), obs("E2", ENTRY, 30)]
    exits = [obs("X1", EXIT, 4)]

    result = pair_observations(entries, exits)

    assert result.max_trips == 1
    assert result.optimal_matchings == 1
    assert [(t.entry.observation_id, t.exit.observation_id) for t in result.confirmed] == [
        ("E1", "X1")
    ]
    assert [(p.observation.observation_id, p.reason) for p in result.pending] == [
        ("E2", "no_optimal_pair")
    ]


def test_duplicate_ids_are_rejected():
    with pytest.raises(PairingError):
        pair_observations(
            [obs("E1", ENTRY, 0), obs("E1", ENTRY, 1)], [obs("X1", EXIT, 5)]
        )


def test_wrong_radar_is_rejected():
    with pytest.raises(PairingError):
        pair_observations([obs("E1", EXIT, 0)], [obs("X1", EXIT, 5)])


def test_too_many_observations_are_rejected():
    entries = [obs(f"E{i}", ENTRY, i) for i in range(MAX_OBSERVATIONS_PER_FILE + 1)]
    with pytest.raises(PairingError):
        pair_observations(entries, [obs("X1", EXIT, 5)])


def test_empty_inputs_are_fine():
    result = pair_observations([], [])
    assert result.max_trips == 0
    assert result.confirmed == []
    assert result.pending == []


def test_results_are_deterministic_and_order_independent():
    entries = [obs("E1", ENTRY, 0), obs("E2", ENTRY, 0), obs("E3", ENTRY, 3)]
    exits = [obs("X1", EXIT, 5), obs("X2", EXIT, 5), obs("X3", EXIT, 8)]

    first = pair_observations(entries, exits)
    second = pair_observations(list(reversed(entries)), list(reversed(exits)))

    assert first.max_trips == second.max_trips
    assert first.min_total_deviation_seconds == second.min_total_deviation_seconds
    assert [(t.entry.observation_id, t.exit.observation_id) for t in first.confirmed] == [
        (t.entry.observation_id, t.exit.observation_id) for t in second.confirmed
    ]


# ---------------------------------------------------------------------------
# Randomised differential test against brute force
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_matches_brute_force(seed: int):
    rng = random.Random(seed)
    n = rng.randint(0, 4)
    m = rng.randint(0, 4)
    entry_times = sorted(rng.randint(0, 12) for _ in range(n))
    exit_times = sorted(rng.randint(0, 18) for _ in range(m))
    entries = [
        obs(f"E{i + 1}", ENTRY, t) for i, t in enumerate(entry_times)
    ]
    exits = [obs(f"X{j + 1}", EXIT, t) for j, t in enumerate(exit_times)]

    result = pair_observations(entries, exits)
    expected_count, expected_cost, expected_common, expected_optima = brute_force(
        entries, exits
    )

    assert result.max_trips == expected_count
    assert result.min_total_deviation_seconds == expected_cost
    assert [
        (t.entry.observation_id, t.exit.observation_id) for t in result.confirmed
    ] == expected_common
    assert result.optimal_matchings == expected_optima
