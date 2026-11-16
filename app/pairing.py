"""Exact pairing engine for tunnel radar passages.

The entry and exit radars do not share a vehicle identifier, so for one
uploaded round we infer a one-to-one matching between entry observations and
exit observations.

Rules
-----
* A pair is feasible only when the passage time is 3..8 minutes inclusive.
* The objective is lexicographic: first maximise the number of pairs, then
  minimise ``sum(abs(duration - 5 minutes))`` over the matched pairs.
* Only pairs that occur in *every* optimal matching may be delivered as
  confirmed trips.  Every other observation is pending verification.

How "confirmed" is decided
--------------------------
A forward/backward dynamic program over (entry index, bitmask of exit
observations) computes

* the optimal value ``V = (max pairs, min total deviation)``,
* the number of optimal matchings,
* for every feasible pair, how many optimal matchings contain it.

A pair is confirmed exactly when that count equals the total number of
optimal matchings.  Because both counts are exact integers, the decision is
exact and does not depend on tie-breaking.  Each CSV is limited to 12 rows,
so the DP is tiny (at most 13 * 2**12 states).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

ENTRY = "entry"
EXIT = "exit"

MIN_DURATION_SECONDS = 3 * 60
MAX_DURATION_SECONDS = 8 * 60
TARGET_DURATION_SECONDS = 5 * 60

MAX_OBSERVATIONS_PER_FILE = 12

REASON_AMBIGUOUS = "ambiguous_pairing"
REASON_NO_OPTIMAL_PAIR = "no_optimal_pair"


class PairingError(ValueError):
    """Raised when the observations cannot be paired as requested."""


@dataclass(frozen=True)
class Observation:
    """A single radar observation at a whole UTC minute."""

    observation_id: str
    radar: str
    utc_minute: int  # whole minutes since the Unix epoch, UTC

    def __post_init__(self) -> None:
        if self.radar not in (ENTRY, EXIT):
            raise PairingError(f"unknown radar: {self.radar!r}")
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise PairingError("observation_id must be a non-empty string")
        if not isinstance(self.utc_minute, int) or isinstance(self.utc_minute, bool):
            raise PairingError("utc_minute must be an integer")


@dataclass(frozen=True)
class Trip:
    entry: Observation
    exit: Observation
    duration_seconds: int
    deviation_seconds: int


@dataclass(frozen=True)
class PendingObservation:
    observation: Observation
    reason: str


@dataclass
class PairingResult:
    confirmed: List[Trip]
    pending: List[PendingObservation]
    max_trips: int
    min_total_deviation_seconds: int
    optimal_matchings: int

    @property
    def pending_entries(self) -> List[PendingObservation]:
        return [p for p in self.pending if p.observation.radar == ENTRY]

    @property
    def pending_exits(self) -> List[PendingObservation]:
        return [p for p in self.pending if p.observation.radar == EXIT]


# ``_State`` is (pair_count, total_deviation, number_of_matchings).
_State = Tuple[int, int, int]


def _merge(
    current: Optional[_State], count: int, cost: int, ways: int
) -> _State:
    """Aggregate ``(count, cost, ways)`` preferring more pairs, then less cost."""

    if current is None:
        return (count, cost, ways)
    cur_count, cur_cost, cur_ways = current
    if count > cur_count or (count == cur_count and cost < cur_cost):
        return (count, cost, ways)
    if count == cur_count and cost == cur_cost:
        return (cur_count, cur_cost, cur_ways + ways)
    return current


def _check_side(observations: Sequence[Observation], radar: str) -> None:
    if len(observations) > MAX_OBSERVATIONS_PER_FILE:
        raise PairingError(
            f"too many {radar} observations: {len(observations)} "
            f"(maximum {MAX_OBSERVATIONS_PER_FILE})"
        )
    seen = set()
    for obs in observations:
        if obs.radar != radar:
            raise PairingError(
                f"observation {obs.observation_id!r} was supplied as {radar} "
                f"but is tagged {obs.radar!r}"
            )
        if obs.observation_id in seen:
            raise PairingError(
                f"duplicate {radar} observation id: {obs.observation_id!r}"
            )
        seen.add(obs.observation_id)


def pair_observations(
    entries: Sequence[Observation], exits: Sequence[Observation]
) -> PairingResult:
    """Pair one round of entry/exit observations exactly as specified."""

    _check_side(entries, ENTRY)
    _check_side(exits, EXIT)

    entries = sorted(entries, key=lambda o: (o.utc_minute, o.observation_id))
    exits = sorted(exits, key=lambda o: (o.utc_minute, o.observation_id))

    n = len(entries)
    m = len(exits)
    full_mask = (1 << m) - 1
    size = 1 << m

    # feasible[i]: list of (exit_index, deviation_seconds)
    feasible: List[List[Tuple[int, int]]] = []
    duration_map: Dict[Tuple[int, int], int] = {}
    for i, entry in enumerate(entries):
        row: List[Tuple[int, int]] = []
        for j, exit_obs in enumerate(exits):
            duration = (exit_obs.utc_minute - entry.utc_minute) * 60
            if MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
                row.append((j, abs(duration - TARGET_DURATION_SECONDS)))
                duration_map[(i, j)] = duration
        feasible.append(row)

    # Forward DP: F[i][mask] = best value for entries [0, i) using exactly
    # the exits in ``mask``.
    forward: List[List[Optional[_State]]] = [[None] * size for _ in range(n + 1)]
    forward[0][0] = (0, 0, 1)
    for i in range(n):
        current_row = forward[i]
        next_row = forward[i + 1]
        for mask in range(size):
            state = current_row[mask]
            if state is None:
                continue
            count, cost, ways = state
            # Leave entry i unpaired.
            next_row[mask] = _merge(next_row[mask], count, cost, ways)
            # Pair entry i with each feasible, still free exit.
            for j, deviation in feasible[i]:
                bit = 1 << j
                if mask & bit:
                    continue
                next_row[mask | bit] = _merge(
                    next_row[mask | bit], count + 1, cost + deviation, ways
                )

    # Backward DP: B[i][mask] = best value for entries [i, n) using a subset
    # of the exits available in ``mask``.
    backward: List[List[_State]] = [[(0, 0, 1)] * size for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        current_row = backward[i]
        next_row = backward[i + 1]
        for mask in range(size):
            state = next_row[mask]  # entry i unpaired
            for j, deviation in feasible[i]:
                bit = 1 << j
                if not (mask & bit):
                    continue
                suffix = next_row[mask ^ bit]
                state = _merge(
                    state, suffix[0] + 1, suffix[1] + deviation, suffix[2]
                )
            current_row[mask] = state

    best = backward[0][full_mask]
    best_count, best_cost, total_optimal = best

    # For every feasible pair, count the optimal matchings that contain it.
    containing: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        prefix_row = forward[i]
        suffix_row = backward[i + 1]
        for j, deviation in feasible[i]:
            bit = 1 << j
            ways = 0
            for mask in range(size):
                if mask & bit:
                    continue
                prefix = prefix_row[mask]
                if prefix is None:
                    continue
                suffix = suffix_row[full_mask ^ mask ^ bit]
                count = prefix[0] + 1 + suffix[0]
                cost = prefix[1] + deviation + suffix[1]
                if count == best_count and cost == best_cost:
                    ways += prefix[2] * suffix[2]
            containing[(i, j)] = ways

    forced = {
        (i, j)
        for (i, j), ways in containing.items()
        if total_optimal > 0 and ways == total_optimal
    }

    confirmed: List[Trip] = []
    for i, j in sorted(forced):
        duration = duration_map[(i, j)]
        confirmed.append(
            Trip(
                entry=entries[i],
                exit=exits[j],
                duration_seconds=duration,
                deviation_seconds=abs(duration - TARGET_DURATION_SECONDS),
            )
        )
    confirmed.sort(
        key=lambda t: (
            t.entry.utc_minute,
            t.entry.observation_id,
            t.exit.observation_id,
        )
    )

    # Any observation that can take part in an optimal matching, but is not
    # confirmed, is ambiguous; otherwise it is absent from every best pairing.
    entries_in_optimal = {i for (i, _j), ways in containing.items() if ways > 0}
    exits_in_optimal = {j for (_i, j), ways in containing.items() if ways > 0}
    confirmed_entry_ids = {t.entry.observation_id for t in confirmed}
    confirmed_exit_ids = {t.exit.observation_id for t in confirmed}

    pending: List[PendingObservation] = []
    for i, entry in enumerate(entries):
        if entry.observation_id in confirmed_entry_ids:
            continue
        reason = REASON_AMBIGUOUS if i in entries_in_optimal else REASON_NO_OPTIMAL_PAIR
        pending.append(PendingObservation(observation=entry, reason=reason))
    for j, exit_obs in enumerate(exits):
        if exit_obs.observation_id in confirmed_exit_ids:
            continue
        reason = (
            REASON_AMBIGUOUS if j in exits_in_optimal else REASON_NO_OPTIMAL_PAIR
        )
        pending.append(PendingObservation(observation=exit_obs, reason=reason))

    pending.sort(
        key=lambda p: (
            p.observation.utc_minute,
            p.observation.radar,
            p.observation.observation_id,
        )
    )

    return PairingResult(
        confirmed=confirmed,
        pending=pending,
        max_trips=best_count,
        min_total_deviation_seconds=best_cost,
        optimal_matchings=total_optimal,
    )
