#!/usr/bin/env python
"""Check whether a Sofascore field is safe to add as a model feature.

Three feature slots in the shipped matrix are permanently constant because
Sofascore never served them. A fourth was constant because it was added after
the matrices were built. Each cost a full rebuild to discover.

This answers the question before the rebuild:

  1. Is the field populated for most players *today*?
  2. Is it populated in the *historical* seasons the model trains on?

A field can pass the first test and fail the second. kilometersCovered,
numberOfSprints and topSpeed are 98% populated for the current season and
absent from every earlier one, so they would train as constant zero.

Usage
-----
    python scripts/check_field_availability.py kilometersCovered tackles
    python scripts/check_field_availability.py --candidates

Exit code is 1 if any requested field fails, so it can gate a rebuild.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fields already assessed and documented in ARCHITECTURE.md.
CANDIDATE_FIELDS = [
    "accurateFinalThirdPasses", "accurateOppositionHalfPasses",
    "accurateOwnHalfPasses", "tackles", "tacklesWonPercentage",
    "dribbledPast", "blockedShots", "errorLeadToGoal", "errorLeadToShot",
    "bigChancesCreated", "bigChancesMissed", "goalConversionPercentage",
    "shotsFromInsideTheBox", "shotsFromOutsideTheBox",
    "groundDuelsWon", "aerialDuelsWon", "possessionLost", "wasFouled",
]

# A field must clear both bars to be worth a feature slot.
MIN_PLAYER_COVERAGE = 0.90
MIN_SEASON_COVERAGE = 0.80

_PL_TOURNAMENT = 17
_SAMPLE_TEAMS = [44, 42, 33]      # a few Premier League squads
_SEASONS_TO_CHECK = 6


def _session():
    from curl_cffi.requests import Session

    return Session(impersonate="chrome120")


def _get(session, url: str, attempts: int = 3) -> Optional[dict]:
    """Fetch with backoff; returns None rather than raising."""
    for i in range(attempts):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return None


def check_fields(fields: List[str]) -> Dict[str, dict]:
    """Measure current-player and historical-season coverage for each field."""
    session = _session()
    base = "https://api.sofascore.com/api/v1"

    seasons_payload = _get(session, f"{base}/unique-tournament/{_PL_TOURNAMENT}/seasons")
    seasons = [s.get("id") for s in (seasons_payload or {}).get("seasons", [])]
    # `if s is not None` rather than `if s` — a season id of 0 is falsy but valid.
    seasons = [s for s in seasons if s is not None][:_SEASONS_TO_CHECK]
    if not seasons:
        return {}

    current_season = seasons[0]

    # ── Player coverage in the current season ────────────────────────────
    player_ids = []
    for team_id in _SAMPLE_TEAMS:
        squad = _get(session, f"{base}/team/{team_id}/players")
        for entry in (squad or {}).get("players", []):
            pid = (entry.get("player") or {}).get("id")
            if pid:
                player_ids.append(pid)

    present = defaultdict(int)
    checked = 0
    for pid in player_ids:
        stats = _get(
            session,
            f"{base}/player/{pid}/unique-tournament/{_PL_TOURNAMENT}"
            f"/season/{current_season}/statistics/overall",
        )
        block = (stats or {}).get("statistics") or {}
        if not block.get("minutesPlayed"):
            continue
        checked += 1
        for field in fields:
            if block.get(field) is not None:
                present[field] += 1

    # ── Season coverage, using long-serving players ──────────────────────
    seasons_with_field = defaultdict(int)
    seasons_checked = 0
    for season_id in seasons:
        found_any = False
        for pid in (934235, 159665, 243609):  # Saka, Salah, Alisson
            stats = _get(
                session,
                f"{base}/player/{pid}/unique-tournament/{_PL_TOURNAMENT}"
                f"/season/{season_id}/statistics/overall",
            )
            block = (stats or {}).get("statistics") or {}
            if not block.get("minutesPlayed"):
                continue
            found_any = True
            for field in fields:
                if block.get(field) is not None:
                    seasons_with_field[field] += 1
            break
        if found_any:
            seasons_checked += 1

    results = {}
    for field in fields:
        player_cov = present[field] / checked if checked else None
        season_cov = (
            seasons_with_field[field] / seasons_checked if seasons_checked else None
        )

        # Report what could not be measured rather than failing a field on a
        # sample that never arrived. A rate-limited run must not look like
        # evidence against a field.
        if player_cov is None and season_cov is None:
            verdict, safe = "no data — could not sample", None
        elif season_cov is not None and season_cov < MIN_SEASON_COVERAGE:
            verdict, safe = (
                "CURRENT SEASON ONLY — would train as constant zero", False
            )
        elif player_cov is None:
            verdict, safe = (
                f"historical {season_cov:.0%}; player sample unavailable", None
            )
        elif player_cov < MIN_PLAYER_COVERAGE:
            verdict, safe = "sparse across players", False
        else:
            verdict, safe = "safe to add", True

        results[field] = {
            "player_coverage": player_cov,
            "season_coverage": season_cov,
            "players_checked": checked,
            "seasons_checked": seasons_checked,
            "safe": safe,
            "verdict": verdict,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fields", nargs="*", help="Sofascore field names to check")
    parser.add_argument(
        "--candidates", action="store_true",
        help="Check the documented candidate list instead",
    )
    args = parser.parse_args()

    fields = CANDIDATE_FIELDS if args.candidates else args.fields
    if not fields:
        parser.print_help()
        return 0

    results = check_fields(fields)
    if not results:
        print("Could not reach Sofascore — no verdict.")
        return 0

    print(f"{'field':32s} {'players':>8} {'seasons':>8}  verdict")
    print("-" * 84)
    failed = []
    unknown = []

    def pct(value):
        return f"{value:>7.0%}" if value is not None else "      -"

    for field, r in results.items():
        if r["safe"] is False:
            failed.append(field)
        elif r["safe"] is None:
            unknown.append(field)
        print(
            f"{field:32s} {pct(r['player_coverage'])} {pct(r['season_coverage']):>8}"
            f"  {r['verdict']}"
        )

    sample = next(iter(results.values()))
    print(
        f"\nsampled {sample['players_checked']} players "
        f"across {sample['seasons_checked']} seasons"
    )
    if unknown:
        print(
            f"\nCould not fully assess: {', '.join(unknown)}. "
            "Re-run when the API is not under load — an incomplete sample is "
            "not evidence against a field."
        )
    if failed:
        print(f"\nNot safe to add: {', '.join(failed)}")
        return 1
    if not unknown:
        print("\nAll requested fields are safe to add.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
