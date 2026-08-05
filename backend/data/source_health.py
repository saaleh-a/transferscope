"""Live health probes for every external data source.

The Diagnostics page used to report a source healthy if its Python module
imported. That is not a health check: ``whoscored_client`` imported perfectly
while every one of its endpoints returned 404/406, and
``worldfootballelo_client`` imported while returning ``None`` for every club,
because eloratings.net serves national teams rather than clubs. Both showed a
green tick for months, and both have since been deleted.

Each probe here performs a real call with a known-good query and reports what
came back. A probe never raises: a source that is down must not take the
Diagnostics page with it.

Two sources were removed entirely after probing proved them non-functional —
see ``REMOVED_SOURCES``. They are recorded rather than silently dropped, so
nobody re-adds them assuming they work.

Statuses
--------
``LIVE``      working, returned usable data
``DEGRADED``  reachable but returning less than expected
``DEAD``      reachable and returning nothing usable, or unreachable
``UNKNOWN``   the probe itself failed in a way we cannot attribute
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Callable, List, Optional

_log = logging.getLogger(__name__)

LIVE = "LIVE"
DEGRADED = "DEGRADED"
DEAD = "DEAD"
UNKNOWN = "UNKNOWN"


@dataclass
class SourceHealth:
    """Outcome of probing one data source."""

    name: str
    status: str
    detail: str
    used_for: str
    elapsed_s: float = 0.0

    @property
    def is_ok(self) -> bool:
        return self.status == LIVE


def _probe(
    name: str,
    used_for: str,
    fn: Callable[[], str],
    timeout_note: str = "",
) -> SourceHealth:
    """Run one probe, converting any failure into a status rather than an error."""
    start = time.time()
    try:
        detail = fn()
    except Exception as exc:  # pragma: no cover - defensive by design
        return SourceHealth(
            name=name,
            status=UNKNOWN,
            detail=f"{type(exc).__name__}: {str(exc)[:80]}",
            used_for=used_for,
            elapsed_s=time.time() - start,
        )

    elapsed = time.time() - start
    if not detail:
        return SourceHealth(name, DEAD, "returned nothing", used_for, elapsed)
    status = DEGRADED if detail.startswith("!") else LIVE
    return SourceHealth(name, status, detail.lstrip("!"), used_for, elapsed)


# ── Individual probes ────────────────────────────────────────────────────────
# Each returns a human-readable detail string, "" for dead, or a string
# prefixed with "!" to signal degraded.


def _probe_sofascore() -> str:
    from backend.data import sofascore_client

    stats = sofascore_client.get_player_stats(934235)  # Bukayo Saka
    per90 = stats.get("per90") or {}
    n = sum(1 for v in per90.values() if v is not None)
    if n == 0:
        return ""
    minutes = stats.get("minutes_played") or 0
    return f"{n} metrics, {minutes:,} mins for sample player"


def _probe_sofascore_value() -> str:
    from backend.data import sofascore_client

    profiles = sofascore_client.get_team_squad_profiles(44)  # a PL squad
    if not profiles:
        return ""
    priced = sum(1 for p in profiles.values() if p.get("market_value"))
    if priced == 0:
        return "!squad resolved but no market values"
    return f"{priced}/{len(profiles)} squad players priced"


def _probe_sofascore_heatmap() -> str:
    from backend.data import sofascore_client

    feats = sofascore_client.compute_territory_features(934235, 17, 76986)
    if not feats:
        return ""
    return f"final third {feats['territory_final_third']:.0%} for sample player"


def _probe_opta() -> str:
    from backend.data import opta_client

    teams = opta_client.get_team_rankings()
    leagues = opta_client.get_league_rankings()
    if not teams:
        return ""
    with_mean = sum(1 for t in teams if t.season_avg_rating)
    detail = f"{len(teams):,} clubs, {len(leagues)} leagues"
    if with_mean < len(teams) * 0.9:
        return f"!{detail}, only {with_mean:,} carry a league average"
    return detail


def _probe_clubelo() -> str:
    from backend.data import clubelo_client

    df = clubelo_client.get_all_by_date(date.today())
    if df is None or len(df) == 0:
        return ""
    return f"{len(df)} European clubs"


def _probe_statsbomb() -> str:
    from backend.data import statsbomb_client

    sample = ["Bukayo Saka", "Erling Haaland", "Mohamed Salah", "Declan Rice"]
    hits = sum(1 for n in sample if statsbomb_client.compute_spatial_features(n))
    if hits == 0:
        return ""
    if hits < len(sample):
        return f"!{hits}/{len(sample)} sampled players covered (historical comps only)"
    return f"{hits}/{len(sample)} sampled players covered"


def _probe_footballdata() -> str:
    from backend.data import footballdata_client

    df = footballdata_client.fetch_season("ENG1", "2425")
    if df is None or len(df) == 0:
        return ""
    return f"{len(df)} matches for a sample league season"


def _probe_reep() -> str:
    from backend.data import reep_registry

    data = reep_registry.enrich_player(934235)
    if not data:
        return ""
    return f"{len(data)} enrichment fields"


# ── Registry ─────────────────────────────────────────────────────────────────

_PROBES = [
    ("Sofascore — player stats", "Per-90 metrics, the model's inputs and labels", _probe_sofascore),
    ("Sofascore — market value", "Budget filter, Value Opportunity Score", _probe_sofascore_value),
    ("Sofascore — heatmap", "Pitch territory features", _probe_sofascore_heatmap),
    ("Opta Power Rankings", "Club and league strength, 0-100", _probe_opta),
    ("ClubElo", "Raw Elo for European clubs", _probe_clubelo),
    ("football-data.co.uk", "League style calibration", _probe_footballdata),
    ("REEP registry", "Club aliases, player height and DOB", _probe_reep),
    ("StatsBomb open data", "Shot maps, pass networks", _probe_statsbomb),
]

# Sources removed from the project after being verified non-functional. Kept as
# a record so nobody re-adds them expecting they work.
REMOVED_SOURCES = {
    "WhoScored": (
        "The /api/v1 paths the old client called do not exist — 404/406. There "
        "is no player-statistics API. Event data IS available at "
        "/Matches/{id}/Live as a JavaScript object, but only through a headless "
        "browser; soccerdata's WhoScored reader does this. Client deleted "
        "2026-08-05; see ARCHITECTURE.md before re-adding."
    ),
    "WorldFootballElo": (
        "eloratings.net rates national teams, not clubs, so no club ever "
        "resolved. Module deleted 2026-08-05."
    ),
}


def probe_all(include_slow: bool = True) -> List[SourceHealth]:
    """Probe every source. Never raises.

    ``include_slow`` skips the probes that fetch large payloads when False.
    """
    slow = {"Opta Power Rankings", "StatsBomb open data", "REEP registry"}
    results: List[SourceHealth] = []
    for name, used_for, fn in _PROBES:
        if not include_slow and name in slow:
            continue
        results.append(_probe(name, used_for, fn))
    return results


def summarise(results: List[SourceHealth]) -> str:
    """One-line summary for logs or a page caption."""
    live = sum(1 for r in results if r.status == LIVE)
    degraded = sum(1 for r in results if r.status == DEGRADED)
    dead = sum(1 for r in results if r.status == DEAD)
    return f"{live} live, {degraded} degraded, {dead} dead of {len(results)} probed"
