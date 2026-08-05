"""Sofascore player attribute overviews — longitudinal skill ratings.

Sofascore publishes five 0-100 attribute ratings per player per year, going
back four years:

    attacking  technical  tactical  defending  creativity

These are the only **longitudinal** signal in the project. Every other feature
is cross-sectional: different players compared at a moment in time. That
distinction killed the age-curve work in :mod:`backend.models.age_curves`,
where a 32-year-old only appears in the sample if he is still good enough to be
bought, so survivorship flattened the decline a career curve should show.

Verified behaviour
------------------
**The ratings are absolute, not scaled within position group.** This was the
open question, and it decides whether the numbers are usable as features.
Measured on a hand-picked sample of unambiguous players:

    mean defending   CB 78 -> DM 63 -> AM 34 -> W 32 -> ST 27
    mean attacking   CB 38 -> DM 64 -> AM 76 -> W 84 -> ST 84
    mean creativity  De Bruyne 97, Odegaard 88 (correctly the top two)

The gradients are monotonic and inverted between attacking and defending, which
is only possible on a shared scale. A position-relative scale would put every
player near the middle of their own group.

**The two ``position`` fields mean different things**, which is what made this
payload look untrustworthy at first:

- ``averageAttributeOverviews[].position`` is the player's own position, used
  to pick the positional-average comparison row. Matched the player's profile
  position on 7 of 7 sampled players.
- ``playerAttributeOverviews[].position`` is the position that player was
  judged to be playing **in that year**, and it moves. Saka reads M for the
  three completed seasons and D for the barely-started current one; Odegaard
  reads M for older seasons and F for recent ones.

The current-year row (``yearShift == 0``) is therefore built from a partial
season and its position label is unreliable. :func:`get_attribute_history`
excludes it by default.

Not yet used by the model. Adding these changes ``FEATURE_DIM`` and needs a
full matrix rebuild, so it is a deliberate phase rather than an incremental
edit. The groundwork is here so that phase starts from verified behaviour
rather than assumptions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.data import cache
from backend.data.sofascore_client import _get

_log = logging.getLogger(__name__)

ATTRIBUTES = ("attacking", "technical", "tactical", "defending", "creativity")

# Attribute ratings move slowly; a week is plenty.
_CACHE_TTL = 604_800


def get_attribute_history(
    player_id: int,
    include_current_year: bool = False,
) -> List[Dict[str, Any]]:
    """Return a player's attribute ratings by year, oldest first.

    Each entry has ``years_ago`` (0 = current), ``position`` (the position the
    player was judged to be playing that year) and the five attributes. A
    missing attribute stays ``None`` rather than becoming 0 — goalkeepers only
    receive ``tactical``, and zero would read as "no attacking ability" instead
    of "not measured".

    ``include_current_year`` is off by default. The ``yearShift == 0`` row
    covers a season in progress, so both its ratings and its position label are
    built from partial data.
    """
    if player_id <= 0:
        return []

    key = cache.make_key("sofascore_attributes", str(player_id))
    cached = cache.get(key, max_age=_CACHE_TTL)
    if cached is None:
        raw = _get(f"/player/{player_id}/attribute-overviews")
        cached = raw if isinstance(raw, dict) else {}
        if cached:
            cache.set(key, cached)

    rows = cached.get("playerAttributeOverviews") or []
    history: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            years_ago = int(row.get("yearShift", -1))
        except (TypeError, ValueError):
            continue
        if years_ago < 0:
            continue
        if years_ago == 0 and not include_current_year:
            continue

        entry: Dict[str, Any] = {
            "years_ago": years_ago,
            "position": row.get("position"),
            "is_partial_season": years_ago == 0,
        }
        for attr in ATTRIBUTES:
            value = row.get(attr)
            try:
                entry[attr] = float(value) if value is not None else None
            except (TypeError, ValueError):
                entry[attr] = None
        history.append(entry)

    history.sort(key=lambda e: -e["years_ago"])  # oldest first
    return history


def get_positional_average(player_id: int) -> Optional[Dict[str, Any]]:
    """Return the positional-average ratings a player is compared against.

    The ``position`` on this row is the player's own position, unlike the
    per-year rows. Returns None when unavailable.
    """
    if player_id <= 0:
        return None

    key = cache.make_key("sofascore_attributes", str(player_id))
    cached = cache.get(key, max_age=_CACHE_TTL)
    if cached is None:
        raw = _get(f"/player/{player_id}/attribute-overviews")
        cached = raw if isinstance(raw, dict) else {}
        if cached:
            cache.set(key, cached)

    rows = cached.get("averageAttributeOverviews") or []
    if not rows or not isinstance(rows[0], dict):
        return None

    row = rows[0]
    result: Dict[str, Any] = {"position": row.get("position")}
    for attr in ATTRIBUTES:
        value = row.get(attr)
        try:
            result[attr] = float(value) if value is not None else None
        except (TypeError, ValueError):
            result[attr] = None
    return result


def compute_trajectory(player_id: int) -> Dict[str, Any]:
    """Summarise how a player's attributes have moved over the available years.

    Returns ``{}`` when fewer than two completed seasons are available, so a
    caller cannot mistake a single data point for a trend.

    Each attribute gets a ``delta`` (latest minus earliest) and the raw series.
    No labels such as "Rising Star" are produced: with three or four annual
    points and no measure of year-to-year noise, a categorical verdict would
    imply confidence the data does not support. That is the same mistake the
    age-curve work exists to avoid.
    """
    history = get_attribute_history(player_id)
    if len(history) < 2:
        return {}

    trajectory: Dict[str, Any] = {
        "years_covered": len(history),
        "from_years_ago": history[0]["years_ago"],
        "to_years_ago": history[-1]["years_ago"],
        "positions_played": [h["position"] for h in history],
        "attributes": {},
    }

    for attr in ATTRIBUTES:
        series = [(h["years_ago"], h[attr]) for h in history if h[attr] is not None]
        if len(series) < 2:
            continue
        earliest = series[0][1]
        latest = series[-1][1]
        trajectory["attributes"][attr] = {
            "earliest": earliest,
            "latest": latest,
            "delta": round(latest - earliest, 1),
            "series": [v for _, v in series],
        }

    # A position change makes an attribute delta ambiguous: a midfielder moved
    # forward will gain attacking without having improved at anything.
    positions = {p for p in trajectory["positions_played"] if p}
    trajectory["position_changed"] = len(positions) > 1
    return trajectory
