"""WorldFootballElo scraper for non-European clubs.

Scrapes eloratings.net for global Elo scores (South America, MLS, etc.).
All responses cached with max_age = 1 day.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional

import requests

from backend.data import cache

_ONE_DAY = 86400
_BASE_URL = "https://www.eloratings.net"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_team_elo(team_name: str, query_date: Optional[date] = None) -> Optional[float]:
    """Fetch Elo rating for a club from eloratings.net.

    Parameters
    ----------
    team_name : str
        Club name as it appears on eloratings.net (URL-friendly slug).
    query_date : date, optional
        Ignored for now — site returns current rating.  Kept for API parity
        with clubelo_client.

    Returns
    -------
    float or None
        The team's current Elo rating, or None if not found.
    """
    slug = _team_slug(team_name)
    key = cache.make_key("worldelo_team", slug)
    cached = cache.get(key, max_age=_ONE_DAY)
    if cached is not None:
        return cached

    elo = _scrape_team_elo(slug)
    if elo is not None:
        cache.set(key, elo)
    return elo


def get_league_teams(league_slug: str) -> List[Dict[str, Any]]:
    """Scrape all teams and their Elo scores for a league page.

    Parameters
    ----------
    league_slug : str
        League page slug on eloratings.net, e.g. ``"Brazil"``, ``"Argentina"``.

    Returns
    -------
    list[dict] with keys ``name``, ``elo``, ``slug``.
    """
    key = cache.make_key("worldelo_league", league_slug)
    cached = cache.get(key, max_age=_ONE_DAY)
    if cached is not None:
        return cached

    teams = _scrape_league_page(league_slug)
    if teams:
        cache.set(key, teams)
    return teams


def is_covered(team_name: str) -> bool:
    """Check whether we can get an Elo for this team from WorldFootballElo."""
    elo = get_team_elo(team_name)
    return elo is not None


# ── Internal helpers ─────────────────────────────────────────────────────────


def _team_slug(name: str) -> str:
    """Convert a human-readable team name to a URL slug."""
    slug = name.strip().replace(" ", "_")
    # Remove special chars except underscores and hyphens
    slug = re.sub(r"[^\w\-]", "", slug)
    return slug


def _scrape_team_elo(slug: str) -> Optional[float]:
    """GET the team page and parse out the current Elo."""
    url = f"{_BASE_URL}/{slug}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    return _parse_elo_from_html(resp.text)


def _parse_elo_from_html(html: str) -> Optional[float]:
    """Extract the Elo number from an eloratings.net team page HTML."""
    # The site typically shows the rating in a pattern like "Rating: 1834"
    # or within a specific element.  We try several patterns.
    patterns = [
        r"Rating[:\s]+(\d{3,4}(?:\.\d+)?)",
        r"<td[^>]*>(\d{4}(?:\.\d+)?)</td>",
        r'"elo"[:\s]+(\d{3,4}(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _scrape_league_page(league_slug: str) -> List[Dict[str, Any]]:
    """Scrape a league page for all team names and Elo values."""
    url = f"{_BASE_URL}/{league_slug}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    teams: list[dict] = []
    # Parse team rows — eloratings.net uses simple HTML tables
    # Pattern: link to team page + Elo value in adjacent cell
    row_pattern = re.compile(
        r'<a\s+href="/([^"]+)"[^>]*>([^<]+)</a>\s*</td>\s*<td[^>]*>(\d{3,4}(?:\.\d+)?)</td>',
        re.IGNORECASE,
    )
    for match in row_pattern.finditer(resp.text):
        slug, name, elo_str = match.group(1), match.group(2), match.group(3)
        try:
            teams.append({"slug": slug, "name": name.strip(), "elo": float(elo_str)})
        except ValueError:
            continue

    return teams
