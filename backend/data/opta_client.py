"""Opta Power Rankings client — scrapes team and league rankings from The Analyst.

Uses SeleniumBase (headless Chrome) to paginate through the dataviz pages:
  - Team rankings:   https://dataviz.theanalyst.com/opta-power-rankings/
  - League rankings: https://dataviz.theanalyst.com/opta-power-rankings/?leagueRankings=true

Caches scraped results for 24 hours via the project's diskcache layer.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from backend.data import cache

_log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
_TEAM_URL = "https://dataviz.theanalyst.com/opta-power-rankings/"
_LEAGUE_URL = (
    "https://dataviz.theanalyst.com/opta-power-rankings/?leagueRankings=true"
)

# Maximum pages of team rankings (~100 teams per page, ~30 pages = ~3000 teams).
_MAX_TEAM_PAGES = int(os.environ.get("OPTA_MAX_TEAM_PAGES", "30"))
# Delay between page navigations (seconds).
_PAGE_DELAY = float(os.environ.get("OPTA_PAGE_DELAY", "2.0"))
# Initial page load wait (seconds).
_LOAD_DELAY = float(os.environ.get("OPTA_LOAD_DELAY", "5.0"))
# Cache TTL — 24 hours.
_CACHE_TTL = 86400


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class OptaTeamRanking:
    """A single team from the Opta Power Rankings table."""

    rank: int
    team: str
    rating: float  # 0-100 scale
    ranking_change_7d: Optional[str]  # e.g. "+2", "−3", "NEW", or ""
    opta_id: str  # Extracted from team logo <img> src


@dataclass
class OptaLeagueRanking:
    """A single league from the Opta League Rankings table."""

    rank: int
    league: str
    rating: float  # 0-100 scale
    ranking_change_7d: Optional[str]


# ── Scraper helpers ───────────────────────────────────────────────────────────

def _parse_float(s: str) -> float:
    """Parse a string to float, stripping non-numeric chars."""
    cleaned = s.strip().replace(",", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def _parse_int(s: str) -> int:
    """Parse a string to int, stripping non-numeric chars."""
    cleaned = s.strip().replace(",", "").replace("#", "")
    try:
        return int(cleaned)
    except (ValueError, TypeError):
        return 0


def _scrape_team_rankings() -> List[OptaTeamRanking]:
    """Scrape all team rankings pages using SeleniumBase (headless Chrome).

    Returns a list of OptaTeamRanking, one per team.
    """
    try:
        from bs4 import BeautifulSoup
        from seleniumbase import Driver
    except ImportError as exc:
        _log.error(
            "seleniumbase and/or beautifulsoup4 not installed. "
            "Install with: pip install seleniumbase beautifulsoup4. Error: %s",
            exc,
        )
        return []

    _log.info("Scraping Opta team rankings from %s", _TEAM_URL)

    driver = None
    try:
        driver = Driver(uc=True, headless=True)
        driver.get(_TEAM_URL)
        time.sleep(_LOAD_DELAY)

        rankings: List[OptaTeamRanking] = []
        page_num = 1

        while page_num <= _MAX_TEAM_PAGES:
            _log.debug("Scraping team rankings page %d", page_num)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            table = soup.find("table")
            if table is None:
                _log.warning(
                    "No table found on page %d — stopping pagination", page_num
                )
                break

            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if not tds or len(tds) < 3:
                    continue

                # Extract opta_id from the team logo image
                img = tr.select_one("img")
                opta_id = ""
                if img and img.get("src"):
                    import re as _re

                    opta_id_match = _re.search(r"[?&]id=([^&]+)", img["src"])
                    opta_id = opta_id_match.group(1) if opta_id_match else ""

                # Columns: Rank | Team | Rating | 7-day change
                rank_text = tds[0].get_text(strip=True)
                team_text = tds[1].get_text(strip=True)
                rating_text = tds[2].get_text(strip=True)
                change_text = tds[3].get_text(strip=True) if len(tds) > 3 else ""

                rank_val = _parse_int(rank_text)
                rating_val = _parse_float(rating_text)

                if team_text and rank_val > 0:
                    rankings.append(
                        OptaTeamRanking(
                            rank=rank_val,
                            team=team_text,
                            rating=rating_val,
                            ranking_change_7d=change_text,
                            opta_id=opta_id,
                        )
                    )

            # Navigate to next page (click the last button on the page)
            if page_num < _MAX_TEAM_PAGES:
                try:
                    buttons = driver.find_elements("css selector", "button")
                    if buttons:
                        buttons[-1].click()
                        time.sleep(_PAGE_DELAY)
                    else:
                        _log.info(
                            "No pagination buttons found — "
                            "stopping at page %d",
                            page_num,
                        )
                        break
                except Exception as exc:
                    _log.warning(
                        "Pagination click failed at page %d: %s", page_num, exc
                    )
                    break

            page_num += 1

        _log.info("Scraped %d team rankings across %d pages", len(rankings), page_num)
        return rankings

    except Exception as exc:
        _log.exception("Opta team rankings scrape failed: %s", exc)
        return []
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


def _scrape_league_rankings() -> List[OptaLeagueRanking]:
    """Scrape league rankings (single page, no pagination needed).

    Returns a list of OptaLeagueRanking.
    """
    try:
        from bs4 import BeautifulSoup
        from seleniumbase import Driver
    except ImportError as exc:
        _log.error(
            "seleniumbase and/or beautifulsoup4 not installed. Error: %s", exc
        )
        return []

    _log.info("Scraping Opta league rankings from %s", _LEAGUE_URL)

    driver = None
    try:
        driver = Driver(uc=True, headless=True)
        driver.get(_LEAGUE_URL)
        time.sleep(_LOAD_DELAY)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table")
        if table is None:
            _log.warning("No table found on Opta league rankings page")
            return []

        rankings: List[OptaLeagueRanking] = []
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds or len(tds) < 3:
                continue

            rank_text = tds[0].get_text(strip=True)
            league_text = tds[1].get_text(strip=True)
            rating_text = tds[2].get_text(strip=True)
            change_text = tds[3].get_text(strip=True) if len(tds) > 3 else ""

            rank_val = _parse_int(rank_text)
            rating_val = _parse_float(rating_text)

            if league_text and rank_val > 0:
                rankings.append(
                    OptaLeagueRanking(
                        rank=rank_val,
                        league=league_text,
                        rating=rating_val,
                        ranking_change_7d=change_text,
                    )
                )

        _log.info("Scraped %d league rankings", len(rankings))
        return rankings

    except Exception as exc:
        _log.exception("Opta league rankings scrape failed: %s", exc)
        return []
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


# ── Public API (cached) ───────────────────────────────────────────────────────

def get_team_rankings(force_refresh: bool = False) -> List[OptaTeamRanking]:
    """Return today's Opta team rankings (cached for 24h).

    Parameters
    ----------
    force_refresh : bool
        If True, bypass cache and re-scrape.

    Returns
    -------
    list[OptaTeamRanking]
    """
    key = cache.make_key("opta_team_rankings", date.today().isoformat())
    if not force_refresh:
        cached = cache.get(key, max_age=_CACHE_TTL)
        if cached is not None:
            _log.debug("Using cached Opta team rankings (%d teams)", len(cached))
            return cached

    rankings = _scrape_team_rankings()
    if rankings:
        cache.set(key, rankings)
    return rankings


def get_league_rankings(force_refresh: bool = False) -> List[OptaLeagueRanking]:
    """Return today's Opta league rankings (cached for 24h).

    Parameters
    ----------
    force_refresh : bool
        If True, bypass cache and re-scrape.

    Returns
    -------
    list[OptaLeagueRanking]
    """
    key = cache.make_key("opta_league_rankings", date.today().isoformat())
    if not force_refresh:
        cached = cache.get(key, max_age=_CACHE_TTL)
        if cached is not None:
            _log.debug("Using cached Opta league rankings (%d leagues)", len(cached))
            return cached

    rankings = _scrape_league_rankings()
    if rankings:
        cache.set(key, rankings)
    return rankings


def get_team_rankings_dict(
    force_refresh: bool = False,
) -> Dict[str, OptaTeamRanking]:
    """Return Opta team rankings as a dict keyed by team name (case-sensitive).

    Convenience wrapper for fast lookups by team name.
    """
    return {r.team: r for r in get_team_rankings(force_refresh)}


def get_league_rankings_dict(
    force_refresh: bool = False,
) -> Dict[str, OptaLeagueRanking]:
    """Return Opta league rankings as a dict keyed by league name."""
    return {r.league: r for r in get_league_rankings(force_refresh)}
