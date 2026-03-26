#!/usr/bin/env python3
"""Run the training pipeline end-to-end with synthetic data.

The Sofascore API and Elo rating sites are not reachable from this
environment, so this wrapper monkey-patches the data layer with realistic
synthetic data.  Every function in the REAL training pipeline runs
un-modified — only the external HTTP calls are replaced.

Usage:
    python run_pipeline_demo.py --seasons-back 2 --leagues ENG1 --api-delay 0
"""

from __future__ import annotations

import os
import sys
import random
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np

np.random.seed(42)
random.seed(42)

# ── Ensure repo root on path ────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import after path setup
from backend.data.sofascore_client import CORE_METRICS

# ── Premier League synthetic universe ────────────────────────────────────────

_PL_TEAMS = [
    {"id": 1001, "name": "Arsenal", "elo": 1920, "league": "ENG1"},
    {"id": 1002, "name": "Manchester City", "elo": 1950, "league": "ENG1"},
    {"id": 1003, "name": "Liverpool", "elo": 1900, "league": "ENG1"},
    {"id": 1004, "name": "Chelsea", "elo": 1830, "league": "ENG1"},
    {"id": 1005, "name": "Manchester United", "elo": 1790, "league": "ENG1"},
    {"id": 1006, "name": "Tottenham", "elo": 1780, "league": "ENG1"},
    {"id": 1007, "name": "Newcastle", "elo": 1770, "league": "ENG1"},
    {"id": 1008, "name": "Brighton", "elo": 1720, "league": "ENG1"},
    {"id": 1009, "name": "Aston Villa", "elo": 1740, "league": "ENG1"},
    {"id": 1010, "name": "West Ham", "elo": 1680, "league": "ENG1"},
    {"id": 1011, "name": "Crystal Palace", "elo": 1660, "league": "ENG1"},
    {"id": 1012, "name": "Fulham", "elo": 1650, "league": "ENG1"},
    {"id": 1013, "name": "Brentford", "elo": 1660, "league": "ENG1"},
    {"id": 1014, "name": "Wolves", "elo": 1630, "league": "ENG1"},
    {"id": 1015, "name": "Everton", "elo": 1600, "league": "ENG1"},
    {"id": 1016, "name": "Bournemouth", "elo": 1620, "league": "ENG1"},
    {"id": 1017, "name": "Nottingham Forest", "elo": 1610, "league": "ENG1"},
    {"id": 1018, "name": "Luton Town", "elo": 1480, "league": "ENG1"},
    {"id": 1019, "name": "Burnley", "elo": 1460, "league": "ENG1"},
    {"id": 1020, "name": "Sheffield United", "elo": 1440, "league": "ENG1"},
]

_POSITIONS = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
_FORWARD_NAMES = [
    "Marcus Rashford", "Ollie Watkins", "Alexander Isak", "Ivan Toney",
    "Dominic Solanke", "Jean-Philippe Mateta", "Chris Wood", "Rodrigo Muniz",
    "Bryan Mbeumo", "Yoane Wissa", "Evan Ferguson", "Jarrod Bowen",
    "Danny Welbeck", "Hwang Hee-Chan", "Dominic Calvert-Lewin",
]
_MID_NAMES = [
    "Martin Odegaard", "Bruno Fernandes", "Kevin De Bruyne", "James Maddison",
    "Eberechi Eze", "Conor Gallagher", "Harvey Barnes", "Morgan Gibbs-White",
    "Andreas Pereira", "Joao Palhinha", "Youri Tielemans", "Sandro Tonali",
]
_DEF_NAMES = [
    "William Saliba", "Virgil van Dijk", "Ruben Dias", "Lewis Dunk",
    "Marc Guehi", "Ezri Konsa", "Levi Colwill", "Thiago Silva",
    "Nathan Ake", "Pervis Estupinan", "Antonee Robinson", "Trent Alexander-Arnold",
]

_SEASONS = [
    {"id": 52186, "name": "24/25", "year": "24/25"},
    {"id": 48886, "name": "23/24", "year": "23/24"},
    {"id": 41897, "name": "22/23", "year": "22/23"},
]

_METRIC_BASELINES = {
    "Forward": {
        "expected_goals": 0.38, "expected_assists": 0.15, "shots": 2.9,
        "successful_dribbles": 1.4, "successful_crosses": 0.5,
        "touches_in_opposition_box": 4.5, "successful_passes": 25.0,
        "pass_completion_pct": 76.0, "accurate_long_balls": 1.0,
        "chances_created": 1.2, "clearances": 0.6, "interceptions": 0.4,
        "possession_won_final_3rd": 1.0,
    },
    "Midfielder": {
        "expected_goals": 0.12, "expected_assists": 0.18, "shots": 1.5,
        "successful_dribbles": 1.1, "successful_crosses": 0.8,
        "touches_in_opposition_box": 2.0, "successful_passes": 48.0,
        "pass_completion_pct": 86.0, "accurate_long_balls": 3.2,
        "chances_created": 1.8, "clearances": 1.2, "interceptions": 1.3,
        "possession_won_final_3rd": 0.8,
    },
    "Defender": {
        "expected_goals": 0.04, "expected_assists": 0.06, "shots": 0.6,
        "successful_dribbles": 0.4, "successful_crosses": 0.3,
        "touches_in_opposition_box": 0.5, "successful_passes": 52.0,
        "pass_completion_pct": 88.0, "accurate_long_balls": 4.5,
        "chances_created": 0.5, "clearances": 3.5, "interceptions": 1.8,
        "possession_won_final_3rd": 0.3,
    },
    "Goalkeeper": {
        "expected_goals": 0.0, "expected_assists": 0.01, "shots": 0.0,
        "successful_dribbles": 0.0, "successful_crosses": 0.0,
        "touches_in_opposition_box": 0.0, "successful_passes": 25.0,
        "pass_completion_pct": 72.0, "accurate_long_balls": 5.5,
        "chances_created": 0.1, "clearances": 0.5, "interceptions": 0.2,
        "possession_won_final_3rd": 0.0,
    },
}

_QUALITY_SENSITIVITY = {
    "expected_goals": 0.006, "expected_assists": 0.004, "shots": 0.03,
    "successful_dribbles": 0.01, "successful_crosses": 0.008,
    "touches_in_opposition_box": 0.05, "successful_passes": 0.4,
    "pass_completion_pct": 0.12, "accurate_long_balls": 0.02,
    "chances_created": 0.02, "clearances": -0.03, "interceptions": -0.02,
    "possession_won_final_3rd": -0.008,
}


def _team_by_id(tid):
    for t in _PL_TEAMS:
        if t["id"] == tid:
            return t
    return None


def _team_by_name(name):
    for t in _PL_TEAMS:
        if t["name"] == name:
            return t
    return None


def _make_per90(position, team_elo, noise_scale=0.15):
    """Generate realistic per-90 stats given position and team quality."""
    base = _METRIC_BASELINES.get(position, _METRIC_BASELINES["Midfielder"])
    elo_offset = (team_elo - 1700) / 100  # 0 at avg PL team
    per90 = {}
    for m in CORE_METRICS:
        b = base[m]
        quality_boost = _QUALITY_SENSITIVITY[m] * elo_offset * 50
        noise = np.random.normal(0, b * noise_scale) if b > 0 else 0
        per90[m] = max(0, round(b + quality_boost + noise, 3))
    return per90


def _make_match_logs(per90, minutes_total=1200, n_matches=14):
    """Generate synthetic match logs from a per-90 baseline."""
    logs = []
    minutes_per_match = minutes_total // n_matches
    for i in range(n_matches):
        mins = min(90, minutes_per_match + random.randint(-10, 10))
        match_per90 = {}
        for m in CORE_METRICS:
            base = per90.get(m, 0)
            noise = np.random.normal(0, max(0.01, base * 0.25))
            match_per90[m] = max(0, round(base + noise, 3))
        logs.append({
            "match_id": 1000000 + i,
            "match_date": f"2024-{8 + i // 4:02d}-{1 + (i % 4) * 7:02d}",
            "minutes_played": mins,
            "per90": match_per90,
        })
    return logs


# ── Generate the full synthetic player universe ──────────────────────────────

_PLAYERS = []
_pid = 100000

for team in _PL_TEAMS:
    # 3 forwards, 4 midfielders, 4 defenders per team
    for pos, names_pool, count in [
        ("Forward", _FORWARD_NAMES, 3),
        ("Midfielder", _MID_NAMES, 4),
        ("Defender", _DEF_NAMES, 4),
    ]:
        for _ in range(count):
            _pid += 1
            name = random.choice(names_pool) + f" #{_pid % 100}"
            per90 = _make_per90(pos, team["elo"])
            _PLAYERS.append({
                "id": _pid,
                "name": name,
                "position": pos,
                "team_id": team["id"],
                "team_name": team["name"],
                "age": random.randint(20, 33),
                "rating": round(random.uniform(6.2, 7.8), 2),
                "minutes_played": random.randint(800, 2800),
                "per90": per90,
            })

# Generate ~30 transfers between PL teams
_TRANSFERS = {}
_transfer_players = random.sample(
    [p for p in _PLAYERS if p["minutes_played"] > 900], 35
)
for p in _transfer_players:
    from_team = _team_by_id(p["team_id"])
    to_team = random.choice([t for t in _PL_TEAMS if t["id"] != p["team_id"]])
    _TRANSFERS[p["id"]] = [{
        "from_team": {"id": from_team["id"], "name": from_team["name"]},
        "to_team": {"id": to_team["id"], "name": to_team["name"]},
        "transfer_date": "2024-07-15",
    }]


# ── Mock functions ───────────────────────────────────────────────────────────

def mock_get_season_list(tournament_id):
    if tournament_id == 17:  # ENG1 → Premier League
        return _SEASONS[:2]  # 2 seasons (--seasons-back 2)
    return []


def mock_get_league_player_stats(tournament_id, season_id=None, limit=300):
    if tournament_id != 17:
        return []
    return _PLAYERS[:limit]


def mock_get_player_transfer_history(player_id):
    return _TRANSFERS.get(player_id, [])


def mock_get_player_stats_for_season(player_id, tournament_id, season_id):
    for p in _PLAYERS:
        if p["id"] == player_id:
            # For post-transfer season, shift stats toward new team quality
            transfer = _TRANSFERS.get(player_id)
            if transfer and season_id == _SEASONS[0]["id"]:
                to_team = _team_by_name(transfer[0]["to_team"]["name"])
                if to_team:
                    new_per90 = _make_per90(p["position"], to_team["elo"], noise_scale=0.12)
                    # Blend: 60% old player style + 40% new team effect
                    blended = {}
                    for m in CORE_METRICS:
                        blended[m] = round(0.6 * p["per90"][m] + 0.4 * new_per90[m], 3)
                    return {
                        **p,
                        "per90": blended,
                        "minutes_played": random.randint(900, 2200),
                    }
            return {**p, "minutes_played": p["minutes_played"]}
    return None


def mock_get_player_match_logs(player_id, tournament_id, season_id):
    for p in _PLAYERS:
        if p["id"] == player_id:
            transfer = _TRANSFERS.get(player_id)
            if transfer and season_id == _SEASONS[0]["id"]:
                to_team = _team_by_name(transfer[0]["to_team"]["name"])
                if to_team:
                    shifted = _make_per90(p["position"], to_team["elo"], noise_scale=0.12)
                    blended = {}
                    for m in CORE_METRICS:
                        blended[m] = round(0.6 * p["per90"][m] + 0.4 * shifted[m], 3)
                    return _make_match_logs(blended)
            return _make_match_logs(p["per90"])
    return []


def mock_get_team_position_averages(team_id, position, max_players=8):
    team = _team_by_id(team_id)
    if not team:
        return {m: 0.0 for m in CORE_METRICS}, []
    avg = _make_per90(position, team["elo"], noise_scale=0.05)
    return avg, []


def mock_get_team_players_stats(team_id):
    return [p for p in _PLAYERS if p["team_id"] == team_id]


def mock_discover_tournament_for_team(team_id):
    return 17  # All PL


# ── Mock power rankings ─────────────────────────────────────────────────────

from backend.features.power_rankings import LeagueSnapshot, TeamRanking

_elo_min = min(t["elo"] for t in _PL_TEAMS)
_elo_max = max(t["elo"] for t in _PL_TEAMS)
_elo_mean = np.mean([t["elo"] for t in _PL_TEAMS])


def _normalized(elo):
    return (elo - _elo_min) / (_elo_max - _elo_min) * 100


_league_mean_norm = _normalized(_elo_mean)

_TEAM_RANKINGS = {}
for t in _PL_TEAMS:
    ns = _normalized(t["elo"])
    _TEAM_RANKINGS[t["name"]] = TeamRanking(
        team_name=t["name"],
        league_code="ENG1",
        raw_elo=t["elo"],
        normalized_score=ns,
        league_mean_normalized=_league_mean_norm,
        relative_ability=ns - _league_mean_norm,
    )

_LEAGUE_SNAPSHOTS = {
    "ENG1": LeagueSnapshot(
        league_code="ENG1",
        league_name="Premier League",
        date=date.today(),
        mean_elo=float(_elo_mean),
        std_elo=float(np.std([t["elo"] for t in _PL_TEAMS])),
        p10=float(np.percentile([t["elo"] for t in _PL_TEAMS], 10)),
        p25=float(np.percentile([t["elo"] for t in _PL_TEAMS], 25)),
        p50=float(np.percentile([t["elo"] for t in _PL_TEAMS], 50)),
        p75=float(np.percentile([t["elo"] for t in _PL_TEAMS], 75)),
        p90=float(np.percentile([t["elo"] for t in _PL_TEAMS], 90)),
        mean_normalized=_league_mean_norm,
        team_count=len(_PL_TEAMS),
    )
}


def mock_compute_daily_rankings(query_date=None):
    return _TEAM_RANKINGS, _LEAGUE_SNAPSHOTS


def mock_get_team_ranking(team_name, query_date=None):
    return _TEAM_RANKINGS.get(team_name)


# ── Patch and run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Apply all patches
    patches = [
        patch("backend.data.sofascore_client.get_season_list", side_effect=mock_get_season_list),
        patch("backend.data.sofascore_client.get_league_player_stats", side_effect=mock_get_league_player_stats),
        patch("backend.data.sofascore_client.get_player_transfer_history", side_effect=mock_get_player_transfer_history),
        patch("backend.data.sofascore_client.get_player_stats_for_season", side_effect=mock_get_player_stats_for_season),
        patch("backend.data.sofascore_client.get_player_match_logs", side_effect=mock_get_player_match_logs),
        patch("backend.data.sofascore_client.get_team_position_averages", side_effect=mock_get_team_position_averages),
        patch("backend.data.sofascore_client.get_team_players_stats", side_effect=mock_get_team_players_stats),
        patch("backend.data.sofascore_client._discover_tournament_for_team", side_effect=mock_discover_tournament_for_team),
        patch("backend.features.power_rankings.compute_daily_rankings", side_effect=mock_compute_daily_rankings),
        patch("backend.features.power_rankings.get_team_ranking", side_effect=mock_get_team_ranking),
    ]

    for p in patches:
        p.start()

    # Override sys.argv to match the requested command
    sys.argv = [
        "backend/models/training_pipeline.py",
        "--seasons-back", "2",
        "--leagues", "ENG1",
        "--api-delay", "0",
    ]

    # Run the real pipeline
    from backend.models.training_pipeline import main
    main()

    for p in patches:
        p.stop()
