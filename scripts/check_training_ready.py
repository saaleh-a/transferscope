#!/usr/bin/env python3
"""TransferScope Training Readiness Check.

Verifies all prerequisites before running the training pipeline.
Reports what is working and what is broken.

Usage:
    python scripts/check_training_ready.py
"""

from __future__ import annotations

import os
import sys

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    failures: list[str] = []

    # ── 1. Import all required modules ───────────────────────────────────
    print("=" * 60)
    print("TransferScope — Training Readiness Check")
    print("=" * 60)

    print("\n1. Checking imports...")
    modules = [
        "backend.data.sofascore_client",
        "backend.data.cache",
        "backend.features.power_rankings",
        "backend.features.adjustment_models",
        "backend.features.rolling_windows",
        "backend.models.transfer_portal",
        "backend.models.training_pipeline",
        "backend.utils.league_registry",
        "numpy",
        "sklearn",
    ]

    for mod_name in modules:
        try:
            __import__(mod_name)
            print(f"   ✅ {mod_name}")
        except ImportError as e:
            print(f"   ❌ {mod_name}: {e}")
            failures.append(f"ImportError: {mod_name} — {e}")

    # ── 2. Check Sofascore API connectivity ──────────────────────────────
    print("\n2. Checking Sofascore API (get_season_list for Premier League)...")
    try:
        from backend.data import sofascore_client

        seasons = sofascore_client.get_season_list(17)
        if seasons:
            print(f"   ✅ Sofascore reachable — {len(seasons)} seasons found")
            print(f"   Latest season: {seasons[0].get('name', '?')}")
        else:
            msg = "Sofascore returned 0 seasons for tournament 17 (Premier League)"
            print(f"   ❌ {msg}")
            failures.append(msg)
    except Exception as e:
        msg = f"Sofascore API unreachable: {e}"
        print(f"   ❌ {msg}")
        failures.append(msg)

    # ── 3. Check league player stats + minutes key ───────────────────────
    print("\n3. Checking get_league_player_stats (Premier League, limit=5)...")
    try:
        from backend.data import sofascore_client

        # Use a valid season_id from the season list
        season_id = None
        try:
            seasons = sofascore_client.get_season_list(17)
            if seasons:
                season_id = seasons[0]["id"]
        except Exception:
            pass

        players = sofascore_client.get_league_player_stats(17, season_id=season_id, limit=5)
        if players:
            first = players[0]
            print(f"   ✅ Returned {len(players)} players")
            print(f"   First player dict keys: {sorted(first.keys())}")
            print(f"   Minutes key value: minutes_played = {first.get('minutes_played')}")
        else:
            msg = "get_league_player_stats returned 0 players"
            print(f"   ❌ {msg}")
            failures.append(msg)
    except Exception as e:
        msg = f"get_league_player_stats failed: {e}"
        print(f"   ❌ {msg}")
        failures.append(msg)

    # ── 4. Check Power Rankings ──────────────────────────────────────────
    print("\n4. Checking Power Rankings (compute_daily_rankings)...")
    try:
        from backend.features import power_rankings

        team_rankings, league_snapshots = power_rankings.compute_daily_rankings()
        if team_rankings:
            print(f"   ✅ Power Rankings loaded — {len(team_rankings)} teams")
        else:
            msg = "compute_daily_rankings returned empty team rankings"
            print(f"   ⚠️  {msg} (may work without ClubElo/soccerdata)")
            failures.append(msg)
        if league_snapshots:
            print(f"   ✅ League snapshots — {len(league_snapshots)} leagues")
        else:
            print("   ⚠️  No league snapshots (predictions will use defaults)")
    except Exception as e:
        msg = f"Power Rankings failed: {e}"
        print(f"   ❌ {msg}")
        failures.append(msg)

    # ── 5. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not failures:
        print("✅ Ready to train — run:")
        print("  python backend/models/training_pipeline.py \\")
        print("    --seasons-back 3 \\")
        print("    --leagues ENG1,ESP1,GER1,ITA1,FRA1 \\")
        print("    --api-delay 1.0")
    else:
        print(f"❌ {len(failures)} issue(s) found:\n")
        for i, f in enumerate(failures, 1):
            print(f"  {i}. {f}")
        print(
            "\nFix the above issues before training. "
            "Common causes:\n"
            "  - Sofascore blocked by Cloudflare (try a VPN or increase delay)\n"
            "  - Missing Python packages (pip install -r requirements.txt)\n"
            "  - No internet access\n"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
