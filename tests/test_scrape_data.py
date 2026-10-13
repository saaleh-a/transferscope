"""Offline coverage for the standalone data collector."""

import json
from unittest.mock import Mock

import pandas as pd
import pytest

from backend.data import clubelo_client, opta_client, sofascore_client
from scripts import scrape_data as scraper


@pytest.fixture
def player():
    return {
        "id": 123, "name": "Example", "team": "Example FC", "team_id": 42,
        "minutes_played": 900, "per90": {"shots": 2.0, "expected_goals": None},
        "market_value": None,
    }


@pytest.fixture
def clients(monkeypatch, player):
    seasons = Mock(return_value=[{"id": 10, "name": "2025/2026"},
                                 {"id": 9, "name": "2024/2025"}])
    players = Mock(return_value=[player])
    monkeypatch.setattr(sofascore_client, "get_season_list", seasons)
    monkeypatch.setattr(sofascore_client, "get_league_player_stats", players)
    monkeypatch.setattr(sofascore_client, "using_curl_impersonation", lambda: True)
    return seasons, players


def command(tmp_path, *extra):
    return ["--sources", "sofascore", "--leagues", "ENG1",
            "--output", str(tmp_path), *extra]


def test_exports_schema_nulls_and_coverage(tmp_path, clients):
    assert scraper.main(command(tmp_path)) == 0
    saved = json.loads((tmp_path / "sofascore/ENG1/10/players.json").read_text())
    assert saved["source"] == "sofascore"
    assert saved["parameters"]["tournament_id"] == 17
    assert saved["record_count"] == 1
    assert saved["records"][0]["per90"]["expected_goals"] is None
    assert saved["records"][0]["market_value"] is None
    assert saved["metric_non_null_counts"]["shots"] == 1
    assert saved["metric_non_null_counts"]["expected_goals"] == 0
    clients[1].assert_called_once_with(
        17, season_id=10, limit=1000, enrich_profiles=True,
    )


def test_resume_overwrite_and_parameter_changes(tmp_path, clients):
    assert scraper.main(command(tmp_path)) == 0
    assert scraper.main(command(tmp_path)) == 0
    assert clients[1].call_count == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["datasets"][0]["status"] == "reused"
    assert scraper.main(command(tmp_path, "--overwrite")) == 0
    assert clients[1].call_count == 2
    assert scraper.main(command(tmp_path, "--limit", "50", "--no-profiles")) == 0
    clients[1].assert_called_with(17, season_id=10, limit=50, enrich_profiles=False)


@pytest.mark.parametrize("broken", ["{", '{"records": []}', "null"])
def test_corrupt_checkpoints_are_retried(tmp_path, clients, broken):
    path = tmp_path / "sofascore/ENG1/10/players.json"
    path.parent.mkdir(parents=True)
    path.write_text(broken)
    assert scraper.main(command(tmp_path)) == 0
    assert clients[1].call_count == 1


def test_empty_results_fail_and_can_be_retried(tmp_path, clients):
    clients[1].side_effect = [[], [{"id": 123, "name": "Example",
                                  "minutes_played": 90, "per90": {}}]]
    assert scraper.main(command(tmp_path)) == 1
    assert not (tmp_path / "sofascore/ENG1/10/players.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["datasets"][0]["status"] == "failed"
    assert scraper.main(command(tmp_path)) == 0


def test_failure_does_not_stop_other_leagues(tmp_path, clients):
    clients[0].side_effect = [RuntimeError("unavailable"), [{"id": 20}]]
    assert scraper.main(command(tmp_path, "--leagues", "ENG1", "ESP1")) == 1
    assert (tmp_path / "sofascore/ESP1/20/players.json").exists()


def test_multiple_seasons_and_duplicate_leagues(tmp_path, clients):
    assert scraper.main(command(
        tmp_path, "--seasons-back", "2", "--leagues", "ENG1", "ENG1",
    )) == 0
    assert clients[0].call_count == 1
    assert clients[1].call_count == 2
    assert (tmp_path / "sofascore/ENG1/9/players.json").exists()


def test_insufficient_or_malformed_seasons_fail(tmp_path, clients):
    assert scraper.main(command(tmp_path, "--seasons-back", "3")) == 1
    clients[0].return_value = [{"id": "../escape"}]
    assert scraper.main(command(tmp_path)) == 1
    clients[1].assert_not_called()


def test_optional_details_are_checkpointed(tmp_path, clients, monkeypatch):
    transfers = Mock(return_value=[{
        "transfer_date": "2025-07-01", "from_team": {"id": 1}, "to_team": {"id": 2},
    }])
    matches = Mock(return_value=[{
        "match_id": 5, "match_date": "2025-08-01",
        "minutes_played": 90, "per90": {"shots": 0},
    }])
    heatmaps = Mock(return_value=[{"x": 70, "y": 20, "count": 3}])
    monkeypatch.setattr(sofascore_client, "get_player_transfer_history", transfers)
    monkeypatch.setattr(sofascore_client, "get_player_match_logs", matches)
    monkeypatch.setattr(sofascore_client, "get_player_heatmap", heatmaps)
    args = command(tmp_path, "--seasons-back", "2", "--details",
                   "transfers", "matches", "heatmaps")
    assert scraper.main(args) == 0
    transfers.assert_called_once_with(123)
    assert matches.call_count == heatmaps.call_count == 2
    assert scraper.main(args) == 0
    assert matches.call_count == heatmaps.call_count == 2


def test_unavailable_details_are_not_saved(tmp_path, clients, monkeypatch):
    monkeypatch.setattr(sofascore_client, "get_player_heatmap", lambda *args: [])
    assert scraper.main(command(tmp_path, "--details", "heatmaps")) == 1
    assert (tmp_path / "sofascore/ENG1/10/players.json").exists()
    assert not (tmp_path / "sofascore/ENG1/10/players/123/heatmaps.json").exists()


def test_opta_and_clubelo_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(opta_client, "get_team_rankings", lambda: [
        opta_client.OptaTeamRanking(1, "Arsenal", 99.0, None, "opta-1"),
    ])
    monkeypatch.setattr(opta_client, "get_league_rankings", lambda: [
        opta_client.OptaLeagueRanking(1, "Premier League", 90.0, None),
    ])
    frame = pd.DataFrame({"elo": [2000.0], "rank": [float("nan")],
                          "from": [pd.Timestamp("2025-01-01")]},
                         index=pd.Index(["Arsenal"], name="team"))
    fetch = Mock(return_value=frame)
    monkeypatch.setattr(clubelo_client, "get_all_by_date", fetch)
    assert scraper.main(["--sources", "opta", "clubelo", "--date", "2025-01-01",
                         "--output", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "clubelo/2025-01-01/teams.json").read_text())
    assert saved["records"][0]["team"] == "Arsenal"
    assert saved["records"][0]["rank"] is None
    assert (tmp_path / "opta/latest/leagues.json").exists()


@pytest.mark.parametrize("record", [
    {"id": "../escape", "name": "Bad", "minutes_played": 90, "per90": {}},
    {"id": 1, "name": "Bad", "minutes_played": 0, "per90": {}},
    {"id": 1, "name": "Bad", "minutes_played": 90, "per90": []},
    {"id": 1, "name": "Bad", "minutes_played": 90, "per90": {"shots": float("nan")}},
])
def test_invalid_player_records_are_not_exported(tmp_path, clients, record):
    clients[1].return_value = [record]
    assert scraper.main(command(tmp_path)) == 1
    assert not (tmp_path / "sofascore/ENG1/10/players.json").exists()


@pytest.mark.parametrize("extra", [
    ["--limit", "0"], ["--seasons-back", "-1"], ["--api-delay", "nan"],
    ["--api-delay", "0"], ["--leagues", "../escape"], ["--date", "not-a-date"],
    ["--sources", "opta", "--details", "transfers"],
])
def test_bad_arguments_are_rejected(tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        scraper.main(command(tmp_path, *extra))
    assert exc.value.code == 2


def test_atomic_write_keeps_previous_export(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text('{"previous": true}')
    with pytest.raises(ValueError):
        scraper._write_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"previous": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_interrupt_leaves_completed_checkpoints(tmp_path, clients):
    clients[1].side_effect = [
        [{"id": 1, "name": "Example", "minutes_played": 90, "per90": {}}],
        KeyboardInterrupt(),
    ]
    assert scraper.main(command(tmp_path, "--seasons-back", "2")) == 130
    assert (tmp_path / "sofascore/ENG1/10/players.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["datasets"][0]["status"] == "collected"
