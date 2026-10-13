#!/usr/bin/env python
"""Collect TransferScope data without an agent or the Streamlit application."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.data import cache, sofascore_client
from backend.utils.league_registry import LEAGUES

_log = logging.getLogger(__name__)
DEFAULT_LEAGUES = ["ENG1", "ESP1", "GER1", "ITA1", "FRA1"]
SOFASCORE_URL = "https://api.sofascore.com/api/v1"
SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    """Replace files atomically, so interruption cannot destroy a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_records(records: object, required: tuple[str, ...]) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError("No usable records returned; source may be unavailable")
    if any(not isinstance(row, dict) or not all(key in row for key in required)
           for row in records):
        raise ValueError(f"Unexpected record shape; required fields: {required}")
    if "per90" in required:
        for row in records:
            if "id" in required and (type(row["id"]) is not int or row["id"] <= 0):
                raise ValueError("Invalid player id")
            if (not isinstance(row["per90"], dict)
                    or not isinstance(row["minutes_played"], (int, float))
                    or not math.isfinite(row["minutes_played"])
                    or row["minutes_played"] <= 0):
                raise ValueError("Invalid player id, minutes or per90 statistics")
            for value in row["per90"].values():
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("Per90 metrics must be finite numbers or null")


class Collector:
    def __init__(self, output: Path, overwrite: bool):
        self.output = output
        self.overwrite = overwrite
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "started_at": _now(),
            "datasets": [],
        }
        self.failures = 0

    def record(self, path: Path, status: str, **extra) -> None:
        self.manifest["datasets"].append({
            "path": str(path.relative_to(self.output)),
            "status": status,
            **extra,
        })
        _write_json(self.output / "manifest.json", self.manifest)

    def error(self, path: Path, message: str) -> None:
        self.failures += 1
        _log.error("%s: %s", path, message)
        self.record(path, "failed", error=message)

    def collect(
        self, relative: str, source: str, dataset: str, parameters: dict,
        source_url: str, fetch: Callable[[], list[dict]],
        required: tuple[str, ...], warnings: tuple[str, ...] = (),
    ) -> list[dict]:
        path = self.output / relative
        identity = {
            "schema_version": SCHEMA_VERSION,
            "source": source,
            "dataset": dataset,
            "parameters": parameters,
            "source_url": source_url,
        }
        if path.exists() and not self.overwrite:
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                if (isinstance(saved, dict)
                        and all(saved.get(k) == v for k, v in identity.items())
                        and isinstance(saved.get("exported_at"), str)):
                    records = saved.get("records")
                    _validate_records(records, required)
                    if saved.get("record_count") == len(records):
                        self.record(path, "reused", record_count=len(records),
                                    exported_at=saved["exported_at"])
                        _log.info("Reused %s (%d records)", path, len(records))
                        return records
            except (OSError, ValueError, TypeError):
                _log.warning("Invalid checkpoint %s; collecting again", path)

        try:
            records = fetch()
            _validate_records(records, required)
            coverage = {}
            if "per90" in required:
                coverage = {
                    metric: sum(row["per90"].get(metric) is not None
                                for row in records)
                    for metric in sofascore_client.ALL_METRICS
                }
            exported_at = _now()
            _write_json(path, {
                **identity,
                "exported_at": exported_at,
                "record_count": len(records),
                "metric_non_null_counts": coverage,
                "warnings": list(warnings),
                "records": records,
            })
        except Exception as exc:
            self.error(path, f"{type(exc).__name__}: {exc}")
            return []
        self.record(path, "collected", record_count=len(records),
                    exported_at=exported_at)
        _log.info("Saved %s (%d records)", path, len(records))
        return records


def _collect_sofascore(collector: Collector, args: argparse.Namespace) -> None:
    for code in dict.fromkeys(args.leagues):
        tournament_id = LEAGUES[code].sofascore_tournament_id
        try:
            seasons = sofascore_client.get_season_list(tournament_id)
            if not isinstance(seasons, list) or not seasons:
                raise ValueError("No seasons returned")
            selected = seasons[:args.seasons_back]
            if len(selected) < args.seasons_back:
                raise ValueError(
                    f"Requested {args.seasons_back} seasons; only {len(selected)} available"
                )
            if any(not isinstance(s, dict) or type(s.get("id")) is not int
                   or s["id"] < 0 for s in selected):
                raise ValueError("Unexpected season response")
        except Exception as exc:
            collector.error(collector.output / "sofascore" / code / "seasons.json",
                            f"{type(exc).__name__}: {exc}")
            continue
        for season in selected:
            sid = season["id"]
            base = f"sofascore/{code}/{sid}"
            params = {
                "league": code, "tournament_id": tournament_id,
                "season_id": sid, "season_name": season.get("name", ""),
                "limit": args.limit, "enrich_profiles": not args.no_profiles,
            }
            players = collector.collect(
                f"{base}/players.json", "sofascore", "league_players", params,
                f"{SOFASCORE_URL}/unique-tournament/{tournament_id}/season/{sid}",
                lambda: sofascore_client.get_league_player_stats(
                    tournament_id, season_id=sid, limit=args.limit,
                    enrich_profiles=not args.no_profiles,
                ),
                ("id", "name", "minutes_played", "per90"),
                (
                    "Player limit is a cap, not proof of complete league coverage.",
                    "Historical fallback uses current squads and can miss departed players.",
                    "Squad profile fields reflect current data, not historical valuations.",
                    "Null metrics mean unavailable, not zero.",
                ),
            )
            if len(players) >= args.limit:
                _log.warning("%s/%s reached the player cap (%d)", code, sid, args.limit)
            for player in players:
                pid = player["id"]
                for detail in dict.fromkeys(args.details):
                    detail_params = {"player_id": pid}
                    if detail == "transfers":
                        relative = f"sofascore/players/{pid}/transfers.json"
                        url = f"{SOFASCORE_URL}/player/{pid}/transfer-history"
                        fetch = lambda: sofascore_client.get_player_transfer_history(pid)
                        required = ("transfer_date", "from_team", "to_team")
                    else:
                        relative = f"{base}/players/{pid}/{detail}.json"
                        detail_params.update(tournament_id=tournament_id, season_id=sid)
                        url = (f"{SOFASCORE_URL}/player/{pid}/unique-tournament/"
                               f"{tournament_id}/season/{sid}")
                        if detail == "matches":
                            fetch = lambda: sofascore_client.get_player_match_logs(
                                pid, tournament_id, sid)
                            required = ("match_id", "match_date", "minutes_played", "per90")
                        else:
                            fetch = lambda: sofascore_client.get_player_heatmap(
                                pid, tournament_id, sid)
                            required = ("x", "y", "count")
                    collector.collect(
                        relative, "sofascore", detail, detail_params, url, fetch, required,
                        ("Empty results may mean no coverage or a failed request; "
                         "the client does not distinguish these cases.",),
                    )


def _collect_opta(collector: Collector) -> None:
    from backend.data import opta_client

    base = "https://dataviz.theanalyst.com/opta-power-rankings"
    for dataset, fetch, url, required in (
        ("teams", opta_client.get_team_rankings, f"{base}/index.js",
         ("rank", "team", "rating", "opta_id")),
        ("leagues", opta_client.get_league_rankings, f"{base}/league-meta.json",
         ("rank", "league", "rating")),
    ):
        collector.collect(
            f"opta/latest/{dataset}.json", "opta", dataset, {}, url,
            lambda: [asdict(row) for row in fetch()], required,
            ("Current rankings only; Opta has no historical API in this project.",),
        )


def _collect_clubelo(collector: Collector, requested_date: date) -> None:
    from backend.data import clubelo_client

    def fetch() -> list[dict]:
        frame = clubelo_client.get_all_by_date(requested_date)
        # pandas JSON conversion handles timestamps and maps NaN to JSON null.
        return json.loads(frame.reset_index().to_json(orient="records", date_format="iso"))

    collector.collect(
        f"clubelo/{requested_date.isoformat()}/teams.json", "clubelo", "teams",
        {"date": requested_date.isoformat()},
        f"http://api.clubelo.com/{requested_date.isoformat()}", fetch, ("team", "elo"),
        ("ClubElo covers European clubs, not global club strength.",),
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _delay(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.5:
        raise argparse.ArgumentTypeError("must be finite and at least 0.5 seconds")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=["sofascore", "opta", "clubelo"],
                        default=["sofascore", "opta", "clubelo"])
    parser.add_argument("--leagues", nargs="+",
                        choices=[k for k, v in LEAGUES.items()
                                 if v.sofascore_tournament_id is not None],
                        default=DEFAULT_LEAGUES)
    parser.add_argument("--seasons-back", type=_positive_int, default=1,
                        help="Newest N seasons per league (default: 1)")
    parser.add_argument("--limit", type=_positive_int, default=1000,
                        help="Player cap per league-season (default: 1000)")
    parser.add_argument("--details", nargs="+", choices=["transfers", "matches", "heatmaps"],
                        default=[], help="Optional extra requests per player")
    parser.add_argument("--no-profiles", action="store_true",
                        help="Skip market-value/contract enrichment")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(),
                        help="ClubElo snapshot date, YYYY-MM-DD (default: today)")
    parser.add_argument("--api-delay", type=_delay, default=1.5,
                        help="Sofascore base delay; adaptive backoff still applies")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "scraped")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate exports; existing client cache TTLs still apply")
    args = parser.parse_args(argv)
    if args.details and "sofascore" not in args.sources:
        parser.error("--details requires --sources sofascore")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    collector = Collector(args.output.resolve(), args.overwrite)
    try:
        sofascore_client.set_inter_request_delay(args.api_delay)
        for source in dict.fromkeys(args.sources):
            try:
                if source == "sofascore":
                    if not sofascore_client.using_curl_impersonation():
                        raise RuntimeError("Sofascore requires the existing curl-cffi dependency")
                    _collect_sofascore(collector, args)
                elif source == "opta":
                    _collect_opta(collector)
                else:
                    _collect_clubelo(collector, args.date)
            except Exception as exc:
                collector.error(collector.output / source,
                                f"{type(exc).__name__}: {exc}")
        collector.manifest["finished_at"] = _now()
        _write_json(collector.output / "manifest.json", collector.manifest)
        _log.info("Finished with %d failed/unavailable datasets. Manifest: %s",
                  collector.failures, collector.output / "manifest.json")
        return 1 if collector.failures else 0
    except KeyboardInterrupt:
        _log.warning("Interrupted; rerun the same command to resume saved exports")
        return 130
    finally:
        cache.close()


if __name__ == "__main__":
    sys.exit(main())
