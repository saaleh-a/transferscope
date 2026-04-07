# TransferScope — Architecture Reference

## What this project is
A football transfer intelligence platform that predicts player performance
at a new club, generates scouting shortlists, and simulates transfer impact.
Based on the Transfer Portal paper (Dinsdale & Gallagher, 2022).
Built for Arsenal scouting. Any player, any club, any league including South America.

---

## Stack — locked, do not deviate

| Layer | Tool |
|---|---|
| Player stats | Sofascore REST API via `backend/data/sofascore_client.py` |
| Power Rankings (Inference) | Opta Power Rankings via `curl_cffi` direct JSON extraction from JS bundle (`backend/data/opta_client.py`) |
| Power Rankings (Training/Historical) | ClubElo via `soccerdata` + WorldFootballElo (`eloratings.net`) via direct HTTP scrape |
| Feature engineering | pandas rolling windows |
| Adjustment models | sklearn LinearRegression + paper-aligned heuristic (`paper_heuristic_predict`) |
| Prediction model | TensorFlow multi-head neural network (4 model groups) — heuristic fallback when untrained |
| UI | Streamlit |
| Spatial data | StatsBomb open data via `statsbombpy` + WhoScored fallback via `curl_cffi` + `mplsoccer` pitch rendering |
| Coefficient calibration | football-data.co.uk match CSVs via `backend/data/footballdata_client.py` |
| Team alias augmentation | REEP register (~45K clubs) via `backend/data/reep_registry.py` |
| Caching | diskcache (SQLite-backed) |
| Python | 3.12 |

---

## Folder structure — do not restructure without asking

```
transferscope/
├── ARCHITECTURE.md
├── README.md
├── METHODOLOGY.md
├── WHITEPAPER.md
├── ONBOARDING.md
├── requirements.txt
├── .python-version                     # Pins Python 3.12 for Streamlit Cloud
├── app.py                              # Streamlit entry point only
├── backend/
│   ├── data/
│   │   ├── sofascore_client.py         # Sofascore REST API — player stats, search, transfers, match logs
│   │   ├── clubelo_client.py           # ClubElo wrapper via soccerdata (Europe)
│   │   ├── worldfootballelo_client.py  # WorldFootballElo scraper (global fallback)
│   │   ├── elo_router.py               # Routes club to correct Elo source, merges scores
│   │   ├── opta_client.py              # Opta Power Rankings — curl_cffi + JSON extraction from JS bundle
│   │   ├── reep_registry.py            # REEP register — dynamic team alias building (~45K clubs)
│   │   ├── statsbomb_client.py         # StatsBomb open-data — shots, passes, heatmaps, spatial features
│   │   ├── whoscored_client.py        # WhoScored spatial data — fallback when StatsBomb misses
│   │   ├── footballdata_client.py      # football-data.co.uk — match CSVs for coefficient calibration
│   │   └── cache.py                    # diskcache layer — all external calls go through here
│   ├── features/
│   │   ├── rolling_windows.py          # 1000-min player rolling averages
│   │   ├── power_rankings.py           # Dynamic league Elo + 0-100 normalization
│   │   └── adjustment_models.py        # sklearn priors + paper_heuristic_predict fallback
│   ├── models/
│   │   ├── transfer_portal.py          # TensorFlow multi-head NN, 4 target groups (per-group feature subsets)
│   │   ├── shortlist_scorer.py         # K-means clustering + weighted Euclidean distance scoring
│   │   ├── training_pipeline.py        # End-to-end training: transfer discovery → sklearn + TF fit
│   │   └── backtester.py              # Compares predictions against actual post-transfer per-90
│   └── utils/
│       └── league_registry.py          # League ID mappings for Sofascore + Elo sources
├── frontend/
│   ├── pages/
│   │   ├── transfer_impact.py          # Fig 1 from paper: predicted perf change dashboard
│   │   ├── shortlist_generator.py      # Fig 2 from paper: replacement shortlist
│   │   ├── hot_or_not.py              # Section 5: quick rumour validator
│   │   ├── backtest_validator.py       # Backtest predictions vs actual post-transfer stats
│   │   ├── diagnostics.py             # System diagnostics — data source status, cache info
│   │   └── about.py                    # In-app methodology & backtest results reference
│   ├── components/
│   │   ├── swarm_plot.py               # Player vs league/team context strip plots
│   │   ├── power_ranking_chart.py      # Before/after team Power Rankings timeline
│   │   ├── metric_bar.py              # Horizontal bar: predicted % change per metric
│   │   ├── pitch_viz.py               # Shot maps, pass networks, heatmaps via mplsoccer
│   │   └── player_pizza.py            # Player pizza/radar chart component
│   ├── constants.py                    # Shared metric labels for UI display
│   └── theme.py                        # "Tactical Noir" dark theme + shared UI components
├── tests/                              # 488 tests across 24 files (all mocked, no network)
├── scripts/
│   └── check_training_ready.py         # Utility to verify training readiness
└── data/
    ├── cache/                          # diskcache files — gitignored
    └── models/                         # Saved TF model weights — gitignored
```

---

## Power Rankings — how they work

### The paper's method
The paper uses a 4-level hierarchy: continent > country > league > team.
Each level has its own Elo. A team's final Power Ranking = sum of all four levels,
then scaled 0-100 daily so best team globally = 100, worst = 0.

### Our implementation (faithful recreation)

**Hybrid data source:** For inference (today's date), Opta Power Rankings are the
primary source. For training (historical dates), ClubElo + WorldFootballElo are
used since Opta has no historical API.

**Step 1 — Get raw team Elo scores**

- **Inference (today):** Opta Power Rankings via `opta_client.py`.
  Fetches the 17MB JS bundle from `https://dataviz.theanalyst.com/opta-power-rankings/index.js`
  and `league-meta.json` using `curl_cffi` (Cloudflare TLS bypass) with `requests` fallback — no Selenium/browser needed.
  Uses regex-based extraction (`_extract_all_json_parse`) to find `JSON.parse(...)` blocks positionally
  (resilient to minified variable name changes between deploys), with legacy marker-based fallback.
  Returns `OptaTeamRanking` (rank, team, rating 0-100, opta_id, domestic_league, domestic_league_id,
  country, confederation, season_avg_rating) and `OptaLeagueRanking` (rank, league,
  rating=seasonAverageRating, number_of_teams=leagueSize, country_name, top5_rating, top10_rating).
  ClubElo provides `raw_elo` for teams it covers; teams not in ClubElo get a linear
  rescale from Opta's 0-100 to ~1000-2100 range.
  League code resolution priority: Opta (domestic_league_name + country from team data) → ClubElo fallback → "UNK".
  Cached with 7-day TTL (Opta updates roughly weekly).
- **Training (historical dates):** ClubElo + WorldFootballElo, same as before:
  - European clubs: ClubElo via `soccerdata` (`sd.ClubElo().read_by_date(date)`)
    Returns rank, club, country, level, Elo, date range for all European clubs.
  - Non-European clubs (South America, MLS, etc.): WorldFootballElo via HTTP scrape
    Endpoint: `http://eloratings.net/{TeamName}` or date-based snapshot
    Returns global Elo scores going back decades.
  - `elo_router.py` checks which source covers the club and fetches accordingly.
    If a club exists in both sources, ClubElo takes priority (more granular for Europe).

**Step 2 — Derive league Power Rankings dynamically**

Do NOT use a static tier table. Instead:
- **Inference:** League means come from **official** `seasonAverageRating` in Opta's
  `league-meta.json`, and team counts come from **official** `leagueSize`.
  This avoids incorrect counts (e.g. Premier League showing 22 teams instead of 20
  when not all teams match by name).
- **Training (historical):**
  ```
  league_elo = mean(team_elo for all teams in that league on transfer_date)
  ```
  This is dynamic — league quality updates as team Elos update.
Store per-league: mean, std, and percentile bands (10th, 25th, 50th, 75th, 90th)
for each snapshot. These power the swarm plot league context.

**Step 3 — Normalize 0-100 globally**
- **Inference:** Opta ratings are already on a native 0-100 scale; no additional
  normalization is needed.
- **Training (historical):**
  ```
  normalized_score = (raw_elo - global_min) / (global_max - global_min) * 100
  ```
  Run daily across ALL clubs in both sources combined.
Best team in the world on that date = 100. Worst = 0.
Cache snapshots with max_age = 7 days for Opta data, 1 day for historical Elo data.

**Step 4 — Relative ability**
```
relative_ability = team_normalized_score - league_mean_normalized_score
```
Positive = better than league average. Negative = worse.
This is a key input feature to the adjustment models.

### Team name resolution

`get_team_ranking()` uses a 3-step lookup:
1. **Exact match** in the rankings dict
2. **Accent-normalized exact match** via `_strip_accents()` (NFKD decomposition)
3. **Fuzzy match** via `_fuzzy_find_team()` — 5-priority cascade:
   - Exact normalized match
   - `_EXTREME_ABBREVS` alias lookup (502 bidirectional entries covering 51 leagues across Europe, South America, MLS, Saudi, J-League: PSG↔Paris Saint-Germain, Orlando City↔Orlando City SC, LAFC↔Los Angeles FC, etc.)
   - Substring containment (≥6 chars, ≥45% overlap ratio)
   - Word-level matching (shared words ≥4 chars)
   - `SequenceMatcher` ratio ≥0.70 (raised from 0.65 to reject "Orlando City SC"→"Man City" false positives)

`_CLUBELO_TO_SOFASCORE` dict (531 entries covering 30+ countries) canonicalizes ClubElo abbreviated
team names (ManCity, ManUtd, etc.) to Sofascore full display names at data-load time.

**Dynamic alias augmentation via REEP:** `_build_dynamic_aliases()` pulls REEP's `teams.csv`
(~45K clubs) at runtime and cross-links provider name columns into normalized bidirectional
aliases. `_get_merged_aliases()` overlays hardcoded entries on top. Graceful degradation —
returns `{}` without caching failure.

---

## Additional data sources

### REEP register (`backend/data/reep_registry.py`)
Provides access to the REEP open football data register (~45,000 clubs, ~430,000 players).
Functions: `get_teams_df()`, `get_people_df()`, `build_clubelo_sofascore_map()`,
`clubelo_to_sofascore_name()`, `sofascore_team_aliases()`, `enrich_player()`.
Used by `power_rankings.py` for dynamic alias building and team name augmentation.
Cached 7 days. Graceful degradation — if unavailable, falls back to hardcoded mappings.

### StatsBomb open data (`backend/data/statsbomb_client.py`)
Provides spatial features via the statsbombpy package.
Functions: `get_player_shots()`, `get_player_passes()`, `get_player_heatmap_data()`,
`compute_spatial_features()`. Rendered by `frontend/components/pitch_viz.py` using mplsoccer.
Integrated into the Transfer Impact page for shot maps, pass networks, and heatmaps.

### WhoScored (`backend/data/whoscored_client.py`)
Fallback spatial data source when StatsBomb open data doesn't cover a player.
Uses `curl_cffi` for HTTP requests with Cloudflare bypass (same as sofascore_client).
Functions: `search_player()`, `get_player_season_stats()`, `get_player_match_history()`,
`get_player_heatmap_data()`, `compute_spatial_features()`.
Player ID lookup via REEP `key_whoscored` column (cross-provider bridge).
Fallback chain: StatsBomb → WhoScored → zeros (0.0).

### football-data.co.uk (`backend/data/footballdata_client.py`)
Provides match-level CSV data from football-data.co.uk for league profiling.
Used by `adjustment_models.calibrate_style_coefficients()` to refine `_LEAGUE_STYLE_COEFF`
and `_OPP_QUALITY_SENS` via cross-league CV analysis. 60/40 blend (data/defaults).

---

## Match-level data

`get_player_match_logs(player_id, tournament_id, season_id)` fetches per-match
stats via Sofascore events endpoint. Paginates up to 10 pages, excludes 0-min
matches, returns [] if fewer than 3 valid matches. Sorted ascending by match_date.
Cached 7 days. Used for rolling window computation when available.

---

## The 13 core metrics (paper, mapped to Sofascore)

| # | Paper Metric | Sofascore Field(s) |
|---|---|---|
| 1 | xG | `expected_goals` (via `expectedGoals`, `xG`) |
| 2 | xA | `expected_assists` (via `expectedAssists`, `xA`) |
| 3 | Shots | `shots` (via `shots`, `totalShots`, `shotAttempts`) |
| 4 | Take-ons | `successful_dribbles` (via `successfulDribbles`, `dribbles`) |
| 5 | Crosses | `successful_crosses` (via `accurateCrosses`, `crossesAccurate`) |
| 6 | Penalty area entries | `touches_in_opposition_box` (via 12+ Sofascore aliases; fallback: estimated from `shots * 2.5`, capped at 30% of `touches`) |
| 7 | Total passes | `successful_passes` (via `accuratePasses`, `passesAccurate`) |
| 8 | Short passes | `pass_completion_pct` (proxy — via `accuratePassesPercentage`, `passAccuracy`) |
| 9 | Long passes | `accurate_long_balls` (via `accurateLongBalls`, `longBallsAccurate`) |
| 10 | Passes in attacking third | `chances_created` (via `keyPasses`, `bigChancesCreated`, `chancesCreated`) |
| 11 | Defensive actions own third | `clearances` |
| 12 | Defensive actions mid third | `interceptions` |
| 13 | Defensive actions att third | `possession_won_final_3rd` (via `wonTackles`, `tacklesWon`, `successfulTackles`) |

All metrics stored and displayed as per-90. Never raw totals.

## Additional Sofascore metrics (beyond paper)

| Sofascore Field | Label |
|---|---|
| `xg_on_target` | xGOT |
| `non_penalty_xg` | Non-penalty xG |
| `dispossessed` | Dispossessed |
| `duels_won_pct` | Duels won % |
| `aerial_duels_won_pct` | Aerial duels won % |
| `recoveries` | Recoveries |
| `fouls_won` | Fouls won |
| `touches` | Touches |
| `goals_conceded_on_pitch` | Goals conceded while on pitch |
| `xg_against_on_pitch` | xG against while on pitch |

---

## Sofascore league coverage (confirmed in scope)

**Europe (39):** Premier League, Championship, Bundesliga, 2. Bundesliga,
La Liga, La Liga 2, Serie A, Serie B, Ligue 1, Ligue 2, Eredivisie,
Primeira Liga, Belgian Pro League, Super Lig, Scottish Premiership,
Austrian Bundesliga, Swiss Super League, Greek Super League,
Czech First League, Danish Superliga, Croatian 1. HNL, Serbian Super Liga,
Norwegian Eliteserien, Swedish Allsvenskan, Polish Ekstraklasa,
Romanian Liga I, Ukrainian/Russian Premier Leagues, Bulgarian/Hungarian/Cypriot/Finnish leagues,
Slovak Super Liga (SVK1), Slovenian PrvaLiga (SLO1), Bosnian Premier Liga (BOS1),
Israeli Premier League (ISR1), Kazakhstan Premier League (KAZ1),
Icelandic Úrvalsdeild (ISL1), League of Ireland Premier Division (IRL1),
Welsh Premier League (WAL1), Georgian Erovnuli Liga (GEO1).

**South America:** Brasileirao Serie A + B, Argentine Primera División,
Colombian Primera A, Chilean Primera División, Uruguayan Primera División,
Ecuadorian Serie A.

**Other:** MLS, Saudi Pro League, J-League.

If a league is on Sofascore it is in scope for TransferScope. 51 leagues across 34 countries registered.

---

## The 4 TensorFlow model groups (paper Table 1)

| Group | Targets | Output heads |
|---|---|---|
| 1 - Shooting | xG, Shots | 2 |
| 2 - Passing | xA, Crosses, Total Passes, Short Passes, Long Passes, Passes Att Third, Penalty Area Entries | 7 |
| 3 - Dribbling | Take-ons | 1 |
| 4 - Defending | Defensive actions own third, mid third, att third | 3 |

Architecture per group:
```
Input (group-specific feature subset)
  -> Dense(128, relu)
  -> BatchNormalization
  -> Dropout(0.3)
  -> Dense(64, relu)
  -> BatchNormalization
  -> Dropout(0.3)
  -> [Linear output head per target]
```

Per-group feature subsets (GROUP_FEATURE_SUBSETS):
- Shooting: 36 features (player metrics + additional metrics + ability + raw Elo + height + spatial shot + team-pos + interaction + relative ability + league norm + league mean ratio)
- Passing: 50 features
- Dribbling: 22 features
- Defending: 33 features

Total input feature dict: 89 keys (13 player core per-90 + 10 additional player metrics
+ 4 team/league ability + 2 raw Elo + 2 REEP metadata
+ 3 relative ability + 13 league norm + 13 league mean ratio
+ 26 team-pos per-90 + 3 interaction: ability_gap, gap², league_gap).

**Raw Elo features** (`raw_elo_current`, `raw_elo_target`) preserve absolute
cross-league strength that normalized 0-100 scores lose (e.g. Arsenal ~1900
vs Sporting CP ~1700 on the same scale).

**REEP metadata** (`player_height_cm`, `player_age`) from the REEP open-data
register (~430K players).  Height aids aerial/crossing prediction; age
captures adaptation speed.

**Spatial features** (`spatial_avg_shot_distance`,
`spatial_shots_inside_box_pct`, `spatial_progressive_pass_pct`,
`spatial_avg_carry_distance`, `spatial_avg_defensive_distance`) from
StatsBomb open data, with WhoScored fallback via REEP `key_whoscored`
bridge.  Fallback chain: StatsBomb → WhoScored → 0.0.  All pages work
regardless of player/club coverage.

Each group slices internally — external API unchanged.

Auto-loads trained weights from `data/models/` when available (`is_trained()` checks
for `.keras` files + `feature_scaler.pkl`). Falls back to `paper_heuristic_predict()`
with a warning when no trained model exists.

---

## Rolling windows

- Player features: 1000-minute rolling window
- Team and team-position features: 3000-minute rolling window
- Prior blend formula for low-data players:
  ```
  weight = min(1, sum(minutes_played) / C)   # C = 1000, adjustable constant
  feature = (1 - weight) * prior + weight * raw_rolling_avg
  ```
- RAG confidence status:
  - Red = weight < 0.3 (heavily prior-dependent, high uncertainty)
  - Amber = weight 0.3 to 0.7 (mixed)
  - Green = weight > 0.7 (data-rich, reliable)

---

## Adjustment models

**Team adjustment — 13 sklearn Ridge regression models (3 features each), one per metric:**
```
target = β₀
       + β₁ * team_relative_feature  (paper A.1 z_{i,j}: (team_per90 - league_mean) / league_mean)
       + β₂ * from_ra                (source team relative ability)
       + β₃ * to_ra                  (target team relative ability)
       + error
```

**Team-position adjustment:**
Scale team-position features by the same percentage change as the team-level adjustment.
Example: if team xG drops 40%, striker xG and CB xG both drop 40%.

**Player adjustment — 13 models per position:**
```
target = intercept
       + b1 * player_previous_per90
       + b2 * avg_position_feature_new_team
       + b3 * diff_avg_position_feature_old_vs_new_team
       + b4 * change_in_relative_ability
       + b5 * change_in_relative_ability^2
       + b6 * change_in_relative_ability^3
       + error
```

**Paper-aligned heuristic fallback (`paper_heuristic_predict`):**

When no trained TF model weights exist, this function produces predictions
using the paper's structure (Appendix A.3) with calibrated default coefficients.

For each metric, three forces compete:
1. **Style shift** — `team_influence * (target_pos_avg - source_pos_avg)`.
   How much does the target team's tactical system differ from the source?
   Per-metric `_TEAM_INFLUENCE` weights (0.15 for dribbles → 0.55 for passing).
   When real team-position data is unavailable, per-metric `_LEAGUE_STYLE_COEFF`
   values estimate style from the league quality gap so metrics produce
   **different** percentage changes (not flat).
2. **Team quality factor** — decomposed into team_gap and league_gap:
   `team_effect = sensitivity * team_gap * (1 - damping * |team_gap|)`
   Asymmetric damping: `_DAMPING_FACTOR_DOWN=0.05` (downgrades) vs
   `_DAMPING_FACTOR_UP=0.10` (upgrades). Extreme downgrades produce larger drops.
   Per-metric `_ABILITY_SENSITIVITY` (offensive positive, defensive negative).
3. **Opposition quality** — `opp_effect = _OPP_QUALITY_SENS[m] * league_gap`.
   Moving to a weaker league means facing weaker opposition, boosting per-90
   offensive output even if the team is weaker. Per-metric `_OPP_QUALITY_SENS`
   (xG=1.30, dribbles=0.12, clearances=-0.55).

Elite player quality_scale is asymmetric: halved protection for downgrades
(even the best players suffer in poor systems).

This means a player moving to a bigger team **can improve or decline**
per-metric depending on whether the target team's style fits them:
- Moving to a high-crossing team → crosses and xA may rise even if league is harder
- Moving to a possession team → passing metrics rise, dribbling stays stable
- Moving to a defensive team → clearances/interceptions may rise, attacking may fall
- Moving to a much weaker team → realistically large drops (not protected by elite status)

**Dual simulation (paper Section 4):**

The Transfer Impact page simulates the player at **both** their current
and target clubs, then compares the two model outputs. This is faithful
to the paper: "we generate performance predictions using Transfer Portal
for players at their current club too — as opposed to using their actual
observed performance measures at their current club." Both sides come from
the same model process, reducing noise sensitivity.

```
predicted_current = predict(player, current_team → current_team, ra=0)
predicted_target  = predict(player, current_team → target_team, ra=Δ)
% change = (predicted_target - predicted_current) / predicted_current
```

---

## Shortlist scoring

K-means clustering + weighted Euclidean distance to a reference player.

**Step 1 — Filter candidates** by age, minutes, position, league, Power Ranking cap.

**Step 2 — Cluster by style** using sklearn KMeans (k = √(n/2), capped 3–10).
Reference player is included in clustering to identify their cluster.
Only when n ≥ 10 candidates; otherwise falls back to direct distance.

**Step 3 — Weighted Euclidean distance:**
```
raw_weighted = predicted_per90 * user_weight_array
standardized = StandardScaler(raw_weighted)
distance = sqrt(sum((candidate - reference)²))
base_score = 1.0 - normalized_distance
```

**Step 4 — Cluster bonus:**
```
if same_cluster_as_reference:
    base_score = min(1.0, base_score * 1.15)   # 15% bonus
```

**Step 5 — Per-metric breakdown:**
```
metric_score[m] = max(0.0, 1.0 - abs(metric_diff) / 3.0)
```

Candidate dataclass has `same_cluster_as_reference` boolean field.
Falls back to z-score ranking without `reference_per90`.

Available filters: age, market value, minutes played, position, league, club Power Ranking cap.

---

## Key decisions already made — do not revisit

- Sofascore not FotMob: team search, transfer history, season selector, league-wide stats, team-position averages
- ClubElo + WorldFootballElo not static tier table: dynamic, global, faithful to paper
- Dynamic league Elo from team mean: updates automatically, no manual maintenance (training/historical mode)
- Opta Power Rankings for inference: primary source for today's date, 0-100 native scale, with ClubElo raw_elo overlay
- `curl_cffi` over Selenium for Opta: direct JSON extraction from static JS bundle is faster, lighter, and more reliable than headless browser automation
- Official league-meta.json values (seasonAverageRating, leagueSize) over computed means: prevents incorrect team counts (e.g. Premier League showing 22 teams instead of 20 when not all teams match by name)
- Streamlit not FastAPI + React: speed of build, sufficient for personal tool
- diskcache not Redis: local tool, SQLite is enough
- All stats stored and displayed as per-90 — never raw totals in UI
- Dual simulation for Transfer Impact: predict at both current and target clubs, compare model vs model (paper Section 4)
- Per-metric style differentiation: `_TEAM_INFLUENCE`, `_ABILITY_SENSITIVITY`, `_OPP_QUALITY_SENS`, `_LEAGUE_STYLE_COEFF` all keyed per-metric, not flat group multipliers
- Asymmetric prediction calibration: `_DAMPING_FACTOR_DOWN=0.05`, `_DAMPING_FACTOR_UP=0.10`; elite quality_scale halved for downgrades
- Multi-tournament stats fallback: when primary tournament returns 0 minutes, try all team tournaments
- Position-aware Hot or Not verdict: offensive metrics 1.5× for forwards, defensive 1.5× for defenders; ±3% thresholds; UNKNOWN when no data
- Position normalization to 4 categories (Forward, Midfielder, Defender, Goalkeeper) via `normalize_position()` — including Sofascore single-letter codes (F/M/D/G)
- K-means clustering for shortlist scoring: weighted Euclidean distance + 15% same-cluster bonus vs simple z-score
- Per-group feature subsets for TF model: shooting=24, passing=32, dribbling=14, defending=20 (not 55 for all groups)
- 3-step team name resolution: exact → accent-normalized → fuzzy (5-priority cascade with 502 extreme abbreviations + dynamic REEP aliases covering 51 leagues)
- `_CLUBELO_TO_SOFASCORE` mapping (531 entries): canonicalize ClubElo names at load time
- Polynomial normalization: `change_ra / 50.0` in PlayerAdjustmentModel (mapping -50..+50 to -1..+1)
- Player-system-reliance scaling: style_diff scaled by player quality vs team average — prevents over-penalizing below-average players
- Diverging butterfly metric bar chart with paper Table 1 group markers (⚡ ◈ ◎ ◆)
- Dynamic REEP alias augmentation: _build_dynamic_aliases() cross-links ~45K clubs from REEP teams.csv at runtime, graceful degradation to hardcoded mappings
- StatsBomb spatial data: shot maps, pass networks, heatmaps via statsbombpy + mplsoccer
- WhoScored fallback for spatial features: when StatsBomb returns {} for a player, fall back to WhoScored via REEP key_whoscored bridge
- REEP enrich_player() returns whoscored_id for cross-provider ID mapping
- football-data.co.uk coefficient calibration: calibrate_style_coefficients() refines per-metric style weights from cross-league match data
- Pizza/radar charts for player profiles via player_pizza.py component
- Backtest Validator page: validates predictions against actual post-transfer outcomes
- Diagnostics page: system health, data source status, cache info
- 89-feature vector: 13 core + 10 additional + 4 ability + 2 raw Elo + 2 REEP + 3 relative ability + 13 league norm + 13 league mean ratio + 26 team-pos + 3 interaction
- Raw Elo features preserve absolute cross-league strength that 0-100 normalization loses
- TeamAdjustmentModel uses 3 features per metric: team_relative_feature (paper A.1 z_{i,j}), from_ra, to_ra
- Per-group architecture overrides: dribbling 64→32 + dropout 0.4 (smaller than default 128→64) to combat overfitting with 14 features / 1 target

---

## Stop and ask before

- Deleting or overwriting anything in `backend/models/` or `data/models/`
- Changing the folder structure
- Switching any package in the stack to an alternative
- Making any external API call not routed through `backend/data/cache.py`
- Installing packages not already in `requirements.txt`
- Modifying saved model weights in `data/models/`

---

## After each phase

Output exactly:
```
✅ Phase [N] complete — [one sentence summary of what was built and tested]
```
Then stop and wait for confirmation before starting the next phase.
