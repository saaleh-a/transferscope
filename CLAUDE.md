# TransferScope — CLAUDE.md

## What this project is
A football transfer intelligence platform that predicts player performance
at a new club, generates scouting shortlists, and simulates transfer impact.
Based on the Transfer Portal paper (Dinsdale & Gallagher, 2022).
Built for Arsenal scouting. Any player, any club, any league including South America.

---

## Stack — locked, do not deviate

| Layer | Tool |
|---|---|
| Player stats | FotMob via `mobfot` Python package |
| Power Rankings (Europe) | ClubElo via `soccerdata` Python package |
| Power Rankings (Global fallback) | WorldFootballElo (`eloratings.net`) via direct HTTP scrape |
| Feature engineering | pandas rolling windows |
| Adjustment models | sklearn LinearRegression |
| Prediction model | TensorFlow multi-head neural network (4 model groups) |
| UI | Streamlit |
| Caching | diskcache (SQLite-backed) |
| Python | 3.11+ |

---

## Folder structure — do not restructure without asking

```
transferscope/
├── CLAUDE.md
├── requirements.txt
├── .env.example
├── app.py                              # Streamlit entry point only
├── backend/
│   ├── data/
│   │   ├── fotmob_client.py            # FotMob API wrapper via mobfot
│   │   ├── clubelo_client.py           # ClubElo wrapper via soccerdata (Europe)
│   │   ├── worldfootballelo_client.py  # WorldFootballElo scraper (global fallback)
│   │   ├── elo_router.py               # Routes club to correct Elo source, merges scores
│   │   └── cache.py                    # diskcache layer — all external calls go through here
│   ├── features/
│   │   ├── rolling_windows.py          # 1000-min player rolling averages
│   │   ├── power_rankings.py           # Dynamic league Elo + 0-100 normalization
│   │   └── adjustment_models.py        # sklearn priors for low-data players and teams
│   ├── models/
│   │   ├── transfer_portal.py          # TensorFlow multi-head NN, 4 target groups
│   │   └── shortlist_scorer.py         # Weighted similarity scoring for shortlist
│   └── utils/
│       └── league_registry.py          # League ID mappings for FotMob + Elo sources
├── frontend/
│   ├── pages/
│   │   ├── transfer_impact.py          # Fig 1 from paper: predicted perf change dashboard
│   │   ├── shortlist_generator.py      # Fig 2 from paper: replacement shortlist
│   │   └── hot_or_not.py              # Section 5: quick rumour validator
│   └── components/
│       ├── swarm_plot.py               # Player vs league/team context strip plots
│       ├── power_ranking_chart.py      # Before/after team Power Rankings timeline
│       └── metric_bar.py              # Horizontal bar: predicted % change per metric
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

**Step 1 — Get raw team Elo scores**

- European clubs: ClubElo via `soccerdata` (`sd.ClubElo().read_by_date(date)`)
  Returns rank, club, country, level, Elo, date range for all European clubs.
- Non-European clubs (South America, MLS, etc.): WorldFootballElo via HTTP scrape
  Endpoint: `http://eloratings.net/{TeamName}` or date-based snapshot
  Returns global Elo scores going back decades.
- `elo_router.py` checks which source covers the club and fetches accordingly.
  If a club exists in both sources, ClubElo takes priority (more granular for Europe).

**Step 2 — Derive league Power Rankings dynamically**

Do NOT use a static tier table. Instead:
```
league_elo = mean(team_elo for all teams in that league on transfer_date)
```
This is dynamic — league quality updates as team Elos update.
Store per-league: mean, std, and percentile bands (10th, 25th, 50th, 75th, 90th)
for each daily snapshot. These power the swarm plot league context.

**Step 3 — Normalize 0-100 globally**
```
normalized_score = (raw_elo - global_min) / (global_max - global_min) * 100
```
Run daily across ALL clubs in both sources combined.
Best team in the world on that date = 100. Worst = 0.
Cache daily snapshots with max_age = 1 day.

**Step 4 — Relative ability**
```
relative_ability = team_normalized_score - league_mean_normalized_score
```
Positive = better than league average. Negative = worse.
This is a key input feature to the adjustment models.

---

## The 13 core metrics (paper, mapped to FotMob)

| # | Paper Metric | FotMob Field |
|---|---|---|
| 1 | xG | `expected_goals` |
| 2 | xA | `expected_assists` |
| 3 | Shots | `shots` |
| 4 | Take-ons | `successful_dribbles` |
| 5 | Crosses | `successful_crosses` |
| 6 | Penalty area entries | `touches_in_opposition_box` |
| 7 | Total passes | `successful_passes` |
| 8 | Short passes | `pass_completion_pct` (proxy) |
| 9 | Long passes | `accurate_long_balls` |
| 10 | Passes in attacking third | `chances_created` |
| 11 | Defensive actions own third | `clearances` |
| 12 | Defensive actions mid third | `interceptions` |
| 13 | Defensive actions att third | `possession_won_final_3rd` |

All metrics stored and displayed as per-90. Never raw totals.

## Additional FotMob metrics (beyond paper)

| FotMob Field | Label |
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

## FotMob league coverage (confirmed in scope)

**Europe:** Premier League, Bundesliga, La Liga, Serie A, Ligue 1, Eredivisie,
Primeira Liga, Championship, Belgian Pro League, Super Lig, and more.

**South America:** Brasileirao Serie A + B, Argentine Primera División,
Colombian Primera A, Chilean Primera División, Uruguayan Primera División,
Ecuadorian Serie A.

**Other:** MLS, Saudi Pro League, J-League, and more.

If a league is on FotMob it is in scope for TransferScope.

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
Input (flattened feature vector)
  -> Dense(128, relu)
  -> Dropout(0.3)
  -> Dense(64, relu)
  -> Dropout(0.3)
  -> [Linear output head per target]
```

Input features: player per-90 metrics (current club) + team ability (current + target)
+ league ability (current + target) + team-position per-90 metrics (current + target).

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

**Team adjustment — 13 sklearn LinearRegression models, one per metric:**
```
target = naive_league_expectation (used as offset)
       + beta * team_relative_feature_in_previous_league
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

---

## Shortlist scoring

```
normalized_target = (predicted_value - mean_across_candidates) / std_across_candidates
weighted_score = normalized_target * user_weight        # weight 0.0-1.0 per metric
final_score = sum(weighted_scores) / sum(weights)       # final range 0-1
```

Available filters: age, market value, minutes played, position, league, club Power Ranking cap.

---

## Key decisions already made — do not revisit

- FotMob not SofaScore: better per-90 coverage, xGOT, npxG, touches in box
- ClubElo + WorldFootballElo not static tier table: dynamic, global, faithful to paper
- Dynamic league Elo from team mean: updates automatically, no manual maintenance
- Streamlit not FastAPI + React: speed of build, sufficient for personal tool
- diskcache not Redis: local tool, SQLite is enough
- All stats stored and displayed as per-90 — never raw totals in UI

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
