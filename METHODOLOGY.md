# TransferScope — Methodology

**A detailed technical description of every step in the TransferScope prediction pipeline, from raw data collection to final output, with plain-English explanations throughout.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Acquisition](#2-data-acquisition)
3. [Per-90 Conversion and Metric Mapping](#3-per-90-conversion-and-metric-mapping)
4. [Rolling Windows and Recent Form](#4-rolling-windows-and-recent-form)
5. [Prior Blending and Confidence Indicators](#5-prior-blending-and-confidence-indicators)
6. [Power Rankings](#6-power-rankings)
7. [Adjustment Models (sklearn)](#7-adjustment-models-sklearn)
8. [Transfer Portal Neural Network (TensorFlow)](#8-transfer-portal-neural-network-tensorflow)
9. [Shortlist Scoring](#9-shortlist-scoring)
10. [End-to-End Prediction Flow](#10-end-to-end-prediction-flow)
11. [Testing and Validation](#11-testing-and-validation)

---

## 1. Overview

TransferScope predicts how a football player's per-90 statistics will change when they transfer from one club to another. The prediction pipeline has five stages:

```
Raw Data → Per-90 Features → Power Rankings → Adjustment Models → Neural Network → Prediction
```

> **In plain English:** We take a player's stats, figure out how strong their current and target teams are, adjust the stats to account for the change in difficulty, then run it all through a neural network to get the final prediction. Each stage refines the estimate.

The system predicts 13 core metrics simultaneously. These cover the full spectrum of outfield play: shooting (xG, shots), passing (xA, crosses, passes, long balls, chances created), possession (dribbling, penalty area entries), and defending (clearances, interceptions, pressing). All predictions are expressed as **per-90 minute** values.

---

## 2. Data Acquisition

### 2.1 Player Statistics — Sofascore REST API

**Technical:** The `backend/data/sofascore_client.py` module communicates with the Sofascore public API at `https://api.sofascore.com/api/v1`. Key endpoints:

| Endpoint | Function | Cache TTL |
|---|---|---|
| `/search/players?q={name}` | `search_player()` — find players by name | 7 days |
| `/search/teams?q={name}` | `search_team()` — find clubs by name | 7 days |
| `/player/{id}` | `get_player_stats()` — profile + season aggregate stats + age | 1 day |
| `/player/{id}/unique-tournament/{tid}/season/{sid}/statistics/overall` | `get_player_stats_for_season()` — stats for a specific season | 1 day |
| `/player/{id}/transfer-history` | `get_player_transfer_history()` — career transfer records | 7 days |
| `/unique-tournament/{tid}/season/{sid}/statistics` | `get_league_player_stats()` — bulk league-wide player stats | 1 day |
| `/unique-tournament/{tid}/seasons` | `get_season_list()` — available seasons for a tournament | 1 day |
| `/team/{id}/players` | `get_team_players_stats()` — squad roster | 1 day |
| `/team/{id}/unique-tournaments` | `_try_all_tournaments_for_player()` — multi-tournament fallback | 7 days |
| `/player/{id}/unique-tournament/{tid}/season/{sid}/events/last/{page}` | `get_player_match_logs()` — per-match stats | 7 days |

Sofascore returns **raw totals** (e.g. 15 goals in 2,000 minutes), not per-90 values. The client converts these in `_parse_stats()`.

**Multi-tournament fallback:** `get_player_stats()` first tries the player's primary domestic league. If that returns 0 minutes (common for youth players or mid-season signings), it iterates through **all tournaments** the player's team participates in (cups, European competitions, secondary divisions) and uses the one with the most minutes. This prevents false "no data" results.

**Match logs:** `get_player_match_logs()` fetches per-match statistics for a player in a specific tournament/season. Paginates up to 10 pages, excludes matches with 0 minutes, returns an empty list if fewer than 3 valid matches. Results sorted ascending by match date for rolling window computation.

**Age extraction:** `get_player_stats()` computes the player's age from the Sofascore profile `dateOfBirthTimestamp` field. This is used by the shortlist generator's age filter to allow age-based candidate filtering. The `_make_empty_result()` template includes `age: None` as a default.

> **In plain English:** Sofascore is where we get the player numbers. When you search for "Bukayo Saka," we hit their search API, get his ID, then fetch his full stats for the season. We also pull team rosters (to populate league context), transfer histories (to train our models), and league-wide data (to build shortlists). Everything gets saved locally so we don't re-download it every time.

### 2.2 European Elo Ratings — ClubElo

**Technical:** `backend/data/clubelo_client.py` uses the `soccerdata` Python package to call `sd.ClubElo().read_by_date(date)`. Returns a DataFrame with columns: rank, club, country, league, Elo, date range. Covers approximately 600 clubs across the top European leagues.

> **In plain English:** ClubElo tracks how strong every European football club is using the Elo system (the same rating system used in chess). Manchester City might be rated 2050, while a mid-table Championship side might be 1400. These numbers update after every match based on results — win against a strong team = big boost, lose to a weak team = big drop.

### 2.3 Global Elo Ratings — WorldFootballElo

**Technical:** `backend/data/worldfootballelo_client.py` scrapes `http://eloratings.net/{country}` pages via HTTP. Parses Elo ratings from the HTML response for clubs in South America, MLS, Asia, etc. — regions not covered by ClubElo.

> **In plain English:** For clubs outside Europe (Brazilian, Argentine, MLS teams, etc.), we use a different source that covers the whole world. It works the same way — every team has a strength score based on their results.

### 2.4 Elo Router

**Technical:** `backend/data/elo_router.py` determines which source to query for a given club. Logic:
1. Check if the club is in ClubElo (European). If yes, use ClubElo (higher granularity).
2. If not, check WorldFootballElo.
3. If neither covers the club, return `None`.

Also provides `normalize_elo(raw, global_min, global_max)` which converts any raw Elo to the 0–100 scale.

> **In plain English:** When we need to know how strong a club is, the router figures out which database to ask. For Arsenal, it asks ClubElo. For Flamengo, it asks WorldFootballElo. It's like a receptionist directing your call to the right department.

### 2.5 Caching

**Technical:** `backend/data/cache.py` wraps `diskcache` (SQLite-backed). All external API calls go through `cache.get(key, max_age)` / `cache.set(key, value)`. Namespaced by data source (e.g. `sofascore_search`, `clubelo`, `worldelo`). TTLs range from 1 day (stats) to 7 days (search results).

> **In plain English:** Every time we download data, we save a copy on your computer. Next time you ask for the same thing, we just read the saved copy instead of downloading it again. This makes the app much faster and avoids hammering external servers.

---

## 3. Per-90 Conversion and Metric Mapping

### 3.1 The Problem

Sofascore returns raw totals: "this player had 15 goals in 2,000 minutes." But comparing a player with 2,000 minutes to one with 500 minutes using totals is misleading. We need a rate metric.

### 3.2 The Conversion

```python
nineties = minutes_played / 90
per_90 = raw_total / nineties
```

Example: 15 goals in 2,000 minutes → 2,000 / 90 = 22.2 "nineties" → 15 / 22.2 = **0.675 goals per 90**.

**Exception:** Percentage metrics (pass completion %, duels won %, aerial duels won %) are stored as-is — they're already rates.

> **In plain English:** "Per-90" is the football analytics standard. It answers "how much does this player produce in a typical full match?" A player with 10 goals in 900 minutes (0.5 per 90 in 10 games) is more prolific than one with 12 goals in 2,700 minutes (0.4 per 90 in 30 games), even though the second has more total goals.

### 3.3 Sofascore-to-Canonical Mapping

Sofascore uses its own field names (e.g. `expectedGoals`, `accurateCrosses`). The `_SOFASCORE_KEY_MAP` dictionary in `sofascore_client.py` maps these to TransferScope's canonical names (e.g. `expected_goals`, `successful_crosses`). Multiple aliases are maintained to handle Sofascore API variations across seasons.

The 13 core metrics and their Sofascore sources:

| Canonical Name | Sofascore Key(s) | Type |
|---|---|---|
| `expected_goals` | `expectedGoals`, `xG` | Per-90 |
| `expected_assists` | `expectedAssists`, `xA` | Per-90 |
| `shots` | `shots`, `totalShots` | Per-90 |
| `successful_dribbles` | `successfulDribbles`, `dribbles` | Per-90 |
| `successful_crosses` | `accurateCrosses` | Per-90 |
| `touches_in_opposition_box` | `penaltyAreaTouches`, `touchInBox`, + 10 more aliases | Per-90 (fallback: estimated from shots×2.5, capped at 30% of touches) |
| `successful_passes` | `accuratePasses` | Per-90 |
| `pass_completion_pct` | `accuratePassesPercentage` | Percentage (as-is) |
| `accurate_long_balls` | `accurateLongBalls` | Per-90 |
| `chances_created` | `keyPasses`, `bigChancesCreated` | Per-90 |
| `clearances` | `clearances` | Per-90 |
| `interceptions` | `interceptions` | Per-90 |
| `possession_won_final_3rd` | `wonTackles`, `tacklesWon` | Per-90 |

---

## 4. Rolling Windows and Recent Form

### 4.1 Player Window (1,000 minutes)

**Technical:** `backend/features/rolling_windows.py` implements `player_rolling_average()`. It iterates through match logs (most recent first), accumulating minute-weighted statistics until 1,000 minutes are reached. The result is a per-90 rolling average.

```python
# For each match in the window:
totals[metric] += value * minutes_in_match
counts[metric] += minutes_in_match

# Final per-90:
result[metric] = totals[metric] / counts[metric]
```

> **In plain English:** Instead of looking at an entire season, we focus on roughly the last 11 matches (1,000 minutes ≈ 11 × 90). This captures recent form. If a player was injured for 3 months but came back strong in the last 6 weeks, the rolling window shows the strong recent performance, not the empty months.

### 4.2 Team Window (3,000 minutes)

Same algorithm but with a 3,000-minute window (~33 matches). Applied to team-level metrics and team-position metrics (e.g. "what do Arsenal's wingers typically produce per-90?").

> **In plain English:** Teams change style more slowly than individual players change form, so we use a bigger window. This tells us: "Over the last ~33 games, what has this team's average striker/winger/centre-back produced per 90 minutes?"

### 4.3 Team-Position Features

`team_position_rolling_average()` computes per-90 averages for a specific position within a team. This is a key input to the neural network — it captures how a team's system uses a particular position.

> **In plain English:** Arsenal's right wingers produce different numbers than Burnley's right wingers — not because of individual talent alone, but because Arsenal's system creates more chances on the right side. This feature captures that system effect.

---

## 5. Prior Blending and Confidence Indicators

### 5.1 The Problem

Some players have very little data — a youth player with 200 minutes, or a new signing mid-season. Their rolling averages are unreliable because the sample is too small.

### 5.2 The Solution — Bayesian-Style Blending

**Technical:** `blend_features()` in `rolling_windows.py`:

```python
weight = min(1, minutes_played / C)   # C = 1000 (adjustable)
blended = (1 - weight) * prior + weight * raw_rolling_average
```

Where `prior` is the league/position average per-90 (a reasonable default for a player we know nothing about).

> **In plain English:** If a player has only played 300 minutes, we don't fully trust their stats. Instead, we mix their actual numbers with the average for their position in their league — heavily weighted toward the average. As they play more, we gradually trust their actual numbers more. By 1,000 minutes, we're using 100% real data.
>
> Think of it like a restaurant review: one 5-star review doesn't mean it's great. But if 50 people all give it 5 stars, you can trust it. The "weight" is like the number of reviews.

### 5.3 RAG Confidence

The blend weight maps directly to a traffic light:

| Weight | Minutes | Confidence | Meaning |
|---|---|---|---|
| < 0.3 | < 300 | 🔴 Red | Heavily prior-dependent, high uncertainty |
| 0.3 – 0.7 | 300 – 700 | 🟡 Amber | Mixed — real data + assumptions |
| > 0.7 | > 700 | 🟢 Green | Data-rich, prediction is reliable |

> **In plain English:** The traffic light is a gut-check for the scout: "How much should I trust this prediction?" Red means "take this with a massive pinch of salt." Green means "this is based on solid data."

---

## 6. Power Rankings

### 6.1 Why Not a Static Tier Table?

Many scouting tools use fixed tiers: "Premier League = Tier 1, Eredivisie = Tier 3." This is imprecise, outdated the moment it's written, and doesn't capture within-league variation (Man City vs. Luton Town are both "Tier 1").

TransferScope computes Power Rankings **dynamically** from actual Elo data.

> **In plain English:** Instead of someone manually deciding "La Liga is the second best league," we calculate it from results. If La Liga has a weak year, its ranking drops automatically. If the Brasileirão strengthens, it rises. No human maintenance needed.

### 6.2 Step-by-Step Calculation

Implemented in `backend/features/power_rankings.py` → `compute_daily_rankings()`:

**Step 1 — Collect all club Elo ratings.**
Query ClubElo for European clubs and WorldFootballElo for non-European clubs. Merge into a single dictionary: `{club_name: (raw_elo, league_code)}`.

**Step 2 — Normalize to 0–100 globally.**
```python
normalized = (raw_elo - global_min) / (global_max - global_min) * 100
```
Best team in the world = 100. Worst = 0.

> **In plain English:** We put every team in the world on the same 0-to-100 scale. If Manchester City has the highest Elo of any club today, they get 100. If a bottom-tier team in Ecuador has the lowest, they get 0. Everyone else falls in between.

**Step 3 — Derive league strength from team mean.**
```python
league_mean = mean(normalized_score for all teams in that league)
```
Also compute standard deviation and percentile bands (10th, 25th, 50th, 75th, 90th) for each league. These power the swarm plots.

> **In plain English:** A league's quality is simply the average of all its teams' scores. The Premier League might average 72, while the Ecuadorian Serie A might average 31. This is calculated fresh every day from actual team ratings, not opinions.

**Step 4 — Compute relative ability.**
```python
relative_ability = team_normalized_score - league_mean_normalized
```

> **In plain English:** This answers: "How much better or worse is this team compared to their own league?" Arsenal might score 85 in a league that averages 72, giving them a relative ability of +13 — they're well above their league average. A mid-table Argentine team might score 40 in a league that averages 38, giving them +2 — they're about average for their league. This relative number is crucial for the models, because it captures a player's context within their team's system.

**Step 5 — Cache for 1 day.**
The entire rankings snapshot is cached. Subsequent requests within 24 hours return the cached result.

### 6.3 Change in Relative Ability

The key transfer feature:
```python
change_RA = target_relative_ability - source_relative_ability
```

> **In plain English:** If a player moves from a team that's +2 above their league average to a team that's +13 above theirs, the change is +11. This player is moving to a much more dominant team (relative to their league). The models use this to predict how stats will shift — moving to a more dominant team typically increases attacking stats and decreases defensive workload.

### 6.4 Team Name Resolution

Matching team names across ClubElo, WorldFootballElo, and Sofascore is non-trivial — each source uses different naming conventions (e.g. "ManCity" vs "Manchester City" vs "Man City"). `get_team_ranking()` uses a 3-step lookup:

1. **Exact match** in the rankings dictionary
2. **Accent-normalized exact match** via `_strip_accents()` (NFKD Unicode decomposition — ü→u, é→e, etc.)
3. **Fuzzy match** via `_fuzzy_find_team()` with a 5-priority cascade:
   - Exact normalized match (lowercase + stripped)
   - `_EXTREME_ABBREVS` alias lookup (180+ bidirectional entries covering Europe, MLS, Saudi Pro League, J-League, and South America — e.g. PSG↔Paris Saint-Germain, LAFC↔Los Angeles FC, Orlando City↔Orlando City SC)
   - Substring containment (≥6 chars, ≥45% overlap ratio)
   - Word-level matching (shared words ≥4 chars)
   - `SequenceMatcher` ratio ≥0.70 (raised from 0.65 to prevent false positives like "Orlando City SC"→"Man City" where the shared suffix "city" inflates similarity)

Additionally, `_CLUBELO_TO_SOFASCORE` (116 entries) canonicalizes ClubElo's abbreviated team names to Sofascore full display names at data-load time, covering the top leagues in England, France, Germany, Spain, Italy, Portugal, Netherlands, Turkey, Scotland, Belgium, and Austria.

> **In plain English:** Team names are a surprisingly hard problem. "PSG" and "Paris Saint-Germain" are the same team, but a computer doesn't know that unless you tell it. We maintain a large dictionary of abbreviations (180+ entries including MLS, Saudi, and Japanese clubs) plus a smart fuzzy-matching system that handles accents, substrings, and similar-sounding names. Importantly, "Orlando City SC" does NOT accidentally match "Manchester City" — the similarity threshold is calibrated to reject false positives from shared suffixes like "City" or "United." This means a user can type "Bayern" or "Bayern Munich" or "FC Bayern München" and they all resolve to the same team, while MLS clubs like "Inter Miami CF" resolve correctly without confusing them with Inter Milan.

---

## 7. Adjustment Models (sklearn)

### 7.1 Purpose

Before the neural network makes its prediction, simpler linear models provide a "first estimate" adjustment. These handle the basic relationships between team/league quality changes and statistical output.

> **In plain English:** Before the main brain (neural network) does its thing, we first do some quick, straightforward maths: "Players who move to stronger teams typically see their xG go up by this much." These simple models set the baseline.

### 7.2 Team Adjustment — 13 LinearRegression Models

Implemented in `TeamAdjustmentModel` in `backend/features/adjustment_models.py`.

**One model per core metric.** Each predicts the adjusted team-level per-90 at the target club.

```
adjusted_per90 = naive_league_expectation + β × team_relative_feature
```

Where:
- `naive_league_expectation` = the target league's average per-90 for that metric (an offset/baseline)
- `team_relative_feature` = the team's relative ability (how much better/worse than league average)
- `β` = learned coefficient from training data

> **In plain English:** This says: "Start with the league average. Then adjust based on how strong the specific team is. If Arsenal is much better than the Premier League average, their xG should be above the league average too, and `β` tells us by how much."

### 7.3 Team-Position Scaling

`scale_team_position_features()` applies the same percentage change from the team adjustment to position-level features.

```python
if team_xG drops 40%:
    striker_xG drops 40%
    CB_xG drops 40%
    # Every position scales proportionally
```

> **In plain English:** If the model says "this team produces 40% less xG than the player's current team," then we assume every position at the team — strikers, midfielders, defenders — also produces about 40% less xG. It's a proportional scaling.

### 7.4 Player Adjustment — 13 Models × Position

Implemented in `PlayerAdjustmentModel`. Uses 6 input features per prediction, with `change_in_relative_ability` normalized by `/50.0` (mapping the -50..+50 range to -1..+1 for numerical stability):

```
norm_ra = change_in_relative_ability / 50.0

predicted_per90 = intercept
   + b1 × player_previous_per90
   + b2 × avg_position_feature_at_new_team
   + b3 × (avg_position_new_team - avg_position_old_team)
   + b4 × norm_ra
   + b5 × norm_ra²
   + b6 × norm_ra³
```

> **In plain English:** This is more sophisticated. For each stat, it considers:
> 1. **What the player was already producing** (their baseline)
> 2. **What players in that position typically produce at the new team** (the system fit)
> 3. **The difference in positional output between the two teams** (is the new team better or worse for that position?)
> 4. **The change in relative team strength** (how much stronger/weaker is the new team vs. their league, compared to the old team vs. theirs?)
> 5. **Squared and cubed versions of #4** (to capture non-linear effects — the difference between moving up 5 points vs 30 points isn't linear)

### 7.5 Auto-Training

`build_training_data_from_transfers(player_id)` creates training rows from Sofascore transfer history. For the most recent transfer, it pairs:
- Before: the previous club's context (relative ability, league context)
- After: the new club's stats (what actually happened)

`auto_train_from_player_history(player_ids)` collects training data from multiple players and fits both model types.

> **In plain English:** The system can learn from real-world transfers. Give it a list of players who've changed clubs, and it will look at their stats before and after each move, then use that data to teach the adjustment models. The more transfers it learns from, the more accurate the adjustments become.

### 7.5.1 Training Pipeline

`backend/models/training_pipeline.py` provides the end-to-end training entry point. It discovers historical transfers across selected leagues, fetches before/after stats, builds training rows, and fits both sklearn adjustment models and the TensorFlow neural network.

```bash
python backend/models/training_pipeline.py [options]
```

| Argument | Default | Description |
|---|---|---|
| `--leagues` | ENG1,ESP1,GER1,ITA1,FRA1,NED1,POR1,BEL1,ENG2,TUR1,SCO1 | Comma-separated league codes |
| `--seasons-back` | 5 | Number of historical seasons to scan |
| `--skip-discovery` | false | Load cached transfer records instead of re-discovering |
| `--skip-training` | false | Skip training, run backtesting only |
| `--val-ratio` | 0.15 | Validation split ratio |
| `--test-ratio` | 0.10 | Test split ratio |
| `--api-delay` | 2.0 | Delay between API calls (seconds) |

Uses **temporal** (not random) train/val/test split — the most recent transfers go into the test set to simulate real-world prediction conditions.

> **In plain English:** One command kicks off the entire training process. It goes league by league, finds players who transferred, fetches their stats before and after, then trains all the models. The temporal split means we test on the most recent transfers — exactly the scenario the tool will face in production.

### 7.5.2 Backtester

`backend/models/backtester.py` compares trained model predictions against actual post-transfer per-90 stats for a held-out test set. Reports per-metric MAE, RMSE, and directional accuracy (did the model correctly predict whether a metric would go up or down?).

### 7.6 Paper-Aligned Heuristic Fallback

When no trained TF model weights exist, `paper_heuristic_predict()` produces predictions using the paper's structure (Appendix A.3) with calibrated default coefficients. For each metric, three forces compete:

**1. Style shift** — weighted by per-metric `_TEAM_INFLUENCE` (0.15 for dribbles → 0.55 for passing):
```python
style_diff = target_pos_avg[metric] - source_pos_avg[metric]
base = player_val + team_influence * style_diff
```

When real team-position data is unavailable (both position averages equal), per-metric `_LEAGUE_STYLE_COEFF` values estimate style from the relative ability gap, attenuated for extreme transfers. This ensures that **different metrics produce different percentage changes** — not a flat decline or increase:
```python
# Each metric has a unique coefficient (e.g. xA=0.40, shots=0.20, dribbles=0.04)
# Attenuated for extreme moves: |ra|=0.15 → 70% retained, |ra|=0.30+ → 40% retained
style_scale = max(0.3, 1.0 - abs(ra) * 2.0)
estimated_style_diff = source_avg * league_style_coeff * ra * style_scale
```

The attenuation prevents double-counting of quality effects: the team_effect and opp_effect already handle how ability differences affect output. The style estimate captures only the residual tactical differences between teams, which become less meaningful for extreme cross-league transfers where quality dominates over tactics.

**2. Ability factor** — polynomial in `change_relative_ability / 100`, decomposed into two orthogonal forces:
```python
# Team quality: how dominant is the new team within their league?
team_gap = ra - league_gap
team_effect = sensitivity * team_gap * (1 - damping * |team_gap|)

# Opposition quality: how strong is the new league's opposition?
league_gap = (source_league_mean - target_league_mean) / 100
opp_effect = opp_quality_sensitivity * league_gap
```

The damping factor is **asymmetric**: less damping for downgrades (large drops are realistic — Mbappé at Ipswich, de Jong at Man Utd) and more damping for upgrades (talent ceiling effect). This ensures extreme transfers produce realistically large predicted changes.

**3. Opposition quality** — per-metric `_OPP_QUALITY_SENS` coefficients model how facing weaker/stronger opponents affects per-90 output, independent of team quality. Moving to a weaker league boosts offensive output (xG sensitivity: 1.30) even if the team itself is weaker. Dribbling is barely affected (sensitivity: 0.12). This faithfully recreates the paper's Section 4.3.1 observation: Doku's xG increases at Gwangju because opposition quality dominates despite the team being worse. **Note:** The opposition boost is additive to the combined adjustment factor, not multiplicative. For extreme downgrades (elite player to a much weaker team), the team quality penalty typically dominates — the opposition boost partially offsets it but doesn't prevent an overall decline. This is realistic: even the best players produce less at much weaker teams.

**Elite player protection (asymmetric):** A player's Sofascore match rating (0-10 scale) modulates team influence. Elite players (7.5+) retain more individual output when moving UP (reduced team-dependence), but this protection is halved when moving DOWN — even the best players suffer in poor systems.

**The key paper insight:** A player at a worse team can do **better or worse** at a bigger team depending on style fit:
- Moving to a high-crossing team → crosses and xA may **rise** even in a harder league
- Moving to a possession team → passing metrics **rise**, dribbling stays stable
- Moving to a counter-attacking team → take-ons **retained**, passing may **drop**
- Defensive metrics at dominant teams → **drop** (less defending needed)

> **In plain English:** The model considers three separate things: (1) how the new team's style differs from the old one, (2) how much stronger or weaker the team is compared to their league, and (3) how strong the league's opponents are overall. A winger moving from Real Madrid to Ipswich Town would see massive drops in attacking output because: the team creates far fewer chances (team quality effect), the style is completely different (style shift), but the league is similar quality so opposition doesn't help. Each of the 13 stats responds differently to these three forces.

---

## 8. Transfer Portal Neural Network (TensorFlow)

### 8.1 Architecture

Implemented in `backend/models/transfer_portal.py`. A 4-group multi-head neural network, following Table 1 from the paper.

**Why 4 groups instead of 1?** Different stat types have different internal relationships. Shooting metrics (xG, shots) relate to each other more than to defensive metrics (clearances, interceptions). Grouping allows each sub-network to specialize.

> **In plain English:** Instead of one big brain trying to predict everything at once, we have four specialist brains:
> - One that's an expert in shooting stats
> - One for passing stats
> - One for dribbling
> - One for defending
>
> Each specialist shares information within its group but focuses on its area of expertise.

**Per-group architecture:**

Each group receives only the features relevant to its metric type (GROUP_FEATURE_SUBSETS), not the full 43-feature vector. This reduces noise — dribbling doesn't need to see passing team-position averages.

```
Input (group-specific feature subset)
  → Dense layer (128 neurons, ReLU activation)
  → Dropout (30%)
  → Dense layer (64 neurons, ReLU activation)
  → Dropout (30%)
  → Linear output head(s) (1 per target metric)
```

| Group | Input features | Hidden | Output | What it predicts |
|---|---|---|---|---|
| Shooting | 16 | 128 → 64 | 2 | xG, Shots |
| Passing | 25 | 128 → 64 | 7 | xA, Crosses, Passes, Pass %, Long Balls, Chances Created, Pen Area |
| Dribbling | 7 | 128 → 64 | 1 | Take-ons |
| Defending | 13 | 128 → 64 | 3 | Clearances, Interceptions, Possession Won |

> **In plain English:**
> - Each specialist brain only sees the information relevant to its job (ranging from 7 to 25 features per group) — e.g., the Dribbling group excludes passing and defending team-position metrics, while the Shooting group excludes defensive metrics.
> - **Dense layer** = a layer of artificial neurons. 128 neurons in the first layer, 64 in the second. Each neuron looks at the group's inputs and learns to focus on certain patterns.
> - **ReLU activation** = "if the answer is negative, just output zero; otherwise output the answer." This helps the network learn non-linear patterns (like "moving up 30 power ranking points affects stats differently than moving up 5").
> - **Dropout 30%** = during training, randomly turn off 30% of neurons. This is like studying by covering up parts of your notes — it forces the model to not rely too heavily on any single piece of information and makes it better at generalizing.
> - **Linear output** = the final prediction is just a number (the predicted per-90 value), with no cap or floor.

### 8.2 Input Features (43-key feature dict, per-group slicing)

`build_feature_dict()` assembles a 43-key dictionary from components. Each model group then slices only its relevant features internally via GROUP_FEATURE_SUBSETS — e.g., the Dribbling group (7 features) only sees player dribbles, the 4 ability scores, and source/target team-position dribbles, excluding all passing and defending metrics.

```
Full feature dict (43 keys):
[ player per-90 (13) | team_ability_current | team_ability_target |
  league_ability_current | league_ability_target |
  team_pos_current per-90 (13) | team_pos_target per-90 (13) ]

Group slicing examples:
  Shooting (16): player xG/shots + all 4 ability scores + target-pos xG/shots + ...
  Dribbling (7):  player dribbles + all 4 ability scores + source/target-pos dribbles
```

| Block | Count | Description |
|---|---|---|
| Player per-90 | 13 | The player's own recent per-90 stats at their current club |
| Team ability (current) | 1 | Normalized 0–100 Power Ranking of the current club |
| Team ability (target) | 1 | Normalized 0–100 Power Ranking of the target club |
| League ability (current) | 1 | Mean normalized Power Ranking of the current league |
| League ability (target) | 1 | Mean normalized Power Ranking of the target league |
| Team-position per-90 (current) | 13 | Average per-90 for the player's position at their current club |
| Team-position per-90 (target) | 13 | Average per-90 for the player's position at the target club |

> **In plain English:** We're telling the neural network 43 things about the transfer:
> - "This player currently produces these 13 stats per 90 minutes"
> - "Their current team is this strong, and the target team is this strong"
> - "Their current league is this competitive, and the target league is this competitive"
> - "Players in this position at the current team typically produce these 13 stats"
> - "Players in this position at the target team typically produce these 13 stats"
>
> From all of that, the network learns to predict: "Here's what this specific player will produce at the new club."

### 8.3 Training

`TransferPortalModel.fit()` trains all 4 groups simultaneously with:
- **Optimizer:** Adam (adaptive learning rate)
- **Loss:** Mean Squared Error (MSE) — penalizes large prediction errors more than small ones
- **Validation split:** 15% of data held out to monitor overfitting
- **Epochs:** 50 passes through the training data

### 8.4 Prediction

`TransferPortalModel.predict(feature_dict)` runs the feature dict through all 4 groups (each slicing its relevant subset) and returns a dictionary of 13 predicted per-90 values.

**Auto-loading:** `predict()` auto-loads trained weights from `data/models/` when available. `is_trained()` checks for both `.keras` model files and `feature_scaler.pkl`. When no trained model exists, falls back to `paper_heuristic_predict()` with a warning log.

---

## 9. Shortlist Scoring

### 9.1 Purpose

The shortlist generator needs to rank hundreds of players by "how well do they match what the user wants?" This is a weighted similarity problem.

### 9.2 Rate-Limit Protection

Sofascore applies aggressive rate-limiting (HTTP 403/429) after 2-3 rapid sequential requests. Without protection, scanning multiple leagues in quick succession results in all subsequent leagues failing silently — producing 0 candidates.

**Solution:** A configurable `_INTER_LEAGUE_DELAY` (default 1.5 seconds) is inserted between league API calls. The default scan is limited to the **Big 5 European leagues** (ENG1, ESP1, GER1, ITA1, FRA1) instead of all 37+ leagues. The **player's own league is always scanned first** (most likely to succeed since season resolution is already cached from the player search). Users can explicitly select additional leagues via the UI.

A per-league diagnostic panel shows which leagues returned data and how many candidates were found from each, making it easy to diagnose API issues.

> **In plain English:** Sofascore blocks you if you ask for too much data too quickly. We solve this by: (1) adding a brief pause between league requests, (2) only searching the top 5 leagues by default instead of all 37+, and (3) always starting with the player's own league since it's most likely to already have cached data. A diagnostic panel shows exactly which leagues succeeded and which didn't.

### 9.3 Filter Design — None-Passthrough

Filter fields (age, minutes played, league, power ranking) use a **None-passthrough** design: when a candidate has `None` for a filtered field (e.g. age unknown from the Sofascore API), the candidate **passes through** the filter rather than being excluded. This means `max_age=30` selects "players aged ≤30 OR players with unknown age."

This is intentional — Sofascore API data is sparse, and excluding unknowns would silently drop valid candidates, giving the false impression of 0 results. Users see "—" in the results table for missing fields and can judge quality themselves.

> **In plain English:** If we don't know a player's age (because Sofascore didn't provide it), we include them anyway rather than silently dropping them. This prevents the tool from showing "0 candidates" when there are actually hundreds of valid players whose metadata is just incomplete.

### 9.4 Algorithm

Implemented in `backend/models/shortlist_scorer.py` → `score_candidates()`:

**Step 1 — Filter candidates.** Apply user constraints: max age, min minutes, position, league, club Power Ranking cap.

**Step 2 — Cluster candidates by playing style** using sklearn KMeans (k = √(n/2), capped 3–10). The reference player is included in clustering to determine their cluster. Only used when n ≥ 10 candidates; otherwise falls back to direct distance.

> **In plain English:** Before ranking, the system groups all candidates into clusters of players with similar playing styles. A high-xG target striker and a creative playmaker will end up in different clusters. This helps find players who aren't just statistically close but who play a similar *type* of game.

**Step 3 — Compute weighted Euclidean distance to reference player.**
```python
raw_weighted = predicted_per90 * user_weight_array
standardized = StandardScaler(raw_weighted)
distance = sqrt(sum((candidate - reference)²))
base_score = 1.0 - normalized_distance   # inverted: closer = higher
```

> **In plain English:** Each candidate is compared to the reference player across all the weighted metrics, and the comparison is done on a standardized scale so that no single metric dominates just because its numbers are bigger.

**Step 4 — Apply cluster bonus.**
```python
if same_cluster_as_reference:
    base_score = min(1.0, base_score * 1.15)   # 15% bonus
```

> **In plain English:** Candidates who play in the same style cluster as the reference player get a 15% score boost. This favours players who are not just numerically similar but stylistically similar.

**Step 5 — Per-metric breakdown.**
```python
metric_score[m] = max(0.0, 1.0 - abs(metric_diff) / 3.0)
```

**Step 6 — Sort descending.** Best match first. Candidate dataclass includes `same_cluster_as_reference` boolean.

### 9.5 Percentage Changes

`compute_percentage_changes()` calculates the percent change from current to predicted per-90 for each metric:

```python
pct_change = ((predicted - current) / abs(current)) * 100
```

This is displayed in metric bar charts and the Hot or Not verdict. Both pages use **dual simulation** per the paper's methodology — percentage changes compare model-predicted-at-target vs model-predicted-at-current (not raw stats vs predicted).

---

## 10. End-to-End Prediction Flow

Here's what happens when a user types "Bukayo Saka → Real Madrid" into the Transfer Impact page:

```
1. search_player("Saka")
   → Sofascore returns player ID 961995

2. search_team("Real Madrid")
   → Sofascore returns team ID, tournament ID

3. get_player_stats(961995)
   → Returns per-90 stats, team, position, minutes, age

4. get_season_list(tournament_id) → optional season selector

5. compute_daily_rankings()
   → ClubElo + WorldFootballElo → normalize 0-100
   → Arsenal Power Ranking: 85, league mean: 72
   → Real Madrid Power Ranking: 91, league mean: 68

6. get_team_position_averages(source_team, position)
   get_team_position_averages(target_team, position)
   → Arsenal wingers avg per-90: {xG: 0.35, xA: 0.18, ...}
   → Real Madrid wingers avg per-90: {xG: 0.42, xA: 0.22, ...}

7. compute_player_features(player_stats)
   → Rolling average or season aggregate
   → Blend with prior if low minutes
   → RAG confidence: GREEN (weight=0.95, 2100 mins)

8. Dual simulation (paper Section 4):
   predicted_current = paper_heuristic_predict(
       player_stats, source_pos_avg → source_pos_avg, ra=0)
   predicted_target = paper_heuristic_predict(
       player_stats, source_pos_avg → target_pos_avg, ra=Δ)
   → Style differences AND ability differences are per-metric

9. compute_percentage_changes(predicted_current, predicted_target)
   → {expected_goals: +14.2%, shots: -3.1%, crosses: +8.5%, ...}
   → Note: some metrics UP, some DOWN — reflects style fit

10. Render:
    → metric_bar.show(...)        # bar chart of changes
    → power_ranking_chart.show()  # timeline comparison
    → swarm_plot.show_swarm_grid()# league context
    → confidence_badge()          # RAG indicator
    → predictions table           # full breakdown
```

> **In plain English:** The system searches for the player and target club, fetches all their data, calculates how strong both teams/leagues are, fetches what each team's position players typically produce (tactical style), simulates the player at both clubs using the same model, then draws everything on screen. The key is that the comparison is model-vs-model (not raw stats vs prediction), and different stats can go up or down depending on style fit.

---

## 11. Testing and Validation

### 11.1 Unit Tests

208 tests across 13 test files, all using `unittest` with `mock.patch` for external API calls:

| Test File | What It Tests | Count |
|---|---|---|
| `test_sofascore_client.py` | Player search, stats parsing, per-90 computation, caching, tournament discovery, match logs | 22 |
| `test_new_features.py` | Team search, transfer history, league stats, seasons, season-specific stats | 16 |
| `test_adjustment_training.py` | Training data builder, auto-training, team-position scaling | 4 |
| `test_cache.py` | Cache set/get, TTL expiry, namespace clearing | 8 |
| `test_clubelo_client.py` | European Elo fetching, caching, league listing | 11 |
| `test_elo_router.py` | Source routing, normalization, coverage detection | 10 |
| `test_improvements.py` | Historical rankings, league comparison, fuzzy matching | 18 |
| `test_worldelo_client.py` | Non-European Elo fetching, HTML parsing, caching | 9 |
| `test_fuzzy_matching.py` | Extreme abbreviations, accent normalization, substring/word matching, ClubElo canonicalization | 64 |
| `test_training_pipeline.py` | First-1000-min targets, non-transfer samples, league means, per-group features, change_ra normalization | 23 |
| `test_backtester.py` | Backtester predictions vs actuals, hold-out evaluation | 3 |
| `test_shortlist_scorer.py` | Filter None-passthrough, score_candidates with clustering, compute_percentage_changes edge cases | 15 |
| `test_new_features.py` | Team search, transfer history, league stats, seasons, season-specific stats | 5 |

All tests use mocked HTTP responses — no network access required. Temporary cache directories are created per test module and cleaned up in `tearDownModule()`.

> **In plain English:** We have 208 automated checks that verify the system works correctly. They use fake data (so they don't need the internet), and they cover everything from "does the search work?" to "is the per-90 math right?" to "does the cache actually cache things?" to "does fuzzy team name matching handle accents and abbreviations?" to "do the shortlist filters correctly handle missing data?" If anyone changes the code and breaks something, these tests catch it immediately.

### 11.2 Run Tests

```bash
python -m pytest tests/ -v
```

### 11.3 Future Validation

- **Backtesting:** Predict historical transfers, compare predicted per-90 to actual post-transfer per-90.
- **Cross-validation:** k-fold CV on the neural network using historical transfer data.
- **Calibration analysis:** Are predicted percentage changes well-calibrated? (If we predict +10%, does the actual outcome average +10%?)

> **In plain English:** Right now we test that the code runs correctly. The next step is testing that the *predictions* are correct — we'd look at past transfers where we know what actually happened and check: "Did the model get it right?" This is the gold standard for any prediction system.

---

## References

1. Dinsdale, J. & Gallagher, J. (2022). *The Transfer Portal: Predicting the Impact of a Player Transfer on the Receiving Club.* Springer. https://doi.org/10.48550/arXiv.2201.11533

2. Elo, A.E. (1978). *The Rating of Chessplayers, Past and Present.* Arco Publishing.

3. ClubElo — http://clubelo.com

4. World Football Elo Ratings — http://eloratings.net

5. Sofascore — https://www.sofascore.com
