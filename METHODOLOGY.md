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
| `/player/{id}` | `get_player_stats()` — profile + season aggregate stats | 1 day |
| `/player/{id}/unique-tournament/{tid}/season/{sid}/statistics/overall` | `get_player_stats_for_season()` — stats for a specific season | 1 day |
| `/player/{id}/transfer-history` | `get_player_transfer_history()` — career transfer records | 7 days |
| `/unique-tournament/{tid}/season/{sid}/statistics` | `get_league_player_stats()` — bulk league-wide player stats | 1 day |
| `/unique-tournament/{tid}/seasons` | `get_season_list()` — available seasons for a tournament | 1 day |
| `/team/{id}/players` | `get_team_players_stats()` — squad roster | 1 day |

Sofascore returns **raw totals** (e.g. 15 goals in 2,000 minutes), not per-90 values. The client converts these in `_parse_stats()`.

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

Implemented in `PlayerAdjustmentModel`. Uses 6 input features per prediction:

```
predicted_per90 = intercept
   + b1 × player_previous_per90
   + b2 × avg_position_feature_at_new_team
   + b3 × (avg_position_new_team - avg_position_old_team)
   + b4 × change_in_relative_ability
   + b5 × change_in_relative_ability²
   + b6 × change_in_relative_ability³
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

### 7.6 Paper-Aligned Heuristic Fallback

When no trained TF model weights exist, `paper_heuristic_predict()` produces predictions using the paper's structure (Appendix A.3) with calibrated default coefficients. For each metric, two forces compete:

**1. Style shift** — weighted by per-metric `_TEAM_INFLUENCE` (0.15 for dribbles → 0.50 for passing):
```python
style_diff = target_pos_avg[metric] - source_pos_avg[metric]
base = player_val + team_influence * style_diff
```

When real team-position data is unavailable (both position averages equal), per-metric `_LEAGUE_STYLE_COEFF` values estimate style from the league quality gap. This ensures that **different metrics produce different percentage changes** — not a flat decline or increase:
```python
# Each metric has a unique coefficient (e.g. xA=0.18, shots=0.08, dribbles=0.02)
estimated_style_diff = source_avg * league_style_coeff * ra
```

**2. Ability factor** — polynomial (linear + quadratic + cubic) in `change_relative_ability / 100`, with per-metric `_ABILITY_SENSITIVITY`:
```python
ability_factor = 1 + sensitivity*ra − 0.15*sensitivity*ra² + 0.02*sensitivity*ra³
```

**The key paper insight:** A player at a worse team can do **better or worse** at a bigger team depending on style fit:
- Moving to a high-crossing team → crosses and xA may **rise** even in a harder league
- Moving to a possession team → passing metrics **rise**, dribbling stays stable
- Moving to a counter-attacking team → take-ons **retained**, passing may **drop**
- Defensive metrics at dominant teams → **drop** (less defending needed)

> **In plain English:** The model doesn't just say "harder league = everything gets worse." It accounts for tactical fit. A creative winger joining Barcelona might see passing stats *increase* despite moving to a harder league, because Barcelona's system demands more passing. Meanwhile their defensive stats drop because Barcelona dominates the ball. And their dribbling stays roughly the same because that's an individual skill. Each of the 13 stats responds differently.

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

```
Input (43 features)
  → Dense layer (128 neurons, ReLU activation)
  → Dropout (30%)
  → Dense layer (64 neurons, ReLU activation)
  → Dropout (30%)
  → Linear output head(s) (1 per target metric)
```

| Group | Input | Hidden | Output | What it predicts |
|---|---|---|---|---|
| Shooting | 43 | 128 → 64 | 2 | xG, Shots |
| Passing | 43 | 128 → 64 | 7 | xA, Crosses, Passes, Pass %, Long Balls, Chances Created, Pen Area |
| Dribbling | 43 | 128 → 64 | 1 | Take-ons |
| Defending | 43 | 128 → 64 | 3 | Clearances, Interceptions, Possession Won |

> **In plain English:**
> - **Dense layer** = a layer of artificial neurons. 128 neurons in the first layer, 64 in the second. Each neuron looks at all 43 inputs and learns to focus on certain patterns.
> - **ReLU activation** = "if the answer is negative, just output zero; otherwise output the answer." This helps the network learn non-linear patterns (like "moving up 30 power ranking points affects stats differently than moving up 5").
> - **Dropout 30%** = during training, randomly turn off 30% of neurons. This is like studying by covering up parts of your notes — it forces the model to not rely too heavily on any single piece of information and makes it better at generalizing.
> - **Linear output** = the final prediction is just a number (the predicted per-90 value), with no cap or floor.

### 8.2 Input Features (43-dimensional)

`build_feature_dict()` assembles the input vector from components:

```
[ player per-90 (13) | team_ability_current | team_ability_target |
  league_ability_current | league_ability_target |
  team_pos_current per-90 (13) | team_pos_target per-90 (13) ]
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

`TransferPortalModel.predict(feature_dict)` runs the 43-feature input through all 4 groups and returns a dictionary of 13 predicted per-90 values.

---

## 9. Shortlist Scoring

### 9.1 Purpose

The shortlist generator needs to rank hundreds of players by "how well do they match what the user wants?" This is a weighted similarity problem.

### 9.2 Algorithm

Implemented in `backend/models/shortlist_scorer.py` → `score_candidates()`:

**Step 1 — Filter candidates.** Apply user constraints: max age, min minutes, position, league, club Power Ranking cap.

**Step 2 — Normalize across candidates.**
```python
for each active metric:
    mean = mean(predicted_per90 across all candidates)
    std  = std(predicted_per90 across all candidates)
    normalized = (candidate_value - mean) / std
```

> **In plain English:** If the average xG across all candidates is 0.3 and the standard deviation is 0.1, a player with 0.5 xG gets a normalized score of +2.0 (excellent), while one with 0.2 gets -1.0 (below average). This puts all metrics on the same scale so we can compare apples to apples.

**Step 3 — Apply user weights.**
```python
weighted_score = normalized_value × user_weight    # weight is 0.0 to 1.0
```

The user sets weights for each metric. "I care a lot about xG (weight 1.0) and less about clearances (weight 0.2)."

**Step 4 — Compute final score.**
```python
final_score = sum(weighted_scores) / sum(weights)
```

> **In plain English:** Each candidate gets a score that reflects: "How good are they at the things the scout cares about?" A player who's exceptional at the high-weighted metrics and mediocre at the low-weighted ones will score higher than a player who's average at everything.

**Step 5 — Sort descending.** Best match first.

### 9.3 Percentage Changes

`compute_percentage_changes()` calculates the percent change from current to predicted per-90 for each metric:

```python
pct_change = ((predicted - current) / abs(current)) * 100
```

This is displayed in metric bar charts and the Hot or Not verdict.

---

## 10. End-to-End Prediction Flow

Here's what happens when a user types "Bukayo Saka → Real Madrid" into the Transfer Impact page:

```
1. search_player("Saka")
   → Sofascore returns player ID 961995

2. search_team("Real Madrid")
   → Sofascore returns team ID, tournament ID

3. get_player_stats(961995)
   → Returns per-90 stats, team, position, minutes

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

97 tests across 8 test files, all using `unittest` with `mock.patch` for external API calls:

| Test File | What It Tests | Count |
|---|---|---|
| `test_sofascore_client.py` | Player search, stats parsing, per-90 computation, caching, tournament discovery | 21 |
| `test_new_features.py` | Team search, transfer history, league stats, seasons, season-specific stats | 16 |
| `test_adjustment_training.py` | Training data builder, auto-training, team-position scaling | 4 |
| `test_cache.py` | Cache set/get, TTL expiry, namespace clearing | 8 |
| `test_clubelo_client.py` | European Elo fetching, caching, league listing | 7 |
| `test_elo_router.py` | Source routing, normalization, coverage detection | 10 |
| `test_improvements.py` | Historical rankings, league comparison, fuzzy matching | 13 |
| `test_worldelo_client.py` | Non-European Elo fetching, HTML parsing, caching | 8 |

All tests use mocked HTTP responses — no network access required. Temporary cache directories are created per test module and cleaned up in `tearDownModule()`.

> **In plain English:** We have 97 automated checks that verify the system works correctly. They use fake data (so they don't need the internet), and they cover everything from "does the search work?" to "is the per-90 math right?" to "does the cache actually cache things?" If anyone changes the code and breaks something, these tests catch it immediately.

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

1. Dinsdale, J. & Gallagher, J. (2022). *The Transfer Portal: Predicting the Impact of a Player Transfer on the Receiving Club.* Springer. https://doi.org/10.1007/978-3-031-02044-5_14

2. Elo, A.E. (1978). *The Rating of Chessplayers, Past and Present.* Arco Publishing.

3. ClubElo — http://clubelo.com

4. World Football Elo Ratings — http://eloratings.net

5. Sofascore — https://www.sofascore.com
