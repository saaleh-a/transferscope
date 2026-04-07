# TransferScope — White Paper

**Predicting Player Performance Across Leagues: A Transfer Intelligence Platform**

---

## Abstract

TransferScope is a football transfer intelligence platform that predicts how a player's statistical output will change when they move to a new club. The system implements and extends the methodology described in *Dinsdale & Gallagher (2022) — "The Transfer Portal"*, combining dynamic Elo-based Power Rankings, per-90 rolling feature windows, sklearn adjustment models, and a TensorFlow multi-head neural network to produce actionable transfer predictions across 51 leagues worldwide. TransferScope surfaces these predictions through three tools — Transfer Impact analysis, multi-league Shortlist Generation, and a rapid rumour validator — delivered via a Streamlit interface designed for scouting workflows.

> **In plain English:** TransferScope is a tool that answers the question: "If we sign this player, what will they actually produce at our club?" It uses a combination of club strength ratings, recent player form, and machine learning to predict how every major stat — goals, assists, passes, defensive actions — will change when a player moves from one team to another. It covers 51 leagues worldwide and comes with a web interface where you can search players, compare clubs, and evaluate transfer rumours.

---

## 1. The Problem

### 1.1 Why Transfer Prediction Matters

Football transfers are among the most financially consequential decisions in professional sport. Premier League clubs collectively spent over £2 billion in the 2023 summer window alone. Yet the fundamental question remains surprisingly difficult to answer with data: **"If we sign this player, what will their actual output look like at our club?"**

A player's per-90 statistics — expected goals, progressive passes, defensive actions — are products of their individual ability *and* the system they play in. A striker averaging 0.6 xG/90 at a mid-table Eredivisie side may produce very different numbers at a dominant Premier League team. The league is harder, but the team creates more chances. These competing forces are what transfer prediction must disentangle.

> **In plain English:** Clubs spend hundreds of millions on players every summer, but they're essentially guessing. A player who looks great in the Dutch league might struggle in England — or might actually do *better* because they're joining a stronger team that creates more chances. Both forces are at play, and right now, most tools just show you raw stats without accounting for this context. That's the gap TransferScope fills.

### 1.2 The Gap

Existing scouting tools present raw player data — current season stats, historical trends, comparison radars. They tell you what a player *has done*. They do not tell you what a player *will do* in a different tactical and competitive context. The missing piece is a model that accounts for:

- **League quality differential** — moving from the Eredivisie to the Premier League
- **Team strength differential** — moving from a relegation side to a title contender
- **Positional context** — how a team's system uses players in that position
- **Data confidence** — whether the player has enough minutes for reliable projections

> **In plain English:** Current scouting platforms are like looking at a player's CV. They tell you what they've done. TransferScope is more like a job interview simulation — it tells you what they'd *actually do* in a new environment, accounting for the difficulty of the league, the strength of the team, and how much data we have to be confident in the prediction.

### 1.3 The Paper

Dinsdale & Gallagher (2022) proposed the "Transfer Portal" — a framework that uses a hierarchical Elo-based Power Ranking system, position-aware adjustment models, and multi-output neural networks to predict post-transfer per-90 statistics. TransferScope is a faithful implementation of this framework, extended with modern data sources, multi-league search, and a production-ready interface.

---

## 2. System Overview

TransferScope has four layers:

```
Data Acquisition → Feature Engineering → Prediction Models → User Interface
```

> **In plain English:** Think of it as an assembly line:
> 1. **Collect** — grab player stats and club ratings from the internet
> 2. **Prepare** — turn raw numbers into things the model can understand (like "how strong is this team compared to their league?")
> 3. **Predict** — feed it all into math models that output "here's what this player will do at the new club"
> 4. **Display** — show it on a web page with charts and tables

### 2.1 Data Acquisition

Seven external sources provide the raw data:

| Source | What It Provides | Coverage |
|---|---|---|
| **Sofascore REST API** | Player stats (per-90), team rosters, transfer histories, season lists, team search | Global — any league on sofascore.com |
| **Opta Power Rankings** (via `curl_cffi`) | Team ratings (0-100), league averages, league sizes | ~14K teams worldwide — primary source for inference |
| **ClubElo** (via `soccerdata`) | Daily Elo ratings for ~600 European clubs | Top 10+ European leagues — used for training/historical |
| **WorldFootballElo** (HTTP scrape) | Daily Elo ratings for global clubs | South America, MLS, Asia, Africa — used for training/historical |
| **REEP Register** (CSV download) | Team name aliases (~45K clubs worldwide) | Dynamic cross-provider name resolution |
| **StatsBomb Open Data** (via `statsbombpy`) | Spatial data — shot locations, pass networks, heatmaps | Shot maps, pass networks, heatmaps in Transfer Impact |
| **football-data.co.uk** (CSV download) | Match-level results for league profiling | Coefficient calibration for style/opposition weights |

All API calls are routed through a `diskcache` SQLite cache layer with configurable time-to-live. Player stats expire after 1 day; Opta rankings refresh weekly; ClubElo/WorldFootballElo ratings refresh daily; search results persist for 7 days.

> **In plain English:** We pull data from multiple places: Sofascore gives us the player numbers (goals, assists, passes, etc.), Opta gives us the official team and league strength rankings for today's predictions, and ClubElo/WorldFootballElo provide historical strength data for training the model. We save results locally so the app doesn't slow down by re-downloading the same data every time.

### 2.2 Feature Engineering

**Rolling Windows.** Player per-90 metrics are computed over a 1,000-minute rolling window (approximately 11 full matches). Team and team-position metrics use a 3,000-minute window.

> **In plain English:** Instead of looking at an entire season's stats (which can be skewed by a hot/cold streak months ago), we focus on roughly the last 11 games' worth of playing time. This gives us a "recent form" picture. For teams, we use a bigger window (about 33 games) because team styles change more slowly.

**Prior Blending.** Players with insufficient data receive blended features using the formula:

```
weight = min(1, minutes_played / 1000)
feature = (1 - weight) × prior + weight × raw_rolling_average
```

This produces a Red / Amber / Green confidence indicator: Red (< 300 minutes), Amber (300–700), Green (> 700).

> **In plain English:** If a player has only played 200 minutes this season, we can't trust their stats much. So we blend their actual numbers with a "guess" based on what an average player in their position typically produces. The traffic light tells you how much to trust the prediction:
> - 🟢 **Green** = plenty of data, prediction is solid
> - 🟡 **Amber** = some data, prediction is reasonable but not certain
> - 🔴 **Red** = very little data, prediction leans heavily on assumptions

**Dynamic Power Rankings.** Rather than using a static league tier table ("Premier League = Tier 1, Eredivisie = Tier 3"), TransferScope computes Power Rankings dynamically using a **hybrid approach**:

- **Inference (today's date):** Opta Power Rankings (0-100 scale) are the primary source. Official league averages (`seasonAverageRating`) and team counts (`leagueSize`) come directly from Opta's `league-meta.json`, ensuring correct values regardless of team name matching. ClubElo provides the raw Elo (on the ~1000-2100 scale the model was trained on) for teams it covers; teams not in ClubElo get a linear rescale from Opta's 0-100.
- **Training (historical dates):** ClubElo + WorldFootballElo, since Opta has no historical archive. League averages are computed as the mean of all teams in each league on the transfer date.

In both cases, each team's **relative ability** = team score − league mean score.

> **In plain English:** For today's predictions, we use Opta's official rankings — the same system used by professional broadcasters. Their data includes official league averages and team counts, so we don't have to guess or compute them ourselves. For training the model on past transfers, we use historical Elo data since Opta doesn't have an archive. Either way, we calculate how much better or worse each team is compared to their own league average.

The full feature vector contains 46 features: 13 player per-90, 4 ability scores (team and league, current and target), 13 team-position per-90 (current), 13 team-position per-90 (target), and 3 interaction features (ability_gap, gap_squared, league_gap).

### 2.3 Prediction Models

Two model tiers operate in sequence, plus a heuristic fallback:

**Adjustment Models (sklearn LinearRegression).** Thirteen linear regression models per metric handle the team-level and player-level adjustments:

- *Team adjustment*: 13 models — one per core metric — that map a team's relative strength to an expected per-90 adjustment at the target league
- *Team-position scaling*: scales position-level features by the same percentage change as the team adjustment
- *Player adjustment*: 13 models × position that use polynomial features (up to cubic) of the change in relative ability

> **In plain English:** These are simpler "rule of thumb" models that answer questions like: "If a team is 15% stronger than their league average, how does that typically affect their strikers' goal output?" and "If a player moves to a team that's 20 points stronger in our ranking, how much does each stat typically change?" We have one of these for each of the 13 stats, and they're specific to each playing position — because moving to a stronger team affects a defender differently than it affects a winger.

**Paper-Aligned Heuristic Fallback (`paper_heuristic_predict`).** When no trained TF model weights exist, this function produces predictions using the paper's structure with calibrated per-metric coefficients. Each metric has three independent weights:
- `_TEAM_INFLUENCE` — how much the metric is team-dependent vs individual (0.15 for dribbles → 0.50 for passing)
- `_ABILITY_SENSITIVITY` — how much league/team quality affects it (offensive positive, defensive negative)
- `_LEAGUE_STYLE_COEFF` — estimated style shift when team-position data is unavailable, attenuated for extreme transfers (prevents double-counting quality effects)

This means a player at a worse team **can improve or decline** at a bigger team, per-metric, depending on whether the target team's style fits them. This is the key paper insight (Sections 4.2–4.3): stylistic differences between teams cause different metrics to move in different directions.

> **In plain English:** Even without a trained neural network, the system makes smart per-metric predictions. A crossing winger joining a team that plays wide will see their crosses and assists go up, even if the league is harder. A dribbler will keep their dribbling numbers almost unchanged because that's an individual skill. A defender joining a dominant team will defend less. Each stat is predicted independently based on both ability and style.

**Transfer Portal Neural Network (TensorFlow).** A 4-group multi-head neural network with per-group feature subsets (not all 46 features to every group) and 13 output heads:

| Group | Input Features | Targets | Heads |
|---|---|---|---|
| Shooting | 36 | xG, Shots | 2 |
| Passing | 50 | xA, Crosses, Passes, Pass %, Long Balls, Chances Created, Pen Area Entries | 7 |
| Dribbling | 22 | Take-ons | 1 |
| Defending | 33 | Clearances, Interceptions, Possession Won Final 3rd | 3 |

Each group has the same architecture: Input → Dense(128, ReLU) → BatchNormalization → Dropout(0.3) → Dense(64, ReLU) → BatchNormalization → Dropout(0.3) → Linear output heads (dribbling uses 64→32 + dropout 0.4). Trained with Adam optimizer and Huber loss (delta=1.0) with L2 regularization. `DELTA_SHRINKAGE=0.85`. Auto-loads trained weights from `data/models/` when available; falls back to `paper_heuristic_predict()` when untrained.

> **In plain English:** The neural network is the "brain" of the system. A full set of 89 numbers about a player is assembled (their current stats including 10 additional metrics, team strength, league strength, raw Elo scores, REEP metadata, relative ability, league normalisation features, what position players typically do at both clubs, plus interaction features). But each specialist group only sees the subset relevant to its job — the shooting brain gets 36 features, the passing brain 50, the dribbling brain 22, and the defending brain 33. Each group outputs its predictions for that stat type. This focus makes each specialist better at its job.
>
> The "Dropout(0.3)" bit means the model randomly ignores 30% of its connections during training — this is like studying by covering up parts of your notes and forcing yourself to remember. It prevents the model from memorizing specific examples and helps it generalize to new players it hasn't seen before.

The full 89-feature dictionary is assembled from:
- 13 core player per-90 metrics (current club)
- 10 additional player metrics (xGOT, npxG, dispossessed, duels won %, aerial duels won %, recoveries, fouls won, touches, goals conceded on pitch, xG against on pitch)
- 4 ability scores (team and league, current and target)
- 2 raw Elo features (current and target)
- 2 REEP metadata features (height, age)
- 3 relative ability features (current, target, gap)
- 13 league normalisation features (player / source league mean)
- 13 league mean ratio features (source mean / target mean)
- 26 team-position per-90 metrics (13 current + 13 target)
- 3 interaction features (ability gap, gap², league gap)

Each group slices only its relevant features internally (GROUP_FEATURE_SUBSETS), reducing noise — dribbling doesn't need passing team-position averages.

### 2.4 User Interface

A Streamlit application with six pages, styled with a custom "Tactical Noir" dark theme (deep charcoal, amber/gold data accents, JetBrains Mono for numbers, Outfit for headings):

**Transfer Impact.** The user enters a player and a target club (with Sofascore autocomplete). The system fetches stats, computes Power Rankings, builds predictions via dual simulation (player simulated at both current and target clubs, per paper Section 4), and displays:
- Metric bars showing predicted percentage changes for all 13 metrics
- Power Ranking timeline comparing source and target clubs
- RAG confidence indicator
- Swarm plots showing the player's position in their current league distribution
- A detailed predictions table with "Simulated Current" vs "Predicted" columns

**Shortlist Generator.** The user selects a player to replace and assigns weights to each metric. The system scans players across selected leagues (defaulting to the Big 5 European leagues for reliability — Sofascore rate-limits rapid sequential requests, so a 1.5-second delay is inserted between league API calls), clusters candidates by playing style using k-means (k=√(n/2), capped 3–10), scores them by weighted Euclidean distance to the reference player with a 15% same-cluster bonus, and returns a ranked table with filters for age, position, league, minutes played, and club Power Ranking cap. The player's own league is always scanned first. Filters use a None-passthrough design — candidates with unknown age or minutes pass through rather than being silently excluded, since Sofascore API data is often sparse. A per-league diagnostic panel shows which leagues returned data and how many candidates were found.

**Hot or Not.** A rapid rumour validator. Enter a player and a target club; receive an instant HOT / TEPID / NOT verdict with the top 3 predicted metric changes, a summary of improving vs declining metrics, the player's transfer history, and Power Ranking context. Uses **dual simulation** (same as Transfer Impact) — compares model-predicted-at-target vs model-predicted-at-current, per paper Section 4. The verdict uses position-aware weighting: offensive metrics count 1.5× for forwards, defensive metrics count 1.5× for defenders. Thresholds are ±3% average predicted change.

**Backtest Validator.** Compares model predictions against actual post-transfer per-90 statistics for validated transfers. Reports per-metric MAE, RMSE, and directional accuracy.

**Diagnostics.** System health page showing data source connectivity status, cache statistics, model loading status, and league registry coverage.

> **In plain English:** You read a rumour — "Osimhen to Arsenal." You type it in, press a button, and get a big verdict: HOT (good move), TEPID (meh), or NOT (bad move). The verdict is smarter for different positions — for a striker, goals and assists matter more than defensive stats. It shows you the top 3 stats that would change, a summary of what improves vs. declines, and the player's entire transfer history. If the data isn't available (e.g. the player hasn't played enough), you'll see UNKNOWN instead of a misleading verdict.

---

## 3. Multi-League Coverage

TransferScope covers 51 leagues across 4 continents:

**Europe (40):** Premier League, Championship, La Liga, La Liga 2, Bundesliga, 2. Bundesliga, Serie A, Serie B, Ligue 1, Ligue 2, Eredivisie, Primeira Liga, Belgian Pro League, Süper Lig, Scottish Premiership, Austrian Bundesliga, Swiss Super League, Greek Super League, Czech First League, Danish Superliga, Croatian 1. HNL, Serbian Super Liga, Norwegian Eliteserien, Swedish Allsvenskan, Polish Ekstraklasa, Romanian Liga I, Ukrainian Premier League, Russian Premier League, Bulgarian First Professional League, Hungarian NB I, Cypriot First Division, Finnish Veikkausliiga, Slovak Super Liga, Slovenian PrvaLiga, Bosnian Premier Liga, Israeli Premier League, Kazakhstan Premier League, Icelandic Úrvalsdeild, League of Ireland Premier Division, Welsh Premier League, Georgian Erovnuli Liga

**South America (7):** Brasileirão Série A, Brasileirão Série B, Argentine Primera División, Colombian Primera A, Chilean Primera División, Uruguayan Primera División, Ecuadorian Serie A

**North America (1):** MLS

**Asia (2):** Saudi Pro League, J-League

> **In plain English:** The tool works across basically all the major leagues in the world — 51 leagues total. If you want to scout a player from the Chilean league and predict how they'd do at Arsenal, it can do that. If you want to compare players across the Bundesliga, Serie A, and the Brasileirão, it can do that too.

---

## 4. Key Innovations Beyond the Paper

While TransferScope faithfully implements the Dinsdale & Gallagher methodology, it extends the original work in several ways:

### 4.1 Multi-League Shortlist Search with K-Means Clustering
The original paper focused on predicting individual transfers. TransferScope adds the ability to scan players across all 51 registered leagues (defaulting to Big 5 for reliability), cluster candidates by playing style using k-means (k=√(n/2), capped 3–10), and rank them by weighted Euclidean distance to a reference player with a 15% same-cluster bonus — turning the model into a **scouting tool**, not just a prediction engine. Rate-limit protection (1.5s inter-league delay, player's own league scanned first) prevents Sofascore API throttling that previously caused 0 results. Filters use a None-passthrough design so candidates with incomplete metadata (unknown age, minutes) aren't silently dropped.

> **In plain English:** The paper told you how to predict *one* transfer. We turned it into a tool that can search *thousands* of players, group them by playing style, and find the best fits — prioritizing players who not only have similar numbers but play a similar type of game. We also solved the rate-limiting problem that was causing 0 results by adding brief pauses between league scans and starting with the most reliable league first.

### 4.2 Dynamic Data Sources
The original paper used static datasets. TransferScope pulls live data from Sofascore, Opta, ClubElo, and WorldFootballElo, with intelligent caching. Opta Power Rankings update weekly; ClubElo/WorldFootballElo ratings update daily. Player stats refresh automatically. The system never goes stale.

### 4.3 Transfer History Integration
Sofascore's transfer history endpoint provides the raw data needed to auto-train the adjustment models. The `build_training_data_from_transfers()` function pairs consecutive transfers to generate training rows, eliminating the need for manual dataset construction.

> **In plain English:** The system can learn from real transfers. It looks at a player's career — "they moved from Club A to Club B" — and uses the before/after stats to teach the model what typically happens when a player changes teams.

### 4.4 Season Selection
Users can analyze historical seasons (e.g. "2023/24") by selecting from the seasons API. This enables retrospective validation: predict what a player's stats would have been, then compare to what actually happened.

### 4.5 Global Power Rankings Coverage
The paper's Power Rankings relied on a single Elo source. TransferScope uses a **triple-source hybrid approach**:
- **Opta Power Rankings** for inference (today's date) — official 0-100 ratings with league averages and team counts from `league-meta.json`, fetched via `curl_cffi` direct JSON extraction from the JS bundle (no Selenium/browser needed)
- **ClubElo** for European clubs on historical dates (higher granularity for training)
- **WorldFootballElo** for non-European clubs on historical dates

League code resolution uses Opta's `domestic_league_name` + `country` compound key (e.g. "Premier League" + "England" → ENG1) to avoid ambiguity, with ClubElo as fallback.

### 4.6 Real League Context for Visualizations
Swarm plots in the Transfer Impact page are populated with actual league-wide per-90 distributions from Sofascore, not synthetic data. This shows exactly where a player sits among their peers.

> **In plain English:** When you see a chart showing "this player is in the 85th percentile for chances created in the Premier League," that's based on real data from every player in the Premier League that season — not a guess.

### 4.7 Paper-Faithful Dual Simulation
The Transfer Impact page simulates each player at **both** their current and target clubs, then compares the two model outputs — faithful to the paper's methodology (Section 4): "we generate performance predictions using Transfer Portal for players at their current club too." This ensures both sides of the comparison come from the same model process, reducing sensitivity to noise in observed data.

> **In plain English:** Instead of comparing "what the player actually did" vs "what we predict at the new club," we compare two predictions: "what the model thinks they'd do at their current club" vs "what the model thinks they'd do at the new club." This is fairer because both numbers come from the same system.

### 4.8 Per-Metric Style Differentiation
Predictions use three per-metric coefficient tables — `_TEAM_INFLUENCE` (how team-dependent a stat is), `_ABILITY_SENSITIVITY` (how much league quality changes it), and `_LEAGUE_STYLE_COEFF` (estimated style shift when team data is unavailable, attenuated for extreme transfers). This means a player moving to a harder league **can see some metrics improve** if the target team's style suits them (e.g., a crossing winger joining a wide-play team), while other metrics decline. Dribbling is treated as near-irreducible (barely changes with team/league), while passing and defending are heavily system-dependent. The style estimation is scaled down proportionally for extreme transfers (|ra| > 0.15) to prevent double-counting of quality effects already handled by team_effect and opp_effect.

> **In plain English:** The model doesn't just say "harder league = everything gets worse." It says "harder league AND this specific team's style = some things get better, others get worse." A creative passer joining Barcelona might see their passing numbers *increase* despite moving to a harder league, because Barcelona's system creates more passing opportunities. Meanwhile, their defensive stats might drop because Barcelona dominates possession and players defend less.

### 4.9 Multi-Tournament Stats Aggregation
When a player's primary domestic league tournament returns 0 minutes (common for young players rotating between league and cup competitions), the system automatically tries **all tournaments** the player's team participates in and uses the one where the player has the most data. This prevents false "no stats available" results for players like youth prospects who may have significant cup or European competition minutes.

> **In plain English:** If a young player hasn't played in the league but has played 1,300 minutes in cup competitions, the system will automatically find and use that data instead of showing "0 minutes."

### 4.10 Opposition Quality Modelling
The heuristic prediction explicitly models **opposition quality** as a separate force from team quality. Moving to a weaker league means facing weaker defenders and goalkeepers, which boosts per-90 offensive output even if the team itself is weaker. This is implemented via per-metric `_OPP_QUALITY_SENS` coefficients (e.g. xG sensitivity 1.30 — weaker opposition significantly boosts expected goals). This faithfully recreates the paper's observation (Section 4.3.1) that Doku's xG increases at Gwangju (weak team, much weaker league) because opposition quality dominates.

**Important caveat:** While opposition quality boosts per-90 output independently of team quality, these boosts are **additive to the combined adjustment factor** (not multiplicative), which means they are typically insufficient to compensate for major downward team moves. An elite player moving from a Champions League contender to a relegation-zone Championship side would still experience a significant net drop in most metrics — the weaker opposition helps, but the team quality penalty dominates. The model correctly captures that even Kvaratskhelia or Mbappé would produce less at a much weaker team, while showing that some metrics (xG, shots) decline less than others (passing, creativity) because of the opposition quality offset.

> **In plain English:** The model knows that playing against weaker defenders means more goals, even if the team creates fewer chances. A player in the K-League faces much easier opposition than in La Liga, so their xG goes up despite being at a worse team. This is a separate effect from "how good is the team" and is modelled independently. However, for extreme downgrades (elite player to a very weak team), the opposition boost only partially offsets the team quality drop — the overall prediction correctly shows most metrics declining, but with some (like xG and shots) declining less than others.

### 4.11 Asymmetric Prediction Calibration
Extreme transfers (elite player to relegation team, or lower-league player to top club) use asymmetric damping: **downgrades allow larger predicted drops** while upgrades are more conservative (talent ceiling effect). Elite player protection is also asymmetric — a high-rated player retains more output when moving to a better team but is less protected when moving to a much weaker team, because even the best players suffer in poor systems.

> **In plain English:** Mbappé moving from Real Madrid to a relegation team would see massive stat drops — the model doesn't protect him just because he's elite. But the same player moving from a good team to a *slightly* better one wouldn't see unrealistically huge improvements. The model is calibrated to be realistic in both directions.

### 4.12 Per-Group Feature Subsets
Each of the 4 TensorFlow model groups receives only the features relevant to its metric type (GROUP_FEATURE_SUBSETS), not the full 46-feature vector. Shooting uses 19 features, Passing 28, Dribbling 10, and Defending 16. This reduces noise and improves generalization — the dribbling model doesn't need to see passing team-position averages.

> **In plain English:** Each specialist brain only looks at the data that matters for its job. The shooting expert doesn't waste time looking at long ball accuracy. This makes each model more focused and less likely to be confused by irrelevant information.

### 4.13 Robust Team Name Resolution
Matching team names across four data sources (Opta, ClubElo, WorldFootballElo, Sofascore) requires handling abbreviations, accents, and regional naming differences. TransferScope uses a 3-step lookup: exact match → accent-normalized match → fuzzy matching with a 5-priority cascade (including 502 extreme abbreviation aliases covering Europe, MLS, Saudi Pro League, and J-League, plus 531 ClubElo-to-Sofascore canonicalization entries). The SequenceMatcher similarity threshold is set at 0.70 to reject false positives where short common suffixes (like "City") drive enough similarity to match unrelated teams (e.g. "Orlando City SC" must not match "Man City"). At runtime, `_build_dynamic_aliases()` augments these with ~45,000 additional team name variants from the REEP open data register, providing near-universal coverage without manual maintenance.

> **In plain English:** "PSG" and "Paris Saint-Germain" and "Paris SG" are all the same team. "Bayern München" and "Bayern Munich" are the same. The system knows 502 abbreviation pairs (including MLS teams, Saudi, and Japanese clubs) and can fuzzy-match team names that are close but not identical. At runtime, it also downloads a comprehensive open-source database of ~45,000 clubs to automatically discover additional name variants. This means data from different sources always connects correctly, without false positives like "Orlando City" matching "Manchester City."

### 4.14 End-to-End Training Pipeline
A CLI-driven training pipeline (`training_pipeline.py`) discovers historical transfers across 11+ leagues and up to 5 seasons, fetches before/after stats, builds training rows with temporal split, and fits both sklearn adjustment models and the TensorFlow neural network. A companion backtester evaluates predictions against actual post-transfer outcomes.

> **In plain English:** One command trains the entire system from scratch: find transfers, collect data, train models, test accuracy. Everything is automated and reproducible.

### 4.15 Match-Level Data Access
Per-match statistics are available via `get_player_match_logs()`, enabling true rolling window computation from game-by-game data rather than season aggregates.

> **In plain English:** Instead of just "season averages," we can get stats from each individual game, allowing more precise recent-form calculations.

### 4.16 REEP-Based Dynamic Alias Resolution
Team name resolution is augmented at runtime by downloading the REEP open data register's `teams.csv` (~45,000 clubs). `_build_dynamic_aliases()` cross-links all provider name columns into normalized bidirectional aliases. `_get_merged_aliases()` overlays curated hardcoded entries on top. This means new club promotions, name changes, and mergers are automatically discovered without code changes.

> **In plain English:** Rather than manually adding every club name variant (there are tens of thousands worldwide), the system downloads a comprehensive open-source database and automatically figures out "these are all the same team." If that download fails, it falls back to over 1,000 hand-curated entries.

### 4.17 Spatial Visualization
Shot maps, pass networks, and heatmaps from StatsBomb open data provide spatial context in the Transfer Impact page. `pitch_viz.py` renders these using `mplsoccer` — showing where on the pitch a player operates, supplementing the numerical per-90 predictions.

> **In plain English:** Numbers tell you "this player takes 3 shots per 90," but the shot map shows you *where* — from 6-yard box tap-ins or 25-yard speculative efforts. This gives scouts a spatial picture that per-90 numbers alone can't provide.

### 4.18 Data-Driven Coefficient Calibration
`calibrate_style_coefficients()` in `adjustment_models.py` uses match-level data from football-data.co.uk to empirically refine `_LEAGUE_STYLE_COEFF` and `_OPP_QUALITY_SENS` via cross-league CV analysis. The final coefficients are a 60/40 blend of data-derived and hand-tuned default values, balancing statistical fit with domain knowledge.

> **In plain English:** Instead of guessing "how much does the Bundesliga's style differ from La Liga's?", the system measures it from real match data and uses that to improve its predictions. The 60/40 blend means if the data disagrees with football common sense, the defaults still have a say.

---

## 5. Technical Stack

| Layer | Technology | What it does |
|---|---|---|
| Player statistics | Sofascore REST API | Gets the actual numbers — goals, passes, etc. |
| Power Rankings (inference) | Opta Power Rankings via `curl_cffi` | Official team and league strength ratings for today |
| Power Rankings (training) | ClubElo + WorldFootballElo | Historical team strength for training the model |
| Feature engineering | pandas, numpy | Crunches raw data into model-ready inputs |
| Adjustment models | scikit-learn Ridge/LinearRegression | Simple "rule of thumb" adjustments for league/team changes |
| Neural network | TensorFlow / Keras | The main prediction brain — learns complex patterns |
| UI | Streamlit | The web interface you interact with |
| Caching | diskcache (SQLite-backed) | Remembers API responses so we don't re-download |
| Visualization | Plotly | Draws the interactive charts |
| Spatial data | StatsBomb via `statsbombpy` + `mplsoccer` | Shot maps, pass networks, heatmaps |
| Coefficient calibration | football-data.co.uk CSVs | Data-driven style coefficient refinement |
| Team alias augmentation | REEP register (~45K clubs) | Dynamic team name resolution |
| Testing | pytest + unittest (488 tests) | Makes sure nothing is broken |

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

- **No match-level granularity in predictions (yet).** Match-level data is accessible via `get_player_match_logs()`, but the prediction pipeline currently uses season-aggregate stats rather than true match-by-match rolling windows.
- **Adjustment model training data.** Auto-training uses the most recent transfer only. All historical transfers with season-specific stats would produce more robust models.
- **No market value data.** Sofascore doesn't provide transfer fees, limiting financial filtering.
- **Training pipeline requires API access.** The end-to-end training pipeline needs live Sofascore data; there is no bundled offline training dataset.
- **Single-tournament season selector.** The season dropdown only shows seasons from the player's primary tournament. Multi-tournament aggregation is used when the primary returns 0 minutes, but the season selector doesn't yet expose all tournaments.
- **No historical data in predictions.** The current system uses single-season snapshot stats. Multi-season trend analysis (is the player improving or declining?) is not yet incorporated into the prediction model.
- **Shortlist default scope.** The shortlist generator defaults to Big 5 leagues only (to avoid Sofascore rate-limiting). Users must manually select additional leagues, which then requires longer scan times with rate-limit delays.
- **Sparse metadata.** Sofascore API doesn't always return player age or complete position data. The None-passthrough filter design mitigates this but means some results lack metadata.

> **In plain English:**
> - Match-by-match data is accessible but not yet wired into the rolling windows for predictions.
> - The learning models get better as we feed them more historical transfer data via the training pipeline.
> - We can't filter by transfer fee because Sofascore doesn't share that information.
> - Training requires an internet connection to fetch real data; there's no pre-packaged training set.

### 6.2 Future Directions

- **Match-level rolling windows in predictions.** Wire `get_player_match_logs()` into the prediction pipeline for true 1000-minute rolling windows instead of season aggregates.
- **Automated model retraining.** Schedule periodic retraining as new transfer data becomes available.
- **Tactical embeddings.** Incorporate formation and tactical style to capture system fit beyond raw metrics.
- **Expected minutes model.** Predict not just per-90 output but likely playing time at the target club.
- **Multi-season validation.** Extend backtester to cover multiple seasons and compute calibration metrics.
- **Improved verdict classification.** Replace the 3-tier Hot/Tepid/Not system with a richer classification (e.g. 5-tier with confidence-weighted thresholds).

---

## 7. Conclusion

TransferScope demonstrates that the methodology from Dinsdale & Gallagher (2022) can be implemented as a production-ready scouting tool with live data, global coverage, and an interface designed for real decision-making. By combining dynamic Power Rankings, position-aware adjustment models, and multi-head neural networks, the system bridges the gap between what a player has done and what they will do — the question at the heart of every transfer decision.

> **In plain English:** This isn't just an academic exercise. TransferScope is a working tool that takes a real research paper and turns it into something you can actually use to evaluate transfers, find replacement players, and make better scouting decisions — all from a web browser, with data that refreshes daily.

---

## References

1. Dinsdale, J. & Gallagher, J. (2022). *The Transfer Portal: Predicting the Impact of a Player Transfer on the Receiving Club.* https://doi.org/10.48550/arXiv.2201.11533

2. Opta Power Rankings — https://dataviz.theanalyst.com/opta-power-rankings/ — Official team and league Power Rankings by Stats Perform.

3. ClubElo — http://clubelo.com — European club Elo rating system.

4. World Football Elo Ratings — http://eloratings.net — Global club Elo ratings.

5. Sofascore — https://www.sofascore.com — Player statistics and match data.

6. Elo, A.E. (1978). *The Rating of Chessplayers, Past and Present.* Arco Publishing.
