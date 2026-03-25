# TransferScope — White Paper

**Predicting Player Performance Across Leagues: A Transfer Intelligence Platform**

---

## Abstract

TransferScope is a football transfer intelligence platform that predicts how a player's statistical output will change when they move to a new club. The system implements and extends the methodology described in *Dinsdale & Gallagher (2022) — "The Transfer Portal"*, combining dynamic Elo-based Power Rankings, per-90 rolling feature windows, sklearn adjustment models, and a TensorFlow multi-head neural network to produce actionable transfer predictions across 37+ leagues worldwide. TransferScope surfaces these predictions through three tools — Transfer Impact analysis, multi-league Shortlist Generation, and a rapid rumour validator — delivered via a Streamlit interface designed for scouting workflows.

> **In plain English:** TransferScope is a tool that answers the question: "If we sign this player, what will they actually produce at our club?" It uses a combination of club strength ratings, recent player form, and machine learning to predict how every major stat — goals, assists, passes, defensive actions — will change when a player moves from one team to another. It covers 37+ leagues worldwide and comes with a web interface where you can search players, compare clubs, and evaluate transfer rumours.

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

Three external sources provide the raw data:

| Source | What It Provides | Coverage |
|---|---|---|
| **Sofascore REST API** | Player stats (per-90), team rosters, transfer histories, season lists, team search | Global — any league on sofascore.com |
| **ClubElo** (via `soccerdata`) | Daily Elo ratings for ~600 European clubs | Top 10+ European leagues |
| **WorldFootballElo** (HTTP scrape) | Daily Elo ratings for global clubs | South America, MLS, Asia, Africa |

All API calls are routed through a `diskcache` SQLite cache layer with configurable time-to-live. Player stats expire after 1 day; Elo ratings refresh daily; search results persist for 7 days.

> **In plain English:** We pull data from three places: Sofascore gives us the player numbers (goals, assists, passes, etc.), ClubElo tells us how strong European clubs are, and WorldFootballElo covers the rest of the world. We save results locally so the app doesn't slow down by re-downloading the same data every time.

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

**Dynamic Power Rankings.** Rather than using a static league tier table ("Premier League = Tier 1, Eredivisie = Tier 3"), TransferScope computes Power Rankings dynamically:

1. Collect all club Elo ratings from ClubElo (Europe) and WorldFootballElo (global)
2. Derive league strength as the mean Elo of all teams in that league on a given date
3. Normalize all clubs to a 0–100 scale globally
4. Compute each team's **relative ability** = team score − league mean score

> **In plain English:** Instead of hard-coding "the Premier League is better than the Eredivisie," we calculate it fresh every day from actual results. This means if the Portuguese league has a strong year, its Power Ranking goes up automatically. Every team gets a score from 0 to 100. Then we calculate how much better or worse each team is compared to their own league average — a mid-table Premier League team and the top Argentine team might be equally strong globally, even though one dominates their league and the other doesn't.

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
- `_LEAGUE_STYLE_COEFF` — estimated style shift when team-position data is unavailable

This means a player at a worse team **can improve or decline** at a bigger team, per-metric, depending on whether the target team's style fits them. This is the key paper insight (Sections 4.2–4.3): stylistic differences between teams cause different metrics to move in different directions.

> **In plain English:** Even without a trained neural network, the system makes smart per-metric predictions. A crossing winger joining a team that plays wide will see their crosses and assists go up, even if the league is harder. A dribbler will keep their dribbling numbers almost unchanged because that's an individual skill. A defender joining a dominant team will defend less. Each stat is predicted independently based on both ability and style.

**Transfer Portal Neural Network (TensorFlow).** A 4-group multi-head neural network with 43 input features and 13 output heads:

| Group | Targets | Heads |
|---|---|---|
| Shooting | xG, Shots | 2 |
| Passing | xA, Crosses, Passes, Pass %, Long Balls, Chances Created, Pen Area Entries | 7 |
| Dribbling | Take-ons | 1 |
| Defending | Clearances, Interceptions, Possession Won Final 3rd | 3 |

Each group has the same architecture: Input → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Linear output heads. Trained with Adam optimizer and MSE loss.

> **In plain English:** The neural network is the "brain" of the system. It takes in 43 numbers about a player (their current stats, how strong their current team is, how strong the target team is, what position players typically do at both clubs) and outputs 13 predictions — one for each stat. It's organized into 4 groups: shooting stats, passing stats, dribbling stats, and defensive stats. Each group has its own little brain that specializes in predicting that type of stat.
>
> The "Dropout(0.3)" bit means the model randomly ignores 30% of its connections during training — this is like studying by covering up parts of your notes and forcing yourself to remember. It prevents the model from memorizing specific examples and helps it generalize to new players it hasn't seen before.

The 43 input features are:
- 13 player per-90 metrics (current club)
- 4 ability scores (team and league, current and target)
- 13 team-position per-90 metrics (current club)
- 13 team-position per-90 metrics (target club)

### 2.4 User Interface

A Streamlit application with three pages, styled with a custom "Tactical Noir" dark theme (deep charcoal, amber/gold data accents, JetBrains Mono for numbers, Outfit for headings):

**Transfer Impact.** The user enters a player and a target club (with Sofascore autocomplete). The system fetches stats, computes Power Rankings, builds predictions via dual simulation (player simulated at both current and target clubs, per paper Section 4), and displays:
- Metric bars showing predicted percentage changes for all 13 metrics
- Power Ranking timeline comparing source and target clubs
- RAG confidence indicator
- Swarm plots showing the player's position in their current league distribution
- A detailed predictions table with "Simulated Current" vs "Predicted" columns

**Shortlist Generator.** The user selects a player to replace and assigns weights to each metric. The system scans players across selected leagues (defaulting to 11 major leagues for speed, expandable to all 37+), scores them by weighted similarity, and returns a ranked table with filters for age, position, league, minutes played, and club Power Ranking cap.

**Hot or Not.** A rapid rumour validator. Enter a player and a target club; receive an instant HOT / TEPID / NOT verdict with the top 3 predicted metric changes, a summary of improving vs declining metrics, and the player's transfer history.

---

## 3. Multi-League Coverage

TransferScope covers 37+ leagues across 4 continents:

**Europe (30+):** Premier League, Championship, La Liga, La Liga 2, Bundesliga, 2. Bundesliga, Serie A, Serie B, Ligue 1, Ligue 2, Eredivisie, Primeira Liga, Belgian Pro League, Süper Lig, Scottish Premiership, Austrian Bundesliga, Swiss Super League, Greek Super League, Danish Superliga, and 15+ additional European leagues

**South America (7):** Brasileirão Série A, Brasileirão Série B, Argentine Primera División, Colombian Primera A, Chilean Primera División, Uruguayan Primera División, Ecuadorian Serie A

**North America (1):** MLS

**Asia (2):** Saudi Pro League, J-League

> **In plain English:** The tool works across basically all the major leagues in the world — 37+ leagues total. If you want to scout a player from the Chilean league and predict how they'd do at Arsenal, it can do that. If you want to compare players across the Bundesliga, Serie A, and the Brasileirão, it can do that too.

---

## 4. Key Innovations Beyond the Paper

While TransferScope faithfully implements the Dinsdale & Gallagher methodology, it extends the original work in several ways:

### 4.1 Multi-League Shortlist Search
The original paper focused on predicting individual transfers. TransferScope adds the ability to scan players across all 37+ registered leagues and rank candidates by weighted metric similarity — turning the model into a **scouting tool**, not just a prediction engine.

> **In plain English:** The paper told you how to predict *one* transfer. We turned it into a tool that can search *thousands* of players and find the best fits.

### 4.2 Dynamic Data Sources
The original paper used static datasets. TransferScope pulls live data from Sofascore, ClubElo, and WorldFootballElo, with intelligent caching. Power Rankings update daily. Player stats refresh automatically. The system never goes stale.

### 4.3 Transfer History Integration
Sofascore's transfer history endpoint provides the raw data needed to auto-train the adjustment models. The `build_training_data_from_transfers()` function pairs consecutive transfers to generate training rows, eliminating the need for manual dataset construction.

> **In plain English:** The system can learn from real transfers. It looks at a player's career — "they moved from Club A to Club B" — and uses the before/after stats to teach the model what typically happens when a player changes teams.

### 4.4 Season Selection
Users can analyze historical seasons (e.g. "2023/24") by selecting from the seasons API. This enables retrospective validation: predict what a player's stats would have been, then compare to what actually happened.

### 4.5 Global Elo Coverage
The paper's Power Rankings relied on a single Elo source. TransferScope uses a dual-source approach — ClubElo for European clubs (more granular) and WorldFootballElo for the rest of the world — with an intelligent router that selects the best source per club.

### 4.6 Real League Context for Visualizations
Swarm plots in the Transfer Impact page are populated with actual league-wide per-90 distributions from Sofascore, not synthetic data. This shows exactly where a player sits among their peers.

> **In plain English:** When you see a chart showing "this player is in the 85th percentile for chances created in the Premier League," that's based on real data from every player in the Premier League that season — not a guess.

### 4.7 Paper-Faithful Dual Simulation
The Transfer Impact page simulates each player at **both** their current and target clubs, then compares the two model outputs — faithful to the paper's methodology (Section 4): "we generate performance predictions using Transfer Portal for players at their current club too." This ensures both sides of the comparison come from the same model process, reducing sensitivity to noise in observed data.

> **In plain English:** Instead of comparing "what the player actually did" vs "what we predict at the new club," we compare two predictions: "what the model thinks they'd do at their current club" vs "what the model thinks they'd do at the new club." This is fairer because both numbers come from the same system.

### 4.8 Per-Metric Style Differentiation
Predictions use three per-metric coefficient tables — `_TEAM_INFLUENCE` (how team-dependent a stat is), `_ABILITY_SENSITIVITY` (how much league quality changes it), and `_LEAGUE_STYLE_COEFF` (estimated style shift when team data is unavailable). This means a player moving to a harder league **can see some metrics improve** if the target team's style suits them (e.g., a crossing winger joining a wide-play team), while other metrics decline. Dribbling is treated as near-irreducible (barely changes with team/league), while passing and defending are heavily system-dependent.

> **In plain English:** The model doesn't just say "harder league = everything gets worse." It says "harder league AND this specific team's style = some things get better, others get worse." A creative passer joining Barcelona might see their passing numbers *increase* despite moving to a harder league, because Barcelona's system creates more passing opportunities. Meanwhile, their defensive stats might drop because Barcelona dominates possession and players defend less.

---

## 5. Technical Stack

| Layer | Technology | What it does |
|---|---|---|
| Player statistics | Sofascore REST API | Gets the actual numbers — goals, passes, etc. |
| European Elo ratings | ClubElo via `soccerdata` | Tells us how strong European teams are |
| Global Elo ratings | WorldFootballElo via HTTP | Same, but for non-European teams |
| Feature engineering | pandas, numpy | Crunches raw data into model-ready inputs |
| Adjustment models | scikit-learn LinearRegression | Simple "rule of thumb" adjustments for league/team changes |
| Neural network | TensorFlow / Keras | The main prediction brain — learns complex patterns |
| UI | Streamlit | The web interface you interact with |
| Caching | diskcache (SQLite-backed) | Remembers API responses so we don't re-download |
| Visualization | Plotly | Draws the interactive charts |
| Testing | pytest + unittest (97 tests) | Makes sure nothing is broken |

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

- **No match-level granularity.** Season-aggregate stats are used rather than true match-by-match rolling windows. Match-level data would improve accuracy.
- **Adjustment model training data.** Auto-training uses the most recent transfer only. All historical transfers with season-specific stats would produce more robust models.
- **No market value data.** Sofascore doesn't provide transfer fees, limiting financial filtering.
- **Single-season neural network.** The TensorFlow model initializes with random weights. Training on historical transfers would significantly improve predictions.

> **In plain English:**
> - We're currently using season averages instead of game-by-game data (which would be more accurate but harder to get).
> - The learning models are still young — they get better as we feed them more historical transfer data.
> - We can't filter by transfer fee because Sofascore doesn't share that information.
> - The neural network hasn't been trained on a huge dataset yet — right now it's smart about structure but hasn't "studied" millions of examples.

### 6.2 Future Directions

- **Match-level data pipeline.** Integrate match-by-match stats to compute true 1000-minute rolling windows.
- **Automated model retraining.** Schedule periodic retraining as new transfer data becomes available.
- **Tactical embeddings.** Incorporate formation and tactical style to capture system fit beyond raw metrics.
- **Expected minutes model.** Predict not just per-90 output but likely playing time at the target club.
- **Multi-season validation.** Backtest predictions against historical transfers to quantify accuracy.

---

## 7. Conclusion

TransferScope demonstrates that the methodology from Dinsdale & Gallagher (2022) can be implemented as a production-ready scouting tool with live data, global coverage, and an interface designed for real decision-making. By combining dynamic Power Rankings, position-aware adjustment models, and multi-head neural networks, the system bridges the gap between what a player has done and what they will do — the question at the heart of every transfer decision.

> **In plain English:** This isn't just an academic exercise. TransferScope is a working tool that takes a real research paper and turns it into something you can actually use to evaluate transfers, find replacement players, and make better scouting decisions — all from a web browser, with data that refreshes daily.

---

## References

1. Dinsdale, J. & Gallagher, J. (2022). *The Transfer Portal: Predicting the Impact of a Player Transfer on the Receiving Club.* In: Brefeld, U., Davis, J., Van Haaren, J., Zimmermann, A. (eds) Machine Learning and Data Mining for Sports Analytics. Springer. https://doi.org/10.1007/978-3-031-02044-5_14

2. ClubElo — http://clubelo.com — European club Elo rating system.

3. World Football Elo Ratings — http://eloratings.net — Global club Elo ratings.

4. Sofascore — https://www.sofascore.com — Player statistics and match data.

5. Elo, A.E. (1978). *The Rating of Chessplayers, Past and Present.* Arco Publishing.
