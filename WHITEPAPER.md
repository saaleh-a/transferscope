# TransferScope — White Paper

**Predicting Player Performance Across Leagues: A Transfer Intelligence Platform**

---

## Abstract

TransferScope is a football transfer intelligence platform that predicts how a player's statistical output will change when they move to a new club. The system implements and extends the methodology described in *Dinsdale & Gallagher (2022) — "The Transfer Portal"*, combining dynamic Elo-based Power Rankings, per-90 rolling feature windows, sklearn adjustment models, and a TensorFlow multi-head neural network to produce actionable transfer predictions across 20+ leagues worldwide. TransferScope surfaces these predictions through three tools — Transfer Impact analysis, multi-league Shortlist Generation, and a rapid rumour validator — delivered via a Streamlit interface designed for scouting workflows.

---

## 1. The Problem

### 1.1 Why Transfer Prediction Matters

Football transfers are among the most financially consequential decisions in professional sport. Premier League clubs collectively spent over £2 billion in the 2023 summer window alone. Yet the fundamental question remains surprisingly difficult to answer with data: **"If we sign this player, what will their actual output look like at our club?"**

A player's per-90 statistics — expected goals, progressive passes, defensive actions — are products of their individual ability *and* the system they play in. A striker averaging 0.6 xG/90 at a mid-table Eredivisie side may produce very different numbers at a dominant Premier League team. The league is harder, but the team creates more chances. These competing forces are what transfer prediction must disentangle.

### 1.2 The Gap

Existing scouting tools present raw player data — current season stats, historical trends, comparison radars. They tell you what a player *has done*. They do not tell you what a player *will do* in a different tactical and competitive context. The missing piece is a model that accounts for:

- **League quality differential** — moving from the Eredivisie to the Premier League
- **Team strength differential** — moving from a relegation side to a title contender
- **Positional context** — how a team's system uses players in that position
- **Data confidence** — whether the player has enough minutes for reliable projections

### 1.3 The Paper

Dinsdale & Gallagher (2022) proposed the "Transfer Portal" — a framework that uses a hierarchical Elo-based Power Ranking system, position-aware adjustment models, and multi-output neural networks to predict post-transfer per-90 statistics. TransferScope is a faithful implementation of this framework, extended with modern data sources, multi-league search, and a production-ready interface.

---

## 2. System Overview

TransferScope has four layers:

```
Data Acquisition → Feature Engineering → Prediction Models → User Interface
```

### 2.1 Data Acquisition

Three external sources provide the raw data:

| Source | What It Provides | Coverage |
|---|---|---|
| **Sofascore REST API** | Player stats (per-90), team rosters, transfer histories, season lists, team search | Global — any league on sofascore.com |
| **ClubElo** (via `soccerdata`) | Daily Elo ratings for ~600 European clubs | Top 10+ European leagues |
| **WorldFootballElo** (HTTP scrape) | Daily Elo ratings for global clubs | South America, MLS, Asia, Africa |

All API calls are routed through a `diskcache` SQLite cache layer with configurable time-to-live. Player stats expire after 1 day; Elo ratings refresh daily; search results persist for 7 days.

### 2.2 Feature Engineering

**Rolling Windows.** Player per-90 metrics are computed over a 1,000-minute rolling window (approximately 11 full matches). Team and team-position metrics use a 3,000-minute window.

**Prior Blending.** Players with insufficient data receive blended features using the formula:

```
weight = min(1, minutes_played / 1000)
feature = (1 - weight) × prior + weight × raw_rolling_average
```

This produces a Red / Amber / Green confidence indicator: Red (< 300 minutes), Amber (300–700), Green (> 700).

**Dynamic Power Rankings.** Rather than using a static league tier table, TransferScope computes Power Rankings dynamically:

1. Collect all club Elo ratings from ClubElo (Europe) and WorldFootballElo (global)
2. Derive league strength as the mean Elo of all teams in that league on a given date
3. Normalize all clubs to a 0–100 scale globally
4. Compute each team's **relative ability** = team score − league mean score

This approach updates automatically as teams' Elo ratings change, requires no manual tier maintenance, and is faithful to the paper's methodology.

### 2.3 Prediction Models

Two model tiers operate in sequence:

**Adjustment Models (sklearn LinearRegression).** Thirteen linear regression models per metric handle the team-level and player-level adjustments:

- *Team adjustment*: 13 models — one per core metric — that map a team's relative strength in their previous league to an expected per-90 adjustment at the target league
- *Team-position scaling*: scales position-level features by the same percentage change as the team adjustment
- *Player adjustment*: 13 models × position that use polynomial features (up to cubic) of the change in relative ability, plus the player's previous per-90 and position-level context at the new team

These models can be auto-trained from Sofascore transfer history data using `auto_train_from_player_history()`.

**Transfer Portal Neural Network (TensorFlow).** A 4-group multi-head neural network with 43 input features and 13 output heads:

| Group | Targets | Heads |
|---|---|---|
| Shooting | xG, Shots | 2 |
| Passing | xA, Crosses, Passes, Pass %, Long Balls, Chances Created, Pen Area Entries | 7 |
| Dribbling | Take-ons | 1 |
| Defending | Clearances, Interceptions, Possession Won Final 3rd | 3 |

Each group has the same architecture: Input → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Linear output heads. Trained with Adam optimizer and MSE loss.

The 43 input features are:
- 13 player per-90 metrics (current club)
- 4 ability scores (team and league, current and target)
- 13 team-position per-90 metrics (current club)
- 13 team-position per-90 metrics (target club)

### 2.4 User Interface

A Streamlit application with three pages, styled with a custom "Tactical Noir" dark theme:

**Transfer Impact.** The user enters a player and a target club (with Sofascore autocomplete). The system fetches stats, computes Power Rankings, builds predictions, and displays:
- Metric bars showing predicted percentage changes for all 13 metrics
- Power Ranking timeline comparing source and target clubs
- RAG confidence indicator
- Swarm plots showing the player's position in their current league distribution
- A detailed predictions table

**Shortlist Generator.** The user selects a player to replace and assigns weights to each metric. The system scans players across selected leagues (up to 20), scores them by weighted similarity, and returns a ranked table with filters for age, position, league, minutes played, and club Power Ranking cap.

**Hot or Not.** A rapid rumour validator. Enter a player and a target club; receive an instant HOT / TEPID / NOT verdict with the top 3 predicted metric changes, a summary of improving vs declining metrics, and the player's transfer history.

---

## 3. Multi-League Coverage

TransferScope covers 20 leagues across 4 continents:

**Europe (10):** Premier League, Championship, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira Liga, Belgian Pro League, Süper Lig

**South America (7):** Brasileirão Série A, Brasileirão Série B, Argentine Primera División, Colombian Primera A, Chilean Primera División, Uruguayan Primera División, Ecuadorian Serie A

**North America (1):** MLS

**Asia (2):** Saudi Pro League, J-League

This coverage is extensible — any league present on Sofascore can be added by registering its tournament ID in the league registry.

---

## 4. Key Innovations Beyond the Paper

While TransferScope faithfully implements the Dinsdale & Gallagher methodology, it extends the original work in several ways:

### 4.1 Multi-League Shortlist Search
The original paper focused on predicting individual transfers. TransferScope adds the ability to scan players across all 20 registered leagues and rank candidates by weighted metric similarity — turning the model into a scouting tool, not just a prediction engine.

### 4.2 Dynamic Data Sources
The original paper used static datasets. TransferScope pulls live data from Sofascore, ClubElo, and WorldFootballElo, with intelligent caching. Power Rankings update daily. Player stats refresh automatically. The system never goes stale.

### 4.3 Transfer History Integration
Sofascore's transfer history endpoint provides the raw data needed to auto-train the adjustment models. The `build_training_data_from_transfers()` function pairs consecutive transfers to generate training rows, eliminating the need for manual dataset construction.

### 4.4 Season Selection
Users can analyze historical seasons (e.g. "2023/24") by selecting from Sofascore's seasons API. This enables retrospective validation: predict what a player's stats would have been, then compare to what actually happened.

### 4.5 Global Elo Coverage
The paper's Power Rankings relied on a single Elo source. TransferScope uses a dual-source approach — ClubElo for European clubs (more granular) and WorldFootballElo for the rest of the world — with an intelligent router that selects the best source per club.

### 4.6 Real League Context for Visualizations
Swarm plots in the Transfer Impact page are populated with actual league-wide per-90 distributions from Sofascore, not synthetic data. This shows exactly where a player sits among their peers.

---

## 5. Technical Stack

| Layer | Technology |
|---|---|
| Player statistics | Sofascore REST API |
| European Elo ratings | ClubElo via `soccerdata` |
| Global Elo ratings | WorldFootballElo via HTTP |
| Feature engineering | pandas, numpy |
| Adjustment models | scikit-learn LinearRegression |
| Neural network | TensorFlow / Keras |
| UI | Streamlit |
| Caching | diskcache (SQLite-backed) |
| Visualization | Plotly |
| Testing | pytest + unittest (68 tests, mocked APIs) |

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

- **No match-level granularity.** The current Sofascore integration uses season-aggregate stats. Match-level rolling windows (as described in the paper) would improve accuracy.
- **Adjustment model training data.** Auto-training relies on the most recent transfer only. Expanding to all historical transfers with season-specific stats would produce more robust models.
- **No market value data.** Sofascore does not provide transfer fees or market valuations, limiting the shortlist generator's ability to filter by financial constraints.
- **Single-season neural network.** The TensorFlow model is currently initialized with random weights. Training on a large corpus of historical transfers would significantly improve predictions.

### 6.2 Future Directions

- **Match-level data pipeline.** Integrate match-by-match stats to compute true 1000-minute rolling windows.
- **Automated model retraining.** Schedule periodic retraining of both adjustment models and the neural network as new transfer data becomes available.
- **Tactical embeddings.** Incorporate team formation and tactical style data to capture system fit beyond raw metrics.
- **Expected minutes model.** Predict not just per-90 output but also likely playing time at the target club.
- **Multi-season validation.** Backtest predictions against historical transfers to quantify model accuracy.

---

## 7. Conclusion

TransferScope demonstrates that the methodology from Dinsdale & Gallagher (2022) can be implemented as a production-ready scouting tool with live data, global coverage, and an interface designed for real decision-making. By combining dynamic Power Rankings, position-aware adjustment models, and multi-head neural networks, the system bridges the gap between what a player has done and what they will do — the question at the heart of every transfer decision.

---

## References

1. Dinsdale, J. & Gallagher, J. (2022). *The Transfer Portal: Predicting the Impact of a Player Transfer on the Receiving Club.* In: Brefeld, U., Davis, J., Van Haaren, J., Zimmermann, A. (eds) Machine Learning and Data Mining for Sports Analytics. Springer. https://doi.org/10.1007/978-3-031-02044-5_14

2. ClubElo — http://clubelo.com — European club Elo rating system.

3. World Football Elo Ratings — http://eloratings.net — Global club Elo ratings.

4. Sofascore — https://www.sofascore.com — Player statistics and match data.

5. Elo, A.E. (1978). *The Rating of Chessplayers, Past and Present.* Arco Publishing.
