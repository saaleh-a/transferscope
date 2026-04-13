# TransferScope

**Football transfer intelligence platform that predicts player performance at a new club, generates scouting shortlists, and validates transfer rumours.**

Built on the methodology from *Dinsdale & Gallagher (2022) — "The Transfer Portal"*. Designed for Arsenal scouting, but works for any player, any club, any league — including South America, MLS, and Asia.

> **In plain English:** You type in a player's name and the club you want to send them to. TransferScope tells you how their stats will change — will they score more? Create fewer chances? Defend better? It also finds replacement players across 51 leagues and gives you a quick "hot or not" verdict on transfer rumours. Think of it like a football version of a "what if?" simulator, powered by maths instead of guesswork.

---

## 🧒 ELI5 (Explain Like I'm 5)

Imagine you have a favourite football player — let's say he plays for a small team and scores lots of goals. Now imagine a really big, famous club wants to buy him. Will he still score lots of goals at the new club? Maybe he'll score *more* because the big club creates more chances. Or maybe he'll score *fewer* because the league is harder. It's really hard to know!

TransferScope is like a crystal ball for football transfers. You tell it "I want to move this player to that club" and it uses maths to figure out what will happen to his stats. It knows how strong every team is, how hard every league is, and what kind of football each team plays. It's looked at thousands of real transfers to learn the patterns.

It's also like a scout's assistant. If your best player gets injured and you need a replacement, TransferScope searches through players in 51 different leagues around the world to find the ones who play most like him. And if you see a transfer rumour on the news, you can type it in and get a quick "thumbs up" or "thumbs down" based on the maths.

The "brain" of the system is a neural network — a computer programme that learns from data, like how you learn from experience. It has six specialist brains: one for shooting, one for creating chances, one for passing, one for crossing, one for dribbling, and one for defending. Each brain only looks at the stats it needs, which makes it really good at its job.

---

## ⚡ TL;DR

- **What:** Football transfer intelligence platform — predicts how a player's per-90 stats change when they move clubs
- **How:** 6-group dual-head neural network (TensorFlow) + sklearn adjustment models, trained on thousands of real transfers
- **Input:** 94-dimensional feature vector per player (stats, team strength, league quality, position, Elo ratings, etc.)
- **Output:** Predicted post-transfer per-90 for 13 core metrics (xG, xA, shots, dribbles, crosses, passes, etc.)
- **Architecture:** Each of 6 groups (shooting, creation, distribution, crossing, dribbling, defending) has its own model with per-group hidden layers, dropout, L2, and Huber loss delta
- **Direction head:** Each model also predicts P(post > pre) via sigmoid — used for direction-aware shrinkage and sign-flipping
- **Ensemble:** 3 seeds per group, predictions averaged before post-processing
- **Training:** 11 default leagues, 5 seasons back, warmup + cosine annealing LR, temporal split with player overlap removal
- **Data:** Sofascore (stats), ClubElo + WorldFootballElo (Elo), Opta (inference power rankings), StatsBomb (spatial), REEP (~45K team aliases)
- **Frontend:** 6 Streamlit pages — Transfer Impact, Shortlist Generator, Hot or Not, Backtest Validator, Diagnostics, About
- **Tests:** 689 automated tests across 27 files, all mocked (no network needed)
- **Coverage:** 51 leagues across 5 continents
- **Shrinkage:** DELTA_SHRINKAGE=0.90 global, plus per-metric shrinkage (0.80 for dribbles → 0.96 for crosses)
- **Fallback:** Paper heuristic prediction when no trained model exists

---

## What It Does

TransferScope answers three questions every sporting director asks:

| Tool | Question |
|---|---|
| **Transfer Impact** | "How will this player's stats change at our club?" |
| **Shortlist Generator** | "Who are the best replacements for this player across all leagues?" |
| **Hot or Not** | "Is this transfer rumour actually a good move?" |
| **About & Methodology** | "How does this work? What leagues are covered?" |
| **Backtest Validator** | "How accurate are the predictions vs actual outcomes?" |
| **Diagnostics** | "Is everything working? What's the cache and data source status?" |

### Transfer Impact

Enter a player and a target club. TransferScope predicts how each of 13 core per-90 metrics will shift based on the difference in team strength, league quality, and playing style. The system simulates the player at **both** their current and target clubs (per the paper), then compares the two model outputs. Displays confidence indicators (Red / Amber / Green) based on data availability.

> **In plain English:** Pick a player (say, a winger from Ajax) and pick a club (say, Arsenal). The tool simulates the player at both clubs using the same model, then calculates: "If this guy moves to Arsenal, his expected goals will go up by 15%, his chance creation will drop by 8%..." and so on for 13 different stats. Crucially, some stats **can go up even when moving to a harder league** if the target team's style suits the player — e.g., a crossing winger moving to a team that plays wide will see crosses increase. It also shows you a traffic light — green means "we have plenty of data, trust this", red means "this player hasn't played much, take it with a pinch of salt."

### Shortlist Generator

Select a player to replace and weight the metrics that matter. TransferScope scans players across 51 leagues (defaulting to the Big 5 European leagues for reliability — scanning too many leagues triggers Sofascore rate-limiting), clusters candidates by playing style using k-means, scores them by weighted Euclidean distance to the reference player (with a 15% same-cluster bonus), and returns a ranked shortlist with filters for age, position, league, minutes played, and club power ranking. The player's own league is always scanned first. A 1.5-second delay between league API calls prevents rate-limiting (403/429 errors) that previously caused 0 results.

> **In plain English:** Say Saka gets injured and you need a replacement right winger. You tell TransferScope which stats matter most to you (e.g. "I care a lot about chance creation and dribbling, less about defensive work"). It then searches through thousands of players across major leagues worldwide, groups them by playing style (using machine learning clustering), and ranks them by how closely they match what you need — with a bonus for players who play a similar style to the reference. You can filter by age, league, how much they've played, etc. The search starts with the player's own league (most reliable) and adds the Big 5 by default — you can expand to more leagues manually.

### Hot or Not

Paste a transfer rumour. Get an instant HOT / TEPID / NOT verdict backed by predicted metric changes, power ranking context, and the player's transfer history. The verdict uses position-aware weighting (offensive metrics matter more for forwards, defensive for defenders) and opposition quality modelling (weaker league = easier opponents). Shows UNKNOWN when insufficient data is available.

> **In plain English:** You read a rumour — "Osimhen to Arsenal." You type it in, press a button, and get a big verdict: HOT (good move), TEPID (meh), or NOT (bad move). It shows you the top 3 stats that would change, a summary of what improves vs. declines, and the player's entire transfer history.

---

## How It Works (The Short Version)

```
1. COLLECT DATA         →  Player stats from Sofascore, club strength ratings from Opta Power Rankings (inference) / ClubElo + WorldFootballElo (training), spatial data from StatsBomb
2. CRUNCH NUMBERS        →  Rolling averages, league quality scores, team strength comparisons
3. PREDICT              →  Neural network + adjustment models predict stats at the new club
4. SHOW RESULTS         →  Charts, tables, and verdicts in a dark-themed web app
```

> **In plain English:** It's like a pipeline in a factory. Raw materials (player data, club ratings) go in one end. They pass through several machines (math models that figure out how league difficulty and team quality affect a player). Out the other end comes a prediction: "Here's what this player will produce at their new club."

---

## Architecture (Technical)

```
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Transfer    │  │  Shortlist   │  │    Hot or Not      │  │
│  │  Impact      │  │  Generator   │  │                    │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘  │
│         │                 │                    │              │
│  ┌──────┴─────────────────┴────────────────────┴──────────┐  │
│  │           Plotly Charts (Dark Theme)                     │  │
│  │  swarm_plot · metric_bar · power_ranking_chart           │  │
│  └────────────────────────┬───────────────────────────────┘  │
└───────────────────────────┼──────────────────────────────────┘
                            │
┌───────────────────────────┼──────────────────────────────────┐
│                     Backend Pipeline                          │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │         TensorFlow Neural Network (the brain)           │  │
│  │   6 model groups · 94 input features · 13 predictions   │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │         sklearn Adjustment Models (the tuners)          │  │
│  │   Team adjustment (13 models)                           │  │
│  │   Player adjustment (13 models × position)              │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │         Feature Engineering (the prep kitchen)          │  │
│  │   Rolling averages · Power Rankings · Prior blending    │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │         Data Sources (the raw ingredients)              │  │
│  │   Sofascore · ClubElo · WorldFootballElo                │  │
│  │   REEP · StatsBomb · football-data.co.uk                │  │
│  │   All calls cached locally to avoid hammering APIs      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
transferscope/
├── app.py                              # Streamlit entry point
├── backend/
│   ├── data/                           # Talks to external data sources
│   │   ├── sofascore_client.py         # Player stats, search, transfers, seasons, match logs
│   │   ├── clubelo_client.py           # European club Elo ratings
│   │   ├── worldfootballelo_client.py  # Global club Elo ratings (non-Europe)
│   │   ├── elo_router.py              # Picks the right Elo source for each club
│   │   ├── reep_registry.py           # REEP open data — ~45K team aliases for fuzzy matching
│   │   ├── statsbomb_client.py        # StatsBomb spatial data — shots, passes, heatmaps
│   │   ├── footballdata_client.py     # football-data.co.uk match CSVs for calibration
│   │   └── cache.py                    # Stores API results locally so we don't re-fetch
│   ├── features/                       # Turns raw data into model-ready numbers
│   │   ├── rolling_windows.py          # Recent-form averages (last ~11 games)
│   │   ├── power_rankings.py           # "How good is this team/league?" scores
│   │   └── adjustment_models.py        # Paper-aligned heuristic + sklearn adjustment models
│   ├── models/                         # The prediction engines
│   │   ├── transfer_portal.py          # Neural network that predicts post-transfer stats
│   │   ├── shortlist_scorer.py         # K-means clustering + weighted Euclidean distance scoring
│   │   ├── training_pipeline.py        # End-to-end training: transfer discovery → sklearn + TF fit
│   │   └── backtester.py              # Compares predictions against actual post-transfer stats
│   └── utils/
│       └── league_registry.py          # Master list of all 51 leagues and their IDs
├── frontend/
│   ├── pages/                          # The six main screens
│   │   ├── transfer_impact.py          # "What happens if this player moves here?"
│   │   ├── shortlist_generator.py      # "Find me a replacement across all leagues"
│   │   ├── hot_or_not.py              # "Is this rumour any good?"
│   │   ├── backtest_validator.py       # "How accurate were past predictions?"
│   │   ├── diagnostics.py             # System health, data sources, cache status
│   │   └── about.py                   # Methodology, league coverage, and limitations
│   ├── components/                     # Reusable chart widgets
│   │   ├── swarm_plot.py              # Shows where a player ranks in their league
│   │   ├── power_ranking_chart.py      # Before/after club strength timeline
│   │   ├── metric_bar.py              # Bar chart of predicted stat changes
│   │   ├── pitch_viz.py               # Shot maps, pass networks, heatmaps (mplsoccer)
│   │   └── player_pizza.py            # Player pizza/radar chart
│   ├── constants.py                    # Shared metric display labels
│   └── theme.py                        # The dark "Tactical Noir" visual design
├── tests/                              # 689 automated tests (no internet needed)
├── scripts/
│   └── check_training_ready.py         # Training readiness verification
├── data/
│   ├── cache/                          # Saved API responses (not in git)
│   └── models/                         # Saved model weights (not in git)
├── ARCHITECTURE.md                     # Full architecture reference (design decisions, metrics, models)
├── WHITEPAPER.md                       # Project white paper
├── METHODOLOGY.md                      # Technical methodology
├── ONBOARDING.md                       # Developer onboarding guide
└── requirements.txt                    # Python package list
```

---

## Quick Start

### Requirements

- Python 3.12
- ~2 GB disk for dependencies (TensorFlow is big)

### Installation

```bash
git clone https://github.com/saaleh-a/transferscope.git
cd transferscope
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. No API keys required — all data sources are publicly accessible.

> **In plain English:** Clone the code, install dependencies, run one command, and a web app opens in your browser. No accounts or passwords needed.

### Run Tests

```bash
python -m pytest tests/ --ignore=tests/test_seleniumbase.py -v
```

All 689 tests use mocked API responses, so they run offline with no network calls.

---

## Data Sources

| Source | What it gives us | Plain English |
|---|---|---|
| **Sofascore** | Player stats, team rosters, transfer history, seasons, match logs | "How many goals/assists/passes did this player make?" |
| **Opta Power Rankings** | Team strength rankings (inference only) | "How strong is this team right now?" (used for live predictions) |
| **ClubElo** | Elo ratings for ~600 European clubs (training/historical) | "How strong was this European club back then?" |
| **WorldFootballElo** | Elo ratings for clubs worldwide (training/historical) | "How strong was this Brazilian/MLS/Saudi club?" |
| **REEP Register** | Team alias mappings (~45K clubs worldwide) | "What other names does this club go by?" |
| **StatsBomb** | Spatial data — shot locations, pass networks, heatmaps | "Where on the pitch does this player operate?" |
| **WhoScored** | Spatial data fallback when StatsBomb doesn't cover the match | "Backup pitch visualisation data" |
| **football-data.co.uk** | Match-level CSVs for coefficient calibration | "How do league playing styles compare?" |

All API calls are routed through a local cache (`backend/data/cache.py`). Player stats cache for 1 day, search results for 7 days, Elo ratings for 1 day. This means the app stays fast and doesn't repeatedly hit external servers. REEP team aliases cache for 7 days.

> **In plain English:** Elo ratings are like a score for how good a team is — the same system chess uses to rank players. A team gains points when they win and loses points when they lose. We use two different Elo providers because no single one covers the entire world.

---

## The 13 Core Metrics

All metrics are stored and displayed as **per-90 minute** values — never raw totals.

> **In plain English:** "Per-90" means "for every 90 minutes played." This makes it fair to compare a player who played 3,000 minutes to one who played 900. Instead of saying "he scored 10 goals" (which depends on how much he played), we say "he scores 0.45 goals per 90 minutes" (which doesn't).

| # | Metric | What it measures |
|---|---|---|
| 1 | xG | How many goals a player "should" score based on shot quality |
| 2 | xA | How many assists a player "should" get based on pass quality |
| 3 | Shots | How often they shoot |
| 4 | Take-ons | How often they beat a defender with a dribble |
| 5 | Crosses | How often they deliver accurate crosses |
| 6 | Penalty area entries | How often they get the ball into the box |
| 7 | Total passes | How many accurate passes they complete |
| 8 | Short passes | Pass completion % (how reliable their passing is) |
| 9 | Long passes | Accurate long balls (switching play, diagonals) |
| 10 | Passes in attacking third | Chances created (the "killer ball") |
| 11 | Defensive actions (own third) | Clearances (last-ditch defending) |
| 12 | Defensive actions (mid third) | Interceptions (reading the game) |
| 13 | Defensive actions (att third) | Winning the ball high up the pitch (pressing) |

Plus 10 additional metrics: xGOT, npxG, dispossessed, duels won %, aerial duels won %, recoveries, fouls won, touches, goals conceded on pitch, xG against on pitch.

---

## League Coverage

**51 leagues across 5 continents:**

- **Europe (39):** Premier League, Championship, La Liga, La Liga 2, Bundesliga, 2. Bundesliga, Serie A, Serie B, Ligue 1, Ligue 2, Eredivisie, Primeira Liga, Belgian Pro League, Süper Lig, Scottish Premiership, Austrian Bundesliga, Swiss Super League, Greek Super League, Czech First League, Danish Superliga, Croatian 1. HNL, Serbian Super Liga, Norwegian Eliteserien, Swedish Allsvenskan, Polish Ekstraklasa, Romanian Liga I, Ukrainian Premier League, Russian Premier League, Slovak Super Liga, Slovenian PrvaLiga, Bosnian Premier Liga, Israeli Premier League, Kazakhstan Premier League, Icelandic Úrvalsdeild, League of Ireland Premier Division, Welsh Premier League, Georgian Erovnuli Liga, Bulgarian/Hungarian/Cypriot/Finnish leagues
- **South America (7):** Brasileirão Série A & B, Argentine Primera, Colombian Primera A, Chilean Primera, Uruguayan Primera, Ecuadorian Serie A
- **North America (1):** MLS
- **Asia (2):** Saudi Pro League, J-League
- **Africa (2):** Egyptian Premier League, South African Premier Division

Any league available on Sofascore can be added by extending the league registry.

---

## Key Design Decisions

| Decision | Why | Plain English |
|---|---|---|
| Sofascore over FotMob | Team search, transfer history, season selector, league-wide stats, team-position averages | Sofascore has more features we need |
| ClubElo + WorldFootballElo | Dynamic, global, faithful to the paper | Two data sources cover the whole world |
| Dynamic league Elo from team mean | Updates automatically, no manual maintenance | League quality is calculated fresh every day, not hard-coded |
| Dual simulation | Predict at both current and target clubs, compare model-vs-model (paper Section 4) | Both predictions use the same model, reducing noise |
| Per-metric style weights | `_TEAM_INFLUENCE`, `_ABILITY_SENSITIVITY`, `_OPP_QUALITY_SENS`, `_LEAGUE_STYLE_COEFF` keyed per-metric | Different stats respond differently to team/league/opposition changes |
| Asymmetric calibration | Less damping for downgrades, more for upgrades; elite protection halved for downgrades | Extreme transfers produce realistically large changes |
| Multi-tournament fallback | When primary tournament returns 0 minutes, try all team tournaments | Fixes data loading for players in cups/European competitions |
| Position-aware verdict | Hot or Not weights offensive metrics 1.5× for forwards, defensive for defenders | More accurate verdicts for different player types |
| K-means shortlist scoring | Cluster candidates by playing style, 15% same-cluster bonus, weighted Euclidean distance | Finds replacements with similar playing profiles, not just similar raw numbers |
| Shortlist rate-limit protection | 1.5s inter-league delay, Big 5 default, player's own league first | Prevents Sofascore 403/429 errors that caused 0 results when scanning too many leagues |
| None-passthrough filters | Candidates with unknown age/minutes pass through filters instead of being excluded | Sparse API data shouldn't silently drop valid candidates |
| Per-group feature subsets | Shooting 41, Creation 37, Distribution 32, Crossing 26, Dribbling 26, Defending 36 features (94 total) | Each model group only sees relevant features, reducing noise |
| 3-step team name matching | Exact → accent-normalized → fuzzy (502 abbreviation aliases + 531 ClubElo mappings + dynamic REEP aliases) | Reliably matches team names across ClubElo, WorldFootballElo, and Sofascore |
| Streamlit | Fast to build; sufficient for a personal tool | Web app framework that gets us a UI without a separate frontend team |
| diskcache | Local tool, SQLite is enough | Simple on-disk cache, no need for a database server |
| Dynamic REEP alias resolution | At runtime, cross-links ~45K clubs from REEP teams.csv for fuzzy matching | Never goes stale — augments hardcoded aliases automatically |
| StatsBomb spatial data | Shot maps, pass networks, heatmaps in Transfer Impact | Visual context beyond raw statistics |
| Coefficient calibration | football-data.co.uk match CSVs refine style coefficients | Data-driven coefficient tuning, not just defaults |
| All stats per-90 | Consistent, comparable, position-agnostic | Fair comparisons regardless of minutes played |
| Dual-head output | Regression (scaled delta) + direction (sigmoid P(post>pre)) per metric | Direction head catches sign errors and modulates shrinkage |
| Direction-aware shrinkage | DELTA_SHRINKAGE=0.90, modulated ±30% by direction confidence | High-confidence direction signals get less conservative shrinkage |
| Log-scale targets | Shooting and crossing groups use log-ratio targets | Multiplicative changes are more natural for count-based stats |
| Ensemble averaging | 3 seeds per group, averaged before post-processing | Reduces variance and single-seed bias |
| Non-transfer samples | 35% of training budget is same-club controls | Stabilises learning by anchoring on no-change baseline |

---

## Neural Network Architecture

TransferScope uses a **6-group dual-head neural network**. Each group specialises in a different type of football output:

| Group | Metrics | Hidden Layers | Dropout | L2 | Huber δ | Normalisation |
|-------|---------|---------------|---------|-----|---------|---------------|
| **Shooting** | xG, shots | 64 → 32 | 0.40 | 4e-4 | 0.3 | LayerNorm |
| **Creation** | xA, chances created, touches in opp box | 64 → 32 | 0.35 | 3e-4 | 0.8 | BatchNorm |
| **Distribution** | passes, pass completion %, long balls | 96 → 48 | 0.35 | 3e-4 | 1.5 | BatchNorm |
| **Crossing** | crosses | 32 → 16 | 0.40 | 4e-4 | 0.3 | BatchNorm |
| **Dribbling** | dribbles | 64 → 32 | 0.40 | 3e-4 | 1.0 | BatchNorm |
| **Defending** | clearances, interceptions, possession won | 96 → 48 | 0.35 | 1.0 | BatchNorm |

> **In plain English:** Instead of one big brain that tries to predict everything, there are six specialist brains. The "shooting brain" only thinks about goals and shots. The "creation brain" only thinks about assists and chance-making. This specialisation makes each brain better at its job. The shooting brain even uses a different kind of maths (LayerNorm instead of BatchNorm) because shooting stats need special handling.

### Feature Vector (94 dimensions)

Each player is described by a 94-number fingerprint:

| Block | Count | What it captures |
|-------|-------|-----------------|
| Core per-90 metrics | 13 | The player's actual stats per 90 min |
| Additional per-90 metrics | 10 | Enrichment stats (duels won %, recoveries, etc.) |
| Team/league abilities | 4 | How strong the current and target teams/leagues are |
| Raw Elo scores | 2 | Current and target club Elo ratings |
| REEP metadata | 2 | Player height (cm) and age |
| Team-position per-90 | 26 | What players in this position typically do at both clubs (13 × 2) |
| Interaction features | 3 | Ability gap, gap², league gap |
| Relative dominance | 3 | How dominant each team is in their league |
| League-normalised stats | 13 | Player stats divided by league average |
| League mean ratios | 13 | Ratio of target league mean to source league mean |
| Position one-hot | 4 | F, M, D, or G |
| Minutes per match | 1 | Starter vs substitute indicator |

### Post-Processing Pipeline

After the neural network produces raw predictions, several safety mechanisms are applied:

1. **Inverse scaling** — Convert from StandardScaler space back to per-90
2. **Direction-aware shrinkage** — `DELTA_SHRINKAGE=0.90` modulated ±30% based on direction head confidence (`DIRECTION_SHRINKAGE_ALPHA=0.30`)
3. **Direction gating** — If the direction head says P(post > pre) < 0.30 but the regression says positive delta (or vice versa), flip the sign (`DIRECTION_FLIP_THRESHOLD=0.70`)
4. **Per-metric shrinkage** — Additional metric-specific shrinkage (dribbles: 0.80, interceptions: 0.82, xA: 0.85, ... crosses: 0.96)
5. **Delta clipping** — Clip to plausible ranges using per-metric floors calibrated from historical P95-P99 distributions

---

## Training Pipeline

The training pipeline (`backend/models/training_pipeline.py`, ~3,117 lines) runs end-to-end:

```
1. Discover transfers    →  Scan 11 leagues × 5 seasons, find all transfers with ≥450 min pre/post
2. Discover non-transfers →  Find same-club controls (35% of training budget)
3. Build feature matrices →  94D feature vectors + 13D label vectors for each sample
4. Temporal split         →  Train/Val/Test split by date (removes player overlap)
5. Train adjustment models →  13 LinearRegression (team) + 4×13 Ridge (player by position)
6. Train neural network   →  6 group models × 3 ensemble seeds
7. Backtest              →  Evaluate on held-out test set
```

**Key training hyperparameters:**
- LR schedule: Linear warmup (10 epochs, 1e-5 → 5e-4) then cosine annealing (to 1e-5 over 150 epochs)
- EarlyStopping: patience=15, min_delta=0.001
- Batch size: 32
- Validation split: 15%
- xG zero-masking: Samples missing xG data get zero weight in shooting group only

**Run training:**
```bash
python -m backend.models.training_pipeline --leagues ENG1,ESP1,GER1 --seasons-back 5
```

**CLI flags:** `--skip-discovery`, `--skip-build`, `--skip-training`, `--val-ratio`, `--test-ratio`, `--api-delay`

---

## References

- Dinsdale, J. & Gallagher, J. (2022). *The Transfer Portal: Predicting the Impact of a Player Transfer on the Receiving Club.* [Paper](https://doi.org/10.48550/arXiv.2201.11533)
- ClubElo: [clubelo.com](http://clubelo.com)
- WorldFootballElo: [eloratings.net](http://eloratings.net)
- Sofascore: [sofascore.com](https://www.sofascore.com)
- REEP Register: [github.com/transfermarkt/reep](https://github.com/transfermarkt/reep)
- StatsBomb Open Data: [statsbomb.com/what-we-do/open-data](https://statsbomb.com/what-we-do/open-data/)
- football-data.co.uk: [football-data.co.uk](https://www.football-data.co.uk)

---

## License

This project is for personal and educational use. Not affiliated with Sofascore, ClubElo, or any football club.
