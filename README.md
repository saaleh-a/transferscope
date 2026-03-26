# TransferScope

**Football transfer intelligence platform that predicts player performance at a new club, generates scouting shortlists, and validates transfer rumours.**

Built on the methodology from *Dinsdale & Gallagher (2022) — "The Transfer Portal"*. Designed for Arsenal scouting, but works for any player, any club, any league — including South America, MLS, and Asia.

> **In plain English:** You type in a player's name and the club you want to send them to. TransferScope tells you how their stats will change — will they score more? Create fewer chances? Defend better? It also finds replacement players across 37+ leagues and gives you a quick "hot or not" verdict on transfer rumours. Think of it like a football version of a "what if?" simulator, powered by maths instead of guesswork.

---

## What It Does

TransferScope answers three questions every sporting director asks:

| Tool | Question |
|---|---|
| **Transfer Impact** | "How will this player's stats change at our club?" |
| **Shortlist Generator** | "Who are the best replacements for this player across all leagues?" |
| **Hot or Not** | "Is this transfer rumour actually a good move?" |
| **About & Methodology** | "How does this work? What leagues are covered?" |

### Transfer Impact

Enter a player and a target club. TransferScope predicts how each of 13 core per-90 metrics will shift based on the difference in team strength, league quality, and playing style. The system simulates the player at **both** their current and target clubs (per the paper), then compares the two model outputs. Displays confidence indicators (Red / Amber / Green) based on data availability.

> **In plain English:** Pick a player (say, a winger from Ajax) and pick a club (say, Arsenal). The tool simulates the player at both clubs using the same model, then calculates: "If this guy moves to Arsenal, his expected goals will go up by 15%, his chance creation will drop by 8%..." and so on for 13 different stats. Crucially, some stats **can go up even when moving to a harder league** if the target team's style suits the player — e.g., a crossing winger moving to a team that plays wide will see crosses increase. It also shows you a traffic light — green means "we have plenty of data, trust this", red means "this player hasn't played much, take it with a pinch of salt."

### Shortlist Generator

Select a player to replace and weight the metrics that matter. TransferScope scans players across 37+ leagues (defaulting to 11 major leagues for speed), clusters candidates by playing style using k-means, scores them by weighted Euclidean distance to the reference player (with a 15% same-cluster bonus), and returns a ranked shortlist with filters for age, position, league, minutes played, and club power ranking.

> **In plain English:** Say Saka gets injured and you need a replacement right winger. You tell TransferScope which stats matter most to you (e.g. "I care a lot about chance creation and dribbling, less about defensive work"). It then searches through thousands of players across major leagues worldwide, groups them by playing style (using machine learning clustering), and ranks them by how closely they match what you need — with a bonus for players who play in a similar style to the reference. You can filter by age, league, how much they've played, etc.

### Hot or Not

Paste a transfer rumour. Get an instant HOT / TEPID / NOT verdict backed by predicted metric changes, power ranking context, and the player's transfer history. The verdict uses position-aware weighting (offensive metrics matter more for forwards, defensive for defenders) and opposition quality modelling (weaker league = easier opponents). Shows UNKNOWN when insufficient data is available.

> **In plain English:** You read a rumour — "Osimhen to Arsenal." You type it in, press a button, and get a big verdict: HOT (good move), TEPID (meh), or NOT (bad move). It shows you the top 3 stats that would change, a summary of what improves vs. declines, and the player's entire transfer history.

---

## How It Works (The Short Version)

```
1. COLLECT DATA         →  Player stats from Sofascore, club strength ratings from Elo systems
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
│  │   4 model groups · 43 input features · 13 predictions   │  │
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
│   │   ├── sofascore_client.py         # Player stats, search, transfers, seasons, match logs, team-position averages
│   │   ├── clubelo_client.py           # European club Elo ratings
│   │   ├── worldfootballelo_client.py  # Global club Elo ratings (non-Europe)
│   │   ├── elo_router.py              # Picks the right Elo source for each club
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
│       └── league_registry.py          # Master list of all 37+ leagues and their IDs
├── frontend/
│   ├── pages/                          # The four main screens
│   │   ├── transfer_impact.py          # "What happens if this player moves here?"
│   │   ├── shortlist_generator.py      # "Find me a replacement across all leagues"
│   │   ├── hot_or_not.py              # "Is this rumour any good?"
│   │   └── about.py                   # Methodology, league coverage, and limitations
│   ├── components/                     # Reusable chart widgets
│   │   ├── swarm_plot.py              # Shows where a player ranks in their league
│   │   ├── power_ranking_chart.py      # Before/after club strength timeline
│   │   └── metric_bar.py              # Bar chart of predicted stat changes
│   └── theme.py                        # The dark "Tactical Noir" visual design
├── tests/                              # 188 automated tests (no internet needed)
├── data/
│   ├── cache/                          # Saved API responses (not in git)
│   └── models/                         # Saved model weights (not in git)
├── CLAUDE.md                           # AI development context
├── WHITEPAPER.md                       # Project white paper
├── METHODOLOGY.md                      # Technical methodology
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
python -m pytest tests/ -v
```

All 188 tests use mocked API responses, so they run offline with no network calls.

---

## Data Sources

| Source | What it gives us | Plain English |
|---|---|---|
| **Sofascore** | Player stats, team rosters, transfer history, seasons, match logs | "How many goals/assists/passes did this player make?" |
| **ClubElo** | Elo ratings for ~600 European clubs | "How strong is this European club right now?" |
| **WorldFootballElo** | Elo ratings for clubs worldwide | "How strong is this Brazilian/MLS/Saudi club?" |

All API calls are routed through a local cache (`backend/data/cache.py`). Player stats cache for 1 day, search results for 7 days, Elo ratings for 1 day. This means the app stays fast and doesn't repeatedly hit external servers.

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

**37+ leagues across 4 continents:**

- **Europe (30+):** Premier League, Championship, La Liga, La Liga 2, Bundesliga, 2. Bundesliga, Serie A, Serie B, Ligue 1, Ligue 2, Eredivisie, Primeira Liga, Belgian Pro League, Süper Lig, Scottish Premiership, Austrian Bundesliga, Swiss Super League, Greek Super League, Czech First League, Danish Superliga, Croatian 1. HNL, Serbian Super Liga, Norwegian Eliteserien, Swedish Allsvenskan, Polish Ekstraklasa, Romanian Liga I, Ukrainian Premier League, Russian Premier League, Bulgarian/Hungarian/Cypriot/Finnish leagues
- **South America (7):** Brasileirão Série A & B, Argentine Primera, Colombian Primera A, Chilean Primera, Uruguayan Primera, Ecuadorian Serie A
- **North America (1):** MLS
- **Asia (2):** Saudi Pro League, J-League

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
| Per-group feature subsets | Shooting 16, Passing 25, Dribbling 7, Defending 13 features | Each model group only sees relevant features, reducing noise |
| 3-step team name matching | Exact → accent-normalized → fuzzy (138 abbreviation aliases) | Reliably matches team names across ClubElo, WorldFootballElo, and Sofascore |
| Streamlit | Fast to build; sufficient for a personal tool | Web app framework that gets us a UI without a separate frontend team |
| diskcache | Local tool, SQLite is enough | Simple on-disk cache, no need for a database server |
| All stats per-90 | Consistent, comparable, position-agnostic | Fair comparisons regardless of minutes played |

---

## References

- Dinsdale, J. & Gallagher, J. (2022). *The Transfer Portal: Predicting the Impact of a Player Transfer on the Receiving Club.* [Paper](https://doi.org/10.48550/arXiv.2201.11533)
- ClubElo: [clubelo.com](http://clubelo.com)
- WorldFootballElo: [eloratings.net](http://eloratings.net)
- Sofascore: [sofascore.com](https://www.sofascore.com)

---

## License

This project is for personal and educational use. Not affiliated with Sofascore, ClubElo, or any football club.
