# TransferScope

**Football transfer intelligence platform that predicts player performance at a new club, generates scouting shortlists, and validates transfer rumours.**

Built on the methodology from *Dinsdale & Gallagher (2022) — "The Transfer Portal"*. Designed for Arsenal scouting, but works for any player, any club, any league — including South America, MLS, and Asia.

---

## What It Does

TransferScope answers three questions every sporting director asks:

| Tool | Question | Paper Reference |
|---|---|---|
| **Transfer Impact** | "How will this player's stats change at our club?" | Fig 1 — predicted performance change dashboard |
| **Shortlist Generator** | "Who are the best replacements for this player across all leagues?" | Fig 2 — replacement shortlist |
| **Hot or Not** | "Is this transfer rumour actually a good move?" | Section 5 — quick rumour validator |

### Transfer Impact
Enter a player and a target club. TransferScope predicts how each of 13 core per-90 metrics will shift based on the difference in team strength, league quality, and playing style. Displays confidence indicators (Red / Amber / Green) based on data availability.

### Shortlist Generator
Select a player to replace and weight the metrics that matter. TransferScope scans players across 20+ leagues, scores them by weighted similarity against predicted performance, and returns a ranked shortlist with filters for age, position, league, minutes played, and club power ranking.

### Hot or Not
Paste a transfer rumour. Get an instant HOT / TEPID / NOT verdict backed by predicted metric changes, power ranking context, and the player's transfer history.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Transfer    │  │  Shortlist   │  │    Hot or Not      │  │
│  │  Impact      │  │  Generator   │  │                    │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘  │
│         │                 │                    │              │
│  ┌──────┴─────────────────┴────────────────────┴──────────┐  │
│  │           Plotly Components (Dark Theme)                │  │
│  │  swarm_plot · metric_bar · power_ranking_chart          │  │
│  └────────────────────────┬───────────────────────────────┘  │
└───────────────────────────┼──────────────────────────────────┘
                            │
┌───────────────────────────┼──────────────────────────────────┐
│                     Backend Pipeline                          │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │              TensorFlow Transfer Portal                 │  │
│  │   4-group multi-head neural network (43 features)       │  │
│  │   Shooting · Passing · Dribbling · Defending            │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │              sklearn Adjustment Models                   │  │
│  │   Team adjustment (13 models)                           │  │
│  │   Player adjustment (13 models × position)              │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │              Feature Engineering                         │  │
│  │   1000-min player rolling windows                       │  │
│  │   3000-min team rolling windows                         │  │
│  │   Prior blend for low-data players                      │  │
│  │   Dynamic Power Rankings (0–100 global)                 │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │              Data Sources                                │  │
│  │   Sofascore API → player stats, team stats, transfers   │  │
│  │   ClubElo (soccerdata) → European Elo ratings           │  │
│  │   WorldFootballElo → non-European Elo ratings           │  │
│  │   All calls routed through diskcache layer              │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
transferscope/
├── app.py                              # Streamlit entry point
├── backend/
│   ├── data/
│   │   ├── sofascore_client.py         # Sofascore API — player stats, search, transfers
│   │   ├── clubelo_client.py           # ClubElo wrapper via soccerdata (Europe)
│   │   ├── worldfootballelo_client.py  # WorldFootballElo scraper (global fallback)
│   │   ├── elo_router.py              # Routes club to correct Elo source
│   │   └── cache.py                    # diskcache layer — all external calls go through here
│   ├── features/
│   │   ├── rolling_windows.py          # 1000-min player / 3000-min team rolling averages
│   │   ├── power_rankings.py           # Dynamic league Elo + 0-100 normalization
│   │   └── adjustment_models.py        # sklearn priors + auto-training from transfer history
│   ├── models/
│   │   ├── transfer_portal.py          # TensorFlow multi-head NN (4 groups, 13 outputs)
│   │   └── shortlist_scorer.py         # Weighted similarity scoring for shortlists
│   └── utils/
│       └── league_registry.py          # League ID mappings across all data sources
├── frontend/
│   ├── pages/
│   │   ├── transfer_impact.py          # Predicted performance change dashboard
│   │   ├── shortlist_generator.py      # Multi-league replacement shortlist
│   │   └── hot_or_not.py              # Quick rumour validator
│   ├── components/
│   │   ├── swarm_plot.py              # Player vs league/team context strip plots
│   │   ├── power_ranking_chart.py      # Before/after team Power Rankings
│   │   └── metric_bar.py              # Horizontal bar: predicted % change per metric
│   └── theme.py                        # Tactical Noir dark theme system
├── tests/                              # 68 unit tests with mocked API calls
├── data/
│   ├── cache/                          # diskcache files (gitignored)
│   └── models/                         # Saved model weights (gitignored)
├── CLAUDE.md                           # AI development context file
├── WHITEPAPER.md                       # Project white paper
├── METHODOLOGY.md                      # Technical methodology
└── requirements.txt
```

---

## Quick Start

### Requirements

- Python 3.11+
- ~2 GB disk for dependencies (TensorFlow)

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

### Run Tests

```bash
python -m pytest tests/ -v
```

All 68 tests use mocked API responses, so they run offline with no network calls.

---

## Data Sources

| Source | Coverage | Access Method |
|---|---|---|
| **Sofascore** | Player stats, team stats, transfers, seasons — all leagues on sofascore.com | REST API via `requests` |
| **ClubElo** | Elo ratings for ~600 European clubs | `soccerdata` Python package |
| **WorldFootballElo** | Elo ratings for clubs worldwide (eloratings.net) | HTTP scrape |

All API calls are routed through a `diskcache` SQLite layer (`backend/data/cache.py`) with configurable TTLs. Player stats cache for 1 day, search results for 7 days, Elo ratings for 1 day.

---

## The 13 Core Metrics

All metrics are stored and displayed as **per-90 minute** values. Never raw totals.

| # | Metric | Description |
|---|---|---|
| 1 | xG | Expected goals |
| 2 | xA | Expected assists |
| 3 | Shots | Total shots |
| 4 | Take-ons | Successful dribbles |
| 5 | Crosses | Accurate crosses |
| 6 | Penalty area entries | Touches in opposition box |
| 7 | Total passes | Accurate passes |
| 8 | Short passes | Pass completion % (proxy) |
| 9 | Long passes | Accurate long balls |
| 10 | Passes in attacking third | Chances created / key passes |
| 11 | Defensive actions (own third) | Clearances |
| 12 | Defensive actions (mid third) | Interceptions |
| 13 | Defensive actions (att third) | Possession won in final third |

Plus 10 additional metrics: xGOT, npxG, dispossessed, duels won %, aerial duels won %, recoveries, fouls won, touches, goals conceded on pitch, xG against on pitch.

---

## League Coverage

**20 leagues across 4 continents:**

- **Europe:** Premier League, Championship, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira Liga, Belgian Pro League, Süper Lig
- **South America:** Brasileirão Série A & B, Argentine Primera, Colombian Primera A, Chilean Primera, Uruguayan Primera, Ecuadorian Serie A
- **North America:** MLS
- **Asia:** Saudi Pro League, J-League

Any league available on Sofascore can be added by extending the league registry.

---

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| Sofascore over FotMob | Broader API: team search, transfer history, season selector, league-wide stats |
| ClubElo + WorldFootballElo over static tier tables | Dynamic, global, faithful to the paper's methodology |
| Dynamic league Elo from team mean | Updates automatically, no manual tier maintenance |
| Streamlit over FastAPI + React | Speed of build; sufficient for a personal scouting tool |
| diskcache over Redis | Local tool; SQLite-backed cache is enough |
| All stats per-90 | Consistent, comparable, position-agnostic |

---

## References

- Dinsdale, J. & Gallagher, J. (2022). *The Transfer Portal: Predicting the Impact of a Player Transfer on the Receiving Club.* [Paper](https://doi.org/10.1007/978-3-031-02044-5_14)
- ClubElo: [clubelo.com](http://clubelo.com)
- WorldFootballElo: [eloratings.net](http://eloratings.net)
- Sofascore: [sofascore.com](https://www.sofascore.com)

---

## License

This project is for personal and educational use. Not affiliated with Sofascore, ClubElo, or any football club.
