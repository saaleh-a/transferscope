# TransferScope — Onboarding Guide

---

## SECTION 1 — Repository Overview

### Purpose

TransferScope is a football transfer intelligence platform that predicts how a player's per-90 statistical output will change if they move to a different club. It implements the methodology from the Dinsdale & Gallagher (2022) paper, which decomposes a transfer's impact into team quality, opposition quality, and tactical style signals. The tool produces three outputs: a Transfer Impact dashboard (predicted % change per metric), a Shortlist Generator (ranked replacement candidates via K-means clustering), and a Hot or Not quick rumour validator. Built for Arsenal scouting, but supports any player/club/league on Sofascore.

### Tech Stack

| Technology | Version | Role in the system |
|---|---|---|
| Python | 3.12 | Runtime |
| Streamlit | ≥1.29.0 | Web UI — single-page app with sidebar navigation |
| TensorFlow | ≥2.15.0 | Multi-head neural network (4 model groups, 13 output heads) |
| scikit-learn | ≥1.3.0 | Ridge regression adjustment models, StandardScaler, KMeans clustering |
| pandas | ≥2.1.0 | DataFrame operations for Elo data, rolling windows, training pipeline |
| numpy | ≥1.24.0 | Array operations throughout features and models |
| Plotly | ≥5.18.0 | Interactive charts (metric bars, swarm plots, power ranking timelines) |
| soccerdata | ≥1.7.0 | ClubElo API wrapper (European club Elo ratings) |
| diskcache | ≥5.6.0 | SQLite-backed caching layer for all external API calls |
| requests | ≥2.31.0 | HTTP client fallback for Sofascore and WorldFootballElo |
| python-dotenv | ≥1.0.0 | Environment variable loading |
| statsbombpy | ≥0.6.0 | StatsBomb open data — spatial features (shots, passes, heatmaps) |
| mplsoccer | ≥1.4.0 | Pitch visualization rendering for StatsBomb data |
| curl-cffi | ≥0.7.1 | HTTP client with Cloudflare bypass for Sofascore |
| joblib | ≥1.3.0 | Parallel processing and model serialization |

### Directory Structure

```
transferscope/
├── app.py                              # Streamlit entry point — page routing, cache warmup, model status
├── requirements.txt                    # All Python dependencies (13 packages)
├── .python-version                     # Pins Python 3.12 for Streamlit Cloud
├── ARCHITECTURE.md                     # Architecture reference (design decisions, metrics, models)
├── METHODOLOGY.md                      # Plain-English methodology explanation
├── WHITEPAPER.md                       # Full paper reference
├── ONBOARDING.md                       # This file
├── backend/
│   ├── data/
│   │   ├── sofascore_client.py         # (1567 lines) Sofascore REST API — player search, stats, transfers, match logs
│   │   ├── clubelo_client.py           # (308 lines) ClubElo wrapper — European club Elo (soccerdata + HTTP fallback)
│   │   ├── worldfootballelo_client.py  # (149 lines) eloratings.net scraper — non-European Elo
│   │   ├── elo_router.py              # (77 lines) Routes team to best Elo source, 0-100 normalization
│   │   ├── reep_registry.py           # (190 lines) REEP register — ~45K team aliases for fuzzy matching
│   │   ├── statsbomb_client.py        # (490 lines) StatsBomb spatial data — shots, passes, heatmaps
│   │   ├── footballdata_client.py     # (279 lines) football-data.co.uk match CSVs for calibration
│   │   └── cache.py                   # (87 lines) diskcache wrapper — namespace-based key-value store
│   ├── features/
│   │   ├── rolling_windows.py          # (188 lines) 1000-min player / 3000-min team rolling per-90 averages
│   │   ├── power_rankings.py           # (1801 lines) Daily Elo aggregation, 0-100 normalization, fuzzy team matching
│   │   └── adjustment_models.py        # (1007 lines) sklearn Ridge models + paper_heuristic_predict fallback
│   ├── models/
│   │   ├── transfer_portal.py          # (861 lines) TensorFlow 4-group NN — build, fit, predict, save/load
│   │   ├── shortlist_scorer.py         # (388 lines) KMeans + weighted Euclidean distance candidate ranking
│   │   ├── training_pipeline.py        # (2193 lines) End-to-end: discover transfers → build features → train
│   │   └── backtester.py              # (293 lines) Post-hoc validation against actual post-transfer per-90
│   └── utils/
│       └── league_registry.py          # (483 lines) Central league ID mappings (Sofascore, ClubElo, WorldElo)
├── frontend/
│   ├── theme.py                        # (699 lines) "Tactical Noir" dark theme — CSS, colors, components
│   ├── constants.py                    # (17 lines) Shared metric display labels
│   ├── pages/
│   │   ├── transfer_impact.py          # Transfer Impact dashboard (paper Fig 1)
│   │   ├── shortlist_generator.py      # Replacement shortlist (paper Fig 2)
│   │   ├── hot_or_not.py              # Quick rumour validator (paper Section 5)
│   │   ├── backtest_validator.py       # Backtest predictions vs actual post-transfer stats
│   │   ├── diagnostics.py             # System diagnostics — data source status, cache info
│   │   └── about.py                   # Methodology explanation page
│   └── components/
│       ├── swarm_plot.py              # (142 lines) Player-vs-league strip plots
│       ├── power_ranking_chart.py     # (108 lines) Before/after Elo timeline
│       ├── metric_bar.py             # (193 lines) Diverging horizontal bar chart
│       ├── pitch_viz.py              # (421 lines) Shot maps, pass networks, heatmaps (mplsoccer)
│       └── player_pizza.py           # (239 lines) Player pizza/radar chart
├── tests/                              # 488 tests across 24 files (all mocked, no network)
├── scripts/
│   └── check_training_ready.py         # Training readiness verification
├── data/
│   ├── cache/                          # diskcache SQLite files (gitignored)
│   └── models/                         # Saved TF weights + scalers (gitignored)
```

### Architecture & Data Flow

**Entry point:** `app.py` → Streamlit. On load, it starts a background thread to warm the Elo cache (`power_rankings.compute_daily_rankings()`), checks whether trained TF model weights exist, and renders sidebar navigation.

**User selects a page → data flows through 4 layers:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  FRONTEND (Streamlit Pages)                                         │
│  transfer_impact.py / shortlist_generator.py / hot_or_not.py /      │
│  backtest_validator.py / diagnostics.py                             │
│    ↓ user enters player name + target club                          │
│    ↓ calls sofascore_client.search_player() / search_team()         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  DATA LAYER (backend/data/)                                         │
│  sofascore_client → HTTP GET → Sofascore API → parse → per-90 dict  │
│  clubelo_client   → soccerdata/HTTP → European Elo ratings          │
│  worldfootballelo → HTML scrape → non-European Elo ratings          │
│  elo_router       → pick best source → raw Elo float                │
│  reep_registry    → CSV download → ~45K team aliases                │
│  statsbomb        → statsbombpy → spatial features (shots, passes)  │
│  footballdata     → CSV download → league match profiles            │
│  cache            → all above route through diskcache               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  FEATURE LAYER (backend/features/)                                  │
│  power_rankings   → aggregate all Elo → normalize 0-100 → relative  │
│  rolling_windows  → match logs → 1000-min weighted averages → blend │
│  adjustment_models→ paper_heuristic_predict() OR trained sklearn     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  MODEL LAYER (backend/models/)                                      │
│  transfer_portal  → build_feature_dict() → TF predict() OR fallback │
│  shortlist_scorer → KMeans cluster → weighted Euclidean → rank      │
│  training_pipeline→ discover_transfers → build samples → fit all    │
│  backtester       → compare predictions vs actuals → report         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  FRONTEND RENDERS OUTPUT                                            │
│  metric_bar.py       → diverging horizontal bars (% change/metric)  │
│  swarm_plot.py       → player vs league context strip plots         │
│  power_ranking_chart → dual-line Elo timeline                       │
│  pitch_viz.py        → shot maps, pass networks, heatmaps           │
│  player_pizza.py     → player pizza/radar chart                     │
│  theme.py            → CSS, stat cards, confidence badges           │
└─────────────────────────────────────────────────────────────────────┘
```

**Concrete request path (Transfer Impact page):**
1. User types "Bukayo Saka" and "Real Madrid", clicks Analyse
2. `sofascore_client.search_player("Bukayo Saka")` → player_id, team_id, tournament_id
3. `sofascore_client.get_player_stats(player_id)` → per-90 dict, minutes, position
4. `power_rankings.get_team_ranking("Arsenal")` and `get_team_ranking("Real Madrid")` → normalized scores, relative abilities
5. `sofascore_client.get_team_position_averages(team_id, position)` for both clubs → tactical style data
6. `transfer_portal.build_feature_dict(...)` assembles 46-key feature dict
7. `TransferPortalModel.predict(feature_dict)` → if trained: TF neural net; else: `paper_heuristic_predict()`
8. Same prediction at current club (dual simulation) → compute % changes
8b. StatsBomb spatial data (if available) → shot maps, pass networks, heatmaps
9. Frontend renders: metric bars, confidence badge, swarm plots, power ranking chart

### Files to Read First

1. **`ARCHITECTURE.md`** — The single source of truth. Contains the entire architecture, all design decisions, metric mappings, and model specs. Read this before any code.
2. **`app.py`** — Entry point. Shows page routing, cache warmup, and model status check. 98 lines, fully readable.
3. **`backend/data/sofascore_client.py`** — The data backbone. Every prediction starts with data from here. Focus on `CORE_METRICS` (line 39), `get_player_stats()`, and `_parse_stats()`.
4. **`backend/features/adjustment_models.py`** — The prediction brain. Start at `paper_heuristic_predict()` (line 644) — this is the default prediction path. Then read `_TEAM_INFLUENCE`, `_ABILITY_SENSITIVITY`, `_OPP_QUALITY_SENS` dicts above it.
5. **`backend/features/power_rankings.py`** — The context engine. `compute_daily_rankings()` (line 201) aggregates all Elo data and normalizes globally. `get_team_ranking()` (line 320) resolves team names with fuzzy matching.
6. **`backend/models/transfer_portal.py`** — The ML model. `build_feature_dict()` shows all 46 feature keys. `predict()` (line 271) shows the TF → heuristic fallback chain.
7. **`frontend/pages/transfer_impact.py`** — The main UI page. `render()` shows the full user-facing flow from input to chart output.

---

## SECTION 2 — Atomic ELI5 Breakdown

### Module: `app.py` (Entry Point)

#### What this does
Streamlit entry point that configures the page, warms the cache, checks model status, and routes to the selected sub-page.

#### Why it exists
Streamlit requires a single Python file as the starting point. This file initializes the app state and dispatches to the correct page renderer.

#### Step-by-step execution flow

1. **Configure the browser tab:**
```python
st.set_page_config(
    page_title="TransferScope",
    page_icon="⚽",
    layout="wide",                    # use full browser width
    initial_sidebar_state="expanded",  # sidebar visible by default
)
```

2. **Inject CSS theme:**
```python
from frontend.theme import inject_css
inject_css()  # writes 516 lines of CSS into the page (dark theme, custom fonts)
```

3. **Warm the Elo cache (once per session):**
```python
if "cache_warmed" not in st.session_state:    # only run once
    def _warmup():
        from backend.features import power_rankings
        power_rankings.compute_daily_rankings()   # fetches ClubElo + WorldElo, normalizes
    threading.Thread(target=_warmup, daemon=True).start()  # background so page loads fast
    st.session_state["cache_warmed"] = True
```
*Analogy: like preheating an oven — the first query will need Elo data, so start fetching it now.*

4. **Check if a trained ML model exists:**
```python
_model_check = TransferPortalModel()
if _model_check.is_trained():    # looks for .keras files + feature_scaler.pkl in data/models/
    st.success("✅ Trained model loaded.")
else:
    st.warning("⚠️ No trained model found. Using heuristic fallback.")
```

5. **Render sidebar navigation and route to page:**
```python
page = st.sidebar.radio("Navigation", ["Transfer Impact", "Shortlist Generator", ...])
if page == "Transfer Impact":
    from frontend.pages.transfer_impact import render
    render()   # <-- the entire Transfer Impact page is in this render() function
```

#### Glossary
- **`st.session_state`**: Streamlit's per-user persistent dict that survives page re-runs.
- **`diskcache`**: SQLite-backed key-value store on disk. Persists between app restarts.
- **Heuristic fallback**: When no trained TensorFlow model exists (default state), predictions use a hand-tuned formula (`paper_heuristic_predict`) instead.

---

### Module: `backend/data/sofascore_client.py` (Data Backbone)

#### What this does
HTTP wrapper around Sofascore's public REST API that fetches player stats, team data, transfer histories, and match logs, converting raw totals to per-90 values.

#### Why it exists
Sofascore is the only data source for player statistics. This module centralizes all API calls, handles rate-limiting/retries, and normalizes the inconsistent field names into 23 canonical metrics.

#### Step-by-step execution flow (for `get_player_stats`)

1. **Try the cache first:**
```python
key = cache.make_key("player_stats", str(player_id), str(season))
cached = cache.get(key, max_age=_ONE_DAY)   # 86400 seconds
if cached:           # truthy check — empty dicts don't count
    return cached
```
*Analogy: checking your notebook before calling the library.*

2. **Discover which tournament/season to query:**
```python
tournament_id = _get_cached_tournament_id(player_id)  # was it saved from a search?
if tournament_id is None:
    # Fall back: GET /api/v1/player/{id} → extract tournament from response
    data = _get(f"/api/v1/player/{player_id}")
    tournament_id = _extract_unique_tournament_id(data.get("team", {}), ...)
```

3. **Fetch raw stats via HTTP:**
```python
data = _get(f"/api/v1/player/{player_id}/unique-tournament/{tid}/season/{sid}/statistics/overall")
# Returns: {"statistics": {"expectedGoals": 3.2, "shotAttempts": 25, ...}, "team": {...}}
```

4. **Parse raw stats into per-90:**
```python
def _parse_stats(stats: dict, minutes_played: int) -> dict:
    result = {}
    for canonical_name in ALL_METRICS:
        # Try every known alias for this metric (Sofascore uses different names)
        raw_val = None
        for alias in _SOFASCORE_KEY_MAP[canonical_name]:
            if alias in stats:
                raw_val = stats[alias]
                break
        if raw_val is not None and minutes_played > 0:
            result[canonical_name] = (raw_val / minutes_played) * 90  # per-90 conversion
    return result
```
*Analogy: like translating measurements from inches to centimeters — same data, standard format.*

5. **Cache and return:**
```python
cache.set(key, result)
return result  # {"name": "Saka", "per90": {"expected_goals": 0.31, ...}, "minutes_played": 2700}
```

#### Glossary
- **Per-90**: A statistic normalized to 90 minutes of play. If a player scored 2 goals in 180 minutes, their per-90 goals = 1.0.
- **`_SOFASCORE_KEY_MAP`**: A dict mapping each of 23 canonical metric names to 3-5 Sofascore API aliases, because Sofascore changes field names between endpoints.
- **`_get(path)`**: The HTTP wrapper. Uses `curl_cffi` (Cloudflare bypass) with stdlib `requests` fallback. Retries on 429/500 errors.

---

### Module: `backend/features/power_rankings.py` (Context Engine)

#### What this does
Collects Elo ratings from two sources (ClubElo for Europe, WorldFootballElo for everywhere else), normalizes all clubs to a 0-100 scale, and computes league means and team relative abilities.

#### Why it exists
The paper's prediction model needs to know *how good* a team is relative to its league. A player moving from a 40-ranked team to a 75-ranked team will see different per-90 changes than one staying at the same level. This module quantifies that context.

#### Step-by-step execution flow (for `compute_daily_rankings`)

1. **Fetch European club Elos from ClubElo:**
```python
ce_df = clubelo_client.get_all_by_date(query_date)
# DataFrame: index=team_name, columns=[elo, league, country, level]
for raw_name in ce_df.index:
    canonical = _CLUBELO_TO_SOFASCORE.get(raw_name, raw_name)  # "ManCity" → "Manchester City"
    all_teams[canonical] = (elo_val, league_code)
```

2. **Fetch non-European Elos from WorldFootballElo:**
```python
for code, info in LEAGUES.items():
    if info.clubelo_league is not None:  # skip — already covered by ClubElo
        continue
    teams = worldfootballelo_client.get_league_teams(info.worldelo_slug)
    # e.g., slug="Brazil" → [{"name": "Flamengo", "elo": 1875}, ...]
```

3. **Normalize globally 0-100:**
```python
global_min = min(all_elos)   # weakest team on Earth today
global_max = max(all_elos)   # strongest team on Earth today
normalized = (raw_elo - global_min) / (global_max - global_min) * 100
# Result: Manchester City ≈ 95, Gwangju FC ≈ 22
```
*Analogy: like grading on a curve — best = 100, worst = 0, everyone else in between.*

4. **Build league snapshots (mean, std, percentiles):**
```python
league_snapshots[code] = LeagueSnapshot(
    mean_normalized=np.mean(norms),   # average team score in this league
    p10=np.percentile(norms, 10),     # bottom 10% cutoff
    # ... p25, p50, p75, p90
)
```

5. **Compute relative ability per team:**
```python
relative_ability = normalized_score - league_mean_normalized
# Man City: 95 - 65 = +30 (much better than PL average)
# Ipswich:  35 - 65 = -30 (much worse than PL average)
```

6. **Fuzzy team name resolution** (when Sofascore says "Tottenham Hotspur" but ClubElo says "Tottenham"):
```python
def _fuzzy_find_team(query, teams_dict):
    # Priority cascade:
    # 1. Exact normalized match (strip accents, lowercase)
    # 2. _EXTREME_ABBREVS lookup (502 entries, augmented by ~45K dynamic REEP aliases at runtime)
    # 3. Substring containment (≥6 chars, ≥45% overlap)
    # 4. Word-level matching (shared words ≥4 chars)
    # 5. SequenceMatcher ratio ≥ 0.70
```

#### Glossary
- **Elo rating**: A numerical strength rating. Higher = stronger. ClubElo typically ranges from ~1100 to ~2100.
- **Relative ability**: How much better or worse a team is compared to its own league average. Key input to the prediction model.
- **League snapshot**: Summary statistics for all teams in a league on a given date.

---

### Module: `backend/features/adjustment_models.py` (Prediction Brain)

#### What this does
Contains three systems: (1) `TeamAdjustmentModel` — 13 Ridge regression models that predict how a team's context changes per-90 output, (2) `PlayerAdjustmentModel` — 13 per-position Ridge models, and (3) `paper_heuristic_predict()` — the hand-tuned fallback used when no trained model exists.

#### Why it exists
The paper argues that a player's per-90 stats are a function of three forces: their individual quality, their team's tactical style, and the opposition quality they face. These models quantify those three forces to predict what happens when one (or more) of those inputs changes.

#### Step-by-step execution flow (for `paper_heuristic_predict`)

1. **Decompose the transfer into team vs opposition effects:**
```python
ra = change_relative_ability / 100.0   # normalize to roughly [-1, 1]
league_gap = (source_league_mean - target_league_mean) / 100.0
# Positive league_gap = moving to weaker league (easier opposition)
team_gap = ra - league_gap
# How much the TEAM quality changes (independent of league difficulty)
```
*Analogy: if you transfer from a top-4 team in the Premier League to a relegation team in Ligue 2, the `team_gap` is negative (worse team) but the `league_gap` is positive (weaker opposition). These partially cancel out.*

2. **Apply player quality modifier:**
```python
quality_scale = 1.0
if player_rating is not None:
    raw_mod = (player_rating - 6.5) * 0.15  # 6.5 = average Sofascore rating
    if ra < -0.1:     # downgrade — halve protection
        raw_mod *= 0.5
    quality_scale = max(0.7, min(1.3, 1.0 - raw_mod))
```
*Elite players (rating 7.5+) retain more of their individual output — but protection is halved when downgrading (even Mbappe would suffer at Ipswich).*

3. **For each of the 13 core metrics, compute 3 forces:**

   a. **Style shift** — how the target team's tactical system differs:
   ```python
   style_diff = tgt_avg - src_avg       # position-average difference
   style_shift = effective_team_inf * style_diff * league_attn
   ```

   b. **Conformity pull** — partial adaptation to the new team:
   ```python
   conformity_pull = 0.25 * effective_team_inf * (tgt_avg - player_val) * league_attn
   ```

   c. **Team + opposition quality factor** — combined multiplicative effect:
   ```python
   team_effect = sensitivity * team_gap * (1 - damp * abs(team_gap))
   opp_effect = opp_sens * league_gap
   combined_factor = 1.0 + team_effect + opp_effect
   ```

4. **Combine and clamp:**
```python
base = player_val + style_shift + conformity_pull
pred = base * combined_factor
predicted[m] = max(pred, 0.0)  # per-90 can never be negative
```

#### Glossary
- **`_TEAM_INFLUENCE`**: Per-metric dict (0-1). How much each metric is driven by team tactics vs individual skill. Dribbling = 0.15 (individual), passing = 0.60 (team-driven).
- **`_ABILITY_SENSITIVITY`**: Per-metric dict. How strongly each metric scales with team quality. Positive for offensive, negative for defensive.
- **`_OPP_QUALITY_SENS`**: Per-metric dict. How much facing weaker/stronger opponents affects output. xG = 1.30 (big effect), dribbling = 0.12 (irreducible).
- **Asymmetric damping**: Downgrades use `_DAMPING_FACTOR_DOWN=0.05` (allows larger drops), upgrades use `_DAMPING_FACTOR_UP=0.10` (ceiling effect prevents unrealistic gains).

---

### Module: `backend/models/transfer_portal.py` (ML Model)

#### What this does
Wraps a TensorFlow multi-head neural network with 4 model groups (shooting, passing, dribbling, defending). Falls back to `paper_heuristic_predict` when no trained model exists.

#### Why it exists
The paper's methodology uses a neural network to learn the non-linear relationships between player features, team context, and post-transfer performance. This module is the production inference engine.

#### Step-by-step execution flow (for `predict`)

1. **Check if trained model exists → load or fallback:**
```python
if not self.models:                    # no models loaded yet
    if self.is_trained():              # check for .keras files on disk
        self._load_trained()           # loads weights + feature scaler + target scalers
    if not self.models:
        return self._heuristic_fallback(feature_dict)  # → paper_heuristic_predict()
```

2. **Prepare feature vector:**
```python
full_X = self._prepare_features(feature_dict).reshape(1, -1)  # shape: (1, 46)
if self._scaler is not None:
    full_X = self._scaler.transform(full_X)  # StandardScaler normalization
```

3. **For each of the 4 groups, slice relevant features and predict:**
```python
for group_name, targets in MODEL_GROUPS.items():   # "shooting", "passing", ...
    group_indices = [key_to_idx[k] for k in GROUP_FEATURE_SUBSETS[group_name]]
    X_group = full_X[:, group_indices]             # slice to 10-28 features
    preds = self.models[group_name].predict(X_group)  # TF inference
    # Inverse-transform back to original scale
    if target_scaler is not None:
        preds = target_scaler.inverse_transform(preds)
```
*Analogy: instead of one model trying to predict everything, there are 4 specialist models — a shooting expert, a passing expert, a dribbling expert, and a defending expert.*

4. **Clamp and return:**
```python
result[target] = max(0.0, float(preds[i]))  # per-90 can't be negative
```

#### Glossary
- **Model groups**: Shooting (2 outputs), Passing (7 outputs), Dribbling (1 output), Defending (3 outputs) = 13 total.
- **Feature subsets**: Each group only uses relevant features (shooting uses 16 of 46, dribbling uses 7). 3 additional interaction features are shared across groups.
- **`build_feature_dict()`**: Assembles the 46-key input dictionary from player stats, team rankings, and position averages.

---

### Module: `backend/features/rolling_windows.py` (Feature Smoothing)

#### What this does
Computes time-weighted per-90 averages from match-level data using a 1000-minute sliding window (player) or 3000-minute window (team), and blends with prior values for low-data players.

#### Why it exists
A player's season aggregate might be 2000+ minutes, but their recent form matters more for transfer predictions. Rolling windows capture recent performance. The prior blend prevents unreliable predictions for players with <1000 minutes of data.

#### Step-by-step execution flow

1. **Accumulate match logs up to the window limit:**
```python
for log in match_logs:           # ordered most-recent first
    mins = log.get("minutes", 0)
    if minutes_accumulated >= window_minutes:  # 1000 for player
        break
    minutes_accumulated += mins
    for metric in ALL_METRICS:
        val = log.get(metric)
        if val is not None:
            totals[metric] += float(val) * mins   # minute-weighted sum
            counts[metric] += mins
```
*Analogy: like calculating a weighted GPA where recent courses count more.*

2. **Compute weighted per-90:**
```python
result[metric] = totals[metric] / counts[metric]  # already expressed as per-90
```

3. **Blend with priors for low-data players:**
```python
weight = min(1.0, minutes_played / 1000)          # 500 min → weight = 0.5
feature = (1 - weight) * prior + weight * raw      # half prior, half data
confidence = "red" if weight < 0.3 else "amber" if weight <= 0.7 else "green"
```

#### Glossary
- **RAG confidence**: Red/Amber/Green indicator. Red (<300 min) = heavily relying on prior assumptions. Green (>700 min) = data-rich, reliable.
- **Prior**: League/position average used as a starting estimate before the player has enough data.

---

## SECTION 3 — Test Coverage Audit

### Current State

- **Framework**: `unittest` (Python standard library), run via `pytest`
- **Command**: `python -m pytest tests/ -v`
- **Total tests**: 488
- **Test files**: 24

### Coverage Table

| File | Functions | Tested | Untested | Coverage |
|---|---|---|---|---|
| `backend/data/sofascore_client.py` | 12 public | 12 | 0 | ~100% |
| `backend/data/clubelo_client.py` | 6 public | 6 | 0 | ~100% |
| `backend/data/worldfootballelo_client.py` | 3 public | 3 | 0 | ~100% |
| `backend/data/elo_router.py` | 4 public | 4 | 0 | ~100% |
| `backend/data/cache.py` | 7 public | 7 | 0 | ~100% |
| `backend/data/reep_registry.py` | 6+ public | 6+ | 0 | ~100% |
| `backend/data/statsbomb_client.py` | 4+ public | 4+ | 0 | ~100% |
| `backend/data/footballdata_client.py` | 3+ public | 3+ | 0 | ~100% |
| `backend/features/rolling_windows.py` | 7 public | 1 | 6 | **14%** |
| `backend/features/power_rankings.py` | 7 public | 3 | 4 | 43% |
| `backend/features/adjustment_models.py` | 14 public | 9 | 5 | **64%** (was 50%, now includes `paper_heuristic_predict`) |
| `backend/models/transfer_portal.py` | 9 public | 4 | 5 | 44% |
| `backend/models/shortlist_scorer.py` | 3 public | 3 | 0 | 100% |
| `backend/models/training_pipeline.py` | 8+ public | 6+ | 2 | ~75% |
| `backend/models/backtester.py` | 2 public | 1 | 1 | 50% |

### Specific Untested Functions

| File | Function | Status |
|---|---|---|
| `rolling_windows.py` | `compute_confidence()` | ❌ No test |
| `rolling_windows.py` | `blend_weight()` | ❌ No test |
| `rolling_windows.py` | `blend_features()` | ❌ No test |
| `rolling_windows.py` | `team_rolling_average()` | ❌ No test |
| `rolling_windows.py` | `team_position_rolling_average()` | ❌ No test |
| `rolling_windows.py` | `compute_player_features()` | ❌ No test |
| `power_rankings.py` | `get_league_snapshot()` | ❌ No test |
| `power_rankings.py` | `get_relative_ability()` | ❌ No test |
| `power_rankings.py` | `get_change_in_relative_ability()` | ❌ No test |
| `power_rankings.py` | `get_historical_rankings()` | ❌ No test |
| `adjustment_models.py` | `TeamAdjustmentModel.predict_all()` | ❌ No test |
| `adjustment_models.py` | `TeamAdjustmentModel.save()` / `.load()` | ❌ No test |
| `adjustment_models.py` | `PlayerAdjustmentModel.predict_all()` | ❌ No test |
| `adjustment_models.py` | `PlayerAdjustmentModel.save()` / `.load()` | ❌ No test |
| `transfer_portal.py` | `build_feature_dict()` | ❌ No test |
| `transfer_portal.py` | `TransferPortalModel.is_trained()` | ❌ No test |
| `transfer_portal.py` | `TransferPortalModel.predict_batch()` | ❌ No test |
| `transfer_portal.py` | `TransferPortalModel.save()` / `.load()` | ❌ No test |
| `backtester.py` | `show_example_predictions()` | ❌ No test |

### Top 5 Highest-Risk Untested Areas

1. **`paper_heuristic_predict()`** — The default prediction engine every user sees. Wrong coefficients = wrong predictions for every transfer query. ✅ **NOW TESTED (32 tests added in this PR)**
2. **`rolling_windows.blend_features()`** — Prior blending for low-data players. Bugs here silently corrupt feature inputs to every prediction.
3. **`power_rankings.get_change_in_relative_ability()`** — Computes the key input (change_relative_ability) that drives 60%+ of prediction magnitude. A sign error here flips all predictions.
4. **`transfer_portal.build_feature_dict()`** — Assembles the 46-key feature dict that feeds the TF model. Wrong key mapping = wrong model input = wrong prediction, silently.
5. **`rolling_windows.compute_confidence()`** — Controls the RAG badge shown to users. Wrong thresholds = misleading confidence indicators.

### Test Implementation

The #1 highest-risk untested function (`paper_heuristic_predict`) now has comprehensive tests in `tests/test_paper_heuristic.py` (32 tests, all passing). Coverage includes:

**Happy path (5 tests):**
- Identity transfer (same team) produces ≈no change
- Output contains exactly 13 core metrics
- All values non-negative
- Upgrade increases offensive metrics
- Downgrade decreases offensive metrics

**Defensive metric inversion (2 tests):**
- Upgrade decreases clearances/interceptions
- Downgrade increases clearances/interceptions

**Dribbling irreducibility (1 test):**
- Take-ons change < 10% for moderate moves

**Opposition quality (3 tests):**
- Weaker league boosts xG
- Stronger league reduces xG
- None league means = no opposition effect

**Player quality modifier (3 tests):**
- High-rated players retain more output
- Elite protection halved for downgrades
- None rating equals center (6.5) rating

**Edge cases (6 tests):**
- Zero player per-90
- Extreme positive RA (doesn't explode)
- Extreme negative RA (clamped to ≥0)
- Missing metrics in player dict
- Missing metrics in position averages
- Large league gap attenuates style

**Style data detection (7 tests):**
- Identical avgs = no style data
- Different avgs = real style data
- One metric different = not enough (threshold = 2)
- Two metrics different = enough
- Empty dicts = no style data
- Player-equal avgs = no style data
- No-style triggers fallback estimation path

**Per-metric differentiation (2 tests):**
- Not all 13 metrics change equally
- Offensive increase / defensive decrease for upgrades

**Asymmetric damping (1 test):**
- At least some metrics show larger drops on downgrade than gains on upgrade

**Type checks (2 tests):**
- All values are float
- Return is dict

---

## SECTION 4 — Developer Mental Model

### 3 Things You MUST Understand Before Touching This Code

1. **Every stat is per-90, always.** The entire system stores, computes, and displays statistics normalized to 90 minutes of play. Raw totals exist only momentarily inside `_parse_stats()`. If you see a raw total anywhere else, it's a bug. This matters because per-90 makes players with different minutes comparable — a player with 0.31 xG per 90 in 2700 minutes is directly comparable to one with 0.31 xG per 90 in 900 minutes (though the second has lower confidence).

2. **There are two prediction paths, and most users see the heuristic.** `TransferPortalModel.predict()` checks `is_trained()` first. If no `.keras` files exist in `data/models/` (which is the default — training requires running the full pipeline with live Sofascore data), it silently falls back to `paper_heuristic_predict()`. The heuristic is not a placeholder — it faithfully implements the paper's methodology with calibrated coefficients. Any change to prediction logic must consider both paths.

3. **Team name resolution is a 5-priority fuzzy cascade.** Data comes from 3 sources (Sofascore, ClubElo, WorldFootballElo) that all use different team names. "ManCity" (ClubElo) must match "Manchester City" (Sofascore). The resolution in `power_rankings.py` uses: exact match → accent-normalized → extreme abbreviation lookup (502 entries covering 51 leagues) → substring containment → SequenceMatcher ratio. False positive prevention (e.g., "Orlando City SC" must NOT match "Man City") is enforced by minimum overlap ratios. 531 ClubElo entries are mapped. If you add a new data source, you may need to add team name mappings — though REEP dynamic aliases provide ~45K automatic mappings at runtime.

### Where Bugs Are Most Likely to Occur and Why

1. **Team name mismatches between data sources.** ClubElo, WorldFootballElo, and Sofascore all use different conventions. A new team promotion, a name change, or a missing mapping in `_CLUBELO_TO_SOFASCORE` silently causes `get_team_ranking()` to return `None`, which cascades to `change_relative_ability=0` and flat predictions. Symptom: all metrics show ≈0% change.

2. **Sofascore API field name changes.** Sofascore is an unofficial API. They periodically rename fields (e.g., `expectedGoals` → `xG`). The `_SOFASCORE_KEY_MAP` has 3-5 aliases per metric, but a new rename breaks silently — the metric returns `None`, which becomes `0.0` in the per-90 dict. Symptom: a specific metric drops to 0 for all players.

3. **Cache poisoning from transient API failures.** If Sofascore returns an HTTP 500 during `get_league_player_stats()`, an empty result might get cached for 24 hours. The code has guards (`if cached:` not `if cached is not None:`) but edge cases remain — especially for functions that return dicts with all-None values (truthy but useless).

4. **The `change_relative_ability / 50.0` vs `/ 100.0` discrepancy.** `PlayerAdjustmentModel.predict()` divides by 50 (mapping ±50 to ±1). `paper_heuristic_predict()` divides by 100 (mapping ±100 to ±1). These are intentionally different scales, but any code that passes `change_relative_ability` to the wrong function gets doubled/halved predictions.

### The Safest First Change to Make

**Add a test for `rolling_windows.compute_confidence()`:**

```python
# In tests/test_rolling_windows.py (new file)
def test_compute_confidence_red():
    assert compute_confidence(0.0) == "red"
    assert compute_confidence(0.29) == "red"

def test_compute_confidence_amber():
    assert compute_confidence(0.3) == "amber"
    assert compute_confidence(0.7) == "amber"

def test_compute_confidence_green():
    assert compute_confidence(0.71) == "green"
    assert compute_confidence(1.0) == "green"
```

This is safe because:
- It's a pure function with no side effects
- It touches no external APIs
- It's read-only (no code changes, just new test code)
- It validates a user-facing feature (the RAG confidence badge)
- Running the test teaches you the testing conventions (unittest, temp CACHE_DIR, `python -m pytest`)

### 3 Questions to Ask Yourself Before Every PR

1. **Does this change work in BOTH prediction paths?** If you modified anything in the feature pipeline or prediction logic, verify the behavior with the TF model AND the heuristic fallback. Most users will see the heuristic.

2. **Did I break team name resolution?** If you changed any data source, team mapping, or fuzzy matching logic, run `test_fuzzy_matching.py` (79 tests) and check that false positive tests still pass (Orlando City ≠ Man City, Inter Miami ≠ Inter Milan).

3. **Am I caching something that could be empty?** If you added or modified a cache call, ensure you're not caching empty/error results from transient API failures. Check for the `if cached:` pattern (not `if cached is not None:`), and verify that empty results are NOT stored.
