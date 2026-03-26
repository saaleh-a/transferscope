"""About & Methodology — in-app reference for how TransferScope works.

Shows:
  (a) How the prediction model works (three forces)
  (b) What data it uses and what it doesn't
  (c) League coverage table
  (d) Model limitations
  (e) The paper reference
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import streamlit as st

from backend.utils.league_registry import LEAGUES
from frontend.theme import section_header, COLORS


def render():
    st.header("About TransferScope")
    st.caption("How it works, what leagues are covered, and known limitations")

    # ── Quick overview ────────────────────────────────────────────────────
    section_header("Overview", "What this tool does")
    st.markdown(
        """
TransferScope predicts how a football player's **per-90 statistics** will change
when they transfer from one club to another. It's based on the
[Transfer Portal paper](https://arxiv.org/abs/2201.11533) (Dinsdale & Gallagher, 2022).

The system predicts **13 core metrics** simultaneously — covering shooting (xG, shots),
passing (xA, crosses, passes, long balls, chances created), possession (dribbling,
penalty area entries), and defending (clearances, interceptions, pressing).

**Four tools:**
- **Transfer Impact** — Predict how each metric changes at a new club (dual simulation)
- **Shortlist Generator** — Find replacement candidates across Big 5 leagues (k-means clustering)
- **Hot or Not** — Quick verdict on a transfer rumour (position-aware weighting)
- **About & Methodology** — This page

> **Key insight:** A player at a worse team *can improve or decline* at a bigger team,
> per-metric, depending on the target team's style. It's not just "harder league =
> everything gets worse."
"""
    )

    # ── How predictions work ──────────────────────────────────────────────
    section_header("How Predictions Work", "Three forces compete for each metric")
    st.markdown(
        """
For each of the 13 metrics, three independent forces determine the prediction:

### 1. 🎯 Style Shift — *"How does the target team play?"*

We compare how each team uses players in the same position. If the target team's
forwards average 0.45 xG/90 and the source team's forwards average 0.30, the model
pulls the player's xG toward the target team's style — but only partially, based on
how team-dependent that metric is.

| Metric | Team Dependence | Why |
|--------|----------------|-----|
| Dribbling | Low (15%) | Highly individual — "irreducible" per the paper |
| Crossing, Passing | High (50-60%) | Tactical system drives volume |
| xG, Shots | Medium (35-40%) | Team creates chances, but finishing is individual |
| Clearances | High (55%) | Defensive approach dictates workload |

When real team-position data isn't available, the system estimates style differences
from the relative ability gap, attenuated for extreme transfers where quality
dominates over tactical style.

### 2. 💪 Team Quality — *"Is the new team better or worse within their league?"*

Moving to a more dominant team (higher Power Ranking relative to their league)
generally boosts offensive output and reduces defensive workload. The sensitivity
differs per metric:

- **Strong effect:** xG (+0.82), penalty area entries (+0.85), chances created (+0.75)
- **Weak/negative:** Clearances (−0.65), interceptions (−0.50) — better teams defend less
- **Minimal:** Dribbling (+0.18) — barely affected by team quality

Damping is **asymmetric**: downgrades allow larger drops (realistic), while upgrades
are more conservative (talent ceiling).

### 3. 🌍 Opposition Quality — *"How strong is the new league?"*

Moving to a weaker league means facing weaker defenders and goalkeepers — boosting
per-90 offensive output *even if the team is worse*. This is independent of team
quality:

- **Elite player → weaker league (hypothetical):** xG rises because weaker-league defenders are
  easier to beat, even though the team itself creates fewer chances
- **Player → K-League:** xG increases significantly despite much weaker team (paper
  Section 4.3.1 — Doku at Gwangju)
- **Dribbling:** Barely changes (0.12 sensitivity) — individual skill

### Combined Effect

These three forces are added together (faithful to the paper's linear regression
structure). This means for a cross-league downgrade:
- 📈 **xG, shots may increase** (opposition quality boost outweighs team quality drop)
- 📉 **Passing, creativity decline** (worse teammates provide fewer opportunities)
- 📈 **Defensive workload increases** (worse team needs more defending)
- ➡️ **Dribbling stays roughly the same** (individual skill)
"""
    )

    # ── Dual simulation ──────────────────────────────────────────────────
    section_header("Dual Simulation", "How we compare predictions")
    st.markdown(
        """
Following the paper's methodology (Section 4), we simulate the player at **both**
their current and target clubs, then compare the two model outputs.

```
predicted_current = model(player, current_team → current_team, ra=0)
predicted_target  = model(player, current_team → target_team, ra=Δ)
% change = (predicted_target − predicted_current) / predicted_current
```

This is more robust than comparing raw observed stats vs predicted stats, because
both sides come from the same model process — reducing noise sensitivity.
"""
    )

    # ── Shortlist methodology ────────────────────────────────────────────
    section_header("Shortlist Generator", "How replacement candidates are found and ranked")
    st.markdown(
        """
### Multi-League Search

The shortlist generator scans players across multiple leagues simultaneously. By default,
it searches the **Big 5 European leagues** (Premier League, La Liga, Bundesliga, Serie A,
Ligue 1), plus the player's own league which is always scanned first.

**Rate-limit protection:** Sofascore applies aggressive rate-limiting (403/429 errors)
after 2-3 rapid requests. A **1.5-second delay** between league API calls prevents this.
Without the delay, scanning multiple leagues causes all but the first 2-3 to fail →
0 candidates.

A diagnostic panel shows which leagues returned data and how many candidates were found
from each.

### K-Means Clustering + Weighted Distance

Candidates are ranked using a two-stage process:

1. **K-means clustering** groups all candidates by playing style (k=√(n/2), capped 3-10).
   The reference player is included to identify their cluster.
2. **Weighted Euclidean distance** measures similarity to the reference player across
   all weighted metrics (user-configurable). Candidates in the same style cluster receive
   a **15% score bonus**.

### Filter Design

Filters use a **None-passthrough** design: candidates with unknown age or minutes
pass through filters rather than being excluded. This means `max_age=30` selects
"players aged ≤30 OR players with unknown age."

This is intentional — Sofascore API data is sparse, and excluding unknowns would
silently drop valid candidates. Missing fields show as "—" in the results table.
"""
    )

    # ── What data it uses ─────────────────────────────────────────────────
    section_header("Data Sources", "What goes into predictions")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
**✅ What the model USES:**
- Current season per-90 stats (Sofascore)
- Team-position averages (tactical style proxy)
- Team & league Elo ratings (Power Rankings)
- Player match rating (Sofascore 0-10)
- Rolling window confidence (minutes played)
"""
        )
    with col2:
        st.markdown(
            """
**❌ What the model does NOT use:**
- Injury or fitness history
- Age / career trajectory
- Multi-season trend data
- Expected playing time at new club
- Transfer fee / market value
- Manager / coaching staff changes
"""
        )

    st.info(
        "💡 The model predicts performance *capacity* — what the player's per-90 "
        "stats would look like given the team and league context. It does not predict "
        "whether the player will actually get playing time, stay fit, or adapt "
        "psychologically to a new environment."
    )

    # ── League coverage ──────────────────────────────────────────────────
    section_header("League Coverage", "37 leagues with Power Rankings support")

    # Group by continent
    continents: Dict[str, list] = {}
    for code, info in LEAGUES.items():
        continents.setdefault(info.continent, []).append((code, info))

    for continent in ["Europe", "South America", "North America", "Asia"]:
        leagues = continents.get(continent, [])
        if not leagues:
            continue

        st.markdown(f"**{continent}** ({len(leagues)} leagues)")
        rows = []
        for code, info in sorted(leagues, key=lambda x: x[1].name):
            elo_source = "ClubElo" if info.clubelo_league else (
                "WorldFootballElo" if info.worldelo_slug else "None"
            )
            rows.append({
                "Code": code,
                "League": info.name,
                "Country": info.country,
                "Elo Source": elo_source,
            })

        import pandas as pd
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        """
**Not currently covered:**
- Women's leagues (NWSL, WSL, Liga F, etc.)
- African leagues
- Most Asian leagues (except J-League, K-League via WorldFootballElo)
- Third-tier leagues and below (League One, 3. Liga, etc.)

If a club can't be found in Power Rankings, the system defaults to a score of 50.0
and shows a warning. Predictions will still work but may be less accurate because the
model can't gauge the true team/league strength.
"""
    )

    # ── Model confidence ─────────────────────────────────────────────────
    section_header("Confidence Indicators", "How to read the RAG badges")
    st.markdown(
        """
The confidence badge uses a **traffic light** system based on how much data the
player has in the rolling window:

| Badge | Weight | Meaning |
|-------|--------|---------|
| 🟢 GREEN | > 0.7 | 700+ minutes played — data-rich, reliable prediction |
| 🟡 AMBER | 0.3–0.7 | Mixed data — prediction blends observed stats with prior estimates |
| 🔴 RED | < 0.3 | Heavily prior-dependent — treat prediction with caution |

The weight formula: `weight = min(1.0, minutes_played / 1000)`. Higher weight means
the prediction relies more on the player's actual observed stats and less on
league/position priors.
"""
    )

    # ── Model training & trust ───────────────────────────────────────────
    section_header("Model Training & Validation", "Can you trust the predictions?")

    # Check if backtest report exists
    backtest_path = os.path.join("data", "models", "backtest_report.json")
    has_trained_model = os.path.exists(os.path.join("data", "models"))

    st.markdown(
        """
**Two prediction modes:**

1. **Trained TensorFlow model** — 4-group multi-head neural network trained on
   historical transfer data (pre/post transfer per-90 stats across 5+ seasons and
   11+ leagues). Validated via held-out temporal test set with backtesting.

2. **Paper-aligned heuristic** (fallback) — when no trained model exists, uses
   calibrated per-metric coefficients derived from the paper's regression structure.
   This is the default mode unless you've run the training pipeline.
"""
    )

    if os.path.exists(backtest_path):
        import json
        try:
            with open(backtest_path) as f:
                report = json.load(f)

            st.success("✅ Trained model with backtest report available")

            overall = report.get("overall", {})
            st.markdown(
                f"**Backtest summary** ({report.get('n_samples', '?')} test transfers):\n"
                f"- Overall improvement vs naive baseline: "
                f"**{overall.get('overall_improvement_pct', 0):.1f}%**\n"
                f"- Metrics improved: **{overall.get('metrics_improved', 0)}** "
                f"/ {overall.get('metrics_total', 13)}"
            )

            per_metric = report.get("per_metric", {})
            if per_metric:
                rows = []
                for metric, data in per_metric.items():
                    rows.append({
                        "Metric": metric.replace("_", " ").title(),
                        "MAE": f"{data.get('mae', 0):.3f}",
                        "Within 10%": f"{data.get('within_10_pct', 0):.0f}%",
                        "Within 20%": f"{data.get('within_20_pct', 0):.0f}%",
                        "Direction Acc.": f"{data.get('direction_accuracy', 0):.0f}%",
                        "vs Naive": f"{data.get('improvement_vs_naive', 0):+.1f}%",
                    })
                import pandas as pd
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception:
            pass
    else:
        # Check if training is currently running
        training_status = st.session_state.get("training_status", "unknown")
        training_step = st.session_state.get("training_step", "")
        if training_status in ("starting", "running"):
            st.info(
                "🔄 **Model training is running automatically in the background.**\n\n"
                f"Current step: {training_step}\n\n"
                "Predictions use the paper-aligned heuristic until training completes. "
                "The trained model will be used automatically once ready."
            )
        elif training_status == "failed":
            st.warning(
                f"⚠️ **Auto-training did not complete:** {training_step}\n\n"
                "The app is using the heuristic fallback. You can retry from the "
                "sidebar, or run manually:\n\n"
                "```\npython backend/models/training_pipeline.py --seasons-back 3 "
                "--leagues ENG1,ESP1,GER1\n```"
            )
        else:
            st.warning(
                "⚠️ No trained model found — using heuristic fallback. "
                "Training starts automatically on app launch. "
                "You can also run it manually:\n\n"
                "```\npython backend/models/training_pipeline.py --seasons-back 3 "
                "--leagues ENG1,ESP1,GER1\n```"
            )

    # ── The 13 metrics ───────────────────────────────────────────────────
    section_header("The 13 Core Metrics", "What the paper predicts")
    st.markdown(
        """
| # | Group | Metric | Sofascore Field | Description |
|---|-------|--------|-----------------|-------------|
| 1 | ⚡ Shooting | xG | `expected_goals` | Shot quality — how many goals expected from chances |
| 2 | ⚡ Shooting | Shots | `shots` | Shot volume per 90 |
| 3 | ◈ Passing | xA | `expected_assists` | Creative quality — expected assists from key passes |
| 4 | ◈ Passing | Crosses | `successful_crosses` | Accurate crosses per 90 |
| 5 | ◈ Passing | Total Passes | `successful_passes` | Completed passes per 90 |
| 6 | ◈ Passing | Short Pass % | `pass_completion_pct` | Pass accuracy percentage |
| 7 | ◈ Passing | Long Passes | `accurate_long_balls` | Accurate long balls per 90 |
| 8 | ◈ Passing | Pen. Area Entries | `touches_in_opposition_box` | Touches in opposition box per 90 |
| 9 | ◈ Passing | Passes Att 3rd | `chances_created` | Chances created / key passes per 90 |
| 10 | ◎ Dribbling | Take-ons | `successful_dribbles` | Successful dribbles per 90 |
| 11 | ◆ Defending | Clearances | `clearances` | Clearances per 90 |
| 12 | ◆ Defending | Interceptions | `interceptions` | Interceptions per 90 |
| 13 | ◆ Defending | Press (Att 3rd) | `possession_won_final_3rd` | Ball recoveries in final third per 90 |
"""
    )

    # ── Paper reference ──────────────────────────────────────────────────
    section_header("Paper Reference", "The research this is based on")
    st.markdown(
        """
> **The Transfer Portal: Predicting the Impact of a Player Transfer on a Football
> Team's Performance**
> — Jake Dinsdale & Ben Gallagher (2022)
> — [arXiv:2201.11533](https://arxiv.org/abs/2201.11533)

The paper introduces a framework for predicting how player performance statistics
change after a transfer, using:
- A 4-level Power Ranking hierarchy (continent → country → league → team)
- Team and player adjustment models (linear regression)
- A TensorFlow multi-head neural network with 4 groups (shooting, passing, dribbling, defending)
- Dual simulation methodology (predict at both clubs, compare)

TransferScope is a faithful recreation of this paper's methodology, with additional
improvements for robustness: dynamic Elo-based Power Rankings, per-metric opposition
quality modelling, asymmetric damping, fuzzy team name matching (180+ aliases),
K-means shortlist clustering with rate-limit protection, None-passthrough filter design,
per-group feature subsets (4 specialist neural networks), and 208 automated tests.
"""
    )
