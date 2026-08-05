"""Model Diagnostics — model status, feature importance, cache health."""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict

import streamlit as st

from frontend.theme import (
    section_header, stat_card, COLORS, PLOTLY_LAYOUT, page_header,
    apply_plotly_theme,
)

_log = logging.getLogger(__name__)


def render():
    page_header(
        "Diagnostics",
        "What the model is made of and whether every data source is still "
        "answering.",
        kicker="System",
    )

    _render_model_status()
    _render_feature_importance()
    _render_cache_health()
    _render_data_source_status()
    _render_system_info()


# ── Section 1: Model Status ─────────────────────────────────────────────────


def _render_model_status():
    section_header("Model Status", "Current prediction mode and model configuration")

    from backend.models.transfer_portal import (
        TransferPortalModel,
        FEATURE_DIM,
        MODEL_GROUPS,
    )

    model = TransferPortalModel()
    trained = model.is_trained()

    mode_label = "Trained (TensorFlow)" if trained else "Heuristic Fallback"
    mode_color = COLORS["accent_green"] if trained else COLORS["accent_amber"]

    cards = [
        stat_card("Prediction Mode", mode_label),
        stat_card("Feature Dimension", str(FEATURE_DIM)),
        stat_card("Model Groups", str(len(MODEL_GROUPS))),
    ]

    # Build group detail cards
    group_details = []
    for group_name, targets in MODEL_GROUPS.items():
        group_details.append(f"{group_name.title()}: {len(targets)} targets")

    cards.append(
        stat_card("Total Targets", str(sum(len(t) for t in MODEL_GROUPS.values())))
    )

    st.markdown(
        '<div style="display:flex; gap:1rem; margin:0.8rem 0; flex-wrap:wrap;">'
        + "".join(
            f'<div style="flex:1; min-width:160px;">{c}</div>' for c in cards
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # Group breakdown table
    import pandas as pd

    rows = []
    for group_name, targets in MODEL_GROUPS.items():
        rows.append(
            {
                "Group": group_name.title(),
                "Targets": len(targets),
                "Metrics": ", ".join(t.replace("_", " ").title() for t in targets),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Section 2: Feature Importance ────────────────────────────────────────────


def _render_feature_importance():
    section_header(
        "Feature Importance", "Gradient-based sensitivity (requires trained model)"
    )

    from backend.data.sofascore_client import CORE_METRICS
    from backend.models.transfer_portal import (
        TransferPortalModel,
        build_feature_dict,
    )

    model = TransferPortalModel()
    if not model.is_trained():
        st.info(
            "Feature importance requires a trained TensorFlow model. "
            "Currently using heuristic fallback."
        )
        return

    try:
        model.load()
    except Exception as exc:
        st.warning(f"Could not load model: {exc}")
        return

    # Build a sample feature dict at the training-data mean so gradient-based
    # importance is computed at a representative operating point.
    # Use real scaler means when a trained scaler is available; otherwise
    # fall back to realistic values that match training distributions.
    from backend.models.transfer_portal import _feature_keys

    all_keys = _feature_keys()
    if model._scaler is not None and hasattr(model._scaler, "mean_"):
        scaler_means = model._scaler.mean_
        key_to_mean = {k: float(scaler_means[i]) for i, k in enumerate(all_keys)}
        # key_to_mean is keyed by *feature* name ("player_expected_goals"), not
        # by bare metric name, so `key_to_mean.get(m, 0.5)` missed on all 13 and
        # evaluated the gradients at a point where a striker's xG was 5x the
        # training mean (0.5 vs 0.1007) and box touches 6x below it (0.5 vs
        # 3.14). The team_pos lines below already got this right, which is what
        # made it easy to miss.
        sample_per90 = {m: key_to_mean.get(f"player_{m}", 0.5) for m in CORE_METRICS}
        sample_fd = build_feature_dict(
            player_per90=sample_per90,
            team_ability_current=key_to_mean.get("team_ability_current", 60.0),
            team_ability_target=key_to_mean.get("team_ability_target", 65.0),
            league_ability_current=key_to_mean.get("league_ability_current", 50.0),
            league_ability_target=key_to_mean.get("league_ability_target", 55.0),
            team_pos_current={m: key_to_mean.get(f"team_pos_current_{m}", 0.4) for m in CORE_METRICS},
            team_pos_target={m: key_to_mean.get(f"team_pos_target_{m}", 0.5) for m in CORE_METRICS},
            raw_elo_current=key_to_mean.get("raw_elo_current", 1700.0),
            raw_elo_target=key_to_mean.get("raw_elo_target", 1700.0),
            player_height_cm=key_to_mean.get("player_height_cm", 180.0),
            player_age=key_to_mean.get("player_age", 25.0),
        )
    else:
        sample_per90 = {m: 0.5 for m in CORE_METRICS}
        sample_fd = build_feature_dict(
            player_per90=sample_per90,
            team_ability_current=60.0,
            team_ability_target=65.0,
            league_ability_current=50.0,
            league_ability_target=55.0,
            team_pos_current={m: 0.4 for m in CORE_METRICS},
            team_pos_target={m: 0.5 for m in CORE_METRICS},
            raw_elo_current=1700.0,
            raw_elo_target=1700.0,
            player_height_cm=180.0,
            player_age=25.0,
        )

    try:
        importance = model.compute_feature_importance(sample_fd)
    except Exception as exc:
        st.warning(f"Could not compute feature importance: {exc}")
        return

    if not importance:
        st.info("No feature importance data available.")
        return

    import plotly.graph_objects as go

    for group_name, features in importance.items():
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        names = [f[0].replace("_", " ").title() for f in sorted_features]
        values = [f[1] for f in sorted_features]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker_color=COLORS["accent_gold"],
            )
        )
        apply_plotly_theme(
            fig,
            title=f"{group_name.title()} Group",
            height=max(200, 28 * len(names)),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Section 3: Cache Health ──────────────────────────────────────────────────


def _render_cache_health():
    section_header("Cache Health", "Size, limit, and namespace breakdown")

    from backend.data import cache as cache_module

    stats = cache_module.stats()
    if "error" in stats:
        st.info("Cache is not available.")
        return

    pct = stats["pct_of_limit"]
    cards = [
        stat_card("Cache Size", f"{stats['mb']:,.0f} MB"),
        stat_card("Entries", f"{stats['entries']:,}"),
        stat_card("Limit", f"{stats['limit_mb']:,} MB"),
        stat_card("Used", f"{pct:.0f}%", delta_positive=pct < 80),
    ]

    st.markdown(
        '<div style="display:flex; gap:1rem; margin:0.8rem 0; flex-wrap:wrap;">'
        + "".join(
            f'<div style="flex:1; min-width:150px;">{c}</div>' for c in cards
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        f"Eviction: {stats['eviction_policy']}. The cache used to be unbounded "
        "and reached 1 GB with no way to reclaim space; it now evicts the "
        "least-recently-used entries once the limit is reached."
    )

    if pct >= 80:
        st.warning(
            "Cache is near its limit. Entries will start being evicted, which "
            "means more live API calls. Prune below, or raise "
            "CACHE_SIZE_LIMIT_MB."
        )

    breakdown = cache_module.namespace_breakdown()
    if breakdown:
        import pandas as pd

        total = sum(breakdown.values()) or 1
        rows = [
            {
                "Namespace": ns,
                "Entries": f"{count:,}",
                "Share": f"{count / total:.0%}",
            }
            for ns, count in breakdown.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        negative = breakdown.get("sofascore_neg", 0)
        if negative > total * 0.2:
            st.caption(
                f"`sofascore_neg` holds {negative:,} entries ({negative / total:.0%}) "
                "— endpoints that returned nothing. These are cheap to keep and "
                "stop the same dead lookups repeating, but they are the first "
                "thing worth pruning if space is tight."
            )
    else:
        st.info("Cache is empty.")

    if st.button("Prune entries older than 7 days", key="diag_prune"):
        removed = cache_module.prune_expired()
        st.success(
            f"Removed {removed:,} entries older than any TTL in use."
            if removed
            else "Nothing to prune — no entry is older than the longest TTL."
        )


# ── Section 4: Data Source Status ────────────────────────────────────────────


def _render_data_source_status():
    section_header(
        "Data Source Status",
        "Live probes — each source is actually called, not just imported",
    )

    st.caption(
        "This page previously reported a source healthy whenever its Python "
        "module imported. That marked WhoScored and WorldFootballElo green for "
        "months while both returned nothing. Each row below is a real call."
    )

    if st.button("Run health probes", key="diag_probe"):
        st.session_state["_diag_health"] = _run_probes()

    results = st.session_state.get("_diag_health")
    if results is None:
        st.info("Press **Run health probes** to call every source live.")
        return

    import pandas as pd

    from backend.data.source_health import DEAD, DEGRADED, LIVE, summarise

    icon = {LIVE: "✅ Live", DEGRADED: "⚠️ Degraded", DEAD: "❌ Dead"}
    rows = [
        {
            "Data Source": r.name,
            "Status": icon.get(r.status, "❓ Unknown"),
            "Detail": r.detail,
            "Used for": r.used_for,
            "Secs": f"{r.elapsed_s:.1f}",
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(summarise(results))

    dead = [r for r in results if r.status == DEAD]
    if dead:
        st.warning(
            "Dead sources: "
            + ", ".join(r.name for r in dead)
            + ". These are documented as non-functional and nothing depends on "
            "them — the probe exists so a *newly* dead source is visible here "
            "rather than failing silently."
        )


@st.cache_data(ttl=300, show_spinner="Probing data sources…")
def _run_probes():
    """Probe all sources, cached briefly so repeated views are cheap."""
    from backend.data.source_health import probe_all

    return probe_all()


# ── Section 5: System Info ───────────────────────────────────────────────────


def _render_system_info():
    section_header("System Info", "Runtime environment details")

    import platform

    info_rows = [
        {"Component": "Python", "Version": sys.version.split()[0]},
        {"Component": "Platform", "Version": platform.platform()},
    ]

    # Optional heavy imports — report version if available
    _optional = [
        ("TensorFlow", "tensorflow"),
        ("NumPy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("Streamlit", "streamlit"),
        ("Plotly", "plotly"),
        ("diskcache", "diskcache"),
        ("soccerdata", "soccerdata"),
    ]

    for label, mod_name in _optional:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            info_rows.append({"Component": label, "Version": str(ver)})
        except ImportError:
            info_rows.append({"Component": label, "Version": "not installed"})

    import pandas as pd

    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)
