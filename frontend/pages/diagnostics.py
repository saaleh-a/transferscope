"""Model Diagnostics — model status, feature importance, cache health."""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict

import streamlit as st

from frontend.theme import section_header, stat_card, COLORS, PLOTLY_LAYOUT

_log = logging.getLogger(__name__)


def render():
    st.header("Diagnostics")
    st.caption("Model status, feature importance, and system health")

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

    # Build a sample feature dict using real training-data means from the
    # fitted scaler so feature importance is computed at a representative
    # operating point (avoids extreme z-scores from arbitrary defaults).
    from backend.models.transfer_portal import _feature_keys

    all_keys = _feature_keys()
    if model._scaler is not None:
        sample_fd = {k: float(model._scaler.mean_[i]) for i, k in enumerate(all_keys)}
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
            raw_elo_target=1750.0,
            player_height_cm=181.0,
            player_age=26.0,
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
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{group_name.title()} Group",
            height=max(200, 28 * len(names)),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Section 3: Cache Health ──────────────────────────────────────────────────


def _render_cache_health():
    section_header("Cache Health", "diskcache storage and namespace breakdown")

    cache_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ),
        "data",
        "cache",
    )

    if not os.path.exists(cache_dir):
        st.info("No cache directory found — cache has not been initialised yet.")
        return

    # Total cache size
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _dn, fn in os.walk(cache_dir)
        for f in fn
    )
    file_count = sum(1 for dp, _dn, fn in os.walk(cache_dir) for _ in fn)

    cards = [
        stat_card("Cache Size", f"{total_size / 1024 / 1024:.1f} MB"),
        stat_card("Files", str(file_count)),
        stat_card("Location", os.path.basename(cache_dir)),
    ]

    st.markdown(
        '<div style="display:flex; gap:1rem; margin:0.8rem 0; flex-wrap:wrap;">'
        + "".join(
            f'<div style="flex:1; min-width:160px;">{c}</div>' for c in cards
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # Namespace breakdown — count keys per namespace via diskcache
    try:
        import diskcache

        cache = diskcache.Cache(cache_dir)
        namespace_counts: Dict[str, int] = {}
        for key in cache:
            if isinstance(key, str) and ":" in key:
                ns = key.split(":")[0]
                namespace_counts[ns] = namespace_counts.get(ns, 0) + 1
            else:
                namespace_counts["(other)"] = namespace_counts.get("(other)", 0) + 1
        cache.close()

        if namespace_counts:
            import pandas as pd

            ns_rows = [
                {"Namespace": ns, "Entries": count}
                for ns, count in sorted(
                    namespace_counts.items(), key=lambda x: x[1], reverse=True
                )
            ]
            st.dataframe(
                pd.DataFrame(ns_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Cache is empty — no entries found.")
    except Exception as exc:
        st.warning(f"Could not read cache entries: {exc}")


# ── Section 4: Data Source Status ────────────────────────────────────────────


def _render_data_source_status():
    section_header(
        "Data Source Status", "Package availability for each data source"
    )

    sources = [
        ("Sofascore REST API", "backend.data.sofascore_client"),
        ("ClubElo (soccerdata)", "soccerdata"),
        ("WorldFootballElo", "backend.data.worldfootballelo_client"),
        ("StatsBomb Open Data", "backend.data.statsbomb_client"),
        ("TensorFlow", "tensorflow"),
        ("scikit-learn", "sklearn"),
    ]

    import pandas as pd

    rows = []
    for name, module_path in sources:
        try:
            __import__(module_path)
            status = "✅ Available"
        except ImportError:
            status = "❌ Not installed"
        except Exception:
            status = "⚠️ Import error"
        rows.append({"Data Source": name, "Status": status})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
