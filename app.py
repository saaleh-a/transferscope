"""TransferScope — Streamlit entry point.

Tactical Noir UI: dark precision instrument for football transfer intelligence.
"""

import logging
import os
import threading

import streamlit as st

_log = logging.getLogger(__name__)

st.set_page_config(
    page_title="TransferScope",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject the Tactical Noir theme
from frontend.theme import inject_css
inject_css()


# ── Cache warmup — run once per session so the first query is fast ───────────
if "cache_warmed" not in st.session_state:
    def _warmup():
        """Pre-load ClubElo data and Power Rankings in the background."""
        try:
            from backend.features import power_rankings
            power_rankings.compute_daily_rankings()
        except Exception:
            pass  # non-critical — user queries will still work

    threading.Thread(target=_warmup, daemon=True).start()
    st.session_state["cache_warmed"] = True


# ── Model status banner — read-only check, no training logic ─────────────────
from backend.models.transfer_portal import TransferPortalModel

_model_check = TransferPortalModel()
if _model_check.is_trained():
    st.success("✅ Trained model loaded.")
else:
    st.warning(
        "⚠️ No trained model found. Predictions are using the heuristic fallback. "
        "Run the training pipeline to enable ML predictions."
    )

# Sidebar brand + navigation
st.sidebar.markdown(
    '<div class="ts-brand">TransferScope</div>'
    '<div class="ts-brand-sub">Transfer Intelligence</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Transfer Impact", "Shortlist Generator", "Hot or Not", "Backtest Validator", "About & Methodology"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-family: \'DM Sans\', sans-serif; font-size: 0.72rem; '
    'color: #484F58; letter-spacing: 0.03em;">'
    'Based on Dinsdale &amp; Gallagher (2022)<br>'
    'Built for Arsenal scouting'
    '</div>',
    unsafe_allow_html=True,
)

# Clear cache button — useful after code updates or API issues
if st.sidebar.button("🗑️ Clear Cache", help="Clear cached API data to force fresh fetches"):
    try:
        from backend.data import cache
        cache.clear()
        st.session_state.pop("cache_warmed", None)
        st.sidebar.success("Cache cleared! Reload the page.")
    except Exception as e:
        st.sidebar.error(f"Failed to clear cache: {e}")

try:
    if page == "Transfer Impact":
        from frontend.pages.transfer_impact import render
        render()
    elif page == "Shortlist Generator":
        from frontend.pages.shortlist_generator import render
        render()
    elif page == "Hot or Not":
        from frontend.pages.hot_or_not import render
        render()
    elif page == "Backtest Validator":
        from frontend.pages.backtest_validator import render
        render()
    elif page == "About & Methodology":
        from frontend.pages.about import render
        render()
except Exception as exc:
    _log.exception("Page render failed for '%s'", page)
    st.error(
        f"⚠️ Failed to render **{page}** — an unexpected error occurred.\n\n"
        f"`{type(exc).__name__}: {exc}`\n\n"
        "Try refreshing the page or clearing the cache (sidebar)."
    )
