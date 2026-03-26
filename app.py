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


# ── Helper: check whether a trained model already exists ─────────────────────
def _model_is_trained() -> bool:
    """Return True if TF model weights + scaler exist on disk."""
    models_dir = os.path.join("data", "models")
    scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    portal_dir = os.path.join(models_dir, "transfer_portal")
    if not os.path.exists(scaler_path) or not os.path.isdir(portal_dir):
        return False
    return any(
        f.endswith(".keras") for f in os.listdir(portal_dir)
    )


# ── Thread-safe training status ──────────────────────────────────────────────
# Module-level dict + lock for cross-thread communication.  The main Streamlit
# thread copies values into session_state each rerun; the background thread
# only mutates this dict (never touches session_state directly).
_training_lock = threading.Lock()
_training_state: dict = {"status": "unknown", "step": ""}


def _run_training_background() -> None:
    """Run the training pipeline in a background thread."""
    try:
        with _training_lock:
            _training_state["status"] = "running"
            _training_state["step"] = "Importing pipeline…"

        from backend.models.training_pipeline import run_pipeline

        def _on_progress(step: str, detail: str = "") -> None:
            with _training_lock:
                _training_state["step"] = (
                    f"{step} — {detail}" if detail else step
                )

        success = run_pipeline(
            seasons_back=3,
            league_codes=["ENG1", "ESP1", "GER1", "ITA1", "FRA1"],
            api_delay=2.0,
            progress_callback=_on_progress,
        )

        with _training_lock:
            if success:
                _training_state["status"] = "done"
                _training_state["step"] = "Training complete ✅"
            else:
                _training_state["status"] = "failed"
                _training_state["step"] = (
                    "Training failed — insufficient data from API"
                )
    except Exception as exc:
        _log.exception("Background training failed")
        with _training_lock:
            _training_state["status"] = "failed"
            _training_state["step"] = f"Training error: {exc}"


def _start_training_thread() -> None:
    """Launch a training thread if one isn't already running."""
    with _training_lock:
        if _training_state["status"] == "running":
            return  # already in progress — don't spawn a second thread
        _training_state["status"] = "starting"
        _training_state["step"] = "Queued…"
    threading.Thread(target=_run_training_background, daemon=True).start()


# ── Sync module-level state → session_state each rerun ───────────────────────
def _sync_training_status() -> None:
    with _training_lock:
        st.session_state["training_status"] = _training_state["status"]
        st.session_state["training_step"] = _training_state["step"]


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


# ── Auto-train: kick off training if no model exists ─────────────────────────
if "training_checked" not in st.session_state:
    st.session_state["training_checked"] = True
    if _model_is_trained():
        with _training_lock:
            _training_state["status"] = "done"
            _training_state["step"] = "Trained model loaded ✅"
    else:
        _start_training_thread()

# Copy background thread status into session_state for this rerun
_sync_training_status()


# Sidebar brand + navigation
st.sidebar.markdown(
    '<div class="ts-brand">TransferScope</div>'
    '<div class="ts-brand-sub">Transfer Intelligence</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

# ── Training status indicator ────────────────────────────────────────────────
_ts = st.session_state.get("training_status", "unknown")
_step = st.session_state.get("training_step", "")

if _ts in ("starting", "running"):
    st.sidebar.info(f"🔄 **Model training in progress**\n\n{_step}")
    st.sidebar.caption(
        "Predictions use the heuristic fallback until training finishes. "
        "This runs once — the trained model is cached for future sessions."
    )
elif _ts == "failed":
    st.sidebar.warning(f"⚠️ **Training incomplete**\n\n{_step}")
    if st.sidebar.button("🔁 Retry Training"):
        _start_training_thread()
        st.rerun()
elif _ts == "done":
    st.sidebar.success("✅ **Trained model active**")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Transfer Impact", "Shortlist Generator", "Hot or Not", "About & Methodology"],
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

if page == "Transfer Impact":
    from frontend.pages.transfer_impact import render
    render()
elif page == "Shortlist Generator":
    from frontend.pages.shortlist_generator import render
    render()
elif page == "Hot or Not":
    from frontend.pages.hot_or_not import render
    render()
elif page == "About & Methodology":
    from frontend.pages.about import render
    render()
