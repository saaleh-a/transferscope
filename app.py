"""TransferScope — Streamlit entry point.

Tactical Noir UI: dark precision instrument for football transfer intelligence.
"""

import streamlit as st

st.set_page_config(
    page_title="TransferScope",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject the Tactical Noir theme
from frontend.theme import inject_css
inject_css()

# Sidebar brand + navigation
st.sidebar.markdown(
    '<div class="ts-brand">TransferScope</div>'
    '<div class="ts-brand-sub">Transfer Intelligence</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Transfer Impact", "Shortlist Generator", "Hot or Not"],
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

if page == "Transfer Impact":
    from frontend.pages.transfer_impact import render
    render()
elif page == "Shortlist Generator":
    from frontend.pages.shortlist_generator import render
    render()
elif page == "Hot or Not":
    from frontend.pages.hot_or_not import render
    render()
