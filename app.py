"""TransferScope — Streamlit entry point."""

import streamlit as st

st.set_page_config(
    page_title="TransferScope",
    page_icon="⚽",
    layout="wide",
)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Transfer Impact", "Shortlist Generator", "Hot or Not"],
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
