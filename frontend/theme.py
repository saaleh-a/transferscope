"""TransferScope — Tactical Noir theme system.

Injects a cohesive dark-mode aesthetic across the entire Streamlit app:
deep charcoal base, amber/gold accents for data highlights, crimson
for alerts, monospaced data accents.  Every element is styled with
intent — no generic defaults.
"""

from __future__ import annotations

import streamlit as st

# ── Color palette ────────────────────────────────────────────────────────────

COLORS = {
    "bg_primary": "#0E1117",
    "bg_secondary": "#161B22",
    "bg_card": "#1C2128",
    "bg_hover": "#21262D",
    "border": "#30363D",
    "border_accent": "#D4A843",
    "text_primary": "#C9D1D9",
    "text_secondary": "#8B949E",
    "text_muted": "#484F58",
    "accent_gold": "#D4A843",
    "accent_amber": "#E3A507",
    "accent_crimson": "#DA3633",
    "accent_green": "#3FB950",
    "accent_blue": "#58A6FF",
    "accent_teal": "#39D2C0",
    "gradient_start": "#D4A843",
    "gradient_end": "#E3A507",
}

# ── Plotly chart theme ───────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#161B22",
    font=dict(
        family="'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
        color="#C9D1D9",
        size=12,
    ),
    title=dict(
        font=dict(
            family="'DM Sans', 'Outfit', sans-serif",
            size=16,
            color="#C9D1D9",
        ),
        x=0,
        xanchor="left",
    ),
    xaxis=dict(
        gridcolor="#21262D",
        zerolinecolor="#30363D",
        tickfont=dict(color="#8B949E", size=10),
        title_font=dict(color="#8B949E", size=11),
    ),
    yaxis=dict(
        gridcolor="#21262D",
        zerolinecolor="#30363D",
        tickfont=dict(color="#8B949E", size=10),
        title_font=dict(color="#8B949E", size=11),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8B949E", size=11),
    ),
    margin=dict(l=10, r=10, t=50, b=30),
)


def inject_css() -> None:
    """Inject the Tactical Noir stylesheet into the current page."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Helper components ────────────────────────────────────────────────────────


def section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section divider."""
    html = (
        f'<div class="ts-section-header">'
        f'<div class="ts-section-rule"></div>'
        f'<h3 class="ts-section-title">{title}</h3>'
    )
    if subtitle:
        html += f'<p class="ts-section-subtitle">{subtitle}</p>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def stat_card(label: str, value: str, delta: str = "", delta_positive: bool = True) -> str:
    """Return HTML for a single stat card (use inside st.markdown)."""
    delta_class = "positive" if delta_positive else "negative"
    delta_html = ""
    if delta:
        arrow = "↑" if delta_positive else "↓"
        delta_html = (
            f'<span class="ts-stat-delta {delta_class}">'
            f'{arrow} {delta}</span>'
        )
    return (
        f'<div class="ts-stat-card">'
        f'<span class="ts-stat-label">{label}</span>'
        f'<span class="ts-stat-value">{value}</span>'
        f'{delta_html}'
        f'</div>'
    )


def confidence_badge(level: str, weight: float, minutes: int) -> None:
    """Render the RAG confidence indicator with the Tactical Noir style."""
    color_map = {
        "green": COLORS["accent_green"],
        "amber": COLORS["accent_amber"],
        "red": COLORS["accent_crimson"],
    }
    color = color_map.get(level, COLORS["text_muted"])
    glow = f"0 0 8px {color}40"

    st.markdown(
        f'<div class="ts-confidence-badge">'
        f'<span class="ts-confidence-dot" style="background:{color}; box-shadow:{glow};"></span>'
        f'<span class="ts-confidence-text">'
        f'<strong style="color:{color};">{level.upper()}</strong>'
        f' — weight {weight:.2f} · {minutes:,} mins'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def verdict_display(verdict: str, player: str, source: str, target: str) -> None:
    """Render the large Hot/Tepid/Not verdict block."""
    color_map = {"HOT": "#3FB950", "TEPID": "#E3A507", "NOT": "#DA3633"}
    color = color_map.get(verdict, "#8B949E")
    st.markdown(
        f'<div class="ts-verdict-block">'
        f'<div class="ts-verdict-label" style="color:{color};">{verdict}</div>'
        f'<div class="ts-verdict-detail">'
        f'{player} <span class="ts-verdict-arrow">→</span> {target}'
        f'</div>'
        f'<div class="ts-verdict-from">from {source}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def player_info_card(
    name: str,
    team: str = "",
    position: str = "",
    minutes: int = 0,
    season_label: str = "Current",
) -> None:
    """Render a consistent player info card across all pages."""
    parts = []
    if team:
        parts.append(f'<span><span class="ts-gold">◆</span> {team}</span>')
    if position:
        parts.append(f"<span>{position}</span>")
    if minutes:
        parts.append(f"<span>{minutes:,} mins</span>")
    if season_label:
        parts.append(f"<span>{season_label}</span>")
    meta = "".join(parts)
    st.markdown(
        f'<div class="ts-player-header">'
        f'<div class="ts-player-name">{name}</div>'
        f'<div class="ts-player-meta">{meta}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Master CSS ───────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300;1,9..40,400&family=JetBrains+Mono:wght@300;400;500;600&family=Outfit:wght@200;300;400;500;600;700;800&display=swap');

/* ── Global resets ────────────────────────────────────────────────────── */
:root {
    --bg-primary: #0E1117;
    --bg-secondary: #161B22;
    --bg-card: #1C2128;
    --bg-hover: #21262D;
    --border: #30363D;
    --border-accent: #D4A843;
    --text-primary: #C9D1D9;
    --text-secondary: #8B949E;
    --text-muted: #484F58;
    --accent-gold: #D4A843;
    --accent-amber: #E3A507;
    --accent-crimson: #DA3633;
    --accent-green: #3FB950;
    --accent-blue: #58A6FF;
    --accent-teal: #39D2C0;
    --radius: 6px;
    --radius-lg: 12px;
}

/* Override Streamlit's base font */
html, body, [class*="css"] {
    font-family: 'DM Sans', 'Outfit', -apple-system, sans-serif !important;
    color: var(--text-primary);
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] .stRadio > label {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 300 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
}

[data-testid="stSidebar"] .stRadio > div > label {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    text-transform: none !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 0.8rem !important;
    border-radius: var(--radius) !important;
    transition: all 0.15s ease !important;
    border: 1px solid transparent !important;
}

[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: var(--bg-hover) !important;
    border-color: var(--border) !important;
}

[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div [aria-checked="true"] + label {
    background: var(--bg-card) !important;
    border-color: var(--accent-gold) !important;
    color: var(--accent-gold) !important;
}

/* ── Page headers ─────────────────────────────────────────────────────── */
h1 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-amber)) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}

/* Streamlit header elements */
.stMarkdown h1, [data-testid="stHeader"] h1 {
    font-size: 2.2rem !important;
}

/* Captions under headers */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 300 !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.02em !important;
}

/* ── Input elements ───────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s ease !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 0 1px var(--accent-gold), 0 0 12px rgba(212, 168, 67, 0.1) !important;
}

.stTextInput label, .stSelectbox label, .stNumberInput label,
.stMultiSelect label, .stSlider label {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border-radius: var(--radius) !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--border) !important;
    padding: 0.5rem 1.5rem !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-amber)) !important;
    color: #0E1117 !important;
    border: none !important;
    font-weight: 700 !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 4px 20px rgba(212, 168, 67, 0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── DataFrames / Tables ──────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}

[data-testid="stDataFrame"] th {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    color: var(--accent-gold) !important;
    background: var(--bg-card) !important;
}

[data-testid="stDataFrame"] td {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: var(--text-primary) !important;
}

/* ── Metric cards ─────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.2rem !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 400 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-size: 0.7rem !important;
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* ── Sliders ──────────────────────────────────────────────────────────── */
.stSlider [data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Progress bar ─────────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-gold), var(--accent-amber)) !important;
}

/* ── Dividers ─────────────────────────────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    opacity: 0.4 !important;
}

/* ── Alerts / Info boxes ──────────────────────────────────────────────── */
.stAlert {
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    border-left: 3px solid var(--accent-gold) !important;
}

/* ── Expanders ────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}

/* ── Custom components ────────────────────────────────────────────────── */

/* Section header */
.ts-section-header {
    margin: 2rem 0 1.2rem 0;
}

.ts-section-rule {
    height: 1px;
    background: linear-gradient(90deg, var(--accent-gold), transparent);
    margin-bottom: 0.8rem;
    opacity: 0.6;
}

.ts-section-title {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    letter-spacing: -0.01em !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
}

.ts-section-subtitle {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 300 !important;
    font-size: 0.85rem !important;
    color: var(--text-muted) !important;
    margin: 0.2rem 0 0 0 !important;
}

/* Stat cards */
.ts-stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1rem 1.2rem;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}

.ts-stat-label {
    font-family: 'Outfit', sans-serif;
    font-weight: 400;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.ts-stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 1.6rem;
    color: var(--text-primary);
    line-height: 1;
}

.ts-stat-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
}

.ts-stat-delta.positive { color: var(--accent-green); }
.ts-stat-delta.negative { color: var(--accent-crimson); }

/* Confidence badge */
.ts-confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.4rem 1rem;
    margin: 0.5rem 0;
}

.ts-confidence-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

.ts-confidence-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: var(--text-secondary);
}

/* Verdict block */
.ts-verdict-block {
    text-align: center;
    padding: 2.5rem 1rem;
    margin: 1rem 0;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    position: relative;
    overflow: hidden;
}

.ts-verdict-block::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
}

.ts-verdict-label {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 4.5rem;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.6rem;
    text-shadow: 0 0 40px currentColor;
}

.ts-verdict-detail {
    font-family: 'DM Sans', sans-serif;
    font-weight: 400;
    font-size: 1.2rem;
    color: var(--text-primary);
}

.ts-verdict-arrow {
    color: var(--accent-gold);
    font-weight: 300;
    margin: 0 0.3rem;
}

.ts-verdict-from {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* Player header card */
.ts-player-header {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}

.ts-player-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-gold), transparent 70%);
}

.ts-player-name {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0;
}

.ts-player-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
    display: flex;
    gap: 1.2rem;
}

.ts-player-meta span {
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

.ts-player-meta .ts-gold { color: var(--accent-gold); }

/* Logo / brand in sidebar */
.ts-brand {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-amber));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.ts-brand-sub {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* Transfer history table */
.ts-transfer-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
}

.ts-transfer-date {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    min-width: 90px;
}

.ts-transfer-clubs {
    color: var(--text-primary);
}

.ts-transfer-type {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--accent-gold);
    margin-left: auto;
}

/* Multiselect pills */
.stMultiSelect [data-baseweb="tag"] {
    background: var(--bg-hover) !important;
    border: 1px solid var(--accent-gold) !important;
    border-radius: 999px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: var(--accent-gold) !important;
}

/* Tabs (if used) */
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}

.stTabs [aria-selected="true"] {
    border-bottom-color: var(--accent-gold) !important;
    color: var(--accent-gold) !important;
}

/* Plotly chart containers */
[data-testid="stPlotlyChart"] {
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    padding: 0.5rem;
    background: var(--bg-secondary);
}
</style>
"""
