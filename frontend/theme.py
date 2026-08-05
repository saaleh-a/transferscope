"""TransferScope — Tactical Noir theme system.

Injects a cohesive dark-mode aesthetic across the entire Streamlit app:
deep charcoal base, amber/gold accents for data highlights, crimson
for alerts, monospaced data accents.  Every element is styled with
intent — no generic defaults.
"""

from __future__ import annotations

import math

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
    "text_secondary": "#A3ACB9",
    "text_muted": "#828B96",
    "accent_gold": "#D4A843",
    "accent_amber": "#E3A507",
    "accent_crimson": "#F45B69",       # Warm coral-crimson (was #DA3633)
    "accent_green": "#2DD4A8",         # Emerald (was #3FB950)
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
        tickfont=dict(color="#A3ACB9", size=10),
        title_font=dict(color="#A3ACB9", size=11),
    ),
    yaxis=dict(
        gridcolor="#21262D",
        zerolinecolor="#30363D",
        tickfont=dict(color="#A3ACB9", size=10),
        title_font=dict(color="#A3ACB9", size=11),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A3ACB9", size=11),
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


def page_header(title: str, subtitle: str = "", kicker: str = "") -> None:
    """Render the page masthead.

    Three tiers, unmistakable in two seconds: a small uppercase ``kicker``
    for category, the ``title`` as the single focal point, and a ``subtitle``
    that says what the page does in one plain sentence.
    """
    html = ['<div class="ts-masthead">']
    if kicker:
        html.append(f'<div class="ts-masthead-kicker">{kicker}</div>')
    html.append(f'<h1 class="ts-masthead-title">{title}</h1>')
    if subtitle:
        html.append(f'<p class="ts-masthead-sub">{subtitle}</p>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def empty_state(
    headline: str,
    body: str,
    examples: list[str] | None = None,
    footnote: str = "",
) -> None:
    """Render the pre-query state.

    Replaces a bare ``st.info`` restating the form labels.  An empty screen is
    the first thing most users see, so it carries the load: what the page
    gives back, a concrete example to copy, and an honest footnote about what
    the number is worth.
    """
    html = [
        '<div class="ts-empty">',
        '<div class="ts-empty-mark"></div>',
        f'<div class="ts-empty-headline">{headline}</div>',
        f'<p class="ts-empty-body">{body}</p>',
    ]
    if examples:
        html.append('<div class="ts-empty-examples">')
        html.append('<span class="ts-empty-examples-label">Try</span>')
        for ex in examples:
            html.append(f'<span class="ts-empty-chip">{ex}</span>')
        html.append("</div>")
    if footnote:
        html.append(f'<div class="ts-empty-footnote">{footnote}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def sidebar_status(ok: bool, label: str, detail: str = "") -> None:
    """Render a compact model-status pill in the sidebar.

    Kept out of the main column deliberately: a healthy system is not news,
    and a green banner in the hero position steals the focal point from the
    page title every single time.
    """
    color = COLORS["accent_green"] if ok else COLORS["accent_crimson"]
    detail_html = (
        f'<div class="ts-status-detail">{detail}</div>' if detail else ""
    )
    st.sidebar.markdown(
        f'<div class="ts-status-pill">'
        f'<span class="ts-status-dot" style="background:{color}; '
        f'box-shadow:0 0 6px {color}66;"></span>'
        f'<div><div class="ts-status-label">{label}</div>{detail_html}</div>'
        f"</div>",
        unsafe_allow_html=True,
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
    color_map = {"HOT": "#2DD4A8", "TEPID": "#E3A507", "NOT": "#F45B69"}
    color = color_map.get(verdict, "#A3ACB9")
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
    rating: float | None = None,
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
    if rating is not None:
        parts.append(f"<span>★ {rating:.2f}</span>")
    meta = '<span style="opacity:0.3; margin:0 0.3em;">·</span>'.join(parts)
    st.markdown(
        f'<div class="ts-player-header">'
        f'<div class="ts-player-name">{name}</div>'
        f'<div class="ts-player-meta">{meta}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Scalar display components ────────────────────────────────────────────────


def _tone_color(tone: str) -> str:
    """Map a semantic tone name to a Tactical Noir accent colour."""
    return {
        "positive": COLORS["accent_green"],
        "warning": COLORS["accent_amber"],
        "negative": COLORS["accent_crimson"],
        "info": COLORS["accent_blue"],
        "neutral": COLORS["text_secondary"],
    }.get(tone, COLORS["text_secondary"])


def tone_for_value(
    value: float, good_above: float = 5.0, bad_below: float = -5.0,
) -> str:
    """Classify a signed value into a semantic tone."""
    if value >= good_above:
        return "positive"
    if value <= bad_below:
        return "negative"
    return "warning"


def badge(text: str, tone: str = "neutral") -> str:
    """Return HTML for a small pill badge (use inside st.markdown).

    Tones: positive, warning, negative, info, neutral.
    """
    color = _tone_color(tone)
    return (
        f'<span style="display:inline-flex; align-items:center; gap:0.4em; '
        f'padding:0.25em 0.7em; border-radius:999px; font-size:0.75rem; '
        f'font-weight:600; letter-spacing:0.04em; text-transform:uppercase; '
        f'color:{color}; background:{color}1A; border:1px solid {color}40;">'
        f'<span style="width:6px; height:6px; border-radius:50%; '
        f'background:{color};"></span>{text}</span>'
    )


def score_ring(
    value: float,
    label: str = "",
    sublabel: str = "",
    vmin: float = -30.0,
    vmax: float = 30.0,
    size: int = 132,
    tone: str | None = None,
) -> str:
    """Return a self-contained SVG circular gauge as an HTML string.

    ``value`` is displayed verbatim (signed, one decimal); the arc length is
    the position of ``value`` within ``[vmin, vmax]``.  No JavaScript and no
    charting library — safe to drop into ``st.markdown(..., unsafe_allow_html=True)``.
    """
    span = (vmax - vmin) or 1.0
    frac = (value - vmin) / span
    frac = min(1.0, max(0.0, frac))

    color = _tone_color(tone if tone is not None else tone_for_value(value))

    radius = size / 2 - 10
    circumference = 2 * math.pi * radius
    dash = circumference * frac
    centre = size / 2

    label_html = ""
    if label:
        label_html = (
            f'<text x="{centre}" y="{centre + 20}" text-anchor="middle" '
            f'fill="{COLORS["text_secondary"]}" font-size="10" '
            f'letter-spacing="1.2" font-family="DM Sans, sans-serif">'
            f'{label.upper()}</text>'
        )

    sub_html = ""
    if sublabel:
        sub_html = (
            f'<div style="margin-top:0.4em; font-size:0.72rem; '
            f'color:{COLORS["text_muted"]}; text-align:center;">{sublabel}</div>'
        )

    return (
        f'<div style="display:flex; flex-direction:column; align-items:center;">'
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'role="img" aria-label="{label or "score"}: {value:+.1f}">'
        f'<circle cx="{centre}" cy="{centre}" r="{radius}" fill="none" '
        f'stroke="{COLORS["border"]}" stroke-width="8"/>'
        f'<circle cx="{centre}" cy="{centre}" r="{radius}" fill="none" '
        f'stroke="{color}" stroke-width="8" stroke-linecap="round" '
        f'stroke-dasharray="{dash:.2f} {circumference:.2f}" '
        f'transform="rotate(-90 {centre} {centre})"/>'
        f'<text x="{centre}" y="{centre + 4}" text-anchor="middle" '
        f'fill="{color}" font-size="24" font-weight="600" '
        f'font-family="JetBrains Mono, monospace">{value:+.1f}%</text>'
        f'{label_html}'
        f'</svg>{sub_html}</div>'
    )


def apply_plotly_theme(fig, title: str = "", **overrides):
    """Stamp the Tactical Noir layout onto a Plotly figure.

    Replaces the ``layout = dict(PLOTLY_LAYOUT); fig.update_layout(**layout)``
    boilerplate repeated across the chart components.
    """
    layout = dict(PLOTLY_LAYOUT)
    if title:
        layout["title"] = dict(text=title, **PLOTLY_LAYOUT["title"])
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


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
    --text-secondary: #A3ACB9;
    --text-muted: #828B96;
    --accent-gold: #D4A843;
    --accent-amber: #E3A507;
    --accent-crimson: #F45B69;
    --accent-green: #2DD4A8;
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

/* ── Masthead ─────────────────────────────────────────────────────────── */
/* One focal point per page. The rule anchors the block to the grid; the
   kicker gives category without competing for size. */
.ts-masthead {
    margin: 0 0 1.9rem 0;
    padding-left: 1.1rem;
    border-left: 3px solid var(--accent-gold);
}

.ts-masthead-kicker {
    font-family: 'Outfit', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent-gold);
    margin-bottom: 0.5rem;
}

.ts-masthead-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.035em !important;
    line-height: 1.05 !important;
    margin: 0 0 0.45rem 0 !important;
    padding: 0 !important;
    color: var(--text-primary) !important;
    /* No gradient fill: the accent is the rule, not the letterforms. */
    background: none !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}

.ts-masthead-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.98rem;
    font-weight: 400;
    line-height: 1.5;
    color: var(--text-secondary);
    margin: 0;
    max-width: 62ch;
}

/* ── Empty state ──────────────────────────────────────────────────────── */
/* The pre-query screen is most users' first impression. It teaches rather
   than restating the form labels directly above it. */
.ts-empty {
    position: relative;
    margin-top: 1.2rem;
    padding: 2.2rem 2.4rem;
    background: linear-gradient(160deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
}

.ts-empty-mark {
    position: absolute;
    top: -60px;
    right: -40px;
    width: 220px;
    height: 220px;
    border: 1px solid var(--accent-gold);
    border-radius: 50%;
    opacity: 0.10;
    pointer-events: none;
}

.ts-empty-headline {
    font-family: 'Outfit', sans-serif;
    font-size: 1.22rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text-primary);
    margin-bottom: 0.6rem;
}

.ts-empty-body {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    line-height: 1.65;
    color: var(--text-secondary);
    max-width: 68ch;
    margin: 0 0 1.35rem 0;
}

.ts-empty-examples {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1.35rem;
}

.ts-empty-examples-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-right: 0.3rem;
}

.ts-empty-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-primary);
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
}

.ts-empty-footnote {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    line-height: 1.6;
    color: var(--text-muted);
    padding-top: 1.05rem;
    border-top: 1px solid var(--border);
    max-width: 74ch;
}

/* ── Sidebar status pill ──────────────────────────────────────────────── */
.ts-status-pill {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.6rem 0.75rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 0.85rem;
}

.ts-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 0.42rem;
}

.ts-status-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--text-primary);
}

.ts-status-detail {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
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
