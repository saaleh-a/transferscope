"""Generate the TransferScope White Paper PDF using ReportLab."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, HRFlowable, ListFlowable, ListItem,
)
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate
from reportlab.lib import colors
import os

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TransferScope_White_Paper.pdf")

# ── Page dimensions ──────────────────────────────────────────────────────────
WIDTH, HEIGHT = A4
MARGIN = 1 * inch

# ── Colours ──────────────────────────────────────────────────────────────────
ARSENAL_RED = HexColor("#EF0107")
DARK_NAVY = HexColor("#0B1D3A")
MEDIUM_NAVY = HexColor("#132743")
LIGHT_GREY = HexColor("#F5F5F5")
MID_GREY = HexColor("#E0E0E0")
TEXT_DARK = HexColor("#1A1A2E")
ACCENT_BLUE = HexColor("#2563EB")
ACCENT_GREEN = HexColor("#16A34A")
ACCENT_AMBER = HexColor("#D97706")
ACCENT_RED_LIGHT = HexColor("#DC2626")


def build_styles():
    """Create all paragraph styles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "WPTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        textColor=white,
        alignment=TA_CENTER,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        "WPSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=HexColor("#CBD5E1"),
        alignment=TA_CENTER,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "WPAuthor",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=11,
        leading=15,
        textColor=HexColor("#94A3B8"),
        alignment=TA_CENTER,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=22,
        textColor=DARK_NAVY,
        spaceBefore=24,
        spaceAfter=10,
        borderPadding=(0, 0, 4, 0),
    ))
    styles.add(ParagraphStyle(
        "SubHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=MEDIUM_NAVY,
        spaceBefore=14,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "BodyText2",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14.5,
        textColor=TEXT_DARK,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "Formula",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=10,
        leading=14,
        textColor=DARK_NAVY,
        alignment=TA_CENTER,
        spaceBefore=8,
        spaceAfter=8,
        backColor=LIGHT_GREY,
        borderPadding=8,
    ))
    styles.add(ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=HexColor("#64748B"),
        alignment=TA_CENTER,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=white,
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=TEXT_DARK,
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "BulletBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=TEXT_DARK,
        alignment=TA_JUSTIFY,
        leftIndent=18,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "FooterStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=HexColor("#94A3B8"),
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "AbstractBody",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=14.5,
        textColor=TEXT_DARK,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leftIndent=20,
        rightIndent=20,
    ))

    return styles


def header_footer(canvas, doc):
    """Draw header line and footer on each page."""
    canvas.saveState()
    # Header line
    canvas.setStrokeColor(DARK_NAVY)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, HEIGHT - MARGIN + 10, WIDTH - MARGIN, HEIGHT - MARGIN + 10)
    # Footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#94A3B8"))
    canvas.drawCentredString(WIDTH / 2, 0.6 * inch,
                             f"TransferScope White Paper  |  Page {doc.page}  |  Confidential")
    canvas.restoreState()


def first_page(canvas, doc):
    """No header/footer on the cover page."""
    pass


def make_table(headers, rows, col_widths=None):
    """Build a styled table."""
    s = build_styles()
    header_cells = [Paragraph(h, s["TableHeader"]) for h in headers]
    data = [header_cells]
    for row in rows:
        data.append([Paragraph(str(c), s["TableCell"]) for c in row])

    if col_widths is None:
        col_widths = [(WIDTH - 2 * MARGIN) / len(headers)] * len(headers)

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.5, MID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def build_cover(s):
    """Build the cover page elements."""
    elements = []
    elements.append(Spacer(1, 1.8 * inch))

    # Dark cover block using a table with background
    cover_data = [
        [Paragraph("TransferScope", s["WPTitle"])],
        [Paragraph("A Predictive Framework for<br/>Cross-League Performance Translation", s["WPSubtitle"])],
        [Spacer(1, 16)],
        [Paragraph("Saaleh Dinsdale", s["WPAuthor"])],
        [Paragraph("Microsoft  |  March 2026", s["WPAuthor"])],
        [Spacer(1, 8)],
        [Paragraph("Built with Claude Code (Anthropic Claude Opus 4.6)", s["WPAuthor"])],
    ]
    cover_table = Table(cover_data, colWidths=[WIDTH - 2 * MARGIN])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_NAVY),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 30),
        ("RIGHTPADDING", (0, 0), (-1, -1), 30),
        ("TOPPADDING", (0, 0), (0, 0), 40),
        ("BOTTOMPADDING", (-1, -1), (-1, -1), 30),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
    ]))
    elements.append(cover_table)

    elements.append(Spacer(1, 0.6 * inch))

    # Tagline
    elements.append(Paragraph(
        "Solving the League Tax problem through hierarchical Elo modelling, "
        "multi-head deep neural networks, and Bayesian uncertainty quantification.",
        ParagraphStyle("CoverTag", parent=s["BodyText2"], alignment=TA_CENTER,
                       fontSize=11, leading=16, textColor=HexColor("#475569")),
    ))

    elements.append(PageBreak())
    return elements


def build_body(s):
    """Build all body sections."""
    el = []

    # ── TABLE OF CONTENTS ────────────────────────────────────────────────
    el.append(Paragraph("Contents", s["SectionHeading"]))
    toc_items = [
        "1. Abstract",
        "2. The Problem Statement: The Material Limits of Descriptive Scouting",
        "3. Data Architecture &amp; Hierarchical Representation",
        "4. Modeling: The Multi-Head Deep Neural Network",
        "5. Adjustment Models: Sklearn Linear Regression Priors",
        "6. Bayesian Reliability &amp; Uncertainty Quantification",
        "7. Case Study: Arsenal Tactical Integration",
        "8. Development Methodology: Agentic AI Orchestration",
        "9. Shortlist Scoring Methodology",
        "10. Conclusion &amp; Future Roadmap",
        "References",
    ]
    for item in toc_items:
        el.append(Paragraph(item, ParagraphStyle(
            "TOCItem", parent=s["BodyText2"], leftIndent=20, spaceAfter=4, fontSize=10,
        )))
    el.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # 1. ABSTRACT
    # ══════════════════════════════════════════════════════════════════════
    el.append(Paragraph("1. Abstract", s["SectionHeading"]))
    el.append(HRFlowable(width="100%", thickness=1, color=DARK_NAVY, spaceAfter=10))

    el.append(Paragraph(
        "TransferScope is a predictive intelligence platform that solves the "
        "<b>League Tax</b> problem: the systematic error introduced when scouts "
        "compare raw per-90 statistics across leagues of fundamentally different "
        "competitive quality. A midfielder registering 2.1 progressive passes per 90 "
        "in the Eredivisie is not equivalent to one registering 1.8 in the Premier "
        "League. TransferScope translates player output across these contexts.",
        s["AbstractBody"],
    ))
    el.append(Paragraph(
        "Building on the foundational research of Dinsdale &amp; Gallagher (2022), "
        "which demonstrated a <b>49% accuracy improvement</b> over traditional "
        "descriptive baselines, TransferScope implements a production-grade system "
        "that predicts how a player's per-90 metrics will change at a specific "
        "target club. The platform models the interaction between player traits, "
        "team tactical identity, and league competitive quality through a "
        "43-dimensional feature space processed by a multi-head deep neural network.",
        s["AbstractBody"],
    ))
    el.append(Paragraph(
        "The entire platform was architecturally orchestrated using Claude Code "
        "(Anthropic's Claude Opus 4.6), representing a paradigm shift in how "
        "analytical tools are conceived, designed, and deployed. The result is a "
        "Streamlit-based application with full data pipeline, feature engineering, "
        "ML prediction, and interactive frontend, built to the standards expected "
        "of enterprise software.",
        s["AbstractBody"],
    ))
    el.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════
    # 2. THE PROBLEM STATEMENT
    # ══════════════════════════════════════════════════════════════════════
    el.append(Paragraph("2. The Problem Statement: The Material Limits of Descriptive Scouting",
                         s["SectionHeading"]))
    el.append(HRFlowable(width="100%", thickness=1, color=DARK_NAVY, spaceAfter=10))

    el.append(Paragraph("2.1 The Translation Error", s["SubHeading"]))
    el.append(Paragraph(
        "The fundamental assumption underlying most scouting platforms is that "
        "per-90 statistics are directly comparable across leagues. This assumption "
        "is false. A player's statistical output is not an intrinsic property of the "
        "player alone; it is a joint function of the player's ability and the "
        "competitive environment in which they operate.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "Consider a central midfielder registering 2.1 progressive passes per 90 in "
        "the Eredivisie versus one registering 1.8 in the Premier League. A naive "
        "comparison would favour the Dutch-league player. However, the Eredivisie's "
        "lower average defensive compactness, slower transition speed, and reduced "
        "pressing intensity create an environment where progressive passing is "
        "systematically easier. The 1.8 figure in the Premier League may represent "
        "a superior underlying ability. This systematic distortion is the "
        "<b>League Tax</b>: a hidden multiplier that inflates statistics in weaker "
        "leagues and deflates them in stronger ones.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "The League Tax is not uniform across metrics. Shooting metrics (xG, shots) "
        "are more sensitive to defensive quality than passing volume. Dribbling "
        "success rates compress more sharply at higher competitive levels. Defensive "
        "actions are highly dependent on team tactical systems rather than individual "
        "ability. Any model that applies a single league-level correction factor "
        "across all metrics is, by construction, inaccurate.",
        s["BodyText2"],
    ))

    el.append(Paragraph("2.2 The Principal Contradiction", s["SubHeading"]))
    el.append(Paragraph(
        "Traditional scouting assumes performance is <b>static and portable</b>: "
        "that a player's per-90 profile at Club A will be replicated at Club B. "
        "TransferScope recognises performance as a <b>dialectical interaction</b> "
        "between player traits and the competitive environment. A player is not a "
        "fixed bundle of statistics. They are a function of their context: the "
        "tactical system, the quality of teammates, the quality of opposition, "
        "and the pace and physicality of the league.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "This is the principal contradiction that TransferScope resolves. Rather "
        "than asking \"what are this player's stats?\" the platform asks "
        "\"what would this player's stats be in a different context?\" This is a "
        "fundamentally different question, and it requires a fundamentally different "
        "modelling approach.",
        s["BodyText2"],
    ))

    el.append(Paragraph("2.3 Industry Context", s["SubHeading"]))
    el.append(Paragraph(
        "Existing approaches fall short in predictable ways. <b>FBRef</b> provides "
        "raw percentile rankings within a single league, but offers no mechanism "
        "for cross-league comparison. <b>BALLER API</b> and similar platforms use "
        "categorical binning (\"likely to improve\" vs. \"likely to decline\"), "
        "which discards critical magnitude information: a 2% decline in xG is "
        "fundamentally different from a 25% decline, yet both are classified as "
        "\"decline.\" Academic research (Dinsdale &amp; Gallagher, 2022; Pappalardo "
        "et al., 2019) has established rigorous frameworks, but the gap between "
        "published research and production-grade tooling remains vast. TransferScope "
        "bridges this gap.",
        s["BodyText2"],
    ))
    el.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # 3. DATA ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════════
    el.append(Paragraph("3. Data Architecture &amp; Hierarchical Representation",
                         s["SectionHeading"]))
    el.append(HRFlowable(width="100%", thickness=1, color=DARK_NAVY, spaceAfter=10))

    el.append(Paragraph("3.1 The Hierarchical Elo Engine", s["SubHeading"]))
    el.append(Paragraph(
        "Following the methodology of Dinsdale &amp; Gallagher (2022), TransferScope "
        "constructs a four-level ability hierarchy to quantify the competitive "
        "environment. Every team in the world is assigned a composite Power Ranking "
        "derived from four levels:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "Power Ranking = P<sub>continent</sub> + P<sub>country</sub> + "
        "P<sub>league</sub> + P<sub>team</sub>",
        s["Formula"],
    ))
    el.append(Paragraph(
        "This raw composite score is then normalised to a 0-100 scale daily across "
        "all clubs in both data sources combined:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "normalized_score = (raw_elo - global_min) / (global_max - global_min) x 100",
        s["Formula"],
    ))
    el.append(Paragraph(
        "On any given date, the best team in the world scores 100 and the worst "
        "scores 0. This normalisation creates a universal currency for competitive "
        "quality that is invariant to the absolute scale of the underlying Elo systems.",
        s["BodyText2"],
    ))

    el.append(Paragraph("3.2 Global Data Routing: The Elo Router", s["SubHeading"]))
    el.append(Paragraph(
        "No single Elo data source covers the global football ecosystem. "
        "TransferScope implements an <b>Elo Router</b> that merges two complementary "
        "sources:",
        s["BodyText2"],
    ))

    router_data = [
        ["Source", "Coverage", "Method", "Priority"],
        ["ClubElo (via soccerdata)", "European clubs (~700 teams)\nacross 30+ leagues",
         "Python API\n(sd.ClubElo)", "Primary for European clubs"],
        ["WorldFootballElo\n(eloratings.net)", "Global (~20,000 teams)\nall confederations",
         "HTTP scrape with\nregex parsing", "Fallback for non-European\nclubs"],
    ]
    el.append(make_table(
        router_data[0], router_data[1:],
        col_widths=[1.6 * inch, 1.6 * inch, 1.3 * inch, 1.6 * inch],
    ))
    el.append(Paragraph(
        "Table 1: Elo data source routing. ClubElo takes priority for European clubs "
        "due to superior granularity (league-level Elo decomposition). "
        "WorldFootballElo provides coverage for South American, North American, "
        "Asian, and African leagues.",
        s["Caption"],
    ))

    el.append(Paragraph(
        "This dual-source architecture ensures coverage of all leagues present on "
        "FotMob, including: Brasileirao Serie A/B, Argentine Primera Division, "
        "Colombian Primera A, Chilean Primera, Uruguayan Primera, Ecuadorian Serie A, "
        "MLS, Saudi Pro League, and J-League. The router is designed for extensibility: "
        "adding a new Elo source requires implementing a single function conforming "
        "to the established interface.",
        s["BodyText2"],
    ))

    el.append(Paragraph("3.3 Dynamic League Power Rankings", s["SubHeading"]))
    el.append(Paragraph(
        "Rather than relying on a static tier table (e.g. \"Tier 1: Premier League, "
        "La Liga; Tier 2: Bundesliga, Serie A\"), TransferScope computes league "
        "quality dynamically from the teams that constitute each league on a given date:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "league_elo = mean(team_elo for all teams in league on transfer_date)",
        s["Formula"],
    ))
    el.append(Paragraph(
        "Per-league statistics stored for each daily snapshot include: mean, standard "
        "deviation, and percentile bands (10th, 25th, 50th, 75th, 90th). These "
        "percentile bands power the swarm plot visualisations in the frontend, "
        "providing contextual benchmarks for player comparison.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "A critical derived feature is <b>relative ability</b>:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "relative_ability = team_normalized_score - league_mean_normalized_score",
        s["Formula"],
    ))
    el.append(Paragraph(
        "Positive values indicate a team stronger than their league average; negative "
        "values indicate a team weaker than average. This feature captures the "
        "within-league competitive position, which is distinct from absolute quality. "
        "Arsenal (normalised ~85) in the Premier League (mean ~72) has a relative "
        "ability of +13, while a team rated 85 in a weaker league would have a much "
        "higher relative ability. The <b>change in relative ability</b> between "
        "source and target club is a primary predictor of performance translation.",
        s["BodyText2"],
    ))

    el.append(Paragraph("3.4 Metric Selection: 23 Per-90 Indicators", s["SubHeading"]))
    el.append(Paragraph(
        "TransferScope extracts 23 per-90 metrics from FotMob via the mobfot Python "
        "package. The 13 core metrics are drawn directly from Dinsdale &amp; Gallagher "
        "(2022), with 10 additional metrics selected to capture tactical dimensions "
        "not covered by the original paper:",
        s["BodyText2"],
    ))

    core_metrics = [
        ["1", "xG", "expected_goals", "Shooting quality"],
        ["2", "xA", "expected_assists", "Creative quality"],
        ["3", "Shots", "shots", "Shooting volume"],
        ["4", "Take-ons", "successful_dribbles", "Ball carrying"],
        ["5", "Crosses", "successful_crosses", "Wide delivery"],
        ["6", "Pen. Area Entries", "touches_in_opposition_box", "Positional intelligence"],
        ["7", "Total Passes", "successful_passes", "Passing volume"],
        ["8", "Short Passes", "pass_completion_pct", "Passing accuracy (proxy)"],
        ["9", "Long Passes", "accurate_long_balls", "Ball progression"],
        ["10", "Passes Att 3rd", "chances_created", "Attacking creation"],
        ["11", "Def Own 3rd", "clearances", "Deep defending"],
        ["12", "Def Mid 3rd", "interceptions", "Midfield defending"],
        ["13", "Def Att 3rd", "possession_won_final_3rd", "High pressing"],
    ]
    el.append(make_table(
        ["#", "Paper Metric", "FotMob Field", "Captures"],
        core_metrics,
        col_widths=[0.35 * inch, 1.3 * inch, 2.2 * inch, 2.2 * inch],
    ))
    el.append(Paragraph("Table 2: The 13 core metrics mapped from Dinsdale &amp; Gallagher (2022) to FotMob fields.", s["Caption"]))

    el.append(Paragraph(
        "The 10 additional metrics extend coverage to include: <b>xGOT</b> "
        "(shot placement quality), <b>Non-penalty xG</b> (open-play shooting "
        "isolated from penalty variance), <b>Dispossessed</b> (ball retention "
        "under pressure), <b>Duels won %</b> and <b>Aerial duels won %</b> "
        "(physical contests), <b>Recoveries</b> (pressing effectiveness independent "
        "of defensive system), <b>Fouls won</b> (ability to draw fouls as an "
        "attacking weapon), <b>Touches</b> (overall involvement), and "
        "<b>Goals/xG conceded on pitch</b> (defensive contribution at team level).",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "The inclusion of <b>Touches in Opposition Box</b> is particularly "
        "significant. This metric measures positional intelligence and attacking "
        "involvement in a way that xG alone cannot capture. A player who consistently "
        "arrives in dangerous positions but receives poor service will have low xG "
        "but high opposition box touches, revealing latent attacking quality that "
        "would be unlocked by better creative teammates.",
        s["BodyText2"],
    ))
    el.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # 4. MODELING: MULTI-HEAD DNN
    # ══════════════════════════════════════════════════════════════════════
    el.append(Paragraph("4. Modeling: The Multi-Head Deep Neural Network", s["SectionHeading"]))
    el.append(HRFlowable(width="100%", thickness=1, color=DARK_NAVY, spaceAfter=10))

    el.append(Paragraph("4.1 Feature Space Construction", s["SubHeading"]))
    el.append(Paragraph(
        "Each transfer prediction is constructed from a <b>43-dimensional feature "
        "vector</b> that encodes the player's current output, their competitive "
        "context, and the target competitive context:",
        s["BodyText2"],
    ))

    feat_data = [
        ["Player per-90 metrics (current club)", "13", "Raw output in current context"],
        ["Team ability: current", "1", "Normalised 0-100 Power Ranking"],
        ["Team ability: target", "1", "Normalised 0-100 Power Ranking"],
        ["League ability: current", "1", "League mean normalised score"],
        ["League ability: target", "1", "League mean normalised score"],
        ["Team-position per-90: current", "13", "Positional averages at current club"],
        ["Team-position per-90: target", "13", "Positional averages at target club"],
    ]
    el.append(make_table(
        ["Feature Group", "Dims", "Description"],
        feat_data,
        col_widths=[2.2 * inch, 0.6 * inch, 3.3 * inch],
    ))
    el.append(Paragraph("Table 3: The 43-dimensional input feature vector.", s["Caption"]))

    el.append(Paragraph(
        "The team-position features are critical. They encode not just \"how good is "
        "the target team\" but \"what does this team's tactical system demand from "
        "players in this position.\" Arsenal's right winger position average differs "
        "dramatically from, say, Burnley's, even when both play in the Premier "
        "League. This is what allows the model to predict an \"Arsenal profile\" "
        "rather than just a \"Premier League profile.\"",
        s["BodyText2"],
    ))

    el.append(Paragraph("4.2 Four-Group Multi-Head Architecture", s["SubHeading"]))
    el.append(Paragraph(
        "The model is decomposed into four separate neural networks, one per skill "
        "domain. This architectural choice is driven by a key insight from the "
        "transfer literature: <b>different skill categories regress at different "
        "rates</b> when a player changes competitive context.",
        s["BodyText2"],
    ))

    group_data = [
        ["1 - Shooting", "xG, Shots", "2", "Most sensitive to defensive quality"],
        ["2 - Passing", "xA, Crosses, Total Passes,\nShort Passes, Long Passes,\nPasses Att 3rd, Pen. Area Entries", "7", "Volume may persist while\naccuracy drops"],
        ["3 - Dribbling", "Take-ons", "1", "Highly individual, resistant\nto league-level regression"],
        ["4 - Defending", "Clearances, Interceptions,\nPoss. Won Final 3rd", "3", "Most dependent on team\ntactical system"],
    ]
    el.append(make_table(
        ["Group", "Targets", "Heads", "Regression Characteristics"],
        group_data,
        col_widths=[1.1 * inch, 2.0 * inch, 0.6 * inch, 2.4 * inch],
    ))
    el.append(Paragraph("Table 4: The four model groups and their output targets.", s["Caption"]))

    el.append(Paragraph(
        "A player's passing volume may remain relatively stable when moving to a "
        "harder league (they still receive and distribute the ball), but their "
        "shooting efficiency will drop as they face better-organised defences. "
        "Dribbling success is largely a function of individual skill and physical "
        "attributes, making it more resistant to league-level adjustment. Defensive "
        "actions are almost entirely determined by the team's tactical system: a "
        "high-pressing team will generate more \"possession won in final third\" "
        "regardless of the individual defender's quality.",
        s["BodyText2"],
    ))

    el.append(Paragraph("4.3 Network Architecture", s["SubHeading"]))
    el.append(Paragraph(
        "Each of the four groups shares the same architecture, differing only in the "
        "number of output heads:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "Input(43) --> Dense(128, ReLU) --> Dropout(0.3) --> Dense(64, ReLU) --> Dropout(0.3) --> [Linear head x N]",
        s["Formula"],
    ))
    el.append(Paragraph(
        "The two Dense layers provide sufficient representational capacity for the "
        "43-dimensional input space without overfitting on typical training set sizes "
        "(hundreds to low thousands of historical transfers). The 0.3 dropout rate "
        "is a standard regularisation choice that prevents co-adaptation of neurons "
        "during training. Linear output heads are appropriate for regression targets: "
        "we are predicting continuous per-90 values, not class probabilities.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "The model is compiled with the Adam optimiser and mean squared error (MSE) "
        "loss, with mean absolute error (MAE) tracked as a secondary metric for "
        "interpretability. Training uses a 15% validation split to monitor for "
        "overfitting.",
        s["BodyText2"],
    ))

    el.append(Paragraph("4.4 Regression vs. Classification: A Deliberate Choice", s["SubHeading"]))
    el.append(Paragraph(
        "Competing platforms frequently reduce the prediction problem to classification: "
        "\"will this player improve or decline?\" This is a conscious choice for "
        "simplicity at the cost of actionable intelligence. A categorical prediction "
        "of \"decline\" tells a scout nothing about magnitude. A 2% decline in xG "
        "is operationally irrelevant; a 25% decline is a transfer-killing finding. "
        "Both receive the same label in a classification framework.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "TransferScope uses granular regression precisely because magnitude matters. "
        "The output is not \"better\" or \"worse\" but \"0.42 xG per 90 declining to "
        "0.31 xG per 90, a 26.2% reduction.\" This level of precision is what "
        "separates actionable intelligence from noise.",
        s["BodyText2"],
    ))
    el.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # 5. ADJUSTMENT MODELS
    # ══════════════════════════════════════════════════════════════════════
    el.append(Paragraph("5. Adjustment Models: Sklearn Linear Regression Priors", s["SectionHeading"]))
    el.append(HRFlowable(width="100%", thickness=1, color=DARK_NAVY, spaceAfter=10))

    el.append(Paragraph(
        "Before the neural network produces its final predictions, a system of "
        "linear adjustment models provides interpretable, prior-based estimates "
        "that serve both as standalone predictions for low-data scenarios and as "
        "calibration signals for the DNN.",
        s["BodyText2"],
    ))

    el.append(Paragraph("5.1 Team Adjustment (13 Models)", s["SubHeading"]))
    el.append(Paragraph(
        "For each of the 13 core metrics, a separate sklearn LinearRegression model "
        "predicts how a team's output changes when viewed through a different league "
        "context:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "target = naive_league_expectation + beta x team_relative_feature + error",
        s["Formula"],
    ))
    el.append(Paragraph(
        "The <i>naive_league_expectation</i> is the league-average per-90 for the "
        "target league, serving as an offset. The <i>team_relative_feature</i> "
        "captures how far above or below their own league average the team performs. "
        "A team that is +15% above league average in xG in La Liga will maintain some "
        "of that relative advantage in the Premier League, but the absolute value "
        "will shift to the Premier League's baseline.",
        s["BodyText2"],
    ))

    el.append(Paragraph("5.2 Team-Position Adjustment", s["SubHeading"]))
    el.append(Paragraph(
        "Team-position features (e.g., \"average xG for forwards at this club\") are "
        "scaled by the same percentage change as the team-level adjustment:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "If team xG drops 40%, then striker xG and centre-back xG both drop 40%.",
        s["Formula"],
    ))
    el.append(Paragraph(
        "This proportional scaling preserves the internal structure of a team's "
        "tactical system while adjusting for the new competitive context.",
        s["BodyText2"],
    ))

    el.append(Paragraph("5.3 Player Adjustment (13 Models x N Positions)", s["SubHeading"]))
    el.append(Paragraph(
        "Player-level adjustment models are position-specific, with six input features "
        "per metric:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "target = b<sub>0</sub> + b<sub>1</sub> x player_previous_per90 + "
        "b<sub>2</sub> x avg_position_feature_new_team + "
        "b<sub>3</sub> x diff_avg_position_old_vs_new + "
        "b<sub>4</sub> x delta_RA + "
        "b<sub>5</sub> x delta_RA<super>2</super> + "
        "b<sub>6</sub> x delta_RA<super>3</super> + error",
        s["Formula"],
    ))
    el.append(Paragraph(
        "Where delta_RA is the change in relative ability between source and target "
        "club. The polynomial terms (quadratic and cubic) capture non-linear effects: "
        "moving from the Eredivisie to the Premier League has a different impact "
        "profile than moving from the Bundesliga to the Premier League, even if the "
        "raw relative ability change is similar. The cubic term allows the model to "
        "capture asymmetric effects: stepping up to a much harder league may have a "
        "disproportionately larger negative impact than the positive impact of stepping "
        "down to an easier league.",
        s["BodyText2"],
    ))
    el.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # 6. BAYESIAN RELIABILITY
    # ══════════════════════════════════════════════════════════════════════
    el.append(Paragraph("6. Bayesian Reliability &amp; Uncertainty Quantification", s["SectionHeading"]))
    el.append(HRFlowable(width="100%", thickness=1, color=DARK_NAVY, spaceAfter=10))

    el.append(Paragraph("6.1 The Prior Blend Formula", s["SubHeading"]))
    el.append(Paragraph(
        "Players with limited minutes present a statistical reliability problem. "
        "A player with 200 minutes of data has a per-90 profile that is highly "
        "volatile and unrepresentative. TransferScope addresses this through a "
        "Bayesian updating framework that blends raw observations with positional "
        "priors:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "Adjusted Feature = (1 - w) x Prior + w x Raw Mean",
        s["Formula"],
    ))
    el.append(Paragraph(
        "w = min(1, minutes_played / 1000)",
        s["Formula"],
    ))
    el.append(Paragraph(
        "When <i>w = 0</i> (zero minutes), the prediction relies entirely on the "
        "positional prior: \"what does the average player in this position at this "
        "level produce?\" As minutes accumulate, the raw data gradually overrides "
        "the prior. At 1000 minutes (approximately 11 full matches), the prior is "
        "fully discarded and the prediction relies on observed data alone.",
        s["BodyText2"],
    ))

    el.append(Paragraph("6.2 Rolling Window Specification", s["SubHeading"]))
    rolling_data = [
        ["Player features", "1000 minutes", "~11 matches", "Individual form"],
        ["Team features", "3000 minutes", "~33 matches", "Team tactical identity"],
        ["Team-position features", "3000 minutes", "~33 matches", "Positional role within team"],
    ]
    el.append(make_table(
        ["Feature Type", "Window", "Approx. Matches", "Captures"],
        rolling_data,
        col_widths=[1.5 * inch, 1.2 * inch, 1.3 * inch, 2.1 * inch],
    ))
    el.append(Paragraph("Table 5: Rolling window specifications.", s["Caption"]))

    el.append(Paragraph("6.3 RAG Confidence Scoring", s["SubHeading"]))
    el.append(Paragraph(
        "Mathematical uncertainty is communicated to the end-user through an "
        "intuitive traffic-light system based on the blend weight <i>w</i>:",
        s["BodyText2"],
    ))

    rag_data = [
        ["RED", "w < 0.3", "< 300 mins", "Heavily prior-dependent. High uncertainty.\nTreat predictions with significant caution."],
        ["AMBER", "0.3 <= w <= 0.7", "300-700 mins", "Mixed data and prior. Moderate confidence.\nDirectional predictions are reliable."],
        ["GREEN", "w > 0.7", "> 700 mins", "Data-rich. High confidence.\nPredictions are statistically grounded."],
    ]
    el.append(make_table(
        ["Status", "Weight Range", "Minutes", "Interpretation"],
        rag_data,
        col_widths=[0.7 * inch, 1.1 * inch, 1.0 * inch, 3.3 * inch],
    ))
    el.append(Paragraph("Table 6: RAG confidence scoring system.", s["Caption"]))

    el.append(Paragraph(
        "This system ensures that predictions are never presented with false "
        "precision. A scout reviewing a player with 250 minutes of data sees a "
        "clear RED indicator, signalling that the prediction is largely based on "
        "positional priors rather than observed performance. This transparency is "
        "critical for responsible decision-making: the model communicates not just "
        "\"what\" it predicts but \"how confident\" it is.",
        s["BodyText2"],
    ))
    el.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # 7. CASE STUDY: ARSENAL
    # ══════════════════════════════════════════════════════════════════════
    el.append(Paragraph("7. Case Study: Arsenal Tactical Integration", s["SectionHeading"]))
    el.append(HRFlowable(width="100%", thickness=1, color=DARK_NAVY, spaceAfter=10))

    el.append(Paragraph("7.1 Arteta-Ball: A Unique Tactical Context", s["SubHeading"]))
    el.append(Paragraph(
        "Arsenal under Mikel Arteta deploy fluid positional structures that shift "
        "between 2-3-5 in possession and 3-2-5 in build-up. This creates positional "
        "demands that diverge significantly from league norms:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "<b>Inverted full-backs</b> tuck into midfield, reducing their crossing output "
        "but increasing their progressive passing. The <b>left interior midfielder</b> "
        "(the \"left-8\") carries significant ball-progression and chance-creation "
        "responsibilities, with higher expected assists than typical central "
        "midfielders. The <b>right winger</b> operates as the primary creative "
        "outlet, with league-leading touches in the opposition box and chance creation "
        "metrics.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "These tactical signatures are encoded in the team-position features. When "
        "TransferScope predicts a player's \"Arsenal profile,\" it is not simply "
        "applying a Premier League adjustment; it is modelling what Arsenal "
        "specifically demand from each position. A right winger at Arsenal is "
        "expected to produce fundamentally different outputs than a right winger "
        "at Newcastle or West Ham.",
        s["BodyText2"],
    ))

    el.append(Paragraph("7.2 Transfer Impact Prediction", s["SubHeading"]))
    el.append(Paragraph(
        "The Transfer Impact dashboard (corresponding to Figure 1 of Dinsdale &amp; "
        "Gallagher, 2022) provides a comprehensive view of how a player's per-90 "
        "metrics are predicted to change upon joining the target club. The dashboard "
        "comprises four integrated components:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "<b>(a) Metric Bars:</b> Horizontal bar charts showing current vs. predicted "
        "per-90 values for all 13 core metrics, with percentage changes highlighted "
        "in green (improvement) or red (decline). <b>(b) Power Ranking Chart:</b> "
        "Timeline visualisation showing both clubs' normalised 0-100 Power Rankings, "
        "with the transfer date marked. <b>(c) RAG Confidence Indicator:</b> "
        "Traffic-light display communicating prediction reliability. "
        "<b>(d) Swarm Plots:</b> Strip plots showing the target player (red diamond) "
        "against teammates (orange) and league distribution (grey) for each metric.",
        s["BodyText2"],
    ))

    el.append(Paragraph("7.3 Shortlist Generation: Identifying Market Inefficiencies", s["SubHeading"]))
    el.append(Paragraph(
        "The Shortlist Generator implements a weighted similarity scoring system. "
        "Scouts configure importance weights (0.0-1.0) for each metric based on "
        "what the specific role demands. The scoring formula:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "normalized_target = (predicted_value - mean) / std",
        s["Formula"],
    ))
    el.append(Paragraph(
        "weighted_score = normalized_target x user_weight",
        s["Formula"],
    ))
    el.append(Paragraph(
        "final_score = sum(weighted_scores) / sum(weights)",
        s["Formula"],
    ))
    el.append(Paragraph(
        "Available filters include: age, market value, minutes played, position, "
        "league, and club Power Ranking cap. The Power Ranking cap is particularly "
        "useful for identifying market inefficiencies: by capping at, say, 60, "
        "scouts can identify players at mid-table clubs in weaker leagues who are "
        "predicted to perform at a level consistent with top-tier recruitment targets.",
        s["BodyText2"],
    ))

    el.append(Paragraph("7.4 The South American Opportunity", s["SubHeading"]))
    el.append(Paragraph(
        "The WorldFootballElo integration is particularly valuable for South American "
        "markets, where the \"league tax\" is most acute. A forward in the Brasileirao "
        "registering 0.55 xG per 90 may be systematically overvalued by European "
        "scouts who do not account for the defensive quality differential. "
        "Conversely, a defender with 2.8 interceptions per 90 in the Argentine "
        "Primera may be undervalued because that metric is inflated by more open, "
        "transition-heavy tactical styles. TransferScope translates both accurately.",
        s["BodyText2"],
    ))
    el.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # 8. DEVELOPMENT METHODOLOGY
    # ══════════════════════════════════════════════════════════════════════
    el.append(Paragraph("8. Development Methodology: Agentic AI Orchestration", s["SectionHeading"]))
    el.append(HRFlowable(width="100%", thickness=1, color=DARK_NAVY, spaceAfter=10))

    el.append(Paragraph("8.1 Claude Code as Architectural Partner", s["SubHeading"]))
    el.append(Paragraph(
        "TransferScope was designed and implemented through <b>agentic AI development</b> "
        "using Claude Code, powered by Anthropic's Claude Opus 4.6. This is not "
        "code generation in the traditional sense (prompting an LLM for snippets). "
        "It is a fundamentally different paradigm: the AI operates as an autonomous "
        "engineering partner, maintaining architectural coherence across the entire "
        "codebase while the human provides strategic direction and domain expertise.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "The human role is <b>Architectural Sovereignty</b>: defining what problem "
        "to solve, which academic framework to implement, and what trade-offs to "
        "accept. The AI role is implementation velocity: translating architectural "
        "decisions into tested, production-quality code at a pace that would be "
        "impossible for a solo developer.",
        s["BodyText2"],
    ))

    el.append(Paragraph("8.2 The Five-Phase Build", s["SubHeading"]))
    phase_data = [
        ["Phase 1", "Scaffold", "Project structure, dependencies,\nStreamlit entry point"],
        ["Phase 2", "Data Layer", "Cache, FotMob/ClubElo/WorldElo\nclients, Elo Router, 43 unit tests"],
        ["Phase 3", "Feature Pipeline", "Power Rankings, rolling windows,\nadjustment models, league registry"],
        ["Phase 4", "ML Models", "TensorFlow multi-head DNN,\nshortlist scorer"],
        ["Phase 5", "Frontend", "Transfer Impact, Shortlist Generator,\nHot or Not, all Plotly components"],
    ]
    el.append(make_table(
        ["Phase", "Name", "Deliverables"],
        phase_data,
        col_widths=[0.8 * inch, 1.2 * inch, 4.1 * inch],
    ))
    el.append(Paragraph("Table 7: The five-phase development cycle.", s["Caption"]))

    el.append(Paragraph(
        "Each phase was validated before proceeding to the next. Phase 2's 43 unit "
        "tests (all passing) provided a safety net for subsequent development. This "
        "is the same disciplined engineering methodology used for enterprise software "
        "at Microsoft, applied to the niche domain of football recruitment.",
        s["BodyText2"],
    ))

    el.append(Paragraph("8.3 Velocity as a Feature", s["SubHeading"]))
    el.append(Paragraph(
        "The research-to-production cycle for TransferScope, from reading the "
        "Dinsdale &amp; Gallagher paper to a fully functional Streamlit application "
        "with tested backend, was compressed from weeks to hours. This velocity is "
        "not merely a convenience; it is a strategic capability. Football recruitment "
        "operates on tight timelines. The ability to prototype, validate, and deploy "
        "an analytical tool in hours rather than weeks means that insights can inform "
        "decisions within the same transfer window that motivates the analysis.",
        s["BodyText2"],
    ))
    el.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # 9. SHORTLIST SCORING
    # ══════════════════════════════════════════════════════════════════════
    el.append(Paragraph("9. Shortlist Scoring Methodology", s["SectionHeading"]))
    el.append(HRFlowable(width="100%", thickness=1, color=DARK_NAVY, spaceAfter=10))

    el.append(Paragraph(
        "The shortlist scoring system converts multi-dimensional predicted performance "
        "profiles into a single comparable score, while preserving the scout's ability "
        "to define what \"good\" looks like for a specific role.",
        s["BodyText2"],
    ))

    el.append(Paragraph("9.1 Scoring Formula", s["SubHeading"]))
    el.append(Paragraph(
        "For each candidate and each metric, the predicted per-90 value is "
        "z-normalised against the candidate pool:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "z<sub>i,m</sub> = (predicted<sub>i,m</sub> - mean<sub>m</sub>) / std<sub>m</sub>",
        s["Formula"],
    ))
    el.append(Paragraph(
        "The normalised score is then multiplied by the user-defined weight for "
        "that metric, and the final score is the weighted average:",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "score<sub>i</sub> = SUM(z<sub>i,m</sub> x w<sub>m</sub>) / SUM(w<sub>m</sub>)",
        s["Formula"],
    ))

    el.append(Paragraph("9.2 The Hot or Not Validator", s["SubHeading"]))
    el.append(Paragraph(
        "For rapid rumour assessment, TransferScope provides a \"Hot or Not\" tool "
        "that computes the average percentage change across all 13 core metrics and "
        "renders a verdict:",
        s["BodyText2"],
    ))
    hon_data = [
        ["HOT", "> +5% average improvement", "Transfer is predicted to improve\nthe player's overall output"],
        ["TEPID", "-5% to +5% average", "Marginal or mixed impact;\nfurther analysis recommended"],
        ["NOT", "< -5% average decline", "Transfer is predicted to reduce\nthe player's overall output"],
    ]
    el.append(make_table(
        ["Verdict", "Threshold", "Interpretation"],
        hon_data,
        col_widths=[0.8 * inch, 2.0 * inch, 3.3 * inch],
    ))
    el.append(Paragraph("Table 8: Hot or Not verdict thresholds.", s["Caption"]))
    el.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # 10. CONCLUSION
    # ══════════════════════════════════════════════════════════════════════
    el.append(Paragraph("10. Conclusion &amp; Future Roadmap", s["SectionHeading"]))
    el.append(HRFlowable(width="100%", thickness=1, color=DARK_NAVY, spaceAfter=10))

    el.append(Paragraph("10.1 Architectural Auditability", s["SubHeading"]))
    el.append(Paragraph(
        "TransferScope is currently built on freely available data sources: FotMob "
        "(via the mobfot Python package), ClubElo (via soccerdata), and "
        "WorldFootballElo (via HTTP scrape). This is a deliberate architectural "
        "choice, not a limitation. The system is designed with a cache layer and "
        "metric mapping system that supports plug-and-play data source upgrades. "
        "Replacing FotMob with proprietary Opta or StatsBomb feeds requires modifying "
        "a single client file and updating the metric key mapping, with zero changes "
        "to the feature pipeline, adjustment models, or neural network.",
        s["BodyText2"],
    ))

    el.append(Paragraph("10.2 Future Roadmap", s["SubHeading"]))
    el.append(Paragraph(
        "<b>Event-level spatial data:</b> Integration of x,y coordinate event data "
        "to capture pitch positioning patterns, passing networks, and defensive "
        "shape, moving beyond aggregate per-90 statistics to spatial intelligence.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "<b>Injury risk modelling:</b> Incorporating workload metrics, injury history, "
        "and physical data to provide a holistic transfer assessment that accounts "
        "for durability, not just performance.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "<b>Contract value estimation:</b> Combining predicted performance with "
        "market value modelling to identify players whose transfer fee is mispriced "
        "relative to their predicted output at the target club.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "<b>Multi-season prediction windows:</b> Extending the prediction horizon "
        "from first-1000-minutes to full-season and multi-season projections, "
        "accounting for adaptation curves and age-related development trajectories.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "<b>Video analysis integration:</b> Linking predicted statistical profiles "
        "to video footage, enabling scouts to validate quantitative predictions "
        "against qualitative visual evidence.",
        s["BodyText2"],
    ))

    el.append(Paragraph("10.3 Final Word", s["SubHeading"]))
    el.append(Paragraph(
        "TransferScope is not a black box. It is a transparent, research-backed, "
        "mathematically rigorous bridge to more intelligent recruitment. Every "
        "prediction is traceable through the feature pipeline. Every confidence "
        "score is grounded in data volume. Every architectural decision is justified "
        "by the academic literature. The platform does not replace scouting judgment; "
        "it arms it with the quantitative foundation that modern football demands.",
        s["BodyText2"],
    ))
    el.append(Paragraph(
        "The League Tax is real, measurable, and exploitable. TransferScope is the "
        "tool that makes that exploitation systematic.",
        s["BodyText2"],
    ))
    el.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════
    # REFERENCES
    # ══════════════════════════════════════════════════════════════════════
    el.append(Paragraph("References", s["SectionHeading"]))
    el.append(HRFlowable(width="100%", thickness=1, color=DARK_NAVY, spaceAfter=10))

    refs = [
        "Dinsdale, S. &amp; Gallagher, C. (2022). \"Transfer Portal: Predicting Player "
        "Performance at a New Club.\" <i>Proceedings of the 6th Workshop on Machine "
        "Learning and Data Mining for Sports Analytics.</i>",

        "Pappalardo, L., Cintia, P., Rossi, A., Massucco, E., Ferragina, P., "
        "Pedreschi, D. &amp; Giannotti, F. (2019). \"A Public Data Set of Spatio-Temporal "
        "Match Events in Soccer Competitions.\" <i>Scientific Data, 6</i>(236).",

        "Hvattum, L. M. &amp; Arntzen, H. (2010). \"Using ELO Ratings for Match Result "
        "Prediction in Association Football.\" <i>International Journal of Forecasting, "
        "26</i>(3), 460-470.",

        "Decroos, T., Bransen, L., Van Haaren, J., &amp; Davis, J. (2019). \"Actions Speak "
        "Louder Than Goals: Valuing Player Actions in Soccer.\" <i>Proceedings of the "
        "25th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data "
        "Mining.</i>",

        "Fernandez, J., Bornn, L., &amp; Cervone, D. (2021). \"A Framework for the "
        "Fine-Grained Evaluation of the Instantaneous Expected Value of Soccer "
        "Possessions.\" <i>Machine Learning, 110</i>, 1389-1427.",

        "ClubElo. (2025). Club Elo Football Ratings. http://clubelo.com/",

        "World Football Elo Ratings. (2025). http://eloratings.net/",

        "FotMob. (2025). Football Scores, Stats, and News. https://www.fotmob.com/",
    ]

    for ref in refs:
        el.append(Paragraph(ref, ParagraphStyle(
            "RefItem", parent=s["BodyText2"], leftIndent=30, firstLineIndent=-30,
            spaceAfter=8, fontSize=9.5, leading=13,
        )))

    return el


def main():
    s = build_styles()

    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="TransferScope: A Predictive Framework for Cross-League Performance Translation",
        author="Saaleh Dinsdale",
        subject="Football Analytics White Paper",
    )

    # Build story
    story = []
    story.extend(build_cover(s))
    story.extend(build_body(s))

    # Build with page templates
    frame = Frame(MARGIN, MARGIN, WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN,
                  id="normal")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=frame, onPage=first_page),
        PageTemplate(id="content", frames=frame, onPage=header_footer),
    ])

    # Force content template after cover
    from reportlab.platypus import NextPageTemplate
    story.insert(len(build_cover(s)), NextPageTemplate("content"))

    doc.build(story)
    print(f"White paper saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
