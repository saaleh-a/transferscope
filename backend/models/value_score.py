"""Value Opportunity Score — who is underpriced relative to what they produce.

TransferScope answers "how will this player perform at a new club".  It has
never answered "is that performance worth the fee", which is the question every
sporting director actually asks second.  This module adds the price dimension.

The design is adapted from the composite-score pattern used by public scouting
dashboards, but every component here is backed by real data rather than a
synthetic proxy:

===========================  ==========================================
Component                    Real signal used
===========================  ==========================================
``output_per_value``         per-90 production ÷ Sofascore market value
``contract_leverage``        contract years remaining (Sofascore)
``age_runway``               years to peak, from ``PEAK_AGE``
``projected_improvement``    TransferScope's own post-transfer prediction
===========================  ==========================================

The last component is the part no comparable public tool has: a model
validated against actual post-transfer outcomes rather than a label that was
constructed and then recovered.

Two rules govern the whole module:

1. **Missing data is never imputed as zero.**  A player with no market value
   gets ``None``, not a flattering score.  Sofascore market-value coverage is
   roughly 90% at major clubs and worse in minor leagues.
2. **Every score is explainable.**  ``components`` and ``reasons`` are returned
   alongside the number so a scout can see why a player surfaced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

_log = logging.getLogger(__name__)

# ── Weights ──────────────────────────────────────────────────────────────────
# Deliberately transparent and tunable.  They sum to 1.0 so the composite is
# directly interpretable as a 0-100 score.
VALUE_WEIGHTS: Dict[str, float] = {
    "output_per_value": 0.40,
    "projected_improvement": 0.25,
    "contract_leverage": 0.20,
    "age_runway": 0.15,
}

# Outfield players peak around 26-27 for most per-90 output metrics.
PEAK_AGE = 26.5
# Beyond this, remaining contract length stops meaningfully affecting fee.
MAX_USEFUL_CONTRACT_YEARS = 5.0

_REASON_TEXT: Dict[str, str] = {
    "output_per_value": "high output for the price",
    "projected_improvement": "projected to improve at a stronger club",
    "contract_leverage": "contract running down",
    "age_runway": "young enough to appreciate",
}

# Position labels vary across Sofascore endpoints ("F", "Forward", "Attacker"),
# so normalise to four coarse groups for cohort comparison.
_POSITION_GROUPS: Dict[str, str] = {
    "g": "Goalkeeper",
    "d": "Defender",
    "m": "Midfielder",
    "f": "Forward",
    "a": "Forward",
}


def _position_group(position: Optional[str]) -> str:
    """Map a Sofascore position label to a coarse group, or 'Unknown'."""
    if not position:
        return "Unknown"
    return _POSITION_GROUPS.get(str(position).strip()[:1].lower(), "Unknown")


@dataclass
class ValueCandidate:
    """One player considered for a value ranking."""

    player_id: int
    name: str
    market_value: Optional[float] = None       # EUR
    contract_years_left: Optional[float] = None
    age: Optional[float] = None
    # Weighted per-90 output — caller decides which metrics matter (see
    # ``composite_output``), so this stays position-agnostic.
    output: Optional[float] = None
    # Predicted % change in key metrics after a move, from TransferPortalModel.
    projected_improvement_pct: Optional[float] = None
    team: str = ""
    position: str = ""


@dataclass
class ValueScore:
    """A scored player, with the reasoning kept attached to the number."""

    player_id: int
    name: str
    score: Optional[float]                     # 0-100, or None if unscoreable
    components: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    coverage: float = 0.0                      # fraction of weight satisfied


def percentile_ranks(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Rank values into [0, 1] by percentile, preserving None.

    Percentile rank is used instead of min-max because market values are
    heavily right-skewed: a single €200M player would compress everyone else
    into the bottom of a min-max scale.  Ties share the average rank.
    """
    indexed = [(v, i) for i, v in enumerate(values) if v is not None]
    out: List[Optional[float]] = [None] * len(values)
    if not indexed:
        return out
    if len(indexed) == 1:
        out[indexed[0][1]] = 0.5  # a lone value is neither best nor worst
        return out

    indexed.sort(key=lambda pair: pair[0])
    n = len(indexed)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][0] == indexed[i][0]:
            j += 1
        # Average rank across the tied block, scaled to [0, 1]
        avg_rank = (i + j) / 2.0
        pct = avg_rank / (n - 1)
        for k in range(i, j + 1):
            out[indexed[k][1]] = pct
        i = j + 1
    return out


def composite_output(
    per90: Dict[str, float],
    weights: Dict[str, float],
) -> Optional[float]:
    """Collapse per-90 metrics into one weighted output number.

    Returns None when no weighted metric is present, so that a player with no
    stats is excluded rather than scored as zero.
    """
    total = 0.0
    weight_used = 0.0
    for metric, weight in weights.items():
        value = per90.get(metric)
        if value is None:
            continue
        total += float(value) * weight
        weight_used += weight
    if weight_used <= 0:
        return None
    return total / weight_used


def _contract_leverage(years_left: Optional[float]) -> Optional[float]:
    """Shorter contract → cheaper to sign → higher leverage (0-1).

    A player inside their final year is the classic underpriced asset; beyond
    ``MAX_USEFUL_CONTRACT_YEARS`` the fee stops responding to contract length.
    """
    if years_left is None:
        return None
    clamped = max(0.0, min(float(years_left), MAX_USEFUL_CONTRACT_YEARS))
    return 1.0 - (clamped / MAX_USEFUL_CONTRACT_YEARS)


def _age_runway(age: Optional[float]) -> Optional[float]:
    """Years of appreciation left before peak, normalised to 0-1.

    Peaks at 1.0 for the youngest players and decays to 0 at and beyond
    ``PEAK_AGE``.  Players past peak score 0 rather than negative — this is an
    upside signal, not a penalty.
    """
    if age is None:
        return None
    try:
        age_f = float(age)
    except (TypeError, ValueError):
        return None
    if age_f <= 0:
        return None
    runway = PEAK_AGE - age_f
    if runway <= 0:
        return 0.0
    # 16-year-olds have ~10 years of runway; normalise against that.
    return min(1.0, runway / (PEAK_AGE - 16.0))


def score_candidates(
    candidates: Sequence[ValueCandidate],
    weights: Optional[Dict[str, float]] = None,
    min_coverage: float = 0.5,
    by_position: bool = False,
) -> List[ValueScore]:
    """Score a cohort of players, ranked highest-opportunity first.

    Scores are **relative to the supplied cohort** — a player is underpriced
    compared to these peers, not in absolute terms. Comparing across leagues
    therefore requires passing those leagues in together.

    ``min_coverage`` is the fraction of total weight that must be backed by
    real data before a score is produced; below it, the score is ``None`` and
    the missing inputs are listed.  This prevents a player with only an age
    from ranking above a fully-documented one.

    ``by_position`` scores players **within their own position group** and is
    strongly recommended for mixed squads.  Output-per-euro is not comparable
    across positions: a cheap goalkeeper trivially out-ranks an expensive
    forward on attacking metrics, because the denominator collapses faster
    than the numerator.  Grouping compares like with like.
    """
    weights = weights or VALUE_WEIGHTS
    if not candidates:
        return []

    if by_position:
        groups: Dict[str, List[ValueCandidate]] = {}
        for cand in candidates:
            groups.setdefault(_position_group(cand.position), []).append(cand)
        combined: List[ValueScore] = []
        for group in groups.values():
            combined.extend(
                score_candidates(
                    group,
                    weights=weights,
                    min_coverage=min_coverage,
                    by_position=False,
                )
            )
        combined.sort(
            key=lambda r: (r.score is not None, r.score or 0.0), reverse=True
        )
        return combined

    # output_per_value: production per € — the core efficiency signal.
    output_per_value: List[Optional[float]] = []
    for c in candidates:
        if c.output is None or not c.market_value or c.market_value <= 0:
            output_per_value.append(None)
        else:
            # Per €1m, to keep the raw numbers legible in debugging.
            output_per_value.append(c.output / (c.market_value / 1_000_000.0))

    raw_components: Dict[str, List[Optional[float]]] = {
        "output_per_value": percentile_ranks(output_per_value),
        "projected_improvement": percentile_ranks(
            [c.projected_improvement_pct for c in candidates]
        ),
        "contract_leverage": [
            _contract_leverage(c.contract_years_left) for c in candidates
        ],
        "age_runway": [_age_runway(c.age) for c in candidates],
    }

    results: List[ValueScore] = []
    for idx, cand in enumerate(candidates):
        components: Dict[str, float] = {}
        missing: List[str] = []
        weighted_sum = 0.0
        weight_available = 0.0

        for name, weight in weights.items():
            value = raw_components.get(name, [None] * len(candidates))[idx]
            if value is None:
                missing.append(name)
                continue
            components[name] = round(float(value), 4)
            weighted_sum += float(value) * weight
            weight_available += weight

        total_weight = sum(weights.values()) or 1.0
        coverage = weight_available / total_weight

        if coverage < min_coverage or weight_available <= 0:
            results.append(
                ValueScore(
                    player_id=cand.player_id,
                    name=cand.name,
                    score=None,
                    components=components,
                    reasons=[],
                    missing=missing,
                    coverage=round(coverage, 3),
                )
            )
            continue

        # Renormalise by available weight so partial data is not penalised
        # twice (once by exclusion, once by a diluted score).
        score = 100.0 * weighted_sum / weight_available

        # Explain the score with its two strongest weighted contributors.
        contributions = sorted(
            ((components[n] * weights[n], n) for n in components),
            reverse=True,
        )
        reasons = [
            _REASON_TEXT[name]
            for contribution, name in contributions[:2]
            if contribution > 0 and name in _REASON_TEXT
        ]

        results.append(
            ValueScore(
                player_id=cand.player_id,
                name=cand.name,
                score=round(score, 1),
                components=components,
                reasons=reasons,
                missing=missing,
                coverage=round(coverage, 3),
            )
        )

    results.sort(key=lambda r: (r.score is not None, r.score or 0.0), reverse=True)
    return results
