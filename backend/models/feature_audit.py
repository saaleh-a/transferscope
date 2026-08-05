"""Detect features that carry no signal.

A constant column trains without complaint and contributes nothing. Four were
found by hand in the shipped matrix:

    player_xg_on_target          Sofascore does not serve xGOT
    player_non_penalty_xg        Sofascore does not serve npxG
    player_xg_against_on_pitch   Sofascore does not serve xGA
    pre_minutes_per_match        added after the matrices were built

The first three are genuinely unavailable: the season statistics endpoint
returns 115 keys and none of them is an xG variant beyond expectedGoals and
expectedAssists. The fourth was a migration artefact.

This module turns that manual sweep into something the test suite runs, so the
next dead feature is found immediately rather than after it has trained into a
shipped model.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

_log = logging.getLogger(__name__)

# Features known to be permanently unavailable from Sofascore. Listed so the
# audit can separate "known gap" from "new regression" — a newly dead feature
# is a bug, these four are documented limitations.
KNOWN_DEAD_FEATURES = frozenset({
    "player_xg_on_target",
    "player_non_penalty_xg",
    "player_xg_against_on_pitch",
})

# Was {"pre_minutes_per_match"}. That whitelisted the one column the audit was
# built to catch: the feature was constant-zero in the shipped matrix because
# no inference caller ever supplied it, and unioning it into `known` meant the
# dead-feature guard stayed quiet about exactly the case it exists for.
# Inference now computes it via `minutes_per_match_from_stats`, so a constant
# column here is a real regression again and should fail.
KNOWN_MIGRATION_GAPS: frozenset = frozenset()

# Above this share of zeros a feature is too sparse to inform much.
SPARSE_THRESHOLD = 0.95


@dataclass
class FeatureAudit:
    """Result of auditing one feature column."""

    name: str
    index: int
    std: float
    zero_fraction: float
    is_constant: bool
    is_sparse: bool
    has_non_finite: bool
    constant_value: Optional[float] = None

    @property
    def is_healthy(self) -> bool:
        return not (self.is_constant or self.has_non_finite)


def audit_features(
    X: np.ndarray,
    feature_names: List[str],
) -> List[FeatureAudit]:
    """Audit every column of a feature matrix."""
    audits: List[FeatureAudit] = []
    for i, name in enumerate(feature_names):
        if i >= X.shape[1]:
            break
        col = X[:, i]
        finite = np.isfinite(col)
        std = float(np.std(col[finite])) if finite.any() else 0.0
        zero_fraction = float((col == 0).mean()) if len(col) else 1.0
        is_constant = std == 0.0
        audits.append(
            FeatureAudit(
                name=name,
                index=i,
                std=std,
                zero_fraction=zero_fraction,
                is_constant=is_constant,
                is_sparse=(not is_constant) and zero_fraction > SPARSE_THRESHOLD,
                has_non_finite=not bool(finite.all()),
                constant_value=float(col[0]) if is_constant and len(col) else None,
            )
        )
    return audits


def unexpected_dead_features(audits: List[FeatureAudit]) -> List[str]:
    """Constant features that are not already documented as unavailable.

    This is the regression signal: a feature going constant that we have not
    accounted for means the source stopped supplying it, a key mapping broke,
    or a migration silently zero-filled a column.
    """
    known = KNOWN_DEAD_FEATURES | KNOWN_MIGRATION_GAPS
    return [a.name for a in audits if a.is_constant and a.name not in known]


def audit_saved_matrices(
    matrices_dir: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    """Audit the matrices on disk. Returns None when they are absent."""
    from backend.models.transfer_portal import _feature_keys

    if matrices_dir is None:
        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        matrices_dir = os.path.join(root, "data", "models", "matrices")

    x_path = os.path.join(matrices_dir, "X.npy")
    if not os.path.exists(x_path):
        return None

    X = np.load(x_path)
    audits = audit_features(X, _feature_keys())
    constant = [a for a in audits if a.is_constant]
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "constant": [a.name for a in constant],
        "sparse": [a.name for a in audits if a.is_sparse],
        "non_finite": [a.name for a in audits if a.has_non_finite],
        "unexpected_dead": unexpected_dead_features(audits),
        "healthy_fraction": (
            sum(1 for a in audits if a.is_healthy) / len(audits) if audits else 0.0
        ),
    }


def format_report(report: Dict[str, object]) -> str:
    """Render an audit report as readable lines."""
    lines = [
        f"features: {report['n_features']} over {report['n_samples']:,} samples",
        f"healthy: {report['healthy_fraction']:.1%}",
    ]
    constant = report.get("constant") or []
    if constant:
        lines.append(f"constant (no signal): {', '.join(constant)}")
    unexpected = report.get("unexpected_dead") or []
    if unexpected:
        lines.append(f"UNEXPECTED dead features: {', '.join(unexpected)}")
    else:
        lines.append("no unexpected dead features")
    sparse = report.get("sparse") or []
    if sparse:
        lines.append(f"sparse (>95% zeros): {', '.join(sparse)}")
    non_finite = report.get("non_finite") or []
    if non_finite:
        lines.append(f"NON-FINITE: {', '.join(non_finite)}")
    return "\n".join(lines)
