"""Empirical age curves from TransferScope's own transfer records — and a
stability check that says when they cannot be trusted.

Public scouting dashboards typically bucket players into trajectory labels
("Rising Star", "Declining Asset") using hand-picked age thresholds.  This
module set out to derive those curves from data instead: roughly 11,500 real
transfers already sitting in ``data/models/matrices/``.

What the data actually says
---------------------------
It mostly does not support age curves, and :func:`curve_is_stable` is what
establishes that rather than assuming it either way.  Splitting the records
into folds and rebuilding each curve shows the peak age wandering by 8-16
years for most metrics.  Forwards' xG looks stable across recency windows but
still fails the fold test, and the underlying curve is close to flat (median
0.26-0.34 xG/90 from age 16 to 34).

That flatness is expected on reflection.  These are *different* players
compared at different ages, not the same player tracked over time, and the
sample is doubly selected: a player only appears if a club bought them, and
only keeps appearing at 32 if they are still good enough to be bought.
Survivorship flattens exactly the decline a career curve is supposed to show.

So this module is deliberately conservative.  It reports curves, marks each
one ``stable`` or not, and exposes :attr:`AgeCurve.trustworthy` so a caller
can refuse to label a player from noise.  The curves are **not** wired into
the shortlist or verdict UI, because a confident-looking peak age drawn from
a flat, survivorship-biased sample is worse than no label at all.

The age bug this surfaced
-------------------------
Building the curves exposed a real defect in the feature matrix.  The
``player_age`` column was the player's age **at build time**, not at the
transfer: the training pipeline computed ``date.today() - date_of_birth``.
Because the records span roughly a decade, this inflated age in proportion to
how long ago the transfer happened — mean stored age 29.1 against a true mean
of 22.4, overstated by up to 24.9 years on the oldest rows.  The model was
being told that a move made by a 20-year-old belonged to a 30-year-old.

The pipeline now measures age at the transfer date and stores it in metadata.
:func:`audit_age_bias` quantifies the distortion in any existing matrices, so
the decision to rebuild is made on measured numbers.  Matrices built before
the fix still carry the old values; :func:`age_at_transfer` corrects them
on read.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_log = logging.getLogger(__name__)

# Column index of ``player_age`` in the feature matrix.  Resolved from the
# canonical key list so it cannot silently drift when features are added.
_AGE_FEATURE_KEY = "player_age"

# Records outside this range are treated as data errors rather than outliers:
# Sofascore transfer histories contain a handful of impossible rows (transfers
# dated to the 1970s attached to players born in the 2000s).
MIN_PLAUSIBLE_AGE = 15.0
MAX_PLAUSIBLE_AGE = 42.0
MIN_PLAUSIBLE_TRANSFER_YEAR = 2000

# A bucket needs enough players before its mean means anything.
DEFAULT_MIN_SAMPLES = 30


@dataclass
class AgeBucket:
    """Aggregated output for one age band."""

    age_from: float
    age_to: float
    n: int
    mean: float
    median: float
    p25: float
    p75: float

    @property
    def midpoint(self) -> float:
        return (self.age_from + self.age_to) / 2.0


@dataclass
class AgeCurve:
    """An empirical age curve for a single metric."""

    metric: str
    buckets: List[AgeBucket] = field(default_factory=list)
    peak_age: Optional[float] = None
    n_total: int = 0
    position: Optional[str] = None
    stable: Optional[bool] = None
    peak_spread_years: Optional[float] = None

    @property
    def has_interior_peak(self) -> bool:
        """True when the maximum falls strictly inside the observed range.

        A "peak" sitting on the first or last bucket means the curve is
        monotonic over the ages we can see — there is no arc, just a slope
        running off the edge of the data.  Reporting that as a peak age is how
        an implausible claim like "xG peaks at 15" gets produced: consistent
        across folds, and still meaningless.
        """
        if len(self.buckets) < 3 or self.peak_age is None:
            return False
        return self.buckets[0].midpoint < self.peak_age < self.buckets[-1].midpoint

    @property
    def trustworthy(self) -> bool:
        """True when the curve has a peak that is both stable and interior.

        Callers should refuse to label a player's trajectory from an
        untrustworthy curve rather than presenting a confident-looking peak
        age that is an artefact of the sample.
        """
        if not self.buckets:
            return False
        if self.stable is False:
            return False
        return self.has_interior_peak

    def value_at(self, age: float) -> Optional[float]:
        """Linear interpolation of the curve at ``age``.

        Returns None when the curve is empty or the age falls outside the
        observed range — extrapolating a career trajectory past the data is
        exactly how spurious "declining asset" labels get produced.
        """
        if not self.buckets:
            return None
        xs = [b.midpoint for b in self.buckets]
        ys = [b.median for b in self.buckets]
        if age < xs[0] or age > xs[-1]:
            return None
        return float(np.interp(age, xs, ys))

    def remaining_upside(self, age: float) -> Optional[float]:
        """Fraction of peak output still ahead of a player at ``age``.

        1.0 means the player is at the start of the observed curve, 0.0 means
        at or past peak.  Returns None when the age is outside the data.
        """
        current = self.value_at(age)
        if current is None or self.peak_age is None:
            return None
        peak_value = self.value_at(self.peak_age)
        if peak_value is None or peak_value <= 0:
            return None
        if age >= self.peak_age:
            return 0.0
        return max(0.0, min(1.0, (peak_value - current) / peak_value))


def age_at_transfer(
    stored_age: float,
    transfer_date: Optional[str],
    reference: Optional[dt.datetime] = None,
) -> Optional[float]:
    """Convert a scrape-time age into the player's age at the transfer.

    ``stored_age`` is what the feature matrix holds: age computed against the
    time the data was fetched.  Subtracting the years elapsed since
    ``transfer_date`` recovers the age the player actually was.

    Returns None when the date is missing, unparseable, implausibly old, or
    when the corrected age falls outside a plausible playing range.
    """
    if not transfer_date or stored_age is None or stored_age <= 0:
        return None
    try:
        parsed = dt.datetime.strptime(str(transfer_date)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    if parsed.year < MIN_PLAUSIBLE_TRANSFER_YEAR:
        return None

    reference = reference or dt.datetime.now()
    years_elapsed = (reference - parsed).days / 365.25
    corrected = float(stored_age) - years_elapsed
    if not (MIN_PLAUSIBLE_AGE <= corrected <= MAX_PLAUSIBLE_AGE):
        return None
    return corrected


def _age_feature_index() -> int:
    """Resolve the age column index from the canonical feature key list."""
    from backend.models.transfer_portal import _feature_keys

    keys = _feature_keys()
    try:
        return keys.index(_AGE_FEATURE_KEY)
    except ValueError as exc:  # pragma: no cover - guards a rename
        raise ValueError(
            f"'{_AGE_FEATURE_KEY}' is not in the feature vector; "
            "age curves cannot be built."
        ) from exc


def load_age_samples(
    matrices_dir: Optional[str] = None,
    reference: Optional[dt.datetime] = None,
) -> List[Tuple[float, Dict[str, float], str]]:
    """Load ``(corrected_age, per90, position)`` triples from saved matrices.

    Records without a usable age or transfer date are skipped rather than
    guessed at.  Returns an empty list when the matrices are absent, so
    callers degrade instead of failing.
    """
    if matrices_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        matrices_dir = os.path.join(root, "data", "models", "matrices")

    x_path = os.path.join(matrices_dir, "X.npy")
    meta_path = os.path.join(matrices_dir, "metadata.json")
    if not (os.path.exists(x_path) and os.path.exists(meta_path)):
        _log.info("Age curves: no feature matrices at %s", matrices_dir)
        return []

    X = np.load(x_path)
    with open(meta_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    age_idx = _age_feature_index()
    if X.shape[1] <= age_idx:
        _log.warning(
            "Age curves: matrix has %d columns, expected age at index %d",
            X.shape[1], age_idx,
        )
        return []

    samples: List[Tuple[float, Dict[str, float], str]] = []
    for i, record in enumerate(metadata):
        if i >= len(X):
            break
        # Matrices rebuilt after the age fix carry the corrected age directly;
        # older ones need it reconstructed from the transfer date.
        stored = record.get("player_age")
        if stored:
            corrected = float(stored)
            if not (MIN_PLAUSIBLE_AGE <= corrected <= MAX_PLAUSIBLE_AGE):
                continue
        else:
            corrected = age_at_transfer(
                float(X[i, age_idx]), record.get("transfer_date"), reference,
            )
            if corrected is None:
                continue
        per90 = record.get("pre_per90") or {}
        if not isinstance(per90, dict) or not per90:
            continue
        samples.append((corrected, per90, record.get("position") or ""))
    return samples


def build_age_curve(
    samples: Sequence[Tuple[float, Dict[str, float], str]],
    metric: str,
    bucket_years: float = 2.0,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    position: Optional[str] = None,
) -> AgeCurve:
    """Aggregate per-90 output for one metric into age buckets.

    Medians are used for the curve because per-90 distributions are
    right-skewed — a handful of elite seasons would drag a mean upward and
    shift the apparent peak.

    Buckets with fewer than ``min_samples`` players are dropped rather than
    reported with a wide error, which is what makes a peak age trustworthy.

    ``position`` should almost always be supplied.  Pooling every position
    into one curve mixes goalkeepers, defenders and forwards, and the result
    is not a football signal: measured on this dataset the pooled xG peak
    moves between ages 15 and 21 depending only on how many seasons of
    transfers are included, whereas the forwards-only peak holds at 29 across
    every window.  :func:`curve_is_stable` checks this directly, and
    :func:`build_all_curves` marks pooled curves as unstable.
    """
    from backend.data.sofascore_client import normalize_position

    by_bucket: Dict[float, List[float]] = {}
    for age, per90, pos in samples:
        if position is not None and normalize_position(pos) != position:
            continue
        value = per90.get(metric)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        bucket_start = float(np.floor(age / bucket_years) * bucket_years)
        by_bucket.setdefault(bucket_start, []).append(value)

    buckets: List[AgeBucket] = []
    for start in sorted(by_bucket):
        values = np.array(by_bucket[start], dtype=float)
        if len(values) < min_samples:
            continue
        buckets.append(
            AgeBucket(
                age_from=start,
                age_to=start + bucket_years,
                n=int(len(values)),
                mean=float(values.mean()),
                median=float(np.median(values)),
                p25=float(np.percentile(values, 25)),
                p75=float(np.percentile(values, 75)),
            )
        )

    peak_age = None
    if buckets:
        peak = max(buckets, key=lambda b: b.median)
        peak_age = peak.midpoint

    return AgeCurve(
        metric=metric,
        buckets=buckets,
        peak_age=peak_age,
        n_total=sum(b.n for b in buckets),
        position=position,
    )


# A peak that moves by more than this across data subsets is not a peak.
PEAK_STABILITY_TOLERANCE_YEARS = 3.0


def curve_is_stable(
    samples: Sequence[Tuple[float, Dict[str, float], str]],
    metric: str,
    position: Optional[str] = None,
    n_folds: int = 3,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    tolerance: float = PEAK_STABILITY_TOLERANCE_YEARS,
) -> Tuple[bool, Optional[float]]:
    """Check whether a curve's peak age survives resampling.

    Splits the samples into ``n_folds`` interleaved subsets, rebuilds the
    curve on each, and returns ``(is_stable, peak_spread_years)``.  A curve
    whose peak wanders more than ``tolerance`` years between folds is
    reporting sampling noise, not a career trajectory, and callers should
    refuse to label players with it.
    """
    if not samples:
        return False, None

    peaks: List[float] = []
    for fold in range(n_folds):
        subset = samples[fold::n_folds]
        curve = build_age_curve(
            subset, metric,
            min_samples=max(5, min_samples // n_folds),
            position=position,
        )
        if curve.peak_age is not None:
            peaks.append(curve.peak_age)

    if len(peaks) < 2:
        return False, None
    spread = float(max(peaks) - min(peaks))
    return spread <= tolerance, spread


def build_all_curves(
    samples: Optional[Sequence[Tuple[float, Dict[str, float], str]]] = None,
    metrics: Optional[Sequence[str]] = None,
    position: Optional[str] = None,
    bucket_years: float = 2.0,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    check_stability: bool = True,
) -> Dict[str, AgeCurve]:
    """Build an age curve for each core metric, flagging unstable ones.

    Every returned curve carries ``stable`` and ``peak_spread_years`` so a
    caller can tell a real career trajectory from sampling noise.  Pass
    ``position`` — pooled curves are usually unstable, and the flag will say so.
    """
    from backend.data.sofascore_client import CORE_METRICS

    if samples is None:
        samples = load_age_samples()
    metrics = metrics or CORE_METRICS

    curves: Dict[str, AgeCurve] = {}
    for metric in metrics:
        curve = build_age_curve(
            samples, metric,
            bucket_years=bucket_years,
            min_samples=min_samples,
            position=position,
        )
        if check_stability and curve.buckets:
            curve.stable, curve.peak_spread_years = curve_is_stable(
                samples, metric, position=position, min_samples=min_samples,
            )
            if curve.stable is False:
                _log.info(
                    "Age curve for %s (%s) is unstable — peak moves %.1f years "
                    "across folds; not safe for trajectory labels.",
                    metric, position or "all positions",
                    curve.peak_spread_years or float("nan"),
                )
        curves[metric] = curve
    return curves


def audit_age_bias(
    matrices_dir: Optional[str] = None,
    reference: Optional[dt.datetime] = None,
) -> Dict[str, float]:
    """Measure how far the stored age drifts from the age at transfer.

    Exists so the scale of the distortion in ``player_age`` is a measured
    number rather than an assumption, since correcting it as a *model input*
    requires rebuilding the feature matrices.
    """
    if matrices_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        matrices_dir = os.path.join(root, "data", "models", "matrices")

    x_path = os.path.join(matrices_dir, "X.npy")
    meta_path = os.path.join(matrices_dir, "metadata.json")
    if not (os.path.exists(x_path) and os.path.exists(meta_path)):
        return {}

    X = np.load(x_path)
    with open(meta_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    age_idx = _age_feature_index()
    stored_vals, corrected_vals = [], []
    skipped = 0
    for i, record in enumerate(metadata):
        if i >= len(X):
            break
        stored = float(X[i, age_idx])
        corrected = age_at_transfer(stored, record.get("transfer_date"), reference)
        if corrected is None:
            skipped += 1
            continue
        stored_vals.append(stored)
        corrected_vals.append(corrected)

    if not corrected_vals:
        return {"n_usable": 0, "n_skipped": float(skipped)}

    stored_arr = np.array(stored_vals)
    corrected_arr = np.array(corrected_vals)
    return {
        "n_usable": float(len(corrected_arr)),
        "n_skipped": float(skipped),
        "mean_stored_age": float(stored_arr.mean()),
        "mean_age_at_transfer": float(corrected_arr.mean()),
        "mean_overstatement_years": float((stored_arr - corrected_arr).mean()),
        "max_overstatement_years": float((stored_arr - corrected_arr).max()),
    }
