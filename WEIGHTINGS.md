# Weightings and Parameters

Every tunable number in TransferScope, where it came from, and whether it has
ever been validated. Written because a review found several that look
authoritative and are not.

**Read this first:** parameters fall into three classes, and the difference
matters more than the values.

| Class | Meaning | Count |
|---|---|---|
| **Learned** | Fitted to data by a training run | ~all NN weights, 26 sklearn adjusters |
| **Measured** | Set from a measurement recorded in this repo | 5 |
| **Asserted** | Hand-set. Plausible, cited to a paper, never fitted or tested | ~70 |

Asserted values are not automatically wrong. They are unvalidated, and the one
group that *was* tested turned out to be actively harmful.

---

## 1. The heuristic coefficients — asserted, and measurably harmful

`backend/features/adjustment_models.py` holds four per-metric tables plus six
structural constants, roughly 58 numbers in total. Each carries a comment
citing the Dinsdale & Gallagher paper.

| Table | Numbers | What it controls |
|---|---|---|
| `_TEAM_INFLUENCE` | 13 | How much a metric is driven by team tactics vs individual skill |
| `_ABILITY_SENSITIVITY` | 13 | How each metric scales with team-quality change |
| `_LEAGUE_STYLE_COEFF` | 13 | Fallback style estimate when team-position data is missing |
| `_OPP_QUALITY_SENS` | 13 | How each metric responds to opposition quality |
| Structural | 6 | `_CONFORMITY_COEFF` 0.25, `_DAMPING_FACTOR_DOWN` 0.05, `_DAMPING_FACTOR_UP` 0.10, `_LEAGUE_ATTN_FACTOR` 1.5, `_LEAGUE_ATTN_FLOOR` 0.25 |

**None of them were fitted.** `calibrate_style_coefficients()` exists in the
same file to fit them from football-data.co.uk match data — and is **never
called from anywhere in the codebase**. The README and this document previously
described football-data.co.uk as powering "coefficient calibration". The data is
fetched and the calibration never runs.

### Measured performance

Benchmarked on the 1,344-transfer temporal test split, against the same
re-anchoring the UI applies:

| Metric | Persistence MAE | Heuristic MAE | Network MAE |
|---|---|---|---|
| expected_goals | 0.0590 | 0.0746 | **0.0554** |
| expected_assists | 0.0393 | 0.0490 | **0.0350** |
| shots | 0.3285 | 0.4642 | **0.3061** |
| successful_dribbles | 0.3080 | 0.3165 | **0.2457** |
| successful_crosses | 0.2297 | 0.3119 | **0.2190** |
| touches_in_opposition_box | 0.8962 | 1.5632 | **0.8399** |
| successful_passes | 5.3725 | 8.7015 | **5.2027** |
| pass_completion_pct | 3.7444 | 5.9636 | **3.5946** |
| accurate_long_balls | 0.5902 | 0.7390 | **0.5428** |
| chances_created | 0.3061 | 0.4371 | **0.2759** |
| clearances | 0.5203 | 0.7667 | **0.4871** |
| interceptions | 0.2697 | 0.3385 | **0.2338** |
| possession_won_final_3rd | 0.2893 | 0.3399 | **0.2560** |

The network beats the heuristic on **13/13** metrics, mean **+32.5%**, paired
bootstrap `[+0.5593, +0.6371]`, significant.

The heuristic is also **worse than persistence on all 13 metrics** — worse than
predicting no change at all. On `successful_passes` it is 62% worse.

### Consequence

Until 2026-08-05 the Shortlist Generator called the heuristic directly and never
touched the network, so its clustering and similarity were both built on the
weaker predictor. It now uses the network and falls back to the heuristic only
when no trained model exists. Transfer Impact and Hot or Not already tried the
network first.

**The heuristic should be treated as a degraded fallback, not a peer.** Either
run the calibration that exists, or replace it with persistence, which is
measurably better.

---

## 2. Neural network parameters — learned, with asserted post-processing

`backend/models/transfer_portal.py`

### Learned
- All layer weights across the six groups, fitted on 7,048 training transfers.
- `feature_scaler.pkl`, `target_scalers.pkl` — fitted `StandardScaler`s.
- `confidence_calibrator.pkl` — isotonic regression mapping raw confidence to
  observed within-20% rate.

### Asserted post-processing
These shape every prediction after the network runs.

| Parameter | Value | Effect |
|---|---|---|
| `DELTA_SHRINKAGE` | 0.90 | Predicted deltas multiplied by this |
| `_METRIC_SHRINKAGE` | per-metric | Overrides the global value |
| `DELTA_CLIP_MULTIPLIER` | 2.0 | Delta capped at ±2× the pre-transfer value |
| `_METRIC_CLIP_FLOORS` | per-metric | Minimum cap for small-valued metrics |
| `DELTA_CLIP_FLOOR` | 1.0 | Fallback cap |
| `DIRECTION_SHRINKAGE_ALPHA` | 0.30 | Direction confidence loosens shrinkage by up to 30% |
| `DIRECTION_FLIP_THRESHOLD` | 0.70 | Above this confidence, a disagreeing delta is flipped |
| `_LOG_EPS` | 0.05 | Floor inside `log()` for the shooting and crossing groups |
| `_MIN_MINUTES_THRESHOLD` | 450 | Players below this are excluded |

Shrinkage at 0.90 is a deliberate 10% pull toward no-change. Given that
persistence beats the heuristic outright, shrinkage is doing real work — but
0.90, 2.0, 0.30 and 0.70 are all round numbers that were never swept.

### Group structure
Six groups over 94 features. Each group sees only its declared feature subset
(`GROUP_FEATURE_SUBSETS`); `scripts/check_artefacts.py` verifies the saved
models match those declarations.

`LOG_SCALE_GROUPS = {shooting, crossing}` predict log-ratios rather than
additive deltas, because both have low base rates with multiplicative error.

---

## 3. Value Opportunity Score — asserted, and deliberately transparent

`backend/models/value_score.py`

| Component | Weight |
|---|---|
| `output_per_value` | 0.40 |
| `projected_improvement` | 0.25 |
| `contract_leverage` | 0.20 |
| `age_runway` | 0.15 |

Plus `PEAK_AGE` 26.5 and `MAX_USEFUL_CONTRACT_YEARS` 5.0.

These are judgement calls and are presented as such: every score ships with its
component breakdown and a plain-English reason, so a user can disagree with the
weighting rather than having to trust it. There is no ground truth to fit them
against — "was this signing good value" is not a labelled quantity.

Two properties are enforced by tests: missing data is never imputed as zero, and
scoring uses percentile ranks rather than min-max, because market values are
right-skewed enough that one €200M player would flatten everyone else.

---

## 4. Shortlist scoring

`backend/models/shortlist_scorer.py`

| Parameter | Value | Status |
|---|---|---|
| Same-cluster bonus | 15% | Asserted |
| `_LOW_CONFIDENCE_THRESHOLD` | 0.3 | Asserted |
| `MIN_MINUTES_THRESHOLD` | 450 | Asserted |
| `_THIN_MINUTES` | 900 | Asserted (~10 matches) |
| `_SOLID_MINUTES` | 1800 | Asserted (~20 matches) |
| k-means `k` | `min(5, n//3)` | Asserted |

Metric weights are set by the user, not by the system.

---

## 5. Power rankings

`backend/features/power_rankings.py`

| Parameter | Value | Status |
|---|---|---|
| `_OPTA_ELO_MIN` / `_OPTA_ELO_MAX` | 1000 / 2100 | **Measured** — the observed ClubElo range, used to rescale Opta 0-100 |
| Fuzzy match threshold | 80 | Asserted |
| League mean | Opta `seasonAverageRating` | **Measured** — read from the source, not computed |

The league mean was previously derived by grouping matched teams, which left 97%
of clubs compared against a global ~51. See the git history for the correction.

---

## 6. Validation parameters — measured

`backend/models/backtester.py`

| Parameter | Value | Status |
|---|---|---|
| Mean-reversion `lambda` | per metric | **Learned** — grid-searched on the training split only |
| Bootstrap resamples | 2000 | Asserted (standard) |
| Confidence interval | 95% | Asserted (standard) |
| Temporal split | 75/15/10 | Asserted |

`backend/models/age_curves.py`

| Parameter | Value | Status |
|---|---|---|
| `PEAK_STABILITY_TOLERANCE_YEARS` | 3.0 | Asserted |
| `DEFAULT_MIN_SAMPLES` | 30 | Asserted |
| `MIN_PLAUSIBLE_AGE` / `MAX` | 15 / 42 | **Measured** — from the observed distribution |

---

## 7. Data-layer parameters

`backend/data/`

| Parameter | Value | Status |
|---|---|---|
| Cache size limit | 2 GB | Asserted |
| Player stats TTL | 1 day | Asserted |
| Squad profiles / REEP TTL | 7 days | Asserted |
| Opta bundle TTL | 7 days | **Measured** — the bundle regenerates roughly weekly |
| Negative-cache TTL | 24 h | Asserted |
| Sofascore retries | 3, exponential | Asserted |
| Inter-request delay | 0.5 s, adaptive to 4 s | Asserted |
| `MIN_PLAYER_COVERAGE` | 0.90 | Asserted |
| `MIN_SEASON_COVERAGE` | 0.80 | Asserted |

---

## What to fix, in order

1. **Run the calibration that already exists**, or delete it and stop claiming
   football-data.co.uk calibrates anything. `calibrate_style_coefficients()` is
   dead code that the docs describe as active.
2. **Consider replacing the heuristic fallback with persistence.** It is
   measurably better on all 13 metrics and has no parameters at all.
3. **Sweep the network's post-processing constants.** Shrinkage, clipping and
   the direction thresholds are round numbers applied to every prediction; a
   sweep against the test split would either justify them or improve them.
4. Leave the Value Opportunity weights alone. They are honest judgement calls
   with no ground truth to fit against, and they are displayed with their
   reasoning.
