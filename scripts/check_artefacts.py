#!/usr/bin/env python
"""Verify that saved model artefacts are consistent with the current code.

TransferScope's neural network is split into groups (see ``MODEL_GROUPS``) and
fed a fixed-width feature vector (``FEATURE_DIM``).  Both have changed over the
project's life — the 4-group/93-feature architecture became 6-group/94-feature
when the combined ``passing`` group was split into ``creation``,
``distribution`` and ``crossing``.

When the code changes but the artefacts in ``data/models/`` do not, the app
fails at prediction time with an opaque error such as::

    ValueError: X has 94 features, but StandardScaler is expecting 93 features

This script turns that late, confusing failure into an early, explicit one.
It is safe to run when no artefacts exist (a fresh clone), in which case the
on-disk checks are skipped.

Exit codes: 0 = consistent, 1 = mismatch.
"""

from __future__ import annotations

import os
import sys

# Keep TensorFlow quiet and CPU-only — this script only inspects shapes.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.transfer_portal import (  # noqa: E402
    FEATURE_DIM,
    GROUP_FEATURE_SUBSETS,
    MODEL_GROUPS,
    _feature_keys,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_ROOT, "data", "models")
_GROUP_DIR = os.path.join(_MODELS_DIR, "transfer_portal")
_SCALER_PATH = os.path.join(_MODELS_DIR, "feature_scaler.pkl")

errors: list[str] = []
notes: list[str] = []


def _check_code_internal_consistency() -> None:
    """Checks that need no artefacts — these always run."""
    keys = _feature_keys()
    if len(keys) != FEATURE_DIM:
        errors.append(
            f"FEATURE_DIM is {FEATURE_DIM} but _feature_keys() returns {len(keys)} keys."
        )

    duplicates = {k for k in keys if keys.count(k) > 1}
    if duplicates:
        errors.append(f"Duplicate feature keys: {sorted(duplicates)}")

    missing_subsets = [g for g in MODEL_GROUPS if g not in GROUP_FEATURE_SUBSETS]
    if missing_subsets:
        errors.append(
            f"MODEL_GROUPS entries with no GROUP_FEATURE_SUBSETS: {missing_subsets}"
        )

    extra_subsets = [g for g in GROUP_FEATURE_SUBSETS if g not in MODEL_GROUPS]
    if extra_subsets:
        errors.append(
            f"GROUP_FEATURE_SUBSETS entries not in MODEL_GROUPS: {extra_subsets}"
        )

    known = set(keys)
    for group, subset in GROUP_FEATURE_SUBSETS.items():
        unknown = [f for f in subset if f not in known]
        if unknown:
            errors.append(
                f"Group '{group}' references features absent from _feature_keys(): {unknown}"
            )


def _check_artefacts() -> None:
    """Checks against the files in data/models/ — skipped when absent."""
    if not os.path.isdir(_GROUP_DIR) and not os.path.exists(_SCALER_PATH):
        notes.append(
            "No trained artefacts in data/models/ — skipping artefact checks. "
            "This is expected on a fresh clone (models are gitignored)."
        )
        return

    # 1. Feature scaler width must match the code's feature vector.
    if os.path.exists(_SCALER_PATH):
        import joblib

        scaler = joblib.load(_SCALER_PATH)
        n_in = getattr(scaler, "n_features_in_", None)
        if n_in is not None and n_in != FEATURE_DIM:
            errors.append(
                f"feature_scaler.pkl expects {n_in} features but the code builds "
                f"{FEATURE_DIM}. Retrain with "
                f"`python -m backend.models.training_pipeline --skip-discovery --skip-build`."
            )
    else:
        errors.append("feature_scaler.pkl is missing but group models exist.")

    if not os.path.isdir(_GROUP_DIR):
        errors.append("data/models/transfer_portal/ is missing but a scaler exists.")
        return

    saved = sorted(f[:-6] for f in os.listdir(_GROUP_DIR) if f.endswith(".keras"))
    # Ensemble members are saved as "<group>_seed0.keras" etc.
    def _base(name: str) -> str:
        return name.rsplit("_seed", 1)[0] if "_seed" in name else name

    saved_groups = sorted({_base(s) for s in saved})

    missing = [g for g in MODEL_GROUPS if g not in saved_groups]
    if missing:
        errors.append(
            f"Missing group models: {missing}. The app would silently lose the "
            f"metrics belonging to those groups. Retrain the pipeline."
        )

    orphans = [g for g in saved_groups if g not in MODEL_GROUPS]
    if orphans:
        errors.append(
            f"Orphaned group models from an older architecture: {orphans}. "
            f"Delete them so they cannot be loaded by mistake."
        )

    # 2. Each model's input width must match its declared feature subset.
    if not missing:
        import tensorflow as tf

        for group in MODEL_GROUPS:
            path = os.path.join(_GROUP_DIR, f"{group}.keras")
            if not os.path.exists(path):
                continue  # ensemble layout — covered by the group check above
            model = tf.keras.models.load_model(path, compile=False)
            expected_in = len(GROUP_FEATURE_SUBSETS[group])
            actual_in = model.input_shape[-1]
            if actual_in != expected_in:
                errors.append(
                    f"Group '{group}' model takes {actual_in} inputs but "
                    f"GROUP_FEATURE_SUBSETS declares {expected_in}."
                )
            expected_out = len(MODEL_GROUPS[group])
            outputs = model.outputs if isinstance(model.outputs, list) else [model.outputs]
            actual_out = outputs[0].shape[-1]
            if actual_out != expected_out:
                errors.append(
                    f"Group '{group}' model predicts {actual_out} targets but "
                    f"MODEL_GROUPS declares {expected_out}."
                )


def _check_feature_health() -> None:
    """Flag features that carry no signal.

    A constant column trains without complaint and contributes nothing. Three
    such features exist because Sofascore genuinely does not serve those stats;
    those are documented. Any *other* constant feature means a source stopped
    supplying data, a key mapping broke, or a migration zero-filled a column.
    """
    from backend.models.feature_audit import audit_saved_matrices, format_report

    report = audit_saved_matrices()
    if report is None:
        notes.append("No saved feature matrices — skipping feature health check.")
        return

    notes.append(format_report(report).replace("\n", "\n      "))

    unexpected = report.get("unexpected_dead") or []
    if unexpected:
        errors.append(
            f"Features carry no signal and are not documented gaps: "
            f"{', '.join(unexpected)}. Either the source stopped supplying "
            f"them, a key mapping broke, or a migration zero-filled the column."
        )

    non_finite = report.get("non_finite") or []
    if non_finite:
        errors.append(
            f"Features contain NaN or inf: {', '.join(non_finite)}."
        )


def main() -> int:
    _check_code_internal_consistency()
    _check_artefacts()
    _check_feature_health()

    for note in notes:
        print(f"note: {note}")

    if errors:
        print("\nArtefact consistency check FAILED:\n")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(
        f"Artefact consistency check passed "
        f"({len(MODEL_GROUPS)} groups, {FEATURE_DIM} features)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
