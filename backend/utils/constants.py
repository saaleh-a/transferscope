"""Shared threshold constants used across backend modules."""

# Minimum minutes a player must have played to be included in shortlists
# and training data. Must be consistent between both to avoid out-of-distribution
# predictions in the shortlist.
MIN_MINUTES_THRESHOLD = 450
